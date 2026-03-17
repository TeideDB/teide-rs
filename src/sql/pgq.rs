// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{ffi, Column, Graph, HnswIndex, Rel, Table};

/// Monotonic counter for unique temp file names.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
use super::pgq_parser::{CreatePropertyGraph as ParsedCPG, PgqStatement};
use super::{ExecResult, Session, SqlError};

/// Write CSV data to a secure temp file and read it back as a Table.
/// Uses `create_new(true)` (O_CREAT|O_EXCL) to prevent symlink attacks:
/// the open fails if the path already exists, so a pre-planted symlink
/// cannot redirect the write to an arbitrary file.
fn csv_to_table(session: &Session, csv: &str, col_names: &[String]) -> Result<Table, SqlError> {
    // Retry with fresh counter values if a stale/colliding file already exists.
    let (tmp_path, mut file) = {
        let mut last_err = None;
        let mut result = None;
        for _ in 0..8 {
            // Mix PID, monotonic counter, and a timestamp-based nonce to make
            // file names unpredictable to local observers (mitigates DoS via
            // pre-creation of deterministic candidate names).
            let nonce: u64 = {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                t.as_nanos() as u64
            };
            let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            // Simple mixing: XOR the nonce with counter shifted, then wrap in hex.
            let tag = nonce ^ (counter.wrapping_mul(0x517cc1b727220a95));
            let p = std::env::temp_dir().join(format!(
                "__pgq_{:x}.csv",
                tag
            ));
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            match opts.open(&p) {
                Ok(f) => { result = Some((p, f)); break; }
                Err(e) => { last_err = Some(e); }
            }
        }
        result.ok_or_else(|| SqlError::Plan(format!(
            "Failed to create temp file after retries: {}",
            last_err.unwrap()
        )))?
    };
    let write_result = file
        .write_all(csv.as_bytes())
        .and_then(|()| file.flush())
        .map_err(|e| SqlError::Plan(format!("Failed to write temp file: {e}")));
    drop(file);
    let result = write_result.and_then(|()| {
        let path_str = tmp_path.to_str().ok_or_else(|| {
            SqlError::Plan("Temp file path not valid UTF-8".into())
        })?;
        let table = session.ctx.read_csv(path_str)?;
        table.with_column_names(col_names).map_err(Into::into)
    });
    // Always clean up the temp file, even on error paths.
    let _ = std::fs::remove_file(&tmp_path);
    result
}

/// Safe conversion of `Table::nrows()` (i64) to usize, returning an error
/// if the C engine returned a negative sentinel.
pub(super) fn checked_nrows(table: &Table) -> Result<usize, SqlError> {
    let n = table.nrows();
    if n < 0 {
        return Err(SqlError::Plan(format!("nrows() returned negative value ({n}); possible engine error")));
    }
    Ok(n as usize)
}

/// Like `checked_nrows` but uses `StoredTable::logical_nrows()` to correct
/// for embedding columns whose flat N*D F32 arrays inflate `table.nrows()`.
pub(super) fn checked_logical_nrows(stored: &super::StoredTable) -> Result<usize, SqlError> {
    let n = stored.logical_nrows();
    if n < 0 {
        return Err(SqlError::Plan(format!("logical_nrows() returned negative value ({n}); possible engine error")));
    }
    Ok(n as usize)
}

// ---------------------------------------------------------------------------
// Property graph catalog types
// ---------------------------------------------------------------------------

/// A vertex label mapping: label name -> session table name.
pub(crate) struct VertexLabel {
    pub table_name: String,
    #[allow(dead_code)]
    pub label: String,
}

/// An edge label mapping: label name -> edge table with source/dest references.
#[derive(Clone)]
pub(crate) struct EdgeLabel {
    pub table_name: String,
    #[allow(dead_code)]
    pub label: String,
    pub src_col: String,
    pub src_ref_table: String,
    pub src_ref_col: String,
    pub dst_col: String,
    pub dst_ref_table: String,
    pub dst_ref_col: String,
}

/// Stored relationship: the built CSR index + its edge label metadata.
pub(crate) struct StoredRel {
    pub rel: Rel,
    pub edge_label: EdgeLabel,
}

/// A property graph defined over session tables.
pub(crate) struct PropertyGraph {
    #[allow(dead_code)]
    pub name: String,
    pub vertex_labels: HashMap<String, VertexLabel>,
    pub edge_labels: HashMap<String, StoredRel>,
}

// ---------------------------------------------------------------------------
// MATCH pattern AST
// ---------------------------------------------------------------------------

/// Direction of an edge in a MATCH pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchDirection {
    Forward,    // ->
    Reverse,    // <-
    Undirected, // - (either direction)
}

/// Quantifier on an edge pattern.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PathQuantifier {
    One,                        // (no quantifier) = exactly 1 hop
    Range { min: u8, max: u8 }, // {min,max}
    Plus,                       // + (1 or more)
    Star,                       // * (0 or more)
}

/// A node pattern: (var:Label WHERE condition)
#[derive(Debug, Clone)]
pub(crate) struct NodePattern {
    #[allow(dead_code)]
    pub variable: Option<String>,
    pub label: Option<String>,
    pub filter: Option<String>, // raw SQL predicate text
}

/// An edge pattern: -[var:Label]-> with optional quantifier
#[derive(Debug, Clone)]
pub(crate) struct EdgePattern {
    #[allow(dead_code)]
    pub variable: Option<String>,
    pub label: Option<String>,
    pub direction: MatchDirection,
    pub quantifier: PathQuantifier,
}

/// A single path pattern: node-edge-node-edge-...-node
#[derive(Debug, Clone)]
pub(crate) struct PathPattern {
    pub nodes: Vec<NodePattern>,
    pub edges: Vec<EdgePattern>,
}

/// Whether this is a shortest-path query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathMode {
    Walk,        // default: all paths
    AnyShortest, // ANY SHORTEST
}

/// A parsed MATCH clause.
#[derive(Debug, Clone)]
pub(crate) struct MatchClause {
    #[allow(dead_code)]
    pub path_variable: Option<String>,
    pub mode: PathMode,
    pub patterns: Vec<PathPattern>, // multiple patterns = comma-separated
}

/// A COLUMNS clause entry: expression AS alias.
#[derive(Debug, Clone)]
pub(crate) struct ColumnEntry {
    pub expr: String, // raw SQL expression (e.g. "a.name", "COUNT(b.id)")
    pub alias: Option<String>,
}

/// A fully parsed GRAPH_TABLE invocation.
#[derive(Debug, Clone)]
pub(crate) struct GraphTableExpr {
    pub graph_name: String,
    pub match_clause: MatchClause,
    pub columns: Vec<ColumnEntry>,
}

// ---------------------------------------------------------------------------
// Build a PropertyGraph from parsed DDL
// ---------------------------------------------------------------------------

/// Build a PropertyGraph from a parsed CREATE PROPERTY GRAPH statement.
/// Validates that all referenced tables exist in the session and builds
/// CSR indexes for each edge table.
pub(crate) fn build_property_graph(
    session: &Session,
    parsed: &ParsedCPG,
) -> Result<PropertyGraph, SqlError> {
    let mut vertex_labels = HashMap::new();

    for vt in &parsed.vertex_tables {
        if !session.tables.contains_key(&vt.table_name) {
            return Err(SqlError::Plan(format!(
                "Vertex table '{}' not found in session",
                vt.table_name
            )));
        }
        let label = vt
            .label
            .as_deref()
            .unwrap_or(&vt.table_name)
            .to_lowercase();
        if vertex_labels.contains_key(&label) {
            return Err(SqlError::Plan(format!(
                "Duplicate vertex label '{label}' in property graph"
            )));
        }
        vertex_labels.insert(
            label.clone(),
            VertexLabel {
                table_name: vt.table_name.clone(),
                label: label.clone(),
            },
        );
    }

    let mut edge_labels = HashMap::new();

    for et in &parsed.edge_tables {
        let edge_stored = session.tables.get(&et.table_name).ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge table '{}' not found in session",
                et.table_name
            ))
        })?;

        let src_stored = session.tables.get(&et.src_ref_table).ok_or_else(|| {
            SqlError::Plan(format!(
                "Source reference table '{}' not found",
                et.src_ref_table
            ))
        })?;
        let dst_stored = session.tables.get(&et.dst_ref_table).ok_or_else(|| {
            SqlError::Plan(format!(
                "Destination reference table '{}' not found",
                et.dst_ref_table
            ))
        })?;

        let n_src_usize = checked_logical_nrows(src_stored).map_err(|_| SqlError::Plan(format!(
            "Source vertex table '{}' has invalid row count", et.src_ref_table
        )))?;
        let n_dst_usize = checked_logical_nrows(dst_stored).map_err(|_| SqlError::Plan(format!(
            "Destination vertex table '{}' has invalid row count", et.dst_ref_table
        )))?;
        let n_src = n_src_usize as i64;
        let n_dst = n_dst_usize as i64;

        // Validate that edge src/dst column values are 0-based row indices.
        // The C engine CSR treats these as row offsets into the vertex tables.
        let n_edges = checked_logical_nrows(edge_stored).map_err(|_| SqlError::Plan(format!(
            "Edge table '{}' has invalid row count", et.table_name
        )))?;
        let src_col_idx = find_col_idx(&edge_stored.table, &et.src_col).ok_or_else(|| {
            SqlError::Plan(format!(
                "Column '{}' not found in edge table '{}'",
                et.src_col, et.table_name
            ))
        })?;
        let dst_col_idx = find_col_idx(&edge_stored.table, &et.dst_col).ok_or_else(|| {
            SqlError::Plan(format!(
                "Column '{}' not found in edge table '{}'",
                et.dst_col, et.table_name
            ))
        })?;
        for row in 0..n_edges {
            let v = edge_stored.table.get_i64(src_col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!(
                    "Edge table '{}' column '{}' row {}: value is NULL. \
                     Edge src/dst values must be non-NULL 0-based row indices.",
                    et.table_name, et.src_col, row
                ))
            })?;
            if v < 0 || v >= n_src {
                return Err(SqlError::Plan(format!(
                    "Edge table '{}' column '{}' row {}: value {} is not a valid row index \
                     into '{}' (expected 0..{}). Edge src/dst values must be 0-based row indices.",
                    et.table_name, et.src_col, row, v, et.src_ref_table, n_src
                )));
            }
            let v = edge_stored.table.get_i64(dst_col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!(
                    "Edge table '{}' column '{}' row {}: value is NULL. \
                     Edge src/dst values must be non-NULL 0-based row indices.",
                    et.table_name, et.dst_col, row
                ))
            })?;
            if v < 0 || v >= n_dst {
                return Err(SqlError::Plan(format!(
                    "Edge table '{}' column '{}' row {}: value {} is not a valid row index \
                     into '{}' (expected 0..{}). Edge src/dst values must be 0-based row indices.",
                    et.table_name, et.dst_col, row, v, et.dst_ref_table, n_dst
                )));
            }
        }

        // Validate that vertex key columns contain 0-based sequential row indices.
        // The traversal code (expand, shortest path BFS) uses key column values as
        // CSR node indices, so they must match row positions.
        validate_key_column_is_rowid(src_stored, &et.src_ref_col, &et.src_ref_table)?;
        validate_key_column_is_rowid(dst_stored, &et.dst_ref_col, &et.dst_ref_table)?;

        let rel = Rel::from_edges(
            &edge_stored.table,
            &et.src_col,
            &et.dst_col,
            n_src,
            n_dst,
            true,
        )?;

        let label = et
            .label
            .as_deref()
            .unwrap_or(&et.table_name)
            .to_lowercase();

        let edge_label = EdgeLabel {
            table_name: et.table_name.clone(),
            label: label.clone(),
            src_col: et.src_col.clone(),
            src_ref_table: et.src_ref_table.clone(),
            src_ref_col: et.src_ref_col.clone(),
            dst_col: et.dst_col.clone(),
            dst_ref_table: et.dst_ref_table.clone(),
            dst_ref_col: et.dst_ref_col.clone(),
        };

        if edge_labels.contains_key(&label) {
            return Err(SqlError::Plan(format!(
                "Duplicate edge label '{label}' in property graph"
            )));
        }
        edge_labels.insert(label, StoredRel { rel, edge_label });
    }

    Ok(PropertyGraph {
        name: parsed.name.clone(),
        vertex_labels,
        edge_labels,
    })
}

// ---------------------------------------------------------------------------
// Execute PGQ statements
// ---------------------------------------------------------------------------

/// Execute a parsed PGQ statement.
pub(crate) fn execute_pgq(
    session: &mut Session,
    stmt: PgqStatement,
) -> Result<ExecResult, SqlError> {
    match stmt {
        PgqStatement::CreatePropertyGraph(parsed) => {
            let name = parsed.name.clone();
            if session.graphs.contains_key(&name) {
                return Err(SqlError::Plan(format!(
                    "Property graph '{name}' already exists"
                )));
            }
            let n_vertices: usize = parsed.vertex_tables.len();
            let n_edges: usize = parsed.edge_tables.len();
            let graph = build_property_graph(session, &parsed)?;
            session.graphs.insert(name.clone(), graph);
            Ok(ExecResult::Ddl(format!(
                "Created property graph '{name}' ({n_vertices} vertex labels, {n_edges} edge labels)"
            )))
        }
        PgqStatement::DropPropertyGraph { name, if_exists } => {
            if session.graphs.remove(&name).is_some() {
                Ok(ExecResult::Ddl(format!(
                    "Dropped property graph '{name}'"
                )))
            } else if if_exists {
                Ok(ExecResult::Ddl(format!(
                    "Property graph '{name}' not found (skipped)"
                )))
            } else {
                Err(SqlError::Plan(format!(
                    "Property graph '{name}' not found"
                )))
            }
        }
        PgqStatement::CreateVectorIndex(parsed) => {
            execute_create_vector_index(session, parsed)
        }
        PgqStatement::DropVectorIndex { name, if_exists } => {
            if session.vector_indexes.remove(&name).is_some() {
                Ok(ExecResult::Ddl(format!(
                    "Dropped vector index '{name}'"
                )))
            } else if if_exists {
                Ok(ExecResult::Ddl(format!(
                    "Vector index '{name}' not found (skipped)"
                )))
            } else {
                Err(SqlError::Plan(format!(
                    "Vector index '{name}' not found"
                )))
            }
        }
    }
}

fn execute_create_vector_index(
    session: &mut Session,
    parsed: super::pgq_parser::CreateVectorIndex,
) -> Result<ExecResult, SqlError> {
    let name = parsed.name.clone();
    if session.vector_indexes.contains_key(&name) {
        return Err(SqlError::Plan(format!(
            "Vector index '{name}' already exists"
        )));
    }

    // Reject duplicate indexes on the same (table, column) pair to ensure
    // deterministic transparent KNN optimisation via find_vector_index.
    let dup_col_key = parsed.column_name.to_lowercase();
    if session.vector_indexes.values().any(|vi| {
        vi.table_name == parsed.table_name && vi.column_name == dup_col_key
    }) {
        return Err(SqlError::Plan(format!(
            "A vector index already exists on {}.{}",
            parsed.table_name, parsed.column_name
        )));
    }

    // Look up the table
    let stored = session.tables.get(&parsed.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", parsed.table_name))
    })?;

    // Find the embedding column and its dimension
    let col_key = parsed.column_name.to_lowercase();
    let dim = stored.embedding_dims.get(&col_key).copied().ok_or_else(|| {
        SqlError::Plan(format!(
            "Column '{}' in table '{}' is not a registered embedding column. \
             Use Session::register_embedding_column or add_embedding_column first.",
            parsed.column_name, parsed.table_name
        ))
    })?;

    // Find the column index
    let col_idx = stored
        .columns
        .iter()
        .position(|c| c.to_lowercase() == col_key)
        .ok_or_else(|| {
            SqlError::Plan(format!(
                "Column '{}' not found in table '{}'",
                parsed.column_name, parsed.table_name
            ))
        })?;

    // Get the raw F32 data from the column
    let col_ptr = stored.table.get_col_idx(col_idx as i64).ok_or_else(|| {
        SqlError::Plan(format!(
            "Cannot access column '{}' data",
            parsed.column_name
        ))
    })?;
    let col_type = unsafe { ffi::td_type(col_ptr) };
    if col_type != ffi::TD_F32 {
        return Err(SqlError::Plan(format!(
            "Column '{}' has type {col_type}, expected TD_F32={} for vector index",
            parsed.column_name,
            ffi::TD_F32
        )));
    }

    let raw_len = unsafe { ffi::td_len(col_ptr) };
    if raw_len <= 0 {
        return Err(SqlError::Plan(format!(
            "Column '{}' is empty or has invalid length {}",
            parsed.column_name, raw_len
        )));
    }
    let n_floats = raw_len as usize;
    if n_floats % (dim as usize) != 0 {
        return Err(SqlError::Plan(format!(
            "Column '{}' has {} floats which is not divisible by dimension {} — data may be corrupt",
            parsed.column_name, n_floats, dim
        )));
    }
    let n_nodes = (n_floats / dim as usize) as i64;

    // Extract the float data
    let data_ptr = unsafe { ffi::td_data(col_ptr) as *const f32 };
    let vectors = unsafe { std::slice::from_raw_parts(data_ptr, n_floats) };

    let m = parsed.m.unwrap_or(16);
    let ef_construction = parsed.ef_construction.unwrap_or(200);

    if m <= 0 {
        return Err(SqlError::Plan(format!(
            "HNSW parameter M must be positive, got {m}"
        )));
    }
    if ef_construction <= 0 {
        return Err(SqlError::Plan(format!(
            "HNSW parameter ef_construction must be positive, got {ef_construction}"
        )));
    }

    let index = HnswIndex::build(&session.ctx, vectors, n_nodes, dim, m, ef_construction)
        .map_err(SqlError::Engine)?;

    session.vector_indexes.insert(
        name.clone(),
        super::VectorIndexInfo {
            table_name: parsed.table_name.clone(),
            column_name: col_key,
            index,
            m,
            ef_construction,
        },
    );

    Ok(ExecResult::Ddl(format!(
        "Created vector index '{name}' on {}.{} (HNSW, M={m}, ef_construction={ef_construction}, {n_nodes} vectors, dim={dim})",
        parsed.table_name, parsed.column_name
    )))
}

// ---------------------------------------------------------------------------
// GRAPH_TABLE planner: translate MATCH patterns into engine ops
// ---------------------------------------------------------------------------

/// Plan and execute a GRAPH_TABLE expression.
/// Returns the result table with the requested COLUMNS.
pub(crate) fn plan_graph_table(
    session: &Session,
    expr: &GraphTableExpr,
) -> Result<(Table, Vec<String>), SqlError> {
    let graph = session.graphs.get(&expr.graph_name).ok_or_else(|| {
        SqlError::Plan(format!(
            "Property graph '{}' not found",
            expr.graph_name
        ))
    })?;

    let match_clause = &expr.match_clause;

    if match_clause.patterns.is_empty() {
        return Err(SqlError::Plan("MATCH requires at least one pattern".into()));
    }

    if match_clause.patterns.len() > 1 {
        return Err(SqlError::Plan(
            "Multiple comma-separated MATCH patterns are not yet supported. \
             Use a single pattern: (a)-[e]->(b)."
                .into(),
        ));
    }

    let pattern = &match_clause.patterns[0];

    // Check for variable-length edge first.
    let is_var_length = pattern.edges.len() == 1
        && !matches!(pattern.edges[0].quantifier, PathQuantifier::One);

    match (pattern.nodes.len(), pattern.edges.len(), match_clause.mode) {
        // ANY SHORTEST always uses the shortest-path planner (supports dst filters,
        // _node/_depth columns).
        (2, 1, PathMode::AnyShortest) => {
            plan_shortest_path(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) if is_var_length => {
            plan_var_length(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) => {
            plan_single_hop(session, graph, pattern, &expr.columns)
        }
        (1, 0, _) => {
            plan_algorithm_query(session, graph, pattern, &expr.columns)
        }
        _ if pattern.nodes.len() >= 3 && pattern.edges.len() >= 2 => {
            let all_fixed = pattern.edges.iter().all(|e| matches!(e.quantifier, PathQuantifier::One));
            if all_fixed {
                plan_multi_hop_fixed(session, graph, pattern, &expr.columns, &match_clause.mode)
            } else {
                plan_multi_hop_variable(session, graph, pattern, &expr.columns, &match_clause.mode)
            }
        }
        _ => {
            Err(SqlError::Plan(format!(
                "Unsupported MATCH pattern: {} nodes, {} edges",
                pattern.nodes.len(),
                pattern.edges.len()
            )))
        }
    }
}

/// Plan a single-hop MATCH: (a:Label)-[e:Label]->(b:Label)
fn plan_single_hop(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let src_node = &pattern.nodes[0];
    let edge = &pattern.edges[0];
    let dst_node = &pattern.nodes[1];

    // Destination node WHERE filters are not yet supported in single-hop
    if dst_node.filter.is_some() {
        return Err(SqlError::Plan(
            "WHERE filters on destination nodes are not yet supported in single-hop patterns. \
             Use a WHERE clause in the outer SELECT instead."
                .into(),
        ));
    }

    // Resolve edge label
    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label: -[:Label]->".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
    })?;

    // For reverse edges, the left node in the pattern is the edge destination
    // and the right node is the edge source, so swap the default table references.
    let is_reverse = edge.direction == MatchDirection::Reverse;
    let (left_default_table, right_default_table, scan_ref_col) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_col)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_col)
    };

    // Resolve source and destination vertex tables
    let src_label = resolve_node_label(src_node, left_default_table, graph)?;
    let dst_label = resolve_node_label(dst_node, right_default_table, graph)?;

    // Validate that explicit node labels match the edge's expected tables.
    // Without this check, a query like (a:City)-[:knows]->(b:Person) could
    // silently bind nodes to wrong table domains in heterogeneous graphs.
    validate_node_table_for_edge(src_node, &src_label.table_name, left_default_table, edge_label, "source")?;
    validate_node_table_for_edge(dst_node, &dst_label.table_name, right_default_table, edge_label, "destination")?;

    let src_stored = session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?;
    let dst_stored = session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?;

    // Undirected traversals on heterogeneous edges (different src/dst tables)
    // would mix node IDs from different tables in the _dst column, producing
    // wrong results when project_columns looks up rows in dst_table.
    if edge.direction == MatchDirection::Undirected
        && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
    {
        return Err(SqlError::Plan(
            "Undirected traversals are not supported on edges with different source and \
             destination vertex tables (heterogeneous graphs). Use a directed pattern instead."
                .into(),
        ));
    }

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // Build graph on the left-side node's table
    let mut g = session.ctx.graph(&src_stored.table)?;

    // Scan the appropriate reference column for the left-side node
    let src_ids = g.scan(scan_ref_col)?;

    // Apply source node filter if present
    let src_ids = if let Some(filter_text) = &src_node.filter {
        apply_node_filter(
            &mut g,
            src_ids,
            filter_text,
            src_node.variable.as_deref(),
        )?
    } else {
        src_ids
    };

    // Expand via CSR
    let expanded = g.expand(src_ids, &stored_rel.rel, direction)?;
    let expand_result = g.execute(expanded)?;

    // Project requested columns
    project_columns(
        session,
        &expand_result,
        columns,
        src_node,
        dst_node,
        &src_stored.table,
        &dst_stored.table,
    )
}

/// Resolve which vertex label a node pattern refers to.
fn resolve_node_label<'a>(
    node: &NodePattern,
    default_table: &str,
    graph: &'a PropertyGraph,
) -> Result<&'a VertexLabel, SqlError> {
    if let Some(label) = &node.label {
        graph.vertex_labels.get(label).ok_or_else(|| {
            SqlError::Plan(format!("Vertex label '{label}' not found in graph"))
        })
    } else {
        graph
            .vertex_labels
            .values()
            .find(|vl| vl.table_name == default_table)
            .ok_or_else(|| {
                SqlError::Plan(format!(
                    "No vertex label found for table '{default_table}'"
                ))
            })
    }
}

/// Validate that a node's resolved table matches the edge's expected table for
/// its position (source or destination). Only checks when the node has an
/// explicit label — if the node uses the default table, it's already correct.
fn validate_node_table_for_edge(
    node: &NodePattern,
    resolved_table: &str,
    expected_table: &str,
    edge_label: &str,
    position: &str,
) -> Result<(), SqlError> {
    if let Some(label) = &node.label {
        if resolved_table != expected_table {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' resolves to table '{resolved_table}', but edge \
                 '{edge_label}' expects {position} table '{expected_table}'"
            )));
        }
    }
    Ok(())
}

/// Apply a WHERE filter from a node pattern.
/// Parses filter text like "a.name = 'Alice'" and applies it.
fn apply_node_filter(
    g: &mut Graph,
    ids: Column,
    filter_text: &str,
    variable: Option<&str>,
) -> Result<Column, SqlError> {
    // Split on '=' first, then check only the LHS for unsupported operators.
    // Checking the full filter_text would false-reject values containing
    // operator characters (e.g. a.name = 'A>B').
    let parts: Vec<&str> = filter_text.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(SqlError::Plan(format!(
            "Unsupported node filter syntax: {filter_text}. Only 'col = value' is supported."
        )));
    }
    let lhs = parts[0].trim();
    // After splitting on '=', a trailing '!', '>', or '<' on the LHS means
    // the original expression used !=, >=, or <= — reject those.
    if lhs.ends_with('!') || lhs.ends_with('>') || lhs.ends_with('<') {
        return Err(SqlError::Plan(format!(
            "Unsupported operator in node filter: {filter_text}. Only 'col = value' is supported."
        )));
    }
    let col_name = if let Some(var) = variable {
        let prefixes = [format!("{var} . "), format!("{var} ."), format!("{var}.")];
        let mut s = lhs.to_string();
        for p in &prefixes {
            if let Some(stripped) = s.strip_prefix(p.as_str()) {
                s = stripped.to_string();
                break;
            }
        }
        s.trim().to_lowercase()
    } else {
        lhs.to_lowercase()
    };
    let value = parts[1].trim();
    // Remove internal spaces from value to handle tokenizer artifacts (e.g., "- 1" -> "-1")
    let value_nospace: String = value.chars().filter(|c| !c.is_whitespace()).collect();

    let scan_col = g.scan(&col_name)?;

    let const_col = if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        let s = value[1..value.len() - 1].replace("''", "'");
        g.const_str(&s)?
    } else if let Ok(n) = value_nospace.parse::<i64>() {
        g.const_i64(n)?
    } else if let Ok(f) = value_nospace.parse::<f64>() {
        g.const_f64(f)?
    } else {
        return Err(SqlError::Plan(format!(
            "Unsupported filter value: {value}"
        )));
    };

    let pred = g.eq(scan_col, const_col)?;
    let filtered = g.filter(ids, pred)?;
    Ok(filtered)
}

/// Project COLUMNS from expand results.
/// Maps column expressions like "b.name" to lookups in source/destination tables.
fn project_columns(
    session: &Session,
    expand_result: &Table,
    columns: &[ColumnEntry],
    src_node: &NodePattern,
    dst_node: &NodePattern,
    src_table: &Table,
    dst_table: &Table,
) -> Result<(Table, Vec<String>), SqlError> {
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    // Use row-level extraction: _src/_dst are row indices into vertex tables
    let nrows = checked_nrows(expand_result)?;

    // Find _src and _dst column indices in expand result
    let src_idx_col = find_col_idx(expand_result, "_src")
        .ok_or_else(|| SqlError::Plan("expand result missing _src column".into()))?;
    let dst_idx_col = find_col_idx(expand_result, "_dst")
        .ok_or_else(|| SqlError::Plan("expand result missing _dst column".into()))?;

    let mut src_indices = Vec::with_capacity(nrows);
    let mut dst_indices = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let src_val = expand_result.get_i64(src_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _src index at row {row}")))?;
        if src_val < 0 {
            return Err(SqlError::Plan(format!("Negative _src index {src_val} at row {row}")));
        }
        src_indices.push(src_val as usize);
        let dst_val = expand_result.get_i64(dst_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _dst index at row {row}")))?;
        if dst_val < 0 {
            return Err(SqlError::Plan(format!("Negative _dst index {dst_val} at row {row}")));
        }
        dst_indices.push(dst_val as usize);
    }

    // Build CSV string for result table
    // Maintain column order as specified in the COLUMNS clause
    let mut col_names = Vec::new();
    struct ColSpec {
        table_col_idx: usize,
        is_src: bool, // true = use src_indices, false = use dst_indices
    }
    let mut col_specs: Vec<ColSpec> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            if var == src_var {
                let col_idx = find_col_idx(src_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in source table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { table_col_idx: col_idx, is_src: true });
            } else if var == dst_var {
                let col_idx = find_col_idx(dst_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in destination table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { table_col_idx: col_idx, is_src: false });
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {src_var}, {dst_var}"
                )));
            }
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS expression must be in 'var.col' format, got: {expr}"
            )));
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Build result via CSV, using unique temp column names to avoid clashes
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();

    let mut csv = csv_col_names.join(",");
    csv.push('\n');
    for row in 0..nrows {
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            let table_row = if spec.is_src { src_indices[row] } else { dst_indices[row] };
            let table = if spec.is_src { src_table } else { dst_table };
            csv.push_str(&get_cell_string(table, spec.table_col_idx, table_row)?);
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

/// Find a column index by name in a table.
pub(super) fn find_col_idx(table: &Table, name: &str) -> Option<usize> {
    let ncols = table.ncols();
    if ncols < 0 {
        return None;
    }
    let ncols = ncols as usize;
    (0..ncols).find(|&i| table.col_name_str(i).to_lowercase() == name)
}

/// Validate that a vertex table's key column contains values 0..n-1 matching row indices.
/// The CSR and traversal code use key column values as node indices, so they must
/// be 0-based sequential integers matching row positions.
pub(super) fn validate_key_column_is_rowid(
    stored: &super::StoredTable,
    key_col: &str,
    table_name: &str,
) -> Result<(), SqlError> {
    let col_idx = find_col_idx(&stored.table, key_col).ok_or_else(|| {
        SqlError::Plan(format!(
            "Key column '{key_col}' not found in vertex table '{table_name}'"
        ))
    })?;
    let nrows = checked_logical_nrows(stored)?;
    for row in 0..nrows {
        let val = stored.table.get_i64(col_idx, row).ok_or_else(|| {
            SqlError::Plan(format!(
                "Vertex table '{table_name}' key column '{key_col}' has NULL at row {row}. \
                 Key column must contain 0-based sequential row indices."
            ))
        })?;
        if val != row as i64 {
            return Err(SqlError::Plan(format!(
                "Vertex table '{table_name}' key column '{key_col}' row {row}: value {val} \
                 does not match row index. Key column must contain 0-based sequential row \
                 indices (0, 1, 2, ...) for CSR graph traversal."
            )));
        }
    }
    Ok(())
}

/// Get a cell value as a string for CSV output.
/// String values are quoted and escaped for CSV safety.
/// Temporal types (DATE, TIME, TIMESTAMP) are formatted as human-readable
/// strings and quoted so the CSV reader preserves them as SYM, not integers.
fn get_cell_string(table: &Table, col: usize, row: usize) -> Result<String, SqlError> {
    let typ = table.col_type(col);
    match typ {
        ffi::TD_SYM => {
            if let Some(s) = table.get_str(col, row) {
                return Ok(csv_quote(&s));
            }
        }
        ffi::TD_DATE => {
            if let Some(v) = table.get_i64(col, row) {
                return Ok(csv_quote(&Table::format_date(v as i32)));
            }
        }
        ffi::TD_TIME => {
            if let Some(v) = table.get_i64(col, row) {
                return Ok(csv_quote(&Table::format_time(v as i32)));
            }
        }
        ffi::TD_TIMESTAMP => {
            if let Some(v) = table.get_i64(col, row) {
                return Ok(csv_quote(&Table::format_timestamp(v)));
            }
        }
        ffi::TD_F64 => {
            if let Some(v) = table.get_f64(col, row) {
                return Ok(v.to_string());
            }
        }
        ffi::TD_BOOL => {
            if let Some(v) = table.get_i64(col, row) {
                return Ok(if v != 0 { "true" } else { "false" }.to_string());
            }
        }
        ffi::TD_U8 | ffi::TD_CHAR | ffi::TD_I16 | ffi::TD_I32 | ffi::TD_I64 => {
            if let Some(v) = table.get_i64(col, row) {
                return Ok(v.to_string());
            }
        }
        other => {
            return Err(SqlError::Plan(format!(
                "unsupported column type {} in GRAPH_TABLE projection",
                other
            )));
        }
    }
    // Matched a supported type but get returned None — NULL
    Ok(String::new())
}

/// Quote a string value for CSV: always wrap in double quotes to preserve
/// string type through CSV round-trip (prevents "123" from being parsed as
/// an integer). Escape embedded double quotes by doubling.
fn csv_quote(s: &str) -> String {
    let escaped = s.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

// ---------------------------------------------------------------------------
// Multi-hop fixed-length path planner (wco_join)
// ---------------------------------------------------------------------------

/// Describes one segment in a multi-hop BFS pattern.
struct BfsSegment<'a> {
    stored_rel: &'a StoredRel,
    direction: u8,
    min_depth: u8,
    max_depth: u8,
    src_table_name: String,
    dst_table_name: String,
}

/// Holds BFS output: node IDs at each position boundary and total path lengths.
struct BfsResult {
    /// `node_ids[position][row]` = node_id at that position in the path.
    /// position 0 = start node, position 1 = node after segment 0, etc.
    node_ids: Vec<Vec<i64>>,
    /// Total path length (sum of hops across all segments) per result row.
    path_lengths: Vec<i64>,
    /// Number of result rows.
    nrows: usize,
}

/// Safety limits for the multi-segment BFS.
const MAX_BFS_STATES: usize = 10_000_000;
const MAX_RESULTS: usize = 1_000_000;

/// Core fused BFS across multiple segments.
///
/// Each BFS state is `(segment_index, node_id, depth_in_segment, path_so_far, total_depth)`.
/// `path_so_far` stores node IDs at each position boundary (between segments).
/// When a segment reaches its `min_depth`, the node can "exit" that segment.
/// If it exits the last segment, a result is recorded. Otherwise, the node
/// transitions to the next segment at depth 0.
fn multi_segment_bfs(
    segments: &[BfsSegment],
    start_nodes: &[i64],
    mode: &PathMode,
    is_cyclic: bool,
) -> Result<BfsResult, SqlError> {
    use std::collections::VecDeque;

    let n_positions = segments.len() + 1; // one position per segment boundary

    let mut result_node_ids: Vec<Vec<i64>> = vec![Vec::new(); n_positions];
    let mut result_path_lengths: Vec<i64> = Vec::new();
    let mut nrows: usize = 0;

    // For AnyShortest: track minimum total_depth found so we can prune longer paths
    let mut best_depth: Option<i64> = None;

    // BFS state: (segment_index, node_id, depth_in_segment, path_so_far, total_depth)
    // path_so_far has n_positions entries; path_so_far[0] = start_node, etc.
    struct BfsState {
        seg_idx: usize,
        node_id: i64,
        depth_in_seg: u8,
        path: Vec<i64>,      // node IDs at position boundaries filled so far
        total_depth: i64,
    }

    let mut queue: VecDeque<BfsState> = VecDeque::new();
    let mut n_states: usize = 0;

    // Initialize queue with start nodes
    for &start in start_nodes {
        let mut path = vec![-1i64; n_positions];
        path[0] = start;
        queue.push_back(BfsState {
            seg_idx: 0,
            node_id: start,
            depth_in_seg: 0,
            path,
            total_depth: 0,
        });
        n_states += 1;
    }

    while let Some(state) = queue.pop_front() {
        if nrows >= MAX_RESULTS {
            break;
        }

        let seg = &segments[state.seg_idx];

        // AnyShortest pruning: skip states already longer than the best known
        // result. Note: because segment transitions break FIFO depth ordering,
        // this is only an optimistic prune — a post-filter below removes any
        // results with total_depth > the true minimum.
        if *mode == PathMode::AnyShortest {
            if let Some(best) = best_depth {
                if state.total_depth > best {
                    continue;
                }
            }
        }

        // Check if current depth_in_seg is within [min_depth, max_depth] for exit
        if state.depth_in_seg >= seg.min_depth {
            // This node can exit the current segment.
            let next_position = state.seg_idx + 1;
            let mut exit_path = state.path.clone();
            exit_path[next_position] = state.node_id;

            if next_position == segments.len() {
                // Last segment — record result
                // For cyclic binding: filter results where first node != last node
                if is_cyclic && exit_path[0] != exit_path[n_positions - 1] {
                    // Not a cycle — skip
                } else {
                    for (pos, ids) in result_node_ids.iter_mut().enumerate() {
                        ids.push(exit_path[pos]);
                    }
                    result_path_lengths.push(state.total_depth);
                    nrows += 1;

                    // Update best_depth for pruning (may not be globally minimal yet)
                    if *mode == PathMode::AnyShortest {
                        best_depth = Some(match best_depth {
                            Some(b) if b < state.total_depth => b,
                            _ => state.total_depth,
                        });
                    }
                }
            } else {
                // Transition to next segment at depth 0
                // The node exiting segment[seg_idx] becomes the start of segment[next_position]
                // It enters at depth_in_seg=0 with the same node_id.
                queue.push_back(BfsState {
                    seg_idx: next_position,
                    node_id: state.node_id,
                    depth_in_seg: 0,
                    path: exit_path,
                    total_depth: state.total_depth,
                });
                n_states += 1;
            }
        }

        // Expand neighbors within the current segment if depth < max_depth
        if state.depth_in_seg < seg.max_depth {
            let neighbors = if seg.direction == 2 {
                // Undirected: merge forward and reverse neighbors
                let fwd = seg.stored_rel.rel.neighbors(state.node_id, 0);
                let rev = seg.stored_rel.rel.neighbors(state.node_id, 1);
                let mut merged = Vec::with_capacity(fwd.len() + rev.len());
                merged.extend_from_slice(fwd);
                merged.extend_from_slice(rev);
                merged
            } else {
                seg.stored_rel.rel.neighbors(state.node_id, seg.direction).to_vec()
            };

            for &next_node in &neighbors {
                if n_states >= MAX_BFS_STATES {
                    return Err(SqlError::Plan(format!(
                        "Multi-segment BFS exceeded {MAX_BFS_STATES} states — graph too large or \
                         path quantifier range too wide. Try narrowing the hop range."
                    )));
                }
                queue.push_back(BfsState {
                    seg_idx: state.seg_idx,
                    node_id: next_node,
                    depth_in_seg: state.depth_in_seg + 1,
                    path: state.path.clone(),
                    total_depth: state.total_depth + 1,
                });
                n_states += 1;
            }
        }
    }

    // AnyShortest post-filter: segment transitions can break FIFO depth ordering,
    // so the early pruning above is optimistic. Remove any results whose
    // total_depth exceeds the true minimum.
    if *mode == PathMode::AnyShortest && nrows > 0 {
        let min_depth = *result_path_lengths.iter().min().unwrap();
        let mut keep = Vec::with_capacity(nrows);
        for i in 0..nrows {
            if result_path_lengths[i] == min_depth {
                keep.push(i);
            }
        }
        if keep.len() < nrows {
            let mut new_node_ids = vec![Vec::with_capacity(keep.len()); n_positions];
            let mut new_path_lengths = Vec::with_capacity(keep.len());
            for &i in &keep {
                for (pos, ids) in new_node_ids.iter_mut().enumerate() {
                    ids.push(result_node_ids[pos][i]);
                }
                new_path_lengths.push(result_path_lengths[i]);
            }
            result_node_ids = new_node_ids;
            result_path_lengths = new_path_lengths;
            nrows = keep.len();
        }
    }

    Ok(BfsResult {
        node_ids: result_node_ids,
        path_lengths: result_path_lengths,
        nrows,
    })
}

/// Plan multi-hop patterns where at least one edge has a variable-length quantifier.
///
/// Builds `BfsSegment`s from the pattern, resolves start nodes, calls
/// `multi_segment_bfs`, then projects the COLUMNS clause.
fn plan_multi_hop_variable(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
    mode: &PathMode,
) -> Result<(Table, Vec<String>), SqlError> {
    let nodes = &pattern.nodes;
    let edges = &pattern.edges;

    // --- Build BfsSegments ---
    let mut segments: Vec<BfsSegment> = Vec::with_capacity(edges.len());

    for (i, edge) in edges.iter().enumerate() {
        let left_node = &nodes[i];
        let right_node = &nodes[i + 1];

        let edge_label = edge.label.as_deref().ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge {} in multi-hop pattern must specify a label: -[:Label]->",
                i
            ))
        })?;
        let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
            SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
        })?;

        // Determine left/right default tables based on edge direction
        let is_reverse = edge.direction == MatchDirection::Reverse;
        let (left_default_table, right_default_table) = if is_reverse {
            (
                &stored_rel.edge_label.dst_ref_table,
                &stored_rel.edge_label.src_ref_table,
            )
        } else {
            (
                &stored_rel.edge_label.src_ref_table,
                &stored_rel.edge_label.dst_ref_table,
            )
        };

        let left_label = resolve_node_label(left_node, left_default_table, graph)?;
        let right_label = resolve_node_label(right_node, right_default_table, graph)?;

        // Validate explicit node labels match edge expectations
        validate_node_table_for_edge(
            left_node,
            &left_label.table_name,
            left_default_table,
            edge_label,
            "source",
        )?;
        validate_node_table_for_edge(
            right_node,
            &right_label.table_name,
            right_default_table,
            edge_label,
            "destination",
        )?;

        // Verify vertex table continuity: segment[i].dst == segment[i+1].src
        if let Some(prev) = segments.last() {
            if prev.dst_table_name != left_label.table_name {
                return Err(SqlError::Plan(format!(
                    "Vertex table mismatch between segment {} and {}: '{}' vs '{}'. \
                     Adjacent segments must share a vertex table.",
                    i - 1,
                    i,
                    prev.dst_table_name,
                    left_label.table_name
                )));
            }
        }

        // Reject undirected edges on heterogeneous segments
        if edge.direction == MatchDirection::Undirected
            && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
        {
            return Err(SqlError::Plan(
                "Undirected traversals are not supported on edges with different source and \
                 destination vertex tables (heterogeneous graphs). Use directed patterns instead."
                    .into(),
            ));
        }

        // Variable-length edges must have same src/dst vertex table (CSR node IDs
        // must be in the same domain across multiple hops).
        let is_variable = !matches!(edge.quantifier, PathQuantifier::One);
        let is_single_range = matches!(edge.quantifier, PathQuantifier::Range { min: 1, max: 1 });
        if is_variable
            && !is_single_range
            && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
        {
            return Err(SqlError::Plan(
                "Variable-length edges within a segment must have the same source and \
                 destination vertex table (CSR node IDs must be in the same domain)."
                    .into(),
            ));
        }

        let direction: u8 = match edge.direction {
            MatchDirection::Forward => 0,
            MatchDirection::Reverse => 1,
            MatchDirection::Undirected => 2,
        };

        let (min_depth, max_depth) = match edge.quantifier {
            PathQuantifier::Range { min, max } => (min, max),
            PathQuantifier::Plus => (1, 255),
            PathQuantifier::Star => (0, 255),
            PathQuantifier::One => (1, 1),
        };

        segments.push(BfsSegment {
            stored_rel,
            direction,
            min_depth,
            max_depth,
            src_table_name: left_label.table_name.clone(),
            dst_table_name: right_label.table_name.clone(),
        });
    }

    // Detect cyclic variable binding
    let is_cyclic = match (&nodes[0].variable, &nodes[nodes.len() - 1].variable) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };

    // Reject WHERE filters on intermediate/destination nodes
    for node in &nodes[1..] {
        if node.filter.is_some() {
            return Err(SqlError::Plan(
                "WHERE filters on intermediate or destination nodes are not yet supported \
                 in multi-hop variable-length patterns."
                    .into(),
            ));
        }
    }

    // --- Determine start nodes ---
    let first_src_table_name = &segments[0].src_table_name;
    let first_src_stored = session.tables.get(first_src_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{first_src_table_name}' not found"))
    })?;

    let start_nodes: Vec<i64> = if let Some(ref filter_text) = nodes[0].filter {
        // Try to extract a single node ID from the WHERE filter
        let src_id = extract_node_id_from_filter(
            &nodes[0],
            first_src_table_name,
            filter_text,
            session,
        )?;
        vec![src_id]
    } else {
        // All nodes in the first vertex table
        let nrows = checked_logical_nrows(first_src_stored)?;
        (0..nrows as i64).collect()
    };

    // --- Run BFS ---
    let bfs_result = multi_segment_bfs(&segments, &start_nodes, mode, is_cyclic)?;

    // --- Build variable map: node variable name -> position index ---
    let n_positions = segments.len() + 1;
    let mut var_map: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some(ref var) = node.variable {
            var_map.entry(var.clone()).or_insert(i);
        }
    }

    // Build table name for each position
    let mut pos_table_names: Vec<String> = Vec::with_capacity(n_positions);
    pos_table_names.push(segments[0].src_table_name.clone());
    for seg in &segments {
        pos_table_names.push(seg.dst_table_name.clone());
    }

    // --- Project COLUMNS ---
    enum MhVarColKind {
        Property { var_idx: usize, table_col_idx: usize, table_name: String },
        PathLength,
    }
    let mut col_names: Vec<String> = Vec::new();
    let mut col_specs: Vec<MhVarColKind> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();
            // Use "var_col" as default name to disambiguate columns from
            // different node variables that share the same property name.
            let default_name = format!("{var}_{col}");
            let out_name = alias.unwrap_or(&default_name).to_string();

            let var_idx = var_map.get(&var).ok_or_else(|| {
                let available: Vec<_> = var_map.keys().collect();
                SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available:?}"
                ))
            })?;

            let table_name = &pos_table_names[*var_idx];
            let vtable = session.tables.get(table_name).ok_or_else(|| {
                SqlError::Plan(format!("Table '{table_name}' not found"))
            })?;
            let table_col_idx = find_col_idx(&vtable.table, &col).ok_or_else(|| {
                SqlError::Plan(format!(
                    "Column '{col}' not found in vertex table '{table_name}'"
                ))
            })?;

            col_names.push(out_name);
            col_specs.push(MhVarColKind::Property {
                var_idx: *var_idx,
                table_col_idx,
                table_name: table_name.clone(),
            });
        } else {
            let lower = expr.to_lowercase();
            if lower.contains("path_length") || lower == "_depth" {
                col_names.push(alias.unwrap_or("path_length").to_string());
                col_specs.push(MhVarColKind::PathLength);
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{expr}'"
                )));
            }
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Handle empty results
    if bfs_result.nrows == 0 {
        let csv_col_names: Vec<String> = (0..col_names.len())
            .map(|i| format!("__c{i}"))
            .collect();
        let csv = format!("{}\n", csv_col_names.join(","));
        let result = csv_to_table(session, &csv, &col_names)?;
        return Ok((result, col_names));
    }

    // Build result via CSV
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();

    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for row in 0..bfs_result.nrows {
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            match spec {
                MhVarColKind::Property { var_idx, table_col_idx, table_name } => {
                    let node_id = bfs_result.node_ids[*var_idx][row];
                    if node_id < 0 {
                        return Err(SqlError::Plan(format!(
                            "Negative node index {} at position {} row {}",
                            node_id, var_idx, row
                        )));
                    }
                    let vtable = session.tables.get(table_name).ok_or_else(|| {
                        SqlError::Plan(format!("Table '{table_name}' not found"))
                    })?;
                    csv.push_str(&get_cell_string(
                        &vtable.table,
                        *table_col_idx,
                        node_id as usize,
                    )?);
                }
                MhVarColKind::PathLength => {
                    csv.push_str(&bfs_result.path_lengths[row].to_string());
                }
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

/// Extract a single node row index from a WHERE filter text.
/// Similar to `extract_node_id` but takes a filter string directly rather
/// than requiring the node to have a filter.
fn extract_node_id_from_filter(
    node: &NodePattern,
    table_name: &str,
    filter_text: &str,
    session: &Session,
) -> Result<i64, SqlError> {
    let parts: Vec<&str> = filter_text.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(SqlError::Plan(format!(
            "Cannot extract node ID from filter: {filter_text}"
        )));
    }
    let lhs = parts[0].trim();
    if lhs.ends_with('!') || lhs.ends_with('>') || lhs.ends_with('<') {
        return Err(SqlError::Plan(format!(
            "Unsupported operator in node filter: {filter_text}. Only 'col = value' is supported."
        )));
    }
    let var = node.variable.as_deref().unwrap_or("");
    let col_name = if !var.is_empty() {
        let prefixes = [format!("{var} . "), format!("{var} ."), format!("{var}.")];
        let mut s = lhs.to_string();
        for p in &prefixes {
            if let Some(stripped) = s.strip_prefix(p.as_str()) {
                s = stripped.to_string();
                break;
            }
        }
        s.trim().to_lowercase()
    } else {
        lhs.to_lowercase()
    };
    let value = parts[1].trim();
    let value_nospace: String = value.chars().filter(|c| !c.is_whitespace()).collect();

    let stored = session.tables.get(table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{table_name}' not found"))
    })?;

    let nrows = checked_logical_nrows(stored)?;
    let col_idx = find_col_idx(&stored.table, &col_name).ok_or_else(|| {
        SqlError::Plan(format!("Column '{col_name}' not found in '{table_name}'"))
    })?;

    let str_val = if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        Some(value[1..value.len() - 1].replace("''", "'"))
    } else {
        None
    };

    let int_val = if str_val.is_none() { value_nospace.parse::<i64>().ok() } else { None };
    let float_val = if str_val.is_none() && int_val.is_none() {
        value_nospace.parse::<f64>().ok()
    } else {
        None
    };

    let mut found_row: Option<usize> = None;
    for row in 0..nrows {
        let matched = if let Some(ref sv) = str_val {
            stored.table.get_str(col_idx, row).as_deref() == Some(sv.as_str())
        } else if let Some(iv) = int_val {
            stored.table.get_i64(col_idx, row) == Some(iv)
        } else if let Some(fv) = float_val {
            stored.table.get_f64(col_idx, row) == Some(fv)
        } else {
            false
        };

        if matched {
            if found_row.is_some() {
                return Err(SqlError::Plan(format!(
                    "Ambiguous node filter: '{filter_text}' matches multiple rows in '{table_name}'."
                )));
            }
            found_row = Some(row);
        }
    }

    found_row.map(|r| r as i64).ok_or_else(|| {
        SqlError::Plan(format!("No matching row for filter: {filter_text}"))
    })
}

/// Plan a multi-hop fixed MATCH: (a)-[e1]->(b)-[e2]->(c) ... where all edges
/// have `PathQuantifier::One`. Uses `wco_join` (Leapfrog Triejoin) to find
/// matching node tuples.
fn plan_multi_hop_fixed(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
    _mode: &PathMode,
) -> Result<(Table, Vec<String>), SqlError> {
    let nodes = &pattern.nodes;
    let edges = &pattern.edges;

    // Collect per-segment info: resolve each edge label and validate node-table
    // continuity between adjacent segments.
    struct Segment<'a> {
        stored_rel: &'a StoredRel,
        src_table_name: String,
        dst_table_name: String,
    }
    let mut segments: Vec<Segment> = Vec::with_capacity(edges.len());

    for (i, edge) in edges.iter().enumerate() {
        let left_node = &nodes[i];
        let right_node = &nodes[i + 1];

        let edge_label = edge.label.as_deref().ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge {} in multi-hop pattern must specify a label: -[:Label]->",
                i
            ))
        })?;
        let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
            SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
        })?;

        // Determine left/right default tables based on edge direction
        let is_reverse = edge.direction == MatchDirection::Reverse;
        let (left_default_table, right_default_table) = if is_reverse {
            (
                &stored_rel.edge_label.dst_ref_table,
                &stored_rel.edge_label.src_ref_table,
            )
        } else {
            (
                &stored_rel.edge_label.src_ref_table,
                &stored_rel.edge_label.dst_ref_table,
            )
        };

        let left_label = resolve_node_label(left_node, left_default_table, graph)?;
        let right_label = resolve_node_label(right_node, right_default_table, graph)?;

        // Validate explicit node labels match edge expectations
        validate_node_table_for_edge(
            left_node,
            &left_label.table_name,
            left_default_table,
            edge_label,
            "source",
        )?;
        validate_node_table_for_edge(
            right_node,
            &right_label.table_name,
            right_default_table,
            edge_label,
            "destination",
        )?;

        // Verify vertex table continuity: segment[i].dst == segment[i+1].src
        if let Some(prev) = segments.last() {
            if prev.dst_table_name != left_label.table_name {
                return Err(SqlError::Plan(format!(
                    "Vertex table mismatch between segment {} and {}: '{}' vs '{}'. \
                     Adjacent segments must share a vertex table.",
                    i - 1,
                    i,
                    prev.dst_table_name,
                    left_label.table_name
                )));
            }
        }

        // Reject undirected edges on heterogeneous segments (same limitation as single-hop)
        if edge.direction == MatchDirection::Undirected
            && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
        {
            return Err(SqlError::Plan(
                "Undirected traversals are not supported on edges with different source and \
                 destination vertex tables (heterogeneous graphs). Use directed patterns instead."
                    .into(),
            ));
        }

        segments.push(Segment {
            stored_rel,
            src_table_name: left_label.table_name.clone(),
            dst_table_name: right_label.table_name.clone(),
        });
    }

    // Detect cyclic variable binding: first and last node share the same variable name.
    // Cyclic patterns require special handling in wco_join (a closing edge) which is
    // not yet implemented — reject for now (handled by Task 4 / plan_multi_hop_variable).
    let is_cyclic = match (&nodes[0].variable, &nodes[nodes.len() - 1].variable) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    if is_cyclic {
        return Err(SqlError::Plan(
            "Cyclic variable binding in fixed-hop multi-edge patterns is not yet supported. \
             Use variable-length patterns for cycles.".into(),
        ));
    }

    // Reject WHERE filters on intermediate/destination nodes — not yet supported.
    for node in &nodes[1..] {
        if node.filter.is_some() {
            return Err(SqlError::Plan(
                "WHERE filters on intermediate or destination nodes are not yet supported \
                 in multi-hop patterns. Use a WHERE clause in the outer SELECT instead.".into(),
            ));
        }
    }

    // n_vars: number of distinct node positions
    let n_nodes = nodes.len();
    let n_vars = n_nodes;

    // Build the wco_join Rel references.
    // For a chain pattern, rel[i] connects var i -> var i+1 (as the C LFTJ expects).
    let rel_refs: Vec<&Rel> = segments.iter().map(|s| &s.stored_rel.rel).collect();

    let n_vars_u8: u8 = n_vars
        .try_into()
        .map_err(|_| SqlError::Plan("Too many node variables in pattern".into()))?;

    // Build the graph on the first segment's source vertex table
    let first_src_table = session
        .tables
        .get(&segments[0].src_table_name)
        .ok_or_else(|| {
            SqlError::Plan(format!(
                "Table '{}' not found",
                segments[0].src_table_name
            ))
        })?;

    let g = session.ctx.graph(&first_src_table.table)?;
    let wco_result_col = g.wco_join(&rel_refs, n_vars_u8)?;
    let wco_result = g.execute(wco_result_col)?;

    // If the source node has a WHERE filter, apply it by filtering the wco_result
    // rows based on the _v0 column values.
    // We need to filter the result table rather than pre-filtering input since
    // wco_join operates on the full CSR.

    let nrows = checked_nrows(&wco_result)?;

    // Build a mapping: node variable name -> var index (position in the pattern)
    let mut var_map: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some(ref var) = node.variable {
            let idx = if is_cyclic && i == n_nodes - 1 { 0 } else { i };
            var_map.entry(var.clone()).or_insert(idx);
        }
    }

    // Collect _v0, _v1, ... column indices from wco_result
    let mut v_col_indices: Vec<usize> = Vec::with_capacity(n_vars);
    for v in 0..n_vars {
        let col_name = format!("_v{v}");
        let idx = find_col_idx(&wco_result, &col_name).ok_or_else(|| {
            SqlError::Plan(format!("wco_join result missing column '{col_name}'"))
        })?;
        v_col_indices.push(idx);
    }

    // Read all node ID arrays from wco_result
    let mut node_ids: Vec<Vec<i64>> = Vec::with_capacity(n_vars);
    for v in 0..n_vars {
        let mut ids = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let val = wco_result.get_i64(v_col_indices[v], row).ok_or_else(|| {
                SqlError::Plan(format!("NULL _v{v} at row {row}"))
            })?;
            ids.push(val);
        }
        node_ids.push(ids);
    }

    // Apply source node (v0) WHERE filter if present
    let src_filter = nodes[0].filter.as_deref();
    let row_mask: Vec<bool> = if let Some(filter_text) = src_filter {
        // Parse the filter to extract column name and value
        let parts: Vec<&str> = filter_text.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(SqlError::Plan(format!(
                "Unsupported node filter syntax: {filter_text}. Only 'col = value' is supported."
            )));
        }
        let lhs = parts[0].trim();
        if lhs.ends_with('!') || lhs.ends_with('>') || lhs.ends_with('<') {
            return Err(SqlError::Plan(format!(
                "Unsupported operator in node filter: {filter_text}. Only 'col = value' is supported."
            )));
        }
        let col_name = if let Some(var) = nodes[0].variable.as_deref() {
            let prefixes = [format!("{var} . "), format!("{var} ."), format!("{var}.")];
            let mut s = lhs.to_string();
            for p in &prefixes {
                if let Some(stripped) = s.strip_prefix(p.as_str()) {
                    s = stripped.to_string();
                    break;
                }
            }
            s.trim().to_lowercase()
        } else {
            lhs.to_lowercase()
        };
        let value = parts[1].trim();

        let src_table = session
            .tables
            .get(&segments[0].src_table_name)
            .ok_or_else(|| {
                SqlError::Plan(format!(
                    "Table '{}' not found",
                    segments[0].src_table_name
                ))
            })?;
        let filter_col_idx = find_col_idx(&src_table.table, &col_name).ok_or_else(|| {
            SqlError::Plan(format!(
                "Column '{col_name}' not found in source table '{}'",
                segments[0].src_table_name
            ))
        })?;

        // For each row in wco_result, check if the source node's column matches the filter value
        let mut mask = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let val = node_ids[0][row];
            if val < 0 {
                return Err(SqlError::Plan(format!("Negative node index {val} for _v0 at row {row}")));
            }
            let node_row = val as usize;
            let matches = if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
                let s = value[1..value.len() - 1].replace("''", "'");
                src_table
                    .table
                    .get_str(filter_col_idx, node_row)
                    .map(|v| v == s)
                    .unwrap_or(false)
            } else if let Ok(int_val) = value.parse::<i64>() {
                src_table
                    .table
                    .get_i64(filter_col_idx, node_row)
                    .map(|v| v == int_val)
                    .unwrap_or(false)
            } else {
                let value_nospace: String = value.chars().filter(|c| !c.is_whitespace()).collect();
                if let Ok(int_val) = value_nospace.parse::<i64>() {
                    src_table
                        .table
                        .get_i64(filter_col_idx, node_row)
                        .map(|v| v == int_val)
                        .unwrap_or(false)
                } else {
                    return Err(SqlError::Plan(format!(
                        "Unsupported filter value: {value}"
                    )));
                }
            };
            mask.push(matches);
        }
        mask
    } else {
        vec![true; nrows]
    };

    // Build a lookup: for each node variable position, which vertex table does it reference?
    let mut var_table_names: Vec<String> = Vec::with_capacity(n_vars);
    var_table_names.push(segments[0].src_table_name.clone());
    for seg in &segments {
        var_table_names.push(seg.dst_table_name.clone());
    }
    // If cyclic, the last entry maps back to var 0 (already handled by truncating to n_vars)
    var_table_names.truncate(n_vars);

    // Project COLUMNS
    struct ColSpec {
        var_idx: usize,    // which _v column to read the node ID from
        table_col_idx: usize, // which column in the vertex table to read
        table_name: String,
    }
    let mut col_names: Vec<String> = Vec::new();
    let mut col_specs: Vec<ColSpec> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            let var_idx = var_map.get(&var).ok_or_else(|| {
                let available: Vec<_> = var_map.keys().collect();
                SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available:?}"
                ))
            })?;

            let table_name = &var_table_names[*var_idx];
            let vtable = session.tables.get(table_name).ok_or_else(|| {
                SqlError::Plan(format!("Table '{table_name}' not found"))
            })?;
            let table_col_idx = find_col_idx(&vtable.table, &col).ok_or_else(|| {
                SqlError::Plan(format!(
                    "Column '{col}' not found in vertex table '{table_name}'"
                ))
            })?;

            col_names.push(out_name);
            col_specs.push(ColSpec {
                var_idx: *var_idx,
                table_col_idx,
                table_name: table_name.clone(),
            });
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS expression must be in 'var.col' format, got: {expr}"
            )));
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Build result via CSV
    let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();

    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for row in 0..nrows {
        if !row_mask[row] {
            continue;
        }
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            let val = node_ids[spec.var_idx][row];
            if val < 0 {
                return Err(SqlError::Plan(format!(
                    "Negative node index {} for _v{} at row {}", val, spec.var_idx, row
                )));
            }
            let node_row = val as usize;
            let vtable = session.tables.get(&spec.table_name).ok_or_else(|| {
                SqlError::Plan(format!("Table '{}' not found", spec.table_name))
            })?;
            csv.push_str(&get_cell_string(&vtable.table, spec.table_col_idx, node_row)?);
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

// ---------------------------------------------------------------------------
// Variable-length path planner
// ---------------------------------------------------------------------------

/// Plan a variable-length MATCH: (a)-[e]->{min,max}(b)
fn plan_var_length(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let src_node = &pattern.nodes[0];
    let edge = &pattern.edges[0];
    let dst_node = &pattern.nodes[1];

    // Destination-node filters are not supported in variable-length patterns
    if dst_node.filter.is_some() {
        return Err(SqlError::Plan(
            "WHERE filters on destination nodes are not yet supported in variable-length patterns. \
             Use a WHERE clause in the outer SELECT instead."
                .into(),
        ));
    }

    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
    })?;

    // For reverse edges, swap default table/column references (same as plan_single_hop)
    let is_reverse = edge.direction == MatchDirection::Reverse;
    let (left_default_table, right_default_table, scan_ref_col) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_col)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_col)
    };

    // Multi-hop traversals require src and dst to reference the same vertex table
    // (node IDs must be in the same domain across hops).
    // Range{1,1} is semantically single-hop, so skip this check for it.
    let is_single_range = matches!(edge.quantifier, PathQuantifier::Range { min: 1, max: 1 });
    if !is_single_range && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table {
        return Err(SqlError::Plan(
            "Variable-length patterns are not supported on edges with different source and \
             destination vertex tables (heterogeneous graphs). Use single-hop patterns instead."
                .into(),
        ));
    }

    let src_label = resolve_node_label(src_node, left_default_table, graph)?;
    validate_node_table_for_edge(src_node, &src_label.table_name, left_default_table, edge_label, "source")?;
    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;

    let dst_label = resolve_node_label(dst_node, right_default_table, graph)?;
    validate_node_table_for_edge(dst_node, &dst_label.table_name, right_default_table, edge_label, "destination")?;
    let dst_table = &session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?.table;

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    let (min_depth, max_depth) = match edge.quantifier {
        PathQuantifier::Range { min, max } => (min, max),
        PathQuantifier::Plus => (1, 255),
        PathQuantifier::Star => (0, 255),
        PathQuantifier::One => (1, 1),
    };

    let mut g = session.ctx.graph(src_table)?;
    let src_ids = g.scan(scan_ref_col)?;

    let src_ids = if let Some(filter_text) = &src_node.filter {
        apply_node_filter(&mut g, src_ids, filter_text, src_node.variable.as_deref())?
    } else {
        src_ids
    };

    let var_exp = g.var_expand(src_ids, &stored_rel.rel, direction, min_depth, max_depth, false)?;
    let result = g.execute(var_exp)?;

    // var_expand result has: _start, _end, _depth
    // Project the requested COLUMNS using _start/_end to look up properties
    project_var_length_columns(session, &result, columns, src_node, dst_node, src_table, dst_table)
}

/// Project COLUMNS from var_expand results.
/// var_expand output: _start, _end, _depth
fn project_var_length_columns(
    session: &Session,
    var_result: &Table,
    columns: &[ColumnEntry],
    src_node: &NodePattern,
    dst_node: &NodePattern,
    src_table: &Table,
    dst_table: &Table,
) -> Result<(Table, Vec<String>), SqlError> {
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    let nrows = checked_nrows(var_result)?;

    let start_idx_col = find_col_idx(var_result, "_start")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _start column".into()))?;
    let end_idx_col = find_col_idx(var_result, "_end")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _end column".into()))?;
    let depth_idx_col = find_col_idx(var_result, "_depth");

    let mut start_indices = Vec::with_capacity(nrows);
    let mut end_indices = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let start_val = var_result.get_i64(start_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _start index at row {row}")))?;
        if start_val < 0 {
            return Err(SqlError::Plan(format!("Negative _start index {start_val} at row {row}")));
        }
        start_indices.push(start_val as usize);
        let end_val = var_result.get_i64(end_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _end index at row {row}")))?;
        if end_val < 0 {
            return Err(SqlError::Plan(format!("Negative _end index {end_val} at row {row}")));
        }
        end_indices.push(end_val as usize);
    }

    // Build column specs in COLUMNS clause order
    let mut col_names = Vec::new();
    struct ColSpec {
        kind: VarColKind,
        table_col_idx: usize, // index into src_table or dst_table (unused for Depth)
    }
    enum VarColKind {
        Src,
        Dst,
        Depth,
    }
    let mut col_specs: Vec<ColSpec> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            if var == src_var {
                let col_idx = find_col_idx(src_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in source table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { kind: VarColKind::Src, table_col_idx: col_idx });
            } else if var == dst_var {
                let col_idx = find_col_idx(dst_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in destination table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { kind: VarColKind::Dst, table_col_idx: col_idx });
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS"
                )));
            }
        } else {
            let lower = expr.to_lowercase();
            if lower.contains("path_length") || lower == "_depth" {
                col_names.push(alias.unwrap_or("depth").to_string());
                col_specs.push(ColSpec { kind: VarColKind::Depth, table_col_idx: 0 });
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{expr}'"
                )));
            }
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Build CSV
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();

    let mut csv = csv_col_names.join(",");
    csv.push('\n');
    for row in 0..nrows {
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            match spec.kind {
                VarColKind::Src => {
                    csv.push_str(&get_cell_string(src_table, spec.table_col_idx, start_indices[row])?);
                }
                VarColKind::Dst => {
                    csv.push_str(&get_cell_string(dst_table, spec.table_col_idx, end_indices[row])?);
                }
                VarColKind::Depth => {
                    if let Some(d_col) = depth_idx_col {
                        let d = var_result.get_i64(d_col, row).unwrap_or(0);
                        csv.push_str(&d.to_string());
                    } else {
                        csv.push('0');
                    }
                }
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

// ---------------------------------------------------------------------------
// Shortest path planner
// ---------------------------------------------------------------------------

/// Plan an ANY SHORTEST MATCH using BFS.
///
/// Reconstructs the shortest path by doing BFS over the edge table's
/// adjacency list, tracking predecessors to build the path.
fn plan_shortest_path(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let src_node = &pattern.nodes[0];
    let edge = &pattern.edges[0];
    let dst_node = &pattern.nodes[1];

    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found"))
    })?;

    // For reverse edges, swap default table/column references
    let is_reverse = edge.direction == MatchDirection::Reverse;
    let (left_table, left_col, right_table, right_col) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.dst_ref_col,
         &stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.src_ref_col)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.src_ref_col,
         &stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.dst_ref_col)
    };

    // Validate explicit node labels against the edge's expected tables.
    if let Some(label) = &src_node.label {
        let src_vl = graph.vertex_labels.get(label).ok_or_else(|| {
            SqlError::Plan(format!("Vertex label '{label}' not found in graph"))
        })?;
        if src_vl.table_name != *left_table {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' resolves to table '{}', but edge '{edge_label}' \
                 expects source table '{left_table}'",
                src_vl.table_name
            )));
        }
    }
    if let Some(label) = &dst_node.label {
        let dst_vl = graph.vertex_labels.get(label).ok_or_else(|| {
            SqlError::Plan(format!("Vertex label '{label}' not found in graph"))
        })?;
        if dst_vl.table_name != *right_table {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' resolves to table '{}', but edge '{edge_label}' \
                 expects destination table '{right_table}'",
                dst_vl.table_name
            )));
        }
    }

    // Shortest-path BFS requires src and dst to reference the same vertex table.
    // Range{1,1} and PathQuantifier::One are semantically single-hop, so skip
    // this check for them (same exemption as plan_var_length).
    let is_single_hop = matches!(edge.quantifier, PathQuantifier::One | PathQuantifier::Range { min: 1, max: 1 });
    if !is_single_hop && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table {
        return Err(SqlError::Plan(
            "SHORTEST_PATH is not supported on edges with different source and \
             destination vertex tables (heterogeneous graphs)."
                .into(),
        ));
    }

    let src_id = extract_node_id(src_node, left_table, left_col, session)?;
    let dst_id = extract_node_id(dst_node, right_table, right_col, session)?;

    let (min_depth, max_depth): (u8, u8) = match edge.quantifier {
        PathQuantifier::Range { min, max } => (min, max),
        PathQuantifier::Plus => (1, 255),
        PathQuantifier::Star => (0, 255),
        PathQuantifier::One => (1, 1),
    };

    // Handle 0-hop match: if min_depth is 0 and src == dst, return immediately
    if min_depth == 0 && src_id == dst_id {
        // Validate columns the same way as the normal path
        let mut col_names: Vec<String> = Vec::new();
        // 0 = _node, 1 = _depth, 2 = path_length (total)
        let mut col_indices: Vec<usize> = Vec::new();
        for entry in columns {
            let lower = entry.expr.to_lowercase();
            let alias = entry.alias.as_deref();
            if lower.contains("path_length") {
                col_names.push(alias.unwrap_or("path_length").to_string());
                col_indices.push(2); // total path length
            } else if lower == "_node" || lower == "node" {
                col_names.push(alias.map(|s| s.to_string()).unwrap_or_else(|| lower.clone()));
                col_indices.push(0);
            } else if lower == "_depth" || lower == "depth" {
                col_names.push(alias.map(|s| s.to_string()).unwrap_or_else(|| lower.clone()));
                col_indices.push(1);
            } else if let Some(dot_pos) = lower.find('.') {
                let var = lower[..dot_pos].trim();
                let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
                if var == dst_var {
                    return Err(SqlError::Plan(
                        "SHORTEST_PATH COLUMNS: use _node, _depth, or path_length(p). \
                         Property lookups on path nodes are not yet supported."
                            .into(),
                    ));
                }
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in SHORTEST_PATH COLUMNS"
                )));
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{}'",
                    entry.expr
                )));
            }
        }
        if col_names.is_empty() {
            return Err(SqlError::Plan("COLUMNS clause is empty".into()));
        }
        // Build CSV with correct values: node=src_id, depth=0, path_length=0
        let csv_col_names: Vec<String> = (0..col_names.len())
            .map(|i| format!("__c{i}"))
            .collect();
        let mut csv = csv_col_names.join(",");
        csv.push('\n');
        for (i, &col_idx) in col_indices.iter().enumerate() {
            if i > 0 { csv.push(','); }
            match col_idx {
                0 => csv.push_str(&src_id.to_string()),
                1 | 2 => csv.push('0'), // depth and path_length are both 0 for 0-hop
                _ => {}
            }
        }
        csv.push('\n');
        let result = csv_to_table(session, &csv, &col_names)?;
        return Ok((result, col_names));
    }

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // BFS over the edge table to find the shortest qualifying path
    let mut path = reconstruct_shortest_path(
        session, src_id, dst_id, min_depth as i64, max_depth as i64, stored_rel, direction,
    )?;

    if path.is_empty() {
        // No path found → return empty result table with proper column names
        // so that the outer query can resolve column references (e.g. ORDER BY _depth).
        // Validate columns the same way as the non-empty branch below.
        let mut display_names: Vec<String> = Vec::new();
        for entry in columns {
            let lower = entry.expr.to_lowercase();
            let alias = entry.alias.as_deref();
            if lower.contains("path_length") {
                display_names.push(alias.unwrap_or("path_length").to_string());
            } else if lower == "_node" || lower == "node"
                || lower == "_depth" || lower == "depth"
            {
                display_names.push(alias.unwrap_or(&lower).to_string());
            } else if let Some(dot_pos) = lower.find('.') {
                let var = lower[..dot_pos].trim();
                let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
                if var == dst_var {
                    return Err(SqlError::Plan(
                        "SHORTEST_PATH COLUMNS: use _node, _depth, or path_length(p). \
                         Property lookups on path nodes are not yet supported."
                            .into(),
                    ));
                }
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in SHORTEST_PATH COLUMNS"
                )));
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{}'",
                    entry.expr
                )));
            }
        }
        if display_names.is_empty() {
            return Err(SqlError::Plan("COLUMNS clause is empty".into()));
        }
        let csv_col_names: Vec<String> = (0..display_names.len())
            .map(|i| format!("__c{i}"))
            .collect();
        let csv = format!("{}\n", csv_col_names.join(","));
        let result = csv_to_table(session, &csv, &display_names)?;
        return Ok((result, display_names));
    }

    path.reverse(); // path is built backwards, reverse to get src -> dst order

    // Build result as CSV with _node and _depth columns
    let mut col_names: Vec<String> = Vec::new();
    // 0 = _node, 1 = _depth, 2 = path_length (total)
    let mut col_indices: Vec<usize> = Vec::new();

    for entry in columns {
        let lower = entry.expr.to_lowercase();
        let alias = entry.alias.as_deref();

        if lower.contains("path_length") {
            col_names.push(alias.unwrap_or("path_length").to_string());
            col_indices.push(2); // total path length (constant per row)
        } else if lower == "_node" || lower == "node" {
            col_names.push(alias.map(|s| s.to_string()).unwrap_or_else(|| entry.expr.to_lowercase()));
            col_indices.push(0); // node
        } else if lower == "_depth" || lower == "depth" {
            col_names.push(alias.map(|s| s.to_string()).unwrap_or_else(|| entry.expr.to_lowercase()));
            col_indices.push(1); // depth
        } else if let Some(dot_pos) = lower.find('.') {
            let var = lower[..dot_pos].trim();
            let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
            if var == dst_var {
                return Err(SqlError::Plan(
                    "SHORTEST_PATH COLUMNS: use _node, _depth, or path_length(p). \
                     Property lookups on path nodes are not yet supported."
                        .into(),
                ));
            }
            return Err(SqlError::Plan(format!(
                "Unknown variable '{var}' in SHORTEST_PATH COLUMNS"
            )));
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS: unsupported expression '{}'",
                entry.expr
            )));
        }
    }

    if col_names.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Total path length = number of edges = number of nodes - 1
    let total_path_length = if path.is_empty() { 0 } else { path.len() - 1 };

    // Build CSV
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for (depth, &node_id) in path.iter().enumerate() {
        for (i, &col_idx) in col_indices.iter().enumerate() {
            if i > 0 { csv.push(','); }
            match col_idx {
                0 => csv.push_str(&node_id.to_string()),
                1 => csv.push_str(&depth.to_string()),
                2 => csv.push_str(&total_path_length.to_string()),
                _ => {}
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

/// Reconstruct shortest path using BFS over the edge table.
/// Returns the path as a Vec of node IDs from dst back to src (reversed).
/// Only returns a path whose hop count is between min_depth and max_depth (inclusive).
fn reconstruct_shortest_path(
    session: &Session,
    src_id: i64,
    dst_id: i64,
    min_depth: i64,
    max_depth: i64,
    stored_rel: &StoredRel,
    direction: u8,
) -> Result<Vec<i64>, SqlError> {
    use std::collections::{HashMap as HM, HashSet, VecDeque};

    // Get all edges for adjacency lookup
    let edge_table_name = &stored_rel.edge_label.table_name;
    let edge_stored = session.tables.get(edge_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge table '{}' not found", edge_table_name))
    })?;

    let src_col_name = &stored_rel.edge_label.src_col;
    let dst_col_name = &stored_rel.edge_label.dst_col;
    let src_col_idx = find_col_idx(&edge_stored.table, src_col_name)
        .ok_or_else(|| SqlError::Plan(format!("Column '{src_col_name}' not found in edge table")))?;
    let dst_col_idx = find_col_idx(&edge_stored.table, dst_col_name)
        .ok_or_else(|| SqlError::Plan(format!("Column '{dst_col_name}' not found in edge table")))?;

    // Build adjacency list from edge table
    let n_edges = checked_logical_nrows(edge_stored)?;
    let mut adj: HM<i64, Vec<i64>> = HM::new();
    for row in 0..n_edges {
        let s = match edge_stored.table.get_i64(src_col_idx, row) {
            Some(v) => v,
            None => continue, // skip edges with NULL endpoints
        };
        let d = match edge_stored.table.get_i64(dst_col_idx, row) {
            Some(v) => v,
            None => continue,
        };
        match direction {
            0 => { adj.entry(s).or_default().push(d); } // forward
            1 => { adj.entry(d).or_default().push(s); } // reverse
            _ => { // undirected
                adj.entry(s).or_default().push(d);
                adj.entry(d).or_default().push(s);
            }
        }
    }

    // BFS keyed on (node, depth) to allow revisiting nodes at different depths.
    // This is needed when min_depth > shortest_depth: the shortest path to an
    // intermediate node may be too short, but a longer path through that node
    // may reach dst_id at the required depth.
    const MAX_BFS_STATES: usize = 1_000_000;
    let mut visited: HashSet<(i64, i64)> = HashSet::new();
    // predecessor map: (node, depth) -> predecessor_node
    let mut pred_map: HM<(i64, i64), i64> = HM::new();
    visited.insert((src_id, 0));
    let mut frontier = VecDeque::new();
    frontier.push_back((src_id, 0i64));

    while let Some((node, depth)) = frontier.pop_front() {
        if visited.len() > MAX_BFS_STATES {
            return Err(SqlError::Plan(format!(
                "SHORTEST_PATH BFS exceeded {MAX_BFS_STATES} states — graph too large or \
                 path quantifier range too wide. Try narrowing the hop range."
            )));
        }
        if node == dst_id && depth >= min_depth {
            // Reconstruct path by following pred_map backwards
            let mut path = vec![dst_id];
            let mut cur_node = dst_id;
            let mut cur_depth = depth;
            while cur_depth > 0 {
                let prev = pred_map[&(cur_node, cur_depth)];
                path.push(prev);
                cur_node = prev;
                cur_depth -= 1;
            }
            return Ok(path);
        }
        if depth >= max_depth {
            continue;
        }
        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors {
                let next_depth = depth + 1;
                if visited.insert((next, next_depth)) {
                    pred_map.insert((next, next_depth), node);
                    frontier.push_back((next, next_depth));
                }
            }
        }
    }

    Ok(Vec::new()) // no path found → empty result
}

/// Extract a node's row index from a WHERE filter.
/// Delegates to `extract_node_id_from_filter` after unwrapping the filter text.
fn extract_node_id(
    node: &NodePattern,
    table_name: &str,
    _key_col: &str,
    session: &Session,
) -> Result<i64, SqlError> {
    let filter = node.filter.as_deref().ok_or_else(|| {
        SqlError::Plan(
            "SHORTEST_PATH requires WHERE filters on both source and destination nodes".into(),
        )
    })?;
    extract_node_id_from_filter(node, table_name, filter, session)
}

// ---------------------------------------------------------------------------
// Graph algorithm planner
// ---------------------------------------------------------------------------

/// Known graph algorithm function names.
const ALGO_FUNCTIONS: &[&str] = &["pagerank", "component", "connected_component", "community", "louvain", "shortest_distance", "dijkstra"];

/// Parse a COLUMNS expression to check if it's a graph algorithm function call.
/// Returns `Some((func_name, args))` if the expression matches `FUNC(arg1, arg2, ...)`.
fn parse_algo_function(expr: &str) -> Option<(String, Vec<String>)> {
    let trimmed = expr.trim();
    let open = trimmed.find('(')?;
    let func_name = trimmed[..open].trim().to_lowercase();
    if !ALGO_FUNCTIONS.contains(&func_name.as_str()) {
        return None;
    }
    // Find matching closing paren
    let inner = trimmed[open + 1..].strip_suffix(')')?.trim().to_string();
    if inner.is_empty() {
        return Some((func_name, Vec::new()));
    }
    let args: Vec<String> = inner.split(',').map(|s| s.trim().to_lowercase()).collect();
    Some((func_name, args))
}

/// Plan a single-node pattern with algorithm function calls in COLUMNS.
/// Handles patterns like: (p:Person) COLUMNS (PAGERANK(social, p) AS rank)
fn plan_algorithm_query(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let node = &pattern.nodes[0];

    // WHERE filters are not supported on algorithm queries -- algorithms operate
    // on the entire graph.  Reject early rather than silently ignoring the filter.
    if node.filter.is_some() {
        return Err(SqlError::Plan(
            "WHERE filters are not supported in algorithm MATCH patterns; \
             use a WHERE clause in the outer SELECT to filter results"
                .into(),
        ));
    }

    // Resolve vertex label for the node
    let vertex_label = if let Some(label) = &node.label {
        graph.vertex_labels.get(label).ok_or_else(|| {
            SqlError::Plan(format!("Vertex label '{label}' not found in graph"))
        })?
    } else if graph.vertex_labels.len() == 1 {
        graph.vertex_labels.values().next().ok_or_else(|| {
            SqlError::Plan("Graph has no vertex labels".into())
        })?
    } else if graph.vertex_labels.is_empty() {
        return Err(SqlError::Plan("Graph has no vertex labels".into()));
    } else {
        return Err(SqlError::Plan(
            "Graph has multiple vertex labels; specify one explicitly in the MATCH pattern (e.g., (n:Label))".into(),
        ));
    };

    let vertex_stored = session.tables.get(&vertex_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", vertex_label.table_name))
    })?;
    let nrows = checked_logical_nrows(vertex_stored)?;

    // Determine edge label: use the only one, or require specification
    let default_edge_label = if graph.edge_labels.len() == 1 {
        graph.edge_labels.keys().next().cloned()
    } else {
        None
    };

    // Process each COLUMNS entry - collect algorithm results and property lookups
    let mut col_names: Vec<String> = Vec::new();
    enum AlgoColKind {
        AlgoResult { algo_col_idx: usize },  // index into algorithm result table
        Property { table_col_idx: usize },    // index into vertex table
    }
    let mut col_specs: Vec<AlgoColKind> = Vec::new();
    let mut algo_result: Option<Table> = None;
    let mut algo_func_used: Option<String> = None;

    let node_var = node.variable.as_deref().unwrap_or("__node");

    for entry in columns {
        let alias = entry.alias.as_deref();

        if let Some((func_name, args)) = parse_algo_function(&entry.expr) {
            let is_dijkstra = matches!(func_name.as_str(), "shortest_distance" | "dijkstra");

            if is_dijkstra {
                // SHORTEST_DISTANCE(graph, src_id, dst_id, weight_col)
                if args.len() != 4 {
                    return Err(SqlError::Plan(format!(
                        "SHORTEST_DISTANCE expects 4 arguments: SHORTEST_DISTANCE({}, src_id, dst_id, 'weight_col'). Got {} argument(s).",
                        graph.name, args.len(),
                    )));
                }
                if args[0] != graph.name.to_lowercase() {
                    return Err(SqlError::Plan(format!(
                        "First argument '{}' does not match graph name '{}'.",
                        args[0], graph.name,
                    )));
                }
            } else {
                // Standard algorithms: FUNC(graph_name, node_var)
                if args.len() != 2 {
                    return Err(SqlError::Plan(format!(
                        "Algorithm function expects exactly 2 arguments: {}({}, {}). Got {} argument(s).",
                        func_name.to_uppercase(), graph.name, node_var, args.len(),
                    )));
                }
                if args[0] != graph.name.to_lowercase() {
                    return Err(SqlError::Plan(format!(
                        "Algorithm argument '{}' does not match graph name '{}'. \
                         Expected: {}({}, {})",
                        args[0], graph.name, func_name.to_uppercase(), graph.name, node_var,
                    )));
                }
                if args[1] != node_var {
                    return Err(SqlError::Plan(format!(
                        "Algorithm argument '{}' does not match node variable '{}'. \
                         Expected: {}({}, {})",
                        args[1], node_var, func_name.to_uppercase(), graph.name, node_var,
                    )));
                }
            }
            // Reject mixing different algorithms in the same COLUMNS clause
            if let Some(ref prev) = algo_func_used {
                let prev_canonical = match prev.as_str() {
                    "component" | "connected_component" => "connected_component",
                    "community" | "louvain" => "louvain",
                    "shortest_distance" | "dijkstra" => "dijkstra",
                    other => other,
                };
                let cur_canonical = match func_name.as_str() {
                    "component" | "connected_component" => "connected_component",
                    "community" | "louvain" => "louvain",
                    "shortest_distance" | "dijkstra" => "dijkstra",
                    other => other,
                };
                if prev_canonical != cur_canonical {
                    return Err(SqlError::Plan(format!(
                        "Cannot mix different algorithms ({prev} and {func_name}) in the same COLUMNS clause. \
                         Use separate GRAPH_TABLE queries for each algorithm."
                    )));
                }
            }
            algo_func_used = Some(func_name.clone());

            // Execute algorithm if not already done
            if algo_result.is_none() {
                let edge_label_name = default_edge_label.as_deref().ok_or_else(|| {
                    SqlError::Plan(
                        "Graph has multiple edge labels; cannot auto-select for algorithm. \
                         Please use a graph with a single edge label.".into()
                    )
                })?;
                if is_dijkstra {
                    algo_result = Some(execute_dijkstra_algorithm(
                        session, graph, edge_label_name, &args,
                    )?);
                } else {
                    algo_result = Some(execute_graph_algorithm(
                        session, graph, &func_name, edge_label_name,
                    )?);
                }
            }

            // Map function name to result column
            let result_col_name = match func_name.as_str() {
                "pagerank" => "_rank",
                "component" | "connected_component" => "_component",
                "community" | "louvain" => "_community",
                "shortest_distance" | "dijkstra" => "_dist",
                _ => return Err(SqlError::Plan(format!("Unknown algorithm: {func_name}"))),
            };
            let default_alias = result_col_name.trim_start_matches('_');
            let out_name = alias.unwrap_or(default_alias).to_string();

            let result_table = algo_result.as_ref().unwrap();
            let algo_col = find_col_idx(result_table, result_col_name).ok_or_else(|| {
                SqlError::Plan(format!(
                    "Algorithm result missing column '{result_col_name}'"
                ))
            })?;
            col_names.push(out_name);
            col_specs.push(AlgoColKind::AlgoResult { algo_col_idx: algo_col });
        } else if let Some(dot_pos) = entry.expr.find('.') {
            let var = entry.expr[..dot_pos].trim().to_lowercase();
            let col = entry.expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            if var == node_var {
                let col_idx = find_col_idx(&vertex_stored.table, &col).ok_or_else(|| {
                    SqlError::Plan(format!("Column '{col}' not found in vertex table"))
                })?;
                col_names.push(out_name);
                col_specs.push(AlgoColKind::Property { table_col_idx: col_idx });
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {node_var}"
                )));
            }
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS: unsupported expression '{}'. Expected algorithm function or var.col format.",
                entry.expr
            )));
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Validate algorithm result row count matches vertex table
    if let Some(ref result_table) = algo_result {
        let algo_nrows = checked_nrows(result_table)?;
        if algo_nrows != nrows {
            return Err(SqlError::Plan(format!(
                "Algorithm result has {algo_nrows} rows but vertex table has {nrows} rows. \
                 Ensure the vertex label matches the edge label's source vertex table."
            )));
        }
    }

    // Build CSV result
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();

    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for row in 0..nrows {
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            match spec {
                AlgoColKind::AlgoResult { algo_col_idx } => {
                    let result_table = algo_result.as_ref().unwrap();
                    csv.push_str(&get_cell_string(result_table, *algo_col_idx, row)?);
                }
                AlgoColKind::Property { table_col_idx } => {
                    csv.push_str(&get_cell_string(&vertex_stored.table, *table_col_idx, row)?);
                }
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

/// Execute a graph algorithm and return its result table.
fn execute_graph_algorithm(
    session: &Session,
    graph: &PropertyGraph,
    func_name: &str,
    edge_label_name: &str,
) -> Result<Table, SqlError> {
    let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label_name}' not found"))
    })?;

    // Need a base table to create a Graph handle
    let src_table_name = &stored_rel.edge_label.src_ref_table;
    let src_table = &session.tables.get(src_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{src_table_name}' not found"))
    })?.table;

    let g = session.ctx.graph(src_table)?;

    let result_col = match func_name {
        "pagerank" => g.pagerank(&stored_rel.rel, 20, 0.85)?,
        "component" | "connected_component" => g.connected_comp(&stored_rel.rel)?,
        "community" | "louvain" => g.louvain(&stored_rel.rel, 100)?,
        "shortest_distance" | "dijkstra" => {
            return Err(SqlError::Plan(
                "SHORTEST_DISTANCE() is not supported as a COLUMNS function in node-only MATCH patterns. \
                 Use the Rust API (Graph::dijkstra) or a two-node MATCH pattern with ANY SHORTEST instead."
                    .into(),
            ));
        }
        _ => return Err(SqlError::Plan(format!(
            "Unknown graph algorithm: {func_name}"
        ))),
    };

    let result = g.execute(result_col)?;
    Ok(result)
}

/// Execute Dijkstra's weighted shortest path algorithm.
/// Args: [graph_name, src_id, dst_id, weight_col]
fn execute_dijkstra_algorithm(
    session: &Session,
    graph: &PropertyGraph,
    edge_label_name: &str,
    args: &[String],
) -> Result<Table, SqlError> {
    let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label_name}' not found"))
    })?;

    let src_id: i64 = args[1].parse().map_err(|_| {
        SqlError::Plan(format!("Invalid source node ID: '{}'", args[1]))
    })?;
    let dst_id: i64 = args[2].parse().map_err(|_| {
        SqlError::Plan(format!("Invalid destination node ID: '{}'", args[2]))
    })?;
    // Strip surrounding quotes from weight column name
    let weight_col = args[3].trim_matches('\'').trim_matches('"');

    // Attach edge properties if not already attached
    let edge_table_name = &stored_rel.edge_label.table_name;
    let edge_table = &session.tables.get(edge_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge table '{edge_table_name}' not found"))
    })?.table;
    stored_rel.rel.set_props(edge_table);

    let src_table_name = &stored_rel.edge_label.src_ref_table;
    let src_table = &session.tables.get(src_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{src_table_name}' not found"))
    })?.table;

    let g = session.ctx.graph(src_table)?;
    let src = g.const_i64(src_id)?;
    let dst = g.const_i64(dst_id)?;
    let result_col = g.dijkstra(src, Some(dst), &stored_rel.rel, weight_col, 255)?;

    let result = g.execute(result_col)?;
    Ok(result)
}
