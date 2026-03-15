// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{Column, Graph, Rel, Table};

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

        let n_src_usize = checked_nrows(&src_stored.table).map_err(|_| SqlError::Plan(format!(
            "Source vertex table '{}' has invalid row count", et.src_ref_table
        )))?;
        let n_dst_usize = checked_nrows(&dst_stored.table).map_err(|_| SqlError::Plan(format!(
            "Destination vertex table '{}' has invalid row count", et.dst_ref_table
        )))?;
        let n_src = n_src_usize as i64;
        let n_dst = n_dst_usize as i64;

        // Validate that edge src/dst column values are 0-based row indices.
        // The C engine CSR treats these as row offsets into the vertex tables.
        let n_edges_i64 = edge_stored.table.nrows();
        if n_edges_i64 < 0 {
            return Err(SqlError::Plan(format!(
                "Edge table '{}' has invalid row count", et.table_name
            )));
        }
        let n_edges = n_edges_i64 as usize;
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
    }
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
        _ => {
            if pattern.nodes.len() >= 3 {
                Err(SqlError::Plan(
                    "Multi-hop and cyclic MATCH patterns are not yet supported. \
                     Use single-hop (a)-[e]->(b)."
                        .into(),
                ))
            } else {
                Err(SqlError::Plan(format!(
                    "Unsupported MATCH pattern: {} nodes, {} edges",
                    pattern.nodes.len(),
                    pattern.edges.len()
                )))
            }
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
    let g = session.ctx.graph(&src_stored.table)?;

    // Scan the appropriate reference column for the left-side node
    let src_ids = g.scan(scan_ref_col)?;

    // Apply source node filter if present
    let src_ids = if let Some(filter_text) = &src_node.filter {
        apply_node_filter(
            &g,
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

/// Apply a WHERE filter from a node pattern.
/// Parses filter text like "a.name = 'Alice'" and applies it.
fn apply_node_filter(
    g: &Graph,
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
    let nrows = checked_nrows(&stored.table)?;
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
fn get_cell_string(table: &Table, col: usize, row: usize) -> Result<String, SqlError> {
    // Try string first (SYM columns)
    if let Some(s) = table.get_str(col, row) {
        return Ok(csv_quote(&s));
    }
    // Try i64 (also covers BOOL which is stored as 0/1 i64)
    if let Some(v) = table.get_i64(col, row) {
        return Ok(v.to_string());
    }
    // Try f64
    if let Some(v) = table.get_f64(col, row) {
        return Ok(v.to_string());
    }
    Err(SqlError::Plan(format!(
        "Unsupported column type at col {col}, row {row} (DATE/TIME/TIMESTAMP not supported in GRAPH_TABLE)"
    )))
}

/// Quote a string value for CSV: always wrap in double quotes to preserve
/// string type through CSV round-trip (prevents "123" from being parsed as
/// an integer). Escape embedded double quotes by doubling.
fn csv_quote(s: &str) -> String {
    let escaped = s.replace('"', "\"\"");
    format!("\"{escaped}\"")
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
    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;

    let dst_label = resolve_node_label(dst_node, right_default_table, graph)?;
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

    let g = session.ctx.graph(src_table)?;
    let src_ids = g.scan(scan_ref_col)?;

    let src_ids = if let Some(filter_text) = &src_node.filter {
        apply_node_filter(&g, src_ids, filter_text, src_node.variable.as_deref())?
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
    let n_edges = checked_nrows(&edge_stored.table)?;
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
/// Looks for patterns like "a.id = 42" or "a.name = 'Alice'" and resolves to the
/// row index in the vertex table. The row index is used as a CSR node index.
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

    // Split on '=' first, then strip variable prefix only from the LHS
    let parts: Vec<&str> = filter.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(SqlError::Plan(format!(
            "Cannot extract node ID from filter: {filter}"
        )));
    }
    let lhs = parts[0].trim();
    // Reject !=, >=, <= operators (detected by trailing !, >, < on LHS after split)
    if lhs.ends_with('!') || lhs.ends_with('>') || lhs.ends_with('<') {
        return Err(SqlError::Plan(format!(
            "Unsupported operator in node filter: {filter}. Only 'col = value' is supported."
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
    // Remove internal spaces from value to handle tokenizer artifacts (e.g., "- 1" -> "-1")
    let value_nospace: String = value.chars().filter(|c| !c.is_whitespace()).collect();

    // Scan the table to find the matching row and return its row index.
    let stored = session.tables.get(table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{table_name}' not found"))
    })?;

    let nrows = checked_nrows(&stored.table)?;
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
            return Ok(row as i64);
        }
    }

    Err(SqlError::Plan(format!(
        "No matching row for filter: {filter}"
    )))
}

// ---------------------------------------------------------------------------
// Graph algorithm planner
// ---------------------------------------------------------------------------

/// Known graph algorithm function names.
const ALGO_FUNCTIONS: &[&str] = &["pagerank", "component", "connected_component", "community", "louvain"];

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
    let nrows = checked_nrows(&vertex_stored.table)?;

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
            // Validate arguments: expected form is FUNC(graph_name, node_var)
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
            // Reject mixing different algorithms in the same COLUMNS clause
            if let Some(ref prev) = algo_func_used {
                let prev_canonical = match prev.as_str() {
                    "component" | "connected_component" => "connected_component",
                    "community" | "louvain" => "louvain",
                    other => other,
                };
                let cur_canonical = match func_name.as_str() {
                    "component" | "connected_component" => "connected_component",
                    "community" | "louvain" => "louvain",
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
                algo_result = Some(execute_graph_algorithm(
                    session, graph, &func_name, edge_label_name,
                )?);
            }

            // Map function name to result column
            let result_col_name = match func_name.as_str() {
                "pagerank" => "_rank",
                "component" | "connected_component" => "_component",
                "community" | "louvain" => "_community",
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
        _ => return Err(SqlError::Plan(format!(
            "Unknown graph algorithm: {func_name}"
        ))),
    };

    let result = g.execute(result_col)?;
    Ok(result)
}
