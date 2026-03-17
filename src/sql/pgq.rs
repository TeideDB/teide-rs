// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{ffi, HnswIndex, Rel, Table};
use sqlparser::ast as sql_ast;
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser as SqlParser;

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

/// A user-facing key value (integer or string) for natural key mapping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum KeyValue {
    Int(i64),
    Str(String),
}

impl KeyValue {
    /// Format for CSV output: integers as plain numbers, strings as CSV-quoted.
    fn to_csv(&self) -> String {
        match self {
            KeyValue::Int(v) => v.to_string(),
            KeyValue::Str(s) => csv_quote(s),
        }
    }
}

/// Which columns are visible in GRAPH_TABLE COLUMNS for this label.
#[derive(Debug, Clone)]
pub(crate) enum ColumnVisibility {
    /// Default: all columns visible.
    All,
    /// PROPERTIES (col1, col2): only these columns are visible.
    Only(HashSet<String>),
    /// PROPERTIES ARE ALL COLUMNS EXCEPT (col): all except these.
    AllExcept(HashSet<String>),
    /// NO PROPERTIES: no columns visible.
    None,
}

impl ColumnVisibility {
    /// Check whether a column name is visible under this visibility rule.
    pub(crate) fn is_visible(&self, col_name: &str) -> bool {
        match self {
            ColumnVisibility::All => true,
            ColumnVisibility::Only(cols) => cols.contains(col_name),
            ColumnVisibility::AllExcept(cols) => !cols.contains(col_name),
            ColumnVisibility::None => false,
        }
    }
}

/// A vertex label mapping: label name -> session table name, with key maps
/// for translating between user-facing key values and internal row indices.
pub(crate) struct VertexLabel {
    pub table_name: String,
    #[allow(dead_code)]
    pub label: String,
    pub key_column: String,
    pub user_to_row: HashMap<KeyValue, usize>,
    pub row_to_user: Vec<KeyValue>,
    /// Column visibility restriction from PROPERTIES clause.
    pub visibility: ColumnVisibility,
}

/// An edge label mapping: label name -> edge table with source/dest references.
#[derive(Clone)]
pub(crate) struct EdgeLabel {
    pub table_name: String,
    #[allow(dead_code)]
    pub label: String,
    pub src_col: String,
    pub src_ref_table: String,
    #[allow(dead_code)]
    pub src_ref_col: String,
    pub dst_col: String,
    pub dst_ref_table: String,
    #[allow(dead_code)]
    pub dst_ref_col: String,
    /// Column visibility restriction from PROPERTIES clause.
    pub visibility: ColumnVisibility,
}

/// Stored relationship: the built CSR index + its edge label metadata.
pub(crate) struct StoredRel {
    pub rel: Rel,
    pub edge_label: EdgeLabel,
    /// Maps (remapped_src_row, remapped_dst_row) to original edge table row indices.
    /// Used to look up edge properties given a (src, dst) pair from CSR traversal.
    pub edge_row_map: HashMap<(i64, i64), Vec<usize>>,
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
    Forward,       // ->
    Reverse,       // <-
    Undirected,    // - (either direction)
    Bidirectional, // <-..-> (edge exists in both directions)
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
/// Labels support OR-union via pipe syntax: (n:Person|Company)
#[derive(Debug, Clone)]
pub(crate) struct NodePattern {
    #[allow(dead_code)]
    pub variable: Option<String>,
    /// One or more vertex labels (OR-union). `None` means unspecified (use default).
    /// Single-element vec is the common case; multiple elements mean label expression.
    pub labels: Option<Vec<String>>,
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
    /// Optional COST expression for weighted shortest path (Dijkstra).
    /// e.g. `COST e.weight` stores `"e.weight"`.
    pub cost_expr: Option<String>,
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
        let stored = session.tables.get(&vt.table_name).ok_or_else(|| {
            SqlError::Plan(format!(
                "Vertex table '{}' not found in session",
                vt.table_name
            ))
        })?;
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

        // Build key maps: user-facing key values <-> row indices
        let key_col_name = vt.key_column.as_deref().unwrap_or("id");
        let key_col_idx = find_col_idx(&stored.table, key_col_name).ok_or_else(|| {
            SqlError::Plan(format!(
                "Key column '{}' not found in vertex table '{}'",
                key_col_name, vt.table_name
            ))
        })?;
        let nrows = checked_logical_nrows(stored)?;
        let mut user_to_row = HashMap::with_capacity(nrows);
        let mut row_to_user = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let key_val = read_key_value(&stored.table, key_col_idx, row, key_col_name, &vt.table_name)?;
            if user_to_row.contains_key(&key_val) {
                return Err(SqlError::Plan(format!(
                    "Duplicate key {:?} in vertex table '{}'", key_val, vt.table_name
                )));
            }
            user_to_row.insert(key_val.clone(), row);
            row_to_user.push(key_val);
        }

        vertex_labels.insert(
            label.clone(),
            VertexLabel {
                table_name: vt.table_name.clone(),
                label: label.clone(),
                key_column: key_col_name.to_string(),
                user_to_row,
                row_to_user,
                visibility: vt.visibility.clone(),
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

        let n_src = checked_logical_nrows(src_stored).map_err(|_| SqlError::Plan(format!(
            "Source vertex table '{}' has invalid row count", et.src_ref_table
        )))? as i64;
        let n_dst = checked_logical_nrows(dst_stored).map_err(|_| SqlError::Plan(format!(
            "Destination vertex table '{}' has invalid row count", et.dst_ref_table
        )))? as i64;

        // Look up vertex labels for source and destination to get key maps
        // Prefer matching by both table_name and key_column for disambiguation
        let src_vl = vertex_labels.values()
            .find(|vl| vl.table_name == et.src_ref_table && vl.key_column == et.src_ref_col)
            .ok_or_else(|| SqlError::Plan(format!(
                "No vertex label found for source table '{}' with key column '{}'",
                et.src_ref_table, et.src_ref_col
            )))?;
        let dst_vl = vertex_labels.values()
            .find(|vl| vl.table_name == et.dst_ref_table && vl.key_column == et.dst_ref_col)
            .ok_or_else(|| SqlError::Plan(format!(
                "No vertex label found for destination table '{}' with key column '{}'",
                et.dst_ref_table, et.dst_ref_col
            )))?;

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
            visibility: et.visibility.clone(),
        };

        // Remap edge FK values through vertex key maps and build CSR
        let (rel, edge_row_map) = remap_and_build_rel(
            session, edge_stored, src_vl, dst_vl, &edge_label, n_src, n_dst,
        )?;

        if edge_labels.contains_key(&label) {
            return Err(SqlError::Plan(format!(
                "Duplicate edge label '{label}' in property graph"
            )));
        }
        edge_labels.insert(label, StoredRel { rel, edge_label, edge_row_map });
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
            let or_replace = parsed.or_replace;
            let if_not_exists = parsed.if_not_exists;
            if session.graphs.contains_key(&name) {
                if if_not_exists {
                    return Ok(ExecResult::Ddl(format!(
                        "Property graph '{name}' already exists (skipped)"
                    )));
                } else if or_replace {
                    session.graphs.remove(&name);
                } else {
                    return Err(SqlError::Plan(format!(
                        "Property graph '{name}' already exists"
                    )));
                }
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
        PgqStatement::DescribePropertyGraph(name) => {
            let graph = session.graphs.get(&name).ok_or_else(|| {
                SqlError::Plan(format!("Property graph '{name}' not found"))
            })?;
            describe_property_graph(session, graph)
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

/// Build a result set describing a property graph's vertex and edge labels.
fn describe_property_graph(
    session: &Session,
    graph: &PropertyGraph,
) -> Result<ExecResult, SqlError> {
    let col_names = vec![
        "element_type".to_string(),
        "label".to_string(),
        "table_name".to_string(),
        "key_column".to_string(),
        "src_table".to_string(),
        "dst_table".to_string(),
    ];
    let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    // Collect and sort vertex labels by label name for deterministic output.
    let mut vertex_entries: Vec<(&String, &VertexLabel)> =
        graph.vertex_labels.iter().collect();
    vertex_entries.sort_by_key(|(name, _)| (*name).clone());
    for (label_name, vl) in &vertex_entries {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            csv_quote("VERTEX"),
            csv_quote(label_name),
            csv_quote(&vl.table_name),
            csv_quote(&vl.key_column),
            csv_quote("-"),
            csv_quote("-"),
        ));
    }

    // Collect and sort edge labels by label name for deterministic output.
    let mut edge_entries: Vec<(&String, &StoredRel)> =
        graph.edge_labels.iter().collect();
    edge_entries.sort_by_key(|(name, _)| (*name).clone());
    for (label_name, stored_rel) in &edge_entries {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            csv_quote("EDGE"),
            csv_quote(label_name),
            csv_quote(&stored_rel.edge_label.table_name),
            csv_quote(&stored_rel.edge_label.src_col),
            csv_quote(&stored_rel.edge_label.src_ref_table),
            csv_quote(&stored_rel.edge_label.dst_ref_table),
        ));
    }

    let table = csv_to_table(session, &csv, &col_names)?;
    let nrows = checked_nrows(&table)?;
    Ok(ExecResult::Query(super::SqlResult {
        table,
        columns: col_names,
        embedding_dims: HashMap::new(),
        nrows,
    }))
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

/// Merge multiple comma-separated MATCH patterns into a single chain by
/// joining on shared variables at junction points.
///
/// For example, `(a)-[:E1]->(b), (b)-[:E2]->(c)` becomes
/// `(a)-[:E1]->(b)-[:E2]->(c)` because variable `b` is shared between
/// the last node of the first pattern and the first node of the second.
///
/// If the shared node has labels or filters in both patterns they are merged:
/// labels must agree (or one is unset), and filters are ANDed.
fn merge_patterns(patterns: &[PathPattern]) -> Result<PathPattern, SqlError> {
    debug_assert!(!patterns.is_empty());
    let mut merged = patterns[0].clone();

    for pattern in &patterns[1..] {
        if pattern.nodes.is_empty() {
            return Err(SqlError::Plan("Empty MATCH pattern".into()));
        }

        let last_node = merged.nodes.last().ok_or_else(|| {
            SqlError::Plan("Empty MATCH pattern".into())
        })?;
        let first_node = &pattern.nodes[0];

        // Check that the two patterns share a named variable at the junction.
        let shared = matches!(
            (&last_node.variable, &first_node.variable),
            (Some(a), Some(b)) if a == b
        );
        if !shared {
            return Err(SqlError::Plan(
                "Multiple MATCH patterns must share a variable at the junction point. \
                 E.g., (a)-[:E1]->(b), (b)-[:E2]->(c) where 'b' is shared."
                    .into(),
            ));
        }

        // Merge labels: if both specify labels, they must agree.
        let merged_labels = match (&last_node.labels, &first_node.labels) {
            (Some(a), Some(b)) if a != b => {
                return Err(SqlError::Plan(format!(
                    "Conflicting labels on shared variable '{}': '{:?}' vs '{:?}'",
                    last_node.variable.as_deref().unwrap_or("?"),
                    a,
                    b
                )));
            }
            (Some(a), _) => Some(a.clone()),
            (_, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        // Merge filters: AND them together if both present.
        let merged_filter = match (&last_node.filter, &first_node.filter) {
            (Some(a), Some(b)) => Some(format!("({a}) AND ({b})")),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        // Update the junction node in the merged pattern with the combined
        // labels and filter.
        let junction = merged.nodes.last_mut().unwrap();
        junction.labels = merged_labels;
        junction.filter = merged_filter;

        // Append the new pattern's edges and remaining nodes (skip the
        // first node since it is the shared junction already present).
        merged.edges.extend(pattern.edges.iter().cloned());
        merged.nodes.extend(pattern.nodes[1..].iter().cloned());
    }

    Ok(merged)
}

/// Check whether any node in a pattern has multiple labels (label expression).
fn has_multi_label_nodes(pattern: &PathPattern) -> bool {
    pattern.nodes.iter().any(|n| {
        n.labels.as_ref().map_or(false, |v| v.len() > 1)
    })
}

/// Expand a pattern with multi-label nodes into all single-label combinations.
/// For example, if node 0 has labels [A, B] and node 1 has labels [C, D],
/// this produces 4 patterns: (A,C), (A,D), (B,C), (B,D).
fn expand_multi_label_patterns(pattern: &PathPattern) -> Vec<PathPattern> {
    // Collect per-node label lists. Nodes without labels or with a single
    // label produce a one-element list of None or Some(single).
    let per_node: Vec<Vec<Option<Vec<String>>>> = pattern.nodes.iter().map(|n| {
        match &n.labels {
            Some(labels) if labels.len() > 1 => {
                labels.iter().map(|l| Some(vec![l.clone()])).collect()
            }
            other => vec![other.clone()],
        }
    }).collect();

    // Cartesian product of all per-node label alternatives.
    let mut combos: Vec<Vec<Option<Vec<String>>>> = vec![vec![]];
    for node_alts in &per_node {
        let mut new_combos = Vec::new();
        for existing in &combos {
            for alt in node_alts {
                let mut combo = existing.clone();
                combo.push(alt.clone());
                new_combos.push(combo);
            }
        }
        combos = new_combos;
    }

    // Build one pattern per combination.
    combos.into_iter().map(|combo| {
        let mut p = pattern.clone();
        for (i, label_opt) in combo.into_iter().enumerate() {
            p.nodes[i].labels = label_opt;
        }
        p
    }).collect()
}

/// Union multiple result tables into a single table via CSV reconstruction.
/// All component tables must have the same column structure.
fn union_result_tables(
    session: &Session,
    results: Vec<(Table, Vec<String>)>,
) -> Result<(Table, Vec<String>), SqlError> {
    if results.is_empty() {
        return Err(SqlError::Plan("No results to union".into()));
    }
    if results.len() == 1 {
        return Ok(results.into_iter().next().unwrap());
    }

    let col_names = results[0].1.clone();
    let ncols = col_names.len();

    // Build CSV header
    let mut csv = col_names
        .iter()
        .map(|n| csv_quote(n))
        .collect::<Vec<_>>()
        .join(",");
    csv.push('\n');

    // Append rows from each result table
    for (table, _) in &results {
        let nrows = checked_nrows(table)?;
        for row in 0..nrows {
            for col in 0..ncols {
                if col > 0 {
                    csv.push(',');
                }
                csv.push_str(&get_cell_string(table, col, row)?);
            }
            csv.push('\n');
        }
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

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

    // Merge multiple comma-separated patterns into a single chain if possible.
    let merged_pattern;
    let pattern = if match_clause.patterns.len() > 1 {
        merged_pattern = merge_patterns(&match_clause.patterns)?;
        &merged_pattern
    } else {
        &match_clause.patterns[0]
    };

    // If any node has a multi-label expression (e.g. Person|Company), expand
    // into separate single-label patterns and union the results.
    if has_multi_label_nodes(pattern) {
        let expanded = expand_multi_label_patterns(pattern);
        let mut results = Vec::with_capacity(expanded.len());
        for variant in &expanded {
            // Build a temporary GraphTableExpr with the single-label variant pattern.
            let variant_expr = GraphTableExpr {
                graph_name: expr.graph_name.clone(),
                match_clause: MatchClause {
                    path_variable: match_clause.path_variable.clone(),
                    mode: match_clause.mode,
                    patterns: vec![variant.clone()],
                },
                columns: expr.columns.clone(),
            };
            match plan_graph_table(session, &variant_expr) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // If a specific label combination fails (e.g. edge doesn't
                    // connect to that label), skip it rather than failing the
                    // entire query.  This is expected for heterogeneous graphs
                    // where not all labels participate in all edge types.
                    if e.to_string().contains("not found")
                        || e.to_string().contains("expects")
                    {
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        if results.is_empty() {
            return Err(SqlError::Plan(
                "No matching label combination found for multi-label pattern".into(),
            ));
        }
        return union_result_tables(session, results);
    }

    // Check for variable-length edge first.
    let is_var_length = pattern.edges.len() == 1
        && !matches!(pattern.edges[0].quantifier, PathQuantifier::One);

    // Detect cyclic variable binding (same variable at first and last node).
    // Cyclic patterns need the Rust BFS planner because the C engine's var_expand
    // uses visited-set deduplication which prevents revisiting the start node.
    let is_cyclic_binding = pattern.nodes.len() >= 2
        && matches!(
            (&pattern.nodes[0].variable, &pattern.nodes.last().unwrap().variable),
            (Some(a), Some(b)) if a == b
        );

    // Check if any edge has a COST expression → route to Dijkstra-based cheapest path planner
    let has_cost = pattern.edges.iter().any(|e| e.cost_expr.is_some());
    if has_cost {
        if pattern.nodes.len() != 2 || pattern.edges.len() != 1 {
            return Err(SqlError::Plan(
                "COST expression is only supported on single-edge patterns: (a)-[e:Label COST expr]->+(b)"
                    .into(),
            ));
        }
        return plan_cheapest_path(session, graph, pattern, &expr.columns);
    }

    match (pattern.nodes.len(), pattern.edges.len(), match_clause.mode) {
        // ANY SHORTEST always uses the shortest-path planner (supports dst filters,
        // _node/_depth columns).
        (2, 1, PathMode::AnyShortest) if !is_cyclic_binding => {
            plan_shortest_path(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) if is_var_length && !is_cyclic_binding => {
            plan_var_length(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) if !is_var_length && !is_cyclic_binding => {
            plan_single_hop(session, graph, pattern, &expr.columns)
        }
        (1, 0, _) => {
            plan_algorithm_query(session, graph, pattern, &expr.columns)
        }
        _ if pattern.nodes.len() >= 2 && pattern.edges.len() >= 1 => {
            let all_fixed = pattern.edges.iter().all(|e| matches!(e.quantifier, PathQuantifier::One));
            if all_fixed && !is_cyclic_binding {
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
    let (left_default_table, right_default_table) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table)
    };

    // Resolve source and destination vertex tables
    let src_label = resolve_node_label(src_node, left_default_table, graph)?;
    let dst_label = resolve_node_label(dst_node, right_default_table, graph)?;

    // Validate that explicit node labels match the edge's expected tables
    // and key columns.
    let (left_ref_col, right_ref_col) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_col, &stored_rel.edge_label.src_ref_col)
    } else {
        (&stored_rel.edge_label.src_ref_col, &stored_rel.edge_label.dst_ref_col)
    };
    validate_node_table_for_edge(src_node, &src_label.table_name, left_default_table, edge_label, "source", Some(&src_label.key_column), Some(left_ref_col))?;
    validate_node_table_for_edge(dst_node, &dst_label.table_name, right_default_table, edge_label, "destination", Some(&dst_label.key_column), Some(right_ref_col))?;

    let src_stored = session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?;
    let dst_stored = session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?;

    // Undirected/bidirectional traversals on heterogeneous edges (different src/dst
    // tables) would mix node IDs from different tables in the _dst column.
    if (edge.direction == MatchDirection::Undirected
        || edge.direction == MatchDirection::Bidirectional)
        && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
    {
        return Err(SqlError::Plan(
            "Undirected/bidirectional traversals are not supported on edges with different source and \
             destination vertex tables (heterogeneous graphs). Use a directed pattern instead."
                .into(),
        ));
    }

    let direction: u8 = match edge.direction {
        MatchDirection::Forward | MatchDirection::Bidirectional => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // Determine source node row indices
    let src_row_indices: Vec<i64> = if let Some(filter_text) = &src_node.filter {
        resolve_filtered_node_ids(filter_text, src_node, src_stored)?
    } else {
        let nrows = checked_logical_nrows(src_stored)?;
        (0..nrows as i64).collect()
    };

    // Build a 1-column table of source row indices for expand
    let rowid_table = build_rowid_table(session, &src_row_indices)?;
    let mut g = session.ctx.graph(&rowid_table)?;
    let src_ids = g.scan("_rowid")?;

    // Expand via CSR
    let expanded = g.expand(src_ids, &stored_rel.rel, direction)?;
    let expand_result = g.execute(expanded)?;

    // For bidirectional edges, filter to keep only (src, dst) pairs where the
    // reverse edge (dst -> src) also exists.
    let expand_result = if edge.direction == MatchDirection::Bidirectional {
        filter_bidirectional(session, &expand_result, &stored_rel.rel)?
    } else {
        expand_result
    };

    // Post-filter destination node
    let expand_result = if let Some(filter_text) = &dst_node.filter {
        post_filter_expand_result(session, &expand_result, filter_text, dst_node, &dst_stored.table, false)?
    } else {
        expand_result
    };

    // Project requested columns
    project_columns(
        session,
        &expand_result,
        columns,
        src_node,
        dst_node,
        &src_stored.table,
        &dst_stored.table,
        edge,
        graph,
        is_reverse,
        src_label,
        dst_label,
    )
}

/// Filter a forward expand result for bidirectional edges: keep only (src, dst)
/// pairs where the reverse edge (dst -> src) also exists.
fn filter_bidirectional(
    session: &Session,
    expand_result: &Table,
    rel: &Rel,
) -> Result<Table, SqlError> {
    use std::collections::HashSet;

    let nrows = checked_nrows(expand_result)?;
    let src_col_idx = find_col_idx(expand_result, "_src")
        .ok_or_else(|| SqlError::Plan("expand result missing _src column".into()))?;
    let dst_col_idx = find_col_idx(expand_result, "_dst")
        .ok_or_else(|| SqlError::Plan("expand result missing _dst column".into()))?;

    // Collect unique dst values and expand forward from them to find all (dst -> ?) edges
    let mut dst_values: Vec<i64> = Vec::new();
    let mut dst_set: HashSet<i64> = HashSet::new();
    for row in 0..nrows {
        if let Some(dst) = expand_result.get_i64(dst_col_idx, row) {
            if dst >= 0 && dst_set.insert(dst) {
                dst_values.push(dst);
            }
        }
    }

    // Expand forward from dst nodes to find reverse edges (dst -> src edges)
    let dst_rowid_table = build_rowid_table(session, &dst_values)?;
    let mut g = session.ctx.graph(&dst_rowid_table)?;
    let dst_ids = g.scan("_rowid")?;
    let reverse_expanded = g.expand(dst_ids, rel, 0)?; // forward expand from dst nodes
    let reverse_result = g.execute(reverse_expanded)?;

    // Build a set of (original_dst, target) from the reverse expansion
    // If (dst, src) exists in this set, then the edge dst->src exists
    let reverse_nrows = checked_nrows(&reverse_result)?;
    let rev_src_col = find_col_idx(&reverse_result, "_src")
        .ok_or_else(|| SqlError::Plan("reverse expand missing _src column".into()))?;
    let rev_dst_col = find_col_idx(&reverse_result, "_dst")
        .ok_or_else(|| SqlError::Plan("reverse expand missing _dst column".into()))?;

    let mut reverse_edges: HashSet<(i64, i64)> = HashSet::new();
    for row in 0..reverse_nrows {
        if let (Some(s), Some(d)) = (
            reverse_result.get_i64(rev_src_col, row),
            reverse_result.get_i64(rev_dst_col, row),
        ) {
            reverse_edges.insert((s, d));
        }
    }

    // Filter original pairs: keep (src, dst) only if (dst, src) exists in reverse_edges
    let mut csv = String::from("_src,_dst\n");
    for row in 0..nrows {
        if let (Some(src), Some(dst)) = (
            expand_result.get_i64(src_col_idx, row),
            expand_result.get_i64(dst_col_idx, row),
        ) {
            if reverse_edges.contains(&(dst, src)) {
                csv.push_str(&format!("{src},{dst}\n"));
            }
        }
    }

    csv_to_table(session, &csv, &["_src".to_string(), "_dst".to_string()])
}

/// Post-filter an expand result by evaluating a WHERE filter on the destination
/// (or source, via `filter_on_src`) node for each row. Returns a new table with
/// only matching rows.
fn post_filter_expand_result(
    session: &Session,
    expand_result: &Table,
    filter_text: &str,
    filter_node: &NodePattern,
    filter_table: &Table,
    filter_on_src: bool,
) -> Result<Table, SqlError> {
    let expr = parse_filter_expr(filter_text)?;
    let var_name = filter_node.variable.as_deref().unwrap_or("");
    let nrows = checked_nrows(expand_result)?;

    let src_col = find_col_idx(expand_result, "_src")
        .ok_or_else(|| SqlError::Plan("expand result missing _src column".into()))?;
    let dst_col = find_col_idx(expand_result, "_dst")
        .ok_or_else(|| SqlError::Plan("expand result missing _dst column".into()))?;

    let filter_col = if filter_on_src { src_col } else { dst_col };

    let mut csv = String::from("_src,_dst\n");
    for row in 0..nrows {
        let filter_row_id = expand_result.get_i64(filter_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL index at row {row}")))?;
        if filter_row_id < 0 {
            continue;
        }
        if evaluate_filter(&expr, filter_table, filter_row_id as usize, var_name)?.unwrap_or(false) {
            let src_val = expand_result.get_i64(src_col, row)
                .ok_or_else(|| SqlError::Plan(format!("NULL _src at row {row}")))?;
            let dst_val = expand_result.get_i64(dst_col, row)
                .ok_or_else(|| SqlError::Plan(format!("NULL _dst at row {row}")))?;
            csv.push_str(&format!("{src_val},{dst_val}\n"));
        }
    }

    csv_to_table(session, &csv, &["_src".to_string(), "_dst".to_string()])
}

/// Build a 1-column table of I64 row indices for use as expand input.
fn build_rowid_table(session: &Session, ids: &[i64]) -> Result<Table, SqlError> {
    let mut csv = String::from("_rowid\n");
    for id in ids {
        csv.push_str(&id.to_string());
        csv.push('\n');
    }
    csv_to_table(session, &csv, &["_rowid".to_string()])
}

/// Resolve which vertex label a node pattern refers to.
/// For multi-label nodes this returns the first label; multi-label union is
/// handled at a higher level by expanding into separate single-label patterns.
fn resolve_node_label<'a>(
    node: &NodePattern,
    default_table: &str,
    graph: &'a PropertyGraph,
) -> Result<&'a VertexLabel, SqlError> {
    if let Some(labels) = &node.labels {
        let label = labels.first().ok_or_else(|| {
            SqlError::Plan("Empty label list in node pattern".into())
        })?;
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
    resolved_key_col: Option<&str>,
    expected_ref_col: Option<&str>,
) -> Result<(), SqlError> {
    if let Some(labels) = &node.labels {
        let label = labels.first().map(|s| s.as_str()).unwrap_or("?");
        if resolved_table != expected_table {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' resolves to table '{resolved_table}', but edge \
                 '{edge_label}' expects {position} table '{expected_table}'"
            )));
        }
        if let (Some(resolved_kc), Some(expected_rc)) = (resolved_key_col, expected_ref_col) {
            if resolved_kc != expected_rc {
                return Err(SqlError::Plan(format!(
                    "Node label '{label}' uses key column '{resolved_kc}', but edge \
                     '{edge_label}' references {position} key column '{expected_rc}'"
                )));
            }
        }
    }
    Ok(())
}

/// Project COLUMNS from expand results.
/// Maps column expressions like "b.name" to lookups in source/destination tables.
/// Also supports edge variable references like "e.weight" via edge_row_map.
fn project_columns(
    session: &Session,
    expand_result: &Table,
    columns: &[ColumnEntry],
    src_node: &NodePattern,
    dst_node: &NodePattern,
    src_table: &Table,
    dst_table: &Table,
    edge_pattern: &EdgePattern,
    graph: &PropertyGraph,
    is_reverse: bool,
    src_vertex_label: &VertexLabel,
    dst_vertex_label: &VertexLabel,
) -> Result<(Table, Vec<String>), SqlError> {
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
    let edge_var = edge_pattern.variable.as_deref();

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
    enum ColSpec {
        Node { table_col_idx: usize, is_src: bool },
        Edge { table_col_idx: usize, edge_label: String },
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
                // Check PROPERTIES visibility for source vertex label
                check_column_visible(&src_vertex_label.visibility, &col, &src_vertex_label.label)?;
                let col_idx = find_col_idx(src_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in source table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec::Node { table_col_idx: col_idx, is_src: true });
            } else if var == dst_var {
                // Check PROPERTIES visibility for destination vertex label
                check_column_visible(&dst_vertex_label.visibility, &col, &dst_vertex_label.label)?;
                let col_idx = find_col_idx(dst_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in destination table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec::Node { table_col_idx: col_idx, is_src: false });
            } else if edge_var.is_some() && var == edge_var.unwrap() {
                // Edge variable: look up property from edge table
                let edge_label_name = edge_pattern.label.as_deref().ok_or_else(|| {
                    SqlError::Plan("Edge variable requires a label to access properties".into())
                })?;
                let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge label '{edge_label_name}' not found in graph"))
                })?;
                // Check PROPERTIES visibility for edge label
                check_column_visible(&stored_rel.edge_label.visibility, &col, edge_label_name)?;
                let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                })?;
                let col_idx = find_col_idx(&edge_table.table, &col).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{col}' not found in edge table '{}'",
                        stored_rel.edge_label.table_name
                    ))
                })?;
                col_names.push(out_name);
                col_specs.push(ColSpec::Edge { table_col_idx: col_idx, edge_label: edge_label_name.to_string() });
            } else {
                let mut available = format!("{src_var}, {dst_var}");
                if let Some(ev) = edge_var {
                    available.push_str(&format!(", {ev}"));
                }
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available}"
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
            match spec {
                ColSpec::Node { table_col_idx, is_src } => {
                    let table_row = if *is_src { src_indices[row] } else { dst_indices[row] };
                    let table = if *is_src { src_table } else { dst_table };
                    csv.push_str(&get_cell_string(table, *table_col_idx, table_row)?);
                }
                ColSpec::Edge { table_col_idx, edge_label } => {
                    let stored_rel = graph.edge_labels.get(edge_label.as_str()).ok_or_else(|| {
                        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
                    })?;
                    // For forward edges: CSR (src, dst) = (_src, _dst)
                    // For reverse edges: pattern _src is edge dst, _dst is edge src
                    let (edge_src, edge_dst) = if is_reverse {
                        (dst_indices[row] as i64, src_indices[row] as i64)
                    } else {
                        (src_indices[row] as i64, dst_indices[row] as i64)
                    };
                    let edge_rows = stored_rel.edge_row_map.get(&(edge_src, edge_dst))
                        .ok_or_else(|| SqlError::Plan(format!(
                            "No edge found for ({edge_src}, {edge_dst}) in edge label '{edge_label}'"
                        )))?;
                    // Use the first matching edge row (for multi-edges, this picks one)
                    let edge_row = edge_rows[0];
                    let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                        SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                    })?;
                    csv.push_str(&get_cell_string(&edge_table.table, *table_col_idx, edge_row)?);
                }
            }
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

/// Check that a column is visible under the given PROPERTIES visibility rule.
/// Returns `Ok(())` if visible, or a descriptive error if restricted.
fn check_column_visible(
    visibility: &ColumnVisibility,
    col_name: &str,
    label_name: &str,
) -> Result<(), SqlError> {
    if visibility.is_visible(col_name) {
        Ok(())
    } else {
        Err(SqlError::Plan(format!(
            "Column '{col_name}' is not accessible on label '{label_name}' \
             (restricted by PROPERTIES clause)"
        )))
    }
}

/// Read a key value (integer or string) from a table cell.
fn read_key_value(
    table: &Table,
    col_idx: usize,
    row: usize,
    col_name: &str,
    table_name: &str,
) -> Result<KeyValue, SqlError> {
    let typ = table.col_type(col_idx);
    match typ {
        ffi::TD_U8 | ffi::TD_CHAR | ffi::TD_I16 | ffi::TD_I32 | ffi::TD_I64 => {
            let v = table.get_i64(col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!(
                    "NULL value at row {row} in column '{col_name}' of '{table_name}'"
                ))
            })?;
            Ok(KeyValue::Int(v))
        }
        ffi::TD_SYM => {
            let s = table.get_str(col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!(
                    "NULL value at row {row} in column '{col_name}' of '{table_name}'"
                ))
            })?;
            Ok(KeyValue::Str(s))
        }
        _ => Err(SqlError::Plan(format!(
            "Unsupported key column type {typ} in '{table_name}'"
        ))),
    }
}

/// Rebuild vertex key maps for a vertex label from the current table data.
/// Called during graph invalidation after DML to keep key maps consistent.
pub(super) fn rebuild_vertex_key_map(
    vl: &mut VertexLabel,
    stored: &super::StoredTable,
) -> Result<(), SqlError> {
    let col_idx = find_col_idx(&stored.table, &vl.key_column).ok_or_else(|| {
        SqlError::Plan(format!(
            "Key column '{}' not found in vertex table '{}'",
            vl.key_column, vl.table_name
        ))
    })?;
    let nrows = checked_logical_nrows(stored)?;
    let mut user_to_row = HashMap::with_capacity(nrows);
    let mut row_to_user = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let key_val = read_key_value(&stored.table, col_idx, row, &vl.key_column, &vl.table_name)?;
        if user_to_row.contains_key(&key_val) {
            return Err(SqlError::Plan(format!(
                "Duplicate key {:?} in vertex table '{}'",
                key_val, vl.table_name
            )));
        }
        user_to_row.insert(key_val.clone(), row);
        row_to_user.push(key_val);
    }
    vl.user_to_row = user_to_row;
    vl.row_to_user = row_to_user;
    Ok(())
}

/// Remap edge FK values through vertex key maps and build a CSR `Rel`.
/// Used during initial graph construction and graph rebuild after DML.
/// Returns the Rel and an edge_row_map: (remapped_src, remapped_dst) -> Vec<original_edge_row>.
pub(super) fn remap_and_build_rel(
    session: &Session,
    edge_stored: &super::StoredTable,
    src_vl: &VertexLabel,
    dst_vl: &VertexLabel,
    el: &EdgeLabel,
    n_src: i64,
    n_dst: i64,
) -> Result<(Rel, HashMap<(i64, i64), Vec<usize>>), SqlError> {
    let n_edges = checked_logical_nrows(edge_stored)?;
    let src_col_idx = find_col_idx(&edge_stored.table, &el.src_col).ok_or_else(|| {
        SqlError::Plan(format!(
            "Column '{}' not found in edge table '{}'",
            el.src_col, el.table_name
        ))
    })?;
    let dst_col_idx = find_col_idx(&edge_stored.table, &el.dst_col).ok_or_else(|| {
        SqlError::Plan(format!(
            "Column '{}' not found in edge table '{}'",
            el.dst_col, el.table_name
        ))
    })?;

    let mut csv = String::from("_src,_dst\n");
    let mut edge_row_map: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for row in 0..n_edges {
        let src_key = read_key_value(&edge_stored.table, src_col_idx, row, &el.src_col, &el.table_name)?;
        let src_row = src_vl.user_to_row.get(&src_key).ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge table '{}' column '{}' row {}: key {:?} not found in vertex table '{}'",
                el.table_name, el.src_col, row, src_key, el.src_ref_table
            ))
        })?;
        let dst_key = read_key_value(&edge_stored.table, dst_col_idx, row, &el.dst_col, &el.table_name)?;
        let dst_row = dst_vl.user_to_row.get(&dst_key).ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge table '{}' column '{}' row {}: key {:?} not found in vertex table '{}'",
                el.table_name, el.dst_col, row, dst_key, el.dst_ref_table
            ))
        })?;
        csv.push_str(&src_row.to_string());
        csv.push(',');
        csv.push_str(&dst_row.to_string());
        csv.push('\n');
        edge_row_map.entry((*src_row as i64, *dst_row as i64)).or_default().push(row);
    }
    let col_names = vec!["_src".to_string(), "_dst".to_string()];
    let remapped_edge_table = csv_to_table(session, &csv, &col_names)?;
    let rel = Rel::from_edges(&remapped_edge_table, "_src", "_dst", n_src, n_dst, true)?;
    Ok((rel, edge_row_map))
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
// Rich WHERE filter evaluation
// ---------------------------------------------------------------------------

/// A scalar value produced during filter expression evaluation.
#[derive(Debug, Clone)]
enum ScalarValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScalarValue::Int(a), ScalarValue::Int(b)) => a == b,
            (ScalarValue::Float(a), ScalarValue::Float(b)) => a == b,
            (ScalarValue::Int(a), ScalarValue::Float(b)) => (*a as f64) == *b,
            (ScalarValue::Float(a), ScalarValue::Int(b)) => *a == (*b as f64),
            (ScalarValue::Str(a), ScalarValue::Str(b)) => a == b,
            (ScalarValue::Bool(a), ScalarValue::Bool(b)) => a == b,
            (ScalarValue::Null, ScalarValue::Null) => false,
            _ => false,
        }
    }
}

impl PartialOrd for ScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ScalarValue::Int(a), ScalarValue::Int(b)) => a.partial_cmp(b),
            (ScalarValue::Float(a), ScalarValue::Float(b)) => a.partial_cmp(b),
            (ScalarValue::Int(a), ScalarValue::Float(b)) => (*a as f64).partial_cmp(b),
            (ScalarValue::Float(a), ScalarValue::Int(b)) => a.partial_cmp(&(*b as f64)),
            (ScalarValue::Str(a), ScalarValue::Str(b)) => a.partial_cmp(b),
            (ScalarValue::Bool(a), ScalarValue::Bool(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

/// Normalize filter text from PGQ tokenizer output.
/// The PGQ tokenizer splits multi-character operators (>=, <=, !=, <>) into
/// separate tokens joined by spaces. This reassembles them for sqlparser.
/// Also normalizes "a . col" → "a.col" compound identifiers.
fn normalize_filter_text(filter_text: &str) -> String {
    // Tokenize quote-aware: preserve single-quoted strings as atomic tokens.
    let mut tokens: Vec<String> = Vec::new();
    let mut chars = filter_text.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '\'' {
            // Collect entire single-quoted string as one token
            let mut tok = String::new();
            tok.push(chars.next().unwrap()); // opening quote
            while let Some(&c) = chars.peek() {
                tok.push(chars.next().unwrap());
                if c == '\'' {
                    // Check for escaped quote ('')
                    if chars.peek() == Some(&'\'') {
                        tok.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
            }
            tokens.push(tok);
        } else {
            // Non-quoted token: collect until whitespace or quote
            let mut tok = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '\'' {
                    break;
                }
                tok.push(chars.next().unwrap());
            }
            tokens.push(tok);
        }
    }

    let mut result = String::with_capacity(filter_text.len());
    let mut i = 0;
    while i < tokens.len() {
        if i > 0 {
            let prev = &tokens[i - 1];
            let curr = &tokens[i];
            if prev == "." || curr == "." {
                // Don't add space around dots: "a . name" -> "a.name"
            } else {
                result.push(' ');
            }
        }
        let tok = &tokens[i];
        // Try to merge multi-char operators
        if i + 1 < tokens.len() {
            let next = &tokens[i + 1];
            let merged = match (tok.as_str(), next.as_str()) {
                (">", "=") => Some(">="),
                ("<", "=") => Some("<="),
                ("!", "=") => Some("!="),
                ("<", ">") => Some("<>"),
                _ => None,
            };
            if let Some(op) = merged {
                result.push_str(op);
                i += 2;
                continue;
            }
        }
        result.push_str(tok);
        i += 1;
    }
    result
}

/// Parse a WHERE filter text into a sqlparser Expr AST.
fn parse_filter_expr(filter_text: &str) -> Result<sql_ast::Expr, SqlError> {
    let normalized = normalize_filter_text(filter_text);
    let dialect = DuckDbDialect {};
    let mut parser = SqlParser::new(&dialect)
        .try_with_sql(&normalized)
        .map_err(|e| SqlError::Parse(format!("Failed to parse WHERE filter: {e}")))?;
    parser
        .parse_expr()
        .map_err(|e| SqlError::Parse(format!("Invalid WHERE expression: {e}")))
}

/// Evaluate a scalar expression against a table row, returning a ScalarValue.
fn eval_scalar(
    expr: &sql_ast::Expr,
    table: &Table,
    row: usize,
    var_name: &str,
) -> Result<ScalarValue, SqlError> {
    match expr {
        sql_ast::Expr::CompoundIdentifier(parts) => {
            // e.g. a.name — parts[0] is variable, parts[1] is column
            if parts.len() == 2 {
                let var = parts[0].value.to_lowercase();
                let col = parts[1].value.to_lowercase();
                if var == var_name {
                    return read_scalar_from_table(table, &col, row);
                }
            }
            Err(SqlError::Plan(format!(
                "Unsupported identifier in filter: {}",
                expr
            )))
        }
        sql_ast::Expr::Identifier(ident) => {
            // Bare column name (no variable prefix)
            let col = ident.value.to_lowercase();
            read_scalar_from_table(table, &col, row)
        }
        sql_ast::Expr::Value(val) => match val {
            sql_ast::Value::Number(n, _) => {
                if let Ok(i) = n.parse::<i64>() {
                    Ok(ScalarValue::Int(i))
                } else if let Ok(f) = n.parse::<f64>() {
                    Ok(ScalarValue::Float(f))
                } else {
                    Err(SqlError::Plan(format!("Cannot parse number: {n}")))
                }
            }
            sql_ast::Value::SingleQuotedString(s) => Ok(ScalarValue::Str(s.clone())),
            sql_ast::Value::Boolean(b) => Ok(ScalarValue::Bool(*b)),
            sql_ast::Value::Null => Ok(ScalarValue::Null),
            _ => Err(SqlError::Plan(format!(
                "Unsupported value in filter: {val}"
            ))),
        },
        sql_ast::Expr::UnaryOp {
            op: sql_ast::UnaryOperator::Minus,
            expr: inner,
        } => {
            let v = eval_scalar(inner, table, row, var_name)?;
            match v {
                ScalarValue::Null => Ok(ScalarValue::Null),
                ScalarValue::Int(i) => Ok(ScalarValue::Int(i.checked_neg().ok_or_else(
                    || SqlError::Plan(format!("Integer overflow negating {i}")),
                )?)),
                ScalarValue::Float(f) => Ok(ScalarValue::Float(-f)),
                _ => Err(SqlError::Plan(format!(
                    "Cannot negate non-numeric value: {expr}"
                ))),
            }
        }
        _ => Err(SqlError::Plan(format!(
            "Unsupported scalar expression in filter: {expr}"
        ))),
    }
}

/// Read a scalar value from a table cell by column name.
fn read_scalar_from_table(
    table: &Table,
    col_name: &str,
    row: usize,
) -> Result<ScalarValue, SqlError> {
    let col_idx = find_col_idx(table, col_name).ok_or_else(|| {
        SqlError::Plan(format!("Column '{col_name}' not found in table"))
    })?;
    let typ = table.col_type(col_idx);
    match typ {
        ffi::TD_BOOL => match table.get_i64(col_idx, row) {
            Some(v) => Ok(ScalarValue::Bool(v != 0)),
            None => Ok(ScalarValue::Null),
        },
        ffi::TD_U8 | ffi::TD_CHAR | ffi::TD_I16 | ffi::TD_I32 | ffi::TD_I64 => {
            match table.get_i64(col_idx, row) {
                Some(v) => Ok(ScalarValue::Int(v)),
                None => Ok(ScalarValue::Null),
            }
        }
        ffi::TD_F64 | ffi::TD_F32 => match table.get_f64(col_idx, row) {
            Some(v) => Ok(ScalarValue::Float(v)),
            None => Ok(ScalarValue::Null),
        },
        ffi::TD_SYM => match table.get_str(col_idx, row) {
            Some(s) => Ok(ScalarValue::Str(s)),
            None => Ok(ScalarValue::Null),
        },
        _ => Err(SqlError::Plan(format!(
            "Unsupported column type {typ} in filter evaluation"
        ))),
    }
}

/// Compare two scalar values with a binary operator.
/// Returns `None` (SQL UNKNOWN) when either operand is NULL.
fn compare_scalars(
    lhs: &ScalarValue,
    rhs: &ScalarValue,
    op: &sql_ast::BinaryOperator,
) -> Result<Option<bool>, SqlError> {
    use sql_ast::BinaryOperator;
    if matches!(lhs, ScalarValue::Null) || matches!(rhs, ScalarValue::Null) {
        return Ok(None); // NULL comparison → UNKNOWN
    }
    match op {
        BinaryOperator::Eq => Ok(Some(lhs.partial_cmp(rhs) == Some(std::cmp::Ordering::Equal))),
        BinaryOperator::NotEq => Ok(Some(lhs.partial_cmp(rhs).map_or(false, |o| o != std::cmp::Ordering::Equal))),
        BinaryOperator::Lt => Ok(Some(lhs.partial_cmp(rhs) == Some(std::cmp::Ordering::Less))),
        BinaryOperator::LtEq => Ok(Some(matches!(
            lhs.partial_cmp(rhs),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ))),
        BinaryOperator::Gt => Ok(Some(lhs.partial_cmp(rhs) == Some(std::cmp::Ordering::Greater))),
        BinaryOperator::GtEq => Ok(Some(matches!(
            lhs.partial_cmp(rhs),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        ))),
        _ => Err(SqlError::Plan(format!(
            "Unsupported comparison operator: {op}"
        ))),
    }
}

/// SQL three-valued AND: false AND _ = false, true AND true = true,
/// otherwise UNKNOWN.
fn tri_and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// SQL three-valued OR: true OR _ = true, false OR false = false,
/// otherwise UNKNOWN.
fn tri_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// SQL three-valued NOT: NOT UNKNOWN = UNKNOWN.
fn tri_not(a: Option<bool>) -> Option<bool> {
    a.map(|v| !v)
}

/// SQL LIKE pattern matching. `%` matches any sequence, `_` matches one char.
/// Supports optional escape character (e.g. `LIKE 'a\%b' ESCAPE '\'`).
fn sql_like_match(text: &str, pattern: &str, escape_char: Option<&str>, case_insensitive: bool) -> bool {
    let (text, pattern): (String, String) = if case_insensitive {
        (text.to_lowercase(), pattern.to_lowercase())
    } else {
        (text.to_string(), pattern.to_string())
    };
    let esc = escape_char.and_then(|e| e.chars().next());
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    like_dp(&t, &p, esc)
}

/// DP-based LIKE matching.
fn like_dp(text: &[char], pattern: &[char], esc: Option<char>) -> bool {
    let (n, m) = (text.len(), pattern.len());
    // dp[j] = whether text[..i] matches pattern[..j]
    let mut dp = vec![false; m + 1];
    dp[0] = true;
    // Initialize: leading %'s match empty text
    for j in 0..m {
        if pattern[j] == '%' && dp[j] {
            dp[j + 1] = true;
        } else {
            break;
        }
    }
    for i in 0..n {
        let mut new_dp = vec![false; m + 1];
        // new_dp[0] is false (non-empty text can't match empty pattern)
        let mut j = 0;
        while j < m {
            if let Some(e) = esc {
                if pattern[j] == e && j + 1 < m {
                    // Escaped character: match literally
                    if dp[j] && text[i] == pattern[j + 1] {
                        new_dp[j + 2] = true;
                    }
                    j += 2;
                    continue;
                }
            }
            match pattern[j] {
                '%' => {
                    // % matches zero or more: dp[j] (skip %) or new_dp[j] (consume char)
                    if dp[j] || new_dp[j] {
                        new_dp[j + 1] = true;
                        new_dp[j] = true; // propagate: % can still consume more
                    }
                }
                '_' => {
                    if dp[j] {
                        new_dp[j + 1] = true;
                    }
                }
                c => {
                    if dp[j] && text[i] == c {
                        new_dp[j + 1] = true;
                    }
                }
            }
            j += 1;
        }
        dp = new_dp;
    }
    dp[m]
}

/// Evaluate a parsed filter expression against a table row.
/// Returns `Some(true)` if the row passes, `Some(false)` if it doesn't,
/// or `None` for SQL UNKNOWN (NULL-involved comparisons).
/// Callers should treat `None` as "exclude row" (i.e. `unwrap_or(false)`).
fn evaluate_filter(
    expr: &sql_ast::Expr,
    table: &Table,
    row: usize,
    var_name: &str,
) -> Result<Option<bool>, SqlError> {
    match expr {
        sql_ast::Expr::BinaryOp { left, op, right } => match op {
            sql_ast::BinaryOperator::And => {
                let l = evaluate_filter(left, table, row, var_name)?;
                if l == Some(false) {
                    return Ok(Some(false));
                }
                let r = evaluate_filter(right, table, row, var_name)?;
                Ok(tri_and(l, r))
            }
            sql_ast::BinaryOperator::Or => {
                let l = evaluate_filter(left, table, row, var_name)?;
                if l == Some(true) {
                    return Ok(Some(true));
                }
                let r = evaluate_filter(right, table, row, var_name)?;
                Ok(tri_or(l, r))
            }
            _ => {
                let lhs = eval_scalar(left, table, row, var_name)?;
                let rhs = eval_scalar(right, table, row, var_name)?;
                compare_scalars(&lhs, &rhs, op)
            }
        },
        sql_ast::Expr::UnaryOp {
            op: sql_ast::UnaryOperator::Not,
            expr: inner,
        } => Ok(tri_not(evaluate_filter(inner, table, row, var_name)?)),
        sql_ast::Expr::InList {
            expr: inner,
            list,
            negated,
        } => {
            let val = eval_scalar(inner, table, row, var_name)?;
            if matches!(val, ScalarValue::Null) {
                return Ok(None); // NULL IN (...) → UNKNOWN
            }
            let mut found = false;
            let mut has_null = false;
            for item in list {
                let v = eval_scalar(item, table, row, var_name)?;
                if matches!(v, ScalarValue::Null) {
                    has_null = true;
                } else if v.partial_cmp(&val) == Some(std::cmp::Ordering::Equal) {
                    found = true;
                    break;
                }
            }
            if found {
                // val IN list → true; val NOT IN list → false
                Ok(Some(!*negated))
            } else if has_null {
                // No match found but NULL in list → UNKNOWN
                Ok(if *negated { None } else { None })
            } else {
                // No match, no NULLs → definite false/true
                Ok(Some(*negated))
            }
        }
        sql_ast::Expr::Between {
            expr: inner,
            negated,
            low,
            high,
        } => {
            let val = eval_scalar(inner, table, row, var_name)?;
            let lo = eval_scalar(low, table, row, var_name)?;
            let hi = eval_scalar(high, table, row, var_name)?;
            if matches!(&val, ScalarValue::Null)
                || matches!(&lo, ScalarValue::Null)
                || matches!(&hi, ScalarValue::Null)
            {
                return Ok(None); // NULL in any operand → UNKNOWN
            }
            let in_range = val >= lo && val <= hi;
            Ok(Some(if *negated { !in_range } else { in_range }))
        }
        sql_ast::Expr::IsNull(inner) => {
            // IS NULL is never UNKNOWN — it always returns true or false
            Ok(Some(matches!(eval_scalar(inner, table, row, var_name)?, ScalarValue::Null)))
        }
        sql_ast::Expr::IsNotNull(inner) => {
            Ok(Some(!matches!(eval_scalar(inner, table, row, var_name)?, ScalarValue::Null)))
        }
        sql_ast::Expr::Nested(inner) => evaluate_filter(inner, table, row, var_name),
        sql_ast::Expr::Like {
            negated,
            expr: inner,
            pattern,
            escape_char,
            ..
        }
        | sql_ast::Expr::ILike {
            negated,
            expr: inner,
            pattern,
            escape_char,
            ..
        } => {
            let case_insensitive = matches!(expr, sql_ast::Expr::ILike { .. });
            let val = eval_scalar(inner, table, row, var_name)?;
            let pat = eval_scalar(pattern, table, row, var_name)?;
            match (&val, &pat) {
                (ScalarValue::Null, _) | (_, ScalarValue::Null) => Ok(None),
                (ScalarValue::Str(s), ScalarValue::Str(p)) => {
                    let matched = sql_like_match(s, p, escape_char.as_deref(), case_insensitive);
                    Ok(Some(if *negated { !matched } else { matched }))
                }
                _ => Err(SqlError::Plan(format!(
                    "LIKE requires string operands, got: {val:?} LIKE {pat:?}"
                ))),
            }
        }
        _ => {
            let v = eval_scalar(expr, table, row, var_name)?;
            match v {
                ScalarValue::Bool(b) => Ok(Some(b)),
                ScalarValue::Null => Ok(None),
                _ => Err(SqlError::Plan(format!(
                    "Unsupported filter expression: {expr}"
                ))),
            }
        }
    }
}

/// Resolve all row indices in a vertex table that match a WHERE filter.
/// Returns the matching row indices.
fn resolve_filtered_node_ids(
    filter_text: &str,
    node: &NodePattern,
    stored: &super::StoredTable,
) -> Result<Vec<i64>, SqlError> {
    let expr = parse_filter_expr(filter_text)?;
    let var = node.variable.as_deref().unwrap_or("");
    let nrows = checked_logical_nrows(stored)?;
    let mut matching = Vec::new();
    for row in 0..nrows {
        if evaluate_filter(&expr, &stored.table, row, var)?.unwrap_or(false) {
            matching.push(row as i64);
        }
    }
    Ok(matching)
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

        // Validate explicit node labels match edge expectations (table + key column)
        let (left_ref_col, right_ref_col) = if is_reverse {
            (&stored_rel.edge_label.dst_ref_col, &stored_rel.edge_label.src_ref_col)
        } else {
            (&stored_rel.edge_label.src_ref_col, &stored_rel.edge_label.dst_ref_col)
        };
        validate_node_table_for_edge(
            left_node,
            &left_label.table_name,
            left_default_table,
            edge_label,
            "source",
            Some(&left_label.key_column),
            Some(left_ref_col),
        )?;
        validate_node_table_for_edge(
            right_node,
            &right_label.table_name,
            right_default_table,
            edge_label,
            "destination",
            Some(&right_label.key_column),
            Some(right_ref_col),
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
            MatchDirection::Bidirectional => {
                return Err(SqlError::Plan(
                    "Bidirectional edges (<-[:]->)  are not supported with variable-length paths".into(),
                ));
            }
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

    // --- Build table name for each position (needed for both post-filtering and projection) ---
    let n_positions = segments.len() + 1;
    let mut pos_table_names: Vec<String> = Vec::with_capacity(n_positions);
    pos_table_names.push(segments[0].src_table_name.clone());
    for seg in &segments {
        pos_table_names.push(seg.dst_table_name.clone());
    }

    // Build per-position vertex label names for PROPERTIES visibility checks.
    let pos_label_names: Vec<String> = nodes.iter().enumerate().map(|(i, node)| {
        if let Some(labels) = &node.labels {
            labels.first().cloned().unwrap_or_default()
        } else if i < pos_table_names.len() {
            graph.vertex_labels.values()
                .find(|vl| vl.table_name == pos_table_names[i])
                .map(|vl| vl.label.clone())
                .unwrap_or_default()
        } else {
            String::new()
        }
    }).collect();

    // --- Determine start nodes ---
    let first_src_table_name = &segments[0].src_table_name;
    let first_src_stored = session.tables.get(first_src_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{first_src_table_name}' not found"))
    })?;

    let start_nodes: Vec<i64> = if let Some(ref filter_text) = nodes[0].filter {
        resolve_filtered_node_ids(filter_text, &nodes[0], first_src_stored)?
    } else {
        let nrows = checked_logical_nrows(first_src_stored)?;
        (0..nrows as i64).collect()
    };

    // --- Run BFS ---
    let bfs_result = multi_segment_bfs(&segments, &start_nodes, mode, is_cyclic)?;

    // Post-filter intermediate/destination nodes
    let bfs_result = {
        let mut keep = vec![true; bfs_result.nrows];
        for (node_pos, node) in nodes.iter().enumerate().skip(1) {
            if let Some(filter_text) = node.filter.as_deref() {
                let expr = parse_filter_expr(filter_text)?;
                let var_name = node.variable.as_deref().unwrap_or("");
                let table_name = &pos_table_names[node_pos];
                let stored_table = session.tables.get(table_name).ok_or_else(|| {
                    SqlError::Plan(format!("Table '{table_name}' not found"))
                })?;
                for row in 0..bfs_result.nrows {
                    if keep[row] {
                        let node_id = bfs_result.node_ids[node_pos][row];
                        if node_id < 0 {
                            keep[row] = false;
                        } else {
                            keep[row] = evaluate_filter(
                                &expr,
                                &stored_table.table,
                                node_id as usize,
                                var_name,
                            )?.unwrap_or(false);
                        }
                    }
                }
            }
        }
        // Rebuild BFS result with only matching rows
        if keep.iter().all(|&k| k) {
            bfs_result
        } else {
            let new_nrows = keep.iter().filter(|&&k| k).count();
            let mut new_node_ids: Vec<Vec<i64>> = (0..bfs_result.node_ids.len())
                .map(|_| Vec::with_capacity(new_nrows))
                .collect();
            let mut new_path_lengths = Vec::with_capacity(new_nrows);
            for row in 0..bfs_result.nrows {
                if keep[row] {
                    for (pos, ids) in bfs_result.node_ids.iter().enumerate() {
                        new_node_ids[pos].push(ids[row]);
                    }
                    new_path_lengths.push(bfs_result.path_lengths[row]);
                }
            }
            BfsResult {
                node_ids: new_node_ids,
                path_lengths: new_path_lengths,
                nrows: new_nrows,
            }
        }
    };

    // --- Build variable map: node variable name -> position index ---
    let mut var_map: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some(ref var) = node.variable {
            var_map.entry(var.clone()).or_insert(i);
        }
    }

    // Build edge variable map: edge variable name -> edge index
    let mut edge_var_map: HashMap<String, usize> = HashMap::new();
    for (i, edge) in edges.iter().enumerate() {
        if let Some(ref var) = edge.variable {
            edge_var_map.entry(var.clone()).or_insert(i);
        }
    }

    // --- Project COLUMNS ---
    enum MhVarColKind {
        Property { var_idx: usize, table_col_idx: usize, table_name: String },
        EdgeProperty { edge_idx: usize, table_col_idx: usize, edge_label: String, is_reverse: bool },
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

            if let Some(var_idx) = var_map.get(&var) {
                let table_name = &pos_table_names[*var_idx];
                // Check PROPERTIES visibility for this vertex label
                let label_name = &pos_label_names[*var_idx];
                if let Some(vl) = graph.vertex_labels.get(label_name.as_str()) {
                    check_column_visible(&vl.visibility, &col, label_name)?;
                }
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
            } else if let Some(edge_idx) = edge_var_map.get(&var) {
                let edge = &edges[*edge_idx];
                let edge_label_name = edge.label.as_deref().ok_or_else(|| {
                    SqlError::Plan("Edge variable requires a label to access properties".into())
                })?;
                let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge label '{edge_label_name}' not found in graph"))
                })?;
                // Check PROPERTIES visibility for this edge label
                check_column_visible(&stored_rel.edge_label.visibility, &col, edge_label_name)?;
                let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                })?;
                let table_col_idx = find_col_idx(&edge_table.table, &col).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{col}' not found in edge table '{}'",
                        stored_rel.edge_label.table_name
                    ))
                })?;
                let is_reverse = edge.direction == MatchDirection::Reverse;

                col_names.push(out_name);
                col_specs.push(MhVarColKind::EdgeProperty {
                    edge_idx: *edge_idx,
                    table_col_idx,
                    edge_label: edge_label_name.to_string(),
                    is_reverse,
                });
            } else {
                let mut available: Vec<_> = var_map.keys().collect();
                available.extend(edge_var_map.keys());
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available:?}"
                )));
            }
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
                MhVarColKind::EdgeProperty { edge_idx, table_col_idx, edge_label, is_reverse } => {
                    let left_node = bfs_result.node_ids[*edge_idx][row];
                    let right_node = bfs_result.node_ids[*edge_idx + 1][row];
                    let (edge_src, edge_dst) = if *is_reverse {
                        (right_node, left_node)
                    } else {
                        (left_node, right_node)
                    };
                    let stored_rel = graph.edge_labels.get(edge_label.as_str()).ok_or_else(|| {
                        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
                    })?;
                    let edge_rows = stored_rel.edge_row_map.get(&(edge_src, edge_dst))
                        .ok_or_else(|| SqlError::Plan(format!(
                            "No edge found for ({edge_src}, {edge_dst}) in edge label '{edge_label}'"
                        )))?;
                    let edge_row = edge_rows[0];
                    let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                        SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                    })?;
                    csv.push_str(&get_cell_string(&edge_table.table, *table_col_idx, edge_row)?);
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


/// Plan a multi-hop fixed MATCH: (a)-[e1]->(b)-[e2]->(c) ... where all edges
/// have `PathQuantifier::One`. Uses `wco_join` (Leapfrog Triejoin) to find
/// matching node tuples.
fn plan_multi_hop_fixed(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
    mode: &PathMode,
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

        // Validate explicit node labels match edge expectations (table + key column)
        let (left_ref_col, right_ref_col) = if is_reverse {
            (&stored_rel.edge_label.dst_ref_col, &stored_rel.edge_label.src_ref_col)
        } else {
            (&stored_rel.edge_label.src_ref_col, &stored_rel.edge_label.dst_ref_col)
        };
        validate_node_table_for_edge(
            left_node,
            &left_label.table_name,
            left_default_table,
            edge_label,
            "source",
            Some(&left_label.key_column),
            Some(left_ref_col),
        )?;
        validate_node_table_for_edge(
            right_node,
            &right_label.table_name,
            right_default_table,
            edge_label,
            "destination",
            Some(&right_label.key_column),
            Some(right_ref_col),
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
    // The C LFTJ default plan doesn't support cyclic variable mappings, so delegate
    // cyclic fixed-hop patterns to the BFS planner which handles is_cyclic natively.
    let is_cyclic = match (&nodes[0].variable, &nodes[nodes.len() - 1].variable) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    if is_cyclic {
        return plan_multi_hop_variable(session, graph, pattern, columns, mode);
    }

    // Detect patterns with reverse edges that cross vertex table boundaries.
    // The C LFTJ chain plan maps rel[i] as var[i]→var[i+1] and doesn't
    // account for direction reversal on heterogeneous edges. Forward cross-table
    // edges work fine (e.g., Person→Knows→Person→LivesIn→City) because the LFTJ
    // naturally resolves the chain. But reverse edges on cross-table segments
    // need BFS to correctly flip the CSR direction.
    let has_reverse_cross_table = edges.iter().enumerate().any(|(i, e)| {
        e.direction == MatchDirection::Reverse
            && segments[i].src_table_name != segments[i].dst_table_name
    });
    if has_reverse_cross_table {
        return plan_multi_hop_variable(session, graph, pattern, columns, mode);
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

    // Build a lookup: for each node variable position, which vertex table does it reference?
    let mut var_table_names: Vec<String> = Vec::with_capacity(n_vars);
    var_table_names.push(segments[0].src_table_name.clone());
    for seg in &segments {
        var_table_names.push(seg.dst_table_name.clone());
    }
    // If cyclic, the last entry maps back to var 0 (already handled by truncating to n_vars)
    var_table_names.truncate(n_vars);

    // Build per-node-position vertex label names for PROPERTIES visibility checks.
    let var_label_names: Vec<String> = nodes.iter().enumerate().map(|(i, node)| {
        if let Some(labels) = &node.labels {
            labels.first().cloned().unwrap_or_default()
        } else if i < var_table_names.len() {
            // Find the label name from the graph that matches this table
            graph.vertex_labels.values()
                .find(|vl| vl.table_name == var_table_names[i])
                .map(|vl| vl.label.clone())
                .unwrap_or_default()
        } else {
            String::new()
        }
    }).take(n_vars).collect();

    // Apply source node (v0) WHERE filter if present
    let mut row_mask: Vec<bool> = if let Some(filter_text) = nodes[0].filter.as_deref() {
        let expr = parse_filter_expr(filter_text)?;
        let var_name = nodes[0].variable.as_deref().unwrap_or("");
        let src_table = session
            .tables
            .get(&segments[0].src_table_name)
            .ok_or_else(|| {
                SqlError::Plan(format!(
                    "Table '{}' not found",
                    segments[0].src_table_name
                ))
            })?;
        let mut mask = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let val = node_ids[0][row];
            if val < 0 {
                return Err(SqlError::Plan(format!("Negative node index {val} for _v0 at row {row}")));
            }
            let node_row = val as usize;
            mask.push(evaluate_filter(&expr, &src_table.table, node_row, var_name)?.unwrap_or(false));
        }
        mask
    } else {
        vec![true; nrows]
    };

    // Apply WHERE filters on intermediate/destination nodes (nodes[1..])
    for (node_pos, node) in nodes.iter().enumerate().skip(1) {
        if let Some(filter_text) = node.filter.as_deref() {
            let expr = parse_filter_expr(filter_text)?;
            let var_name = node.variable.as_deref().unwrap_or("");
            // For cyclic patterns, the last node maps to var 0
            let var_idx = if is_cyclic && node_pos == n_nodes - 1 { 0 } else { node_pos };
            let table_name = &var_table_names[var_idx];
            let stored_table = session.tables.get(table_name).ok_or_else(|| {
                SqlError::Plan(format!("Table '{table_name}' not found"))
            })?;
            for row in 0..nrows {
                if row_mask[row] {
                    let node_id = node_ids[var_idx][row];
                    if node_id < 0 {
                        row_mask[row] = false;
                    } else {
                        row_mask[row] = evaluate_filter(
                            &expr,
                            &stored_table.table,
                            node_id as usize,
                            var_name,
                        )?.unwrap_or(false);
                    }
                }
            }
        }
    }

    // Build edge variable map: edge variable name -> edge index
    let mut edge_var_map: HashMap<String, usize> = HashMap::new();
    for (i, edge) in edges.iter().enumerate() {
        if let Some(ref var) = edge.variable {
            edge_var_map.entry(var.clone()).or_insert(i);
        }
    }

    // Project COLUMNS
    enum ColSpec {
        Node { var_idx: usize, table_col_idx: usize, table_name: String },
        Edge { edge_idx: usize, table_col_idx: usize, edge_label: String, is_reverse: bool },
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

            if let Some(var_idx) = var_map.get(&var) {
                let table_name = &var_table_names[*var_idx];
                // Check PROPERTIES visibility for this vertex label
                let label_name = &var_label_names[*var_idx];
                if let Some(vl) = graph.vertex_labels.get(label_name.as_str()) {
                    check_column_visible(&vl.visibility, &col, label_name)?;
                }
                let vtable = session.tables.get(table_name).ok_or_else(|| {
                    SqlError::Plan(format!("Table '{table_name}' not found"))
                })?;
                let table_col_idx = find_col_idx(&vtable.table, &col).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{col}' not found in vertex table '{table_name}'"
                    ))
                })?;

                col_names.push(out_name);
                col_specs.push(ColSpec::Node {
                    var_idx: *var_idx,
                    table_col_idx,
                    table_name: table_name.clone(),
                });
            } else if let Some(edge_idx) = edge_var_map.get(&var) {
                let edge = &edges[*edge_idx];
                let edge_label_name = edge.label.as_deref().ok_or_else(|| {
                    SqlError::Plan("Edge variable requires a label to access properties".into())
                })?;
                let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge label '{edge_label_name}' not found in graph"))
                })?;
                // Check PROPERTIES visibility for this edge label
                check_column_visible(&stored_rel.edge_label.visibility, &col, edge_label_name)?;
                let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                    SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                })?;
                let table_col_idx = find_col_idx(&edge_table.table, &col).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{col}' not found in edge table '{}'",
                        stored_rel.edge_label.table_name
                    ))
                })?;
                let is_reverse = edge.direction == MatchDirection::Reverse;

                col_names.push(out_name);
                col_specs.push(ColSpec::Edge {
                    edge_idx: *edge_idx,
                    table_col_idx,
                    edge_label: edge_label_name.to_string(),
                    is_reverse,
                });
            } else {
                let mut available: Vec<_> = var_map.keys().collect();
                available.extend(edge_var_map.keys());
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available:?}"
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
            match spec {
                ColSpec::Node { var_idx, table_col_idx, table_name } => {
                    let val = node_ids[*var_idx][row];
                    if val < 0 {
                        return Err(SqlError::Plan(format!(
                            "Negative node index {} for _v{} at row {}", val, var_idx, row
                        )));
                    }
                    let node_row = val as usize;
                    let vtable = session.tables.get(table_name.as_str()).ok_or_else(|| {
                        SqlError::Plan(format!("Table '{}' not found", table_name))
                    })?;
                    csv.push_str(&get_cell_string(&vtable.table, *table_col_idx, node_row)?);
                }
                ColSpec::Edge { edge_idx, table_col_idx, edge_label, is_reverse } => {
                    // For edge at index i, the node pair is (node[i], node[i+1])
                    let left_node = node_ids[*edge_idx][row];
                    let right_node = node_ids[*edge_idx + 1][row];
                    // Convert to edge's natural direction for edge_row_map lookup
                    let (edge_src, edge_dst) = if *is_reverse {
                        (right_node, left_node)
                    } else {
                        (left_node, right_node)
                    };
                    let stored_rel = graph.edge_labels.get(edge_label.as_str()).ok_or_else(|| {
                        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
                    })?;
                    let edge_rows = stored_rel.edge_row_map.get(&(edge_src, edge_dst))
                        .ok_or_else(|| SqlError::Plan(format!(
                            "No edge found for ({edge_src}, {edge_dst}) in edge label '{edge_label}'"
                        )))?;
                    let edge_row = edge_rows[0];
                    let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or_else(|| {
                        SqlError::Plan(format!("Edge table '{}' not found", stored_rel.edge_label.table_name))
                    })?;
                    csv.push_str(&get_cell_string(&edge_table.table, *table_col_idx, edge_row)?);
                }
            }
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

    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
    })?;

    // For reverse edges, swap default table/column references (same as plan_single_hop)
    let is_reverse = edge.direction == MatchDirection::Reverse;
    let (left_default_table, right_default_table) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table)
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

    let (left_ref_col_vl, right_ref_col_vl) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_col, &stored_rel.edge_label.src_ref_col)
    } else {
        (&stored_rel.edge_label.src_ref_col, &stored_rel.edge_label.dst_ref_col)
    };
    let src_label = resolve_node_label(src_node, left_default_table, graph)?;
    validate_node_table_for_edge(src_node, &src_label.table_name, left_default_table, edge_label, "source", Some(&src_label.key_column), Some(left_ref_col_vl))?;
    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;

    let dst_label = resolve_node_label(dst_node, right_default_table, graph)?;
    validate_node_table_for_edge(dst_node, &dst_label.table_name, right_default_table, edge_label, "destination", Some(&dst_label.key_column), Some(right_ref_col_vl))?;
    let dst_table = &session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?.table;

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
        MatchDirection::Bidirectional => {
            return Err(SqlError::Plan(
                "Bidirectional edges (<-[:]->)  are not supported with variable-length paths".into(),
            ));
        }
    };

    let (min_depth, max_depth) = match edge.quantifier {
        PathQuantifier::Range { min, max } => (min, max),
        PathQuantifier::Plus => (1, 255),
        PathQuantifier::Star => (0, 255),
        PathQuantifier::One => (1, 1),
    };

    // Determine source row indices
    let src_stored = session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?;
    let src_row_indices: Vec<i64> = if let Some(filter_text) = &src_node.filter {
        resolve_filtered_node_ids(filter_text, src_node, src_stored)?
    } else {
        let nrows = checked_logical_nrows(src_stored)?;
        (0..nrows as i64).collect()
    };

    let rowid_table = build_rowid_table(session, &src_row_indices)?;
    let mut g = session.ctx.graph(&rowid_table)?;
    let src_ids = g.scan("_rowid")?;

    let var_exp = g.var_expand(src_ids, &stored_rel.rel, direction, min_depth, max_depth, false)?;
    let result = g.execute(var_exp)?;

    // Post-filter destination node on var_expand result
    let result = if let Some(filter_text) = &dst_node.filter {
        let expr = parse_filter_expr(filter_text)?;
        let var_name = dst_node.variable.as_deref().unwrap_or("");
        let dst_stored_table = session.tables.get(&dst_label.table_name).ok_or_else(|| {
            SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
        })?;
        let nrows = checked_nrows(&result)?;
        let start_col = find_col_idx(&result, "_start")
            .ok_or_else(|| SqlError::Plan("var_expand result missing _start column".into()))?;
        let end_col = find_col_idx(&result, "_end")
            .ok_or_else(|| SqlError::Plan("var_expand result missing _end column".into()))?;
        let depth_col = find_col_idx(&result, "_depth");

        let mut csv = String::from("_start,_end,_depth\n");
        for row in 0..nrows {
            let end_val = result.get_i64(end_col, row)
                .ok_or_else(|| SqlError::Plan(format!("NULL _end at row {row}")))?;
            if end_val < 0 { continue; }
            if evaluate_filter(&expr, &dst_stored_table.table, end_val as usize, var_name)?.unwrap_or(false) {
                let start_val = result.get_i64(start_col, row)
                    .ok_or_else(|| SqlError::Plan(format!("NULL _start at row {row}")))?;
                let depth_val = depth_col
                    .and_then(|ci| result.get_i64(ci, row))
                    .unwrap_or(0);
                csv.push_str(&format!("{start_val},{end_val},{depth_val}\n"));
            }
        }
        csv_to_table(session, &csv, &["_start".to_string(), "_end".to_string(), "_depth".to_string()])?
    } else {
        result
    };

    // var_expand result has: _start, _end, _depth
    project_var_length_columns(session, &result, columns, src_node, dst_node, src_table, dst_table, edge, src_label, dst_label)
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
    edge_pattern: &EdgePattern,
    src_vertex_label: &VertexLabel,
    dst_vertex_label: &VertexLabel,
) -> Result<(Table, Vec<String>), SqlError> {
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
    let edge_var = edge_pattern.variable.as_deref();

    let nrows = checked_nrows(var_result)?;

    let start_idx_col = find_col_idx(var_result, "_start")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _start column".into()))?;
    let end_idx_col = find_col_idx(var_result, "_end")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _end column".into()))?;
    let depth_idx_col = find_col_idx(var_result, "_depth");

    let mut start_indices = Vec::with_capacity(nrows);
    let mut end_indices = Vec::with_capacity(nrows);
    let mut depth_values: Vec<i64> = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let start_val = var_result.get_i64(start_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _start index at row {row}")))?;
        if start_val < 0 {
            return Err(SqlError::Plan(format!("Negative _start index {start_val} at row {row}")));
        }
        let end_val = var_result.get_i64(end_idx_col, row)
            .ok_or_else(|| SqlError::Plan(format!("NULL _end index at row {row}")))?;
        if end_val < 0 {
            return Err(SqlError::Plan(format!("Negative _end index {end_val} at row {row}")));
        }
        start_indices.push(start_val as usize);
        end_indices.push(end_val as usize);
        depth_values.push(
            depth_idx_col
                .and_then(|ci| var_result.get_i64(ci, row))
                .unwrap_or(0),
        );
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
                // Check PROPERTIES visibility for source vertex label
                check_column_visible(&src_vertex_label.visibility, &col, &src_vertex_label.label)?;
                let col_idx = find_col_idx(src_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in source table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { kind: VarColKind::Src, table_col_idx: col_idx });
            } else if var == dst_var {
                // Check PROPERTIES visibility for destination vertex label
                check_column_visible(&dst_vertex_label.visibility, &col, &dst_vertex_label.label)?;
                let col_idx = find_col_idx(dst_table, &col)
                    .ok_or_else(|| SqlError::Plan(format!("Column '{col}' not found in destination table")))?;
                col_names.push(out_name);
                col_specs.push(ColSpec { kind: VarColKind::Dst, table_col_idx: col_idx });
            } else if edge_var.is_some() && var == edge_var.unwrap() {
                return Err(SqlError::Plan(format!(
                    "Edge property access on variable-length edges is not supported. \
                     Edge variable '{var}' cannot be used with property access in variable-length patterns."
                )));
            } else {
                let mut available = format!("{src_var}, {dst_var}");
                if let Some(ev) = edge_var {
                    available.push_str(&format!(", {ev}"));
                }
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS. Available: {available}"
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
                    csv.push_str(&depth_values[row].to_string());
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

/// Column specification for shortest-path COLUMNS clause.
enum SpColKind {
    /// `_node` — path node ID (user key) at this depth
    Node,
    /// `_depth` — integer depth in the path
    Depth,
    /// `path_length(p)` — total path length (constant per row)
    PathLength,
    /// `vertices(p)` — list of vertex IDs along the path as text
    Vertices,
    /// `edges(p)` — list of edge pairs along the path as text
    Edges,
    /// Source variable property lookup — column index in the left vertex table
    SrcProp(usize),
    /// Destination variable property lookup — column index in the right vertex table
    DstProp(usize),
}

/// Parse the COLUMNS clause for a shortest-path query, producing column names
/// and `SpColKind` descriptors.  This is shared by the 0-hop, empty-path, and
/// normal-path code paths in `plan_shortest_path`.
fn parse_sp_columns(
    columns: &[ColumnEntry],
    src_var: &str,
    dst_var: &str,
    left_table: &str,
    right_table: &str,
    session: &Session,
    graph: &PropertyGraph,
) -> Result<(Vec<String>, Vec<SpColKind>), SqlError> {
    let mut col_names: Vec<String> = Vec::new();
    let mut col_kinds: Vec<SpColKind> = Vec::new();

    for entry in columns {
        let lower = entry.expr.to_lowercase();
        let alias = entry.alias.as_deref();

        // Strip whitespace for matching function-call syntax like "vertices ( p )"
        let compact = lower.replace(' ', "");
        if compact.starts_with("vertices(") {
            col_names.push(alias.unwrap_or("vertices").to_string());
            col_kinds.push(SpColKind::Vertices);
        } else if compact.starts_with("edges(") {
            col_names.push(alias.unwrap_or("edges").to_string());
            col_kinds.push(SpColKind::Edges);
        } else if lower.contains("path_length") {
            col_names.push(alias.unwrap_or("path_length").to_string());
            col_kinds.push(SpColKind::PathLength);
        } else if lower == "_node" || lower == "node" {
            col_names.push(
                alias
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| lower.clone()),
            );
            col_kinds.push(SpColKind::Node);
        } else if lower == "_depth" || lower == "depth" {
            col_names.push(
                alias
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| lower.clone()),
            );
            col_kinds.push(SpColKind::Depth);
        } else if let Some(dot_pos) = lower.find('.') {
            let var = lower[..dot_pos].trim();
            let prop = lower[dot_pos + 1..].trim();
            if var == src_var {
                // Check PROPERTIES visibility for source vertex label
                if let Some(vl) = graph.vertex_labels.values().find(|vl| vl.table_name == left_table) {
                    check_column_visible(&vl.visibility, prop, &vl.label)?;
                }
                let stored = session.tables.get(left_table).ok_or_else(|| {
                    SqlError::Plan(format!("Table '{left_table}' not found"))
                })?;
                let col_idx = find_col_idx(&stored.table, prop).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{prop}' not found in vertex table '{left_table}'"
                    ))
                })?;
                col_names.push(alias.unwrap_or(prop).to_string());
                col_kinds.push(SpColKind::SrcProp(col_idx));
            } else if var == dst_var {
                // Check PROPERTIES visibility for destination vertex label
                if let Some(vl) = graph.vertex_labels.values().find(|vl| vl.table_name == right_table) {
                    check_column_visible(&vl.visibility, prop, &vl.label)?;
                }
                let stored = session.tables.get(right_table).ok_or_else(|| {
                    SqlError::Plan(format!("Table '{right_table}' not found"))
                })?;
                let col_idx = find_col_idx(&stored.table, prop).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Column '{prop}' not found in vertex table '{right_table}'"
                    ))
                })?;
                col_names.push(alias.unwrap_or(prop).to_string());
                col_kinds.push(SpColKind::DstProp(col_idx));
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in SHORTEST_PATH COLUMNS"
                )));
            }
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
    Ok((col_names, col_kinds))
}

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
    let (left_table, right_table) = if is_reverse {
        (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table)
    } else {
        (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table)
    };

    // Validate explicit node labels against the edge's expected tables and
    // key columns.  Checking only table_name is insufficient when the graph
    // defines multiple vertex labels over the same table with different KEY
    // columns — the label must also match the edge's referenced key column.
    let left_ref_col_for_validate = if is_reverse {
        &stored_rel.edge_label.dst_ref_col
    } else {
        &stored_rel.edge_label.src_ref_col
    };
    let right_ref_col_for_validate = if is_reverse {
        &stored_rel.edge_label.src_ref_col
    } else {
        &stored_rel.edge_label.dst_ref_col
    };
    if let Some(labels) = &src_node.labels {
        let label = labels.first().map(|s| s.as_str()).unwrap_or("?");
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
        if src_vl.key_column != *left_ref_col_for_validate {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' uses key column '{}', but edge '{edge_label}' \
                 references source key column '{left_ref_col_for_validate}'",
                src_vl.key_column
            )));
        }
    }
    if let Some(labels) = &dst_node.labels {
        let label = labels.first().map(|s| s.as_str()).unwrap_or("?");
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
        if dst_vl.key_column != *right_ref_col_for_validate {
            return Err(SqlError::Plan(format!(
                "Node label '{label}' uses key column '{}', but edge '{edge_label}' \
                 references destination key column '{right_ref_col_for_validate}'",
                dst_vl.key_column
            )));
        }
    }

    // Undirected traversals on heterogeneous edges (different src/dst tables)
    // would mix node IDs from different tables in the BFS, producing wrong results.
    // This mirrors the same check in plan_single_hop.
    if edge.direction == MatchDirection::Undirected
        && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
    {
        return Err(SqlError::Plan(
            "Undirected traversals are not supported on edges with different source and \
             destination vertex tables (heterogeneous graphs). Use a directed pattern instead."
                .into(),
        ));
    }

    // Shortest-path BFS requires src and dst to reference the same vertex table.
    // When both sides reference the same table (even with different key columns),
    // n_src == n_dst so BFS over row indices is safe.
    // Range{1,1} and PathQuantifier::One are semantically single-hop, so skip
    // this check for them (same exemption as plan_var_length).
    let is_single_hop = matches!(edge.quantifier, PathQuantifier::One | PathQuantifier::Range { min: 1, max: 1 });
    if !is_single_hop
        && stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table
    {
        return Err(SqlError::Plan(
            "SHORTEST_PATH with variable-length quantifiers is not supported on edges \
             whose source and destination reference different vertex tables (heterogeneous endpoints)."
                .into(),
        ));
    }

    let src_id = extract_node_id(src_node, left_table, session)?;
    let dst_id = extract_node_id(dst_node, right_table, session)?;

    // Look up vertex labels for _node output (row index -> user key).
    // Match on both table name and key column (from the edge's ref column) to
    // handle graphs with multiple labels on the same table keyed differently.
    let left_ref_col = if is_reverse {
        &stored_rel.edge_label.dst_ref_col
    } else {
        &stored_rel.edge_label.src_ref_col
    };
    let left_vertex_label = graph.vertex_labels.values()
        .find(|vl| vl.table_name == *left_table && vl.key_column == *left_ref_col)
        .ok_or_else(|| SqlError::Plan(format!(
            "No vertex label found for table '{left_table}' with key column '{left_ref_col}'"
        )))?;
    // The destination nodes may need a separate vertex label: either because
    // they belong to a different table, or because they reference the same
    // table with a different key column.  This applies to both single-hop and
    // multi-hop paths (for multi-hop same-table edges with mixed keys, the
    // terminal node must be decoded via the destination key map).
    let right_vertex_label = {
        let right_ref_col = if is_reverse {
            &stored_rel.edge_label.src_ref_col
        } else {
            &stored_rel.edge_label.dst_ref_col
        };
        if left_table != right_table || left_ref_col != right_ref_col {
            Some(graph.vertex_labels.values()
                .find(|vl| vl.table_name == *right_table && vl.key_column == *right_ref_col)
                .ok_or_else(|| SqlError::Plan(format!(
                    "No vertex label found for table '{right_table}' with key column '{right_ref_col}'"
                )))?)
        } else {
            None
        }
    };

    let (min_depth, max_depth): (u8, u8) = match edge.quantifier {
        PathQuantifier::Range { min, max } => (min, max),
        PathQuantifier::Plus => (1, 255),
        PathQuantifier::Star => (0, 255),
        PathQuantifier::One => (1, 1),
    };

    // Handle 0-hop match: if min_depth is 0 and src == dst, return immediately
    if min_depth == 0 && src_id == dst_id {
        let src_var = src_node.variable.as_deref().unwrap_or("__src");
        let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
        let (col_names, col_kinds) =
            parse_sp_columns(columns, src_var, dst_var, left_table, right_table, session, graph)?;

        // Build CSV with correct values: node=user_key, depth=0, path_length=0
        let src_key = left_vertex_label.row_to_user.get(src_id as usize)
            .ok_or_else(|| SqlError::Plan(format!(
                "Row index {src_id} out of bounds in vertex table '{left_table}'"
            )))?;
        let left_stored = session.tables.get(left_table.as_str());
        let right_stored = session.tables.get(right_table.as_str());
        let csv_col_names: Vec<String> = (0..col_names.len())
            .map(|i| format!("__c{i}"))
            .collect();
        let mut csv = csv_col_names.join(",");
        csv.push('\n');
        for (i, kind) in col_kinds.iter().enumerate() {
            if i > 0 { csv.push(','); }
            match kind {
                SpColKind::Node => csv.push_str(&src_key.to_csv()),
                SpColKind::Depth | SpColKind::PathLength => csv.push('0'),
                SpColKind::Vertices => {
                    // 0-hop: single node
                    csv.push_str(&csv_quote(&format!("[{}]", src_key.to_csv())));
                }
                SpColKind::Edges => {
                    // 0-hop: no edges
                    csv.push_str(&csv_quote("[]"));
                }
                SpColKind::SrcProp(idx) => {
                    let t = left_stored.ok_or_else(|| {
                        SqlError::Plan(format!("Table '{left_table}' not found"))
                    })?;
                    csv.push_str(&get_cell_string(&t.table, *idx, src_id as usize)?);
                }
                SpColKind::DstProp(idx) => {
                    let t = right_stored.ok_or_else(|| {
                        SqlError::Plan(format!("Table '{right_table}' not found"))
                    })?;
                    csv.push_str(&get_cell_string(&t.table, *idx, dst_id as usize)?);
                }
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
        MatchDirection::Bidirectional => {
            return Err(SqlError::Plan(
                "Bidirectional edges (<-[:]->)  are not supported with shortest path queries".into(),
            ));
        }
    };

    // BFS over the CSR to find the shortest qualifying path
    let mut path = reconstruct_shortest_path(
        src_id, dst_id, min_depth as i64, max_depth as i64, stored_rel, direction,
    )?;

    if path.is_empty() {
        // No path found → return empty result table with proper column names
        // so that the outer query can resolve column references (e.g. ORDER BY _depth).
        let src_var = src_node.variable.as_deref().unwrap_or("__src");
        let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
        let (display_names, _col_kinds) =
            parse_sp_columns(columns, src_var, dst_var, left_table, right_table, session, graph)?;

        let csv_col_names: Vec<String> = (0..display_names.len())
            .map(|i| format!("__c{i}"))
            .collect();
        let csv = format!("{}\n", csv_col_names.join(","));
        let result = csv_to_table(session, &csv, &display_names)?;
        return Ok((result, display_names));
    }

    path.reverse(); // path is built backwards, reverse to get src -> dst order

    // Parse COLUMNS clause
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
    let (col_names, col_kinds) =
        parse_sp_columns(columns, src_var, dst_var, left_table, right_table, session, graph)?;

    // Total path length = number of edges = number of nodes - 1
    let total_path_length = if path.is_empty() { 0 } else { path.len() - 1 };

    // Pre-fetch table references for property lookups (only if needed)
    let left_stored = session.tables.get(left_table.as_str());
    let right_stored = session.tables.get(right_table.as_str());

    // Build CSV
    let csv_col_names: Vec<String> = (0..col_names.len())
        .map(|i| format!("__c{i}"))
        .collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    let last_idx = path.len().saturating_sub(1);

    // Pre-compute vertices(p) and edges(p) strings (constant per path).
    // These are CSV-quoted because they contain commas.
    let needs_vertices = col_kinds.iter().any(|k| matches!(k, SpColKind::Vertices));
    let needs_edges = col_kinds.iter().any(|k| matches!(k, SpColKind::Edges));
    let vertices_str = if needs_vertices {
        let mut parts = Vec::with_capacity(path.len());
        for (i, &nid) in path.iter().enumerate() {
            let vl = if i == last_idx && right_vertex_label.is_some() {
                right_vertex_label.unwrap()
            } else {
                left_vertex_label
            };
            parts.push(
                vl.row_to_user
                    .get(nid as usize)
                    .map(|k| k.to_csv())
                    .unwrap_or_else(|| nid.to_string()),
            );
        }
        csv_quote(&format!("[{}]", parts.join(", ")))
    } else {
        String::new()
    };
    let edges_str = if needs_edges {
        let mut parts = Vec::with_capacity(path.len().saturating_sub(1));
        for i in 0..path.len().saturating_sub(1) {
            let src_vl = if i == last_idx && right_vertex_label.is_some() {
                right_vertex_label.unwrap()
            } else {
                left_vertex_label
            };
            let dst_vl = if i + 1 == last_idx && right_vertex_label.is_some() {
                right_vertex_label.unwrap()
            } else {
                left_vertex_label
            };
            let sk = src_vl
                .row_to_user
                .get(path[i] as usize)
                .map(|k| k.to_csv())
                .unwrap_or_else(|| path[i].to_string());
            let dk = dst_vl
                .row_to_user
                .get(path[i + 1] as usize)
                .map(|k| k.to_csv())
                .unwrap_or_else(|| path[i + 1].to_string());
            parts.push(format!("({sk}, {dk})"));
        }
        csv_quote(&format!("[{}]", parts.join(", ")))
    } else {
        String::new()
    };

    for (depth, &node_id) in path.iter().enumerate() {
        // For single-hop heterogeneous edges, the last node in the path
        // belongs to the right (destination) table and needs a different
        // vertex label for correct key decoding.
        let vl = if depth == last_idx && right_vertex_label.is_some() {
            right_vertex_label.unwrap()
        } else {
            left_vertex_label
        };
        let decode_table = if depth == last_idx && right_vertex_label.is_some() {
            right_table
        } else {
            left_table
        };
        let user_key = vl.row_to_user.get(node_id as usize)
            .ok_or_else(|| SqlError::Plan(format!(
                "Row index {node_id} out of bounds in vertex table '{decode_table}'"
            )))?;
        for (i, kind) in col_kinds.iter().enumerate() {
            if i > 0 { csv.push(','); }
            match kind {
                SpColKind::Node => csv.push_str(&user_key.to_csv()),
                SpColKind::Depth => csv.push_str(&depth.to_string()),
                SpColKind::PathLength => csv.push_str(&total_path_length.to_string()),
                SpColKind::Vertices => csv.push_str(&vertices_str),
                SpColKind::Edges => csv.push_str(&edges_str),
                SpColKind::SrcProp(idx) => {
                    let t = left_stored.ok_or_else(|| {
                        SqlError::Plan(format!("Table '{left_table}' not found"))
                    })?;
                    csv.push_str(&get_cell_string(&t.table, *idx, src_id as usize)?);
                }
                SpColKind::DstProp(idx) => {
                    let t = right_stored.ok_or_else(|| {
                        SqlError::Plan(format!("Table '{right_table}' not found"))
                    })?;
                    csv.push_str(&get_cell_string(&t.table, *idx, dst_id as usize)?);
                }
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}

// ---------------------------------------------------------------------------
// Cheapest (weighted shortest) path planner
// ---------------------------------------------------------------------------

/// Plan an ANY SHORTEST MATCH with COST expression using Dijkstra.
///
/// When an edge pattern carries a COST expression (e.g. `COST r.weight`),
/// the planner invokes the C engine's Dijkstra kernel to find the
/// minimum-weight path between source and destination nodes.
///
/// Result: one summary row per query with source/destination properties
/// and `path_cost(p)` for the total weight of the cheapest path.
fn plan_cheapest_path(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let src_node = &pattern.nodes[0];
    let edge = &pattern.edges[0];
    let dst_node = &pattern.nodes[1];

    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label for COST queries".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found"))
    })?;

    // Dijkstra requires same src/dst vertex table (CSR constraint).
    if stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table {
        return Err(SqlError::Plan(
            "CHEAPEST path (COST) is not supported on edges with different source and \
             destination vertex tables (heterogeneous endpoints)."
                .into(),
        ));
    }

    let is_reverse = edge.direction == MatchDirection::Reverse;
    let (left_table, right_table) = if is_reverse {
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

    // Resolve source and destination node IDs from WHERE filters
    let src_id = extract_node_id(src_node, left_table, session)?;
    let dst_id = extract_node_id(dst_node, right_table, session)?;

    // Resolve cost expression to weight column name.
    // COST expression is like "e.distance" or "r.weight" — extract the column part.
    let cost_text = edge.cost_expr.as_deref().ok_or_else(|| {
        SqlError::Plan("Internal error: plan_cheapest_path called without cost_expr".into())
    })?;
    let weight_col = if let Some(dot_pos) = cost_text.find('.') {
        &cost_text[dot_pos + 1..]
    } else {
        cost_text
    };

    // Attach edge properties so Dijkstra can read weight values.
    let edge_table_name = &stored_rel.edge_label.table_name;
    let edge_table = &session
        .tables
        .get(edge_table_name)
        .ok_or_else(|| SqlError::Plan(format!("Edge table '{edge_table_name}' not found")))?
        .table;
    stored_rel.rel.set_props(edge_table);

    let src_table_name = &stored_rel.edge_label.src_ref_table;
    let src_table = &session
        .tables
        .get(src_table_name)
        .ok_or_else(|| SqlError::Plan(format!("Table '{src_table_name}' not found")))?
        .table;

    let (_min_depth, max_depth): (u8, u8) = match edge.quantifier {
        PathQuantifier::Range { min, max } => (min, max),
        PathQuantifier::Plus => (1, 255),
        PathQuantifier::Star => (0, 255),
        PathQuantifier::One => (1, 1),
    };

    let g = session.ctx.graph(src_table)?;
    let src_col = g.const_i64(src_id)?;
    let dst_col = g.const_i64(dst_id)?;
    let result_col = g.dijkstra(src_col, Some(dst_col), &stored_rel.rel, weight_col, max_depth)?;
    let result = g.execute(result_col)?;

    // Dijkstra result has _node (I64) and _dist (F64) columns.
    let nrows = checked_nrows(&result)?;

    let _node_col_idx = find_col_idx(&result, "_node")
        .ok_or_else(|| SqlError::Plan("Dijkstra result missing _node column".into()))?;
    let dist_col_idx = find_col_idx(&result, "_dist")
        .ok_or_else(|| SqlError::Plan("Dijkstra result missing _dist column".into()))?;

    // Total cost is the _dist value at the destination row (last row).
    let total_cost: f64 = if nrows > 0 {
        result.get_f64(dist_col_idx, nrows - 1).unwrap_or(0.0)
    } else {
        0.0
    };

    // Get vertex tables for property lookups
    let src_vtable = &session
        .tables
        .get(left_table.as_str())
        .ok_or_else(|| SqlError::Plan(format!("Table '{left_table}' not found")))?
        .table;
    let dst_vtable = &session
        .tables
        .get(right_table.as_str())
        .ok_or_else(|| SqlError::Plan(format!("Table '{right_table}' not found")))?
        .table;

    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    // Project COLUMNS — cheapest path returns one summary row.
    enum CpColKind {
        SrcProp(usize), // column index into src vertex table
        DstProp(usize), // column index into dst vertex table
        PathCost,       // total path cost
    }
    let mut col_names: Vec<String> = Vec::new();
    let mut col_specs: Vec<CpColKind> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();
        let lower = expr.to_lowercase();

        if lower.contains("path_cost") {
            col_names.push(alias.unwrap_or("path_cost").to_string());
            col_specs.push(CpColKind::PathCost);
        } else if let Some(dot_pos) = lower.find('.') {
            let var = lower[..dot_pos].trim();
            let col = lower[dot_pos + 1..].trim();
            let out_name = alias.unwrap_or(col).to_string();

            if var == src_var {
                // Check PROPERTIES visibility for source vertex label
                if let Some(vl) = graph.vertex_labels.values().find(|vl| vl.table_name == *left_table) {
                    check_column_visible(&vl.visibility, col, &vl.label)?;
                }
                let col_idx = find_col_idx(src_vtable, col).ok_or_else(|| {
                    SqlError::Plan(format!("Column '{col}' not found in source table"))
                })?;
                col_names.push(out_name);
                col_specs.push(CpColKind::SrcProp(col_idx));
            } else if var == dst_var {
                // Check PROPERTIES visibility for destination vertex label
                if let Some(vl) = graph.vertex_labels.values().find(|vl| vl.table_name == *right_table) {
                    check_column_visible(&vl.visibility, col, &vl.label)?;
                }
                let col_idx = find_col_idx(dst_vtable, col).ok_or_else(|| {
                    SqlError::Plan(format!("Column '{col}' not found in destination table"))
                })?;
                col_names.push(out_name);
                col_specs.push(CpColKind::DstProp(col_idx));
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in CHEAPEST path COLUMNS. Available: {src_var}, {dst_var}"
                )));
            }
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS: unsupported expression '{expr}' in CHEAPEST path"
            )));
        }
    }

    if col_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Build a single result row via CSV.
    let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    // If path was found (nrows > 0), emit one result row.
    if nrows > 0 {
        for (i, spec) in col_specs.iter().enumerate() {
            if i > 0 {
                csv.push(',');
            }
            match spec {
                CpColKind::SrcProp(col_idx) => {
                    csv.push_str(&get_cell_string(src_vtable, *col_idx, src_id as usize)?);
                }
                CpColKind::DstProp(col_idx) => {
                    csv.push_str(&get_cell_string(dst_vtable, *col_idx, dst_id as usize)?);
                }
                CpColKind::PathCost => {
                    // Ensure the float always has a decimal point so the C
                    // engine reads it as F64 rather than I64.
                    let s = total_cost.to_string();
                    csv.push_str(&s);
                    if !s.contains('.') {
                        csv.push_str(".0");
                    }
                }
            }
        }
        csv.push('\n');
    }

    let result_table = csv_to_table(session, &csv, &col_names)?;
    Ok((result_table, col_names))
}

/// Reconstruct shortest path using BFS over the CSR adjacency structure.
/// Returns the path as a Vec of row indices from dst back to src (reversed).
/// Only returns a path whose hop count is between min_depth and max_depth (inclusive).
fn reconstruct_shortest_path(
    src_id: i64,
    dst_id: i64,
    min_depth: i64,
    max_depth: i64,
    stored_rel: &StoredRel,
    direction: u8,
) -> Result<Vec<i64>, SqlError> {
    use std::collections::{HashMap as HM, HashSet, VecDeque};

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
                let prev = *pred_map.get(&(cur_node, cur_depth)).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "BFS path reconstruction failed: no predecessor for node {cur_node} at depth {cur_depth}"
                    ))
                })?;
                path.push(prev);
                cur_node = prev;
                cur_depth -= 1;
            }
            return Ok(path);
        }
        if depth >= max_depth {
            continue;
        }
        // Use the CSR for neighbor lookup — works correctly with remapped row indices
        let neighbors = if direction == 2 {
            // Undirected: merge forward and reverse neighbors
            let fwd = stored_rel.rel.neighbors(node, 0);
            let rev = stored_rel.rel.neighbors(node, 1);
            let mut merged = Vec::with_capacity(fwd.len() + rev.len());
            merged.extend_from_slice(fwd);
            merged.extend_from_slice(rev);
            merged
        } else {
            stored_rel.rel.neighbors(node, direction).to_vec()
        };
        for &next in &neighbors {
            let next_depth = depth + 1;
            if visited.insert((next, next_depth)) {
                pred_map.insert((next, next_depth), node);
                frontier.push_back((next, next_depth));
            }
        }
    }

    Ok(Vec::new()) // no path found → empty result
}

/// Extract a node's row index from a WHERE filter.
/// Uses the rich filter evaluator and requires exactly one matching row.
fn extract_node_id(
    node: &NodePattern,
    table_name: &str,
    session: &Session,
) -> Result<i64, SqlError> {
    let filter = node.filter.as_deref().ok_or_else(|| {
        SqlError::Plan(
            "SHORTEST_PATH requires WHERE filters on both source and destination nodes".into(),
        )
    })?;
    let stored = session.tables.get(table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{table_name}' not found"))
    })?;
    let matches = resolve_filtered_node_ids(filter, node, stored)?;
    match matches.len() {
        0 => Err(SqlError::Plan(format!("No matching row for filter: {filter}"))),
        1 => Ok(matches[0]),
        n => Err(SqlError::Plan(format!(
            "SHORTEST_PATH filter must match exactly one node, but '{filter}' matched {n} rows in '{table_name}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Quote-aware argument parsing helpers
// ---------------------------------------------------------------------------

/// Split a string on commas, but respect single-quoted SQL literals so that
/// commas inside quotes are not treated as separators.
fn split_respecting_quotes(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if in_quote {
                // Check for escaped quote ('')
                if chars.peek() == Some(&'\'') {
                    current.push('\'');
                    current.push('\'');
                    chars.next();
                } else {
                    current.push(ch);
                    in_quote = false;
                }
            } else {
                current.push(ch);
                in_quote = true;
            }
        } else if ch == ',' && !in_quote {
            args.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() || !args.is_empty() {
        args.push(trimmed);
    }
    args
}

/// Unescape a SQL string literal: strip surrounding single quotes and replace
/// doubled single quotes ('') with a single quote (').
fn unescape_sql_string(s: &str) -> String {
    let trimmed = s.trim();
    let inner = if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        &trimmed[1..trimmed.len() - 1]
    } else if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return trimmed.to_string();
    };
    if trimmed.starts_with('\'') {
        inner.replace("''", "'")
    } else {
        inner.replace("\"\"", "\"")
    }
}

// ---------------------------------------------------------------------------
// Graph algorithm planner
// ---------------------------------------------------------------------------

/// Known graph algorithm function names.
const ALGO_FUNCTIONS: &[&str] = &["pagerank", "component", "connected_component", "community", "louvain", "shortest_distance", "dijkstra", "clustering_coefficient", "local_clustering_coeff", "clustering_coeff"];

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
    let args: Vec<String> = split_respecting_quotes(&inner);
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
    let vertex_label = if let Some(labels) = &node.labels {
        let label = labels.first().ok_or_else(|| {
            SqlError::Plan("Empty label list in node pattern".into())
        })?;
        graph.vertex_labels.get(label.as_str()).ok_or_else(|| {
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
                if args[0].to_lowercase() != graph.name.to_lowercase() {
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
                if args[0].to_lowercase() != graph.name.to_lowercase() {
                    return Err(SqlError::Plan(format!(
                        "Algorithm argument '{}' does not match graph name '{}'. \
                         Expected: {}({}, {})",
                        args[0], graph.name, func_name.to_uppercase(), graph.name, node_var,
                    )));
                }
                if args[1].to_lowercase() != node_var {
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
                    "clustering_coefficient" | "local_clustering_coeff" | "clustering_coeff" => "clustering_coeff",
                    other => other,
                };
                let cur_canonical = match func_name.as_str() {
                    "component" | "connected_component" => "connected_component",
                    "community" | "louvain" => "louvain",
                    "shortest_distance" | "dijkstra" => "dijkstra",
                    "clustering_coefficient" | "local_clustering_coeff" | "clustering_coeff" => "clustering_coeff",
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
                "clustering_coefficient" | "local_clustering_coeff" | "clustering_coeff" => "_clustering_coeff",
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
                // Check PROPERTIES visibility for this vertex label
                check_column_visible(&vertex_label.visibility, &col, &vertex_label.label)?;
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
        "clustering_coefficient" | "local_clustering_coeff" | "clustering_coeff" => {
            g.clustering_coeff(&stored_rel.rel)?
        }
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

    // Dijkstra's C kernel allocates dist/visited/depth arrays sized by
    // rel->fwd.n_nodes (source domain).  On heterogeneous edges where
    // src and dst reference different tables, the CSR targets live in
    // the destination domain which can exceed the source domain size
    // and cause out-of-bounds access.  When both sides reference the
    // same table (even with different key columns), n_src == n_dst so
    // the kernel is safe.
    if stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table {
        return Err(SqlError::Plan(
            "SHORTEST_DISTANCE is not supported on edges whose source and destination \
             reference different vertex tables (heterogeneous endpoints)."
                .into(),
        ));
    }

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

    // Remap user-facing IDs to internal row indices via natural key maps
    let dst_table_name = &stored_rel.edge_label.dst_ref_table;
    let src_vl = graph.vertex_labels.values()
        .find(|vl| vl.table_name == *src_table_name && vl.key_column == stored_rel.edge_label.src_ref_col)
        .ok_or_else(|| {
            SqlError::Plan(format!("No vertex label found for table '{src_table_name}' with key column '{}'",
                stored_rel.edge_label.src_ref_col))
        })?;
    let dst_vl = graph.vertex_labels.values()
        .find(|vl| vl.table_name == *dst_table_name && vl.key_column == stored_rel.edge_label.dst_ref_col)
        .ok_or_else(|| {
            SqlError::Plan(format!("No vertex label found for table '{dst_table_name}' with key column '{}'",
                stored_rel.edge_label.dst_ref_col))
        })?;
    // Determine key type from each vertex label and construct appropriate KeyValue.
    // Unescape SQL string literals (handles '' escaping and surrounding quotes).
    let src_arg = unescape_sql_string(&args[1]);
    let dst_arg = unescape_sql_string(&args[2]);
    let src_uses_string_keys = src_vl.row_to_user.first()
        .map(|k| matches!(k, KeyValue::Str(_)))
        .unwrap_or(false);
    let dst_uses_string_keys = dst_vl.row_to_user.first()
        .map(|k| matches!(k, KeyValue::Str(_)))
        .unwrap_or(false);
    let src_key = if src_uses_string_keys {
        KeyValue::Str(src_arg.to_string())
    } else {
        let src_id: i64 = src_arg.parse().map_err(|_| {
            SqlError::Plan(format!("Invalid source node ID: '{}' (expected integer)", src_arg))
        })?;
        KeyValue::Int(src_id)
    };
    let dst_key = if dst_uses_string_keys {
        KeyValue::Str(dst_arg.to_string())
    } else {
        let dst_id: i64 = dst_arg.parse().map_err(|_| {
            SqlError::Plan(format!("Invalid destination node ID: '{}' (expected integer)", dst_arg))
        })?;
        KeyValue::Int(dst_id)
    };
    let internal_src = *src_vl.user_to_row.get(&src_key).ok_or_else(|| {
        SqlError::Plan(format!("Source node '{}' not found in vertex table '{src_table_name}'", args[1]))
    })? as i64;
    let internal_dst = *dst_vl.user_to_row.get(&dst_key).ok_or_else(|| {
        SqlError::Plan(format!("Destination node '{}' not found in vertex table '{dst_table_name}'", args[2]))
    })? as i64;

    let g = session.ctx.graph(src_table)?;
    let src = g.const_i64(internal_src)?;
    let dst = g.const_i64(internal_dst)?;
    let result_col = g.dijkstra(src, Some(dst), &stored_rel.rel, weight_col, 255)?;

    let result = g.execute(result_col)?;
    Ok(result)
}
