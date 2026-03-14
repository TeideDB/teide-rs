// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{Column, Graph, Rel, Table};

/// Monotonic counter for unique temp file names (avoids race conditions).
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
use super::pgq_parser::{CreatePropertyGraph as ParsedCPG, PgqStatement};
use super::{ExecResult, Session, SqlError};

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

        let n_src = src_stored.table.nrows();
        let n_dst = dst_stored.table.nrows();

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

    let pattern = &match_clause.patterns[0];

    // Check for variable-length edge first
    let is_var_length = pattern.edges.len() == 1
        && !matches!(pattern.edges[0].quantifier, PathQuantifier::One);

    match (pattern.nodes.len(), pattern.edges.len(), match_clause.mode) {
        (2, 1, PathMode::AnyShortest) => {
            plan_shortest_path(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) if is_var_length => {
            plan_var_length(session, graph, pattern, &expr.columns)
        }
        (2, 1, PathMode::Walk) => {
            plan_single_hop(session, graph, pattern, &expr.columns)
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

    // Resolve source and destination vertex tables
    let src_label = resolve_node_label(src_node, &stored_rel.edge_label.src_ref_table, graph)?;
    let dst_label = resolve_node_label(dst_node, &stored_rel.edge_label.dst_ref_table, graph)?;

    let src_stored = session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?;
    let dst_stored = session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?;

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // Build graph on the source table
    let g = session.ctx.graph(&src_stored.table)?;

    // Scan the source reference column (the join key, e.g. "id")
    let src_ref_col = &stored_rel.edge_label.src_ref_col;
    let src_ids = g.scan(src_ref_col)?;

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
    // Strip the variable prefix (e.g., "a.name" -> "name")
    let clean = if let Some(var) = variable {
        filter_text.replace(&format!("{var} ."), "").replace(&format!("{var}."), "")
    } else {
        filter_text.to_string()
    };

    // Parse: col = 'value' or col = number
    let parts: Vec<&str> = clean.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(SqlError::Plan(format!(
            "Unsupported node filter syntax: {filter_text}. Only 'col = value' is supported."
        )));
    }
    let col_name = parts[0].trim().to_lowercase();
    let value = parts[1].trim();

    let scan_col = g.scan(&col_name)?;

    let const_col = if value.starts_with('\'') && value.ends_with('\'') {
        let s = &value[1..value.len() - 1];
        g.const_str(s)?
    } else if let Ok(n) = value.parse::<i64>() {
        g.const_i64(n)?
    } else if let Ok(f) = value.parse::<f64>() {
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
    let nrows = expand_result.nrows() as usize;

    // Find _src and _dst column indices in expand result
    let src_idx_col = find_col_idx(expand_result, "_src")
        .ok_or_else(|| SqlError::Plan("expand result missing _src column".into()))?;
    let dst_idx_col = find_col_idx(expand_result, "_dst")
        .ok_or_else(|| SqlError::Plan("expand result missing _dst column".into()))?;

    let mut src_indices = Vec::with_capacity(nrows);
    let mut dst_indices = Vec::with_capacity(nrows);
    for row in 0..nrows {
        src_indices.push(expand_result.get_i64(src_idx_col, row).unwrap_or(0) as usize);
        dst_indices.push(expand_result.get_i64(dst_idx_col, row).unwrap_or(0) as usize);
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
            csv.push_str(&get_cell_string(table, spec.table_col_idx, table_row));
        }
        csv.push('\n');
    }

    // Write to temp file and read back as table
    let tmp_path = std::env::temp_dir().join(format!(
        "__pgq_{}_{}.csv",
        std::process::id(),
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp_path, csv.as_bytes())
        .map_err(|e| SqlError::Plan(format!("Failed to write temp file: {e}")))?;
    let path_str = tmp_path.to_str().ok_or_else(|| {
        SqlError::Plan("Temp file path not valid UTF-8".into())
    })?;
    let result = session.ctx.read_csv(path_str)?;
    let _ = std::fs::remove_file(&tmp_path);
    let result = result.with_column_names(&col_names)?;

    Ok((result, col_names))
}

/// Find a column index by name in a table.
fn find_col_idx(table: &Table, name: &str) -> Option<usize> {
    let ncols = table.ncols() as usize;
    (0..ncols).find(|&i| table.col_name_str(i).to_lowercase() == name)
}

/// Get a cell value as a string for CSV output.
/// String values are quoted and escaped for CSV safety.
fn get_cell_string(table: &Table, col: usize, row: usize) -> String {
    // Try string first (SYM columns)
    if let Some(s) = table.get_str(col, row) {
        return csv_quote(&s);
    }
    // Try i64
    if let Some(v) = table.get_i64(col, row) {
        return v.to_string();
    }
    // Try f64
    if let Some(v) = table.get_f64(col, row) {
        return v.to_string();
    }
    String::new()
}

/// Quote a string value for CSV: wrap in double quotes if it contains
/// commas, newlines, or quotes. Escape embedded double quotes by doubling.
fn csv_quote(s: &str) -> String {
    if s.contains(',') || s.contains('\n') || s.contains('"') || s.contains('\r') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
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

    let src_label = resolve_node_label(src_node, &stored_rel.edge_label.src_ref_table, graph)?;
    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;

    let dst_label = resolve_node_label(dst_node, &stored_rel.edge_label.dst_ref_table, graph)?;
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
    let src_ref_col = &stored_rel.edge_label.src_ref_col;
    let src_ids = g.scan(src_ref_col)?;

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

    let nrows = var_result.nrows() as usize;

    let start_idx_col = find_col_idx(var_result, "_start")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _start column".into()))?;
    let end_idx_col = find_col_idx(var_result, "_end")
        .ok_or_else(|| SqlError::Plan("var_expand result missing _end column".into()))?;
    let depth_idx_col = find_col_idx(var_result, "_depth");

    let mut start_indices = Vec::with_capacity(nrows);
    let mut end_indices = Vec::with_capacity(nrows);
    for row in 0..nrows {
        start_indices.push(var_result.get_i64(start_idx_col, row).unwrap_or(0) as usize);
        end_indices.push(var_result.get_i64(end_idx_col, row).unwrap_or(0) as usize);
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
                    csv.push_str(&get_cell_string(src_table, spec.table_col_idx, start_indices[row]));
                }
                VarColKind::Dst => {
                    csv.push_str(&get_cell_string(dst_table, spec.table_col_idx, end_indices[row]));
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

    let tmp_path = std::env::temp_dir().join(format!(
        "__pgq_vl_{}_{}.csv",
        std::process::id(),
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp_path, csv.as_bytes())
        .map_err(|e| SqlError::Plan(format!("Failed to write temp file: {e}")))?;
    let path_str = tmp_path.to_str().ok_or_else(|| {
        SqlError::Plan("Temp file path not valid UTF-8".into())
    })?;
    let result = session.ctx.read_csv(path_str)?;
    let _ = std::fs::remove_file(&tmp_path);
    let result = result.with_column_names(&col_names)?;

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

    let src_id = extract_node_id(src_node, &stored_rel.edge_label.src_ref_table, session)?;
    let dst_id = extract_node_id(dst_node, &stored_rel.edge_label.dst_ref_table, session)?;

    let max_depth: u8 = match edge.quantifier {
        PathQuantifier::Range { max, .. } => max,
        _ => 255,
    };

    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // BFS over the edge table to find and reconstruct the shortest path
    let mut path = reconstruct_shortest_path(
        session, src_id, dst_id, max_depth as i64, stored_rel, direction,
    )?;
    path.reverse(); // path is built backwards, reverse to get src -> dst order

    // Build result as CSV with _node and _depth columns
    let mut col_names: Vec<String> = Vec::new();
    let mut col_indices: Vec<usize> = Vec::new(); // 0 = _node col, 1 = _depth col

    for entry in columns {
        let lower = entry.expr.to_lowercase();
        let alias = entry.alias.as_deref();

        if lower.contains("path_length") {
            col_names.push(alias.unwrap_or("path_length").to_string());
            col_indices.push(1); // depth
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
                _ => {}
            }
        }
        csv.push('\n');
    }

    let tmp_path = std::env::temp_dir().join(format!(
        "__pgq_sp_{}_{}.csv",
        std::process::id(),
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp_path, csv.as_bytes())
        .map_err(|e| SqlError::Plan(format!("Failed to write temp file: {e}")))?;
    let path_str = tmp_path.to_str().ok_or_else(|| {
        SqlError::Plan("Temp file path not valid UTF-8".into())
    })?;
    let result = session.ctx.read_csv(path_str)?;
    let _ = std::fs::remove_file(&tmp_path);
    let result = result.with_column_names(&col_names)?;

    Ok((result, col_names))
}

/// Reconstruct shortest path using BFS over the edge table.
/// Returns the path as a Vec of node IDs from dst back to src (reversed).
fn reconstruct_shortest_path(
    session: &Session,
    src_id: i64,
    dst_id: i64,
    max_depth: i64,
    stored_rel: &StoredRel,
    direction: u8,
) -> Result<Vec<i64>, SqlError> {
    use std::collections::{HashMap as HM, VecDeque};

    // BFS using single-hop expand, tracking predecessors
    let mut visited: HM<i64, i64> = HM::new(); // node -> predecessor
    visited.insert(src_id, -1);
    let mut frontier = VecDeque::new();
    frontier.push_back((src_id, 0i64));

    // Get all edges for adjacency lookup
    let edge_table_name = &stored_rel.edge_label.table_name;
    let edge_stored = session.tables.get(edge_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge table '{}' not found", edge_table_name))
    })?;

    let src_col_name = &stored_rel.edge_label.src_col;
    let dst_col_name = &stored_rel.edge_label.dst_col;
    let src_col_idx = edge_stored.columns.iter().position(|c| c == src_col_name)
        .ok_or_else(|| SqlError::Plan(format!("Column '{src_col_name}' not found in edge table")))?;
    let dst_col_idx = edge_stored.columns.iter().position(|c| c == dst_col_name)
        .ok_or_else(|| SqlError::Plan(format!("Column '{dst_col_name}' not found in edge table")))?;

    // Build adjacency list from edge table
    let n_edges = edge_stored.table.nrows() as usize;
    let mut adj: HM<i64, Vec<i64>> = HM::new();
    for row in 0..n_edges {
        let s = edge_stored.table.get_i64(src_col_idx, row).unwrap_or(-1);
        let d = edge_stored.table.get_i64(dst_col_idx, row).unwrap_or(-1);
        match direction {
            0 => { adj.entry(s).or_default().push(d); } // forward
            1 => { adj.entry(d).or_default().push(s); } // reverse
            _ => { // undirected
                adj.entry(s).or_default().push(d);
                adj.entry(d).or_default().push(s);
            }
        }
    }

    // BFS
    while let Some((node, depth)) = frontier.pop_front() {
        if node == dst_id {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = dst_id;
            while cur != -1 {
                path.push(cur);
                cur = *visited.get(&cur).unwrap_or(&-1);
            }
            return Ok(path);
        }
        if depth >= max_depth {
            continue;
        }
        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors {
                if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(next) {
                    e.insert(node);
                    frontier.push_back((next, depth + 1));
                }
            }
        }
    }

    Err(SqlError::Plan(format!(
        "No path found from node {src_id} to node {dst_id}"
    )))
}

/// Extract a specific node ID from a WHERE filter.
/// Looks for patterns like "a.id = 42" or "a.name = 'Alice'" and resolves to row index.
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

    let var = node.variable.as_deref().unwrap_or("");
    let clean = if !var.is_empty() {
        filter.replace(&format!("{var} ."), "").replace(&format!("{var}."), "")
    } else {
        filter.to_string()
    };

    let parts: Vec<&str> = clean.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(SqlError::Plan(format!(
            "Cannot extract node ID from filter: {filter}"
        )));
    }
    let col_name = parts[0].trim().to_lowercase();
    let value = parts[1].trim();

    // If filtering by ID directly
    if col_name == "id" {
        if let Ok(id) = value.parse::<i64>() {
            return Ok(id);
        }
    }

    // Otherwise, scan the table to find the matching row index
    let stored = session.tables.get(table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{table_name}' not found"))
    })?;

    let nrows = stored.table.nrows() as usize;
    let col_idx = stored.columns.iter().position(|c| c == &col_name).ok_or_else(|| {
        SqlError::Plan(format!("Column '{col_name}' not found in '{table_name}'"))
    })?;

    let str_val = if value.starts_with('\'') && value.ends_with('\'') {
        Some(&value[1..value.len() - 1])
    } else {
        None
    };

    // Find the ID column to return the actual node ID, not the row index.
    let id_col_idx = stored.columns.iter().position(|c| c == "id");

    for row in 0..nrows {
        let matched = if let Some(sv) = str_val {
            stored.table.get_str(col_idx, row).as_deref() == Some(sv)
        } else if let Ok(iv) = value.parse::<i64>() {
            stored.table.get_i64(col_idx, row) == Some(iv)
        } else {
            false
        };

        if matched {
            // Return the actual ID value from the id column, not the row index.
            if let Some(id_idx) = id_col_idx {
                if let Some(id_val) = stored.table.get_i64(id_idx, row) {
                    return Ok(id_val);
                }
            }
            // Fall back to row index only if there's no id column.
            return Ok(row as i64);
        }
    }

    Err(SqlError::Plan(format!(
        "No matching row for filter: {filter}"
    )))
}
