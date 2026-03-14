// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use crate::{Column, Graph, Rel, Table};
use super::pgq_parser::{CreatePropertyGraph as ParsedCPG, PgqStatement};
use super::{ExecResult, Session, SqlError};

// ---------------------------------------------------------------------------
// Property graph catalog types
// ---------------------------------------------------------------------------

/// A vertex label mapping: label name -> session table name.
pub(crate) struct VertexLabel {
    pub table_name: String,
    pub label: String,
}

/// An edge label mapping: label name -> edge table with source/dest references.
pub(crate) struct EdgeLabel {
    pub table_name: String,
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
    pub variable: Option<String>,
    pub label: Option<String>,
    pub filter: Option<String>, // raw SQL predicate text
}

/// An edge pattern: -[var:Label]-> with optional quantifier
#[derive(Debug, Clone)]
pub(crate) struct EdgePattern {
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
            Err(SqlError::Plan(
                "ANY SHORTEST path queries are not yet supported. \
                 Use single-hop (a)-[e]->(b) patterns."
                    .into(),
            ))
        }
        (2, 1, PathMode::Walk) if is_var_length => {
            Err(SqlError::Plan(
                "Variable-length path queries are not yet supported. \
                 Use single-hop (a)-[e]->(b) patterns."
                    .into(),
            ))
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
        edge,
        &src_stored.table,
        &dst_stored.table,
        &stored_rel.edge_label,
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
    _edge: &EdgePattern,
    src_table: &Table,
    dst_table: &Table,
    _edge_label: &EdgeLabel,
) -> Result<(Table, Vec<String>), SqlError> {
    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    // Collect which columns we need from src and dst tables
    let mut src_cols_needed: Vec<(String, String)> = Vec::new(); // (col_name, output_name)
    let mut dst_cols_needed: Vec<(String, String)> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            if var == src_var {
                src_cols_needed.push((col, out_name));
            } else if var == dst_var {
                dst_cols_needed.push((col, out_name));
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

    if src_cols_needed.is_empty() && dst_cols_needed.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

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
            }
        }
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
    let tmp_path = std::env::temp_dir().join(format!("__pgq_{}.csv", std::process::id()));
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
    for i in 0..ncols {
        if table.col_name_str(i).to_lowercase() == name {
            return Some(i);
        }
    }
    None
}

/// Get a cell value as a string for CSV output.
fn get_cell_string(table: &Table, col: usize, row: usize) -> String {
    // Try string first (SYM columns)
    if let Some(s) = table.get_str(col, row) {
        return s;
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
