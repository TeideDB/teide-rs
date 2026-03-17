//   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
//   All rights reserved.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

// teide-db: SQL parser and planner for the Teide columnar table engine.
//
// Translates SQL queries into Teide execution graphs and runs them against
// CSV files.

pub mod expr;
pub mod pgq;
pub mod pgq_parser;
pub mod planner;

use crate::{Context, HnswIndex, Table};
use std::collections::HashMap;

/// Errors produced by the SQL layer.
#[derive(Debug)]
pub enum SqlError {
    /// SQL syntax error from the parser.
    Parse(String),
    /// Planning error (unknown column, unsupported feature, etc.).
    Plan(String),
    /// Teide engine execution error.
    Engine(crate::Error),
}

impl std::fmt::Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SqlError::Parse(msg) => write!(f, "SQL parse error: {msg}"),
            SqlError::Plan(msg) => write!(f, "SQL planning error: {msg}"),
            SqlError::Engine(err) => write!(f, "Engine error: {err}"),
        }
    }
}

impl std::error::Error for SqlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SqlError::Engine(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::Error> for SqlError {
    fn from(err: crate::Error) -> Self {
        SqlError::Engine(err)
    }
}

/// Result of executing a SQL query.
pub struct SqlResult {
    /// The result table.
    pub table: Table,
    /// Column names/aliases as they appear in the SELECT list.
    pub columns: Vec<String>,
    /// Embedding column dimensions inherited from the source table (if any).
    pub embedding_dims: HashMap<String, i32>,
    /// True row count.  For tables with embedding columns, `table.nrows()`
    /// returns N*D (the flat F32 array length).  This field holds the corrected
    /// count so that consumers do not need to be aware of the embedding layout.
    pub nrows: usize,
}

/// Result of executing a SQL statement via a Session.
pub enum ExecResult {
    /// A SELECT query that produced a result set.
    Query(SqlResult),
    /// A DDL statement (CREATE TABLE, DROP TABLE) with a status message.
    Ddl(String),
}

/// A stored table in the session registry.
pub(crate) struct StoredTable {
    pub table: Table,
    pub columns: Vec<String>,
    /// Per-column embedding dimensions (column name → dim).
    /// Populated by `Session::register_embedding_column`; used by the SQL
    /// planner to validate query-vector lengths in COSINE_SIMILARITY /
    /// EUCLIDEAN_DISTANCE calls.
    pub embedding_dims: HashMap<String, i32>,
}

impl StoredTable {
    /// Return the logical row count, correcting for embedding columns whose
    /// flat F32 arrays make `table.nrows()` return N*D instead of N.
    pub(crate) fn logical_nrows(&self) -> i64 {
        if self.embedding_dims.is_empty() {
            return self.table.nrows();
        }
        let raw = self.table.nrows();
        // Find the first non-embedding column and use its length.
        for ci in 0..self.table.ncols() {
            let name = self.table.col_name_str(ci as usize).to_lowercase();
            if !self.embedding_dims.contains_key(&name) {
                if let Some(col) = self.table.get_col_idx(ci) {
                    let col_type = unsafe { crate::raw::td_type(col) };
                    if col_type > 0
                        && !crate::ffi::td_is_parted(col_type)
                        && col_type != crate::ffi::TD_MAPCOMMON
                    {
                        return unsafe { crate::raw::td_len(col) };
                    }
                }
                return raw;
            }
        }
        // All columns are embeddings — derive N from the first column.
        if self.table.ncols() > 0 {
            let name = self.table.col_name_str(0).to_lowercase();
            if let Some(&d) = self.embedding_dims.get(&name) {
                if d > 0 {
                    return raw / d as i64;
                }
            }
        }
        raw
    }
}

impl Clone for StoredTable {
    fn clone(&self) -> Self {
        StoredTable {
            table: self.table.clone_ref(),
            columns: self.columns.clone(),
            embedding_dims: self.embedding_dims.clone(),
        }
    }
}

/// Metadata for a stored vector index.
#[allow(dead_code)]
pub(crate) struct VectorIndexInfo {
    pub table_name: String,
    pub column_name: String,
    pub index: HnswIndex,
    pub m: i32,
    pub ef_construction: i32,
}

/// A stateful SQL session that maintains a table registry across queries.
///
// pub(crate) access allows planner to manage table registry directly.
// This is intentional for simplicity; encapsulation via methods would
// add complexity without safety benefit since planner is the only consumer.
pub struct Session {
    pub(crate) tables: HashMap<String, StoredTable>,
    pub(crate) graphs: HashMap<String, pgq::PropertyGraph>,
    pub(crate) vector_indexes: HashMap<String, VectorIndexInfo>,
    pub(crate) ctx: Context,
}

impl Session {
    /// Create a new session, initializing the Teide engine.
    pub fn new() -> Result<Self, SqlError> {
        let ctx = Context::new()?;
        Ok(Session {
            ctx,
            tables: HashMap::new(),
            graphs: HashMap::new(),
            vector_indexes: HashMap::new(),
        })
    }

    /// Register an embedding column's dimension for a stored table.
    /// This enables the SQL planner to validate query vector lengths
    /// in COSINE_SIMILARITY / EUCLIDEAN_DISTANCE calls.
    ///
    /// Returns an error if the column does not exist or is not of type `TD_F32`.
    pub fn register_embedding_column(
        &mut self,
        table_name: &str,
        column_name: &str,
        dim: i32,
    ) -> Result<(), SqlError> {
        self.register_embedding_dim_inner(table_name, column_name, dim, true)
    }

    /// Register an embedding dimension for a column without checking its type.
    ///
    /// Unlike [`register_embedding_column`], this method does not verify that
    /// the column is TD_F32.  It is intended for internal use (e.g., when the
    /// planner propagates embedding metadata through query results whose
    /// intermediate columns may not be typed TD_F32).
    ///
    /// Prefer [`register_embedding_column`] when registering user-facing
    /// embedding columns, as it enforces the TD_F32 invariant.
    #[doc(hidden)]
    pub fn register_embedding_dim(
        &mut self,
        table_name: &str,
        column_name: &str,
        dim: i32,
    ) -> Result<(), SqlError> {
        self.register_embedding_dim_inner(table_name, column_name, dim, false)
    }

    fn register_embedding_dim_inner(
        &mut self,
        table_name: &str,
        column_name: &str,
        dim: i32,
        check_type: bool,
    ) -> Result<(), SqlError> {
        if dim <= 0 {
            return Err(SqlError::Plan(format!(
                "Embedding dimension must be positive, got {dim}"
            )));
        }
        let table_key = table_name.to_lowercase();
        let col_key = column_name.to_lowercase();
        let stored = self.tables.get_mut(&table_key).ok_or_else(|| {
            SqlError::Plan(format!("Table '{table_name}' not found"))
        })?;
        let col_idx = stored
            .columns
            .iter()
            .position(|c| c.to_lowercase() == col_key)
            .ok_or_else(|| {
                SqlError::Plan(format!(
                    "Column '{column_name}' not found in table '{table_name}'"
                ))
            })?;
        if check_type {
            // Check the raw type tag (not the normalized one from col_type(),
            // which maps parted columns to their base type).  Parted F32 columns
            // have a different physical layout and are not valid embedding storage.
            let raw_type = stored.table.get_col_idx(col_idx as i64)
                .map(|p| unsafe { crate::ffi::td_type(p) })
                .unwrap_or(0);
            if raw_type != crate::ffi::TD_F32 {
                return Err(SqlError::Plan(format!(
                    "Column '{column_name}' in table '{table_name}' has type {raw_type}, \
                     expected TD_F32={} for embedding columns \
                     (use Table::create_embedding_column to create proper embedding columns)",
                    crate::ffi::TD_F32,
                )));
            }
        }
        stored.embedding_dims.insert(col_key, dim);
        Ok(())
    }

    /// Add a TD_F32 embedding column to a stored table and register its dimension.
    ///
    /// Creates a proper embedding column via `Table::create_embedding_column`,
    /// adds it to the table, and registers the dimension for SQL planner
    /// validation.  The `data` slice must contain exactly `nrows * dim`
    /// elements where `nrows` matches the current row count of the table.
    pub fn add_embedding_column(
        &mut self,
        table_name: &str,
        column_name: &str,
        dim: i32,
        data: &[f32],
    ) -> Result<(), SqlError> {
        let table_key = table_name.to_lowercase();
        let col_key = column_name.to_lowercase();
        let stored = self.tables.get_mut(&table_key).ok_or_else(|| {
            SqlError::Plan(format!("Table '{table_name}' not found"))
        })?;
        if stored.columns.iter().any(|c| c == &col_key) {
            return Err(SqlError::Plan(format!(
                "Column '{column_name}' already exists in table '{table_name}'"
            )));
        }
        let nrows = stored.logical_nrows();
        let emb_col = Table::create_embedding_column(&self.ctx, nrows, dim, data)
            .map_err(|e| SqlError::Plan(format!(
                "Failed to create embedding column '{column_name}': {e}"
            )))?;
        let col_sym = match crate::sym_intern(&col_key) {
            Ok(s) => s,
            Err(e) => {
                // Release the C object to avoid a leak on failure.
                unsafe { crate::ffi::td_release(emb_col) };
                return Err(SqlError::Plan(format!(
                    "Failed to intern symbol for column '{column_name}': {e}"
                )));
            }
        };
        // Use the safe add_column_raw wrapper which handles COW pointer
        // semantics correctly (updating self.raw in place without
        // double-releasing the old handle via Drop).
        // SAFETY: emb_col is a valid TD_F32 column from create_embedding_column.
        if let Err(e) = unsafe { stored.table.add_column_raw(col_sym, emb_col) } {
            unsafe { crate::ffi::td_release(emb_col) };
            return Err(SqlError::Plan(format!(
                "Failed to add embedding column '{column_name}' to table '{table_name}': {e}"
            )));
        }
        // add_column_raw does NOT release the caller's reference to emb_col;
        // td_table_add_col retained it internally, so release ours now.
        unsafe { crate::ffi::td_release(emb_col) };
        stored.columns.push(col_key.clone());
        stored.embedding_dims.insert(col_key, dim);
        Ok(())
    }

    /// Execute a SQL statement, which may be a SELECT, CREATE TABLE AS, DROP TABLE,
    /// INSERT INTO, UPDATE, or DELETE.
    pub fn execute(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
        // Check for PGQ statements first (sqlparser has no SQL/PGQ support)
        if let Some(pgq_stmt) = pgq_parser::try_parse_pgq(sql)? {
            return pgq::execute_pgq(self, pgq_stmt);
        }

        // Check for GRAPH_TABLE in FROM clause and extract/execute
        if contains_graph_table_keyword(sql) {
            return self.execute_with_graph_table(sql);
        }

        planner::session_execute(self, sql)
    }

    /// Execute SQL containing GRAPH_TABLE expressions.
    /// Extracts GRAPH_TABLE, executes the graph query, stores the result
    /// as a temporary table, rewrites the SQL, and runs the modified SQL.
    fn execute_with_graph_table(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
        let (rewritten_sql, graph_exprs) = pgq_parser::extract_graph_tables(sql)?;

        // Save any user tables that would be shadowed by temp names, so we can restore them.
        let mut saved_tables: Vec<(String, Option<StoredTable>)> = Vec::new();

        // Execute each GRAPH_TABLE and store results as temp tables
        let graph_result = (|| {
            for (temp_name, expr) in &graph_exprs {
                let (table, columns) = pgq::plan_graph_table(self, expr)?;
                let previous = self.tables.insert(
                    temp_name.clone(),
                    StoredTable { table, columns, embedding_dims: HashMap::new() },
                );
                saved_tables.push((temp_name.clone(), previous));
            }

            // Run the rewritten SQL (which references temp tables)
            planner::session_execute(self, &rewritten_sql)
        })();

        // Restore previous table entries or remove temp tables
        for (name, previous) in saved_tables {
            if let Some(prev) = previous {
                self.tables.insert(name, prev);
            } else {
                self.tables.remove(&name);
            }
        }

        graph_result
    }

    /// Execute a multi-statement SQL script (statements separated by `;`).
    /// Returns the result of the last statement.
    ///
    /// Splits on `;` while respecting single-quoted string literals,
    /// so that PGQ statements (which sqlparser cannot parse) are handled.
    pub fn execute_script(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
        let stmts = split_sql_statements(sql);
        if stmts.is_empty() {
            return Err(SqlError::Plan("Empty script".into()));
        }
        let mut last = None;
        for stmt in &stmts {
            last = Some(self.execute(stmt)?);
        }
        last.ok_or_else(|| SqlError::Plan("Empty script".into()))
    }

    /// Execute a SQL script from a file path.
    pub fn execute_script_file(&mut self, path: &std::path::Path) -> Result<ExecResult, SqlError> {
        let sql = std::fs::read_to_string(path)
            .map_err(|e| SqlError::Plan(format!("Failed to read {}: {e}", path.display())))?;
        self.execute_script(&sql)
    }

    /// Rebuild CSR indexes for any property graphs that reference the given table.
    /// Called after INSERT, DELETE, UPDATE, or CREATE OR REPLACE to keep cached
    /// CSR indexes consistent. Returns an error if validation fails (e.g. edge
    /// endpoints out of range after a vertex table change), which should cause
    /// the DML operation to fail rather than silently corrupting graph data.
    pub(crate) fn invalidate_graphs_for_table(
        &mut self,
        table_name: &str,
    ) -> Result<(), SqlError> {
        let affected: Vec<String> = self
            .graphs
            .iter()
            .filter(|(_, graph)| {
                graph
                    .vertex_labels
                    .values()
                    .any(|vl| vl.table_name == table_name)
                    || graph.edge_labels.values().any(|sr| {
                        sr.edge_label.table_name == table_name
                            || sr.edge_label.src_ref_table == table_name
                            || sr.edge_label.dst_ref_table == table_name
                    })
            })
            .map(|(name, _)| name.clone())
            .collect();

        // Save old vertex key maps so we can roll back if edge rebuild fails
        let mut saved_key_maps: Vec<(String, String, HashMap<pgq::KeyValue, usize>, Vec<pgq::KeyValue>)> = Vec::new();
        for graph_name in &affected {
            let graph = self.graphs.get(graph_name).unwrap();
            for (label, vl) in &graph.vertex_labels {
                if vl.table_name == table_name {
                    saved_key_maps.push((
                        graph_name.clone(),
                        label.clone(),
                        vl.user_to_row.clone(),
                        vl.row_to_user.clone(),
                    ));
                }
            }
        }

        // Rebuild vertex key maps for any affected vertex tables.
        // On any error, roll back all key maps already rebuilt.
        if let Err(e) = self.rebuild_vertex_key_maps_for_table(&affected, table_name) {
            self.restore_vertex_key_maps(&saved_key_maps);
            return Err(e);
        }

        // Collect all rebuilt edges first, only apply if ALL graphs validate.
        // Any error in this phase must roll back vertex key maps.
        let rebuilt = self.rebuild_edges_for_graphs(&affected);
        let rebuilt = match rebuilt {
            Ok(r) => r,
            Err(e) => {
                self.restore_vertex_key_maps(&saved_key_maps);
                return Err(e);
            }
        };

        // All graphs validated successfully — apply all updates atomically
        for (graph_name, new_edges) in rebuilt {
            let graph = self.graphs.get_mut(&graph_name).unwrap();
            graph.edge_labels = new_edges.into_iter().collect();
        }
        Ok(())
    }

    /// Restore saved vertex key maps (used to roll back after failed edge rebuild).
    fn restore_vertex_key_maps(
        &mut self,
        saved: &[(String, String, HashMap<pgq::KeyValue, usize>, Vec<pgq::KeyValue>)],
    ) {
        for (graph_name, label, user_to_row, row_to_user) in saved {
            if let Some(graph) = self.graphs.get_mut(graph_name) {
                if let Some(vl) = graph.vertex_labels.get_mut(label) {
                    vl.user_to_row = user_to_row.clone();
                    vl.row_to_user = row_to_user.clone();
                }
            }
        }
    }

    /// Rebuild vertex key maps for all vertex labels in the given graphs
    /// that reference `table_name`.  Returns an error if any rebuild fails;
    /// callers are responsible for rolling back via `restore_vertex_key_maps`.
    fn rebuild_vertex_key_maps_for_table(
        &mut self,
        affected: &[String],
        table_name: &str,
    ) -> Result<(), SqlError> {
        for graph_name in affected {
            let graph = self.graphs.get_mut(graph_name).unwrap();
            for vl in graph.vertex_labels.values_mut() {
                if vl.table_name == table_name {
                    let stored = self.tables.get(&vl.table_name).ok_or_else(|| {
                        SqlError::Plan(format!(
                            "Cannot rebuild graph '{}': vertex table '{}' not found",
                            graph_name, vl.table_name
                        ))
                    })?;
                    pgq::rebuild_vertex_key_map(vl, stored)?;
                }
            }
        }
        Ok(())
    }

    /// Rebuild edge labels (CSR relations) for the given graphs.
    /// Returns the rebuilt edges grouped by graph name, or an error if any
    /// graph fails validation.  Callers are responsible for rolling back
    /// vertex key maps on error.
    fn rebuild_edges_for_graphs(
        &self,
        affected: &[String],
    ) -> Result<Vec<(String, Vec<(String, pgq::StoredRel)>)>, SqlError> {
        let mut rebuilt = Vec::new();
        for graph_name in affected {
            let graph = self.graphs.get(graph_name).unwrap();
            let mut new_edges: Vec<(String, pgq::StoredRel)> = Vec::new();
            for (label, sr) in &graph.edge_labels {
                let el = &sr.edge_label;
                let edge_stored = self.tables.get(&el.table_name).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Cannot rebuild graph '{}': edge table '{}' not found",
                        graph_name, el.table_name
                    ))
                })?;
                let src_stored = self.tables.get(&el.src_ref_table).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Cannot rebuild graph '{}': source vertex table '{}' not found",
                        graph_name, el.src_ref_table
                    ))
                })?;
                let dst_stored = self.tables.get(&el.dst_ref_table).ok_or_else(|| {
                    SqlError::Plan(format!(
                        "Cannot rebuild graph '{}': destination vertex table '{}' not found",
                        graph_name, el.dst_ref_table
                    ))
                })?;

                let n_src = pgq::checked_logical_nrows(src_stored)? as i64;
                let n_dst = pgq::checked_logical_nrows(dst_stored)? as i64;

                let src_vl = graph.vertex_labels.values()
                    .find(|vl| vl.table_name == el.src_ref_table && vl.key_column == el.src_ref_col)
                    .ok_or_else(|| SqlError::Plan(format!(
                        "No vertex label found for source table '{}' with key column '{}'",
                        el.src_ref_table, el.src_ref_col
                    )))?;
                let dst_vl = graph.vertex_labels.values()
                    .find(|vl| vl.table_name == el.dst_ref_table && vl.key_column == el.dst_ref_col)
                    .ok_or_else(|| SqlError::Plan(format!(
                        "No vertex label found for destination table '{}' with key column '{}'",
                        el.dst_ref_table, el.dst_ref_col
                    )))?;

                let (rel, edge_row_map) = pgq::remap_and_build_rel(
                    self, edge_stored, src_vl, dst_vl, el, n_src, n_dst,
                )?;
                new_edges.push((
                    label.clone(),
                    pgq::StoredRel {
                        rel,
                        edge_label: el.clone(),
                        edge_row_map,
                    },
                ));
            }
            rebuilt.push((graph_name.clone(), new_edges));
        }
        Ok(rebuilt)
    }

    /// Remove any property graphs that reference the given table.
    /// Used when a table is dropped — no rebuild is possible, so just clean up.
    pub(crate) fn remove_graphs_for_table(&mut self, table_name: &str) {
        self.graphs.retain(|_, graph| {
            let references_table = graph
                .vertex_labels
                .values()
                .any(|vl| vl.table_name == table_name)
                || graph.edge_labels.values().any(|sr| {
                    sr.edge_label.table_name == table_name
                        || sr.edge_label.src_ref_table == table_name
                        || sr.edge_label.dst_ref_table == table_name
                });
            !references_table
        });
    }

    /// List stored table names.
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Get (nrows, ncols) for a stored table, or None if not found.
    pub fn table_info(&self, name: &str) -> Option<(i64, usize)> {
        self.tables
            .get(name)
            .map(|st| (st.logical_nrows(), st.columns.len()))
    }

    /// Find a vector index for a given table and column.
    pub(crate) fn find_vector_index(
        &self,
        table_name: &str,
        column_name: &str,
    ) -> Option<&VectorIndexInfo> {
        self.vector_indexes.values().find(|vi| {
            vi.table_name == table_name && vi.column_name == column_name
        })
    }

    /// Remove any vector indexes that reference the given table.
    pub(crate) fn remove_vector_indexes_for_table(&mut self, table_name: &str) {
        self.vector_indexes
            .retain(|_, vi| vi.table_name != table_name);
    }
}

/// Split a SQL script into individual statements on `;`, respecting
/// single-quoted strings, double-quoted identifiers, `--` line comments,
/// and `/* */` block comments so that semicolons inside them are not
/// treated as statement separators.
fn split_sql_statements(sql: &str) -> Vec<&str> {
    let mut stmts = Vec::new();
    let mut start = 0;
    let bytes = sql.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        match bytes[i] {
            b'\'' => {
                // Single-quoted string: skip until closing quote ('' is escaped quote)
                i += 1;
                while i < len {
                    if bytes[i] == b'\'' {
                        i += 1;
                        if i < len && bytes[i] == b'\'' {
                            i += 1; // escaped quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'"' => {
                // Double-quoted identifier: skip until closing quote ("" is escaped)
                i += 1;
                while i < len {
                    if bytes[i] == b'"' {
                        i += 1;
                        if i < len && bytes[i] == b'"' {
                            i += 1; // escaped quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'-' if i + 1 < len && bytes[i + 1] == b'-' => {
                // Line comment: skip until newline
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                if i < len {
                    i += 1; // skip newline
                }
            }
            b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                // Block comment: skip until */
                i += 2;
                while i + 1 < len {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            b';' => {
                let s = sql[start..i].trim();
                if !s.is_empty() {
                    stmts.push(s);
                }
                i += 1;
                start = i;
            }
            _ => {
                i += 1;
            }
        }
    }
    let s = sql[start..].trim();
    if !s.is_empty() {
        stmts.push(s);
    }
    stmts
}

/// Skip whitespace and SQL comments (both `--` line comments and `/* */` block
/// comments) at the start of `s`, returning the remaining suffix.
pub(super) fn skip_whitespace_and_comments(s: &str) -> &str {
    let mut rest = s;
    loop {
        rest = rest.trim_start();
        if rest.starts_with("--") {
            // Line comment: skip to end of line
            match rest.find('\n') {
                Some(pos) => rest = &rest[pos + 1..],
                None => return "",
            }
        } else if rest.starts_with("/*") {
            // Block comment: skip to closing */
            match rest[2..].find("*/") {
                Some(pos) => rest = &rest[pos + 4..],
                None => return "",
            }
        } else {
            return rest;
        }
    }
}

/// Check whether `sql` contains the keyword `GRAPH_TABLE` outside of
/// single-quoted strings, double-quoted identifiers, and SQL comments.
fn contains_graph_table_keyword(sql: &str) -> bool {
    let bytes = sql.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        match bytes[i] {
            b'\'' => {
                i += 1;
                while i < len {
                    if bytes[i] == b'\'' {
                        i += 1;
                        if i < len && bytes[i] == b'\'' {
                            i += 1;
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'"' => {
                i += 1;
                while i < len {
                    if bytes[i] == b'"' {
                        i += 1;
                        if i < len && bytes[i] == b'"' {
                            i += 1;
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'-' if i + 1 < len && bytes[i + 1] == b'-' => {
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
            }
            b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < len {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            _ => {
                if i + 11 <= len {
                    if let Some(slice) = sql.get(i..i + 11) {
                        if slice.eq_ignore_ascii_case("GRAPH_TABLE") {
                            let before_ok = i == 0
                                || (!bytes[i - 1].is_ascii_alphanumeric()
                                    && bytes[i - 1] != b'_');
                            let after_pos = i + 11;
                            let after_ok = after_pos >= len
                                || (!bytes[after_pos].is_ascii_alphanumeric()
                                    && bytes[after_pos] != b'_');
                            if before_ok && after_ok {
                                // Verify GRAPH_TABLE is followed by '(' (with optional whitespace/comments)
                                let rest = &sql[after_pos..];
                                let trimmed = skip_whitespace_and_comments(rest);
                                if trimmed.starts_with('(') {
                                    return true;
                                }
                            }
                        }
                    }
                }
                // Advance by UTF-8 char width to avoid landing on continuation bytes
                let ch_len = sql[i..].chars().next().map_or(1, |c| c.len_utf8());
                i += ch_len;
            }
        }
    }
    false
}

/// Parse and execute a SQL query, returning the result table and column list.
/// Stateless single-query mode (no session registry).
pub fn execute_sql(ctx: &Context, sql: &str) -> Result<SqlResult, SqlError> {
    planner::plan_and_execute(ctx, sql, None)
}
