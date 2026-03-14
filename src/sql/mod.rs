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

use crate::{Context, Rel, Table};
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
}

impl Clone for StoredTable {
    fn clone(&self) -> Self {
        StoredTable {
            table: self.table.clone_ref(),
            columns: self.columns.clone(),
        }
    }
}

/// A stateful SQL session that maintains a table registry across queries.
///
// pub(crate) access allows planner to manage table registry directly.
// This is intentional for simplicity; encapsulation via methods would
// add complexity without safety benefit since planner is the only consumer.
pub struct Session {
    pub(crate) tables: HashMap<String, StoredTable>,
    pub(crate) graphs: HashMap<String, pgq::PropertyGraph>,
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
        })
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
                    StoredTable { table, columns },
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

        // Collect all rebuilt edges first, only apply if ALL graphs validate
        let mut rebuilt: Vec<(String, Vec<(String, pgq::StoredRel)>)> = Vec::new();

        for graph_name in &affected {
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

                // Re-validate vertex key columns are still valid rowid sequences
                pgq::validate_key_column_is_rowid(
                    src_stored,
                    &el.src_ref_col,
                    &el.src_ref_table,
                )?;
                pgq::validate_key_column_is_rowid(
                    dst_stored,
                    &el.dst_ref_col,
                    &el.dst_ref_table,
                )?;

                // Re-validate edge endpoint values are within vertex table bounds
                let n_src = src_stored.table.nrows();
                let n_dst = dst_stored.table.nrows();
                let n_edges = pgq::checked_nrows(&edge_stored.table)?;
                let src_col_idx =
                    pgq::find_col_idx(&edge_stored.table, &el.src_col).ok_or_else(|| {
                        SqlError::Plan(format!(
                            "Column '{}' not found in edge table '{}'",
                            el.src_col, el.table_name
                        ))
                    })?;
                let dst_col_idx =
                    pgq::find_col_idx(&edge_stored.table, &el.dst_col).ok_or_else(|| {
                        SqlError::Plan(format!(
                            "Column '{}' not found in edge table '{}'",
                            el.dst_col, el.table_name
                        ))
                    })?;
                for row in 0..n_edges {
                    if let Some(v) = edge_stored.table.get_i64(src_col_idx, row) {
                        if v < 0 || v >= n_src {
                            return Err(SqlError::Plan(format!(
                                "Graph '{}' integrity violation: edge table '{}' column '{}' \
                                 row {}: value {} is out of range for vertex table '{}' \
                                 (0..{}). DML on referenced vertex tables can invalidate \
                                 edge endpoints.",
                                graph_name, el.table_name, el.src_col, row, v,
                                el.src_ref_table, n_src
                            )));
                        }
                    }
                    if let Some(v) = edge_stored.table.get_i64(dst_col_idx, row) {
                        if v < 0 || v >= n_dst {
                            return Err(SqlError::Plan(format!(
                                "Graph '{}' integrity violation: edge table '{}' column '{}' \
                                 row {}: value {} is out of range for vertex table '{}' \
                                 (0..{}). DML on referenced vertex tables can invalidate \
                                 edge endpoints.",
                                graph_name, el.table_name, el.dst_col, row, v,
                                el.dst_ref_table, n_dst
                            )));
                        }
                    }
                }

                let rel = Rel::from_edges(
                    &edge_stored.table,
                    &el.src_col,
                    &el.dst_col,
                    n_src,
                    n_dst,
                    true,
                )?;
                new_edges.push((
                    label.clone(),
                    pgq::StoredRel {
                        rel,
                        edge_label: el.clone(),
                    },
                ));
            }
            rebuilt.push((graph_name.clone(), new_edges));
        }

        // All graphs validated successfully — apply all updates atomically
        for (graph_name, new_edges) in rebuilt {
            let graph = self.graphs.get_mut(&graph_name).unwrap();
            graph.edge_labels = new_edges.into_iter().collect();
        }
        Ok(())
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
            .map(|st| (st.table.nrows(), st.columns.len()))
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
