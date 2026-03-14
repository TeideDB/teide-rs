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

use crate::{Context, Table};
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
        let upper = sql.to_uppercase();
        if upper.contains("GRAPH_TABLE") {
            return self.execute_with_graph_table(sql);
        }

        planner::session_execute(self, sql)
    }

    /// Execute SQL containing GRAPH_TABLE expressions.
    /// Extracts GRAPH_TABLE, executes the graph query, stores the result
    /// as a temporary table, rewrites the SQL, and runs the modified SQL.
    fn execute_with_graph_table(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
        let (rewritten_sql, graph_exprs) = pgq_parser::extract_graph_tables(sql)?;

        // Execute each GRAPH_TABLE and store results as temp tables
        let mut temp_names = Vec::new();
        let graph_result = (|| {
            for (i, expr) in graph_exprs.iter().enumerate() {
                let temp_name = format!("__pgq_result_{i}");
                let (table, columns) = pgq::plan_graph_table(self, expr)?;
                self.tables.insert(
                    temp_name.clone(),
                    StoredTable { table, columns },
                );
                temp_names.push(temp_name);
            }

            // Run the rewritten SQL (which references __pgq_result_N tables)
            planner::session_execute(self, &rewritten_sql)
        })();

        // Always clean up temp tables, even on error
        for name in &temp_names {
            self.tables.remove(name);
        }

        graph_result
    }

    /// Execute a multi-statement SQL script (statements separated by `;`).
    /// Returns the result of the last statement.
    ///
    /// Uses sqlparser to split statements correctly, respecting string
    /// literals and quoted identifiers.
    pub fn execute_script(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
        use sqlparser::dialect::DuckDbDialect;
        use sqlparser::parser::Parser;

        let dialect = DuckDbDialect {};
        let stmts = Parser::parse_sql(&dialect, sql).map_err(|e| SqlError::Parse(e.to_string()))?;
        if stmts.is_empty() {
            return Err(SqlError::Plan("Empty script".into()));
        }
        let mut last = None;
        for stmt in &stmts {
            let s = stmt.to_string();
            last = Some(self.execute(&s)?);
        }
        last.ok_or_else(|| SqlError::Plan("Empty script".into()))
    }

    /// Execute a SQL script from a file path.
    pub fn execute_script_file(&mut self, path: &std::path::Path) -> Result<ExecResult, SqlError> {
        let sql = std::fs::read_to_string(path)
            .map_err(|e| SqlError::Plan(format!("Failed to read {}: {e}", path.display())))?;
        self.execute_script(&sql)
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

/// Parse and execute a SQL query, returning the result table and column list.
/// Stateless single-query mode (no session registry).
pub fn execute_sql(ctx: &Context, sql: &str) -> Result<SqlResult, SqlError> {
    planner::plan_and_execute(ctx, sql, None)
}
