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

//! PG wire protocol handler that bridges pgwire traits to the Teide engine.
//!
//! Architecture: Session (which contains !Send C pointers) lives on a dedicated
//! OS thread. The async pgwire handler communicates with it via channels,
//! avoiding any Send/Sync issues with the C engine's thread-local arenas.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use futures::sink::Sink;
use pgwire::api::auth::{DefaultServerParameterProvider, StartupHandler};
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::{ExtendedQueryHandler, SimpleQueryHandler};
use pgwire::api::results::{
    DescribePortalResponse, DescribeResponse, DescribeStatementResponse, FieldFormat, FieldInfo,
    Response, Tag,
};
use pgwire::api::stmt::{QueryParser, StoredStatement};
use pgwire::api::store::PortalStore;
use pgwire::api::ClientPortalStore;
use pgwire::api::{ClientInfo, PgWireServerHandlers, Type};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use tokio::sync::Mutex;

use super::catalog;
use super::encode;
use crate::sql::{ExecResult, Session, StoredTable};

// ---------------------------------------------------------------------------
// SendableTables: safe cross-thread transfer of StoredTable collections
// ---------------------------------------------------------------------------

/// Wrapper for cross-thread transfer of StoredTable collections.
///
/// # Safety
/// `StoredTable` contains `Table { ptr: *mut td_t, .. }` which is `!Send`.
/// This wrapper is safe because:
/// - Clones use `clone_ref()` (atomic `td_retain`) — the refcount is atomic.
/// - Each clone is moved exactly once into a dedicated engine thread.
/// - The engine thread exclusively owns its Session and StoredTable copies.
/// - The factory only creates clones; it never dereferences the raw pointers.
struct SendableTables(Vec<(String, StoredTable)>);
unsafe impl Send for SendableTables {}
unsafe impl Sync for SendableTables {}

impl SendableTables {
    fn into_inner(self) -> Vec<(String, StoredTable)> {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Engine thread: owns Session, processes queries sequentially
// ---------------------------------------------------------------------------

/// A query result serialized into Send-safe types.
/// All cell values are pre-formatted as text-protocol strings.
pub struct WireResult {
    pub columns: Vec<(String, i8)>, // (name, td_type)
    pub rows: Vec<Vec<Option<String>>>,
}

/// Response from the engine thread.
pub enum EngineResponse {
    Query(WireResult),
    Ddl(String),
}

/// Request sent to the engine thread.
struct EngineRequest {
    sql: String,
    reply: tokio::sync::oneshot::Sender<Result<EngineResponse, String>>,
}

/// A Send-safe handle to a Session running on a dedicated OS thread.
/// Cloneable — each clone shares the same engine thread.
#[derive(Clone)]
pub struct SessionBridge {
    tx: std::sync::mpsc::Sender<EngineRequest>,
}

impl SessionBridge {
    /// Spawn a dedicated OS thread that owns a Session with the given tables.
    fn spawn(tables: SendableTables) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<EngineRequest>();

        std::thread::spawn(move || {
            // into_inner() consumes `tables` (SendableTables), forcing Rust 2021
            // edition to capture the whole Send wrapper, not the inner !Send Vec.
            let base_tables = tables.into_inner();

            // Create a fresh Session on this thread (initializes arena)
            let mut session = match Session::new() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[engine] Failed to create session: {e}");
                    return;
                }
            };
            // Insert base tables into this session
            for (name, stored) in base_tables {
                session.tables.insert(name, stored);
            }

            // Process requests until the channel is closed
            while let Ok(req) = rx.recv() {
                let result = match session.execute(&req.sql) {
                    Ok(ExecResult::Query(sql_result)) => {
                        let ncols = sql_result.table.ncols() as usize;
                        let nrows = sql_result.table.nrows() as usize;

                        let columns: Vec<(String, i8)> = (0..ncols)
                            .map(|i| {
                                let name = if i < sql_result.columns.len() {
                                    sql_result.columns[i].clone()
                                } else {
                                    sql_result.table.col_name_str(i)
                                };
                                let td_type = sql_result.table.col_type(i);
                                (name, td_type)
                            })
                            .collect();

                        let mut rows = Vec::with_capacity(nrows);
                        for r in 0..nrows {
                            let mut row = Vec::with_capacity(ncols);
                            for c in 0..ncols {
                                row.push(super::types::format_cell(&sql_result.table, c, r));
                            }
                            rows.push(row);
                        }

                        Ok(EngineResponse::Query(WireResult { columns, rows }))
                    }
                    Ok(ExecResult::Ddl(msg)) => Ok(EngineResponse::Ddl(msg)),
                    Err(e) => Err(e.to_string()),
                };

                // If the receiver dropped, the connection closed — just move on
                let _ = req.reply.send(result);
            }
        });

        SessionBridge { tx }
    }

    /// Send a SQL query to the engine thread and await the result.
    pub async fn query(&self, sql: String) -> Result<EngineResponse, String> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(EngineRequest {
                sql,
                reply: reply_tx,
            })
            .map_err(|_| "engine thread stopped".to_string())?;

        reply_rx
            .await
            .map_err(|_| "engine thread dropped reply".to_string())?
    }
}

// ---------------------------------------------------------------------------
// Table metadata (Send-safe snapshot of session table registry)
// ---------------------------------------------------------------------------

/// Send-safe metadata about a stored table (no raw pointers).
#[derive(Clone)]
pub struct TableMeta {
    pub columns: Vec<(String, i8)>, // (col_name, td_type)
    pub nrows: i64,
}

/// Send-safe snapshot of the session's table registry for catalog queries.
#[derive(Clone)]
pub struct SessionMeta {
    pub tables: Vec<(String, TableMeta)>,
}

impl SessionMeta {
    pub fn from_session(session: &Session) -> Self {
        let tables = session
            .table_names()
            .into_iter()
            .filter_map(|name| {
                session.tables.get(name).map(|st| {
                    let ncols = st.columns.len().max(st.table.ncols() as usize);
                    let columns: Vec<(String, i8)> = (0..ncols)
                        .map(|i| {
                            let col_name = if i < st.columns.len() {
                                st.columns[i].clone()
                            } else {
                                st.table.col_name_str(i)
                            };
                            let td_type = st.table.col_type(i);
                            (col_name, td_type)
                        })
                        .collect();
                    let nrows = st.table.nrows();
                    (name.to_string(), TableMeta { columns, nrows })
                })
            })
            .collect();
        SessionMeta { tables }
    }
}

// ---------------------------------------------------------------------------
// Per-connection pgwire handler
// ---------------------------------------------------------------------------

/// Per-connection handler. Owns a SessionBridge (engine-thread handle) and
/// SessionMeta (for catalog queries without hitting the engine).
///
/// The `describe_cache` holds query results from Describe (statement) so that
/// the subsequent Execute can return the cached result without re-running.
/// This is needed because the extended protocol requires column metadata
/// from Describe before Execute sends any data rows.
pub struct TeideHandler {
    bridge: SessionBridge,
    meta: SessionMeta,
    describe_cache: Arc<Mutex<HashMap<String, WireResult>>>,
}

impl TeideHandler {
    /// Execute a SQL query through the catalog handler or engine bridge.
    /// Returns the pgwire Response(s).
    async fn execute_sql(&self, query: &str) -> PgWireResult<Vec<Response>> {
        // Check for catalog queries (handled locally, no engine needed)
        if catalog::is_catalog_query(query) {
            if let Some(result) = catalog::handle_catalog_query(query, &self.meta) {
                return result;
            }
        }

        // Strip schema prefix
        let sql = query.replace("public.", "").replace("\"public\".", "");

        let result = self.bridge.query(sql).await.map_err(|e| {
            PgWireError::UserError(Box::new(ErrorInfo::new(
                "ERROR".to_string(),
                "XX000".to_string(),
                e,
            )))
        })?;

        match result {
            EngineResponse::Query(wire_result) => {
                let qr = encode::encode_wire_result(&wire_result, false)?;
                Ok(vec![Response::Query(qr)])
            }
            EngineResponse::Ddl(msg) => Ok(vec![Response::Execution(Tag::new(&msg).with_rows(0))]),
        }
    }

    /// Execute a query and cache the WireResult for subsequent do_query.
    /// Returns the column schema for Describe responses.
    ///
    /// All column types are mapped to VARCHAR for the extended protocol.
    /// tokio-postgres (and most JDBC drivers) always request binary format
    /// in Bind. Since we only have text-encoded values, VARCHAR is the
    /// correct type — its binary representation is identical to text
    /// (raw UTF-8 bytes).
    async fn describe_and_cache(&self, sql: &str) -> PgWireResult<Vec<FieldInfo>> {
        let lower = sql.to_lowercase();
        let lower = lower.trim();

        // Catalog queries
        if catalog::is_catalog_query(sql) {
            // DDL-like: SET, BEGIN, COMMIT, DEALLOCATE, etc. → no result columns
            if lower.starts_with("set ")
                || lower == "begin"
                || lower == "commit"
                || lower == "rollback"
                || lower == "end"
                || lower.starts_with("begin;")
                || lower.starts_with("commit;")
                || lower.starts_with("rollback;")
                || lower.starts_with("deallocate ")
                || lower.starts_with("close ")
                || lower.starts_with("discard ")
            {
                return Ok(vec![]);
            }

            // SELECT <constant> (e.g. SELECT 1, SELECT 'hello') → cache result
            if let Some(rest) = lower.strip_prefix("select ") {
                let val = rest.trim().trim_end_matches(';').trim();
                let is_constant = val.parse::<i64>().is_ok()
                    || (val.starts_with('\'') && val.ends_with('\''))
                    || val == "true"
                    || val == "false"
                    || val == "null";
                if is_constant {
                    let display_val = val.trim_matches('\'').to_string();
                    let wr = WireResult {
                        columns: vec![("?column?".to_string(), 0)],
                        rows: vec![vec![Some(display_val)]],
                    };
                    let fields = vec![FieldInfo::new(
                        "?column?".to_string(),
                        None,
                        None,
                        Type::VARCHAR,
                        FieldFormat::Text,
                    )];
                    let mut cache = self.describe_cache.lock().await;
                    cache.insert(sql.to_string(), wr);
                    return Ok(fields);
                }
            }

            // SELECT version(), SELECT current_schema, etc. → single VARCHAR column
            if lower.starts_with("select version()")
                || lower.starts_with("select current_schema")
                || lower.starts_with("select current_database")
                || lower.contains("pg_backend_pid()")
            {
                // Run catalog handler to get the actual value
                if let Some(Ok(responses)) = catalog::handle_catalog_query(sql, &self.meta) {
                    // These catalog functions return single-column, single-row results.
                    // Cache a generic WireResult so do_query can skip re-running.
                    let fields = vec![FieldInfo::new(
                        "?column?".to_string(),
                        None,
                        None,
                        Type::VARCHAR,
                        FieldFormat::Text,
                    )];
                    // We can't extract data from Response, but do_query will re-run
                    // the catalog handler if there's no cache hit.
                    drop(responses);
                    return Ok(fields);
                }
            }

            // Other catalog queries: return empty schema.
            // do_query will handle them by running the catalog handler.
            return Ok(vec![]);
        }

        let clean_sql = sql.replace("public.", "").replace("\"public\".", "");

        let result = self.bridge.query(clean_sql).await.map_err(|e| {
            PgWireError::UserError(Box::new(ErrorInfo::new(
                "ERROR".to_string(),
                "XX000".to_string(),
                e,
            )))
        })?;

        match result {
            EngineResponse::Query(wire_result) => {
                // All columns as VARCHAR for extended protocol compatibility
                let schema: Vec<FieldInfo> = wire_result
                    .columns
                    .iter()
                    .map(|(name, _td_type)| {
                        FieldInfo::new(name.clone(), None, None, Type::VARCHAR, FieldFormat::Text)
                    })
                    .collect();

                // Cache the result for subsequent do_query
                let mut cache = self.describe_cache.lock().await;
                cache.insert(sql.to_string(), wire_result);

                Ok(schema)
            }
            EngineResponse::Ddl(_) => Ok(vec![]),
        }
    }
}

#[async_trait]
impl SimpleQueryHandler for TeideHandler {
    async fn do_query<C>(&self, _client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        eprintln!("[query] {query}");
        self.execute_sql(query).await
    }
}

// ---------------------------------------------------------------------------
// Extended Query Protocol: TeideQueryParser + ExtendedQueryHandler
// ---------------------------------------------------------------------------

/// Query parser for the extended protocol. Stores the raw SQL string as the
/// "statement" — Teide's SQL planner handles actual parsing during execution.
pub struct TeideQueryParser;

#[async_trait]
impl QueryParser for TeideQueryParser {
    type Statement = String;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<String>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        Ok(sql.to_string())
    }

    fn get_parameter_types(&self, _stmt: &String) -> PgWireResult<Vec<Type>> {
        // No parameterized queries yet
        Ok(vec![])
    }

    fn get_result_schema(
        &self,
        _stmt: &String,
        _column_format: Option<&Format>,
    ) -> PgWireResult<Vec<pgwire::api::results::FieldInfo>> {
        // Lazy: schema determined at execution time, not parse time.
        // Describe returns NoData; JDBC handles this gracefully.
        Ok(vec![])
    }
}

#[async_trait]
impl ExtendedQueryHandler for TeideHandler {
    type Statement = String;
    type QueryParser = TeideQueryParser;

    fn query_parser(&self) -> Arc<Self::QueryParser> {
        Arc::new(TeideQueryParser)
    }

    async fn do_query<C>(
        &self,
        _client: &mut C,
        portal: &Portal<String>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = String>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let query = portal.statement.statement.as_str();
        eprintln!("[extended] {query}");

        // Check describe cache first (Describe already ran the query)
        {
            let mut cache = self.describe_cache.lock().await;
            if let Some(wire_result) = cache.remove(query) {
                // all_text=true: map all types to VARCHAR for extended protocol
                let qr = encode::encode_wire_result(&wire_result, true)?;
                return Ok(Response::Query(qr));
            }
        }

        // Not cached (e.g. DDL, catalog, or Describe was skipped) — execute now
        let mut responses = self.execute_sql(query).await?;
        Ok(responses.remove(0))
    }

    async fn do_describe_statement<C>(
        &self,
        _client: &mut C,
        statement: &StoredStatement<String>,
    ) -> PgWireResult<DescribeStatementResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = String>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let sql = &statement.statement;

        // Execute the query now, cache results for do_query, return schema
        let fields = self.describe_and_cache(sql).await?;

        // No params (we don't support parameterized queries yet)
        Ok(DescribeStatementResponse::new(vec![], fields))
    }

    async fn do_describe_portal<C>(
        &self,
        _client: &mut C,
        portal: &Portal<String>,
    ) -> PgWireResult<DescribePortalResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = String>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let sql = &portal.statement.statement;

        // Check if we already described (and cached) this statement
        let cache = self.describe_cache.lock().await;
        if let Some(wire_result) = cache.get(sql.as_str()) {
            let fields: Vec<FieldInfo> = wire_result
                .columns
                .iter()
                .map(|(name, _td_type)| {
                    // All VARCHAR for extended protocol compatibility
                    FieldInfo::new(name.clone(), None, None, Type::VARCHAR, FieldFormat::Text)
                })
                .collect();
            return Ok(DescribePortalResponse::new(fields));
        }

        Ok(DescribePortalResponse::no_data())
    }
}

// ---------------------------------------------------------------------------
// Custom startup handler with clean ParameterStatus
// ---------------------------------------------------------------------------

/// Startup handler that sends ParameterStatus with clean server_version
/// (no pgwire suffix) and proper settings for JDBC compatibility.
pub struct TeideStartupHandler;

#[async_trait]
impl StartupHandler for TeideStartupHandler {
    async fn on_startup<C>(
        &self,
        client: &mut C,
        message: pgwire::messages::PgWireFrontendMessage,
    ) -> PgWireResult<()>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        if let pgwire::messages::PgWireFrontendMessage::Startup(ref startup) = message {
            pgwire::api::auth::protocol_negotiation(client, startup).await?;
            pgwire::api::auth::save_startup_parameters_to_metadata(client, startup);

            // Customized parameters: clean server_version, ISO MDY date style
            let mut params = DefaultServerParameterProvider::default();
            params.server_version = "16.6".to_string();
            params.date_style = "ISO, MDY".to_string();

            pgwire::api::auth::finish_authentication(client, &params).await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Handler factory
// ---------------------------------------------------------------------------

/// Factory that creates per-connection handlers.
/// Each connection gets its own engine thread with a cloned Session.
pub struct TeideHandlerFactory {
    base_tables: Arc<SendableTables>,
    meta: SessionMeta,
}

impl TeideHandlerFactory {
    /// Create a factory from an existing session. Snapshots the current
    /// table registry as the template for new connections.
    pub fn from_session(session: &Session) -> Self {
        let base_tables: Vec<(String, StoredTable)> = session
            .tables
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let meta = SessionMeta::from_session(session);
        TeideHandlerFactory {
            base_tables: Arc::new(SendableTables(base_tables)),
            meta,
        }
    }

    fn make_handler(&self) -> TeideHandler {
        let tables = SendableTables(self.base_tables.0.clone());
        let bridge = SessionBridge::spawn(tables);
        TeideHandler {
            bridge,
            meta: self.meta.clone(),
            describe_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl PgWireServerHandlers for TeideHandlerFactory {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        Arc::new(self.make_handler())
    }

    fn extended_query_handler(&self) -> Arc<impl ExtendedQueryHandler> {
        Arc::new(self.make_handler())
    }

    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        Arc::new(TeideStartupHandler)
    }
}
