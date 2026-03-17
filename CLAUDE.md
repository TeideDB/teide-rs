# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is teide-rs?

Rust bindings for the [Teide](https://github.com/TeideDB/teide) C17 columnar dataframe engine. Provides safe FFI wrappers, a SQL parser/planner, an interactive REPL, and a PostgreSQL wire protocol server.

## Build & Test Commands

```bash
# Build everything (CLI + server + library)
cargo build --all-features

# Build just the CLI or server
cargo build --features cli
cargo build --features server

# Run all tests (CI skips server/extended tests)
cargo test --all-features
cargo test --all-features -- --skip server_ --skip extended_   # CI equivalent

# Run a single test
cargo test --all-features -- test_name

# Run SLT (SQL logic tests) only
cargo test --all-features -- slt

# Run the REPL
cargo run --features cli

# Run the PgWire server
cargo run --features server -- --port 5433

# Benchmarks
cargo bench --all-features
```

## Architecture

### Crate Structure

- `src/ffi.rs` — Raw FFI bindings (hand-written from `vendor/teide/include/teide/td.h`)
- `src/engine.rs` — Safe Rust wrappers: `Context`, `Table`, `Graph`, `Column`, `Rel`, `Error`, `HnswIndex`, embedding/vector similarity ops
- `src/sql/mod.rs` — `Session`, `ExecResult`, `execute_sql` entry point
- `src/sql/planner.rs` — SQL AST → Teide Graph translation
- `src/sql/expr.rs` — Expression tree walker, aggregate collection
- `src/sql/pgq.rs` — Property graph catalog types, MATCH pattern AST, and GRAPH_TABLE planner
- `src/sql/pgq_parser.rs` — Pre-parser intercepting PGQ syntax (CREATE/DROP PROPERTY GRAPH, GRAPH_TABLE) and vector index DDL (CREATE/DROP VECTOR INDEX) before sqlparser
- `src/cli/` — REPL binary (feature-gated on `cli`)
- `src/server/` — PgWire server binary (feature-gated on `server`)
- `tests/engine_api.rs` — Rust API integration tests for engine, graph, and algorithm ops
- `tests/slt/*.slt` — SQL logic tests (SLT format) run by `tests/slt_runner.rs`

### C Engine Singleton

The C engine uses global state (thread-local arenas, global symbol table). Key constraints:

- `EngineGuard` manages init/destroy lifecycle via `OnceLock<Mutex<Weak<EngineGuard>>>`
- `Context` is `!Send + !Sync` (via `PhantomData<*mut ()>`) because the C engine uses thread-local arenas
- Multiple `Context` handles share one `Arc<EngineGuard>`; engine tears down when all drop

### Graph API (Lazy DAG)

All computation goes through a lazy DAG: `ctx.graph(&table)` → chain ops (`scan`, `filter`, `add`, etc.) → `g.execute(root)`. `Column` is a non-owning `Copy` handle (raw `*mut td_op_t`). On `execute()`, the C optimizer runs (type inference, constant fold, predicate pushdown, CSE, fusion, DCE), then the morsel-driven executor runs. Graph algorithm ops (`pagerank`, `connected_comp`, `dijkstra`, `louvain`) operate directly on CSR indexes and return TD_TABLE results with computed per-node values. These kernels allocate scratch buffers via `td_scratch_arena_t` (bump arena backed by 64KB buddy blocks): `td_scratch_arena_init()` at entry, `td_scratch_arena_push()` per buffer, single `td_scratch_arena_reset()` on every exit path. Do not use `scratch_alloc()`/`scratch_free()` for graph algorithm kernels (pagerank, connected_comp, dijkstra, louvain). Vector similarity ops (`cosine_sim`, `euclidean_dist`, `knn`) operate on TD_F32 embedding columns (flat N*D float arrays created via `Table::create_embedding_column`). Each has an `unsafe` borrowed variant and a safe `_owned` variant that pins the query `Vec<f32>` in the Graph's `_pinned` storage. HNSW approximate nearest neighbor search is available via `HnswIndex`, an RAII wrapper around `td_hnsw_t`. `HnswIndex::build()` constructs a multi-layer proximity graph; `search()` performs beam search. The index supports `save`/`load`/`mmap` persistence. `Rel` exposes direct CSR accessors: `neighbors(node, direction)` returns a `&[i64]` slice of neighbor node IDs (zero-copy into the CSR targets array, valid while the `Rel` lives); `n_nodes(direction)` returns the node count for one side. Direction `0` = forward (src->dst), `1` = reverse (dst->src). Both are O(1), zero-allocation.

### SQL Layer

Uses `sqlparser` crate with `DuckDbDialect`. `Session` holds a `HashMap<String, StoredTable>` as table registry and a `HashMap<String, PropertyGraph>` for graph metadata. Statements dispatch through `planner::session_execute()` which builds Graph ops and executes them. Supports SELECT, CREATE TABLE AS, DROP TABLE, INSERT INTO, UPDATE, DELETE, CREATE/DROP PROPERTY GRAPH, and GRAPH_TABLE with MATCH patterns. PGQ syntax is intercepted by a custom pre-parser (`pgq_parser.rs`) before reaching sqlparser. Vertex tables support natural (non-sequential) keys via `KEY (col)` in CREATE PROPERTY GRAPH: `VertexLabel` stores bidirectional maps (`user_to_row: HashMap<KeyValue, usize>`, `row_to_user: Vec<KeyValue>`) so user-facing integer or string keys are transparently remapped to internal 0-based row indices during CSR construction and reversed during column projection and shortest-path output. GRAPH_TABLE WHERE clauses support rich filter expressions (`AND`, `OR`, `NOT`, comparisons, `IS NULL`, `IS NOT NULL`, `IN`, `BETWEEN`) evaluated via `evaluate_filter()` in `pgq.rs`, which walks the sqlparser `Expr` tree against row data. GRAPH_TABLE COLUMNS supports algorithm functions: `PAGERANK()`, `COMPONENT()` (alias: `CONNECTED_COMPONENT`), and `COMMUNITY()` (alias: `LOUVAIN`). Scalar functions `COSINE_SIMILARITY(emb_col, ARRAY[...])` and `EUCLIDEAN_DISTANCE(emb_col, ARRAY[...])` compute vector similarity; query vectors are specified as ARRAY literals. `Session::add_embedding_column` adds a TD_F32 embedding column to a stored table and registers its dimension. `Session::register_embedding_column` registers an existing TD_F32 column's dimension for query-vector validation. `StoredTable` carries per-column `embedding_dims` metadata that the planner uses to validate query vector lengths at plan time. Tables with registered embedding columns have DML restrictions: UPDATE is not supported, DELETE with WHERE is not supported (DELETE without WHERE truncates the table but preserves embedding column metadata), and SELECT with WHERE/ORDER BY/LIMIT is not yet supported. These restrictions exist because the C engine's filter, sort, and head kernels operate element-wise on flat F32 arrays and are not dimension-aware. `CREATE VECTOR INDEX name ON table(column) USING HNSW(M=16, ef_construction=200)` builds an HNSW index on an embedding column; `DROP VECTOR INDEX [IF EXISTS] name` removes it. These are parsed by the PGQ pre-parser. `Session` holds a `HashMap<String, VectorIndexInfo>` for vector index metadata. DML operations (INSERT, DELETE) on a table with vector indexes automatically drop those indexes since the underlying column data is reallocated.

### Server Thread Model

The C engine's `!Send` constraint requires special handling: each pgwire connection spawns a dedicated OS thread owning a `Session`. The async pgwire handler communicates with the engine thread via channels.

### Type System

Every C object is a `td_t*` (32-byte header + data). Key type constants in `ffi.rs`: `TD_BOOL=1`, `TD_I16=4`, `TD_I32=5`, `TD_I64=6`, `TD_F64=7`, `TD_F32=8`, `TD_DATE=9`, `TD_TIME=10`, `TD_TIMESTAMP=11`, `TD_SYM=20`, `TD_TABLE=13`. Positive type = vector, negative = atom, 0 = list. Error returns use low pointer values (< 32) as sentinels checked via `td_is_err()`.

## Critical: Test Serialization

All tests **must** acquire `ENGINE_LOCK: Mutex<()>` before creating a `Context`, because the C engine's global state cannot be initialized/destroyed concurrently in the same process. Every test function follows this pattern:

```rust
let _guard = ENGINE_LOCK.lock().unwrap();
let ctx = Context::new().unwrap();
```

## Build System

`build.rs` compiles the vendored C source tree at `vendor/teide/` using the `cc` crate (C17 standard). It also embeds `GIT_HASH` for the CLI banner. The `Cargo.toml` declares `links = "teide"`.
