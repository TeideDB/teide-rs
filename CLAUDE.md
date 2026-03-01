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
- `src/engine.rs` — Safe Rust wrappers: `Context`, `Table`, `Graph`, `Column`, `Error`
- `src/sql/mod.rs` — `Session`, `ExecResult`, `execute_sql` entry point
- `src/sql/planner.rs` — SQL AST → Teide Graph translation
- `src/sql/expr.rs` — Expression tree walker, aggregate collection
- `src/cli/` — REPL binary (feature-gated on `cli`)
- `src/server/` — PgWire server binary (feature-gated on `server`)

### C Engine Singleton

The C engine uses global state (thread-local arenas, global symbol table). Key constraints:

- `EngineGuard` manages init/destroy lifecycle via `OnceLock<Mutex<Weak<EngineGuard>>>`
- `Context` is `!Send + !Sync` (via `PhantomData<*mut ()>`) because the C engine uses thread-local arenas
- Multiple `Context` handles share one `Arc<EngineGuard>`; engine tears down when all drop

### Graph API (Lazy DAG)

All computation goes through a lazy DAG: `ctx.graph(&table)` → chain ops (`scan`, `filter`, `add`, etc.) → `g.execute(root)`. `Column` is a non-owning `Copy` handle (raw `*mut td_op_t`). On `execute()`, the C optimizer runs (type inference, constant fold, predicate pushdown, CSE, fusion, DCE), then the morsel-driven executor runs.

### SQL Layer

Uses `sqlparser` crate with `DuckDbDialect`. `Session` holds a `HashMap<String, StoredTable>` as table registry. Statements dispatch through `planner::session_execute()` which builds Graph ops and executes them.

### Server Thread Model

The C engine's `!Send` constraint requires special handling: each pgwire connection spawns a dedicated OS thread owning a `Session`. The async pgwire handler communicates with the engine thread via channels.

### Type System

Every C object is a `td_t*` (32-byte header + data). Key type constants in `ffi.rs`: `TD_BOOL=1`, `TD_I16=4`, `TD_I32=5`, `TD_I64=6`, `TD_F64=7`, `TD_DATE=9`, `TD_TIME=10`, `TD_TIMESTAMP=11`, `TD_SYM=20`, `TD_TABLE=13`. Positive type = vector, negative = atom, 0 = list. Error returns use low pointer values (< 32) as sentinels checked via `td_is_err()`.

## Critical: Test Serialization

All tests **must** acquire `ENGINE_LOCK: Mutex<()>` before creating a `Context`, because the C engine's global state cannot be initialized/destroyed concurrently in the same process. Every test function follows this pattern:

```rust
let _guard = ENGINE_LOCK.lock().unwrap();
let ctx = Context::new().unwrap();
```

## Build System

`build.rs` compiles the vendored C source tree at `vendor/teide/` using the `cc` crate (C17 standard). It also embeds `GIT_HASH` for the CLI banner. The `Cargo.toml` declares `links = "teide"`.
