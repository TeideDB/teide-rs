# Teide RS

Rust bindings for the [Teide](https://github.com/TeideDB/teide) columnar dataframe engine, with SQL frontend, interactive REPL, and PostgreSQL wire protocol server.

## Features

- **SQL engine** — parse and execute SQL queries against Teide tables
- **Interactive REPL** — syntax highlighting, tab completion, command history
- **PgWire server** — connect with any PostgreSQL client (`psql`, DBeaver, etc.)
- **FFI bindings** — safe Rust wrappers around the C17 core

## Build

The C core is automatically vendored via `git clone` during build:

```bash
cargo build --all-features
```

## Usage

### REPL

```bash
cargo run --features cli
```

### PostgreSQL Server

```bash
cargo run --features server -- --port 5433
```

Then connect with: `psql -h localhost -p 5433`

## Testing

```bash
cargo test --all-features
```

Includes SQL logic tests (`.slt` files) for comprehensive SQL coverage.

## License

MIT
