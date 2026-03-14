# SQL/PGQ Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ISO SQL/PGQ graph pattern matching to Teide — `CREATE PROPERTY GRAPH`, `GRAPH_TABLE` with `MATCH`, variable-length paths, shortest path, and cyclic pattern detection.

**Architecture:** Custom pre-parser intercepts PGQ syntax before sqlparser (which has no SQL/PGQ support). A new `pgq` module translates MATCH patterns into existing engine graph ops (`expand`, `var_expand`, `shortest_path`, `wco_join`). Property graph metadata lives in the `Session` alongside existing table registry. No C engine changes.

**Tech Stack:** Rust, sqlparser 0.53.0 (DuckDB dialect), existing Teide C engine graph ops (CSR, expand, var_expand, shortest_path, wco_join).

**Design doc:** `docs/plans/graph-sql-proposal.html`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/sql/pgq.rs` | Property graph catalog types (`PropertyGraph`, `VertexLabel`, `EdgeLabel`) and MATCH pattern AST + planner |
| `src/sql/pgq_parser.rs` | Pre-parser for PGQ syntax: intercepts `CREATE PROPERTY GRAPH`, `DROP PROPERTY GRAPH`, and rewrites `GRAPH_TABLE(...)` in FROM clauses |
| `tests/slt/pgq.slt` | SQL logic tests for all PGQ features |
| `tests/slt/pgq_paths.slt` | SQL logic tests for variable-length paths and shortest path |

### Modified Files
| File | Lines | Change |
|------|-------|--------|
| `src/sql/mod.rs:27` | Add `pub mod pgq; pub mod pgq_parser;` |
| `src/sql/mod.rs:105-108` | Add `graphs` field to `Session` |
| `src/sql/mod.rs:112-118` | Init `graphs` in `Session::new()` |
| `src/sql/planner.rs:51-148` | Add match arms in `session_execute()` for CREATE/DROP PROPERTY GRAPH |
| `src/sql/planner.rs:122` | Call pre-parser before sqlparser in `Session::execute()` |
| `src/sql/planner.rs:1628-1664` | Add `GRAPH_TABLE` handling in `resolve_table_factor()` |
| `tests/slt_runner.rs` | Add `slt_pgq` and `slt_pgq_paths` test functions |

---

## Task 1: Property Graph Catalog Types

Define the data structures for storing property graph metadata in the session.

**Files:**
- Create: `src/sql/pgq.rs`
- Modify: `src/sql/mod.rs:27` (add module declaration)
- Modify: `src/sql/mod.rs:30` (add `Rel` import)
- Modify: `src/sql/mod.rs:105-118` (add `graphs` to Session)

- [x] **Step 1: Create `src/sql/pgq.rs` with catalog types**

```rust
// src/sql/pgq.rs
// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use crate::{Rel, Table};
use super::SqlError;

// ---------------------------------------------------------------------------
// Property graph catalog types
// ---------------------------------------------------------------------------

/// A vertex label mapping: label name → session table name.
pub(crate) struct VertexLabel {
    pub table_name: String,
    pub label: String,
}

/// An edge label mapping: label name → edge table with source/dest references.
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
```

- [x] **Step 2: Add module declarations to `src/sql/mod.rs`**

Add after line 28 (`pub mod planner;`):
```rust
pub mod pgq;
pub mod pgq_parser;
```

Add `Rel` to the import at line 30:
```rust
use crate::{Context, Rel, Table};
```

- [x] **Step 3: Add `graphs` field to `Session`**

In `src/sql/mod.rs`, change the Session struct (line 105-108):
```rust
pub struct Session {
    pub(crate) tables: HashMap<String, StoredTable>,
    pub(crate) graphs: HashMap<String, pgq::PropertyGraph>,
    pub(crate) ctx: Context,
}
```

Update `Session::new()` (line 112-118):
```rust
pub fn new() -> Result<Self, SqlError> {
    let ctx = Context::new()?;
    Ok(Session {
        ctx,
        tables: HashMap::new(),
        graphs: HashMap::new(),
    })
}
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: compilation succeeds (pgq_parser.rs doesn't exist yet — create an empty placeholder)

Create `src/sql/pgq_parser.rs`:
```rust
// src/sql/pgq_parser.rs
// Pre-parser for SQL/PGQ syntax.
```

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS (warnings about unused imports are OK)

- [x] **Step 5: Commit**

```bash
git add src/sql/pgq.rs src/sql/pgq_parser.rs src/sql/mod.rs
git commit -m "feat(pgq): add property graph catalog types and Session.graphs field"
```

---

## Task 2: CREATE PROPERTY GRAPH Parser

Parse `CREATE PROPERTY GRAPH` statements. Since sqlparser has no PGQ support, we intercept the SQL string before it reaches sqlparser.

**Files:**
- Modify: `src/sql/pgq_parser.rs`

- [x] **Step 1: Write the CREATE PROPERTY GRAPH parser**

The parser handles this syntax:
```sql
CREATE PROPERTY GRAPH <name>
VERTEX TABLES (
  <table> [LABEL <label>] [, ...]
)
EDGE TABLES (
  <table>
    SOURCE KEY (<col>) REFERENCES <table> (<col>)
    DESTINATION KEY (<col>) REFERENCES <table> (<col>)
    LABEL <label>
  [, ...]
);
```

```rust
// src/sql/pgq_parser.rs
// Pre-parser for SQL/PGQ syntax.
//
// sqlparser 0.53 has no SQL/PGQ support, so we intercept PGQ statements
// before they reach the SQL parser and handle them directly.

use super::SqlError;

// ---------------------------------------------------------------------------
// PGQ statement types (parsed from raw SQL)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct ParsedVertexTable {
    pub table_name: String,
    pub label: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ParsedEdgeTable {
    pub table_name: String,
    pub src_col: String,
    pub src_ref_table: String,
    pub src_ref_col: String,
    pub dst_col: String,
    pub dst_ref_table: String,
    pub dst_ref_col: String,
    pub label: Option<String>,
}

#[derive(Debug)]
pub(crate) struct CreatePropertyGraph {
    pub name: String,
    pub vertex_tables: Vec<ParsedVertexTable>,
    pub edge_tables: Vec<ParsedEdgeTable>,
}

#[derive(Debug)]
pub(crate) enum PgqStatement {
    CreatePropertyGraph(CreatePropertyGraph),
    DropPropertyGraph { name: String, if_exists: bool },
}

// ---------------------------------------------------------------------------
// Pre-parser: detect and extract PGQ statements from raw SQL
// ---------------------------------------------------------------------------

/// Check if a SQL string is a PGQ statement and parse it.
/// Returns None if the SQL is not a PGQ statement (should be passed to sqlparser).
pub(crate) fn try_parse_pgq(sql: &str) -> Result<Option<PgqStatement>, SqlError> {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();

    if upper.starts_with("CREATE PROPERTY GRAPH") {
        return Ok(Some(parse_create_property_graph(trimmed)?));
    }
    if upper.starts_with("DROP PROPERTY GRAPH") {
        return Ok(Some(parse_drop_property_graph(trimmed)?));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Token-based parser helpers
// ---------------------------------------------------------------------------

struct Tokens {
    tokens: Vec<String>,
    pos: usize,
}

impl Tokens {
    fn new(sql: &str) -> Self {
        let tokens = tokenize(sql);
        Tokens { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Result<String, SqlError> {
        if self.pos >= self.tokens.len() {
            return Err(SqlError::Parse("Unexpected end of input".into()));
        }
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        Ok(tok)
    }

    fn expect(&mut self, expected: &str) -> Result<(), SqlError> {
        let tok = self.next()?;
        if tok.to_uppercase() != expected.to_uppercase() {
            return Err(SqlError::Parse(format!(
                "Expected '{expected}', got '{tok}'"
            )));
        }
        Ok(())
    }

    fn expect_upper(&mut self, expected: &str) -> Result<(), SqlError> {
        self.expect(expected)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }
}

/// Tokenize SQL into words and punctuation, respecting parentheses and commas.
fn tokenize(sql: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = sql.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '(' | ')' | ',' | ';' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
                chars.next();
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                chars.next();
            }
            _ => {
                current.push(ch);
                chars.next();
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    // Remove trailing semicolons
    while tokens.last().map(|t| t.as_str()) == Some(";") {
        tokens.pop();
    }
    tokens
}

// ---------------------------------------------------------------------------
// CREATE PROPERTY GRAPH
// ---------------------------------------------------------------------------

fn parse_create_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect_upper("CREATE")?;
    t.expect_upper("PROPERTY")?;
    t.expect_upper("GRAPH")?;
    let name = t.next()?.to_lowercase();

    t.expect_upper("VERTEX")?;
    t.expect_upper("TABLES")?;
    t.expect("(")?;
    let vertex_tables = parse_vertex_tables(&mut t)?;
    t.expect(")")?;

    let mut edge_tables = Vec::new();
    if t.peek().map(|s| s.to_uppercase()) == Some("EDGE".into()) {
        t.expect_upper("EDGE")?;
        t.expect_upper("TABLES")?;
        t.expect("(")?;
        edge_tables = parse_edge_tables(&mut t)?;
        t.expect(")")?;
    }

    Ok(PgqStatement::CreatePropertyGraph(CreatePropertyGraph {
        name,
        vertex_tables,
        edge_tables,
    }))
}

fn parse_vertex_tables(t: &mut Tokens) -> Result<Vec<ParsedVertexTable>, SqlError> {
    let mut tables = Vec::new();
    loop {
        let table_name = t.next()?.to_lowercase();
        let mut label = None;
        if t.peek().map(|s| s.to_uppercase()) == Some("LABEL".into()) {
            t.next()?; // consume LABEL
            label = Some(t.next()?);
        }
        tables.push(ParsedVertexTable { table_name, label });
        if t.peek() == Some(",") {
            t.next()?; // consume comma
        } else {
            break;
        }
    }
    Ok(tables)
}

fn parse_edge_tables(t: &mut Tokens) -> Result<Vec<ParsedEdgeTable>, SqlError> {
    let mut tables = Vec::new();
    loop {
        let table_name = t.next()?.to_lowercase();

        // SOURCE KEY (<col>) REFERENCES <table> (<col>)
        t.expect_upper("SOURCE")?;
        t.expect_upper("KEY")?;
        t.expect("(")?;
        let src_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect_upper("REFERENCES")?;
        let src_ref_table = t.next()?.to_lowercase();
        t.expect("(")?;
        let src_ref_col = t.next()?.to_lowercase();
        t.expect(")")?;

        // DESTINATION KEY (<col>) REFERENCES <table> (<col>)
        t.expect_upper("DESTINATION")?;
        t.expect_upper("KEY")?;
        t.expect("(")?;
        let dst_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect_upper("REFERENCES")?;
        let dst_ref_table = t.next()?.to_lowercase();
        t.expect("(")?;
        let dst_ref_col = t.next()?.to_lowercase();
        t.expect(")")?;

        // LABEL <name>
        let mut label = None;
        if t.peek().map(|s| s.to_uppercase()) == Some("LABEL".into()) {
            t.next()?; // consume LABEL
            label = Some(t.next()?);
        }

        tables.push(ParsedEdgeTable {
            table_name,
            src_col,
            src_ref_table,
            src_ref_col,
            dst_col,
            dst_ref_table,
            dst_ref_col,
            label,
        });

        if t.peek() == Some(",") {
            t.next()?; // consume comma
        } else {
            break;
        }
    }
    Ok(tables)
}

// ---------------------------------------------------------------------------
// DROP PROPERTY GRAPH
// ---------------------------------------------------------------------------

fn parse_drop_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect_upper("DROP")?;
    t.expect_upper("PROPERTY")?;
    t.expect_upper("GRAPH")?;
    let mut if_exists = false;
    let name_or_if = t.next()?;
    let name = if name_or_if.to_uppercase() == "IF" {
        t.expect_upper("EXISTS")?;
        if_exists = true;
        t.next()?.to_lowercase()
    } else {
        name_or_if.to_lowercase()
    };
    Ok(PgqStatement::DropPropertyGraph { name, if_exists })
}
```

- [x] **Step 2: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add src/sql/pgq_parser.rs
git commit -m "feat(pgq): add pre-parser for CREATE/DROP PROPERTY GRAPH"
```

---

## Task 3: CREATE PROPERTY GRAPH Execution

Wire the parsed `CreatePropertyGraph` statement into `session_execute()` to build CSR indexes and register the graph.

**Files:**
- Modify: `src/sql/pgq.rs` (add `build_property_graph` function)
- Modify: `src/sql/planner.rs:51-148` (add PGQ dispatch)
- Modify: `src/sql/mod.rs:120-124` (call pre-parser)

- [x] **Step 1: Add `build_property_graph` to `src/sql/pgq.rs`**

Append to `src/sql/pgq.rs`:

```rust
use super::pgq_parser::{CreatePropertyGraph as ParsedCPG, ParsedEdgeTable, ParsedVertexTable};
use super::{Session, StoredTable};

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
        // Verify table exists
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
        // Verify edge table exists
        let edge_stored = session.tables.get(&et.table_name).ok_or_else(|| {
            SqlError::Plan(format!(
                "Edge table '{}' not found in session",
                et.table_name
            ))
        })?;

        // Verify source and destination tables exist
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

        // Build CSR index from the edge table
        let rel = Rel::from_edges(
            &edge_stored.table,
            &et.src_col,
            &et.dst_col,
            n_src,
            n_dst,
            true, // sort targets for deterministic traversal
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
```

- [x] **Step 2: Add PGQ dispatch in `Session::execute()`**

In `src/sql/mod.rs`, change `Session::execute()` (line 122-124) to call the pre-parser first:

```rust
pub fn execute(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
    // Check for PGQ statements first (sqlparser has no SQL/PGQ support)
    if let Some(pgq_stmt) = pgq_parser::try_parse_pgq(sql)? {
        return pgq::execute_pgq(self, pgq_stmt);
    }
    planner::session_execute(self, sql)
}
```

- [x] **Step 3: Add `execute_pgq` dispatch function to `src/sql/pgq.rs`**

Append to `src/sql/pgq.rs`:

```rust
use super::pgq_parser::PgqStatement;
use super::ExecResult;

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
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Write initial SLT test for CREATE/DROP PROPERTY GRAPH**

Create `tests/slt/pgq.slt`:

```
# SQL/PGQ: Property graph creation and deletion

# Setup: create vertex and edge tables
statement ok
CREATE TABLE persons (id INTEGER, name VARCHAR, city VARCHAR)

statement ok
INSERT INTO persons VALUES (0, 'Alice', 'NYC'), (1, 'Bob', 'SF'), (2, 'Carol', 'NYC'), (3, 'Dave', 'LA'), (4, 'Eve', 'SF')

statement ok
CREATE TABLE knows_edges (src INTEGER, dst INTEGER)

statement ok
INSERT INTO knows_edges VALUES (0, 1), (0, 2), (1, 3), (2, 3), (3, 4)

# Test: CREATE PROPERTY GRAPH
statement ok
CREATE PROPERTY GRAPH social VERTEX TABLES (persons LABEL Person) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows)

# Test: duplicate graph name should fail
statement error
CREATE PROPERTY GRAPH social VERTEX TABLES (persons LABEL Person) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows)

# Test: DROP PROPERTY GRAPH
statement ok
DROP PROPERTY GRAPH social

# Test: DROP non-existent should fail
statement error
DROP PROPERTY GRAPH social

# Test: DROP IF EXISTS non-existent should succeed
statement ok
DROP PROPERTY GRAPH IF EXISTS social

# Test: referencing non-existent table should fail
statement error
CREATE PROPERTY GRAPH bad VERTEX TABLES (nonexistent LABEL Foo) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES nonexistent (id) DESTINATION KEY (dst) REFERENCES nonexistent (id) LABEL Bar)
```

- [x] **Step 6: Add SLT test runner**

Add to `tests/slt_runner.rs`:

```rust
#[test]
fn slt_pgq() {
    run_slt("tests/slt/pgq.slt");
}
```

- [x] **Step 7: Run tests**

Run: `cargo test --all-features -- slt_pgq`
Expected: PASS

- [x] **Step 8: Commit**

```bash
git add src/sql/pgq.rs src/sql/mod.rs tests/slt/pgq.slt tests/slt_runner.rs
git commit -m "feat(pgq): implement CREATE/DROP PROPERTY GRAPH with CSR index building"
```

---

## Task 4: GRAPH_TABLE Pre-Parser

Parse `GRAPH_TABLE(graph_name MATCH pattern COLUMNS (...))` in FROM clauses. Since sqlparser can't parse this, we rewrite it to a placeholder table function before sqlparser sees it, then intercept it in `resolve_table_factor`.

**Files:**
- Modify: `src/sql/pgq_parser.rs` (add GRAPH_TABLE rewriting + MATCH parser)
- Modify: `src/sql/pgq.rs` (add MATCH AST types)

- [x] **Step 1: Add MATCH pattern AST types to `src/sql/pgq.rs`**

Append to `src/sql/pgq.rs`:

```rust
// ---------------------------------------------------------------------------
// MATCH pattern AST
// ---------------------------------------------------------------------------

/// Direction of an edge in a MATCH pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchDirection {
    Forward,   // ->
    Reverse,   // <-
    Undirected, // - (either direction)
}

/// Quantifier on an edge pattern.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PathQuantifier {
    One,                         // (no quantifier) = exactly 1 hop
    Range { min: u8, max: u8 },  // {min,max}
    Plus,                        // + (1 or more)
    Star,                        // * (0 or more)
}

/// A node pattern: (var:Label WHERE condition)
#[derive(Debug, Clone)]
pub(crate) struct NodePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub filter: Option<String>,  // raw SQL predicate text
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
    Walk,         // default: all paths
    AnyShortest,  // ANY SHORTEST
}

/// A parsed MATCH clause.
#[derive(Debug, Clone)]
pub(crate) struct MatchClause {
    pub path_variable: Option<String>,
    pub mode: PathMode,
    pub patterns: Vec<PathPattern>,  // multiple patterns = comma-separated
}

/// A COLUMNS clause entry: expression AS alias.
#[derive(Debug, Clone)]
pub(crate) struct ColumnEntry {
    pub expr: String,   // raw SQL expression (e.g. "a.name", "COUNT(b.id)")
    pub alias: Option<String>,
}

/// A fully parsed GRAPH_TABLE invocation.
#[derive(Debug, Clone)]
pub(crate) struct GraphTableExpr {
    pub graph_name: String,
    pub match_clause: MatchClause,
    pub columns: Vec<ColumnEntry>,
}
```

- [x] **Step 2: Add GRAPH_TABLE pre-parser to `src/sql/pgq_parser.rs`**

Append to `src/sql/pgq_parser.rs`:

```rust
use super::pgq::{
    ColumnEntry, EdgePattern, GraphTableExpr, MatchClause, MatchDirection,
    NodePattern, PathMode, PathPattern, PathQuantifier,
};

// ---------------------------------------------------------------------------
// GRAPH_TABLE rewriting
// ---------------------------------------------------------------------------

/// Scan SQL for GRAPH_TABLE(...) in FROM clauses and extract them.
/// Returns a list of extracted GraphTableExpr and the rewritten SQL
/// (with GRAPH_TABLE replaced by a placeholder like `__pgq_result_0`).
///
/// This is a simple text-level scan. It finds `GRAPH_TABLE` followed by `(`
/// and extracts the balanced parenthesized content.
pub(crate) fn extract_graph_tables(
    sql: &str,
) -> Result<(String, Vec<GraphTableExpr>), SqlError> {
    let upper = sql.to_uppercase();
    let mut result = String::with_capacity(sql.len());
    let mut exprs = Vec::new();
    let mut pos = 0;
    let bytes = sql.as_bytes();

    while pos < bytes.len() {
        // Find next GRAPH_TABLE
        if let Some(gt_pos) = upper[pos..].find("GRAPH_TABLE") {
            let abs_pos = pos + gt_pos;
            // Copy everything before GRAPH_TABLE
            result.push_str(&sql[pos..abs_pos]);

            // Find the opening paren
            let after_gt = abs_pos + "GRAPH_TABLE".len();
            let paren_start = sql[after_gt..]
                .find('(')
                .ok_or_else(|| SqlError::Parse("Expected '(' after GRAPH_TABLE".into()))?
                + after_gt;

            // Find matching closing paren
            let inner_end = find_matching_paren(sql, paren_start)?;

            // Extract the inner content (between parens)
            let inner = &sql[paren_start + 1..inner_end];

            // Parse it
            let expr = parse_graph_table_inner(inner)?;
            let idx = exprs.len();
            exprs.push(expr);

            // Replace with placeholder table name
            result.push_str(&format!("__pgq_result_{idx}"));

            pos = inner_end + 1;
        } else {
            result.push_str(&sql[pos..]);
            break;
        }
    }

    Ok((result, exprs))
}

/// Find the matching closing parenthesis for an opening one at `start`.
fn find_matching_paren(sql: &str, start: usize) -> Result<usize, SqlError> {
    let mut depth = 0;
    let mut in_string = false;
    for (i, ch) in sql[start..].char_indices() {
        match ch {
            '\'' if !in_string => in_string = true,
            '\'' if in_string => in_string = false,
            '(' if !in_string => depth += 1,
            ')' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Ok(start + i);
                }
            }
            _ => {}
        }
    }
    Err(SqlError::Parse("Unmatched parenthesis in GRAPH_TABLE".into()))
}

/// Parse the content inside GRAPH_TABLE(...).
/// Expected: <graph_name> MATCH <pattern> COLUMNS (<col_list>)
fn parse_graph_table_inner(inner: &str) -> Result<GraphTableExpr, SqlError> {
    let mut t = Tokens::new(inner);

    let graph_name = t.next()?.to_lowercase();

    // Parse optional path_var = ANY SHORTEST before MATCH
    let mut path_variable = None;
    let mut mode = PathMode::Walk;

    t.expect_upper("MATCH")?;

    // Check for: p = ANY SHORTEST or just pattern
    // Peek ahead to see if next token is an identifier followed by '='
    let checkpoint = t.pos;
    if let Ok(maybe_var) = t.next() {
        if t.peek() == Some("=") {
            // This is: var = [ANY SHORTEST] pattern
            path_variable = Some(maybe_var.to_lowercase());
            t.next()?; // consume '='
            if t.peek().map(|s| s.to_uppercase()) == Some("ANY".into()) {
                t.next()?; // consume ANY
                t.expect_upper("SHORTEST")?;
                mode = PathMode::AnyShortest;
            }
        } else {
            // Not a var assignment, rewind
            t.pos = checkpoint;
        }
    } else {
        t.pos = checkpoint;
    }

    // Parse patterns (comma-separated path patterns)
    let patterns = parse_match_patterns(&mut t)?;

    // Parse COLUMNS clause
    t.expect_upper("COLUMNS")?;
    t.expect("(")?;
    let columns = parse_columns_clause(&mut t)?;
    t.expect(")")?;

    Ok(GraphTableExpr {
        graph_name,
        match_clause: MatchClause {
            path_variable,
            mode,
            patterns,
        },
        columns,
    })
}

/// Parse comma-separated path patterns.
/// Each pattern is: (node)-[edge]->(node)-[edge]->(node)...
fn parse_match_patterns(t: &mut Tokens) -> Result<Vec<PathPattern>, SqlError> {
    let mut patterns = Vec::new();
    patterns.push(parse_single_path(t)?);

    // Check for comma-separated additional patterns
    while t.peek() == Some(",") {
        // But not if the next pattern is COLUMNS (end of patterns)
        let saved = t.pos;
        t.next()?; // consume comma
        if t.peek().map(|s| s.to_uppercase()) == Some("COLUMNS".into()) {
            t.pos = saved;
            break;
        }
        // Check if next token is '(' which starts a node pattern
        if t.peek() == Some("(") {
            patterns.push(parse_single_path(t)?);
        } else {
            t.pos = saved;
            break;
        }
    }

    Ok(patterns)
}

/// Parse a single path: (node)-[edge]->(node)...
fn parse_single_path(t: &mut Tokens) -> Result<PathPattern, SqlError> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // First node
    nodes.push(parse_node_pattern(t)?);

    // Alternating edge-node pairs
    loop {
        // Check for edge pattern start: - or <
        match t.peek() {
            Some("-") | Some("<") => {
                let (edge, node) = parse_edge_and_node(t)?;
                edges.push(edge);
                nodes.push(node);
            }
            _ => break,
        }
    }

    if nodes.len() < 2 {
        // Single node pattern (no edges) — valid for node-only queries
    }

    Ok(PathPattern { nodes, edges })
}

/// Parse a node pattern: (var:Label WHERE condition)
fn parse_node_pattern(t: &mut Tokens) -> Result<NodePattern, SqlError> {
    t.expect("(")?;

    let mut variable = None;
    let mut label = None;
    let mut filter = None;

    // Could be: empty (), (var), (var:Label), (:Label), (var:Label WHERE ...)
    if t.peek() != Some(")") {
        // First token: variable or :Label
        let first = t.next()?;
        if first == ":" {
            // :Label (no variable)
            label = Some(t.next()?.to_lowercase());
        } else if t.peek() == Some(":") {
            // var:Label
            variable = Some(first.to_lowercase());
            t.next()?; // consume ':'
            // Handle case where ':' might be attached to label
            if t.peek() != Some(")") && t.peek().map(|s| s.to_uppercase()) != Some("WHERE".into()) {
                label = Some(t.next()?.to_lowercase());
            }
        } else {
            // Just a variable name
            variable = Some(first.to_lowercase());
        }

        // Check for WHERE
        if t.peek().map(|s| s.to_uppercase()) == Some("WHERE".into()) {
            t.next()?; // consume WHERE
            // Collect everything until closing paren (respecting nested parens)
            let mut filter_tokens = Vec::new();
            let mut depth = 0;
            loop {
                match t.peek() {
                    Some(")") if depth == 0 => break,
                    Some("(") => {
                        depth += 1;
                        filter_tokens.push(t.next()?);
                    }
                    Some(")") => {
                        depth -= 1;
                        filter_tokens.push(t.next()?);
                    }
                    Some(_) => filter_tokens.push(t.next()?),
                    None => return Err(SqlError::Parse("Unexpected end in node pattern".into())),
                }
            }
            if !filter_tokens.is_empty() {
                filter = Some(filter_tokens.join(" "));
            }
        }
    }

    t.expect(")")?;

    Ok(NodePattern {
        variable,
        label,
        filter,
    })
}

/// Parse an edge pattern and the following node: -[e:Label]->(node)
fn parse_edge_and_node(
    t: &mut Tokens,
) -> Result<(EdgePattern, NodePattern), SqlError> {
    // Determine direction from prefix: - or <-
    let first = t.next()?;
    let starts_reverse = first == "<";
    if starts_reverse {
        // Expect '-' after '<'
        t.expect("-")?;
    }
    // first == "-" for forward/undirected

    // Parse edge details: [var:Label] or empty
    let mut variable = None;
    let mut label = None;
    let mut quantifier = PathQuantifier::One;

    if t.peek() == Some("[") {
        t.next()?; // consume '['

        // Parse contents until ']'
        if t.peek() != Some("]") {
            let first_tok = t.next()?;
            if first_tok == ":" {
                label = Some(t.next()?.to_lowercase());
            } else if t.peek() == Some(":") {
                variable = Some(first_tok.to_lowercase());
                t.next()?; // consume ':'
                if t.peek() != Some("]") {
                    label = Some(t.next()?.to_lowercase());
                }
            } else {
                variable = Some(first_tok.to_lowercase());
            }
        }

        t.expect("]")?;
    }

    // Parse direction suffix: -> or -
    t.expect("-")?;
    let direction = if t.peek() == Some(">") {
        t.next()?; // consume '>'
        if starts_reverse {
            return Err(SqlError::Parse("Invalid edge direction: <-..->".into()));
        }
        MatchDirection::Forward
    } else if starts_reverse {
        MatchDirection::Reverse
    } else {
        MatchDirection::Undirected
    };

    // Parse optional quantifier: +, *, {min,max}
    match t.peek() {
        Some("+") => {
            t.next()?;
            quantifier = PathQuantifier::Plus;
        }
        Some("*") => {
            t.next()?;
            quantifier = PathQuantifier::Star;
        }
        Some("{") => {
            t.next()?; // consume '{'
            let min_str = t.next()?;
            let min: u8 = min_str
                .parse()
                .map_err(|_| SqlError::Parse(format!("Invalid min depth: {min_str}")))?;
            t.expect(",")?;
            let max_str = t.next()?;
            let max: u8 = max_str
                .parse()
                .map_err(|_| SqlError::Parse(format!("Invalid max depth: {max_str}")))?;
            t.expect("}")?;
            quantifier = PathQuantifier::Range { min, max };
        }
        _ => {}
    }

    // Parse following node
    let node = parse_node_pattern(t)?;

    Ok((
        EdgePattern {
            variable,
            label,
            direction,
            quantifier,
        },
        node,
    ))
}

/// Parse COLUMNS clause entries: expr [AS alias], ...
fn parse_columns_clause(t: &mut Tokens) -> Result<Vec<ColumnEntry>, SqlError> {
    let mut entries = Vec::new();
    loop {
        if t.peek() == Some(")") {
            break;
        }

        // Collect expression tokens until AS, comma, or closing paren
        let mut expr_tokens = Vec::new();
        let mut depth = 0;
        loop {
            match t.peek() {
                Some(")") if depth == 0 => break,
                Some(",") if depth == 0 => break,
                Some(tok) if tok.to_uppercase() == "AS" && depth == 0 => break,
                Some("(") => {
                    depth += 1;
                    expr_tokens.push(t.next()?);
                }
                Some(")") => {
                    depth -= 1;
                    expr_tokens.push(t.next()?);
                }
                Some(_) => expr_tokens.push(t.next()?),
                None => return Err(SqlError::Parse("Unexpected end in COLUMNS".into())),
            }
        }

        let expr = expr_tokens.join(" ");
        let mut alias = None;

        if t.peek().map(|s| s.to_uppercase()) == Some("AS".into()) {
            t.next()?; // consume AS
            alias = Some(t.next()?.to_lowercase());
        }

        entries.push(ColumnEntry { expr, alias });

        if t.peek() == Some(",") {
            t.next()?; // consume comma
        }
    }

    Ok(entries)
}
```

- [x] **Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add src/sql/pgq_parser.rs src/sql/pgq.rs
git commit -m "feat(pgq): add GRAPH_TABLE and MATCH pattern pre-parser"
```

---

## Task 5: GRAPH_TABLE Planner — Basic 1-Hop MATCH

Plan and execute basic 1-hop `MATCH (a)-[e]->(b)` patterns by translating them into `graph.expand()` calls.

**Files:**
- Modify: `src/sql/pgq.rs` (add `plan_graph_table` function)
- Modify: `src/sql/mod.rs:120-124` (add GRAPH_TABLE extraction)
- Modify: `src/sql/planner.rs:1628-1664` (handle `__pgq_result_N` placeholder tables)

- [x] **Step 1: Add `plan_graph_table` to `src/sql/pgq.rs`**

This function takes a `GraphTableExpr` and a `Session`, and returns a `(Table, Vec<String>)` — the result table and column names.

Append to `src/sql/pgq.rs`:

```rust
use crate::{Column, Context, Graph};

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

    // For now, support single-pattern matches
    if match_clause.patterns.is_empty() {
        return Err(SqlError::Plan("MATCH requires at least one pattern".into()));
    }

    let pattern = &match_clause.patterns[0];

    match (pattern.nodes.len(), pattern.edges.len(), match_clause.mode) {
        // Single-hop: (a)-[e]->(b)
        (2, 1, PathMode::Walk) => {
            plan_single_hop(session, graph, pattern, &expr.columns)
        }
        // Variable-length: (a)-[e]->{min,max}(b) or -[e]->+(b)
        (2, 1, PathMode::Walk)
            if !matches!(pattern.edges[0].quantifier, PathQuantifier::One) =>
        {
            plan_var_length(session, graph, pattern, &expr.columns)
        }
        // Shortest path: ANY SHORTEST (a)-[e]->+(b)
        (2, 1, PathMode::AnyShortest) => {
            plan_shortest_path(session, graph, pattern, match_clause, &expr.columns)
        }
        _ => {
            // Check if it's a variable-length walk
            if pattern.nodes.len() == 2
                && pattern.edges.len() == 1
                && !matches!(pattern.edges[0].quantifier, PathQuantifier::One)
            {
                plan_var_length(session, graph, pattern, &expr.columns)
            } else if pattern.nodes.len() >= 3 {
                // Multi-hop or cyclic pattern
                Err(SqlError::Plan(
                    "Multi-hop and cyclic MATCH patterns are not yet supported. \
                     Use single-hop (a)-[e]->(b) or variable-length (a)-[e]->{1,3}(b)."
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

    // Resolve edge label → StoredRel
    let edge_label = edge.label.as_deref().ok_or_else(|| {
        SqlError::Plan("Edge pattern must specify a label: -[:Label]->".into())
    })?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label}' not found in graph"))
    })?;

    // Resolve source and destination vertex tables
    let src_label = resolve_node_label(src_node, &stored_rel.edge_label.src_ref_table, graph)?;
    let dst_label = resolve_node_label(dst_node, &stored_rel.edge_label.dst_ref_table, graph)?;

    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;
    let dst_table = &session.tables.get(&dst_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", dst_label.table_name))
    })?.table;

    // Build source node IDs (applying optional WHERE filter)
    let direction: u8 = match edge.direction {
        MatchDirection::Forward => 0,
        MatchDirection::Reverse => 1,
        MatchDirection::Undirected => 2,
    };

    // Create a graph on the source table
    let g = session.ctx.graph(src_table)?;

    // Scan the source reference column (the join key, e.g. "id")
    let src_ref_col = &stored_rel.edge_label.src_ref_col;
    let src_ids = g.scan(src_ref_col)?;

    // Apply source node filter if present
    let src_ids = if let Some(filter_text) = &src_node.filter {
        apply_node_filter(&g, src_table, src_ids, filter_text, src_node.variable.as_deref(), session)?
    } else {
        src_ids
    };

    // Expand
    let expanded = g.expand(src_ids, &stored_rel.rel, direction)?;
    let expand_result = g.execute(expanded)?;

    // expand_result has columns: _src, _dst
    // Now build the final result by projecting the requested COLUMNS
    project_columns(
        session,
        &expand_result,
        columns,
        src_node,
        dst_node,
        edge,
        src_table,
        dst_table,
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
        // Find vertex label by table name
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
    table: &Table,
    ids: Column,
    filter_text: &str,
    variable: Option<&str>,
    session: &Session,
) -> Result<Column, SqlError> {
    // Simple filter parsing: handle "var.col = 'value'" and "var.col = number"
    // Strip the variable prefix (e.g., "a.name" -> "name")
    let clean = if let Some(var) = variable {
        filter_text.replace(&format!("{var}."), "")
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

/// Project COLUMNS from expand/var_expand results.
/// Maps column expressions like "b.name" to lookups in the destination table.
fn project_columns(
    session: &Session,
    expand_result: &Table,
    columns: &[ColumnEntry],
    src_node: &NodePattern,
    dst_node: &NodePattern,
    edge: &EdgePattern,
    src_table: &Table,
    dst_table: &Table,
    edge_label: &EdgeLabel,
) -> Result<(Table, Vec<String>), SqlError> {
    // Build a graph on the expand result to project columns
    let g = session.ctx.graph(expand_result)?;
    let src_table_id = g.add_table(src_table);
    let dst_table_id = g.add_table(dst_table);

    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    let _src_col = g.scan("_src")?;
    let _dst_col = g.scan("_dst")?;

    let mut result_cols: Vec<Column> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        // Parse "var.col" format
        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();

            if var == src_var {
                // Source node property: look up in src_table via _src
                let prop = g.scan_table(src_table_id, &col)?;
                result_cols.push(prop);
                col_names.push(alias.unwrap_or(&col).to_string());
            } else if var == dst_var {
                // Destination node property: look up in dst_table via _dst
                let prop = g.scan_table(dst_table_id, &col)?;
                result_cols.push(prop);
                col_names.push(alias.unwrap_or(&col).to_string());
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

    if result_cols.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Project
    let projected = g.select(result_cols[0], &result_cols)?;
    let result = g.execute(projected)?;
    let result = result.with_column_names(&col_names)?;

    Ok((result, col_names))
}
```

- [x] **Step 2: Wire GRAPH_TABLE extraction into `Session::execute()`**

In `src/sql/mod.rs`, update `Session::execute()`:

```rust
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
```

Add a new method to `Session`:

```rust
/// Execute SQL containing GRAPH_TABLE expressions.
/// Extracts GRAPH_TABLE, executes the graph query, stores the result
/// as a temporary table, rewrites the SQL, and runs the modified SQL.
fn execute_with_graph_table(&mut self, sql: &str) -> Result<ExecResult, SqlError> {
    let (rewritten_sql, graph_exprs) = pgq_parser::extract_graph_tables(sql)?;

    // Execute each GRAPH_TABLE and store results as temp tables
    let mut temp_names = Vec::new();
    for (i, expr) in graph_exprs.iter().enumerate() {
        let temp_name = format!("__pgq_result_{i}");
        let (table, columns) = pgq::plan_graph_table(self, expr)?;
        self.tables.insert(
            temp_name.clone(),
            StoredTable {
                table,
                columns,
            },
        );
        temp_names.push(temp_name);
    }

    // Run the rewritten SQL (which references __pgq_result_N tables)
    let result = planner::session_execute(self, &rewritten_sql);

    // Clean up temp tables
    for name in &temp_names {
        self.tables.remove(name);
    }

    result
}
```

- [x] **Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 4: Add basic GRAPH_TABLE SLT tests**

Append to `tests/slt/pgq.slt`:

```
# Re-create the graph for GRAPH_TABLE tests
statement ok
CREATE PROPERTY GRAPH social VERTEX TABLES (persons LABEL Person) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows)

# Basic 1-hop: who does Alice know?
query TT
SELECT * FROM GRAPH_TABLE (social MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person) COLUMNS (b.name, b.city))
----
Bob SF
Carol NYC

# 1-hop without source filter: all knows relationships
query TT
SELECT * FROM GRAPH_TABLE (social MATCH (a:Person)-[:Knows]->(b:Person) COLUMNS (a.name, b.name)) ORDER BY a.name, b.name
----
Alice Bob
Alice Carol
Bob Dave
Carol Dave
Dave Eve

# Composable with SQL: graph + aggregation
query TI
SELECT name, COUNT(*) AS cnt FROM GRAPH_TABLE (social MATCH (a:Person)-[:Knows]->(b:Person) COLUMNS (a.name AS name)) GROUP BY name ORDER BY cnt DESC
----
Alice 2
Bob 1
Carol 1
Dave 1
```

- [x] **Step 5: Run tests**

Run: `cargo test --all-features -- slt_pgq`
Expected: PASS

- [x] **Step 6: Run full test suite to check for regressions**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All existing tests PASS

- [x] **Step 7: Commit**

```bash
git add src/sql/pgq.rs src/sql/pgq_parser.rs src/sql/mod.rs tests/slt/pgq.slt
git commit -m "feat(pgq): implement GRAPH_TABLE with basic 1-hop MATCH patterns"
```

---

## Task 6: Variable-Length Paths

Add support for path quantifiers: `->{1,3}`, `->+`, `->*`.

**Files:**
- Modify: `src/sql/pgq.rs` (add `plan_var_length` function)
- Create: `tests/slt/pgq_paths.slt`

- [x] **Step 1: Add `plan_var_length` to `src/sql/pgq.rs`**

```rust
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
        apply_node_filter(&g, src_table, src_ids, filter_text, src_node.variable.as_deref(), session)?
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
    let g = session.ctx.graph(var_result)?;
    let src_table_id = g.add_table(src_table);
    let dst_table_id = g.add_table(dst_table);

    let src_var = src_node.variable.as_deref().unwrap_or("__src");
    let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");

    let mut result_cols: Vec<Column> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();

    for entry in columns {
        let expr = &entry.expr;
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = expr.find('.') {
            let var = expr[..dot_pos].trim().to_lowercase();
            let col = expr[dot_pos + 1..].trim().to_lowercase();

            if var == src_var {
                let prop = g.scan_table(src_table_id, &col)?;
                result_cols.push(prop);
                col_names.push(alias.unwrap_or(&col).to_string());
            } else if var == dst_var {
                let prop = g.scan_table(dst_table_id, &col)?;
                result_cols.push(prop);
                col_names.push(alias.unwrap_or(&col).to_string());
            } else {
                return Err(SqlError::Plan(format!(
                    "Unknown variable '{var}' in COLUMNS"
                )));
            }
        } else {
            // Could be a built-in like path_length(p)
            let lower = expr.to_lowercase();
            if lower.contains("path_length") || lower == "_depth" {
                let depth = g.scan("_depth")?;
                result_cols.push(depth);
                col_names.push(alias.unwrap_or("depth").to_string());
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{expr}'"
                )));
            }
        }
    }

    if result_cols.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    let projected = g.select(result_cols[0], &result_cols)?;
    let result = g.execute(projected)?;
    let result = result.with_column_names(&col_names)?;

    Ok((result, col_names))
}
```

- [x] **Step 2: Create `tests/slt/pgq_paths.slt`**

```
# SQL/PGQ: Variable-length paths and shortest path

# Setup
statement ok
CREATE TABLE persons (id INTEGER, name VARCHAR, city VARCHAR)

statement ok
INSERT INTO persons VALUES (0, 'Alice', 'NYC'), (1, 'Bob', 'SF'), (2, 'Carol', 'NYC'), (3, 'Dave', 'LA'), (4, 'Eve', 'SF')

statement ok
CREATE TABLE knows_edges (src INTEGER, dst INTEGER)

statement ok
INSERT INTO knows_edges VALUES (0, 1), (0, 2), (1, 3), (2, 3), (3, 4)

statement ok
CREATE PROPERTY GRAPH social VERTEX TABLES (persons LABEL Person) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows)

# Variable-length: 1-2 hops from Alice
query T
SELECT * FROM GRAPH_TABLE (social MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->{1,2}(b:Person) COLUMNS (b.name)) ORDER BY b.name
----
Bob
Carol
Dave

# Variable-length: exactly 2 hops from Alice (friends-of-friends)
query T
SELECT * FROM GRAPH_TABLE (social MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->{2,2}(b:Person) COLUMNS (b.name)) ORDER BY b.name
----
Dave

# One-or-more hops from Alice (all reachable)
query T
SELECT * FROM GRAPH_TABLE (social MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->+(b:Person) COLUMNS (b.name)) ORDER BY b.name
----
Bob
Carol
Dave
Eve
```

- [x] **Step 3: Add SLT runner**

Add to `tests/slt_runner.rs`:
```rust
#[test]
fn slt_pgq_paths() {
    run_slt("tests/slt/pgq_paths.slt");
}
```

- [x] **Step 4: Run tests**

Run: `cargo test --all-features -- slt_pgq`
Expected: PASS

Run: `cargo test --all-features -- slt_pgq_paths`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_paths.slt tests/slt_runner.rs
git commit -m "feat(pgq): add variable-length path support ({min,max}, +, *)"
```

---

## Task 7: Shortest Path

Add `ANY SHORTEST` path support to MATCH patterns.

**Files:**
- Modify: `src/sql/pgq.rs` (add `plan_shortest_path` function)
- Modify: `tests/slt/pgq_paths.slt`

- [x] **Step 1: Add `plan_shortest_path` to `src/sql/pgq.rs`**

```rust
/// Plan an ANY SHORTEST MATCH.
fn plan_shortest_path(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    match_clause: &MatchClause,
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

    // For shortest path we need specific src and dst node IDs.
    // Extract from WHERE filters.
    let src_id = extract_node_id(src_node, &stored_rel.edge_label.src_ref_table, session)?;
    let dst_id = extract_node_id(dst_node, &stored_rel.edge_label.dst_ref_table, session)?;

    let src_label = resolve_node_label(src_node, &stored_rel.edge_label.src_ref_table, graph)?;
    let src_table = &session.tables.get(&src_label.table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", src_label.table_name))
    })?.table;

    let g = session.ctx.graph(src_table)?;
    let src_const = g.const_i64(src_id)?;
    let dst_const = g.const_i64(dst_id)?;

    let max_depth: u8 = match edge.quantifier {
        PathQuantifier::Range { max, .. } => max,
        _ => 255,
    };

    let sp = g.shortest_path(src_const, dst_const, &stored_rel.rel, max_depth)?;
    let result = g.execute(sp)?;

    // shortest_path result: _node, _depth
    // Project COLUMNS
    let g2 = session.ctx.graph(&result)?;
    let mut result_cols: Vec<Column> = Vec::new();
    let mut col_names: Vec<String> = Vec::new();

    for entry in columns {
        let lower = entry.expr.to_lowercase();
        let alias = entry.alias.as_deref();

        if lower.contains("path_length") {
            // path_length(p) = max depth in result
            let depth = g2.scan("_depth")?;
            result_cols.push(depth);
            col_names.push(alias.unwrap_or("path_length").to_string());
        } else if lower == "_node" || lower == "node" {
            let node = g2.scan("_node")?;
            result_cols.push(node);
            col_names.push(alias.unwrap_or("node").to_string());
        } else if lower == "_depth" || lower == "depth" {
            let depth = g2.scan("_depth")?;
            result_cols.push(depth);
            col_names.push(alias.unwrap_or("depth").to_string());
        } else if let Some(dot_pos) = lower.find('.') {
            let var = lower[..dot_pos].trim();
            let col = lower[dot_pos + 1..].trim();
            let dst_var = dst_node.variable.as_deref().unwrap_or("__dst");
            if var == dst_var {
                // Look up destination property — but shortest_path returns
                // the full path, not just the destination.
                // For now, return _node and _depth directly.
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

    if result_cols.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    let projected = g2.select(result_cols[0], &result_cols)?;
    let result = g2.execute(projected)?;
    let result = result.with_column_names(&col_names)?;

    Ok((result, col_names))
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
        filter.replace(&format!("{var}."), "")
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

    for row in 0..nrows {
        if let Some(sv) = str_val {
            if stored.table.get_str(col_idx, row) == Some(sv) {
                return Ok(row as i64);
            }
        } else if let Ok(iv) = value.parse::<i64>() {
            if stored.table.get_i64(col_idx, row) == Some(iv) {
                return Ok(row as i64);
            }
        }
    }

    Err(SqlError::Plan(format!(
        "No matching row for filter: {filter}"
    )))
}
```

- [x] **Step 2: Add shortest path SLT tests**

Append to `tests/slt/pgq_paths.slt`:

```
# Shortest path from Alice (0) to Eve (4)
query II
SELECT * FROM GRAPH_TABLE (social MATCH p = ANY SHORTEST (a:Person WHERE a.id = 0)-[:Knows]->+(b:Person WHERE b.id = 4) COLUMNS (_node, _depth)) ORDER BY _depth
----
0 0
1 1
3 2
4 3
```

- [x] **Step 3: Run tests**

Run: `cargo test --all-features -- slt_pgq_paths`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_paths.slt
git commit -m "feat(pgq): add ANY SHORTEST path support in MATCH patterns"
```

---

## Task 8: Full Regression Test + Documentation

Run the complete test suite, verify no regressions, and update documentation.

**Files:**
- Modify: `docs/plans/graph-sql-proposal.html` (mark Phase 1+2 as implemented)

- [ ] **Step 1: Run complete test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All tests PASS including new slt_pgq and slt_pgq_paths

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | tail -10`
Expected: No warnings

- [ ] **Step 3: Verify REPL works with PGQ**

Run interactive test:
```bash
echo "CREATE TABLE persons (id INTEGER, name VARCHAR);
INSERT INTO persons VALUES (0, 'Alice'), (1, 'Bob'), (2, 'Carol');
CREATE TABLE edges (src INTEGER, dst INTEGER);
INSERT INTO edges VALUES (0, 1), (0, 2), (1, 2);
CREATE PROPERTY GRAPH g VERTEX TABLES (persons LABEL Person) EDGE TABLES (edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows);
SELECT * FROM GRAPH_TABLE (g MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person) COLUMNS (b.name));" | cargo run --features cli 2>&1
```
Expected: Output shows Bob and Carol

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat(pgq): complete SQL/PGQ Phase 1+2 implementation

Adds CREATE/DROP PROPERTY GRAPH DDL, GRAPH_TABLE with MATCH patterns,
variable-length paths ({min,max}, +, *), and ANY SHORTEST path support.
All graph operations map to existing C engine ops (expand, var_expand,
shortest_path) with no engine changes."
```

---

## Notes for Phase 3 (Future)

Cyclic pattern detection (triangles via `wco_join`) is deferred. It requires:
1. Multi-pattern MATCH with shared variables: `(a)->(b)->(c), (a)->(c)`
2. Pattern analysis to detect cycles and map to `wco_join`
3. This is mechanically straightforward (the engine op exists) but the pattern analysis adds complexity

Graph algorithms (PageRank, connected components) can be added as SQL functions that internally loop over `expand`/`var_expand` until convergence.
