# PGQ Gap-Closing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close all 16 identified gaps between TeideDB and DuckPGQ's SQL/PGQ support, achieving full feature parity.

**Architecture:** Three-layer approach: (A) foundational changes that other features depend on, (B) core features that build on Layer A, (C) independent DDL/syntax features. Each task is TDD: write failing test → implement → verify → commit.

**Tech Stack:** Rust SQL layer (`pgq.rs`, `pgq_parser.rs`, `planner.rs`), C engine libteide (`vendor/teide/`), sqlparser crate.

**Design document:** `docs/plans/2026-03-17-pgq-gap-closing-design.md`

---

## Pre-Implementation: Verify Current State

**Step 1: Run full test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: 250 tests, 0 failures

**Step 2: Create feature branch**

```bash
git checkout -b pgq-gap-closing
```

---

## Task 1: Natural Key Support

**Goal:** Remove the requirement that vertex keys must be 0-based sequential integers.

**Files:**
- Modify: `src/sql/pgq.rs:98-102` — `VertexLabel` struct (add key mapping fields)
- Modify: `src/sql/pgq.rs:217-555` — `build_property_graph` (build key maps, remove validation)
- Modify: `src/sql/pgq.rs:966-996` — `validate_key_column_is_rowid` (replace with new validation)
- Modify: `src/sql/pgq.rs:1566-1640` — `extract_node_id_from_filter` (use key maps for resolution)
- Modify: `src/sql/pgq.rs:851-954` — `project_columns` (reverse-map row indices to user keys)
- Test: `tests/slt/pgq_natural_keys.slt` — new SLT test file

### Step 1.1: Write the failing SLT test

Create `tests/slt/pgq_natural_keys.slt`:

```sql
# SQL/PGQ: Natural key support (non-sequential integer keys)

# Setup: persons with non-sequential IDs
statement ok
CREATE TABLE nk_persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO nk_persons VALUES (100, 'Alice'), (200, 'Bob'), (300, 'Carol')

statement ok
CREATE TABLE nk_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO nk_knows VALUES (100, 200), (200, 300)

statement ok
CREATE PROPERTY GRAPH nk_graph VERTEX TABLES (nk_persons KEY (id) LABEL Person) EDGE TABLES (nk_knows SOURCE KEY (src) REFERENCES nk_persons (id) DESTINATION KEY (dst) REFERENCES nk_persons (id) LABEL Knows)

# 1-hop: Alice knows Bob
query TT
SELECT * FROM GRAPH_TABLE (nk_graph MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person) COLUMNS (a.name, b.name)) ORDER BY b.name
----
Alice Bob
```

Register in `tests/slt_runner.rs`:

```rust
#[test]
fn slt_pgq_natural_keys() {
    run_slt("tests/slt/pgq_natural_keys.slt");
}
```

### Step 1.2: Run test to verify it fails

Run: `cargo test --all-features -- slt_pgq_natural_keys`
Expected: FAIL — current validation rejects non-0-based keys

### Step 1.3: Update `ParsedVertexTable` to accept KEY clause

In `src/sql/pgq_parser.rs`, the `ParsedVertexTable` struct (line 37-39) needs a `key_column` field:

```rust
#[derive(Debug)]
pub(crate) struct ParsedVertexTable {
    pub table_name: String,
    pub label: Option<String>,
    pub key_column: Option<String>,  // NEW: KEY (col) clause
}
```

In `parse_vertex_tables` (line 366-382), after parsing LABEL, parse optional KEY clause:

```rust
// After parsing label...
let mut key_column = None;
if t.peek().map(|s| s.to_uppercase()) == Some("KEY".into()) {
    t.next()?; // consume KEY
    t.expect("(")?;
    key_column = Some(t.next()?.to_lowercase());
    t.expect(")")?;
}
tables.push(ParsedVertexTable { table_name, label, key_column });
```

**Important:** The KEY clause can appear before or after LABEL. Handle both orderings.

### Step 1.4: Add key mapping types to `VertexLabel`

In `src/sql/pgq.rs`, update `VertexLabel` (lines 98-102):

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum KeyValue {
    Int(i64),
    Str(String),
}

pub(crate) struct VertexLabel {
    pub table_name: String,
    pub label: String,
    pub key_column: String,           // NEW
    pub user_to_row: HashMap<KeyValue, usize>,  // NEW: user PK → row index
    pub row_to_user: Vec<KeyValue>,             // NEW: row index → user PK
}
```

### Step 1.5: Build key maps in `build_property_graph`

Replace `validate_key_column_is_rowid` calls with key-map construction. In `build_property_graph` (line 217), after looking up the stored table for each vertex:

```rust
let key_col_name = vt.key_column.as_deref().unwrap_or("id");
let key_col_idx = find_col_idx(&stored.table, key_col_name).ok_or_else(|| {
    SqlError::Plan(format!("Key column '{}' not found in vertex table '{}'", key_col_name, vt.table_name))
})?;
let nrows = checked_logical_nrows(stored)?;
let mut user_to_row = HashMap::with_capacity(nrows);
let mut row_to_user = Vec::with_capacity(nrows);
for row in 0..nrows {
    let key_val = match stored.table.col_type(key_col_idx) {
        ffi::TD_I16 | ffi::TD_I32 | ffi::TD_I64 => {
            let v = stored.table.get_i64(key_col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!("NULL key at row {row} in '{}'", vt.table_name))
            })?;
            KeyValue::Int(v)
        }
        ffi::TD_SYM => {
            let s = stored.table.get_str(key_col_idx, row).ok_or_else(|| {
                SqlError::Plan(format!("NULL key at row {row} in '{}'", vt.table_name))
            })?;
            KeyValue::Str(s)
        }
        t => return Err(SqlError::Plan(format!("Unsupported key column type {t} in '{}'", vt.table_name))),
    };
    if user_to_row.contains_key(&key_val) {
        return Err(SqlError::Plan(format!("Duplicate key {:?} in vertex table '{}'", key_val, vt.table_name)));
    }
    user_to_row.insert(key_val.clone(), row);
    row_to_user.push(key_val);
}
```

### Step 1.6: Remap edge FK values through key maps during CSR construction

In `build_property_graph` where edge tables are processed (around line 298), instead of directly using FK column values as row indices, look them up through `user_to_row`:

For each edge row, resolve `src_col` value → `src_vertex_label.user_to_row[value]` → row index. Use these remapped row indices for `Rel::from_edges`.

**Strategy:** Build a temporary remapped edge table with 0-based row indices, then pass that to `Rel::from_edges`. Alternatively, build the Rel manually from a `Vec<(i64, i64)>` of `(src_row_idx, dst_row_idx)` pairs.

### Step 1.7: Update `extract_node_id_from_filter` to use key maps

Replace the current logic that parses `col = value` and returns the value directly. Instead:
1. Parse the filter as before to get column name and value
2. Look up `value` in the vertex label's `user_to_row` to get the row index
3. Return the row index

### Step 1.8: Update `project_columns` for key column reverse mapping

When projecting `var.col` and the column is the key column, use `row_to_user[row_idx]` to return the original user-facing key value.

### Step 1.9: Run test and iterate

Run: `cargo test --all-features -- slt_pgq_natural_keys`
Expected: PASS

### Step 1.10: Add more SLT tests for natural keys

Add tests to `tests/slt/pgq_natural_keys.slt`:
- String keys (VARCHAR key column)
- Large/negative integer keys
- Multi-hop with natural keys
- Variable-length with natural keys
- Key column projection (SELECT id should show original values)

### Step 1.11: Run full test suite

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: ALL tests pass (including all existing PGQ tests — backward compat)

### Step 1.12: Commit

```bash
git add src/sql/pgq.rs src/sql/pgq_parser.rs tests/slt/pgq_natural_keys.slt tests/slt_runner.rs
git commit -m "feat(pgq): natural key support — remove 0-based sequential key requirement"
```

---

## Task 2: Rich WHERE Filters

**Goal:** Support full expression evaluation in node WHERE clauses (not just `col = value`).

**Files:**
- Modify: `src/sql/pgq.rs:154-160` — `NodePattern` (add `filter_expr` field)
- Modify: `src/sql/pgq.rs` — new `evaluate_filter` function
- Modify: `src/sql/pgq.rs:1566-1640` — replace `extract_node_id_from_filter` callers
- Modify: `src/sql/pgq_parser.rs:766-829` — parse WHERE into `Expr`
- Test: `tests/slt/pgq_filters.slt` — new SLT test file

### Step 2.1: Write the failing SLT test

Create `tests/slt/pgq_filters.slt`:

```sql
# SQL/PGQ: Rich WHERE filters

statement ok
CREATE TABLE rf_persons (id INTEGER, name VARCHAR, age INTEGER)

statement ok
INSERT INTO rf_persons VALUES (0, 'Alice', 30), (1, 'Bob', 25), (2, 'Carol', 35), (3, 'Dave', 25)

statement ok
CREATE TABLE rf_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO rf_knows VALUES (0, 1), (0, 2), (0, 3), (1, 2)

statement ok
CREATE PROPERTY GRAPH rf_graph VERTEX TABLES (rf_persons LABEL Person) EDGE TABLES (rf_knows SOURCE KEY (src) REFERENCES rf_persons (id) DESTINATION KEY (dst) REFERENCES rf_persons (id) LABEL Knows)

# Greater-than filter
query TT
SELECT * FROM GRAPH_TABLE (rf_graph MATCH (a:Person WHERE a.age > 28)-[:Knows]->(b:Person) COLUMNS (a.name, b.name)) ORDER BY a.name, b.name
----
Alice Bob
Alice Carol
Alice Dave
Carol (empty — Carol has no outgoing Knows edges)

# IN filter
query TT
SELECT * FROM GRAPH_TABLE (rf_graph MATCH (a:Person WHERE a.name IN ('Alice', 'Bob'))-[:Knows]->(b:Person) COLUMNS (a.name, b.name)) ORDER BY a.name, b.name
----
Alice Bob
Alice Carol
Alice Dave
Bob Carol

# AND filter
query TT
SELECT * FROM GRAPH_TABLE (rf_graph MATCH (a:Person WHERE a.age >= 25 AND a.name != 'Dave')-[:Knows]->(b:Person) COLUMNS (a.name, b.name)) ORDER BY a.name, b.name
----
Alice Bob
Alice Carol
Alice Dave
Bob Carol
```

Register in `tests/slt_runner.rs`:

```rust
#[test]
fn slt_pgq_filters() {
    run_slt("tests/slt/pgq_filters.slt");
}
```

### Step 2.2: Run test to verify it fails

Run: `cargo test --all-features -- slt_pgq_filters`
Expected: FAIL — "Unsupported operator" or similar error from `extract_node_id_from_filter`

### Step 2.3: Add `filter_expr` to `NodePattern`

In `src/sql/pgq.rs` (line 154):

```rust
pub(crate) struct NodePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub filter: Option<String>,
    pub filter_expr: Option<sqlparser::ast::Expr>,  // NEW
}
```

Add `sqlparser` dependency usage at top of file:
```rust
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser as SqlParser;
```

### Step 2.4: Parse WHERE text into `Expr`

In `src/sql/pgq_parser.rs`, after building the `filter` string in `parse_node_pattern` (line 816-818), also parse it into an Expr. This can be done in `pgq.rs` when the `NodePattern` is first used, or in the parser itself. Preferred: do it in `pgq.rs` at planning time since `pgq_parser.rs` doesn't use sqlparser.

In `pgq.rs`, add a helper:

```rust
fn parse_filter_expr(filter_text: &str) -> Result<sqlparser::ast::Expr, SqlError> {
    let dialect = DuckDbDialect {};
    let mut parser = SqlParser::new(&dialect)
        .try_with_sql(filter_text)
        .map_err(|e| SqlError::Parse(format!("Failed to parse WHERE filter: {e}")))?;
    parser.parse_expr()
        .map_err(|e| SqlError::Parse(format!("Invalid WHERE expression: {e}")))
}
```

### Step 2.5: Implement `evaluate_filter`

New function in `src/sql/pgq.rs`:

```rust
/// Evaluate a parsed filter expression against a table row.
/// Returns true if the row passes the filter.
fn evaluate_filter(
    expr: &sqlparser::ast::Expr,
    table: &Table,
    row: usize,
    var_name: &str,
) -> Result<bool, SqlError> {
    use sqlparser::ast::{Expr, BinaryOperator, UnaryOperator, Value};
    match expr {
        Expr::BinaryOp { left, op, right } => {
            match op {
                BinaryOperator::And => {
                    Ok(evaluate_filter(left, table, row, var_name)?
                        && evaluate_filter(right, table, row, var_name)?)
                }
                BinaryOperator::Or => {
                    Ok(evaluate_filter(left, table, row, var_name)?
                        || evaluate_filter(right, table, row, var_name)?)
                }
                // Comparison operators: evaluate both sides and compare
                _ => {
                    let lhs = eval_scalar(left, table, row, var_name)?;
                    let rhs = eval_scalar(right, table, row, var_name)?;
                    compare_scalars(&lhs, &rhs, op)
                }
            }
        }
        Expr::UnaryOp { op: UnaryOperator::Not, expr: inner } => {
            Ok(!evaluate_filter(inner, table, row, var_name)?)
        }
        Expr::InList { expr: inner, list, negated } => {
            let val = eval_scalar(inner, table, row, var_name)?;
            let found = list.iter().any(|item| {
                eval_scalar(item, table, row, var_name)
                    .map(|v| v == val)
                    .unwrap_or(false)
            });
            Ok(if *negated { !found } else { found })
        }
        Expr::Between { expr: inner, negated, low, high } => {
            let val = eval_scalar(inner, table, row, var_name)?;
            let lo = eval_scalar(low, table, row, var_name)?;
            let hi = eval_scalar(high, table, row, var_name)?;
            let in_range = val >= lo && val <= hi;
            Ok(if *negated { !in_range } else { in_range })
        }
        Expr::IsNull(inner) => {
            Ok(eval_scalar(inner, table, row, var_name).is_err())
        }
        Expr::IsNotNull(inner) => {
            Ok(eval_scalar(inner, table, row, var_name).is_ok())
        }
        _ => Err(SqlError::Plan(format!("Unsupported filter expression: {expr}"))),
    }
}
```

Also implement `eval_scalar` (returns a comparable `ScalarValue` enum) and `compare_scalars`.

### Step 2.6: Replace filter extraction in all planners

In `plan_single_hop`, `plan_var_length`, `plan_multi_hop_fixed`, `plan_multi_hop_variable`, and `plan_shortest_path`:

Where currently calling `extract_node_id_from_filter` for source nodes:
1. Parse the filter text via `parse_filter_expr`
2. For simple equality on the key column, use the fast path (directly resolve via `user_to_row`)
3. For complex expressions, scan all rows in the vertex table and collect matching row indices

```rust
fn resolve_filtered_node_ids(
    filter_text: &str,
    node: &NodePattern,
    vertex_label: &VertexLabel,
    stored: &StoredTable,
) -> Result<Vec<usize>, SqlError> {
    let expr = parse_filter_expr(filter_text)?;
    let var = node.variable.as_deref().unwrap_or("");
    let nrows = checked_logical_nrows(stored)?;
    let mut matching = Vec::new();
    for row in 0..nrows {
        if evaluate_filter(&expr, &stored.table, row, var)? {
            matching.push(row);
        }
    }
    Ok(matching)
}
```

### Step 2.7: Run test and iterate

Run: `cargo test --all-features -- slt_pgq_filters`
Expected: PASS

### Step 2.8: Run full test suite

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: ALL pass

### Step 2.9: Commit

```bash
git add src/sql/pgq.rs src/sql/pgq_parser.rs tests/slt/pgq_filters.slt tests/slt_runner.rs
git commit -m "feat(pgq): rich WHERE filters with full expression evaluation"
```

---

## Task 3: Edge Property Access in COLUMNS

**Goal:** Support `edge_var.col` in COLUMNS to project edge table properties.

**Files:**
- Modify: `src/sql/pgq.rs:119-122` — `StoredRel` (add `edge_row_map`)
- Modify: `src/sql/pgq.rs:217-555` — `build_property_graph` (build edge_row_map)
- Modify: `src/sql/pgq.rs:851-954` — `project_columns` (handle edge variables)
- Test: `tests/slt/pgq_edge_props.slt`

### Step 3.1: Write the failing SLT test

Create `tests/slt/pgq_edge_props.slt`:

```sql
# SQL/PGQ: Edge property access in COLUMNS

statement ok
CREATE TABLE ep_persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO ep_persons VALUES (0, 'Alice'), (1, 'Bob'), (2, 'Carol')

statement ok
CREATE TABLE ep_transfers (src INTEGER, dst INTEGER, amount DOUBLE)

statement ok
INSERT INTO ep_transfers VALUES (0, 1, 100.0), (0, 2, 50.0), (1, 2, 75.0)

statement ok
CREATE PROPERTY GRAPH ep_graph VERTEX TABLES (ep_persons LABEL Person) EDGE TABLES (ep_transfers SOURCE KEY (src) REFERENCES ep_persons (id) DESTINATION KEY (dst) REFERENCES ep_persons (id) LABEL Transfer)

# Access edge property: amount
query TTR
SELECT * FROM GRAPH_TABLE (ep_graph MATCH (a:Person WHERE a.name = 'Alice')-[t:Transfer]->(b:Person) COLUMNS (a.name, b.name, t.amount)) ORDER BY b.name
----
Alice Bob 100.0
Alice Carol 50.0
```

Register in `tests/slt_runner.rs`.

### Step 3.2: Run test to verify it fails

Run: `cargo test --all-features -- slt_pgq_edge_props`
Expected: FAIL — "Unknown variable 't'" error

### Step 3.3: Add `edge_row_map` to `StoredRel`

In `src/sql/pgq.rs` (line 119):

```rust
pub(crate) struct StoredRel {
    pub rel: Rel,
    pub edge_label: EdgeLabel,
    pub edge_row_map: HashMap<(i64, i64), Vec<usize>>,  // (src_row, dst_row) → edge row(s)
}
```

### Step 3.4: Build `edge_row_map` in `build_property_graph`

After building the Rel for each edge table, also build the edge_row_map:

```rust
let mut edge_row_map: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
for row in 0..n_edges {
    let src_row = /* remapped src row index */;
    let dst_row = /* remapped dst row index */;
    edge_row_map.entry((src_row as i64, dst_row as i64))
        .or_default()
        .push(row);
}
```

### Step 3.5: Update `project_columns` to handle edge variables

In `project_columns` (line 851), after checking for `src_var` and `dst_var` in the dot-expression, add:

```rust
// Check if variable matches an edge pattern variable
let edge_vars: HashMap<&str, &EdgePattern> = pattern.edges.iter()
    .filter_map(|e| e.variable.as_deref().map(|v| (v, e)))
    .collect();

if let Some(edge_pattern) = edge_vars.get(var.as_str()) {
    let edge_label = edge_pattern.label.as_deref().ok_or(...)?;
    let stored_rel = graph.edge_labels.get(edge_label).ok_or(...)?;
    let edge_table = session.tables.get(&stored_rel.edge_label.table_name).ok_or(...)?;
    let col_idx = find_col_idx(&edge_table.table, &col).ok_or(...)?;
    // For each result row, look up (src_row, dst_row) → edge_row_map → edge table row
    // ...
}
```

**Note:** `project_columns` currently doesn't take `pattern` or `graph` as parameters. These need to be threaded through.

### Step 3.6: Run test and iterate

Run: `cargo test --all-features -- slt_pgq_edge_props`
Expected: PASS

### Step 3.7: Add multi-hop edge property test, run full suite

Test edge properties in multi-hop patterns. Run full suite.

### Step 3.8: Commit

```bash
git add src/sql/pgq.rs tests/slt/pgq_edge_props.slt tests/slt_runner.rs
git commit -m "feat(pgq): edge property access in COLUMNS projection"
```

---

## Task 4: LIST Column Type

**Goal:** Expose `TD_LIST` through Rust layer and SQL. Needed for path accessor functions.

**Files:**
- Modify: `src/ffi.rs:691-694` — verify existing bindings
- Modify: `src/engine.rs` — add LIST wrappers (`get_list_i64`, `create_list`)
- Modify: `tests/slt_runner.rs:80-130` — add type 0 formatting
- Modify: `tests/engine_api.rs` — Rust API test for LIST
- Test: `tests/slt/pgq_list.slt`

### Step 4.1: Write Rust API test for LIST

In `tests/engine_api.rs`, add:

```rust
#[test]
fn test_list_create_and_read() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let ctx = Context::new().unwrap();
    // Create a list of i64 values
    let list = ctx.list_new(4).unwrap();
    // ... verify list operations
}
```

### Step 4.2: Run test to verify it fails

Run: `cargo test --all-features -- test_list_create_and_read`
Expected: FAIL — `list_new` not defined

### Step 4.3: Add safe LIST wrappers in `engine.rs`

```rust
/// RAII wrapper for a TD_LIST.
pub struct List {
    raw: *mut ffi::td_t,
    _engine: Arc<EngineGuard>,
}

impl List {
    pub fn new(capacity: i64) -> Result<Self> { ... }
    pub fn append_i64(&mut self, value: i64) -> Result<()> { ... }
    pub fn get_i64(&self, idx: i64) -> Option<i64> { ... }
    pub fn len(&self) -> i64 { ... }
}
```

### Step 4.4: Run test

Run: `cargo test --all-features -- test_list_create_and_read`
Expected: PASS

### Step 4.5: Add LIST formatting in SLT runner

In `tests/slt_runner.rs`, `format_cell` function (line 80), add case for type 0:

```rust
0 => {
    // TD_LIST: format as [elem1, elem2, ...]
    // Read list pointer and iterate elements
    // For now, format elements as integers
    format_list_cell(table, col, row)
}
```

This requires access to the raw `td_t*` and list iteration. Add a helper in `engine.rs`:

```rust
impl Table {
    /// Format a LIST cell as a string: "[1, 2, 3]"
    pub fn format_list(&self, col: usize, row: usize) -> Option<String> { ... }
}
```

### Step 4.6: Write SLT test

Create `tests/slt/pgq_list.slt` with a basic test that verifies LIST columns display correctly. This will be used later by path accessor functions.

### Step 4.7: Run full test suite and commit

```bash
git add src/engine.rs src/ffi.rs tests/slt_runner.rs tests/engine_api.rs tests/slt/pgq_list.slt tests/slt_runner.rs
git commit -m "feat: expose TD_LIST through Rust layer with safe wrappers and SLT formatting"
```

---

## Task 5: WHERE on Destination/Intermediate Nodes

**Goal:** Allow WHERE filters on non-source nodes via post-filtering.

**Files:**
- Modify: `src/sql/pgq.rs:647-654` — remove rejection in `plan_single_hop`
- Modify: `src/sql/pgq.rs` — remove rejections in all other planners
- Modify: `src/sql/pgq.rs` — add post-filter logic after execution
- Test: `tests/slt/pgq_dst_filters.slt`

### Step 5.1: Write the failing SLT test

Create `tests/slt/pgq_dst_filters.slt`:

```sql
# SQL/PGQ: WHERE on destination/intermediate nodes

statement ok
CREATE TABLE df_persons (id INTEGER, name VARCHAR, city VARCHAR)

statement ok
INSERT INTO df_persons VALUES (0, 'Alice', 'NYC'), (1, 'Bob', 'SF'), (2, 'Carol', 'NYC'), (3, 'Dave', 'LA')

statement ok
CREATE TABLE df_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO df_knows VALUES (0, 1), (0, 2), (0, 3), (1, 2)

statement ok
CREATE PROPERTY GRAPH df_graph VERTEX TABLES (df_persons LABEL Person) EDGE TABLES (df_knows SOURCE KEY (src) REFERENCES df_persons (id) DESTINATION KEY (dst) REFERENCES df_persons (id) LABEL Knows)

# Destination filter: Alice knows someone in NYC
query TT
SELECT * FROM GRAPH_TABLE (df_graph MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person WHERE b.city = 'NYC') COLUMNS (a.name, b.name)) ORDER BY b.name
----
Alice Carol

# Intermediate node filter in multi-hop
query TTT
SELECT * FROM GRAPH_TABLE (df_graph MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person WHERE b.city = 'SF')-[:Knows]->(c:Person) COLUMNS (a.name, b.name, c.name)) ORDER BY c.name
----
Alice Bob Carol
```

Register in `tests/slt_runner.rs`.

### Step 5.2: Run test to verify it fails

Run: `cargo test --all-features -- slt_pgq_dst_filters`
Expected: FAIL — "WHERE filters on destination nodes are not yet supported"

### Step 5.3: Remove rejection and add post-filter

In `plan_single_hop` (line 647-654), remove the `dst_node.filter.is_some()` rejection.

After executing the expand and getting results, add post-filter:

```rust
// Post-filter destination nodes
if let Some(filter_text) = &dst_node.filter {
    let expr = parse_filter_expr(filter_text)?;
    let dst_var = dst_node.variable.as_deref().unwrap_or("");
    // Filter rows where destination doesn't match
    // Rebuild result with only matching rows
}
```

Apply same pattern to `plan_multi_hop_fixed`, `plan_multi_hop_variable`, `plan_var_length`.

### Step 5.4: Run test and iterate

Run: `cargo test --all-features -- slt_pgq_dst_filters`
Expected: PASS

### Step 5.5: Run full suite and commit

```bash
git add src/sql/pgq.rs tests/slt/pgq_dst_filters.slt tests/slt_runner.rs
git commit -m "feat(pgq): WHERE filters on destination and intermediate nodes"
```

---

## Task 6: CHEAPEST Path (COST Expression)

**Goal:** Support `COST <expr>` on edges for Dijkstra-based shortest weighted paths.

**Files:**
- Modify: `src/sql/pgq.rs:164-170` — `EdgePattern` (add `cost_expr`)
- Modify: `src/sql/pgq_parser.rs:832-923` — parse COST in edge brackets
- Modify: `src/sql/pgq.rs` — new `plan_cheapest_path` function
- Test: `tests/slt/pgq_cheapest.slt`

### Step 6.1: Write the failing SLT test

Create `tests/slt/pgq_cheapest.slt`:

```sql
# SQL/PGQ: CHEAPEST path with COST expression

statement ok
CREATE TABLE cp_cities (id INTEGER, name VARCHAR)

statement ok
INSERT INTO cp_cities VALUES (0, 'NYC'), (1, 'Chicago'), (2, 'Denver'), (3, 'SF')

statement ok
CREATE TABLE cp_routes (src INTEGER, dst INTEGER, distance DOUBLE)

statement ok
INSERT INTO cp_routes VALUES (0, 1, 790.0), (1, 2, 920.0), (2, 3, 1240.0), (0, 3, 2900.0)

statement ok
CREATE PROPERTY GRAPH cp_graph VERTEX TABLES (cp_cities LABEL City) EDGE TABLES (cp_routes SOURCE KEY (src) REFERENCES cp_cities (id) DESTINATION KEY (dst) REFERENCES cp_cities (id) LABEL Route)

# Cheapest path NYC to SF
query TTR
SELECT * FROM GRAPH_TABLE (cp_graph MATCH p = ANY SHORTEST (a:City WHERE a.name = 'NYC')-[r:Route COST r.distance]->+(b:City WHERE b.name = 'SF') COLUMNS (a.name, b.name, path_cost(p))) ORDER BY 1
----
NYC SF 2950.0
```

Register in `tests/slt_runner.rs`.

### Step 6.2: Run test to verify it fails

Run: `cargo test --all-features -- slt_pgq_cheapest`
Expected: FAIL — parse error on COST keyword

### Step 6.3: Add `cost_expr` to `EdgePattern`

In `src/sql/pgq.rs` (line 164):

```rust
pub(crate) struct EdgePattern {
    pub variable: Option<String>,
    pub label: Option<String>,
    pub direction: MatchDirection,
    pub quantifier: PathQuantifier,
    pub cost_expr: Option<String>,  // NEW: raw COST expression text
}
```

### Step 6.4: Parse COST in edge brackets

In `src/sql/pgq_parser.rs`, `parse_edge_and_node` (line 832), after parsing the edge label and before `t.expect("]")`, check for COST:

```rust
let mut cost_expr = None;
if t.peek().map(|s| s.to_uppercase()) == Some("COST".into()) {
    t.next()?; // consume COST
    let mut expr_tokens = Vec::new();
    while t.peek() != Some("]") {
        expr_tokens.push(t.next()?);
    }
    if !expr_tokens.is_empty() {
        cost_expr = Some(expr_tokens.join(" "));
    }
}
```

### Step 6.5: Implement `plan_cheapest_path`

New function that:
1. Resolves source/destination from WHERE filters
2. Evaluates cost expression to build a weight vector
3. Attaches weights to Rel via `Rel::set_props`
4. Calls `Graph::dijkstra`
5. Reconstructs path and projects COLUMNS including `path_cost(p)`

### Step 6.6: Wire into `plan_graph_table` dispatcher

Add detection: if any edge has `cost_expr.is_some()`, route to `plan_cheapest_path`.

### Step 6.7: Run test and iterate

Run: `cargo test --all-features -- slt_pgq_cheapest`
Expected: PASS

### Step 6.8: Run full suite and commit

```bash
git add src/sql/pgq.rs src/sql/pgq_parser.rs tests/slt/pgq_cheapest.slt tests/slt_runner.rs
git commit -m "feat(pgq): CHEAPEST path with COST expression via Dijkstra"
```

---

## Task 7: Property Lookups on Path Nodes

**Goal:** Allow `var.col` in shortest-path COLUMNS instead of only `_node`/`_depth`.

**Files:**
- Modify: `src/sql/pgq.rs` — update shortest-path COLUMNS projection
- Test: `tests/slt/pgq_paths.slt` — extend existing test

### Step 7.1: Write the failing test

Add to `tests/slt/pgq_paths.slt`:

```sql
# Property lookup on shortest path nodes
query TT
SELECT * FROM GRAPH_TABLE (sp_graph MATCH path = ANY SHORTEST (a:Person WHERE a.name = 'Alice')-[:Knows]->+(b:Person) COLUMNS (a.name, b.name)) ORDER BY b.name
----
Alice Bob
Alice Carol
```

### Step 7.2: Run test to verify it fails

Expected: FAIL — "Property lookups on path nodes are not yet supported"

### Step 7.3: Remove rejection and implement lookup

Find the error message in pgq.rs and replace with actual property lookup logic:
- Determine which vertex table the variable refers to
- Use the node ID from the BFS/shortest-path result as a row index
- Look up the property column in that vertex table

### Step 7.4: Run test and full suite, commit

```bash
git commit -m "feat(pgq): property lookups on shortest-path nodes"
```

---

## Task 8: Path Accessor Functions

**Goal:** Implement `vertices(p)`, `edges(p)`, `element_id(p)` returning LIST columns.

**Files:**
- Modify: `src/sql/pgq.rs` — COLUMNS projection in variable-length and shortest-path planners
- Modify: `src/engine.rs` — helper to build LIST columns for result tables
- Test: `tests/slt/pgq_path_accessors.slt`

### Step 8.1: Write the failing SLT test

Create `tests/slt/pgq_path_accessors.slt`:

```sql
# SQL/PGQ: Path accessor functions

statement ok
CREATE TABLE pa_persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO pa_persons VALUES (0, 'Alice'), (1, 'Bob'), (2, 'Carol')

statement ok
CREATE TABLE pa_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO pa_knows VALUES (0, 1), (1, 2)

statement ok
CREATE PROPERTY GRAPH pa_graph VERTEX TABLES (pa_persons LABEL Person) EDGE TABLES (pa_knows SOURCE KEY (src) REFERENCES pa_persons (id) DESTINATION KEY (dst) REFERENCES pa_persons (id) LABEL Knows)

# vertices(p): list of vertex IDs along the path
query T
SELECT * FROM GRAPH_TABLE (pa_graph MATCH p = ANY SHORTEST (a:Person WHERE a.name = 'Alice')-[:Knows]->+(b:Person WHERE b.name = 'Carol') COLUMNS (vertices(p) AS v))
----
[0, 1, 2]
```

### Step 8.2: Run, fail, implement, verify, commit

Implementation: For each result path, build a `TD_LIST` via FFI containing the vertex IDs at each position. Return a column of type `TD_LIST`.

```bash
git commit -m "feat(pgq): path accessor functions — vertices(), edges(), element_id()"
```

---

## Task 9: PROPERTIES Clause in DDL

**Goal:** Support `PROPERTIES (col1, col2)` / `NO PROPERTIES` / `ARE ALL COLUMNS EXCEPT (...)`.

**Files:**
- Modify: `src/sql/pgq_parser.rs:335-439` — parse PROPERTIES variants in vertex/edge tables
- Modify: `src/sql/pgq.rs:98-130` — add `ColumnVisibility` to labels
- Modify: `src/sql/pgq.rs` — validate property access in COLUMNS projection
- Test: `tests/slt/pgq_properties.slt`

### Step 9.1: Write the failing SLT test

```sql
# SQL/PGQ: PROPERTIES clause

statement ok
CREATE TABLE pr_persons (id INTEGER, name VARCHAR, ssn VARCHAR)

statement ok
INSERT INTO pr_persons VALUES (0, 'Alice', '123-45-6789'), (1, 'Bob', '987-65-4321')

statement ok
CREATE TABLE pr_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO pr_knows VALUES (0, 1)

statement ok
CREATE PROPERTY GRAPH pr_graph VERTEX TABLES (pr_persons LABEL Person PROPERTIES (name)) EDGE TABLES (pr_knows SOURCE KEY (src) REFERENCES pr_persons (id) DESTINATION KEY (dst) REFERENCES pr_persons (id) LABEL Knows)

# Can access visible property
query TT
SELECT * FROM GRAPH_TABLE (pr_graph MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person) COLUMNS (a.name, b.name))
----
Alice Bob

# Cannot access hidden property (ssn not in PROPERTIES)
statement error
SELECT * FROM GRAPH_TABLE (pr_graph MATCH (a:Person)-[:Knows]->(b:Person) COLUMNS (a.ssn, b.name))
```

### Step 9.2-9.5: Implement, test, commit

Add `ColumnVisibility` enum and `visible_columns` to `VertexLabel`/`EdgeLabel`. Parse `PROPERTIES (...)` / `NO PROPERTIES` / `ARE ALL COLUMNS EXCEPT (...)` in the parser. Validate in COLUMNS projection.

```bash
git commit -m "feat(pgq): PROPERTIES clause for column visibility in DDL"
```

---

## Task 10: CREATE OR REPLACE / IF NOT EXISTS

**Files:**
- Modify: `src/sql/pgq_parser.rs:335-364` — detect OR REPLACE / IF NOT EXISTS tokens
- Modify: `src/sql/pgq.rs` — logic in `session_execute` for create_property_graph
- Test: `tests/slt/pgq_ddl.slt` — extend existing `ddl.slt` or new file

### Step 10.1: Write failing SLT test

```sql
# CREATE OR REPLACE: should succeed even if graph exists
statement ok
CREATE OR REPLACE PROPERTY GRAPH test_graph ...

# IF NOT EXISTS: should silently succeed
statement ok
CREATE PROPERTY GRAPH IF NOT EXISTS test_graph ...
```

### Step 10.2-10.4: Implement, test, commit

```bash
git commit -m "feat(pgq): CREATE OR REPLACE and IF NOT EXISTS for property graphs"
```

---

## Task 11: DESCRIBE PROPERTY GRAPH

**Files:**
- Modify: `src/sql/pgq_parser.rs` — detect `DESCRIBE PROPERTY GRAPH <name>`
- Modify: `src/sql/pgq.rs` — new `describe_property_graph` function
- Test: `tests/slt/pgq_describe.slt`

### Step 11.1: Write failing SLT test

```sql
statement ok
DESCRIBE PROPERTY GRAPH some_graph

# Expected output: table with element_type, table_name, label, etc.
```

### Step 11.2-11.4: Implement, test, commit

Build a result table with columns: `element_type`, `table_name`, `label`, `key_column`, `src_table`, `dst_table`.

```bash
git commit -m "feat(pgq): DESCRIBE PROPERTY GRAPH statement"
```

---

## Task 12: Label Expressions

**Goal:** Support `(n:Person|Company)` — OR-union of vertex labels.

**Files:**
- Modify: `src/sql/pgq_parser.rs:766-829` — parse `|` in label position
- Modify: `src/sql/pgq.rs` — resolve label set, run per-table, union results
- Test: `tests/slt/pgq_label_expr.slt`

### Step 12.1: Write failing SLT test

```sql
# Label expression: Person|City
query T
SELECT * FROM GRAPH_TABLE (some_graph MATCH (n:Person|City) COLUMNS (n.name)) ORDER BY n.name
----
Alice
Bob
NYC
SF
```

### Step 12.2-12.4: Implement, test, commit

Parse `Label1|Label2` into `LabelExpr::Or(Vec<String>)`. In each planner, resolve all matching vertex tables, run the pattern for each, union results.

```bash
git commit -m "feat(pgq): label expressions with OR union (Person|Company)"
```

---

## Task 13: Multiple MATCH Patterns

**Goal:** Support comma-separated patterns joined on shared variables.

**Files:**
- Modify: `src/sql/pgq.rs:580-586` — remove rejection
- Modify: `src/sql/pgq.rs:563-634` — handle `patterns.len() > 1`
- Test: `tests/slt/pgq_multi_match.slt`

### Step 13.1: Write failing SLT test

```sql
# Multiple patterns: comma-separated, joined on shared variable 'b'
query TTT
SELECT * FROM GRAPH_TABLE (some_graph MATCH (a:Person)-[:Knows]->(b:Person), (b:Person)-[:LivesIn]->(c:City) COLUMNS (a.name, b.name, c.city))
----
(same as 2-hop)
```

### Step 13.2-13.4: Implement, test, commit

Execute each pattern independently, then join on shared variable names using hash join.

```bash
git commit -m "feat(pgq): multiple comma-separated MATCH patterns"
```

---

## Task 14: Bidirectional Edges

**Goal:** Support `<-[:Label]->` meaning edge exists in BOTH directions.

**Files:**
- Modify: `src/sql/pgq_parser.rs:866-878` — detect `<-..->` pattern
- Modify: `src/sql/pgq.rs` — intersection logic
- Test: `tests/slt/pgq_bidir.slt`

### Step 14.1: Write failing SLT test

```sql
# Bidirectional: edge must exist in both directions
query TT
SELECT * FROM GRAPH_TABLE (cycle_graph MATCH (a:Person WHERE a.name = 'Alice')<-[:Knows]->(b:Person) COLUMNS (a.name, b.name))
----
Alice Bob  (if both Alice→Bob and Bob→Alice exist)
```

### Step 14.2: Fix parser

Currently `<-..->` returns error "Invalid edge direction: <-..->". Change to detect it as `MatchDirection::Bidirectional`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchDirection {
    Forward,
    Reverse,
    Undirected,
    Bidirectional,  // NEW
}
```

In `parse_edge_and_node`, the case `starts_reverse && peek == ">"`:

```rust
let direction = if t.peek() == Some(">") {
    t.next()?;
    if starts_reverse {
        MatchDirection::Bidirectional  // <-[...]->
    } else {
        MatchDirection::Forward  // -[...]->
    }
} else if starts_reverse {
    MatchDirection::Reverse  // <-[...]-
} else {
    MatchDirection::Undirected  // -[...]-
};
```

### Step 14.3-14.4: Implement intersection in planner, test, commit

For bidirectional: expand forward, expand reverse, intersect result sets.

```bash
git commit -m "feat(pgq): bidirectional edge patterns <-[:Label]->"
```

---

## Task 15: Quantifier Shorthands

**Goal:** Support `{,m}` → `{0,m}` and `{n,}` → `{n,255}`.

**Files:**
- Modify: `src/sql/pgq_parser.rs:890-910` — quantifier parsing
- Test: `tests/slt/pgq_quantifiers.slt`

### Step 15.1: Write failing SLT test

```sql
# {,3} means {0,3}
query TI
SELECT * FROM GRAPH_TABLE (some_graph MATCH (a:Person)-[:Knows]->{,3}(b:Person) COLUMNS (b.name, path_length(a, b))) ORDER BY 1, 2
----
...

# {2,} means {2,255}
query TI
SELECT * FROM GRAPH_TABLE (some_graph MATCH (a:Person)-[:Knows]->{2,}(b:Person) COLUMNS (b.name, path_length(a, b))) ORDER BY 1, 2
----
...
```

### Step 15.2: Fix parser

In `parse_edge_and_node`, the `{min,max}` case (line 890):

```rust
Some("{") => {
    t.next()?; // consume '{'
    let first = t.peek().ok_or_else(|| SqlError::Parse("Unexpected end in quantifier".into()))?;
    if first == "," {
        // {,max} shorthand
        t.next()?; // consume ','
        let max_str = t.next()?;
        let max: u8 = max_str.parse().map_err(|_| ...)?;
        t.expect("}")?;
        quantifier = PathQuantifier::Range { min: 0, max };
    } else {
        let min_str = t.next()?;
        let min: u8 = min_str.parse().map_err(|_| ...)?;
        t.expect(",")?;
        let next = t.peek().ok_or_else(|| ...)?;
        if next == "}" {
            // {n,} shorthand
            t.next()?; // consume '}'
            quantifier = PathQuantifier::Range { min, max: 255 };
        } else {
            let max_str = t.next()?;
            let max: u8 = max_str.parse().map_err(|_| ...)?;
            t.expect("}")?;
            if min > max { return Err(...); }
            quantifier = PathQuantifier::Range { min, max };
        }
    }
}
```

### Step 15.3: Run test and commit

```bash
git commit -m "feat(pgq): quantifier shorthands {,m} and {n,}"
```

---

## Task 16: Local Clustering Coefficient

**Goal:** New graph algorithm kernel in libteide C + Rust exposure + SQL function.

**Files:**
- Modify: `vendor/teide/include/teide/td.h` — declare `td_local_clustering_coeff`
- Modify: `vendor/teide/src/ops/graph.c` (or new file) — implement kernel
- Modify: `src/ffi.rs` — FFI binding
- Modify: `src/engine.rs` — safe wrapper `Graph::clustering_coeff`
- Modify: `src/sql/pgq.rs:2669` — add to algorithm dispatch
- Test: `tests/slt/pgq_algorithms.slt` — extend
- Test: `tests/engine_api.rs` — Rust API test

### Step 16.1: Write failing Rust API test

In `tests/engine_api.rs`:

```rust
#[test]
fn test_clustering_coefficient() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let ctx = Context::new().unwrap();
    // Build triangle: 0→1, 1→2, 2→0, 0→2, 1→0, 2→1
    // All nodes should have LCC = 1.0 (complete graph)
    // ...
}
```

### Step 16.2: Run test to verify it fails

### Step 16.3: Implement C kernel

In `vendor/teide/src/ops/graph.c` (find the right file — it may be `exec.c`):

```c
td_t* td_local_clustering_coeff(td_rel_t* rel) {
    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    int64_t n = td_rel_n_nodes(rel);
    // Allocate result arrays
    int64_t* nodes = td_scratch_arena_push(&arena, n * sizeof(int64_t));
    double* lcc = td_scratch_arena_push(&arena, n * sizeof(double));

    for (int64_t v = 0; v < n; v++) {
        nodes[v] = v;
        int64_t deg = td_rel_degree(rel, v);
        if (deg < 2) {
            lcc[v] = 0.0;
            continue;
        }
        // Get neighbors
        const int64_t* nbrs = td_rel_neighbors(rel, v);
        int64_t triangles = 0;
        for (int64_t i = 0; i < deg; i++) {
            for (int64_t j = i + 1; j < deg; j++) {
                // Check if edge (nbrs[i], nbrs[j]) exists
                if (td_rel_has_edge(rel, nbrs[i], nbrs[j])) {
                    triangles++;
                }
            }
        }
        lcc[v] = (2.0 * triangles) / ((double)deg * (deg - 1));
    }

    // Build result table with _node and _lcc columns
    td_t* result = td_table_new();
    // ... add columns from arena data ...

    td_scratch_arena_reset(&arena);
    return result;
}
```

**Note:** The exact C API functions (`td_rel_degree`, `td_rel_neighbors`, `td_rel_has_edge`) need to be verified against the actual CSR API in `include/teide/td.h`. The CSR stores adjacency lists — binary search within a neighbor list checks edge existence.

### Step 16.4: Declare in `td.h`

```c
td_t* td_local_clustering_coeff(td_rel_t* rel);
```

### Step 16.5: Add FFI binding

In `src/ffi.rs`:

```rust
pub fn td_local_clustering_coeff(rel: *mut td_rel_t) -> *mut td_t;
```

### Step 16.6: Add safe wrapper

In `src/engine.rs`:

```rust
impl<'a> Graph<'a> {
    pub fn clustering_coeff(&self, rel: &Rel) -> Result<Table> {
        let result = unsafe { ffi::td_local_clustering_coeff(rel.as_ptr()) };
        // ... wrap result ...
    }
}
```

### Step 16.7: Wire into SQL algorithm dispatch

In `src/sql/pgq.rs`, `plan_algorithm_query` (line 2669), add:

```rust
"clustering_coefficient" | "local_clustering_coeff" => {
    // Similar to pagerank/louvain dispatch
}
```

### Step 16.8: Add SLT test

Extend `tests/slt/pgq_algorithms.slt`:

```sql
# Clustering coefficient on triangle graph
query IR
SELECT * FROM GRAPH_TABLE (algo_graph MATCH (n:Person) COLUMNS (n.id, CLUSTERING_COEFFICIENT(algo_graph, n) AS lcc)) ORDER BY n.id
----
...
```

### Step 16.9: Run full suite and commit

```bash
git add vendor/teide/ src/ffi.rs src/engine.rs src/sql/pgq.rs tests/
git commit -m "feat: local clustering coefficient algorithm (libteide kernel + SQL function)"
```

---

## Final Verification

### Step F.1: Run full test suite

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: ALL pass (250+ original + new tests)

### Step F.2: Run SLT tests specifically

Run: `cargo test --all-features -- slt`
Expected: ALL SLT tests pass

### Step F.3: Build check

Run: `cargo build --all-features`
Expected: Clean build, no warnings

---

## Summary

| Task | Feature | Layer | Commits |
|------|---------|-------|---------|
| 1 | Natural key support | A | 1 |
| 2 | Rich WHERE filters | A | 1 |
| 3 | Edge property access | A | 1 |
| 4 | LIST column type | A | 1 |
| 5 | WHERE on dst/intermediate | B | 1 |
| 6 | CHEAPEST path (COST) | B | 1 |
| 7 | Property lookups on path nodes | B | 1 |
| 8 | Path accessor functions | B | 1 |
| 9 | PROPERTIES clause | C | 1 |
| 10 | CREATE OR REPLACE / IF NOT EXISTS | C | 1 |
| 11 | DESCRIBE PROPERTY GRAPH | C | 1 |
| 12 | Label expressions | C | 1 |
| 13 | Multiple MATCH patterns | C | 1 |
| 14 | Bidirectional edges | C | 1 |
| 15 | Quantifier shorthands | C | 1 |
| 16 | Local Clustering Coefficient | C | 1 |

**Total:** 16 tasks, ~16 commits, ~16 new SLT test files

**libteide C changes:** Tasks 4 (LIST wrappers) and 16 (clustering coefficient kernel). Task 6 may need C changes depending on Dijkstra API evaluation.
