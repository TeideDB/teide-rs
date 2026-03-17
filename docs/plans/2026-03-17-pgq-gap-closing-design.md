# PGQ Gap-Closing Design: Full DuckPGQ Feature Parity

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the detailed implementation plan from this design.

**Goal:** Close all 15 identified gaps between TeideDB and DuckPGQ's SQL/PGQ support, achieving full feature parity and surpassing DuckPGQ in graph algorithms.

**Architecture:** Three-layer approach: (A) foundational changes that other features depend on, (B) core features that build on Layer A, (C) independent DDL/syntax features. Items implemented strictly in order within each layer.

**Tech Stack:** Rust SQL layer (`pgq.rs`, `pgq_parser.rs`, `planner.rs`), C engine libteide (`vendor/teide/`), sqlparser crate.

---

## Gap Summary

| # | Feature | Layer | Requires libteide C changes? |
|---|---------|-------|------------------------------|
| 1 | Natural key support | A | No (Rust remapping layer) |
| 2 | Rich WHERE filters | A | No (Rust expression evaluator) |
| 3 | Edge property access in COLUMNS | A | No (Rust projection + edge-row map) |
| 4 | LIST column type | A | Yes — Rust wrappers for existing `TD_LIST` API; add `td_table_add_list_col` helper; format/display in executor |
| 5 | WHERE on destination/intermediate nodes | B | No (remove Rust rejection, add post-filter) |
| 6 | CHEAPEST path (COST expression) | B | Possibly — may need `td_dijkstra` enhancements for multi-source or cost-expression columns |
| 7 | Property lookups on path nodes | B | No (Rust projection change) |
| 8 | Path accessor functions (vertices, edges, element_id) | B | Yes — build `TD_LIST` columns from Rust via FFI |
| 9 | PROPERTIES clause in DDL | C | No (Rust metadata filtering) |
| 10 | CREATE OR REPLACE / IF NOT EXISTS | C | No (Rust DDL logic) |
| 11 | DESCRIBE PROPERTY GRAPH | C | No (Rust metadata query) |
| 12 | Label expressions (Person\|Company) | C | No (Rust planner union) |
| 13 | Multiple MATCH patterns (comma-separated) | C | No (Rust join of pattern results) |
| 14 | Bidirectional edges (<-[]->)  | C | No (Rust intersect of forward+reverse) |
| 15 | Quantifier shorthands ({,m} and {n,}) | C | No (parser default values) |
| 16 | Local Clustering Coefficient | C | **Yes — new `td_local_clustering_coeff` kernel in libteide** |

### libteide C Changes Required

1. **TD_LIST Rust wrappers (Item 4):** The C engine already has `td_list_new`, `td_list_append`, `td_list_get`, `td_list_set` with COW semantics and serialization. Need safe Rust wrappers in `engine.rs` and result formatting in the SQL layer and SLT runner. May need a `td_table_add_list_col` helper or a way to build a vector-of-lists column for table results.

2. **TD_LIST display in executor (Item 4):** The C dump/display code (`dump.c`) prints "LIST" for type 0 but doesn't recursively format list contents. Need a display path that renders `[1, 2, 3]` for SQL output.

3. **Dijkstra enhancements (Item 6):** The existing `exec_dijkstra` takes a single `(src, dst)` pair and uses `Rel::set_props` for weights. May need enhancement if the COST expression is dynamic (per-query edge weight computation) rather than a static column. Evaluate whether the existing API suffices or needs a new `td_dijkstra_cost` variant.

4. **Local Clustering Coefficient (Item 16):** New kernel `td_local_clustering_coeff(td_rel_t* rel)` that for each node: gets its neighbor list from the CSR, counts triangles among neighbors (intersect neighbor lists pairwise), computes `2 * triangles / (degree * (degree - 1))`. Returns a `TD_TABLE` with `_node` (I64) and `_lcc` (F64) columns. Uses `td_scratch_arena_t` for temporary buffers.

---

## Layer A — Foundation

### Item 1: Natural Key Support

**Problem:** Vertex keys must be 0-based sequential integers matching row positions. Real data has arbitrary PKs.

**Design:**

Add bidirectional ID mapping to `VertexLabel`:

```rust
struct VertexLabel {
    table_name: String,
    key_column: String,
    key_type: KeyType,          // NEW: Integer or String
    user_to_row: HashMap<KeyValue, usize>,  // NEW: user PK → row index
    row_to_user: Vec<KeyValue>,             // NEW: row index → user PK
}

enum KeyValue {
    Int(i64),
    Str(String),
}
```

**At CREATE PROPERTY GRAPH time:**
- Scan the key column of each vertex table
- Build `user_to_row` and `row_to_user` maps
- Validate uniqueness (reject duplicate keys)
- For edge tables: resolve FK values through `user_to_row` to get 0-based row indices for CSR construction
- Remove the current validation that keys must be `0, 1, 2, ...`

**At query time:**
- WHERE filters: resolve `a.id = 'uuid-123'` → row index via `user_to_row`
- COLUMNS projection: when projecting a key column, use `row_to_user[row_idx]` to return the user-facing key
- For non-key columns: row index is used directly (no change)

**Backward compatibility:** When keys happen to be 0-based sequential integers, the maps are identity mappings. Existing queries continue to work with zero overhead in the hot path (CSR traversal unchanged).

**Files:**
- Modify: `src/sql/pgq.rs` — `VertexLabel` struct, `create_property_graph`, all planners that resolve node IDs
- Modify: `src/sql/pgq_parser.rs` — remove 0-based key validation

---

### Item 2: Rich WHERE Filters

**Problem:** Only `col = value` equality on source nodes. Need full expression evaluation on any node.

**Design:**

Replace string-based filter handling with parsed AST expressions:

```rust
struct NodePattern {
    variable: Option<String>,
    label: Option<String>,
    filter: Option<String>,        // keep raw text for backward compat
    filter_expr: Option<sqlparser::ast::Expr>,  // NEW: parsed expression
}
```

**Parser change:** After extracting the WHERE text, parse it:
```rust
let expr = sqlparser::parser::Parser::parse_expr(&dialect, filter_text)?;
```

**Expression evaluator:** New function `evaluate_filter(expr: &Expr, table: &Table, row: usize) -> Result<bool>`:
- Handles: `=`, `!=`, `>`, `<`, `>=`, `<=`, `LIKE`, `IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`
- Handles: `AND`, `OR`, `NOT` compound expressions
- Handles: column references (resolve via table column lookup), string/int/float literals
- Returns `true` if the row passes the filter

**Optimization:** For simple equality filters on source nodes, preserve the existing fast path (pre-filter start nodes by scanning the vertex table once). For complex expressions, fall back to post-filter.

**Files:**
- Modify: `src/sql/pgq.rs` — new `evaluate_filter` function, update `NodePattern`, update all planners
- Modify: `src/sql/pgq_parser.rs` — parse filter into `Expr`

---

### Item 3: Edge Property Access in COLUMNS

**Problem:** Cannot project edge properties like `t.amount` in COLUMNS.

**Design:**

**Edge-row mapping:** At `CREATE PROPERTY GRAPH` time, for each edge table build a lookup from `(src_row, dst_row)` pairs to edge table row indices:

```rust
struct StoredRel {
    rel: Rel,
    edge_label: EdgeLabel,
    edge_row_map: HashMap<(i64, i64), Vec<usize>>,  // NEW: (src, dst) → edge row(s)
}
```

`Vec<usize>` handles multigraphs (multiple edges between same node pair). For simple graphs, each vec has exactly one element.

**COLUMNS resolution:** When `t.col` is encountered and `t` is an edge variable:
1. Identify which edge in the pattern `t` refers to (edge index `i`)
2. For each result row, get `(node[i], node[i+1])` — the source and destination of edge `i`
3. Look up `edge_row_map[(src, dst)]` to get the edge table row index
4. Project `col` from the edge table at that row

**Files:**
- Modify: `src/sql/pgq.rs` — `StoredRel` struct, `create_property_graph`, all COLUMNS projection code

---

### Item 4: LIST Column Type

**Problem:** `TD_LIST` exists in the C engine but isn't exposed through Rust or SQL. Needed for path accessor functions and general ARRAY support.

**Design:**

**libteide C changes:**

1. **Display helper** — add `td_list_format(td_t* list, char* buf, size_t bufsz)` that renders `[1, 2, 3]` format. Alternatively, handle in the Rust layer since we already format cells there.

2. **Table column support** — verify `td_table_add_col` works with a `TD_LIST` typed vector (a vector where each element is a `td_t*` pointing to a list). If not, add `td_vec_new_list(int64_t nrows)` that creates a vector of LIST pointers.

**Rust layer (`engine.rs`):**

```rust
impl Table {
    /// Read a LIST cell: returns the list elements as i64 values.
    pub fn get_list_i64(&self, col: usize, row: usize) -> Option<Vec<i64>>;

    /// Create a column of LIST values from Rust Vec<Vec<i64>> data.
    pub fn add_list_column(&mut self, name: &str, data: &[Vec<i64>]) -> Result<(), Error>;
}
```

Safe wrappers around `td_list_new`, `td_list_append`, `td_list_get`.

**SQL layer:**

- `slt_runner.rs`: Add type `0` (TD_LIST) case to `format_cell` — render as `[1, 2, 3]`
- `pgq.rs` projection: Build LIST columns natively for path accessors
- ARRAY literal support: `ARRAY[1,2,3]` in SQL → build TD_LIST via FFI
- `array_length(col)` / `list_length(col)`: Return `td_len(list)` as I64
- `unnest(col)`: Expand LIST column into rows (standard SQL array flattening)
- List indexing `col[i]`: Map to `td_list_get`

**PgWire server:** Map TD_LIST to PostgreSQL ARRAY type for wire protocol compatibility.

**Files:**
- Modify: `vendor/teide/src/ops/dump.c` — list display (optional, can do in Rust)
- Modify: `src/ffi.rs` — already has bindings, just verify
- Modify: `src/engine.rs` — add `get_list_i64`, `add_list_column` wrappers
- Modify: `tests/slt_runner.rs` — format LIST cells
- Modify: `src/sql/planner.rs` — ARRAY literal support, `array_length`, `unnest`, list indexing
- Modify: `src/server/` — PgWire ARRAY type mapping

---

## Layer B — Core Features

### Item 5: WHERE on Destination/Intermediate Nodes

**Problem:** All planners reject filters on non-source nodes.

**Design:**

- Remove the `return Err(...)` rejection checks in: `plan_single_hop`, `plan_var_length`, `plan_multi_hop_fixed`, `plan_multi_hop_variable`
- After the C engine returns raw results, post-filter each row:
  - For each node position that has a filter expression, evaluate it against that position's vertex table row
  - Keep only rows where ALL node filters pass
- Optimization: For single-hop with destination equality filter, check destination before building full result row

**Files:**
- Modify: `src/sql/pgq.rs` — remove rejection errors, add post-filter loops

---

### Item 6: CHEAPEST Path (COST Expression)

**Problem:** DuckPGQ's headline feature. Find lowest-cost paths using edge weight expressions.

**Syntax:**
```sql
MATCH p = ANY SHORTEST (a:Account)-[t:Transfer COST t.amount]->+(b:Account)
COLUMNS (a.name, b.name, path_cost(p))
```

**Design:**

**Parser:** Extend `EdgePattern` with `cost_expr: Option<Expr>`. Parse `COST <expr>` inside edge brackets.

**Planner:** New `plan_cheapest_path` function:
1. Evaluate cost expression for each edge row → build a weight vector (F64)
2. Attach weights to Rel via `Rel::set_props`
3. Resolve source/destination node IDs from WHERE filters
4. Call `Graph::dijkstra(src_id, dst_id)` (existing C kernel)
5. Reconstruct path from Dijkstra result
6. Project COLUMNS including new `path_cost(p)` function (sum of edge weights)

**libteide consideration:** The existing `td_dijkstra` takes a Rel with attached props as weights. If the COST expression is a simple column reference (`t.amount`), we just attach that column. If it's a computed expression (`t.fee + t.tax`), we evaluate it in Rust and create a temporary weight column. The C API should suffice — no changes needed unless multi-source Dijkstra is required.

**Limitation (initial):** Single-edge COST paths only. Multi-segment weighted paths (e.g., Person→Owns→Account→Transfer COST→Account) would require a custom weighted BFS in Rust — defer to future work.

**New path function:** `path_cost(p)` returns the total accumulated cost along the path.

**Files:**
- Modify: `src/sql/pgq_parser.rs` — parse COST in edge pattern
- Modify: `src/sql/pgq.rs` — new `plan_cheapest_path`, `path_cost` function
- Test: `tests/slt/pgq_paths.slt` — CHEAPEST path tests

---

### Item 7: Property Lookups on Path Nodes

**Problem:** Shortest-path queries reject `var.col` with error. Only `_node` and `_depth` available.

**Design:**

- In `project_shortest_path_columns`: when encountering `var.col`, look up the vertex table for that variable's position, get the node ID from the BFS result, use it as row index (with natural key remapping) to read the property value
- Remove the error "Property lookups on path nodes are not yet supported"
- Works for both single-edge shortest paths and multi-segment BFS results

**Files:**
- Modify: `src/sql/pgq.rs` — `project_shortest_path_columns`

---

### Item 8: Path Accessor Functions

**Problem:** No `vertices(p)`, `edges(p)`, `element_id(p)` functions.

**Syntax:**
```sql
COLUMNS (vertices(p), edges(p), element_id(p), path_length(p))
```

**Design:**

Uses TD_LIST columns from Item 4.

- **`vertices(p)`**: For each result path, build a `TD_LIST` containing the vertex IDs at each position. The BFS already tracks these in `result_node_ids`. Build the list via `td_list_new` + `td_list_append` through FFI.
- **`edges(p)`**: For each adjacent pair `(node[i], node[i+1])`, look up the edge row index from `edge_row_map` (Item 3). Build a `TD_LIST` of edge row indices.
- **`element_id(p)`**: Interleave vertex IDs and edge row indices into a single `TD_LIST`.
- **Column type**: Each function produces a column of type `TD_LIST` in the result table.

**libteide C changes:** Build LIST values from Rust via existing `td_list_new`/`td_list_append` FFI. May need to verify that a "vector of lists" (where each row is a TD_LIST pointer) works correctly with `td_table_add_col`.

**Files:**
- Modify: `src/sql/pgq.rs` — path accessor functions in all COLUMNS projection code
- Modify: `src/engine.rs` — helper to build LIST columns

---

## Layer C — DDL & Syntax Features

### Item 9: PROPERTIES Clause in DDL

**Design:** Add `visible_columns: Option<ColumnVisibility>` to `VertexLabel`/`EdgeLabel`.

```rust
enum ColumnVisibility {
    All,                          // default
    Include(HashSet<String>),     // PROPERTIES (col1, col2)
    Exclude(HashSet<String>),     // PROPERTIES ARE ALL COLUMNS EXCEPT (col1)
    None,                         // NO PROPERTIES
}
```

Enforce in COLUMNS projection — reject references to non-visible columns.

**Files:**
- Modify: `src/sql/pgq_parser.rs` — parse PROPERTIES variants
- Modify: `src/sql/pgq.rs` — `VertexLabel`/`EdgeLabel` structs, projection validation

---

### Item 10: CREATE OR REPLACE / IF NOT EXISTS

**Design:**
- `OR REPLACE`: Drop existing graph before creating new one
- `IF NOT EXISTS`: Skip silently if graph exists
- Reject combining both

**Files:**
- Modify: `src/sql/pgq_parser.rs` — detect tokens
- Modify: `src/sql/pgq.rs` — `create_property_graph` logic

---

### Item 11: DESCRIBE PROPERTY GRAPH

**Design:** New statement returning a result table:

| element_type | table_name | label | key_column | src_table | dst_table | properties |
|---|---|---|---|---|---|---|
| VERTEX | persons | Person | id | NULL | NULL | name, age |
| EDGE | knows | Knows | NULL | persons | persons | since |

**Files:**
- Modify: `src/sql/pgq_parser.rs` — detect `DESCRIBE PROPERTY GRAPH`
- Modify: `src/sql/pgq.rs` — new `describe_property_graph` function

---

### Item 12: Label Expressions

**Design:** Support `|` (OR) in node labels. Parse into `LabelExpr::Or(Vec<String>)`.

Planner resolves to set of vertex tables, runs pattern per matching table, unions results. Defer `&` (AND) and `!` (NOT).

**Files:**
- Modify: `src/sql/pgq_parser.rs` — parse `Label1|Label2` syntax
- Modify: `src/sql/pgq.rs` — `resolve_node_label` returns set, planners iterate

---

### Item 13: Multiple MATCH Patterns

**Design:** Comma-separated patterns planned independently, joined on shared variable names.

```sql
MATCH (a)-[:Knows]->(b), (b)-[:LivesIn]->(c)
```

Each pattern produces a result table. Shared variables (`b`) become equi-join keys. No shared variables = cross product.

**Files:**
- Modify: `src/sql/pgq.rs` — `plan_graph_table` handles `patterns.len() > 1`

---

### Item 14: Bidirectional Edges

**Design:** `<-[:Label]->` means edge exists in BOTH directions.

New `MatchDirection::Bidirectional`. Planner: run forward + reverse expand, intersect result sets.

**Files:**
- Modify: `src/sql/pgq_parser.rs` — detect `<-[...]->`
- Modify: `src/sql/pgq.rs` — intersection logic in all planners

---

### Item 15: Quantifier Shorthands

**Design:** Parser defaults: `{,m}` → `{0,m}`, `{n,}` → `{n,255}`.

**Files:**
- Modify: `src/sql/pgq_parser.rs` — quantifier parsing

---

### Item 16: Local Clustering Coefficient

**libteide C changes required.**

**New kernel:** `td_local_clustering_coeff(td_rel_t* rel)`:
- For each node `v` with degree `d`:
  - Get neighbor list `N(v)` from CSR
  - For each pair `(u, w)` in `N(v)`: check if edge `(u, w)` exists (binary search in CSR)
  - Count triangles `t`
  - LCC(v) = `2t / (d * (d-1))` (0 if d < 2)
- Returns `TD_TABLE` with `_node` (I64) and `_lcc` (F64) columns
- Uses `td_scratch_arena_t` for temporary buffers

**Rust/SQL:**
- Expose as `CLUSTERING_COEFFICIENT(graph_name, node_var)` in COLUMNS
- Same pattern as PAGERANK/COMPONENT/COMMUNITY

**Files:**
- Create: `vendor/teide/src/ops/graph.c` — add `td_local_clustering_coeff`
- Modify: `vendor/teide/include/teide/td.h` — declare new function
- Modify: `src/ffi.rs` — FFI binding
- Modify: `src/engine.rs` — safe wrapper
- Modify: `src/sql/pgq.rs` — algorithm dispatch

---

## Implementation Order

```
Layer A (foundation):
  1. Natural key support
  2. Rich WHERE filters
  3. Edge property access in COLUMNS
  4. LIST column type (libteide + Rust)

Layer B (core features):
  5. WHERE on destination/intermediate nodes
  6. CHEAPEST path / COST expression
  7. Property lookups on path nodes
  8. Path accessor functions (vertices, edges, element_id)

Layer C (DDL & syntax — can be parallelized):
  9.  PROPERTIES clause
  10. CREATE OR REPLACE / IF NOT EXISTS
  11. DESCRIBE PROPERTY GRAPH
  12. Label expressions
  13. Multiple MATCH patterns
  14. Bidirectional edges
  15. Quantifier shorthands
  16. Local Clustering Coefficient (libteide)
```

## libteide C Change Summary

| Change | File(s) | Complexity |
|--------|---------|------------|
| Verify TD_LIST works in table columns | `src/vec/vec.c`, `src/table/table.c` | Low — likely already works |
| Optional: `td_list_format` display helper | `src/ops/dump.c` | Low |
| `td_local_clustering_coeff` kernel | `src/ops/graph.c`, `include/teide/td.h` | Medium — triangle counting via CSR neighbor intersection |
| Possible: `td_dijkstra` multi-source or cost-column variant | `src/ops/exec.c` | Low-Medium — evaluate after Item 6 design |

Total: 2-3 functions added to libteide. The vast majority of work is Rust-layer.
