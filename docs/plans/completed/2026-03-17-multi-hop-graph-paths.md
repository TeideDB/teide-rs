# Multi-Hop Graph Path Queries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Support DuckPGQ-style multi-edge MATCH patterns with mixed quantifiers, cyclic variable binding, ANY SHORTEST mode, and cross-table traversal.

**Architecture:** Add a thin public CSR neighbor-access API to libteide (`td_rel_neighbors`, `td_rel_n_nodes`) — ~20 lines of C. All multi-hop orchestration lives in Rust: fixed-hop patterns route to existing `wco_join`; variable-length patterns use a new fused multi-segment BFS that walks CSR directly via the new accessor. The parser already handles multi-edge patterns and produces the right AST — only the planner dispatcher needs to route to the new handlers.

**Tech Stack:** C17 (libteide CSR accessor), Rust (planner/BFS), existing `wco_join` (LFTJ), existing `PathPattern` AST.

---

## Task 1: Add `td_rel_neighbors` and `td_rel_n_nodes` to libteide

- [x] Add C declarations to td.h
- [x] Add C implementations to csr.c
- [x] Add FFI bindings to ffi.rs
- [x] Add Rust wrappers to engine.rs
- [x] Add test to engine_api.rs
- [x] All tests pass

**Files:**
- Modify: `vendor/teide/include/teide/td.h` (add 2 function declarations after `td_rel_free`, line ~956)
- Modify: `vendor/teide/src/store/csr.c` (add 2 function implementations at end of file)

These are read-only accessors into the existing double-indexed CSR. Zero allocation, O(1) lookup.

**Step 1: Write the C test (engine_api.rs Rust-side, since C tests are in vendor)**

Add a test to `tests/engine_api.rs` that builds a Rel from an edge table and calls the new neighbor accessor:

```rust
#[test]
fn test_rel_neighbors() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let ctx = Context::new().unwrap();

    // Build a small graph: 3 nodes, edges: 0->1, 0->2, 1->2
    let edges = ctx.read_csv_str("src,dst\n0,1\n0,2\n1,2").unwrap();
    let rel = Rel::from_edges(&edges, "src", "dst", 3, 3, true).unwrap();

    // Forward: node 0 has neighbors [1, 2]
    let (ptr, count) = rel.neighbors(0, 0); // direction 0 = forward
    assert_eq!(count, 2);
    let neighbors = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    assert!(neighbors.contains(&1));
    assert!(neighbors.contains(&2));

    // Forward: node 1 has neighbors [2]
    let (ptr, count) = rel.neighbors(1, 0);
    assert_eq!(count, 1);
    let neighbors = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    assert_eq!(neighbors[0], 2);

    // Forward: node 2 has no outgoing neighbors
    let (_, count) = rel.neighbors(2, 0);
    assert_eq!(count, 0);

    // Reverse: node 2 has incoming from [0, 1]
    let (ptr, count) = rel.neighbors(2, 1); // direction 1 = reverse
    assert_eq!(count, 2);
    let neighbors = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
    assert!(neighbors.contains(&0));
    assert!(neighbors.contains(&1));

    // n_nodes
    assert_eq!(rel.n_nodes(0), 3); // forward: n_src_nodes
    assert_eq!(rel.n_nodes(1), 3); // reverse: n_dst_nodes
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- test_rel_neighbors`
Expected: FAIL — `neighbors` and `n_nodes` methods don't exist on `Rel`.

**Step 3: Add C declarations to `td.h`**

In `vendor/teide/include/teide/td.h`, after `void td_rel_free(td_rel_t* rel);` (~line 956), add:

```c
/* Direct CSR neighbor access — zero-copy, O(1).
 * direction: 0=fwd (src→dst), 1=rev (dst→src).
 * Returns pointer into the CSR targets array; *out_count set to neighbor count.
 * Returns NULL with *out_count=0 if node is out of range or rel is NULL. */
const int64_t* td_rel_neighbors(td_rel_t* rel, int64_t node,
                                 uint8_t direction, int64_t* out_count);

/* Number of source (direction=0) or destination (direction=1) nodes in the rel. */
int64_t td_rel_n_nodes(td_rel_t* rel, uint8_t direction);
```

**Step 4: Add C implementations to `csr.c`**

At the end of `vendor/teide/src/store/csr.c`, add:

```c
/* --- Public CSR neighbor access ------------------------------------------- */

const int64_t* td_rel_neighbors(td_rel_t* rel, int64_t node,
                                 uint8_t direction, int64_t* out_count) {
    if (!rel) { if (out_count) *out_count = 0; return NULL; }
    td_csr_t* csr = (direction == 1) ? &rel->rev : &rel->fwd;
    return td_csr_neighbors(csr, node, out_count);
}

int64_t td_rel_n_nodes(td_rel_t* rel, uint8_t direction) {
    if (!rel) return 0;
    td_csr_t* csr = (direction == 1) ? &rel->rev : &rel->fwd;
    return csr->n_nodes;
}
```

**Step 5: Add FFI bindings to `ffi.rs`**

In `src/ffi.rs`, in the `extern "C"` block, after `td_rel_free`:

```rust
    pub fn td_rel_neighbors(
        rel: *mut td_rel_t,
        node: i64,
        direction: u8,
        out_count: *mut i64,
    ) -> *const i64;

    pub fn td_rel_n_nodes(rel: *mut td_rel_t, direction: u8) -> i64;
```

**Step 6: Add Rust wrappers to `engine.rs`**

In `src/engine.rs`, in the `impl Rel` block (after `as_raw`):

```rust
    /// Get direct read-only access to a node's CSR neighbor list.
    /// Returns (pointer, count). The pointer is into the CSR's internal
    /// targets array and is valid as long as this `Rel` is alive.
    /// direction: 0=fwd (src→dst), 1=rev (dst→src).
    pub fn neighbors(&self, node: i64, direction: u8) -> (*const i64, i64) {
        let mut count: i64 = 0;
        let ptr = unsafe {
            ffi::td_rel_neighbors(self.ptr, node, direction, &mut count)
        };
        (ptr, count)
    }

    /// Number of nodes on one side of the relationship.
    /// direction: 0 → source node count, 1 → destination node count.
    pub fn n_nodes(&self, direction: u8) -> i64 {
        unsafe { ffi::td_rel_n_nodes(self.ptr, direction) }
    }
```

**Step 7: Run test to verify it passes**

Run: `cargo test --all-features -- test_rel_neighbors`
Expected: PASS

**Step 8: Commit**

```bash
git add vendor/teide/include/teide/td.h vendor/teide/src/store/csr.c \
        src/ffi.rs src/engine.rs tests/engine_api.rs
git commit -m "feat: add td_rel_neighbors/td_rel_n_nodes CSR accessors to libteide"
```

---

## Task 2: Multi-hop fixed-hop patterns via `wco_join`

Route patterns where ALL edges have `PathQuantifier::One` (e.g., `(a)-[e1]->(b)-[e2]->(c)`) to the existing `wco_join` engine op.

**Files:**
- Modify: `src/sql/pgq.rs` (update dispatcher ~line 594, add `plan_multi_hop_fixed`)
- Create: `tests/slt/pgq_multi_hop.slt`

**Step 1: Write the SLT test**

Create `tests/slt/pgq_multi_hop.slt`:

```sql
# SQL/PGQ: Multi-hop fixed patterns

# Setup: Person -[Knows]-> Person -[LivesIn]-> City
statement ok
CREATE TABLE mh_persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO mh_persons VALUES (0, 'Alice'), (1, 'Bob'), (2, 'Carol')

statement ok
CREATE TABLE mh_cities (id INTEGER, city VARCHAR)

statement ok
INSERT INTO mh_cities VALUES (0, 'NYC'), (1, 'SF')

statement ok
CREATE TABLE mh_knows (src INTEGER, dst INTEGER)

statement ok
INSERT INTO mh_knows VALUES (0, 1), (0, 2), (1, 2)

statement ok
CREATE TABLE mh_lives (person_id INTEGER, city_id INTEGER)

statement ok
INSERT INTO mh_lives VALUES (0, 0), (1, 1), (2, 0)

statement ok
CREATE PROPERTY GRAPH mh_graph VERTEX TABLES (mh_persons LABEL Person, mh_cities LABEL City) EDGE TABLES (mh_knows SOURCE KEY (src) REFERENCES mh_persons (id) DESTINATION KEY (dst) REFERENCES mh_persons (id) LABEL Knows, mh_lives SOURCE KEY (person_id) REFERENCES mh_persons (id) DESTINATION KEY (city_id) REFERENCES mh_cities (id) LABEL LivesIn)

# 2-hop: Alice knows someone who lives in a city
query TT
SELECT * FROM GRAPH_TABLE (mh_graph MATCH (a:Person WHERE a.name = 'Alice')-[:Knows]->(b:Person)-[:LivesIn]->(c:City) COLUMNS (b.name, c.city)) ORDER BY b.name
----
Bob SF
Carol NYC
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- slt`
Expected: FAIL — "Multi-hop and cyclic MATCH patterns are not yet supported"

**Step 3: Update the dispatcher in `plan_graph_table`**

In `src/sql/pgq.rs`, replace the `_ =>` arm (~line 609-623) in the `plan_graph_table` match with:

```rust
        _ => {
            if pattern.edges.len() >= 2 {
                // Check if ALL edges are single-hop (PathQuantifier::One)
                let all_fixed = pattern.edges.iter().all(|e| {
                    matches!(e.quantifier, PathQuantifier::One)
                });
                if all_fixed {
                    plan_multi_hop_fixed(session, graph, pattern, &expr.columns)
                } else {
                    // At least one variable-length edge → fused BFS (Task 3)
                    plan_multi_hop_variable(session, graph, pattern, &expr.columns, &match_clause.mode)
                }
            } else {
                Err(SqlError::Plan(format!(
                    "Unsupported MATCH pattern: {} nodes, {} edges",
                    pattern.nodes.len(),
                    pattern.edges.len()
                )))
            }
        }
```

**Step 4: Add stub for `plan_multi_hop_variable`**

```rust
/// Plan a multi-edge MATCH with at least one variable-length edge.
/// Implemented in Task 3.
fn plan_multi_hop_variable(
    _session: &Session,
    _graph: &PropertyGraph,
    _pattern: &PathPattern,
    _columns: &[ColumnEntry],
    _mode: &PathMode,
) -> Result<(Table, Vec<String>), SqlError> {
    Err(SqlError::Plan(
        "Multi-hop patterns with variable-length edges are not yet supported.".into(),
    ))
}
```

**Step 5: Implement `plan_multi_hop_fixed`**

This function maps the multi-edge pattern to `wco_join`:

```rust
/// Plan a multi-edge fixed-hop MATCH via worst-case optimal join.
///
/// Pattern: (n0)-[e0]->(n1)-[e1]->(n2)-...-[eK]->(nK+1)
/// All edges have PathQuantifier::One.
///
/// Maps to wco_join with N+1 variables (one per node in the pattern)
/// and K rels (one per edge). Each Rel connects the appropriate pair
/// of node variables.
fn plan_multi_hop_fixed(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
) -> Result<(Table, Vec<String>), SqlError> {
    let n_nodes = pattern.nodes.len();
    let n_edges = pattern.edges.len();
    assert!(n_edges >= 2);
    assert_eq!(n_nodes, n_edges + 1);

    // Resolve edge labels → StoredRel, and validate node labels against edge expectations.
    struct SegmentInfo<'a> {
        stored_rel: &'a StoredRel,
        direction: u8,            // 0=fwd, 1=rev
        src_table_name: String,   // vertex table for left node
        dst_table_name: String,   // vertex table for right node
    }

    let mut segments: Vec<SegmentInfo> = Vec::new();

    for (i, edge) in pattern.edges.iter().enumerate() {
        let edge_label = edge.label.as_deref().ok_or_else(|| {
            SqlError::Plan(format!("Edge {} must specify a label", i))
        })?;
        let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
            SqlError::Plan(format!("Edge label '{}' not found in graph", edge_label))
        })?;

        let is_reverse = edge.direction == MatchDirection::Reverse;
        let (src_table, dst_table) = if is_reverse {
            (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table)
        } else {
            (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table)
        };

        // Validate explicit node labels if present
        let left_node = &pattern.nodes[i];
        if let Some(label) = &left_node.label {
            let vl = graph.vertex_labels.get(label).ok_or_else(|| {
                SqlError::Plan(format!("Vertex label '{}' not found", label))
            })?;
            if vl.table_name != *src_table {
                return Err(SqlError::Plan(format!(
                    "Node label '{}' resolves to '{}' but edge '{}' expects source '{}'",
                    label, vl.table_name, edge_label, src_table
                )));
            }
        }
        let right_node = &pattern.nodes[i + 1];
        if let Some(label) = &right_node.label {
            let vl = graph.vertex_labels.get(label).ok_or_else(|| {
                SqlError::Plan(format!("Vertex label '{}' not found", label))
            })?;
            if vl.table_name != *dst_table {
                return Err(SqlError::Plan(format!(
                    "Node label '{}' resolves to '{}' but edge '{}' expects destination '{}'",
                    label, vl.table_name, edge_label, dst_table
                )));
            }
        }

        let direction: u8 = match edge.direction {
            MatchDirection::Forward => 0,
            MatchDirection::Reverse => 1,
            MatchDirection::Undirected => 2,
        };

        segments.push(SegmentInfo {
            stored_rel,
            direction,
            src_table_name: src_table.clone(),
            dst_table_name: dst_table.clone(),
        });
    }

    // Verify node-table continuity: segment[i].dst_table must equal segment[i+1].src_table
    for i in 0..segments.len() - 1 {
        if segments[i].dst_table_name != segments[i + 1].src_table_name {
            return Err(SqlError::Plan(format!(
                "Vertex table mismatch between edge {} ('{}') and edge {} ('{}'): \
                 '{}' != '{}'",
                i, pattern.edges[i].label.as_deref().unwrap_or("?"),
                i + 1, pattern.edges[i + 1].label.as_deref().unwrap_or("?"),
                segments[i].dst_table_name, segments[i + 1].src_table_name
            )));
        }
    }

    // Detect cyclic binding: first and last node share the same variable name
    let is_cyclic = match (&pattern.nodes[0].variable, &pattern.nodes[n_nodes - 1].variable) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    if is_cyclic && segments[0].src_table_name != segments.last().unwrap().dst_table_name {
        return Err(SqlError::Plan(
            "Cyclic pattern binding requires the first and last nodes to be in the same vertex table.".into()
        ));
    }

    // Build node-to-table mapping for COLUMNS projection
    struct NodeInfo {
        table_name: String,
        variable: Option<String>,
    }
    let mut node_infos: Vec<NodeInfo> = Vec::new();
    node_infos.push(NodeInfo {
        table_name: segments[0].src_table_name.clone(),
        variable: pattern.nodes[0].variable.clone(),
    });
    for seg in &segments {
        node_infos.push(NodeInfo {
            table_name: seg.dst_table_name.clone(),
            variable: pattern.nodes[node_infos.len()].variable.clone(),
        });
    }

    // Use the first segment's source table as the Graph base (needed by wco_join).
    let base_table = &session.tables.get(&segments[0].src_table_name)
        .ok_or_else(|| SqlError::Plan(format!(
            "Table '{}' not found", segments[0].src_table_name
        )))?.table;

    let mut g = session.ctx.graph(base_table)?;

    // Collect Rel references for wco_join (borrow from segments)
    let rels: Vec<&Rel> = segments.iter().map(|s| &s.stored_rel.rel).collect();

    // Determine n_vars: number of unique node positions.
    // If cyclic, first and last are the same variable, so n_vars = n_nodes - 1.
    let n_vars: u8 = if is_cyclic {
        (n_nodes - 1).try_into().map_err(|_| {
            SqlError::Plan("Too many nodes in pattern (max 255)".into())
        })?
    } else {
        n_nodes.try_into().map_err(|_| {
            SqlError::Plan("Too many nodes in pattern (max 255)".into())
        })?
    };

    let join_result = g.wco_join(&rels, n_vars)?;

    // Apply source-node WHERE filter if present
    let join_result = if let Some(filter_text) = &pattern.nodes[0].filter {
        // _v0 is the first node variable; filter on its properties
        apply_node_filter(&mut g, join_result, filter_text, pattern.nodes[0].variable.as_deref())?
    } else {
        join_result
    };

    let result = g.execute(join_result)?;
    let nrows = checked_nrows(&result)?;

    // wco_join output: _v0, _v1, ..., _v{n_vars-1} (all I64 node IDs)
    // Project COLUMNS by looking up var.col in the appropriate vertex table.
    let mut col_names: Vec<String> = Vec::new();
    struct ProjSpec {
        var_idx: usize,      // which _v column to use as row index
        table_col_idx: usize, // column index in the vertex table
        table_name: String,
    }
    let mut proj_specs: Vec<ProjSpec> = Vec::new();

    for entry in columns {
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = entry.expr.find('.') {
            let var = entry.expr[..dot_pos].trim().to_lowercase();
            let col = entry.expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            // Find which node position this variable refers to
            let var_idx = node_infos.iter().position(|ni| {
                ni.variable.as_ref().map(|v| v.to_lowercase()) == Some(var.clone())
            }).ok_or_else(|| {
                SqlError::Plan(format!("Unknown variable '{}' in COLUMNS", var))
            })?;

            // Map to wco_join var index (for cyclic, last node maps to _v0)
            let wco_var_idx = if is_cyclic && var_idx == n_nodes - 1 { 0 } else { var_idx };

            let table = &session.tables.get(&node_infos[var_idx].table_name)
                .ok_or_else(|| SqlError::Plan(format!(
                    "Table '{}' not found", node_infos[var_idx].table_name
                )))?.table;
            let table_col_idx = find_col_idx(table, &col)
                .ok_or_else(|| SqlError::Plan(format!(
                    "Column '{}' not found in table '{}'", col, node_infos[var_idx].table_name
                )))?;

            col_names.push(out_name);
            proj_specs.push(ProjSpec {
                var_idx: wco_var_idx,
                table_col_idx,
                table_name: node_infos[var_idx].table_name.clone(),
            });
        } else {
            return Err(SqlError::Plan(format!(
                "COLUMNS: unsupported expression '{}'. Use var.column syntax.", entry.expr
            )));
        }
    }

    if proj_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Read _v{i} columns from wco_join result
    let mut var_col_indices: Vec<usize> = Vec::new();
    for i in 0..n_vars as usize {
        let col_name = format!("_v{i}");
        let idx = find_col_idx(&result, &col_name).ok_or_else(|| {
            SqlError::Plan(format!("wco_join result missing column '{col_name}'"))
        })?;
        var_col_indices.push(idx);
    }

    // Build CSV result
    let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for row in 0..nrows {
        for (i, spec) in proj_specs.iter().enumerate() {
            if i > 0 { csv.push(','); }
            let node_id = result.get_i64(var_col_indices[spec.var_idx], row)
                .ok_or_else(|| SqlError::Plan(format!("NULL _v{} at row {}", spec.var_idx, row)))?;
            if node_id < 0 {
                return Err(SqlError::Plan(format!(
                    "Negative node ID {} from _v{} at row {}", node_id, spec.var_idx, row
                )));
            }
            let vtable = &session.tables.get(&spec.table_name)
                .ok_or_else(|| SqlError::Plan(format!("Table '{}' not found", spec.table_name)))?.table;
            csv.push_str(&get_cell_string(vtable, spec.table_col_idx, node_id as usize)?);
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}
```

**Step 6: Run test to verify it passes**

Run: `cargo test --all-features -- slt`
Expected: PASS (both old tests and new `pgq_multi_hop.slt`)

**Step 7: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_multi_hop.slt
git commit -m "feat: multi-hop fixed-hop MATCH patterns via wco_join"
```

---

## Task 3: Fused multi-segment BFS for variable-length multi-edge patterns

This is the core new capability: a BFS that walks across multiple edge types with mixed quantifiers, using `Rel::neighbors()` for zero-copy CSR access.

**Files:**
- Modify: `src/sql/pgq.rs` (implement `plan_multi_hop_variable`, add `multi_segment_bfs`)
- Modify: `tests/slt/pgq_multi_hop.slt` (add variable-length tests)

**Step 1: Add SLT tests for variable-length multi-hop**

Append to `tests/slt/pgq_multi_hop.slt`:

```sql
# --- Variable-length multi-hop ---

# Setup: financial graph
# Person -[Owns]-> Account -[Transfer]->+ Account
statement ok
CREATE TABLE fin_persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO fin_persons VALUES (0, 'Alice'), (1, 'Bob')

statement ok
CREATE TABLE fin_accounts (id INTEGER, acct_name VARCHAR)

statement ok
INSERT INTO fin_accounts VALUES (0, 'Checking_A'), (1, 'Savings_A'), (2, 'Checking_B')

statement ok
CREATE TABLE fin_owns (person_id INTEGER, account_id INTEGER)

statement ok
INSERT INTO fin_owns VALUES (0, 0), (0, 1), (1, 2)

statement ok
CREATE TABLE fin_transfers (src_acct INTEGER, dst_acct INTEGER)

statement ok
INSERT INTO fin_transfers VALUES (0, 2), (2, 1)

statement ok
CREATE PROPERTY GRAPH fin_graph VERTEX TABLES (fin_persons LABEL Person, fin_accounts LABEL Account) EDGE TABLES (fin_owns SOURCE KEY (person_id) REFERENCES fin_persons (id) DESTINATION KEY (account_id) REFERENCES fin_accounts (id) LABEL Owns, fin_transfers SOURCE KEY (src_acct) REFERENCES fin_accounts (id) DESTINATION KEY (dst_acct) REFERENCES fin_accounts (id) LABEL Transfer)

# Multi-hop: Alice's accounts that transferred to any account in 1+ hops
query TT
SELECT * FROM GRAPH_TABLE (fin_graph MATCH (p:Person WHERE p.name = 'Alice')-[:Owns]->(a1:Account)-[:Transfer]->+(a2:Account) COLUMNS (a1.acct_name, a2.acct_name)) ORDER BY a1.acct_name, a2.acct_name
----
Checking_A Checking_B
Checking_A Savings_A
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- slt`
Expected: FAIL — "Multi-hop patterns with variable-length edges are not yet supported."

**Step 3: Implement `multi_segment_bfs`**

This is the core BFS engine. Add to `src/sql/pgq.rs`:

```rust
/// Describes one segment in a multi-hop pattern.
struct BfsSegment<'a> {
    stored_rel: &'a StoredRel,
    direction: u8,
    min_depth: u8,
    max_depth: u8,
    src_table_name: String,
    dst_table_name: String,
}

/// BFS result: one row per result tuple.
struct BfsResult {
    /// For each node position in the pattern, the node IDs per result row.
    /// node_ids[position][row] = node_id
    node_ids: Vec<Vec<i64>>,
    /// Total path length per result row (sum of depths across all segments).
    path_lengths: Vec<i64>,
    /// Number of result rows.
    nrows: usize,
}

/// Fused multi-segment BFS.
///
/// Walks a chain of edge types with mixed quantifiers:
///   (n0)-[e0{min0,max0}]->(n1)-[e1{min1,max1}]->(n2)-...
///
/// For WALK mode: explores all paths, collects all result tuples.
/// For ANY SHORTEST mode: BFS per-segment, stops at first complete match.
///
/// Uses Rel::neighbors() for zero-copy CSR access. Never materializes
/// intermediate tables — works directly with node ID sets.
fn multi_segment_bfs(
    segments: &[BfsSegment],
    start_nodes: &[i64],
    mode: &PathMode,
    is_cyclic: bool, // if true, filter: last node == first node
) -> Result<BfsResult, SqlError> {
    use std::collections::{HashSet, VecDeque};

    let n_segments = segments.len();
    let n_positions = n_segments + 1; // one node per segment boundary

    const MAX_RESULTS: usize = 1_000_000;
    const MAX_BFS_STATES: usize = 10_000_000;

    // BFS state: (segment_index, node_id, depth_in_segment, path_so_far)
    // path_so_far[i] = node_id at position i
    struct BfsState {
        seg_idx: usize,
        node: i64,
        depth_in_seg: u8,
        path: Vec<i64>,        // node IDs at each position boundary so far
        total_depth: i64,
    }

    let mut results = BfsResult {
        node_ids: vec![Vec::new(); n_positions],
        path_lengths: Vec::new(),
        nrows: 0,
    };

    let mut frontier: VecDeque<BfsState> = VecDeque::new();
    let mut visited_count: usize = 0;

    // Initialize: one state per start node, at segment 0, depth 0
    for &start in start_nodes {
        frontier.push_back(BfsState {
            seg_idx: 0,
            node: start,
            depth_in_seg: 0,
            path: vec![start],
            total_depth: 0,
        });
    }

    // For ANY SHORTEST, track minimum total depth found
    let mut shortest_found: Option<i64> = None;

    while let Some(state) = frontier.pop_front() {
        visited_count += 1;
        if visited_count > MAX_BFS_STATES {
            return Err(SqlError::Plan(format!(
                "Multi-hop BFS exceeded {} states — pattern too broad or graph too large. \
                 Try narrowing the hop range.",
                MAX_BFS_STATES
            )));
        }

        // For ANY SHORTEST: prune if we already found a shorter path
        if *mode == PathMode::AnyShortest {
            if let Some(best) = shortest_found {
                if state.total_depth > best {
                    continue;
                }
            }
        }

        let seg = &segments[state.seg_idx];

        // Check if this state has reached the segment's min_depth
        let at_or_past_min = state.depth_in_seg >= seg.min_depth;

        // If at_or_past_min, this node can be the "exit" of this segment
        if at_or_past_min {
            if state.seg_idx == n_segments - 1 {
                // Last segment — this is a complete path
                let mut final_path = state.path.clone();
                final_path.push(state.node);

                // Cyclic check: first node == last node
                if is_cyclic && final_path[0] != *final_path.last().unwrap() {
                    // Not cyclic — skip
                } else {
                    // Record result
                    if *mode == PathMode::AnyShortest {
                        match shortest_found {
                            None => shortest_found = Some(state.total_depth),
                            Some(best) if state.total_depth < best => {
                                // Found shorter: clear previous results
                                for v in &mut results.node_ids { v.clear(); }
                                results.path_lengths.clear();
                                results.nrows = 0;
                                shortest_found = Some(state.total_depth);
                            }
                            Some(best) if state.total_depth > best => continue,
                            _ => {} // same length, add to results
                        }
                    }

                    for (i, &nid) in final_path.iter().enumerate() {
                        results.node_ids[i].push(nid);
                    }
                    results.path_lengths.push(state.total_depth);
                    results.nrows += 1;

                    if results.nrows >= MAX_RESULTS {
                        return Err(SqlError::Plan(format!(
                            "Multi-hop BFS exceeded {} results — pattern too broad.",
                            MAX_RESULTS
                        )));
                    }

                    if *mode == PathMode::AnyShortest {
                        // Don't stop — other start nodes might also have shortest paths.
                        // But prune will kick in for longer paths.
                        continue;
                    }
                }
            } else {
                // Transition to the next segment: record current node as
                // the boundary, start next segment at depth 0
                let mut next_path = state.path.clone();
                next_path.push(state.node);
                frontier.push_back(BfsState {
                    seg_idx: state.seg_idx + 1,
                    node: state.node,
                    depth_in_seg: 0,
                    path: next_path,
                    total_depth: state.total_depth,
                });
            }
        }

        // Expand within the current segment if depth < max_depth
        if state.depth_in_seg < seg.max_depth {
            let (ptr, count) = seg.stored_rel.rel.neighbors(state.node, seg.direction);
            if count > 0 && !ptr.is_null() {
                let neighbors = unsafe { std::slice::from_raw_parts(ptr, count as usize) };
                for &next in neighbors {
                    frontier.push_back(BfsState {
                        seg_idx: state.seg_idx,
                        node: next,
                        depth_in_seg: state.depth_in_seg + 1,
                        path: state.path.clone(),
                        total_depth: state.total_depth + 1,
                    });
                }
            }
        }
    }

    Ok(results)
}
```

**Step 4: Implement `plan_multi_hop_variable`**

Replace the stub from Task 2:

```rust
/// Plan a multi-edge MATCH with at least one variable-length edge.
///
/// Decomposes the pattern into segments, each with its own Rel and quantifier.
/// Uses fused multi-segment BFS via `Rel::neighbors()` for zero-copy CSR access.
fn plan_multi_hop_variable(
    session: &Session,
    graph: &PropertyGraph,
    pattern: &PathPattern,
    columns: &[ColumnEntry],
    mode: &PathMode,
) -> Result<(Table, Vec<String>), SqlError> {
    let n_nodes = pattern.nodes.len();
    let n_edges = pattern.edges.len();
    assert!(n_edges >= 1);
    assert_eq!(n_nodes, n_edges + 1);

    // Build segments
    let mut segments: Vec<BfsSegment> = Vec::new();

    for (i, edge) in pattern.edges.iter().enumerate() {
        let edge_label = edge.label.as_deref().ok_or_else(|| {
            SqlError::Plan(format!("Edge {} must specify a label", i))
        })?;
        let stored_rel = graph.edge_labels.get(edge_label).ok_or_else(|| {
            SqlError::Plan(format!("Edge label '{}' not found in graph", edge_label))
        })?;

        let is_reverse = edge.direction == MatchDirection::Reverse;
        let (src_table, dst_table) = if is_reverse {
            (&stored_rel.edge_label.dst_ref_table, &stored_rel.edge_label.src_ref_table)
        } else {
            (&stored_rel.edge_label.src_ref_table, &stored_rel.edge_label.dst_ref_table)
        };

        // Validate node labels
        let left_node = &pattern.nodes[i];
        if let Some(label) = &left_node.label {
            let vl = graph.vertex_labels.get(label).ok_or_else(|| {
                SqlError::Plan(format!("Vertex label '{}' not found", label))
            })?;
            if vl.table_name != *src_table {
                return Err(SqlError::Plan(format!(
                    "Node label '{}' resolves to '{}' but edge '{}' expects source '{}'",
                    label, vl.table_name, edge_label, src_table
                )));
            }
        }
        let right_node = &pattern.nodes[i + 1];
        if let Some(label) = &right_node.label {
            let vl = graph.vertex_labels.get(label).ok_or_else(|| {
                SqlError::Plan(format!("Vertex label '{}' not found", label))
            })?;
            if vl.table_name != *dst_table {
                return Err(SqlError::Plan(format!(
                    "Node label '{}' resolves to '{}' but edge '{}' expects destination '{}'",
                    label, vl.table_name, edge_label, dst_table
                )));
            }
        }

        // Variable-length edges within a segment require same src/dst table
        let (min_depth, max_depth) = match edge.quantifier {
            PathQuantifier::One => (1u8, 1u8),
            PathQuantifier::Range { min, max } => (min, max),
            PathQuantifier::Plus => (1, 255),
            PathQuantifier::Star => (0, 255),
        };
        if min_depth != 1 || max_depth != 1 {
            if stored_rel.edge_label.src_ref_table != stored_rel.edge_label.dst_ref_table {
                return Err(SqlError::Plan(format!(
                    "Variable-length edge '{}' requires same source and destination vertex table, \
                     but connects '{}' → '{}'.",
                    edge_label, stored_rel.edge_label.src_ref_table,
                    stored_rel.edge_label.dst_ref_table
                )));
            }
        }

        let direction: u8 = match edge.direction {
            MatchDirection::Forward => 0,
            MatchDirection::Reverse => 1,
            MatchDirection::Undirected => 2,
        };

        segments.push(BfsSegment {
            stored_rel,
            direction,
            min_depth,
            max_depth,
            src_table_name: src_table.clone(),
            dst_table_name: dst_table.clone(),
        });
    }

    // Verify table continuity between adjacent segments
    for i in 0..segments.len() - 1 {
        if segments[i].dst_table_name != segments[i + 1].src_table_name {
            return Err(SqlError::Plan(format!(
                "Vertex table mismatch between edge {} and edge {}: '{}' != '{}'",
                i, i + 1, segments[i].dst_table_name, segments[i + 1].src_table_name
            )));
        }
    }

    // Detect cyclic binding
    let is_cyclic = match (&pattern.nodes[0].variable, &pattern.nodes[n_nodes - 1].variable) {
        (Some(a), Some(b)) => a.to_lowercase() == b.to_lowercase(),
        _ => false,
    };

    // Determine start nodes from the first node's WHERE filter
    let first_table_name = &segments[0].src_table_name;
    let first_stored = session.tables.get(first_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{}' not found", first_table_name))
    })?;
    let first_n = checked_logical_nrows(first_stored)?;

    let start_nodes: Vec<i64> = if let Some(filter_text) = &pattern.nodes[0].filter {
        // Extract specific node(s) from filter
        let first_ref_col = &segments[0].stored_rel.edge_label.src_ref_col;
        let node_id = extract_node_id(&pattern.nodes[0], first_table_name, first_ref_col, session)?;
        vec![node_id]
    } else {
        // All nodes in the first table
        (0..first_n as i64).collect()
    };

    // Run the BFS
    let bfs_result = multi_segment_bfs(&segments, &start_nodes, mode, is_cyclic)?;

    if bfs_result.nrows == 0 {
        // Return empty table with proper column names
        let mut col_names = Vec::new();
        for entry in columns {
            let alias = entry.alias.as_deref();
            let lower = entry.expr.to_lowercase();
            if lower.contains("path_length") {
                col_names.push(alias.unwrap_or("path_length").to_string());
            } else if let Some(dot_pos) = lower.find('.') {
                let col = lower[dot_pos + 1..].trim();
                col_names.push(alias.unwrap_or(col).to_string());
            } else {
                col_names.push(alias.unwrap_or(&lower).to_string());
            }
        }
        if col_names.is_empty() {
            return Err(SqlError::Plan("COLUMNS clause is empty".into()));
        }
        let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();
        let csv = format!("{}\n", csv_col_names.join(","));
        let result = csv_to_table(session, &csv, &col_names)?;
        return Ok((result, col_names));
    }

    // Build node-to-table mapping
    struct NodeInfo {
        table_name: String,
        variable: Option<String>,
    }
    let mut node_infos: Vec<NodeInfo> = Vec::new();
    node_infos.push(NodeInfo {
        table_name: segments[0].src_table_name.clone(),
        variable: pattern.nodes[0].variable.clone(),
    });
    for (i, seg) in segments.iter().enumerate() {
        node_infos.push(NodeInfo {
            table_name: seg.dst_table_name.clone(),
            variable: pattern.nodes[i + 1].variable.clone(),
        });
    }

    // Project COLUMNS
    let mut col_names: Vec<String> = Vec::new();
    enum ProjKind { NodeProp { pos: usize, col_idx: usize, table_name: String }, PathLength }
    let mut proj_specs: Vec<ProjKind> = Vec::new();

    for entry in columns {
        let alias = entry.alias.as_deref();

        if let Some(dot_pos) = entry.expr.find('.') {
            let var = entry.expr[..dot_pos].trim().to_lowercase();
            let col = entry.expr[dot_pos + 1..].trim().to_lowercase();
            let out_name = alias.unwrap_or(&col).to_string();

            let pos = node_infos.iter().position(|ni| {
                ni.variable.as_ref().map(|v| v.to_lowercase()) == Some(var.clone())
            }).ok_or_else(|| {
                SqlError::Plan(format!("Unknown variable '{}' in COLUMNS", var))
            })?;

            let vtable = &session.tables.get(&node_infos[pos].table_name)
                .ok_or_else(|| SqlError::Plan(format!(
                    "Table '{}' not found", node_infos[pos].table_name
                )))?.table;
            let col_idx = find_col_idx(vtable, &col).ok_or_else(|| {
                SqlError::Plan(format!("Column '{}' not found in '{}'", col, node_infos[pos].table_name))
            })?;

            col_names.push(out_name);
            proj_specs.push(ProjKind::NodeProp { pos, col_idx, table_name: node_infos[pos].table_name.clone() });
        } else {
            let lower = entry.expr.to_lowercase();
            if lower.contains("path_length") || lower == "_depth" {
                col_names.push(alias.unwrap_or("path_length").to_string());
                proj_specs.push(ProjKind::PathLength);
            } else {
                return Err(SqlError::Plan(format!(
                    "COLUMNS: unsupported expression '{}'", entry.expr
                )));
            }
        }
    }

    if proj_specs.is_empty() {
        return Err(SqlError::Plan("COLUMNS clause is empty".into()));
    }

    // Build CSV
    let csv_col_names: Vec<String> = (0..col_names.len()).map(|i| format!("__c{i}")).collect();
    let mut csv = csv_col_names.join(",");
    csv.push('\n');

    for row in 0..bfs_result.nrows {
        for (i, spec) in proj_specs.iter().enumerate() {
            if i > 0 { csv.push(','); }
            match spec {
                ProjKind::NodeProp { pos, col_idx, table_name } => {
                    let node_id = bfs_result.node_ids[*pos][row];
                    let vtable = &session.tables.get(table_name)
                        .ok_or_else(|| SqlError::Plan(format!("Table '{}' not found", table_name)))?.table;
                    csv.push_str(&get_cell_string(vtable, *col_idx, node_id as usize)?);
                }
                ProjKind::PathLength => {
                    csv.push_str(&bfs_result.path_lengths[row].to_string());
                }
            }
        }
        csv.push('\n');
    }

    let result = csv_to_table(session, &csv, &col_names)?;
    Ok((result, col_names))
}
```

**Step 5: Run tests**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 6: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_multi_hop.slt
git commit -m "feat: fused multi-segment BFS for variable-length multi-hop patterns"
```

---

## Task 4: Cyclic variable binding

Add tests and ensure cyclic binding (`(p:Person)...(p:Person)`) works for both fixed-hop and variable-length paths.

**Files:**
- Modify: `tests/slt/pgq_multi_hop.slt` (add cyclic tests)

**Step 1: Add SLT tests for cyclic patterns**

Append to `tests/slt/pgq_multi_hop.slt`:

```sql
# --- Cyclic variable binding ---

# Setup: triangle graph
statement ok
CREATE TABLE tri_nodes (id INTEGER, name VARCHAR)

statement ok
INSERT INTO tri_nodes VALUES (0, 'X'), (1, 'Y'), (2, 'Z')

statement ok
CREATE TABLE tri_edges (src INTEGER, dst INTEGER)

statement ok
INSERT INTO tri_edges VALUES (0, 1), (1, 2), (2, 0), (0, 2)

statement ok
CREATE PROPERTY GRAPH tri_graph VERTEX TABLES (tri_nodes LABEL Node) EDGE TABLES (tri_edges SOURCE KEY (src) REFERENCES tri_nodes (id) DESTINATION KEY (dst) REFERENCES tri_nodes (id) LABEL Edge)

# Fixed-hop cycle: find all triangles (a)->(b)->(c)->(a)
query TTT
SELECT * FROM GRAPH_TABLE (tri_graph MATCH (a:Node)-[:Edge]->(b:Node)-[:Edge]->(c:Node)-[:Edge]->(a:Node) COLUMNS (a.name, b.name, c.name)) ORDER BY a.name, b.name, c.name
----
X Y Z
Z X Y

# Variable-length cycle: (a)-[e]->+(a) — cycles of any length
query T
SELECT * FROM GRAPH_TABLE (tri_graph MATCH (a:Node)-[:Edge]->+(a:Node) COLUMNS (a.name)) ORDER BY a.name
----
X
Y
Z
```

**Step 2: Run test to verify it passes (or identify what needs fixing)**

Run: `cargo test --all-features -- slt`

The cyclic detection logic is already in both `plan_multi_hop_fixed` (via `is_cyclic` flag and wco_join `n_vars` adjustment) and `multi_segment_bfs` (via the `is_cyclic` filter). This task is primarily about testing.

If the fixed-hop cyclic test fails, debug and fix the wco_join variable mapping. If the variable-length cyclic test fails, it goes through the single-edge `plan_var_length` path (2 nodes, 1 edge) — we may need to extend the dispatcher to detect cyclic single-edge patterns and route them to `plan_multi_hop_variable`.

**Step 3: Handle single-edge cyclic patterns in the dispatcher**

If needed, update the dispatcher in `plan_graph_table` to detect when a single-edge pattern has cyclic binding:

```rust
        // Before the main match, check for cyclic single-edge patterns
        let is_single_edge_cyclic = pattern.edges.len() == 1
            && pattern.nodes.len() == 2
            && match (&pattern.nodes[0].variable, &pattern.nodes[1].variable) {
                (Some(a), Some(b)) => a.to_lowercase() == b.to_lowercase(),
                _ => false,
            };
```

If `is_single_edge_cyclic` is true and the edge is variable-length, route to `plan_multi_hop_variable` (which handles cyclic via the BFS filter).

**Step 4: Run tests and verify**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_multi_hop.slt
git commit -m "feat: cyclic variable binding in multi-hop MATCH patterns"
```

---

## Task 5: ANY SHORTEST with multi-edge patterns

Extend the multi-hop paths to support `p = ANY SHORTEST` with `path_length(p)`.

**Files:**
- Modify: `tests/slt/pgq_multi_hop.slt` (add ANY SHORTEST multi-hop tests)
- Potentially modify: `src/sql/pgq.rs` (adjust dispatcher for multi-edge ANY SHORTEST)

**Step 1: Add SLT tests**

Append to `tests/slt/pgq_multi_hop.slt`:

```sql
# --- ANY SHORTEST multi-hop ---

# Shortest path from Alice's accounts to Bob's accounts via transfers
query TTI
SELECT * FROM GRAPH_TABLE (fin_graph MATCH p = ANY SHORTEST (p1:Person WHERE p1.name = 'Alice')-[:Owns]->(a1:Account)-[:Transfer]->+(a2:Account)<-[:Owns]-(p2:Person WHERE p2.name = 'Bob') COLUMNS (a1.acct_name, a2.acct_name, path_length(p))) ORDER BY a1.acct_name
----
Checking_A Checking_B 2
```

**Step 2: Run test to verify behavior**

Run: `cargo test --all-features -- slt`

The dispatcher routes multi-edge patterns with at least one variable-length edge to `plan_multi_hop_variable`, which already receives `mode`. The `multi_segment_bfs` already implements ANY SHORTEST pruning. This task verifies the integration works end-to-end.

**Step 3: Fix dispatcher for multi-edge ANY SHORTEST**

The current dispatcher only checks `PathMode::AnyShortest` for the `(2, 1, ...)` case. For multi-edge patterns, ensure the mode is passed through. Update the dispatcher:

In the `_ =>` arm of the match in `plan_graph_table`, the `plan_multi_hop_variable` call already passes `&match_clause.mode`. For `plan_multi_hop_fixed`, if someone writes `p = ANY SHORTEST (a)-[e1]->(b)-[e2]->(c)` with all fixed-hop edges, wco_join returns all matches (no shortest semantics needed since all paths have the same length). Verify this works and add a comment.

**Step 4: Run tests**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_multi_hop.slt
git commit -m "feat: ANY SHORTEST with multi-edge MATCH patterns"
```

---

## Task 6: Cross-table traversal (heterogeneous vertex tables)

Verify and test hopping across different vertex tables (Person→Account→Person).

**Files:**
- Modify: `tests/slt/pgq_multi_hop.slt` (add cross-table round-trip test)

**Step 1: Add SLT test for full DuckPGQ-style pattern**

Append to `tests/slt/pgq_multi_hop.slt`:

```sql
# --- Cross-table: Person -> Account -> Account -> Person ---

# Who can Alice reach via her accounts' transfer chains?
query TI
SELECT * FROM GRAPH_TABLE (fin_graph MATCH (p1:Person WHERE p1.name = 'Alice')-[:Owns]->(a1:Account)-[:Transfer]->+(a2:Account)<-[:Owns]-(p2:Person) COLUMNS (p2.name, path_length(p))) ORDER BY p2.name
----
Bob 3

# Reverse: who sent money to Bob's accounts?
query TI
SELECT * FROM GRAPH_TABLE (fin_graph MATCH (p1:Person WHERE p1.name = 'Bob')-[:Owns]->(a1:Account)<-[:Transfer]-+(a2:Account)<-[:Owns]-(p2:Person) COLUMNS (p2.name, path_length(p))) ORDER BY p2.name
----
Alice 3
```

**Step 2: Run tests**

Run: `cargo test --all-features -- slt`

Cross-table traversal should already work since each segment tracks its own `src_table_name` and `dst_table_name`, and the BFS transitions between segments when crossing table boundaries.

**Step 3: Fix any issues found**

Likely edge cases:
- Reverse edge direction (`<-`) in the last segment needs correct table resolution
- `path_length()` must count across all segments correctly

**Step 4: Run full test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/slt/pgq_multi_hop.slt
git commit -m "test: cross-table multi-hop traversal (Person→Account→Person)"
```

---

## Task 7: Run existing tests to verify no regressions

**Step 1: Run full test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All existing tests pass (pgq.slt, pgq_paths.slt, pgq_algorithms.slt, engine_api.rs)

**Step 2: Run the specific SLT tests**

Run: `cargo test --all-features -- slt`
Expected: All SLT files pass

**Step 3: Final commit (if any cleanup needed)**

```bash
git commit -m "chore: cleanup after multi-hop graph paths implementation"
```
