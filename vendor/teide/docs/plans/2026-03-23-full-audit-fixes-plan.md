# Full Audit Fixes — Prioritized Plan

## Priority 1: Immediate fixes (confidence 100, one-line or trivial)

### Fix 1.1: CSV double-close fd (C)
**File:** `src/io/csv.c:1081`
**Fix:** Remove `close(fd)` — fd was already closed at line 1044 after mmap.

### Fix 1.2: Remove dead FFI declarations (Rust)
**File:** `teide-rs/src/ffi.rs`
**Fix:** Remove `td_project`, `td_rel_neighbors`, `td_rel_n_nodes` declarations. Remove `OP_PROJECT` constant. Remove `Graph::project()` from engine.rs.

### Fix 1.3: Fix opcode constants in FFI (Rust)
**File:** `teide-rs/src/ffi.rs`
**Fix:** Remove `OP_LOCAL_CLUSTERING_COEFF = 92`. Add all batch 1-3 opcodes:
```rust
pub const OP_DEGREE_CENT:   u16 = 92;
pub const OP_TOPSORT:       u16 = 93;
pub const OP_DFS:           u16 = 94;
pub const OP_ASTAR:         u16 = 95;
pub const OP_K_SHORTEST:    u16 = 96;
pub const OP_CLUSTER_COEFF: u16 = 97;
pub const OP_RANDOM_WALK:   u16 = 98;
pub const OP_BETWEENNESS:   u16 = 99;
pub const OP_CLOSENESS:     u16 = 100;
pub const OP_MST:           u16 = 101;
```

### Fix 1.4: Dijkstra weight column type check (C)
**File:** `src/ops/exec.c:13241`
**Fix:** Add `if (weight_vec->type != TD_F64) return TD_ERR_PTR(TD_ERR_SCHEMA);` after `td_table_get_col`.

### Fix 1.5: Dijkstra dst_id bounds validation (C)
**File:** `src/ops/exec.c:13235`
**Fix:** Add `if (dst_id != -1 && (dst_id < 0 || dst_id >= n)) return TD_ERR_PTR(TD_ERR_RANGE);`

## Priority 2: Safety hardening (confidence 85-95, small changes)

### Fix 2.1: exec_replace off-by-one (C)
**File:** `src/ops/exec.c:10849,10852`
**Fix:** Change `bi + to_len < buf_cap` to `bi + to_len <= buf_cap - 1` (or add an assert that the condition never fails given correct worst-case sizing).

### Fix 2.2: td_table_add_col return check in graph algorithms (C)
**File:** `src/ops/exec.c` — 7 functions
**Fix:** Add `tmp` pattern (already used in astar/k_shortest/betweenness/closeness) to: exec_pagerank, exec_connected_comp, exec_dijkstra, exec_louvain, exec_degree_cent, exec_topsort, exec_dfs.

### Fix 2.3: Dijkstra heap capacity guard (C)
**File:** `src/ops/exec.c:13134`
**Fix:** Add `cap` parameter to `dijk_heap_push` and guard: `if (*size >= cap) return;`

### Fix 2.4: CSR load offset monotonicity validation (C)
**File:** `src/store/csr.c:411`
**Fix:** After loading offsets, scan `for (i = 0; i < n_nodes; i++) if (off[i+1] < off[i]) return TD_ERR_CORRUPT;`

### Fix 2.5: Zero-default INSERT fix for STR/SYM columns (Rust)
**File:** `teide-rs/src/sql/planner.rs:1638-1647`
**Fix:** For `TD_STR` columns, fill with empty string vectors via `td_str_vec_append(vec, "", 0)`. For `TD_SYM`, fill with `td_sym_intern("", 0)`. Check column type before defaulting.

### Fix 2.6: DELETE row count check (Rust)
**File:** `teide-rs/src/sql/planner.rs:972`
**Fix:** Validate `new_nrows >= 0` before computing `deleted = old_nrows - new_nrows`.

### Fix 2.7: Memory leak in Table::with_column_names (Rust)
**File:** `teide-rs/src/engine.rs:672,701`
**Fix:** Replace `sym_intern(name)?` with explicit match that releases `new_raw`/`tbl` on error.

### Fix 2.8: wco_join pin ptrs Vec (Rust)
**File:** `teide-rs/src/engine.rs:2005-2016`
**Fix:** Take `&mut self`, pin the `ptrs` Vec in `self._pinned` after the FFI call.

### Fix 2.9: td_window_join FFI signature (Rust)
**File:** `teide-rs/src/ffi.rs:855`
**Fix:** Update to match C's `td_asof_join` signature, or remove if dead code.

## Priority 3: Systemic improvements (high effort, high impact)

### Fix 3.1: PageRank dangling node handling (C)
**File:** `src/ops/exec.c:12970`
**Fix:** Add pre-pass per iteration to sum rank of zero-out-degree nodes, redistribute as `dangling_sum * damping / n` added to base.

### Fix 3.2: td_col_save crash safety (C)
**File:** `src/store/col.c:448`
**Fix:** Write to temp file, fsync, atomic rename — matching the pattern `td_sym_save` already uses.

### Fix 3.3: exec_if atom type dispatch (C)
**File:** `src/ops/exec.c:10222`
**Fix:** Use `atom_to_numeric()` for scalar then/else values instead of direct `->f64`/`->i64` field reads.

### Fix 3.4: VLA replacement on parallel path (C)
**File:** `src/ops/exec.c:1501`
**Fix:** Replace `int64_t lsym_buf[n]` VLA with `scratch_alloc` for the narrow SYM buffers.

## Priority 4: Deferred (systemic debt, large refactors)

### Deferred 4.1: td_sym_intern return value checking
~80+ call sites. Requires either: (a) a checked wrapper `td_sym_intern_or_err()`, or (b) systematic audit of all call sites. Best addressed as a dedicated cleanup pass.

### Deferred 4.2: td_vec_set_null error propagation
Requires changing return type from `void` to `td_err_t` — breaks all callers. Needs careful migration.

### Deferred 4.3: td_sym_init error propagation
Requires changing return type from `void` to `td_err_t`. Relatively contained — few callers.

### Deferred 4.4: CSV intern failure propagation
`csv_intern_strings` return value needs to be checked and propagated to `td_read_csv_opts`.

### Deferred 4.5: td_heap_merge pool overflow handling
Only triggers at 512 pools (16GB+). Add assert/abort or pool eviction.

### Deferred 4.6: PGQ expression rewriting prefix collision
`extract_var_col_refs` needs word-boundary-aware replacement or AST-based rewrite.

### Deferred 4.7: PGQ quoted identifier keyword collision
Tokenizer needs to distinguish quoted identifiers from keywords.

### Deferred 4.8: Original 4 graph algorithms need C-level tests
PageRank, Connected Components, Dijkstra, Louvain — only tested through Rust FFI, bypassing ASan/UBSan.

### Deferred 4.9: HNSW index needs C-level tests
Complex data structure with zero C tests.
