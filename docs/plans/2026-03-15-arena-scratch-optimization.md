# Arena-Based Scratch Allocation for Graph Algorithms

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace buddy-allocator-based `scratch_alloc()` with bump arena allocation in graph algorithm kernels to eliminate 50% memory waste from power-of-2 rounding.

**Architecture:** The C engine already has `td_scratch_arena_t` (bump allocator backed by 64KB buddy blocks, 16-byte aligned). Graph algorithm kernels (`exec_pagerank`, `exec_connected_comp`, `exec_dijkstra`, `exec_louvain`) currently use `scratch_alloc()` which routes to `td_alloc()` (buddy allocator, power-of-2 rounding). Switch them to use a local `td_scratch_arena_t` — exact-size allocations, single `arena_reset()` frees everything.

**Tech Stack:** C17, existing `td_scratch_arena_t` API.

**Impact:** ~50% memory reduction for graph algorithm scratch buffers (e.g., Louvain on 1M nodes: 96MB → 48MB).

---

## File Structure

| File | Change |
|------|--------|
| `vendor/teide/src/ops/exec.c` | Refactor `exec_pagerank`, `exec_connected_comp`, `exec_dijkstra`, `exec_louvain` to use arena |

No new files. No API changes. No Rust changes.

---

## Task 1: Refactor Graph Algorithm Kernels to Use Arena

**Files:**
- Modify: `vendor/teide/src/ops/exec.c`

- [ ] **Step 1: Refactor `exec_pagerank` to use arena**

Replace individual `scratch_alloc`/`scratch_free` with arena:

```c
static td_t* exec_pagerank(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n       = rel->fwd.n_nodes;
    uint16_t iters  = ext->graph.max_iter;
    double damping  = ext->graph.damping;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    /* Arena for all scratch memory — freed in one shot */
    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    double* rank     = (double*)td_scratch_arena_push(&arena, n * sizeof(double), 16);
    double* rank_new = (double*)td_scratch_arena_push(&arena, n * sizeof(double), 16);
    if (!rank || !rank_new) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* ... algorithm unchanged ... */

    /* Build output table (uses buddy allocator — these are long-lived) */
    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* rank_vec = td_vec_new(TD_F64, n);
    if (!node_vec || !rank_vec) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* ... fill output ... */

    td_scratch_arena_reset(&arena);  /* free all scratch at once */
    return result;
}
```

- [ ] **Step 2: Refactor `exec_connected_comp` to use arena**

Same pattern — replace `scratch_alloc(label)` + `scratch_free(label)` with arena push + reset.

- [ ] **Step 3: Refactor `exec_dijkstra` to use arena**

Replace 5 individual `scratch_alloc`/`scratch_free` calls (dist, visited, depth, prev, heap) with arena:

```c
    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    double*        dist    = (double*)td_scratch_arena_push(&arena, n * sizeof(double), 16);
    bool*          visited = (bool*)td_scratch_arena_push(&arena, n * sizeof(bool), 16);
    int64_t*       prev    = (int64_t*)td_scratch_arena_push(&arena, n * sizeof(int64_t), 16);
    int64_t*       depth   = (int64_t*)td_scratch_arena_push(&arena, n * sizeof(int64_t), 16);
    dijk_entry_t*  heap    = (dijk_entry_t*)td_scratch_arena_push(&arena, heap_cap * sizeof(dijk_entry_t), 16);
    if (!dist || !visited || !prev || !depth || !heap) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* ... algorithm ... */

    td_scratch_arena_reset(&arena);
    return result;
```

This also simplifies error handling — no need to track which buffers to free individually.

- [ ] **Step 4: Refactor `exec_louvain` to use arena**

Replace 6 individual scratch buffers (community, degree, comm_tot, comm_int, k_i_in, remap) with arena.

- [ ] **Step 5: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 6: Run all tests**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS (behavior unchanged, only allocation strategy differs)

- [ ] **Step 7: Commit**

```bash
git add -f vendor/teide/src/ops/exec.c
git commit -m "perf(engine): use bump arena for graph algorithm scratch buffers

Replaces buddy-allocator scratch_alloc() with td_scratch_arena_t in
PageRank, connected components, Dijkstra, and Louvain kernels.
Eliminates ~50% memory waste from power-of-2 rounding on large buffers.
Single arena_reset() replaces multiple scratch_free() calls."
```

---

## Memory Savings

| Algorithm | 1M nodes | Before (buddy) | After (arena) | Saved |
|---|---|---|---|---|
| PageRank | 2 × 8MB scratch | 32MB | 16MB | 16MB (50%) |
| Connected Components | 1 × 8MB scratch | 16MB | 8MB | 8MB (50%) |
| Dijkstra | ~30MB scratch | ~55MB | ~30MB | ~25MB (45%) |
| Louvain | 6 × 8MB scratch | 96MB | 48MB | 48MB (50%) |
