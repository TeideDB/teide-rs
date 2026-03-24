# Graph Algorithms Batch 1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add degree centrality, topological sort, and DFS traversal to the graph engine — 3 trivial O(n+m) algorithms that operate directly on CSR arrays.

**Architecture:** Each algorithm follows the established pattern: opcode constant → DAG constructor in graph.c → executor kernel in exec.c → result as `TD_TABLE`. All scratch memory via `td_scratch_arena_t`. Tests in test_csr.c using the existing `make_edge_table` helper (4 nodes, 6 edges: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0).

**Tech Stack:** Pure C17. Build: `cmake --build build`. Test: `./build/test_teide --suite /csr`.

**Design doc:** `docs/plans/2026-03-23-graph-algorithms-design.md`

**Reference graph (used in all tests):**
```
  0 → 1, 0 → 2, 1 → 2, 1 → 3, 2 → 3, 3 → 0
  (4 nodes, 6 edges, cyclic)
```

---

### Task 1: Opcodes and declarations [x]

**Files:**
- Modify: `include/teide/td.h:481-487` (add opcodes)
- Modify: `include/teide/td.h:1000-1007` (add declarations)

**Step 1: Add opcodes**

After `OP_LOUVAIN` (line 481) and before the vector similarity opcodes, add:

```c
/* Opcodes — Graph algorithms (batch 1) */
#define OP_DEGREE_CENT     92   /* degree centrality                  */
#define OP_TOPSORT         93   /* topological sort (Kahn's)          */
#define OP_DFS             94   /* depth-first search traversal       */
```

**Step 2: Add declarations**

After the `td_louvain` declaration (line 1007), add:

```c
td_op_t* td_degree_cent(td_graph_t* g, td_rel_t* rel);
td_op_t* td_topsort(td_graph_t* g, td_rel_t* rel);
td_op_t* td_dfs(td_graph_t* g, td_op_t* src, td_rel_t* rel, uint8_t max_depth);
```

**Step 3: Build**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds (no references to new opcodes yet)

**Step 4: Commit**

```bash
git add include/teide/td.h
git commit -m "feat: add opcodes and declarations for degree centrality, topsort, DFS"
```

---

### Task 2: Degree centrality — DAG constructor + executor + test [x]

**Files:**
- Modify: `src/ops/graph.c` (add td_degree_cent)
- Modify: `src/ops/exec.c` (add exec_degree_cent + switch case)
- Test: `test/test_csr.c`

**Step 1: Write failing test**

Add to `test/test_csr.c`:

```c
static MunitResult test_degree_cent(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* dc = td_degree_cent(g, rel);
    munit_assert_ptr_not_null(dc);

    td_t* result = td_execute(g, dc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* Graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0
     * Node 0: out=2, in=1, total=3
     * Node 1: out=2, in=1, total=3
     * Node 2: out=1, in=2, total=3
     * Node 3: out=1, in=2, total=3 */
    int64_t out_sym = td_sym_intern("_out_degree", 11);
    int64_t in_sym  = td_sym_intern("_in_degree", 10);
    int64_t deg_sym = td_sym_intern("_degree", 7);

    td_t* out_col = td_table_get_col(result, out_sym);
    td_t* in_col  = td_table_get_col(result, in_sym);
    td_t* deg_col = td_table_get_col(result, deg_sym);

    munit_assert_ptr_not_null(out_col);
    munit_assert_ptr_not_null(in_col);
    munit_assert_ptr_not_null(deg_col);

    int64_t* out_d = (int64_t*)td_data(out_col);
    int64_t* in_d  = (int64_t*)td_data(in_col);
    int64_t* deg_d = (int64_t*)td_data(deg_col);

    munit_assert_int64(out_d[0], ==, 2);
    munit_assert_int64(in_d[0],  ==, 1);
    munit_assert_int64(deg_d[0], ==, 3);

    munit_assert_int64(out_d[2], ==, 1);
    munit_assert_int64(in_d[2],  ==, 2);
    munit_assert_int64(deg_d[2], ==, 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Register in the `csr_tests[]` array:
```c
{ "/degree_cent", test_degree_cent, NULL, NULL, 0, NULL },
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake --build . && ./test_teide --suite /csr/degree_cent`
Expected: FAIL — td_degree_cent not defined (linker error) or returns NULL

**Step 3: Implement DAG constructor**

In `src/ops/graph.c`, after `td_louvain`:

```c
td_op_t* td_degree_cent(td_graph_t* g, td_rel_t* rel) {
    if (!g || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode   = OP_DEGREE_CENT;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel     = rel;
    ext->graph.direction = 2;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 4: Implement executor kernel**

In `src/ops/exec.c`, after `exec_louvain`:

```c
/* --------------------------------------------------------------------------
 * exec_degree_cent: in/out/total degree from CSR offsets. O(n).
 * -------------------------------------------------------------------------- */
static td_t* exec_degree_cent(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* rev_off = (int64_t*)td_data(rel->rev.offsets);

    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* in_vec   = td_vec_new(TD_I64, n);
    td_t* out_vec  = td_vec_new(TD_I64, n);
    td_t* deg_vec  = td_vec_new(TD_I64, n);
    if (!node_vec || TD_IS_ERR(node_vec) ||
        !in_vec   || TD_IS_ERR(in_vec)   ||
        !out_vec  || TD_IS_ERR(out_vec)  ||
        !deg_vec  || TD_IS_ERR(deg_vec)) {
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        if (in_vec   && !TD_IS_ERR(in_vec))   td_release(in_vec);
        if (out_vec  && !TD_IS_ERR(out_vec))  td_release(out_vec);
        if (deg_vec  && !TD_IS_ERR(deg_vec))  td_release(deg_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata   = (int64_t*)td_data(node_vec);
    int64_t* in_data = (int64_t*)td_data(in_vec);
    int64_t* out_data= (int64_t*)td_data(out_vec);
    int64_t* deg_data= (int64_t*)td_data(deg_vec);

    for (int64_t i = 0; i < n; i++) {
        ndata[i]    = i;
        out_data[i] = fwd_off[i + 1] - fwd_off[i];
        in_data[i]  = rev_off[i + 1] - rev_off[i];
        deg_data[i] = out_data[i] + in_data[i];
    }
    node_vec->len = n;
    in_vec->len   = n;
    out_vec->len  = n;
    deg_vec->len  = n;

    td_t* result = td_table_new(4);
    if (!result || TD_IS_ERR(result)) {
        td_release(node_vec); td_release(in_vec);
        td_release(out_vec);  td_release(deg_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    result = td_table_add_col(result, td_sym_intern("_in_degree", 10), in_vec);
    td_release(in_vec);
    result = td_table_add_col(result, td_sym_intern("_out_degree", 11), out_vec);
    td_release(out_vec);
    result = td_table_add_col(result, td_sym_intern("_degree", 7), deg_vec);
    td_release(deg_vec);

    return result;
}
```

**Step 5: Add switch case**

In the executor dispatch switch (after OP_LOUVAIN case ~line 14570), add:

```c
        case OP_DEGREE_CENT: {
            return exec_degree_cent(g, op);
        }
```

**Step 6: Run test**

Run: `cd build && cmake --build . && ./test_teide --suite /csr/degree_cent`
Expected: PASS

**Step 7: Commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: degree centrality — in/out/total degree from CSR offsets"
```

---

### Task 3: Topological sort — DAG constructor + executor + test [x]

**Files:**
- Modify: `src/ops/graph.c`
- Modify: `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing tests**

Need a DAG (no cycles) for topsort. Add a helper and two tests:

```c
/* DAG: 0→1, 0→2, 1→3, 2→3 (4 nodes, 4 edges, no cycles) */
static td_t* make_dag_edge_table(void) {
    int64_t src_data[] = {0, 0, 1, 2};
    int64_t dst_data[] = {1, 2, 3, 3};
    int64_t n = 4;

    td_t* src_vec = td_vec_from_raw(TD_I64, src_data, n);
    td_t* dst_vec = td_vec_from_raw(TD_I64, dst_data, n);

    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);

    td_t* tbl = td_table_new(2);
    tbl = td_table_add_col(tbl, src_sym, src_vec);
    td_release(src_vec);
    tbl = td_table_add_col(tbl, dst_sym, dst_vec);
    td_release(dst_vec);
    return tbl;
}

static MunitResult test_topsort(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_dag_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* ts = td_topsort(g, rel);
    munit_assert_ptr_not_null(ts);

    td_t* result = td_execute(g, ts);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* DAG: 0→1, 0→2, 1→3, 2→3
     * Valid orderings: 0 must come before 1,2; 1,2 before 3 */
    int64_t order_sym = td_sym_intern("_order", 6);
    td_t* order_col = td_table_get_col(result, order_sym);
    munit_assert_ptr_not_null(order_col);
    int64_t* ord = (int64_t*)td_data(order_col);

    /* Node 0 must have lowest order */
    munit_assert_int64(ord[0], <, ord[1]);
    munit_assert_int64(ord[0], <, ord[2]);
    /* Node 3 must have highest order */
    munit_assert_int64(ord[3], >, ord[1]);
    munit_assert_int64(ord[3], >, ord[2]);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_topsort_cycle(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Cyclic graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0 */
    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* ts = td_topsort(g, rel);
    td_t* result = td_execute(g, ts);

    /* Cycle detected — should return error */
    munit_assert_true(TD_IS_ERR(result));

    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Register:
```c
{ "/topsort",       test_topsort,       NULL, NULL, 0, NULL },
{ "/topsort_cycle", test_topsort_cycle, NULL, NULL, 0, NULL },
```

**Step 2: Run to verify failure**

Expected: FAIL

**Step 3: Implement DAG constructor**

```c
td_op_t* td_topsort(td_graph_t* g, td_rel_t* rel) {
    if (!g || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode   = OP_TOPSORT;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel     = rel;
    ext->graph.direction = 0;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 4: Implement executor kernel (Kahn's algorithm)**

```c
/* --------------------------------------------------------------------------
 * exec_topsort: topological sort via Kahn's algorithm. O(n+m).
 * Returns error if graph contains a cycle.
 * -------------------------------------------------------------------------- */
static td_t* exec_topsort(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);
    int64_t* rev_off = (int64_t*)td_data(rel->rev.offsets);

    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    int64_t* in_deg = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* queue  = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* order  = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    if (!in_deg || !queue || !order) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* Compute in-degrees from reverse CSR */
    for (int64_t i = 0; i < n; i++)
        in_deg[i] = rev_off[i + 1] - rev_off[i];

    /* Enqueue zero-degree nodes */
    int64_t head = 0, tail = 0;
    for (int64_t i = 0; i < n; i++) {
        if (in_deg[i] == 0) queue[tail++] = i;
    }

    /* BFS — decrement in-degrees, enqueue new zeros */
    int64_t count = 0;
    while (head < tail) {
        int64_t v = queue[head++];
        order[v] = count++;

        int64_t start = fwd_off[v];
        int64_t end   = fwd_off[v + 1];
        for (int64_t j = start; j < end; j++) {
            int64_t u = fwd_tgt[j];
            if (--in_deg[u] == 0) queue[tail++] = u;
        }
    }

    /* Cycle detection: not all nodes processed */
    if (count < n) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_SCHEMA);  /* cycle detected */
    }

    /* Build result */
    td_t* node_vec  = td_vec_new(TD_I64, n);
    td_t* order_vec = td_vec_new(TD_I64, n);
    if (!node_vec || TD_IS_ERR(node_vec) || !order_vec || TD_IS_ERR(order_vec)) {
        td_scratch_arena_reset(&arena);
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        if (order_vec && !TD_IS_ERR(order_vec)) td_release(order_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)td_data(node_vec);
    int64_t* odata = (int64_t*)td_data(order_vec);
    for (int64_t i = 0; i < n; i++) {
        ndata[i] = i;
        odata[i] = order[i];
    }
    node_vec->len  = n;
    order_vec->len = n;

    td_scratch_arena_reset(&arena);

    td_t* result = td_table_new(2);
    if (!result || TD_IS_ERR(result)) {
        td_release(node_vec); td_release(order_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    result = td_table_add_col(result, td_sym_intern("_order", 6), order_vec);
    td_release(order_vec);

    return result;
}
```

**Step 5: Add switch case**

```c
        case OP_TOPSORT: {
            return exec_topsort(g, op);
        }
```

**Step 6: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /csr/topsort && ./test_teide --suite /csr/topsort_cycle`
Expected: PASS

**Step 7: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 8: Commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: topological sort — Kahn's algorithm with cycle detection"
```

---

### Task 4: DFS traversal — DAG constructor + executor + test [x]

**Files:**
- Modify: `src/ops/graph.c`
- Modify: `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_dfs(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* Build a table with source node = 0 */
    td_t* src_tbl = td_table_new(1);
    td_t* src_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    src_tbl = td_table_add_col(src_tbl, td_sym_intern("src", 3), src_vec);
    td_release(src_vec);

    td_graph_t* g = td_graph_new(src_tbl);
    td_op_t* src_op = td_scan(g, "src");
    td_op_t* dfs = td_dfs(g, src_op, rel, 255);
    munit_assert_ptr_not_null(dfs);

    td_t* result = td_execute(g, dfs);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* All 4 nodes should be reachable from node 0 */
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* Node 0 should have depth 0 and parent -1 */
    int64_t node_sym   = td_sym_intern("_node", 5);
    int64_t depth_sym  = td_sym_intern("_depth", 6);
    int64_t parent_sym = td_sym_intern("_parent", 7);

    td_t* node_col   = td_table_get_col(result, node_sym);
    td_t* depth_col  = td_table_get_col(result, depth_sym);
    td_t* parent_col = td_table_get_col(result, parent_sym);
    munit_assert_ptr_not_null(node_col);
    munit_assert_ptr_not_null(depth_col);
    munit_assert_ptr_not_null(parent_col);

    int64_t* nodes   = (int64_t*)td_data(node_col);
    int64_t* depths  = (int64_t*)td_data(depth_col);
    int64_t* parents = (int64_t*)td_data(parent_col);

    /* First node in DFS order should be source (node 0) */
    munit_assert_int64(nodes[0], ==, 0);
    munit_assert_int64(depths[0], ==, 0);
    munit_assert_int64(parents[0], ==, -1);

    /* All depths should be >= 0 */
    for (int64_t i = 0; i < 4; i++) {
        munit_assert_int64(depths[i], >=, 0);
    }

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(src_tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_dfs_max_depth(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_dag_edge_table();  /* 0→1, 0→2, 1→3, 2→3 */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* src_tbl = td_table_new(1);
    td_t* src_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    src_tbl = td_table_add_col(src_tbl, td_sym_intern("src", 3), src_vec);
    td_release(src_vec);

    td_graph_t* g = td_graph_new(src_tbl);
    td_op_t* src_op = td_scan(g, "src");
    td_op_t* dfs = td_dfs(g, src_op, rel, 1);  /* max depth = 1 */

    td_t* result = td_execute(g, dfs);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* With max_depth=1 from node 0: nodes 0, 1, 2 (not 3) */
    munit_assert_int(td_table_nrows(result), ==, 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(src_tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Register:
```c
{ "/dfs",           test_dfs,           NULL, NULL, 0, NULL },
{ "/dfs_max_depth", test_dfs_max_depth, NULL, NULL, 0, NULL },
```

**Step 2: Run to verify failure**

Expected: FAIL

**Step 3: Implement DAG constructor**

```c
td_op_t* td_dfs(td_graph_t* g, td_op_t* src, td_rel_t* rel, uint8_t max_depth) {
    if (!g || !src || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    uint32_t src_id = src->id;
    src = &g->nodes[src_id];

    ext->base.opcode   = OP_DFS;
    ext->base.arity    = 1;
    ext->base.inputs[0] = src;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel       = rel;
    ext->graph.direction  = 0;
    ext->graph.max_depth  = max_depth;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 4: Implement executor kernel**

```c
/* --------------------------------------------------------------------------
 * exec_dfs: depth-first search from source node. O(n+m).
 * -------------------------------------------------------------------------- */
static td_t* exec_dfs(td_graph_t* g, td_op_t* op, td_t* src_val) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    uint8_t max_depth = ext->graph.max_depth;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    /* Get source node ID */
    int64_t start_node;
    if (td_is_atom(src_val)) {
        start_node = src_val->i64;
    } else {
        start_node = ((int64_t*)td_data(src_val))[0];
    }
    if (start_node < 0 || start_node >= n) return TD_ERR_PTR(TD_ERR_RANGE);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);

    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    /* DFS uses paired stack: node + depth */
    int64_t* stack_node  = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* stack_depth = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    uint8_t* visited     = (uint8_t*)td_scratch_arena_push(&arena, (size_t)n);
    int64_t* res_node    = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* res_depth   = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* res_parent  = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* parent_map  = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    if (!stack_node || !stack_depth || !visited ||
        !res_node || !res_depth || !res_parent || !parent_map) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    memset(visited, 0, (size_t)n);
    for (int64_t i = 0; i < n; i++) parent_map[i] = -1;

    /* Push source */
    int64_t sp = 0;
    stack_node[sp]  = start_node;
    stack_depth[sp] = 0;
    sp++;

    int64_t count = 0;

    while (sp > 0) {
        sp--;
        int64_t v = stack_node[sp];
        int64_t d = stack_depth[sp];

        if (visited[v]) continue;
        visited[v] = 1;

        res_node[count]   = v;
        res_depth[count]  = d;
        res_parent[count] = parent_map[v];
        count++;

        if (d < max_depth) {
            /* Push neighbors in reverse order so first neighbor is visited first */
            int64_t start = fwd_off[v];
            int64_t end   = fwd_off[v + 1];
            for (int64_t j = end - 1; j >= start; j--) {
                int64_t u = fwd_tgt[j];
                if (!visited[u]) {
                    stack_node[sp]  = u;
                    stack_depth[sp] = d + 1;
                    parent_map[u]   = v;
                    sp++;
                }
            }
        }
    }

    /* Build result vectors */
    td_t* node_vec   = td_vec_new(TD_I64, count);
    td_t* depth_vec  = td_vec_new(TD_I64, count);
    td_t* parent_vec = td_vec_new(TD_I64, count);
    if (!node_vec || TD_IS_ERR(node_vec) ||
        !depth_vec || TD_IS_ERR(depth_vec) ||
        !parent_vec || TD_IS_ERR(parent_vec)) {
        td_scratch_arena_reset(&arena);
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        if (depth_vec && !TD_IS_ERR(depth_vec)) td_release(depth_vec);
        if (parent_vec && !TD_IS_ERR(parent_vec)) td_release(parent_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    memcpy(td_data(node_vec),   res_node,   (size_t)count * sizeof(int64_t));
    memcpy(td_data(depth_vec),  res_depth,  (size_t)count * sizeof(int64_t));
    memcpy(td_data(parent_vec), res_parent, (size_t)count * sizeof(int64_t));
    node_vec->len   = count;
    depth_vec->len  = count;
    parent_vec->len = count;

    td_scratch_arena_reset(&arena);

    td_t* result = td_table_new(3);
    if (!result || TD_IS_ERR(result)) {
        td_release(node_vec); td_release(depth_vec); td_release(parent_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    result = td_table_add_col(result, td_sym_intern("_depth", 6), depth_vec);
    td_release(depth_vec);
    result = td_table_add_col(result, td_sym_intern("_parent", 7), parent_vec);
    td_release(parent_vec);

    return result;
}
```

**Step 5: Add switch case**

DFS takes a source input, same pattern as Dijkstra:

```c
        case OP_DFS: {
            td_t* src = exec_node(g, op->inputs[0]);
            if (!src || TD_IS_ERR(src)) return src;
            td_t* result = exec_dfs(g, op, src);
            td_release(src);
            return result;
        }
```

**Step 6: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /csr/dfs && ./test_teide --suite /csr/dfs_max_depth`
Expected: PASS

**Step 7: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 8: Commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: DFS traversal — stack-based with depth limit, parent tracking"
```

---

## Summary

| Task | Algorithm | Complexity | Result Columns |
|------|-----------|------------|----------------|
| 1 | Opcodes + declarations | — | — |
| 2 | Degree centrality | O(n) | `_node`, `_in_degree`, `_out_degree`, `_degree` |
| 3 | Topological sort | O(n+m) | `_node`, `_order` |
| 4 | DFS traversal | O(n+m) | `_node`, `_depth`, `_parent` |
