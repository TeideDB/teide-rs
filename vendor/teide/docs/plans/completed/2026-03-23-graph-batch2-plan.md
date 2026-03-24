# Graph Algorithms Batch 2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add A*, Yen's k-shortest paths, clustering coefficients, and random walk — 4 medium-complexity graph algorithms with both C kernels and Rust/SQL integration.

**Architecture:** Each algorithm: C opcode + DAG constructor + executor kernel (in `teide/`) → FFI binding + Graph method + SQL table function (in `teide-rs/`). A* and Yen's reuse Dijkstra's `dijk_entry_t` heap. All scratch memory via `td_scratch_arena_t`.

**Tech Stack:** C17 (teide), Rust (teide-rs). Build C: `cmake --build build`. Build Rust: `cd ../teide-rs && cargo build --all-features`. Test C: `./build/test_teide --suite /csr`. Test Rust: `cd ../teide-rs && cargo test --all-features`.

**Design doc:** `docs/plans/2026-03-23-graph-algorithms-design.md`

**Two repos:**
- `teide/` — C engine: opcodes, DAG constructors, executor kernels, C tests
- `teide-rs/` — Rust bindings: `ffi.rs`, `engine.rs`, `planner.rs`, `pgq.rs`, Rust tests

---

### Task 1: Opcodes, struct extension, and declarations (C only)

- [x] Add opcodes, struct extension, and declarations to td.h
- [x] Build succeeds

**Files:**
- Modify: `include/teide/td.h:488` (add opcodes after OP_DFS)
- Modify: `include/teide/td.h:593-594` (add coord_col_syms to graph struct)
- Modify: `include/teide/td.h:1007+` (add declarations)

**Step 1: Add opcodes**

After `OP_DFS 94`:

```c
#define OP_ASTAR           95   /* A* shortest path (coordinate heuristic) */
#define OP_K_SHORTEST      96   /* Yen's k-shortest paths                 */
#define OP_CLUSTER_COEFF   97   /* clustering coefficients                */
#define OP_RANDOM_WALK     98   /* random walk traversal                  */
```

**Step 2: Add coord_col_syms to graph struct**

After `weight_col_sym` (line ~598):

```c
            int64_t   weight_col_sym; /* Dijkstra/A*/Yen weight column name  */
            int64_t   coord_col_syms[2]; /* A*: lat/lon property column names */
```

**Step 3: Add declarations**

```c
td_op_t* td_astar(td_graph_t* g, td_op_t* src, td_op_t* dst,
                  td_rel_t* rel, const char* weight_col,
                  const char* lat_col, const char* lon_col, uint8_t max_depth);
td_op_t* td_k_shortest(td_graph_t* g, td_op_t* src, td_op_t* dst,
                       td_rel_t* rel, const char* weight_col, uint16_t k);
td_op_t* td_cluster_coeff(td_graph_t* g, td_rel_t* rel);
td_op_t* td_random_walk(td_graph_t* g, td_op_t* src, td_rel_t* rel,
                        uint16_t walk_length);
```

**Step 4: Build**

Run: `cmake --build build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add include/teide/td.h
git commit -m "feat: add opcodes and declarations for A*, k-shortest, clustering coeff, random walk"
```

---

### Task 2: Clustering coefficients — C kernel + test

- [x] Write failing test
- [x] Implement DAG constructor and executor kernel
- [x] Tests pass

Simplest of the batch — no inputs, no edge weights. Triangle counting via neighbor set intersection.

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_cluster_coeff(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0
     * Treat as undirected: node 0 neighbors={1,2,3}, node 1 neighbors={0,2,3}, etc.
     * All 4 nodes form a near-clique. */
    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* cc = td_cluster_coeff(g, rel);
    munit_assert_ptr_not_null(cc);

    td_t* result = td_execute(g, cc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* All coefficients should be between 0.0 and 1.0 */
    int64_t coeff_sym = td_sym_intern("_coefficient", 12);
    td_t* coeff_col = td_table_get_col(result, coeff_sym);
    munit_assert_ptr_not_null(coeff_col);
    double* coeffs = (double*)td_data(coeff_col);
    for (int i = 0; i < 4; i++) {
        munit_assert_double(coeffs[i], >=, 0.0);
        munit_assert_double(coeffs[i], <=, 1.0);
    }

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

**Step 2: Implement DAG constructor**

```c
td_op_t* td_cluster_coeff(td_graph_t* g, td_rel_t* rel) {
    if (!g || !rel) return NULL;
    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;
    ext->base.opcode   = OP_CLUSTER_COEFF;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel     = rel;
    ext->graph.direction = 2;
    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 3: Implement executor kernel**

For each node v, get undirected neighbors (union of fwd + rev). For each pair (u, w) of neighbors, check if edge (u, w) exists by scanning u's fwd neighbors for w. Count triangles. `coeff = 2*triangles / (d*(d-1))` where d = undirected degree.

Use scratch arena for: `int64_t* neighbors` (merged fwd+rev, deduplicated), `uint8_t* neighbor_set` (bitset for O(1) lookup).

```c
static td_t* exec_cluster_coeff(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);
    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);
    int64_t* rev_off = (int64_t*)td_data(rel->rev.offsets);
    int64_t* rev_tgt = (int64_t*)td_data(rel->rev.targets);

    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    /* Bitset for O(1) neighbor lookup: one byte per node */
    uint8_t* nbr_set = (uint8_t*)td_scratch_arena_push(&arena, (size_t)n);
    /* Merged neighbor list (max degree = fwd+rev) */
    int64_t max_deg = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t d = (fwd_off[i+1]-fwd_off[i]) + (rev_off[i+1]-rev_off[i]);
        if (d > max_deg) max_deg = d;
    }
    int64_t* nbrs = (int64_t*)td_scratch_arena_push(&arena, (size_t)max_deg * sizeof(int64_t));
    if (!nbr_set || !nbrs) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    td_t* node_vec  = td_vec_new(TD_I64, n);
    td_t* coeff_vec = td_vec_new(TD_F64, n);
    if (!node_vec || TD_IS_ERR(node_vec) || !coeff_vec || TD_IS_ERR(coeff_vec)) {
        td_scratch_arena_reset(&arena);
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        if (coeff_vec && !TD_IS_ERR(coeff_vec)) td_release(coeff_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)td_data(node_vec);
    double*  cdata = (double*)td_data(coeff_vec);

    for (int64_t v = 0; v < n; v++) {
        ndata[v] = v;
        memset(nbr_set, 0, (size_t)n);

        /* Build undirected neighbor set for v */
        int64_t deg = 0;
        for (int64_t j = fwd_off[v]; j < fwd_off[v+1]; j++) {
            int64_t u = fwd_tgt[j];
            if (u != v && !nbr_set[u]) { nbr_set[u] = 1; nbrs[deg++] = u; }
        }
        for (int64_t j = rev_off[v]; j < rev_off[v+1]; j++) {
            int64_t u = rev_tgt[j];
            if (u != v && !nbr_set[u]) { nbr_set[u] = 1; nbrs[deg++] = u; }
        }

        if (deg < 2) { cdata[v] = 0.0; continue; }

        /* Count triangles: for each pair of neighbors, check edge */
        int64_t triangles = 0;
        for (int64_t a = 0; a < deg; a++) {
            int64_t u = nbrs[a];
            /* Check which of v's other neighbors are also neighbors of u */
            for (int64_t j = fwd_off[u]; j < fwd_off[u+1]; j++) {
                if (nbr_set[fwd_tgt[j]] && fwd_tgt[j] != v) triangles++;
            }
            for (int64_t j = rev_off[u]; j < rev_off[u+1]; j++) {
                if (nbr_set[rev_tgt[j]] && rev_tgt[j] != v) triangles++;
            }
        }
        /* Each triangle counted twice (once from each endpoint of the edge) */
        cdata[v] = (double)triangles / (double)(deg * (deg - 1));
    }

    node_vec->len  = n;
    coeff_vec->len = n;
    td_scratch_arena_reset(&arena);

    td_t* result = td_table_new(2);
    if (!result || TD_IS_ERR(result)) {
        td_release(node_vec); td_release(coeff_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    result = td_table_add_col(result, td_sym_intern("_coefficient", 12), coeff_vec);
    td_release(coeff_vec);
    return result;
}
```

Add switch case:
```c
        case OP_CLUSTER_COEFF: {
            return exec_cluster_coeff(g, op);
        }
```

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /csr/cluster_coeff`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: clustering coefficients — triangle counting with neighbor bitset"
```

---

### Task 3: Random walk — C kernel + test

- [x] Write failing test
- [x] Implement DAG constructor and executor kernel
- [x] Tests pass

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_random_walk(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* src_tbl = td_table_new(1);
    td_t* src_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    src_tbl = td_table_add_col(src_tbl, td_sym_intern("src", 3), src_vec);
    td_release(src_vec);

    td_graph_t* g = td_graph_new(src_tbl);
    td_op_t* src_op = td_scan(g, "src");
    td_op_t* rw = td_random_walk(g, src_op, rel, 10);
    munit_assert_ptr_not_null(rw);

    td_t* result = td_execute(g, rw);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* Should have 11 rows (start + 10 steps) */
    munit_assert_int(td_table_nrows(result), ==, 11);

    /* First node should be source (0) */
    int64_t node_sym = td_sym_intern("_node", 5);
    td_t* node_col = td_table_get_col(result, node_sym);
    int64_t* nodes = (int64_t*)td_data(node_col);
    munit_assert_int64(nodes[0], ==, 0);

    /* All nodes should be valid (0..3) */
    for (int i = 0; i < 11; i++) {
        munit_assert_int64(nodes[i], >=, 0);
        munit_assert_int64(nodes[i], <, 4);
    }

    /* Step column should be 0..10 */
    int64_t step_sym = td_sym_intern("_step", 5);
    td_t* step_col = td_table_get_col(result, step_sym);
    int64_t* steps = (int64_t*)td_data(step_col);
    for (int i = 0; i < 11; i++) {
        munit_assert_int64(steps[i], ==, i);
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
```

**Step 2: Implement DAG constructor**

```c
td_op_t* td_random_walk(td_graph_t* g, td_op_t* src, td_rel_t* rel,
                        uint16_t walk_length) {
    if (!g || !src || !rel) return NULL;
    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;
    uint32_t src_id = src->id;
    src = &g->nodes[src_id];
    ext->base.opcode    = OP_RANDOM_WALK;
    ext->base.arity     = 1;
    ext->base.inputs[0] = src;
    ext->base.out_type  = TD_TABLE;
    ext->base.est_rows  = walk_length + 1;
    ext->graph.rel      = rel;
    ext->graph.max_iter = walk_length;
    ext->graph.direction = 0;
    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 3: Implement executor kernel**

Uses xorshift64 PRNG seeded from source node ID + a constant:

```c
static td_t* exec_random_walk(td_graph_t* g, td_op_t* op, td_t* src_val) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);
    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    uint16_t walk_len = ext->graph.max_iter;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t start_node;
    if (td_is_atom(src_val)) start_node = src_val->i64;
    else start_node = ((int64_t*)td_data(src_val))[0];
    if (start_node < 0 || start_node >= n) return TD_ERR_PTR(TD_ERR_RANGE);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);

    int64_t total = (int64_t)walk_len + 1;
    td_t* step_vec = td_vec_new(TD_I64, total);
    td_t* node_vec = td_vec_new(TD_I64, total);
    if (!step_vec || TD_IS_ERR(step_vec) || !node_vec || TD_IS_ERR(node_vec)) {
        if (step_vec && !TD_IS_ERR(step_vec)) td_release(step_vec);
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* sdata = (int64_t*)td_data(step_vec);
    int64_t* ndata = (int64_t*)td_data(node_vec);

    /* xorshift64 PRNG */
    uint64_t rng = (uint64_t)start_node * 6364136223846793005ULL + 1442695040888963407ULL;
    if (rng == 0) rng = 1;

    int64_t current = start_node;
    for (int64_t i = 0; i < total; i++) {
        sdata[i] = i;
        ndata[i] = current;
        if (i < walk_len) {
            int64_t deg = fwd_off[current + 1] - fwd_off[current];
            if (deg == 0) break;  /* stuck at dead end */
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            int64_t pick = (int64_t)(rng % (uint64_t)deg);
            current = fwd_tgt[fwd_off[current] + pick];
        }
    }

    /* Actual length may be shorter if dead end hit */
    step_vec->len = /* count written */ total;  /* adjust if break hit */
    node_vec->len = total;

    td_t* result = td_table_new(2);
    if (!result || TD_IS_ERR(result)) {
        td_release(step_vec); td_release(node_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_step", 5), step_vec);
    td_release(step_vec);
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    return result;
}
```

Note: The implementer should handle the early break on dead-end properly — track actual count written and set `step_vec->len = count`, `node_vec->len = count`.

Add switch case (same pattern as DFS — takes src input):
```c
        case OP_RANDOM_WALK: {
            td_t* src = exec_node(g, op->inputs[0]);
            if (!src || TD_IS_ERR(src)) return src;
            td_t* result = exec_random_walk(g, op, src);
            td_release(src);
            return result;
        }
```

**Step 4: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: random walk — xorshift64 PRNG with dead-end handling"
```

---

### Task 4: A* shortest path — C kernel + test

- [x] Write failing test
- [x] Implement DAG constructor and executor kernel
- [x] Tests pass

Reuses Dijkstra's heap. Adds Euclidean heuristic from node coordinate columns.

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

Create a helper with coordinate properties:

```c
/* Weighted graph with lat/lon node properties:
 * 5 nodes, 6 edges:
 *   0→1 (w=1.0), 0→2 (w=4.0), 1→3 (w=2.0), 2→3 (w=1.0), 3→4 (w=3.0), 1→4 (w=10.0)
 * Node coordinates (for heuristic): 0=(0,0), 1=(1,0), 2=(0,2), 3=(2,1), 4=(3,0) */
static void make_astar_graph(td_t** out_edges, td_rel_t** out_rel,
                             td_t** out_node_props) {
    int64_t src[] = {0, 0, 1, 2, 3, 1};
    int64_t dst[] = {1, 2, 3, 3, 4, 4};
    double  wts[] = {1.0, 4.0, 2.0, 1.0, 3.0, 10.0};
    int64_t ne = 6;

    td_t* sv = td_vec_from_raw(TD_I64, src, ne);
    td_t* dv = td_vec_from_raw(TD_I64, dst, ne);
    td_t* wv = td_vec_new(TD_F64, ne);
    memcpy(td_data(wv), wts, sizeof(wts));
    wv->len = ne;

    td_t* edges = td_table_new(3);
    edges = td_table_add_col(edges, td_sym_intern("src", 3), sv); td_release(sv);
    edges = td_table_add_col(edges, td_sym_intern("dst", 3), dv); td_release(dv);
    edges = td_table_add_col(edges, td_sym_intern("weight", 6), wv); td_release(wv);

    *out_rel = td_rel_from_edges(edges, "src", "dst", 5, 5, false);
    td_rel_set_props(*out_rel, edges);
    *out_edges = edges;

    /* Node property table with lat/lon */
    double lat[] = {0.0, 1.0, 0.0, 2.0, 3.0};
    double lon[] = {0.0, 0.0, 2.0, 1.0, 0.0};
    td_t* nv = td_vec_from_raw(TD_I64, (int64_t[]){0,1,2,3,4}, 5);
    td_t* latv = td_vec_new(TD_F64, 5); memcpy(td_data(latv), lat, sizeof(lat)); latv->len = 5;
    td_t* lonv = td_vec_new(TD_F64, 5); memcpy(td_data(lonv), lon, sizeof(lon)); lonv->len = 5;

    td_t* np = td_table_new(3);
    np = td_table_add_col(np, td_sym_intern("_node", 5), nv); td_release(nv);
    np = td_table_add_col(np, td_sym_intern("lat", 3), latv); td_release(latv);
    np = td_table_add_col(np, td_sym_intern("lon", 3), lonv); td_release(lonv);
    *out_node_props = np;
}

static MunitResult test_astar(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges; td_rel_t* rel; td_t* node_props;
    make_astar_graph(&edges, &rel, &node_props);

    td_graph_t* g = td_graph_new(edges);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 4);
    td_op_t* as = td_astar(g, src, dst, rel, "weight", "lat", "lon", 255);
    munit_assert_ptr_not_null(as);

    td_t* result = td_execute(g, as);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* Should find path to node 4 with dist=6.0 (0→1→3→4: 1+2+3) */
    int64_t nrows = td_table_nrows(result);
    munit_assert_int64(nrows, >, 0);

    int64_t node_sym = td_sym_intern("_node", 5);
    int64_t dist_sym = td_sym_intern("_dist", 5);
    td_t* node_col = td_table_get_col(result, node_sym);
    td_t* dist_col = td_table_get_col(result, dist_sym);
    int64_t* nodes = (int64_t*)td_data(node_col);
    double* dists = (double*)td_data(dist_col);

    bool found = false;
    for (int64_t i = 0; i < nrows; i++) {
        if (nodes[i] == 4) {
            munit_assert_double(dists[i], >=, 5.99);
            munit_assert_double(dists[i], <=, 6.01);
            found = true;
        }
    }
    munit_assert_true(found);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(node_props);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Implement**

A* is Dijkstra with priority = `g_cost + h_cost`. The heuristic reads lat/lon from a node property table (passed via the ext node, or looked up from the rel's props). The `coord_col_syms[2]` field stores the interned names of the coordinate columns.

The implementer should:
1. DAG constructor: intern lat_col/lon_col into `coord_col_syms[0]`/`[1]`
2. Executor: load coordinate arrays from props table, compute Euclidean distance `sqrt((lat[v]-lat[dst])^2 + (lon[v]-lon[dst])^2)` as heuristic
3. Same heap, same result format as Dijkstra

Note: Node property coordinates need to be accessible. The simplest approach: the executor reads coordinates from `rel->fwd.props` (the edge property table also has node IDs, but that's per-edge). The design should store node properties as a separate table passed through the ext node. The implementer can use the `damping` field (unused by A*) to store a pointer to the node properties table, or add a `node_props` field to the graph struct.

**Simpler alternative**: Since A* with coordinates is the only consumer, store lat/lon as `double*` arrays resolved at construction time from the edge properties. The heuristic indexes by node ID directly.

**Step 3: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: A* shortest path — Dijkstra with Euclidean coordinate heuristic"
```

---

### Task 5: Yen's k-shortest paths — C kernel + test

- [x] Write failing test
- [x] Implement DAG constructor and executor kernel
- [x] Tests pass

**Files:**
- Modify: `src/ops/exec.c`
- Modify: `src/ops/graph.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_k_shortest(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges; td_rel_t* rel; td_t* node_props;
    make_astar_graph(&edges, &rel, &node_props);

    td_graph_t* g = td_graph_new(edges);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 4);
    td_op_t* ks = td_k_shortest(g, src, dst, rel, "weight", 3);
    munit_assert_ptr_not_null(ks);

    td_t* result = td_execute(g, ks);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* Should find at least 2 paths: 0→1→3→4 (6.0) and 0→2→3→4 (8.0) */
    int64_t nrows = td_table_nrows(result);
    munit_assert_int64(nrows, >=, 2);

    /* Check path_id column exists */
    int64_t pid_sym = td_sym_intern("_path_id", 8);
    td_t* pid_col = td_table_get_col(result, pid_sym);
    munit_assert_ptr_not_null(pid_col);

    /* First path should have lowest total distance */
    int64_t dist_sym = td_sym_intern("_dist", 5);
    td_t* dist_col = td_table_get_col(result, dist_sym);
    double* dists = (double*)td_data(dist_col);
    /* First row of path 0 should be source with dist 0 */
    munit_assert_double(dists[0], >=, -0.01);
    munit_assert_double(dists[0], <=, 0.01);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(node_props);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Implement**

Yen's algorithm:
1. Run Dijkstra to find shortest path P[0]
2. For k=1..K-1: for each spur node s on P[k-1], mask edges used by paths sharing the root (prefix), run Dijkstra from s to dst, combine root+spur
3. Keep a sorted candidate list; next shortest becomes P[k]

The implementer should extract a reusable `dijkstra_inner()` function from `exec_dijkstra` that takes edge mask arrays and returns dist+parent arrays, so Yen's can call it repeatedly without rebuilding the full executor context.

**Step 3: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: Yen's k-shortest paths — iterative Dijkstra with edge masking"
```

---

### Task 6: Rust bindings for all 4 algorithms (teide-rs)

- [x] Add FFI declarations and Graph methods
- [x] Add integration tests and SQL dispatch
- [x] Build and test both repos

**Files:**
- Modify: `../teide-rs/src/ffi.rs` (add FFI declarations)
- Modify: `../teide-rs/src/engine.rs` (add Graph methods)
- Test: `../teide-rs/tests/engine_api.rs`

**Step 1: Add FFI declarations**

In `teide-rs/src/ffi.rs`, after the existing graph algo FFI:

```rust
    pub fn td_cluster_coeff(
        g: *mut td_graph_t,
        rel: *mut td_rel_t,
    ) -> *mut td_op_t;

    pub fn td_random_walk(
        g: *mut td_graph_t,
        src: *mut td_op_t,
        rel: *mut td_rel_t,
        walk_length: u16,
    ) -> *mut td_op_t;

    pub fn td_astar(
        g: *mut td_graph_t,
        src: *mut td_op_t,
        dst: *mut td_op_t,
        rel: *mut td_rel_t,
        weight_col: *const c_char,
        lat_col: *const c_char,
        lon_col: *const c_char,
        max_depth: u8,
    ) -> *mut td_op_t;

    pub fn td_k_shortest(
        g: *mut td_graph_t,
        src: *mut td_op_t,
        dst: *mut td_op_t,
        rel: *mut td_rel_t,
        weight_col: *const c_char,
        k: u16,
    ) -> *mut td_op_t;
```

**Step 2: Add Graph methods**

In `teide-rs/src/engine.rs`:

```rust
    pub fn cluster_coeff(&self, rel: &'a Rel) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_cluster_coeff(self.raw, rel.ptr) })
    }

    pub fn random_walk(&self, src: Column, rel: &'a Rel, walk_length: u16) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_random_walk(self.raw, src.raw, rel.ptr, walk_length) })
    }

    pub fn astar(
        &self, src: Column, dst: Column, rel: &'a Rel,
        weight_col: &str, lat_col: &str, lon_col: &str, max_depth: u8,
    ) -> Result<Column> {
        let c_w = CString::new(weight_col).map_err(|_| Error::InvalidInput)?;
        let c_lat = CString::new(lat_col).map_err(|_| Error::InvalidInput)?;
        let c_lon = CString::new(lon_col).map_err(|_| Error::InvalidInput)?;
        Self::check_op(unsafe {
            ffi::td_astar(self.raw, src.raw, dst.raw, rel.ptr,
                         c_w.as_ptr(), c_lat.as_ptr(), c_lon.as_ptr(), max_depth)
        })
    }

    pub fn k_shortest(
        &self, src: Column, dst: Column, rel: &'a Rel,
        weight_col: &str, k: u16,
    ) -> Result<Column> {
        let c_col = CString::new(weight_col).map_err(|_| Error::InvalidInput)?;
        Self::check_op(unsafe {
            ffi::td_k_shortest(self.raw, src.raw, dst.raw, rel.ptr, c_col.as_ptr(), k)
        })
    }
```

**Step 3: Add Rust integration tests**

Follow the pattern from `graph_dijkstra()` in `tests/engine_api.rs`. Create tests for each algorithm.

**Step 4: Add SQL table function dispatch**

In `teide-rs/src/sql/planner.rs:2421`, extend the match:

```rust
"pagerank" | "connected_component" | "louvain" | "clustering_coefficient"
    | "degree_centrality" | "random_walk" => { ... }
```

In `teide-rs/src/sql/pgq.rs`, add dispatch for the new algorithms in `execute_standalone_algorithm` and `ALGO_FUNCTIONS`.

**Step 5: Update vendored C source**

Copy the updated teide C files to `teide-rs/vendor/teide/`.

**Step 6: Build and test both repos**

Run: `cmake --build build && cd ../teide-rs && cargo test --all-features`
Expected: All pass

**Step 7: Commit both repos**

```bash
# In teide-rs/
git add src/ffi.rs src/engine.rs src/sql/planner.rs src/sql/pgq.rs tests/ vendor/
git commit -m "feat: Rust bindings + SQL for clustering coeff, random walk, A*, k-shortest"
```

---

## Summary

| Task | Algorithm | C files | Rust files | Complexity |
|------|-----------|---------|------------|------------|
| 1 | Opcodes + declarations | td.h | — | Trivial |
| 2 | Clustering coefficients | graph.c, exec.c | — | O(n·d²) |
| 3 | Random walk | graph.c, exec.c | — | O(walk_len) |
| 4 | A* shortest path | graph.c, exec.c | — | O(m·log(n)) |
| 5 | Yen's k-shortest | graph.c, exec.c | — | O(k·n·m·log(n)) |
| 6 | Rust bindings + SQL | — | ffi.rs, engine.rs, planner.rs, pgq.rs | Integration |
