# Graph Algorithms Batch 3 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add betweenness centrality (Brandes with optional sampling), closeness centrality (with optional sampling), and MST (Kruskal with union-find) — 3 heavy O(n·m) graph algorithms. Both C kernels and Rust/SQL integration.

**Architecture:** Betweenness and closeness use BFS from each source node (or sampled subset). MST collects all weighted edges, sorts by weight, and builds forest via union-find with path compression + union by rank. All follow the established pattern: opcode → DAG constructor → executor kernel → result table → Rust FFI + SQL.

**Tech Stack:** C17 (teide), Rust (teide-rs). Build C: `cmake --build build`. Build Rust: `cd ../teide-rs && cargo build --all-features`. Test C: `./build/test_teide --suite /csr`. Test Rust: `cd ../teide-rs && cargo test --all-features`.

**Design doc:** `docs/plans/2026-03-23-graph-algorithms-design.md`

**Reference graph (test_csr.c):**
```
make_edge_table: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0  (4 nodes, 6 edges, cyclic)
make_dag_edge_table: 0→1, 0→2, 1→3, 2→3  (4 nodes, 4 edges, DAG)
make_astar_graph: 5 nodes, 6 weighted edges with coords
```

---

### Task 1: Opcodes and declarations

- [x] Add OP_BETWEENNESS, OP_CLOSENESS, OP_MST opcodes to td.h
- [x] Add td_betweenness, td_closeness, td_mst declarations to td.h
- [x] Build and verify

**Files:**
- Modify: `include/teide/td.h` (add opcodes after OP_RANDOM_WALK 98)

**Step 1: Add opcodes and declarations**

```c
#define OP_BETWEENNESS     99   /* betweenness centrality (Brandes)   */
#define OP_CLOSENESS      100   /* closeness centrality               */
#define OP_MST            101   /* minimum spanning forest (Kruskal)  */
```

Declarations:

```c
td_op_t* td_betweenness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size);
td_op_t* td_closeness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size);
td_op_t* td_mst(td_graph_t* g, td_rel_t* rel, const char* weight_col);
```

**Step 2: Build, commit**

```bash
git add include/teide/td.h
git commit -m "feat: add opcodes and declarations for betweenness, closeness, MST"
```

---

### Task 2: Betweenness centrality (Brandes) — C kernel + test

- [x] Write failing tests (test_betweenness, test_betweenness_sampled)
- [x] Implement DAG constructor td_betweenness in graph.c
- [x] Implement exec_betweenness kernel in exec.c
- [x] Tests pass

Brandes' algorithm: BFS from each source, accumulate dependency scores backward. O(n·m) exact, O(sample·m) approximate when `sample_size > 0`.

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_betweenness(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* DAG: 0→1, 0→2, 1→3, 2→3
     * Node 0 is the only source to 3, via 1 and 2.
     * Nodes 1 and 2 are bridges — should have nonzero betweenness. */
    td_t* edges = make_dag_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* bc = td_betweenness(g, rel, 0);  /* exact */
    munit_assert_ptr_not_null(bc);

    td_t* result = td_execute(g, bc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    int64_t cent_sym = td_sym_intern("_centrality", 11);
    td_t* cent_col = td_table_get_col(result, cent_sym);
    munit_assert_ptr_not_null(cent_col);
    double* cents = (double*)td_data(cent_col);

    /* All centrality values should be >= 0 */
    for (int i = 0; i < 4; i++) {
        munit_assert_double(cents[i], >=, 0.0);
    }

    /* Node 0 is source-only, node 3 is sink-only — lower betweenness.
     * Nodes 1 and 2 are intermediaries — should have higher betweenness. */
    munit_assert_double(cents[1] + cents[2], >, cents[0] + cents[3]);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_betweenness_sampled(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* bc = td_betweenness(g, rel, 2);  /* sample 2 sources */
    td_t* result = td_execute(g, bc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

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

Register: `{ "/betweenness", test_betweenness, ... }`, `{ "/betweenness_sampled", test_betweenness_sampled, ... }`

**Step 2: Implement DAG constructor**

```c
td_op_t* td_betweenness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size) {
    if (!g || !rel) return NULL;
    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;
    ext->base.opcode   = OP_BETWEENNESS;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel       = rel;
    ext->graph.direction = 2;  /* undirected BFS */
    ext->graph.max_iter  = sample_size;  /* 0 = exact */
    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 3: Implement executor kernel (Brandes)**

```c
static td_t* exec_betweenness(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);
    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);
    uint16_t sample = ext->graph.max_iter;
    int64_t n_sources = (sample > 0 && sample < n) ? sample : n;

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);
    int64_t* rev_off = (int64_t*)td_data(rel->rev.offsets);
    int64_t* rev_tgt = (int64_t*)td_data(rel->rev.targets);

    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    double*  cb      = (double*)td_scratch_arena_push(&arena, (size_t)n * sizeof(double));
    double*  sigma   = (double*)td_scratch_arena_push(&arena, (size_t)n * sizeof(double));
    double*  delta   = (double*)td_scratch_arena_push(&arena, (size_t)n * sizeof(double));
    int64_t* dist    = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* queue   = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* stack   = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    /* Predecessor lists: flat array with offsets */
    int64_t* pred_off  = (int64_t*)td_scratch_arena_push(&arena, (size_t)(n + 1) * sizeof(int64_t));
    int64_t m_total = rel->fwd.n_edges + rel->rev.n_edges;
    int64_t* pred_data = (int64_t*)td_scratch_arena_push(&arena, (size_t)m_total * sizeof(int64_t));
    int64_t* pred_count = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));

    if (!cb || !sigma || !delta || !dist || !queue || !stack ||
        !pred_off || !pred_data || !pred_count) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    memset(cb, 0, (size_t)n * sizeof(double));

    /* Source selection: first n_sources nodes (or random if sampled) */
    /* For simplicity, use deterministic selection: nodes 0..n_sources-1
     * when exact (sample=0 → all), or strided selection for sampling */
    int64_t stride = (sample > 0 && sample < n) ? (n / sample) : 1;

    for (int64_t si = 0; si < n_sources; si++) {
        int64_t s = (si * stride) % n;

        /* Initialize */
        for (int64_t i = 0; i < n; i++) {
            sigma[i] = 0.0;
            delta[i] = 0.0;
            dist[i] = -1;
            pred_count[i] = 0;
        }
        sigma[s] = 1.0;
        dist[s] = 0;

        /* BFS from s — build stack in order of discovery */
        int64_t q_head = 0, q_tail = 0;
        int64_t stack_top = 0;
        queue[q_tail++] = s;

        while (q_head < q_tail) {
            int64_t v = queue[q_head++];
            stack[stack_top++] = v;

            /* Forward neighbors */
            for (int64_t j = fwd_off[v]; j < fwd_off[v+1]; j++) {
                int64_t w = fwd_tgt[j];
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    queue[q_tail++] = w;
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred_data[w * (m_total / n + 2) + pred_count[w]++] = v;
                    /* Note: pred storage is approximate; see implementation note */
                }
            }
            /* Reverse neighbors (undirected) */
            for (int64_t j = rev_off[v]; j < rev_off[v+1]; j++) {
                int64_t w = rev_tgt[j];
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    queue[q_tail++] = w;
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred_data[w * (m_total / n + 2) + pred_count[w]++] = v;
                }
            }
        }

        /* Back-propagation of dependencies */
        while (stack_top > 0) {
            int64_t w = stack[--stack_top];
            int64_t pred_max = m_total / n + 2;
            for (int64_t pi = 0; pi < pred_count[w]; pi++) {
                int64_t v = pred_data[w * pred_max + pi];
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s) cb[w] += delta[w];
        }
    }

    /* Normalize if sampled */
    if (sample > 0 && sample < n) {
        double scale = (double)n / (double)sample;
        for (int64_t i = 0; i < n; i++) cb[i] *= scale;
    }

    /* Build result */
    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* cent_vec = td_vec_new(TD_F64, n);
    if (!node_vec || TD_IS_ERR(node_vec) || !cent_vec || TD_IS_ERR(cent_vec)) {
        td_scratch_arena_reset(&arena);
        if (node_vec && !TD_IS_ERR(node_vec)) td_release(node_vec);
        if (cent_vec && !TD_IS_ERR(cent_vec)) td_release(cent_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    int64_t* ndata = (int64_t*)td_data(node_vec);
    double*  cdata = (double*)td_data(cent_vec);
    for (int64_t i = 0; i < n; i++) { ndata[i] = i; cdata[i] = cb[i]; }
    node_vec->len = n; cent_vec->len = n;
    td_scratch_arena_reset(&arena);

    td_t* result = td_table_new(2);
    if (!result || TD_IS_ERR(result)) {
        td_release(node_vec); td_release(cent_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_node", 5), node_vec);
    td_release(node_vec);
    result = td_table_add_col(result, td_sym_intern("_centrality", 11), cent_vec);
    td_release(cent_vec);
    return result;
}
```

**Implementation note on predecessor storage:** The plan uses a flat array indexed by `w * pred_max + pi` where `pred_max = m_total / n + 2`. This is an approximation — the actual number of predecessors per node can exceed this. The implementer should use a more robust approach: either a per-BFS arena with linked lists, or a flat array sized to `m_total` with a separate offset array. The existing scratch arena makes the flat approach cleanest.

Add switch case:
```c
        case OP_BETWEENNESS: {
            return exec_betweenness(g, op);
        }
```

**Step 4: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: betweenness centrality — Brandes algorithm with optional sampling"
```

---

### Task 3: Closeness centrality — C kernel + test

- [x] Write failing test (test_closeness)
- [x] Implement DAG constructor td_closeness in graph.c
- [x] Implement exec_closeness kernel in exec.c
- [x] Tests pass

BFS from each source, sum distances. `closeness[v] = (n-1) / sum_dist[v]`.

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_closeness(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges = make_edge_table();  /* 4-node cycle */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* cc = td_closeness(g, rel, 0);  /* exact */
    td_t* result = td_execute(g, cc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    int64_t cent_sym = td_sym_intern("_centrality", 11);
    td_t* cent_col = td_table_get_col(result, cent_sym);
    double* cents = (double*)td_data(cent_col);

    /* All nodes should have positive closeness in a connected graph */
    for (int i = 0; i < 4; i++) {
        munit_assert_double(cents[i], >, 0.0);
        munit_assert_double(cents[i], <=, 1.0);
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

**Step 2: Implement**

Closeness is simpler than betweenness — no back-propagation needed. For each source s, BFS to compute distances, sum them. `closeness[s] = (reachable - 1) / sum_dist`. Same sampling logic as betweenness (`sample_size > 0` → stride selection).

```c
td_op_t* td_closeness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size) {
    /* Same pattern as td_betweenness — OP_CLOSENESS, sample in max_iter */
}

static td_t* exec_closeness(td_graph_t* g, td_op_t* op) {
    /* For each source s:
     *   BFS (fwd + rev for undirected) to compute dist[] from s
     *   sum_dist = sum of all finite distances
     *   reachable = count of finite distances
     *   closeness[s] += (reachable - 1) / sum_dist
     * If sampled, normalize by n/sample */
}
```

Scratch: `double* closeness[n]`, `int64_t* dist[n]`, `int64_t* queue[n]`.

**Step 3: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: closeness centrality — BFS distance sums with optional sampling"
```

---

### Task 4: MST (Kruskal) — C kernel + test

- [x] Write failing test (test_mst)
- [x] Implement DAG constructor td_mst in graph.c
- [x] Implement exec_mst kernel in exec.c
- [x] Tests pass

Collect weighted edges, sort by weight, union-find to build forest.

**Files:**
- Modify: `src/ops/graph.c`, `src/ops/exec.c`
- Test: `test/test_csr.c`

**Step 1: Write failing test**

```c
static MunitResult test_mst(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* edges; td_rel_t* rel; td_t* node_props;
    make_astar_graph(&edges, &rel, &node_props);
    /* 5 nodes, 6 edges: 0→1(1), 0→2(4), 1→3(2), 2→3(1), 3→4(3), 1→4(10)
     * MST (undirected) should pick: 0-1(1), 2-3(1), 1-3(2), 3-4(3) = total 7
     * (skip 0-2(4) and 1-4(10)) */

    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6),
                           td_vec_from_raw(TD_I64, (int64_t[]){0}, 1));
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* mst = td_mst(g, rel, "weight");
    munit_assert_ptr_not_null(mst);

    td_t* result = td_execute(g, mst);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* MST of 5 nodes has 4 edges */
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* Total weight should be 7.0 */
    int64_t w_sym = td_sym_intern("_weight", 7);
    td_t* w_col = td_table_get_col(result, w_sym);
    munit_assert_ptr_not_null(w_col);
    double* ws = (double*)td_data(w_col);
    double total = 0.0;
    for (int i = 0; i < 4; i++) total += ws[i];
    munit_assert_double(total, >=, 6.99);
    munit_assert_double(total, <=, 7.01);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(node_props);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Implement DAG constructor**

```c
td_op_t* td_mst(td_graph_t* g, td_rel_t* rel, const char* weight_col) {
    if (!g || !rel) return NULL;
    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;
    ext->base.opcode   = OP_MST;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)(rel->fwd.n_nodes - 1);
    ext->graph.rel     = rel;
    ext->graph.direction = 2;
    ext->graph.weight_col_sym = td_sym_intern(weight_col, strlen(weight_col));
    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

**Step 3: Implement executor kernel (Kruskal)**

```c
typedef struct { double w; int64_t src; int64_t dst; } mst_edge_t;

static int mst_edge_cmp(const void* a, const void* b) {
    double da = ((const mst_edge_t*)a)->w;
    double db = ((const mst_edge_t*)b)->w;
    return (da > db) - (da < db);
}

/* Union-Find with path compression + union by rank */
static int64_t uf_find(int64_t* parent, int64_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}
static bool uf_union(int64_t* parent, int64_t* rank, int64_t a, int64_t b) {
    a = uf_find(parent, a); b = uf_find(parent, b);
    if (a == b) return false;
    if (rank[a] < rank[b]) { int64_t tmp = a; a = b; b = tmp; }
    parent[b] = a;
    if (rank[a] == rank[b]) rank[a]++;
    return true;
}

static td_t* exec_mst(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);
    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel || !rel->fwd.props) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    int64_t m = rel->fwd.n_edges;

    int64_t weight_sym = ext->graph.weight_col_sym;
    td_t* weight_vec = td_table_get_col(rel->fwd.props, weight_sym);
    if (!weight_vec || weight_vec->type != TD_F64) return TD_ERR_PTR(TD_ERR_SCHEMA);
    double* weights = (double*)td_data(weight_vec);

    int64_t* fwd_off = (int64_t*)td_data(rel->fwd.offsets);
    int64_t* fwd_tgt = (int64_t*)td_data(rel->fwd.targets);
    int64_t* fwd_row = (int64_t*)td_data(rel->fwd.rowmap);

    td_scratch_arena_t arena;
    td_scratch_arena_init(&arena);

    /* Collect all edges (undirected: use forward CSR, treat as undirected) */
    mst_edge_t* edges_arr = (mst_edge_t*)td_scratch_arena_push(&arena,
                                (size_t)m * sizeof(mst_edge_t));
    int64_t* uf_parent = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    int64_t* uf_rank   = (int64_t*)td_scratch_arena_push(&arena, (size_t)n * sizeof(int64_t));
    if (!edges_arr || !uf_parent || !uf_rank) {
        td_scratch_arena_reset(&arena);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* Fill edge array from forward CSR */
    int64_t ei = 0;
    for (int64_t u = 0; u < n; u++) {
        for (int64_t j = fwd_off[u]; j < fwd_off[u+1]; j++) {
            edges_arr[ei].src = u;
            edges_arr[ei].dst = fwd_tgt[j];
            edges_arr[ei].w   = weights[fwd_row[j]];
            ei++;
        }
    }

    /* Sort edges by weight */
    qsort(edges_arr, (size_t)ei, sizeof(mst_edge_t), mst_edge_cmp);

    /* Initialize union-find */
    for (int64_t i = 0; i < n; i++) { uf_parent[i] = i; uf_rank[i] = 0; }

    /* Build MST */
    int64_t mst_count = 0;
    /* Result arrays (at most n-1 edges) */
    td_t* src_vec = td_vec_new(TD_I64, n - 1);
    td_t* dst_vec = td_vec_new(TD_I64, n - 1);
    td_t* wt_vec  = td_vec_new(TD_F64, n - 1);
    if (!src_vec || TD_IS_ERR(src_vec) ||
        !dst_vec || TD_IS_ERR(dst_vec) ||
        !wt_vec  || TD_IS_ERR(wt_vec)) {
        td_scratch_arena_reset(&arena);
        if (src_vec && !TD_IS_ERR(src_vec)) td_release(src_vec);
        if (dst_vec && !TD_IS_ERR(dst_vec)) td_release(dst_vec);
        if (wt_vec  && !TD_IS_ERR(wt_vec))  td_release(wt_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* sdata = (int64_t*)td_data(src_vec);
    int64_t* ddata = (int64_t*)td_data(dst_vec);
    double*  wdata = (double*)td_data(wt_vec);

    for (int64_t i = 0; i < ei && mst_count < n - 1; i++) {
        if (uf_union(uf_parent, uf_rank, edges_arr[i].src, edges_arr[i].dst)) {
            sdata[mst_count] = edges_arr[i].src;
            ddata[mst_count] = edges_arr[i].dst;
            wdata[mst_count] = edges_arr[i].w;
            mst_count++;
        }
    }

    src_vec->len = mst_count;
    dst_vec->len = mst_count;
    wt_vec->len  = mst_count;

    td_scratch_arena_reset(&arena);

    td_t* result = td_table_new(3);
    if (!result || TD_IS_ERR(result)) {
        td_release(src_vec); td_release(dst_vec); td_release(wt_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    result = td_table_add_col(result, td_sym_intern("_src", 4), src_vec);
    td_release(src_vec);
    result = td_table_add_col(result, td_sym_intern("_dst", 4), dst_vec);
    td_release(dst_vec);
    result = td_table_add_col(result, td_sym_intern("_weight", 7), wt_vec);
    td_release(wt_vec);
    return result;
}
```

Add switch cases for all three:
```c
        case OP_BETWEENNESS: { return exec_betweenness(g, op); }
        case OP_CLOSENESS:   { return exec_closeness(g, op); }
        case OP_MST:         { return exec_mst(g, op); }
```

**Step 4: Run tests, commit**

```bash
git add src/ops/graph.c src/ops/exec.c test/test_csr.c
git commit -m "feat: MST (Kruskal) — union-find with path compression, sorted edges"
```

---

### Task 5: Rust bindings + SQL for all 3 algorithms (teide-rs)

- [x] Add FFI declarations in ffi.rs
- [x] Add Graph methods in engine.rs
- [x] Add SQL table function + GRAPH_TABLE dispatch
- [x] Vendor update, Rust tests pass

**Files:**
- Modify: `../teide-rs/src/ffi.rs` (add FFI declarations)
- Modify: `../teide-rs/src/engine.rs` (add Graph methods)
- Modify: `../teide-rs/src/sql/planner.rs` (add table function dispatch)
- Modify: `../teide-rs/src/sql/pgq.rs` (add algorithm dispatch)
- Test: `../teide-rs/tests/engine_api.rs`
- Update: `../teide-rs/vendor/teide/` (copy updated C sources)

**Step 1: FFI declarations**

```rust
    pub fn td_betweenness(g: *mut td_graph_t, rel: *mut td_rel_t, sample_size: u16) -> *mut td_op_t;
    pub fn td_closeness(g: *mut td_graph_t, rel: *mut td_rel_t, sample_size: u16) -> *mut td_op_t;
    pub fn td_mst(g: *mut td_graph_t, rel: *mut td_rel_t, weight_col: *const c_char) -> *mut td_op_t;
```

**Step 2: Graph methods**

```rust
    pub fn betweenness(&self, rel: &'a Rel, sample_size: u16) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_betweenness(self.raw, rel.ptr, sample_size) })
    }
    pub fn closeness(&self, rel: &'a Rel, sample_size: u16) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_closeness(self.raw, rel.ptr, sample_size) })
    }
    pub fn mst(&self, rel: &'a Rel, weight_col: &str) -> Result<Column> {
        let c = CString::new(weight_col).map_err(|_| Error::InvalidInput)?;
        Self::check_op(unsafe { ffi::td_mst(self.raw, rel.ptr, c.as_ptr()) })
    }
```

**Step 3: SQL table function + GRAPH_TABLE dispatch**

Add `"betweenness" | "closeness" | "mst"` to the table function match in planner.rs and the algorithm dispatch in pgq.rs.

**Step 4: Rust tests, vendor update, build both**

```bash
cd ../teide-rs && cargo test --all-features
```

**Step 5: Commit both repos**

```bash
# teide-rs
git add src/ffi.rs src/engine.rs src/sql/ tests/ vendor/
git commit -m "feat: Rust bindings + SQL for betweenness, closeness, MST"
```

---

## Summary

| Task | Algorithm | Complexity | Result Columns |
|------|-----------|------------|----------------|
| 1 | Opcodes + declarations | — | — |
| 2 | Betweenness centrality | O(sample·m) | `_node`, `_centrality` |
| 3 | Closeness centrality | O(sample·m) | `_node`, `_centrality` |
| 4 | MST (Kruskal) | O(m·log(m)) | `_src`, `_dst`, `_weight` |
| 5 | Rust bindings + SQL | Integration | — |

After this batch, Teide has all 10 planned graph algorithms — full parity with CozoDB plus LFTJ and SIP which CozoDB doesn't have.
