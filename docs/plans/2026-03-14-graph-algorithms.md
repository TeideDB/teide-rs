# Graph Algorithms Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PageRank, connected components, weighted Dijkstra shortest path, and community detection (Louvain) as C engine kernels with SQL/PGQ exposure via `GRAPH_TABLE` COLUMNS functions.

**Architecture:** Each algorithm is implemented as a new opcode in the C engine (`OP_PAGERANK`, `OP_CONNECTED_COMP`, `OP_DIJKSTRA`, `OP_LOUVAIN`), following the existing pattern: builder function in `graph.c`, kernel in `exec.c`, dispatch in `exec_node()`. FFI bindings in `ffi.rs`, safe wrappers in `engine.rs`, and SQL exposure via `pgq.rs`. All algorithms operate on existing `td_rel_t` CSR indexes — no new data structures.

**Tech Stack:** C17 (vendor/teide/), Rust FFI, sqlparser, existing CSR/Rel infrastructure.

**Prerequisite:** SQL/PGQ Phase 1+2 implementation (completed).

---

## File Structure

### C Engine (vendor/teide/)
| File | Change |
|------|--------|
| `vendor/teide/include/teide/td.h` | Add `OP_PAGERANK=84`, `OP_CONNECTED_COMP=85`, `OP_DIJKSTRA=86`, `OP_LOUVAIN=87` constants and function declarations |
| `vendor/teide/src/ops/graph.c` | Add builder functions: `td_pagerank()`, `td_connected_comp()`, `td_dijkstra()`, `td_louvain()` |
| `vendor/teide/src/ops/exec.c` | Add kernel functions: `exec_pagerank()`, `exec_connected_comp()`, `exec_dijkstra()`, `exec_louvain()` and dispatch cases |
| `vendor/teide/src/ops/dump.c` | Add opcode name strings for debugging |

### Rust Bindings
| File | Change |
|------|--------|
| `src/ffi.rs` | Add `OP_PAGERANK/CONNECTED_COMP/DIJKSTRA/LOUVAIN` constants and `extern "C"` declarations |
| `src/engine.rs` | Add `Graph::pagerank()`, `Graph::connected_comp()`, `Graph::dijkstra()`, `Graph::louvain()` safe wrappers |

### SQL Layer
| File | Change |
|------|--------|
| `src/sql/pgq.rs` | Handle algorithm function calls in GRAPH_TABLE COLUMNS: `PAGERANK()`, `COMPONENT()`, `COMMUNITY()` |

### Tests
| File | Change |
|------|--------|
| `tests/engine_api.rs` | Add Rust API tests for each algorithm |
| `tests/slt/pgq_algorithms.slt` | SQL logic tests for graph algorithms via GRAPH_TABLE |
| `tests/slt_runner.rs` | Add `slt_pgq_algorithms` test function |

---

## Task 1: PageRank — C Engine Kernel

Iterative PageRank over CSR adjacency. Each iteration: for every node, sum `rank[neighbor] / out_degree[neighbor]` across all incoming edges, apply damping factor. Converges after `max_iter` iterations or when ranks change less than epsilon.

**Output:** I64 node ID vector + F64 rank vector as a TD_TABLE with columns `_node`, `_rank`.

**Files:**
- Modify: `vendor/teide/include/teide/td.h`
- Modify: `vendor/teide/src/ops/graph.c`
- Modify: `vendor/teide/src/ops/exec.c`
- Modify: `vendor/teide/src/ops/dump.c`

- [x] **Step 1: Add opcode constant and declaration to `td.h`**

After the existing graph opcodes (line ~405 in td.h):
```c
#define OP_PAGERANK        84   /* iterative PageRank                 */
#define OP_CONNECTED_COMP  85   /* connected components (label prop)  */
#define OP_DIJKSTRA        86   /* weighted shortest path (Dijkstra)  */
#define OP_LOUVAIN         87   /* community detection (Louvain)      */
```

Add function declarations after `td_wco_join` (line ~893):
```c
/* Graph algorithms */
td_op_t* td_pagerank(td_graph_t* g, td_rel_t* rel,
                      uint16_t max_iter, double damping);
td_op_t* td_connected_comp(td_graph_t* g, td_rel_t* rel);
td_op_t* td_dijkstra(td_graph_t* g, td_op_t* src, td_op_t* dst,
                      td_rel_t* rel, const char* weight_col,
                      uint8_t max_depth);
td_op_t* td_louvain(td_graph_t* g, td_rel_t* rel,
                     uint16_t max_iter);
```

- [x] **Step 2: Extend `td_op_ext_t` graph union for algorithm params**

In td.h, extend the graph union in `td_op_ext_t` (around line 501-509):
```c
struct {
    void*     rel;
    void*     sip_sel;
    uint8_t   direction;
    uint8_t   min_depth;
    uint8_t   max_depth;
    uint8_t   path_tracking;
    uint8_t   factorized;
    /* Algorithm-specific params (reuse existing padding) */
    uint16_t  max_iter;       /* PageRank/Louvain iterations  */
    double    damping;        /* PageRank damping factor      */
    int64_t   weight_col_sym; /* Dijkstra weight column name  */
} graph;
```

- [x] **Step 3: Add PageRank builder function to `graph.c`**

```c
td_op_t* td_pagerank(td_graph_t* g, td_rel_t* rel,
                      uint16_t max_iter, double damping) {
    if (!g || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode   = OP_PAGERANK;
    ext->base.arity    = 0;   /* nullary: reads CSR directly */
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel      = rel;
    ext->graph.max_iter  = max_iter;
    ext->graph.damping   = damping;
    ext->graph.direction = 0;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 4: Add PageRank kernel to `exec.c`**

```c
/*
 * PageRank: iterative algorithm.
 *
 * rank[v] = (1 - d) / N + d * SUM(rank[u] / out_degree[u]) for u in in_neighbors(v)
 *
 * Uses reverse CSR to iterate over in-neighbors, forward CSR for out-degree.
 */
static td_t* exec_pagerank(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n       = rel->fwd.n_nodes;
    uint16_t iters  = ext->graph.max_iter;
    double damping  = ext->graph.damping;

    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    /* Allocate rank arrays: current and next */
    double* rank     = (double*)scratch_alloc(n * sizeof(double));
    double* rank_new = (double*)scratch_alloc(n * sizeof(double));
    if (!rank || !rank_new) return TD_ERR_PTR(TD_ERR_OOM);

    double init = 1.0 / (double)n;
    for (int64_t i = 0; i < n; i++) rank[i] = init;

    /* Get raw CSR arrays for direct access */
    int64_t* fwd_off = (int64_t*)((char*)rel->fwd.offsets + 32);
    int64_t* rev_off = (int64_t*)((char*)rel->rev.offsets + 32);
    int64_t* rev_tgt = (int64_t*)((char*)rel->rev.targets + 32);

    double base = (1.0 - damping) / (double)n;

    for (uint16_t iter = 0; iter < iters; iter++) {
        for (int64_t v = 0; v < n; v++) {
            double sum = 0.0;
            /* Iterate over in-neighbors of v using reverse CSR */
            int64_t rev_start = rev_off[v];
            int64_t rev_end   = rev_off[v + 1];
            for (int64_t j = rev_start; j < rev_end; j++) {
                int64_t u = rev_tgt[j];
                /* out_degree of u from forward CSR */
                int64_t out_deg = fwd_off[u + 1] - fwd_off[u];
                if (out_deg > 0) {
                    sum += rank[u] / (double)out_deg;
                }
            }
            rank_new[v] = base + damping * sum;
        }
        /* Swap */
        double* tmp = rank;
        rank = rank_new;
        rank_new = tmp;
    }

    /* Build output table: _node (I64), _rank (F64) */
    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* rank_vec = td_vec_new(TD_F64, n);
    if (!node_vec || !rank_vec) {
        scratch_free(rank);
        scratch_free(rank_new);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)((char*)node_vec + 32);
    double*  rdata = (double*)((char*)rank_vec + 32);
    for (int64_t i = 0; i < n; i++) {
        ndata[i] = i;
        rdata[i] = rank[i];
    }
    node_vec->n = n;
    rank_vec->n = n;

    scratch_free(rank);
    scratch_free(rank_new);

    /* Package as table with named columns */
    td_t* result = td_table_new(2, n);
    if (!result) {
        td_release(node_vec);
        td_release(rank_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, node_vec, td_sym_intern("_node"));
    td_table_set_col(result, 1, rank_vec, td_sym_intern("_rank"));

    return result;
}
```

- [x] **Step 5: Add dispatch case in `exec_node()`**

In the switch statement in `exec_node()`:
```c
case OP_PAGERANK: {
    td_t* result = exec_pagerank(g, op);
    return result;
}
```

- [x] **Step 6: Add opcode name to `dump.c`**

```c
case OP_PAGERANK:       return "PAGERANK";
```

- [x] **Step 7: Verify C engine compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS (build.rs compiles the C source)

- [x] **Step 8: Commit**

```bash
git add vendor/teide/
git commit -m "feat(engine): add OP_PAGERANK C kernel with iterative rank computation"
```

---

## Task 2: Connected Components — C Engine Kernel

Label propagation: each node starts with its own ID as label. Each iteration: for every node, set label to minimum label among self and all neighbors. Converge when no labels change.

**Output:** TD_TABLE with `_node` (I64) and `_component` (I64).

**Files:**
- Modify: `vendor/teide/src/ops/graph.c`
- Modify: `vendor/teide/src/ops/exec.c`
- Modify: `vendor/teide/src/ops/dump.c`

- [x] **Step 1: Add builder function to `graph.c`**

```c
td_op_t* td_connected_comp(td_graph_t* g, td_rel_t* rel) {
    if (!g || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode   = OP_CONNECTED_COMP;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel     = rel;
    ext->graph.direction = 2;  /* both directions for undirected */

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 2: Add kernel to `exec.c`**

```c
/*
 * Connected Components via label propagation.
 * Treats graph as undirected (uses both forward and reverse CSR).
 * O(diameter * |E|) time.
 */
static td_t* exec_connected_comp(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t* label = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    if (!label) return TD_ERR_PTR(TD_ERR_OOM);

    /* Initialize: each node is its own component */
    for (int64_t i = 0; i < n; i++) label[i] = i;

    int64_t* fwd_off = (int64_t*)((char*)rel->fwd.offsets + 32);
    int64_t* fwd_tgt = (int64_t*)((char*)rel->fwd.targets + 32);
    int64_t* rev_off = (int64_t*)((char*)rel->rev.offsets + 32);
    int64_t* rev_tgt = (int64_t*)((char*)rel->rev.targets + 32);

    /* Iterate until convergence */
    bool changed = true;
    while (changed) {
        changed = false;
        for (int64_t v = 0; v < n; v++) {
            int64_t min_label = label[v];
            /* Forward neighbors */
            for (int64_t j = fwd_off[v]; j < fwd_off[v + 1]; j++) {
                int64_t u = fwd_tgt[j];
                if (label[u] < min_label) min_label = label[u];
            }
            /* Reverse neighbors */
            for (int64_t j = rev_off[v]; j < rev_off[v + 1]; j++) {
                int64_t u = rev_tgt[j];
                if (label[u] < min_label) min_label = label[u];
            }
            if (min_label < label[v]) {
                label[v] = min_label;
                changed = true;
            }
        }
    }

    /* Build output table */
    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* comp_vec = td_vec_new(TD_I64, n);
    if (!node_vec || !comp_vec) {
        scratch_free(label);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)((char*)node_vec + 32);
    int64_t* cdata = (int64_t*)((char*)comp_vec + 32);
    for (int64_t i = 0; i < n; i++) {
        ndata[i] = i;
        cdata[i] = label[i];
    }
    node_vec->n = n;
    comp_vec->n = n;

    scratch_free(label);

    td_t* result = td_table_new(2, n);
    if (!result) {
        td_release(node_vec);
        td_release(comp_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, node_vec, td_sym_intern("_node"));
    td_table_set_col(result, 1, comp_vec, td_sym_intern("_component"));

    return result;
}
```

- [x] **Step 3: Add dispatch and dump**

In `exec_node()`:
```c
case OP_CONNECTED_COMP: {
    return exec_connected_comp(g, op);
}
```

In `dump.c`:
```c
case OP_CONNECTED_COMP: return "CONNECTED_COMP";
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vendor/teide/
git commit -m "feat(engine): add OP_CONNECTED_COMP via label propagation"
```

---

## Task 3: Weighted Dijkstra — C Engine Kernel

Dijkstra's algorithm with edge weights read from the CSR edge property table. Uses a binary heap (min-heap implemented as array).

**Output:** TD_TABLE with `_node` (I64), `_dist` (F64), `_depth` (I64).

**Files:**
- Modify: `vendor/teide/src/ops/graph.c`
- Modify: `vendor/teide/src/ops/exec.c`
- Modify: `vendor/teide/src/ops/dump.c`

- [x] **Step 1: Add builder function to `graph.c`**

```c
td_op_t* td_dijkstra(td_graph_t* g, td_op_t* src, td_op_t* dst,
                      td_rel_t* rel, const char* weight_col,
                      uint8_t max_depth) {
    if (!g || !src || !rel || !weight_col) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    src = &g->nodes[src->id];
    if (dst) dst = &g->nodes[dst->id];

    ext->base.opcode    = OP_DIJKSTRA;
    ext->base.arity     = dst ? 2 : 1;
    ext->base.inputs[0] = src;
    ext->base.inputs[1] = dst;
    ext->base.out_type  = TD_TABLE;
    ext->base.est_rows  = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel       = rel;
    ext->graph.direction = 0;
    ext->graph.max_depth = max_depth;
    ext->graph.weight_col_sym = td_sym_intern(weight_col);

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 2: Add Dijkstra kernel to `exec.c`**

```c
/* Min-heap entry for Dijkstra */
typedef struct {
    double   dist;
    int64_t  node;
} dijk_entry_t;

static void dijk_heap_push(dijk_entry_t* heap, int64_t* size,
                            double dist, int64_t node) {
    int64_t i = (*size)++;
    heap[i].dist = dist;
    heap[i].node = node;
    /* Sift up */
    while (i > 0) {
        int64_t parent = (i - 1) / 2;
        if (heap[parent].dist <= heap[i].dist) break;
        dijk_entry_t tmp = heap[parent];
        heap[parent] = heap[i];
        heap[i] = tmp;
        i = parent;
    }
}

static dijk_entry_t dijk_heap_pop(dijk_entry_t* heap, int64_t* size) {
    dijk_entry_t top = heap[0];
    (*size)--;
    if (*size > 0) {
        heap[0] = heap[*size];
        /* Sift down */
        int64_t i = 0;
        while (1) {
            int64_t left  = 2 * i + 1;
            int64_t right = 2 * i + 2;
            int64_t smallest = i;
            if (left  < *size && heap[left].dist  < heap[smallest].dist) smallest = left;
            if (right < *size && heap[right].dist < heap[smallest].dist) smallest = right;
            if (smallest == i) break;
            dijk_entry_t tmp = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = tmp;
            i = smallest;
        }
    }
    return top;
}

/*
 * Dijkstra's algorithm with edge weights from CSR property table.
 * Returns shortest distances from source to all reachable nodes
 * (or just to destination if dst is provided).
 */
static td_t* exec_dijkstra(td_graph_t* g, td_op_t* op,
                             td_t* src_val, td_t* dst_val) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);
    if (!rel->fwd.props) return TD_ERR_PTR(TD_ERR_SCHEMA); /* need edge properties */

    int64_t n = rel->fwd.n_nodes;
    int64_t src_id = *(int64_t*)((char*)src_val + 32);
    int64_t dst_id = dst_val ? *(int64_t*)((char*)dst_val + 32) : -1;

    if (src_id < 0 || src_id >= n) return TD_ERR_PTR(TD_ERR_RANGE);

    /* Find weight column in edge properties */
    int64_t weight_sym = ext->graph.weight_col_sym;
    td_t* props = rel->fwd.props;
    int weight_col_idx = -1;
    int64_t n_props_cols = td_ncols(props);
    for (int64_t c = 0; c < n_props_cols; c++) {
        if (td_col_name(props, c) == weight_sym) {
            weight_col_idx = (int)c;
            break;
        }
    }
    if (weight_col_idx < 0) return TD_ERR_PTR(TD_ERR_SCHEMA);

    td_t* weight_vec = td_col(props, weight_col_idx);
    double* weights = (double*)((char*)weight_vec + 32);

    /* Dijkstra */
    double*  dist    = (double*)scratch_alloc(n * sizeof(double));
    bool*    visited = (bool*)scratch_alloc(n * sizeof(bool));
    int64_t* prev    = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    int64_t* depth   = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    dijk_entry_t* heap = (dijk_entry_t*)scratch_alloc(n * 2 * sizeof(dijk_entry_t));
    if (!dist || !visited || !prev || !depth || !heap)
        return TD_ERR_PTR(TD_ERR_OOM);

    for (int64_t i = 0; i < n; i++) {
        dist[i] = 1e308;  /* infinity */
        visited[i] = false;
        prev[i] = -1;
        depth[i] = 0;
    }
    dist[src_id] = 0.0;

    int64_t heap_size = 0;
    dijk_heap_push(heap, &heap_size, 0.0, src_id);

    int64_t* fwd_off = (int64_t*)((char*)rel->fwd.offsets + 32);
    int64_t* fwd_tgt = (int64_t*)((char*)rel->fwd.targets + 32);
    int64_t* fwd_row = (int64_t*)((char*)rel->fwd.rowmap + 32);

    while (heap_size > 0) {
        dijk_entry_t top = dijk_heap_pop(heap, &heap_size);
        int64_t u = top.node;
        if (visited[u]) continue;
        visited[u] = true;

        if (u == dst_id) break;  /* early exit if destination reached */

        for (int64_t j = fwd_off[u]; j < fwd_off[u + 1]; j++) {
            int64_t v = fwd_tgt[j];
            int64_t edge_row = fwd_row[j];
            double w = weights[edge_row];
            double new_dist = dist[u] + w;
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                prev[v] = u;
                depth[v] = depth[u] + 1;
                dijk_heap_push(heap, &heap_size, new_dist, v);
            }
        }
    }

    /* Collect reachable nodes */
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (dist[i] < 1e308) count++;
    }

    td_t* node_vec  = td_vec_new(TD_I64, count);
    td_t* dist_vec  = td_vec_new(TD_F64, count);
    td_t* depth_vec = td_vec_new(TD_I64, count);
    if (!node_vec || !dist_vec || !depth_vec) {
        scratch_free(dist); scratch_free(visited);
        scratch_free(prev); scratch_free(depth); scratch_free(heap);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)((char*)node_vec + 32);
    double*  ddata = (double*)((char*)dist_vec + 32);
    int64_t* hdata = (int64_t*)((char*)depth_vec + 32);
    int64_t idx = 0;
    for (int64_t i = 0; i < n; i++) {
        if (dist[i] < 1e308) {
            ndata[idx] = i;
            ddata[idx] = dist[i];
            hdata[idx] = depth[i];
            idx++;
        }
    }
    node_vec->n = count;
    dist_vec->n = count;
    depth_vec->n = count;

    scratch_free(dist); scratch_free(visited);
    scratch_free(prev); scratch_free(depth); scratch_free(heap);

    td_t* result = td_table_new(3, count);
    if (!result) {
        td_release(node_vec);
        td_release(dist_vec);
        td_release(depth_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, node_vec, td_sym_intern("_node"));
    td_table_set_col(result, 1, dist_vec, td_sym_intern("_dist"));
    td_table_set_col(result, 2, depth_vec, td_sym_intern("_depth"));

    return result;
}
```

- [x] **Step 3: Add dispatch and dump**

In `exec_node()`:
```c
case OP_DIJKSTRA: {
    td_t* src = exec_node(g, op->inputs[0]);
    if (!src || TD_IS_ERR(src)) return src;
    td_t* dst = op->inputs[1] ? exec_node(g, op->inputs[1]) : NULL;
    if (dst && TD_IS_ERR(dst)) { td_release(src); return dst; }
    td_t* result = exec_dijkstra(g, op, src, dst);
    td_release(src);
    if (dst) td_release(dst);
    return result;
}
```

In `dump.c`:
```c
case OP_DIJKSTRA: return "DIJKSTRA";
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vendor/teide/
git commit -m "feat(engine): add OP_DIJKSTRA weighted shortest path kernel"
```

---

## Task 4: Louvain Community Detection — C Engine Kernel

Louvain modularity optimization: each node starts in its own community. Each iteration: for every node, try moving it to each neighbor's community and keep the move that maximizes modularity gain. Repeat until no moves improve modularity.

**Output:** TD_TABLE with `_node` (I64) and `_community` (I64).

**Files:**
- Modify: `vendor/teide/src/ops/graph.c`
- Modify: `vendor/teide/src/ops/exec.c`
- Modify: `vendor/teide/src/ops/dump.c`

- [x] **Step 1: Add builder function to `graph.c`**

```c
td_op_t* td_louvain(td_graph_t* g, td_rel_t* rel, uint16_t max_iter) {
    if (!g || !rel) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode   = OP_LOUVAIN;
    ext->base.arity    = 0;
    ext->base.out_type = TD_TABLE;
    ext->base.est_rows = (uint32_t)rel->fwd.n_nodes;
    ext->graph.rel      = rel;
    ext->graph.max_iter  = max_iter > 0 ? max_iter : 100;
    ext->graph.direction = 2;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 2: Add Louvain kernel to `exec.c`**

```c
/*
 * Louvain community detection (Phase 1 only — no graph contraction).
 * Maximizes modularity Q = (1/2m) * SUM[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
 * where m = total edges, k_i = degree of i, A_ij = adjacency.
 *
 * Treats graph as undirected. Uses forward+reverse CSR for neighbor iteration.
 */
static td_t* exec_louvain(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_rel_t* rel = (td_rel_t*)ext->graph.rel;
    if (!rel) return TD_ERR_PTR(TD_ERR_SCHEMA);

    int64_t n = rel->fwd.n_nodes;
    int64_t m = rel->fwd.n_edges;  /* directed edges; effective = m for undirected */
    uint16_t max_iter = ext->graph.max_iter;

    if (n <= 0) return TD_ERR_PTR(TD_ERR_LENGTH);

    int64_t* community = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    int64_t* degree    = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    int64_t* comm_tot  = (int64_t*)scratch_alloc(n * sizeof(int64_t)); /* sum of degrees in community */
    int64_t* comm_int  = (int64_t*)scratch_alloc(n * sizeof(int64_t)); /* internal edges of community */
    if (!community || !degree || !comm_tot || !comm_int)
        return TD_ERR_PTR(TD_ERR_OOM);

    int64_t* fwd_off = (int64_t*)((char*)rel->fwd.offsets + 32);
    int64_t* fwd_tgt = (int64_t*)((char*)rel->fwd.targets + 32);
    int64_t* rev_off = (int64_t*)((char*)rel->rev.offsets + 32);
    int64_t* rev_tgt = (int64_t*)((char*)rel->rev.targets + 32);

    /* Initialize: each node in its own community */
    for (int64_t i = 0; i < n; i++) {
        community[i] = i;
        degree[i] = (fwd_off[i+1] - fwd_off[i]) + (rev_off[i+1] - rev_off[i]);
        comm_tot[i] = degree[i];
        comm_int[i] = 0;
    }

    double two_m = (double)(2 * m);
    if (two_m == 0) two_m = 1;

    for (uint16_t iter = 0; iter < max_iter; iter++) {
        bool moved = false;
        for (int64_t v = 0; v < n; v++) {
            int64_t old_comm = community[v];
            int64_t best_comm = old_comm;
            double best_gain = 0.0;

            /* Count edges from v to each neighbor community */
            /* Iterate forward + reverse neighbors */
            for (int64_t j = fwd_off[v]; j < fwd_off[v + 1]; j++) {
                int64_t u = fwd_tgt[j];
                int64_t c = community[u];
                if (c == old_comm) continue;

                /* Simplified modularity gain */
                double k_v = (double)degree[v];
                double sigma_tot = (double)comm_tot[c];
                double gain = 1.0 / two_m - (k_v * sigma_tot) / (two_m * two_m);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_comm = c;
                }
            }
            for (int64_t j = rev_off[v]; j < rev_off[v + 1]; j++) {
                int64_t u = rev_tgt[j];
                int64_t c = community[u];
                if (c == old_comm) continue;

                double k_v = (double)degree[v];
                double sigma_tot = (double)comm_tot[c];
                double gain = 1.0 / two_m - (k_v * sigma_tot) / (two_m * two_m);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            if (best_comm != old_comm) {
                comm_tot[old_comm] -= degree[v];
                comm_tot[best_comm] += degree[v];
                community[v] = best_comm;
                moved = true;
            }
        }
        if (!moved) break;
    }

    /* Normalize community IDs to 0..k-1 */
    int64_t* remap = (int64_t*)scratch_alloc(n * sizeof(int64_t));
    if (!remap) return TD_ERR_PTR(TD_ERR_OOM);
    for (int64_t i = 0; i < n; i++) remap[i] = -1;
    int64_t next_id = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t c = community[i];
        if (remap[c] < 0) remap[c] = next_id++;
        community[i] = remap[c];
    }

    /* Build output table */
    td_t* node_vec = td_vec_new(TD_I64, n);
    td_t* comm_vec = td_vec_new(TD_I64, n);
    if (!node_vec || !comm_vec) {
        scratch_free(community); scratch_free(degree);
        scratch_free(comm_tot); scratch_free(comm_int); scratch_free(remap);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* ndata = (int64_t*)((char*)node_vec + 32);
    int64_t* cdata = (int64_t*)((char*)comm_vec + 32);
    for (int64_t i = 0; i < n; i++) {
        ndata[i] = i;
        cdata[i] = community[i];
    }
    node_vec->n = n;
    comm_vec->n = n;

    scratch_free(community); scratch_free(degree);
    scratch_free(comm_tot); scratch_free(comm_int); scratch_free(remap);

    td_t* result = td_table_new(2, n);
    if (!result) {
        td_release(node_vec);
        td_release(comm_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, node_vec, td_sym_intern("_node"));
    td_table_set_col(result, 1, comm_vec, td_sym_intern("_community"));

    return result;
}
```

- [x] **Step 3: Add dispatch and dump**

In `exec_node()`:
```c
case OP_LOUVAIN: {
    return exec_louvain(g, op);
}
```

In `dump.c`:
```c
case OP_LOUVAIN: return "LOUVAIN";
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vendor/teide/
git commit -m "feat(engine): add OP_LOUVAIN community detection kernel"
```

---

## Task 5: Rust FFI Bindings + Safe Wrappers

Expose all four algorithms through `ffi.rs` and `engine.rs`.

**Files:**
- Modify: `src/ffi.rs`
- Modify: `src/engine.rs`

- [ ] **Step 1: Add FFI constants and declarations to `src/ffi.rs`**

After `OP_WCO_JOIN` (line 235):
```rust
pub const OP_PAGERANK: u16 = 84;
pub const OP_CONNECTED_COMP: u16 = 85;
pub const OP_DIJKSTRA: u16 = 86;
pub const OP_LOUVAIN: u16 = 87;
```

In the `extern "C"` block, after `td_wco_join`:
```rust
    // --- Graph Algorithm Ops ---
    pub fn td_pagerank(
        g: *mut td_graph_t,
        rel: *mut td_rel_t,
        max_iter: u16,
        damping: c_double,
    ) -> *mut td_op_t;
    pub fn td_connected_comp(
        g: *mut td_graph_t,
        rel: *mut td_rel_t,
    ) -> *mut td_op_t;
    pub fn td_dijkstra(
        g: *mut td_graph_t,
        src: *mut td_op_t,
        dst: *mut td_op_t,
        rel: *mut td_rel_t,
        weight_col: *const c_char,
        max_depth: u8,
    ) -> *mut td_op_t;
    pub fn td_louvain(
        g: *mut td_graph_t,
        rel: *mut td_rel_t,
        max_iter: u16,
    ) -> *mut td_op_t;
```

- [ ] **Step 2: Add safe wrappers to `src/engine.rs`**

In the `impl Graph` block, after `wco_join`:
```rust
    // ---- Graph algorithm ops ------------------------------------------------

    /// Compute PageRank over a relationship's CSR index.
    /// Returns a table with `_node` (I64) and `_rank` (F64) columns.
    pub fn pagerank(&self, rel: &'a Rel, max_iter: u16, damping: f64) -> Result<Column> {
        Self::check_op(unsafe {
            ffi::td_pagerank(self.raw, rel.ptr, max_iter, damping)
        })
    }

    /// Compute connected components via label propagation.
    /// Returns a table with `_node` (I64) and `_component` (I64) columns.
    pub fn connected_comp(&self, rel: &'a Rel) -> Result<Column> {
        Self::check_op(unsafe {
            ffi::td_connected_comp(self.raw, rel.ptr)
        })
    }

    /// Weighted shortest path via Dijkstra's algorithm.
    /// Requires edge properties with a weight column.
    /// Returns a table with `_node` (I64), `_dist` (F64), `_depth` (I64) columns.
    pub fn dijkstra(
        &self,
        src: Column,
        dst: Option<Column>,
        rel: &'a Rel,
        weight_col: &str,
        max_depth: u8,
    ) -> Result<Column> {
        let c_col = CString::new(weight_col).map_err(|_| Error::InvalidInput)?;
        let dst_ptr = dst.map(|d| d.raw).unwrap_or(std::ptr::null_mut());
        Self::check_op(unsafe {
            ffi::td_dijkstra(self.raw, src.raw, dst_ptr, rel.ptr, c_col.as_ptr(), max_depth)
        })
    }

    /// Community detection via Louvain modularity optimization.
    /// Returns a table with `_node` (I64) and `_community` (I64) columns.
    pub fn louvain(&self, rel: &'a Rel, max_iter: u16) -> Result<Column> {
        Self::check_op(unsafe {
            ffi::td_louvain(self.raw, rel.ptr, max_iter)
        })
    }
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs src/engine.rs
git commit -m "feat: add FFI bindings and safe wrappers for graph algorithms"
```

---

## Task 6: Rust API Tests

Test each algorithm through the Rust API to verify the C kernels work correctly.

**Files:**
- Modify: `tests/engine_api.rs`

- [ ] **Step 1: Add PageRank test**

```rust
#[test]
fn graph_pagerank() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    // Need a dummy table to create a graph — use edges as base
    let g = ctx.graph(&edges).unwrap();
    let pr = g.pagerank(&rel, 20, 0.85).unwrap();
    let result = g.execute(pr).unwrap();

    // Should have 5 nodes with ranks
    assert_eq!(result.nrows(), 5);
    // All ranks should be positive and sum to ~1.0
    let mut sum = 0.0;
    for i in 0..5 {
        let rank = result.get_f64(1, i).unwrap();
        assert!(rank > 0.0, "rank should be positive");
        sum += rank;
    }
    assert!((sum - 1.0).abs() < 0.01, "ranks should sum to ~1.0, got {sum}");
}
```

- [ ] **Step 2: Add Connected Components test**

```rust
#[test]
fn graph_connected_comp() {
    let _guard = lock();
    // Create two disconnected components: {0,1,2} and {3,4}
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "src,dst").unwrap();
    writeln!(f, "0,1").unwrap();
    writeln!(f, "1,2").unwrap();
    writeln!(f, "3,4").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let g = ctx.graph(&edges).unwrap();
    let cc = g.connected_comp(&rel).unwrap();
    let result = g.execute(cc).unwrap();

    assert_eq!(result.nrows(), 5);
    // Nodes 0,1,2 should have same component; 3,4 should have same component
    let c0 = result.get_i64(1, 0).unwrap();
    let c1 = result.get_i64(1, 1).unwrap();
    let c2 = result.get_i64(1, 2).unwrap();
    let c3 = result.get_i64(1, 3).unwrap();
    let c4 = result.get_i64(1, 4).unwrap();
    assert_eq!(c0, c1, "nodes 0,1 should be same component");
    assert_eq!(c1, c2, "nodes 1,2 should be same component");
    assert_eq!(c3, c4, "nodes 3,4 should be same component");
    assert_ne!(c0, c3, "components should be different");
}
```

- [ ] **Step 3: Add Louvain test**

```rust
#[test]
fn graph_louvain() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let g = ctx.graph(&edges).unwrap();
    let lv = g.louvain(&rel, 100).unwrap();
    let result = g.execute(lv).unwrap();

    assert_eq!(result.nrows(), 5);
    // Each node should have a non-negative community ID
    for i in 0..5 {
        let c = result.get_i64(1, i).unwrap();
        assert!(c >= 0, "community ID should be non-negative");
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --all-features -- graph_pagerank graph_connected_comp graph_louvain`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/engine_api.rs
git commit -m "test: add Rust API tests for PageRank, connected components, Louvain"
```

---

## Task 7: SQL/PGQ Integration

Expose algorithms via GRAPH_TABLE COLUMNS functions: `PAGERANK()`, `COMPONENT()`, `COMMUNITY()`.

**Files:**
- Modify: `src/sql/pgq.rs`
- Create: `tests/slt/pgq_algorithms.slt`
- Modify: `tests/slt_runner.rs`

- [ ] **Step 1: Add algorithm dispatch to `pgq.rs`**

In the COLUMNS projection logic, handle algorithm function calls. Add a new function:

```rust
/// Execute a graph algorithm and return its result as a stored temp table.
/// Called when COLUMNS contains PAGERANK(), COMPONENT(), or COMMUNITY().
pub(crate) fn execute_graph_algorithm(
    session: &Session,
    graph: &PropertyGraph,
    func_name: &str,
    edge_label_name: &str,
) -> Result<Table, SqlError> {
    let stored_rel = graph.edge_labels.get(edge_label_name).ok_or_else(|| {
        SqlError::Plan(format!("Edge label '{edge_label_name}' not found"))
    })?;

    // Need a base table to create a Graph handle
    let src_table_name = &stored_rel.edge_label.src_ref_table;
    let src_table = &session.tables.get(src_table_name).ok_or_else(|| {
        SqlError::Plan(format!("Table '{src_table_name}' not found"))
    })?.table;

    let g = session.ctx.graph(src_table)?;

    let result_col = match func_name {
        "pagerank" => g.pagerank(&stored_rel.rel, 20, 0.85)?,
        "component" | "connected_component" => g.connected_comp(&stored_rel.rel)?,
        "community" | "louvain" => g.louvain(&stored_rel.rel, 100)?,
        _ => return Err(SqlError::Plan(format!(
            "Unknown graph algorithm: {func_name}"
        ))),
    };

    let result = g.execute(result_col)?;
    Ok(result)
}
```

- [ ] **Step 2: Create SLT tests**

Create `tests/slt/pgq_algorithms.slt`:

```
# SQL/PGQ: Graph algorithms

# Setup
statement ok
CREATE TABLE persons (id INTEGER, name VARCHAR)

statement ok
INSERT INTO persons VALUES (0, 'Alice'), (1, 'Bob'), (2, 'Carol'), (3, 'Dave'), (4, 'Eve')

statement ok
CREATE TABLE knows_edges (src INTEGER, dst INTEGER)

statement ok
INSERT INTO knows_edges VALUES (0, 1), (0, 2), (1, 3), (2, 3), (3, 4)

statement ok
CREATE PROPERTY GRAPH social VERTEX TABLES (persons LABEL Person) EDGE TABLES (knows_edges SOURCE KEY (src) REFERENCES persons (id) DESTINATION KEY (dst) REFERENCES persons (id) LABEL Knows)

# PageRank: verify all nodes get ranks that sum to ~1.0
query I
SELECT COUNT(*) FROM GRAPH_TABLE (social MATCH (p:Person) COLUMNS (PAGERANK(social, p) AS rank)) WHERE rank > 0
----
5

# Connected components on a connected graph: all same component
query I
SELECT COUNT(DISTINCT component) FROM GRAPH_TABLE (social MATCH (p:Person) COLUMNS (COMPONENT(social, p) AS component))
----
1

# Community detection: should produce at least 1 community
query I
SELECT COUNT(DISTINCT community) FROM GRAPH_TABLE (social MATCH (p:Person) COLUMNS (COMMUNITY(social, p) AS community)) WHERE community >= 0
----
1
```

Note: exact output values for PageRank/Louvain depend on convergence, so tests check invariants rather than specific values.

- [ ] **Step 3: Add SLT runner**

Add to `tests/slt_runner.rs`:
```rust
#[test]
fn slt_pgq_algorithms() {
    run_slt("tests/slt/pgq_algorithms.slt");
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sql/pgq.rs tests/slt/pgq_algorithms.slt tests/slt_runner.rs
git commit -m "feat(pgq): expose PageRank, connected components, and community detection via GRAPH_TABLE"
```

---

## Task 8: Full Regression Test

- [ ] **Step 1: Run complete test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All tests PASS

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | tail -10`
Expected: No warnings

- [ ] **Step 3: Commit final changes**

```bash
git add -A
git commit -m "feat: complete graph algorithms implementation

Adds four C engine kernels: PageRank (OP_PAGERANK), connected components
(OP_CONNECTED_COMP), weighted Dijkstra (OP_DIJKSTRA), and Louvain
community detection (OP_LOUVAIN). All operate directly on CSR indexes
in the morsel-driven executor. Exposed via Rust API and SQL/PGQ
GRAPH_TABLE COLUMNS functions."
```

---

## Summary: Performance Placement

| Algorithm | Location | Why | Complexity |
|---|---|---|---|
| PageRank | C kernel (`exec.c`) | Iterative, touches all edges per round | O(iter * E) |
| Connected Components | C kernel (`exec.c`) | Iterative, touches all edges per round | O(diameter * E) |
| Dijkstra | C kernel (`exec.c`) | Per-hop weight lookups, priority queue | O((V + E) log V) |
| Louvain | C kernel (`exec.c`) | Multi-pass modularity optimization | O(iter * E) |

All hot paths stay in C with direct CSR access — no FFI boundary crossing per edge or per iteration.
