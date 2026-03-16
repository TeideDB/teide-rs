# HNSW Vector Index Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index to the C engine, reducing vector search from O(N*D) brute-force to O(D * log N), enabling efficient similarity search on 1M+ embedding collections.

**Architecture:** A new `td_hnsw_t` data structure stores the multi-layer proximity graph alongside the embedding column. Construction uses the standard HNSW insertion algorithm (greedy search + neighbor selection with `M`, `ef_construction` parameters). Search uses greedy beam search with `ef_search` parameter. The index is persisted via save/load/mmap alongside CSR indexes. SQL exposure via `CREATE VECTOR INDEX` DDL and transparent query optimization (planner rewrites KNN scans to use the index when available).

**Tech Stack:** C17 (vendor/teide/), Rust FFI, SQL planner.

**Prerequisite:** Vector search plan (TD_F32, cosine similarity, brute-force KNN).

**References:**
- [HNSW Paper (Malkov & Yashunin, 2016)](https://arxiv.org/pdf/1603.09320)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Zilliz HNSW Learn](https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW)

---

## File Structure

### C Engine (vendor/teide/)
| File | Change |
|------|--------|
| `vendor/teide/src/store/hnsw.h` | Create: `td_hnsw_t` struct, neighbor list layout, API declarations |
| `vendor/teide/src/store/hnsw.c` | Create: HNSW build, search, save, load, mmap, free |
| `vendor/teide/include/teide/td.h` | Add `OP_HNSW_KNN=91`, `td_hnsw_build()`, `td_hnsw_search()`, `td_hnsw_knn()` declarations |
| `vendor/teide/src/ops/graph.c` | Add `td_hnsw_knn()` DAG builder |
| `vendor/teide/src/ops/exec.c` | Add `exec_hnsw_knn()` kernel |
| `vendor/teide/src/ops/dump.c` | Add opcode name |
| `vendor/teide/test/test_hnsw.c` | Create: C-level unit tests for HNSW build + search |

### Rust Bindings
| File | Change |
|------|--------|
| `src/ffi.rs` | Add HNSW FFI declarations |
| `src/engine.rs` | Add `HnswIndex` RAII wrapper with build/search/save/load/mmap |

### SQL Layer
| File | Change |
|------|--------|
| `src/sql/pgq_parser.rs` | Parse `CREATE VECTOR INDEX` / `DROP VECTOR INDEX` |
| `src/sql/pgq.rs` | Store indexes in Session, transparent KNN optimization |
| `src/sql/mod.rs` | Add `vector_indexes` field to Session |

### Tests
| File | Change |
|------|--------|
| `tests/engine_api.rs` | HNSW build, search, save/load roundtrip |
| `tests/slt/vector.slt` | SQL tests for CREATE VECTOR INDEX + KNN with index |
| `tests/slt_runner.rs` | Already has `slt_vector` |

---

## Task 1: HNSW Data Structure — C Implementation

The core data structure: multi-layer proximity graph where each node has up to `M` neighbors per layer. Layer 0 has all nodes; higher layers have exponentially fewer.

**Files:**
- Create: `vendor/teide/src/store/hnsw.h`
- Create: `vendor/teide/src/store/hnsw.c`

- [x] **Step 1: Define `td_hnsw_t` structure in `hnsw.h`**

```c
#ifndef TD_HNSW_H
#define TD_HNSW_H

#include "teide/td.h"

/* ---------- HNSW Index ----------
 *
 * Multi-layer proximity graph for approximate nearest neighbor search.
 *
 * Memory layout per node:
 *   - Layer 0: up to M_max0 neighbors (default 2*M)
 *   - Layers 1+: up to M neighbors each
 *
 * Neighbor lists stored as flat arrays:
 *   neighbors[node * M_max + i] = neighbor_id  (or -1 if unused)
 *
 * Each layer stores its own neighbor array for all nodes at that layer.
 */

#define HNSW_MAX_LAYERS    16
#define HNSW_DEFAULT_M     16
#define HNSW_DEFAULT_EF_C  200
#define HNSW_DEFAULT_EF_S  50

typedef struct td_hnsw_layer {
    int64_t*  neighbors;     /* flat array: n_nodes_in_layer * M_max entries */
    int64_t   n_nodes;       /* number of nodes in this layer */
    int64_t   M_max;         /* max neighbors per node in this layer */
    int64_t*  node_ids;      /* mapping: layer_idx -> global node id */
} td_hnsw_layer_t;

typedef struct td_hnsw {
    int64_t          n_nodes;         /* total number of vectors */
    int32_t          dim;             /* embedding dimension */
    int32_t          n_layers;        /* number of layers (including layer 0) */
    int32_t          M;               /* max neighbors per node (layers 1+) */
    int32_t          M_max0;          /* max neighbors per node (layer 0) */
    int32_t          ef_construction;  /* beam width during construction */
    int64_t          entry_point;     /* entry point node (highest layer) */
    int8_t*          node_level;      /* max layer for each node (n_nodes entries) */
    td_hnsw_layer_t  layers[HNSW_MAX_LAYERS];
    const float*     vectors;         /* pointer to embedding data (not owned) */
} td_hnsw_t;

/* --- Build / Free --- */
td_hnsw_t* td_hnsw_build(const float* vectors, int64_t n_nodes, int32_t dim,
                           int32_t M, int32_t ef_construction);
void td_hnsw_free(td_hnsw_t* idx);

/* --- Search --- */
/* Returns top-K nearest neighbors as (node_id, distance) pairs.
 * out_ids and out_dists must be pre-allocated with k entries.
 * Returns actual number of results (may be < k). */
int64_t td_hnsw_search(const td_hnsw_t* idx,
                         const float* query, int32_t dim,
                         int64_t k, int32_t ef_search,
                         int64_t* out_ids, double* out_dists);

/* --- Persistence --- */
td_err_t td_hnsw_save(const td_hnsw_t* idx, const char* dir);
td_hnsw_t* td_hnsw_load(const char* dir);
td_hnsw_t* td_hnsw_mmap(const char* dir);

#endif /* TD_HNSW_H */
```

- [x] **Step 2: Implement HNSW construction in `hnsw.c`**

```c
#include "hnsw.h"
#include <math.h>
#include <string.h>

/* --- Distance function (cosine similarity → distance = 1 - sim) --- */
static double hnsw_cosine_dist(const float* a, const float* b, int32_t dim) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int32_t i = 0; i < dim; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    return (denom > 0.0) ? 1.0 - dot / denom : 1.0;
}

/* --- Random level assignment --- */
static int32_t hnsw_random_level(int32_t M) {
    double ml = 1.0 / log((double)M);
    /* floor(-ln(uniform(0,1)) * ml) */
    double r = (double)rand() / (double)RAND_MAX;
    if (r == 0.0) r = 1e-10;
    int32_t level = (int32_t)floor(-log(r) * ml);
    if (level >= HNSW_MAX_LAYERS) level = HNSW_MAX_LAYERS - 1;
    return level;
}

/* --- Priority queue (min-heap by distance) for beam search --- */
typedef struct {
    int64_t id;
    double  dist;
} hnsw_candidate_t;

/* (heap helpers omitted for brevity — same pattern as Dijkstra min-heap) */

/* --- Greedy search on a single layer --- */
/* Starts from entry_points, explores ef nearest candidates, returns them. */
static int64_t hnsw_search_layer(
    const td_hnsw_t* idx,
    const float* query,
    int64_t* entry_points, int64_t n_entries,
    int32_t layer_idx,
    int32_t ef,
    hnsw_candidate_t* results  /* pre-allocated, ef entries */
) {
    /* Standard HNSW layer search:
     * 1. Initialize candidate set C and visited set V with entry points
     * 2. While C is not empty:
     *    a. Pick closest unvisited candidate c from C
     *    b. If c is farther than the farthest in results (W), break
     *    c. For each neighbor n of c in this layer:
     *       - If n not visited: compute dist, add to C and W if closer
     * 3. Return W (the ef nearest found)
     */
    /* Implementation uses the layer's neighbor array directly */

    td_hnsw_layer_t* layer = &((td_hnsw_t*)idx)->layers[layer_idx];
    /* ... full implementation ... */

    return 0; /* actual result count */
}

/* --- Main build function --- */
td_hnsw_t* td_hnsw_build(const float* vectors, int64_t n_nodes, int32_t dim,
                           int32_t M, int32_t ef_construction) {
    if (!vectors || n_nodes <= 0 || dim <= 0) return NULL;
    if (M <= 0) M = HNSW_DEFAULT_M;
    if (ef_construction <= 0) ef_construction = HNSW_DEFAULT_EF_C;

    td_hnsw_t* idx = (td_hnsw_t*)td_sys_alloc(sizeof(td_hnsw_t));
    if (!idx) return NULL;
    memset(idx, 0, sizeof(td_hnsw_t));

    idx->n_nodes = n_nodes;
    idx->dim = dim;
    idx->M = M;
    idx->M_max0 = 2 * M;
    idx->ef_construction = ef_construction;
    idx->entry_point = 0;
    idx->vectors = vectors;

    /* Allocate node levels */
    idx->node_level = (int8_t*)td_sys_alloc((size_t)n_nodes * sizeof(int8_t));
    if (!idx->node_level) { td_sys_free(idx); return NULL; }

    /* Assign random levels to all nodes */
    int32_t max_level = 0;
    for (int64_t i = 0; i < n_nodes; i++) {
        int32_t level = hnsw_random_level(M);
        idx->node_level[i] = (int8_t)level;
        if (level > max_level) max_level = level;
    }
    idx->n_layers = max_level + 1;

    /* Allocate layers */
    for (int32_t l = 0; l < idx->n_layers; l++) {
        td_hnsw_layer_t* layer = &idx->layers[l];

        /* Count nodes at this layer */
        int64_t count = 0;
        for (int64_t i = 0; i < n_nodes; i++) {
            if (idx->node_level[i] >= l) count++;
        }
        layer->n_nodes = count;
        layer->M_max = (l == 0) ? idx->M_max0 : M;

        /* Allocate neighbor array and node_ids mapping */
        size_t nb_size = (size_t)count * layer->M_max * sizeof(int64_t);
        layer->neighbors = (int64_t*)td_sys_alloc(nb_size);
        layer->node_ids  = (int64_t*)td_sys_alloc((size_t)count * sizeof(int64_t));
        if (!layer->neighbors || !layer->node_ids) {
            td_hnsw_free(idx);
            return NULL;
        }

        /* Initialize neighbors to -1 (empty) */
        memset(layer->neighbors, 0xFF, nb_size); /* -1 in two's complement */

        /* Fill node_ids mapping */
        int64_t j = 0;
        for (int64_t i = 0; i < n_nodes; i++) {
            if (idx->node_level[i] >= l) {
                layer->node_ids[j++] = i;
            }
        }
    }

    /* Insert nodes one by one (standard HNSW construction) */
    /* First node is the entry point */
    for (int64_t i = 1; i < n_nodes; i++) {
        const float* vec = vectors + i * dim;
        int32_t node_level = idx->node_level[i];

        /* Greedy search from top layer down to node_level+1 */
        int64_t ep = idx->entry_point;
        for (int32_t l = idx->n_layers - 1; l > node_level; l--) {
            /* Find closest node in this layer */
            /* ... greedy 1-nn search ... */
        }

        /* Insert into layers node_level down to 0 */
        for (int32_t l = node_level; l >= 0; l--) {
            /* Search for ef_construction nearest neighbors */
            /* Connect node to M nearest found */
            /* Prune if any neighbor exceeds M_max */
        }

        /* Update entry point if this node has higher level */
        if (node_level >= idx->n_layers - 1) {
            idx->entry_point = i;
        }
    }

    return idx;
}

/* --- Free --- */
void td_hnsw_free(td_hnsw_t* idx) {
    if (!idx) return;
    for (int32_t l = 0; l < idx->n_layers; l++) {
        if (idx->layers[l].neighbors) td_sys_free(idx->layers[l].neighbors);
        if (idx->layers[l].node_ids) td_sys_free(idx->layers[l].node_ids);
    }
    if (idx->node_level) td_sys_free(idx->node_level);
    td_sys_free(idx);
}
```

Note: The full HNSW construction is ~300 lines of C. The plan provides the structure and key functions; the implementer should follow the HNSW paper (Algorithm 1) for the complete insertion logic including neighbor selection heuristic.

- [x] **Step 3: Implement HNSW search in `hnsw.c`**

```c
/* --- Search: find K approximate nearest neighbors --- */
int64_t td_hnsw_search(const td_hnsw_t* idx,
                         const float* query, int32_t dim,
                         int64_t k, int32_t ef_search,
                         int64_t* out_ids, double* out_dists) {
    if (!idx || !query || dim != idx->dim || k <= 0) return 0;
    if (ef_search < k) ef_search = (int32_t)k;
    if (idx->n_nodes == 0) return 0;

    /* Phase 1: Greedy descent from top layer to layer 1 */
    int64_t ep = idx->entry_point;
    for (int32_t l = idx->n_layers - 1; l >= 1; l--) {
        /* 1-nearest-neighbor search from ep in layer l */
        /* ep = closest node found */
    }

    /* Phase 2: Beam search on layer 0 with ef_search width */
    hnsw_candidate_t* candidates = (hnsw_candidate_t*)td_sys_alloc(
        (size_t)ef_search * sizeof(hnsw_candidate_t));
    if (!candidates) return 0;

    int64_t n_found = hnsw_search_layer(idx, query, &ep, 1, 0, ef_search, candidates);

    /* Extract top-K from candidates (sorted by distance) */
    /* Sort candidates by distance ascending */
    /* ... insertion sort (ef_search is small) ... */

    int64_t result_count = (n_found < k) ? n_found : k;
    for (int64_t i = 0; i < result_count; i++) {
        out_ids[i]   = candidates[i].id;
        out_dists[i] = candidates[i].dist;
    }

    td_sys_free(candidates);
    return result_count;
}
```

- [x] **Step 4: Implement persistence (save/load/mmap) in `hnsw.c`**

File layout on disk:
```
header:    n_nodes(i64) dim(i32) n_layers(i32) M(i32) M_max0(i32) entry_point(i64)
levels:    node_level[n_nodes] (i8 array)
per layer: n_nodes_in_layer(i64) M_max(i64)
           neighbors[n_nodes_in_layer * M_max] (i64 array)
           node_ids[n_nodes_in_layer] (i64 array)
```

```c
td_err_t td_hnsw_save(const td_hnsw_t* idx, const char* dir) {
    /* Create directory, write header file + per-layer neighbor files */
    /* Same pattern as td_rel_save */
}

td_hnsw_t* td_hnsw_load(const char* dir) {
    /* Read header, allocate, read neighbor arrays */
}

td_hnsw_t* td_hnsw_mmap(const char* dir) {
    /* Memory-map neighbor arrays for zero-copy */
}
```

- [x] **Step 5: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add -f vendor/teide/src/store/hnsw.h vendor/teide/src/store/hnsw.c
git commit -m "feat(engine): add HNSW index data structure with build, search, and persistence"
```

---

## Task 2: HNSW DAG Integration — C Engine Op

Wire HNSW search into the execution DAG as `OP_HNSW_KNN`.

**Files:**
- Modify: `vendor/teide/include/teide/td.h`
- Modify: `vendor/teide/src/ops/graph.c`
- Modify: `vendor/teide/src/ops/exec.c`
- Modify: `vendor/teide/src/ops/dump.c`

- [x] **Step 1: Add opcode and declaration to `td.h`**

```c
#define OP_HNSW_KNN  91   /* HNSW approximate K nearest neighbors */
```

```c
/* HNSW-accelerated KNN (uses pre-built index instead of brute-force) */
td_op_t* td_hnsw_knn(td_graph_t* g, td_hnsw_t* idx,
                       const float* query_vec, int32_t dim,
                       int64_t k, int32_t ef_search);
```

Extend `td_op_ext_t` union:
```c
struct {  /* OP_HNSW_KNN */
    void*     hnsw_idx;       /* td_hnsw_t* (opaque) */
    float*    query_vec;
    int32_t   dim;
    int64_t   k;
    int32_t   ef_search;
} hnsw;
```

- [x] **Step 2: Add builder and kernel**

Builder in `graph.c`:
```c
td_op_t* td_hnsw_knn(td_graph_t* g, td_hnsw_t* idx,
                       const float* query_vec, int32_t dim,
                       int64_t k, int32_t ef_search) {
    if (!g || !idx || !query_vec || dim <= 0 || k <= 0) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    ext->base.opcode    = OP_HNSW_KNN;
    ext->base.arity     = 0;  /* nullary: reads from index directly */
    ext->base.out_type  = TD_TABLE;
    ext->base.est_rows  = (uint32_t)k;
    ext->hnsw.hnsw_idx  = idx;
    ext->hnsw.query_vec = (float*)query_vec;
    ext->hnsw.dim       = dim;
    ext->hnsw.k         = k;
    ext->hnsw.ef_search = ef_search > 0 ? ef_search : HNSW_DEFAULT_EF_S;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

Kernel in `exec.c`:
```c
static td_t* exec_hnsw_knn(td_graph_t* g, td_op_t* op) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    td_hnsw_t* idx = (td_hnsw_t*)ext->hnsw.hnsw_idx;
    const float* query = ext->hnsw.query_vec;
    int32_t dim = ext->hnsw.dim;
    int64_t k = ext->hnsw.k;
    int32_t ef = ext->hnsw.ef_search;

    if (!idx || !query) return TD_ERR_PTR(TD_ERR_SCHEMA);

    /* Pre-allocate output arrays */
    int64_t* ids   = (int64_t*)td_sys_alloc((size_t)k * sizeof(int64_t));
    double*  dists = (double*)td_sys_alloc((size_t)k * sizeof(double));
    if (!ids || !dists) {
        if (ids) td_sys_free(ids);
        if (dists) td_sys_free(dists);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t n_found = td_hnsw_search(idx, query, dim, k, ef, ids, dists);

    /* Build output table: _rowid (I64), _similarity (F64) */
    td_t* rowid_vec = td_vec_new(TD_I64, n_found);
    td_t* sim_vec   = td_vec_new(TD_F64, n_found);
    if (!rowid_vec || !sim_vec) {
        td_sys_free(ids); td_sys_free(dists);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* rdata = (int64_t*)td_data(rowid_vec);
    double*  sdata = (double*)td_data(sim_vec);
    for (int64_t i = 0; i < n_found; i++) {
        rdata[i] = ids[i];
        sdata[i] = 1.0 - dists[i];  /* convert distance back to similarity */
    }
    rowid_vec->len = n_found;
    sim_vec->len   = n_found;

    td_sys_free(ids);
    td_sys_free(dists);

    td_t* result = td_table_new(2, n_found);
    if (!result) {
        td_release(rowid_vec); td_release(sim_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, rowid_vec, td_sym_intern("_rowid"));
    td_table_set_col(result, 1, sim_vec, td_sym_intern("_similarity"));

    return result;
}
```

- [x] **Step 3: Add dispatch and dump**

```c
case OP_HNSW_KNN: return exec_hnsw_knn(g, op);
```

```c
case OP_HNSW_KNN: return "HNSW_KNN";
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -f vendor/teide/
git commit -m "feat(engine): add OP_HNSW_KNN execution kernel"
```

---

## Task 3: Rust FFI + Safe Wrapper

**Files:**
- Modify: `src/ffi.rs`
- Modify: `src/engine.rs`

- [x] **Step 1: Add FFI declarations**

```rust
// --- HNSW Index ---
pub const OP_HNSW_KNN: u16 = 91;

// Opaque pointer type
pub enum td_hnsw_t {}

extern "C" {
    pub fn td_hnsw_build(
        vectors: *const f32,
        n_nodes: i64,
        dim: i32,
        m: i32,
        ef_construction: i32,
    ) -> *mut td_hnsw_t;
    pub fn td_hnsw_free(idx: *mut td_hnsw_t);
    pub fn td_hnsw_search(
        idx: *const td_hnsw_t,
        query: *const f32,
        dim: i32,
        k: i64,
        ef_search: i32,
        out_ids: *mut i64,
        out_dists: *mut f64,
    ) -> i64;
    pub fn td_hnsw_save(idx: *const td_hnsw_t, dir: *const c_char) -> td_err_t;
    pub fn td_hnsw_load(dir: *const c_char) -> *mut td_hnsw_t;
    pub fn td_hnsw_mmap(dir: *const c_char) -> *mut td_hnsw_t;
    pub fn td_hnsw_knn(
        g: *mut td_graph_t,
        idx: *mut td_hnsw_t,
        query_vec: *const f32,
        dim: i32,
        k: i64,
        ef_search: i32,
    ) -> *mut td_op_t;
}
```

- [x] **Step 2: Add `HnswIndex` RAII wrapper in `engine.rs`**

```rust
/// RAII wrapper for a C-allocated HNSW index.
pub struct HnswIndex {
    ptr: *mut ffi::td_hnsw_t,
    _engine: Arc<EngineGuard>,
}

impl HnswIndex {
    /// Build an HNSW index from embedding data.
    pub fn build(
        ctx: &Context,
        vectors: &[f32],
        n_nodes: i64,
        dim: i32,
        m: i32,
        ef_construction: i32,
    ) -> Result<Self> {
        if vectors.len() != (n_nodes * dim as i64) as usize {
            return Err(Error::Length);
        }
        let ptr = unsafe {
            ffi::td_hnsw_build(vectors.as_ptr(), n_nodes, dim, m, ef_construction)
        };
        if ptr.is_null() {
            return Err(Error::Oom);
        }
        Ok(HnswIndex {
            ptr,
            _engine: ctx.engine_guard(),
        })
    }

    /// Search for K nearest neighbors.
    pub fn search(&self, query: &[f32], k: i64, ef_search: i32) -> Result<Vec<(i64, f64)>> {
        let dim = query.len() as i32;
        let mut ids = vec![0i64; k as usize];
        let mut dists = vec![0f64; k as usize];
        let n_found = unsafe {
            ffi::td_hnsw_search(
                self.ptr, query.as_ptr(), dim, k, ef_search,
                ids.as_mut_ptr(), dists.as_mut_ptr(),
            )
        };
        ids.truncate(n_found as usize);
        dists.truncate(n_found as usize);
        Ok(ids.into_iter().zip(dists).collect())
    }

    /// Save index to disk.
    pub fn save(&self, dir: &str) -> Result<()> {
        let c_dir = CString::new(dir).map_err(|_| Error::InvalidInput)?;
        let err = unsafe { ffi::td_hnsw_save(self.ptr, c_dir.as_ptr()) };
        if err != ffi::td_err_t::TD_OK {
            return Err(Error::from_code(err));
        }
        Ok(())
    }

    /// Load index from disk.
    pub fn load(dir: &str) -> Result<Self> {
        let engine = acquire_existing_engine_guard()?;
        let c_dir = CString::new(dir).map_err(|_| Error::InvalidInput)?;
        let ptr = unsafe { ffi::td_hnsw_load(c_dir.as_ptr()) };
        if ptr.is_null() { return Err(Error::Io); }
        Ok(HnswIndex { ptr, _engine: engine })
    }

    /// Memory-map index from disk (zero-copy).
    pub fn mmap(dir: &str) -> Result<Self> {
        let engine = acquire_existing_engine_guard()?;
        let c_dir = CString::new(dir).map_err(|_| Error::InvalidInput)?;
        let ptr = unsafe { ffi::td_hnsw_mmap(c_dir.as_ptr()) };
        if ptr.is_null() { return Err(Error::Io); }
        Ok(HnswIndex { ptr, _engine: engine })
    }

    pub fn as_raw(&self) -> *mut ffi::td_hnsw_t { self.ptr }
}

impl Drop for HnswIndex {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::td_hnsw_free(self.ptr); }
        }
    }
}
```

- [x] **Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add src/ffi.rs src/engine.rs
git commit -m "feat: add HnswIndex RAII wrapper with build/search/save/load/mmap"
```

---

## Task 4: Rust API Tests

**Files:**
- Modify: `tests/engine_api.rs`

- [ ] **Step 1: Add HNSW build + search test**

```rust
#[test]
fn hnsw_build_and_search() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 4;
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // 0: x-axis
        0.0, 1.0, 0.0, 0.0,  // 1: y-axis
        0.0, 0.0, 1.0, 0.0,  // 2: z-axis
        0.9, 0.1, 0.0, 0.0,  // 3: near x-axis
        0.1, 0.9, 0.0, 0.0,  // 4: near y-axis
    ];

    let idx = HnswIndex::build(&ctx, &vectors, 5, dim, 4, 20).unwrap();

    // Search for vectors near x-axis
    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let results = idx.search(&query, 3, 10).unwrap();

    assert_eq!(results.len(), 3);
    // Top result should be node 0 (exact match)
    assert_eq!(results[0].0, 0, "nearest should be node 0");
    // Node 3 should be second (0.9, 0.1, 0, 0)
    assert_eq!(results[1].0, 3, "second nearest should be node 3");
}
```

- [ ] **Step 2: Add HNSW save/load roundtrip test**

```rust
#[test]
fn hnsw_save_load_roundtrip() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    let idx = HnswIndex::build(&ctx, &vectors, 3, dim, 4, 20).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap();
    idx.save(dir_path).unwrap();

    let idx2 = HnswIndex::load(dir_path).unwrap();
    let query = vec![1.0f32, 0.0, 0.0];
    let results = idx2.search(&query, 1, 10).unwrap();
    assert_eq!(results[0].0, 0, "loaded index should find node 0");
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --all-features -- hnsw_build_and_search hnsw_save_load_roundtrip`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/engine_api.rs
git commit -m "test: add HNSW index build, search, and persistence tests"
```

---

## Task 5: SQL Integration — CREATE VECTOR INDEX

**Files:**
- Modify: `src/sql/pgq_parser.rs` (parse CREATE/DROP VECTOR INDEX)
- Modify: `src/sql/pgq.rs` (execute, store in session)
- Modify: `src/sql/mod.rs` (add vector_indexes to Session)

- [ ] **Step 1: Add vector index storage to Session**

In `src/sql/mod.rs`:
```rust
pub struct Session {
    pub(crate) tables: HashMap<String, StoredTable>,
    pub(crate) graphs: HashMap<String, pgq::PropertyGraph>,
    pub(crate) vector_indexes: HashMap<String, engine::HnswIndex>,
    pub(crate) ctx: Context,
}
```

- [ ] **Step 2: Parse CREATE/DROP VECTOR INDEX in `pgq_parser.rs`**

SQL syntax:
```sql
CREATE VECTOR INDEX idx_name ON table(column) USING HNSW(M=16, ef_construction=200);
DROP VECTOR INDEX idx_name;
```

Add to `try_parse_pgq()`:
```rust
if upper.starts_with("CREATE VECTOR INDEX") {
    return Ok(Some(parse_create_vector_index(trimmed)?));
}
if upper.starts_with("DROP VECTOR INDEX") {
    return Ok(Some(parse_drop_vector_index(trimmed)?));
}
```

- [ ] **Step 3: Implement index building in `pgq.rs`**

When `CREATE VECTOR INDEX` is executed:
1. Look up the table and embedding column
2. Extract the F32 data
3. Call `HnswIndex::build()`
4. Store in `session.vector_indexes`

- [ ] **Step 4: Transparent KNN optimization**

When the planner sees a KNN() table function and a vector index exists on the target column, automatically rewrite to use `OP_HNSW_KNN` instead of `OP_KNN` (brute-force).

- [ ] **Step 5: Add SLT tests**

Append to `tests/slt/vector.slt`:
```
# CREATE VECTOR INDEX
statement ok
CREATE VECTOR INDEX emb_idx ON docs(embedding) USING HNSW(M=16, ef_construction=200)

# KNN should use the index (transparent optimization)
# ... query tests ...

# DROP VECTOR INDEX
statement ok
DROP VECTOR INDEX emb_idx
```

- [ ] **Step 6: Run all tests**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/sql/pgq_parser.rs src/sql/pgq.rs src/sql/mod.rs tests/slt/vector.slt
git commit -m "feat(sql): add CREATE/DROP VECTOR INDEX with transparent KNN optimization"
```

---

## Task 6: Full Regression Test

- [ ] **Step 1: Run complete test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | tail -10`
Expected: No warnings

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: complete HNSW vector index implementation

Adds td_hnsw_t data structure with multi-layer proximity graph,
O(D * log N) approximate nearest neighbor search, save/load/mmap
persistence, and transparent SQL optimization via CREATE VECTOR INDEX."
```

---

## Performance Characteristics

| Operation | Brute-force KNN | HNSW KNN |
|---|---|---|
| Build | N/A | O(N * D * M * log N) |
| Search | O(N * D) | O(D * log N) |
| Memory | 0 (scan in place) | O(N * M * layers) |
| 100K vectors, dim=768 | ~50ms | ~0.5ms |
| 1M vectors, dim=768 | ~500ms | ~1ms |
| 10M vectors, dim=768 | ~5s | ~2ms |
| Recall@10 | 100% (exact) | ~95-99% (approximate) |

HNSW parameters:
- `M=16`: good balance of speed and recall
- `ef_construction=200`: high quality graph
- `ef_search=50`: tunable at query time (higher = better recall, slower)
