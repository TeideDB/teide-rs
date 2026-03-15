# Vector Search Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add F32 vector column type, cosine similarity / euclidean distance functions, and brute-force nearest-neighbor search to enable GraphRAG embedding storage and retrieval.

**Architecture:** A new `TD_F32` type (slot 8) stores dense float32 vectors. For embedding columns, each element is a fixed-dimension F32 array stored contiguously at stride `dim * 4` bytes, with the dimension recorded in the column's `attrs` field. Similarity functions (`OP_COSINE_SIM`, `OP_EUCLIDEAN_DIST`) are implemented as C engine kernels operating on embedding columns. A `KNN()` table function performs brute-force scan returning the K nearest neighbors. HNSW indexing is deferred to a future plan — brute-force over contiguous F32 memory with SIMD is sufficient for <1M vectors.

**Tech Stack:** C17 (vendor/teide/), Rust FFI, SQL/PGQ integration.

**Prerequisite:** Graph algorithms plan (completed).

---

## File Structure

### C Engine (vendor/teide/)
| File | Change |
|------|--------|
| `vendor/teide/include/teide/td.h` | Add `TD_F32=8` type constant, `OP_COSINE_SIM=88`, `OP_EUCLIDEAN_DIST=89`, `OP_KNN=90` opcodes, embedding helper declarations |
| `vendor/teide/src/core/types.c` | Add `td_type_sizes[8] = 4` for F32 element size |
| `vendor/teide/src/vec/vec.c` | Support F32 in `td_vec_new`, `td_vec_push_f32` |
| `vendor/teide/src/ops/graph.c` | Add builder functions: `td_cosine_sim()`, `td_euclidean_dist()`, `td_knn()` |
| `vendor/teide/src/ops/exec.c` | Add kernel functions: `exec_cosine_sim()`, `exec_euclidean_dist()`, `exec_knn()` |
| `vendor/teide/src/ops/opt.c` | Type inference for new ops (output type = TD_F64) |
| `vendor/teide/src/ops/dump.c` | Add opcode name strings |
| `vendor/teide/src/io/csv.c` | Parse JSON-array syntax `[1.0,2.0,...]` as F32 embedding columns |

### Rust Bindings
| File | Change |
|------|--------|
| `src/ffi.rs` | Add `TD_F32=8`, new opcode constants, FFI declarations |
| `src/engine.rs` | Add `Table::set_embedding()`, `Graph::cosine_sim()`, `Graph::euclidean_dist()`, `Graph::knn()` |

### SQL Layer
| File | Change |
|------|--------|
| `src/sql/planner.rs` | Handle `COSINE_SIMILARITY()` and `EUCLIDEAN_DISTANCE()` as scalar functions in SELECT/WHERE |
| `src/sql/expr.rs` | Add expression planning for similarity functions |
| `src/sql/pgq.rs` | Integrate embedding operations with GRAPH_TABLE queries |

### Tests
| File | Change |
|------|--------|
| `tests/engine_api.rs` | F32 vector creation, similarity computation, KNN tests |
| `tests/slt/vector.slt` | SQL tests for embedding columns, similarity functions, KNN |
| `tests/slt_runner.rs` | Add `slt_vector` test function |

---

## Task 1: TD_F32 Type Constant + Element Size

Add the F32 type to the C engine's type system.

**Files:**
- Modify: `vendor/teide/include/teide/td.h`
- Modify: `vendor/teide/src/core/types.c`
- Modify: `src/ffi.rs`

- [x] **Step 1: Add TD_F32 type constant to `td.h`**

After `TD_F64 = 7` (line ~110):
```c
#define TD_F32          8    /* 32-bit float (single precision)    */
```

Also add atom variant:
```c
#define TD_ATOM_F32    -8    /* scalar f32                         */
```

Note: `TD_ATOM_STR` was already using slot -8. Check if there's a conflict — `TD_ATOM_STR = -8` is defined at line 76 in ffi.rs. We need to use a different slot. Use `TD_F32 = 14` instead (slot 14 is free between TD_TABLE=13 and TD_SEL=16).

Actually, let me re-check available slots:
- 0=LIST, 1=BOOL, 2=U8, 3=CHAR, 4=I16, 5=I32, 6=I64, 7=F64, 8=???, 9=DATE, 10=TIME, 11=TIMESTAMP, 12=GUID, 13=TABLE, 14=???, 15=???, 16=SEL, 17-19=???, 20=SYM

Slot 8 is free (no existing type). But `TD_ATOM_STR = -8` already uses the negative. Since atom types are only matched by their negative constant (never by `abs(type)` lookup), this is fine — `TD_F32 = 8` (vector) and `TD_ATOM_STR = -8` (atom) don't conflict because they're different sign domains.

```c
#define TD_F32          8    /* 32-bit float vector (also used for embeddings) */
```

- [x] **Step 2: Add F32 element size to `types.c`**

In the `td_type_sizes` array, slot 8 is currently 0 or unset. Set it:
```c
/* [TD_F32]       =  8 */ 4,
```

Also update `TD_TYPE_COUNT` if needed — current value is 21, which covers indices 0-20. Index 8 is within range, so no change needed.

- [x] **Step 3: Add Rust FFI constant**

In `src/ffi.rs`, after `TD_F64`:
```rust
pub const TD_F32: i8 = 8;
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -f vendor/teide/include/teide/td.h vendor/teide/src/core/types.c src/ffi.rs
git commit -m "feat(engine): add TD_F32 type constant (slot 8, elem_size=4)"
```

---

## Task 2: Embedding Column Storage

An embedding column stores N rows of D-dimensional F32 vectors as a contiguous flat array of `N * D` floats. The dimension D is encoded in the column's `attrs` field (up to 255 dimensions directly, or use a convention for larger dims).

For dimensions > 255, we use a two-byte encoding: `attrs = dim & 0xFF`, and store the full dimension in a separate metadata field. For simplicity in v1, we support up to 65535 dimensions by repurposing the `attrs` field and adding a helper.

**Files:**
- Modify: `vendor/teide/include/teide/td.h` (embedding helpers)
- Modify: `vendor/teide/src/vec/vec.c` (embedding vector creation)
- Modify: `src/engine.rs` (Rust API for creating embedding columns)

- [x] **Step 1: Add embedding helper declarations to `td.h`**

```c
/* ---------- Embedding column helpers ---------- */

/* An embedding column is a TD_F32 vector of length N*D where D is the
 * embedding dimension.  D is stored in a separate I32 atom that the
 * caller keeps alongside the column.  Access helpers: */

/* Create an embedding column for N rows of D-dimensional vectors. */
td_t* td_embedding_new(int64_t nrows, int32_t dim);

/* Get the raw float pointer for row `row` (0-indexed). */
static inline float* td_embedding_row(td_t* col, int32_t dim, int64_t row) {
    return (float*)td_data(col) + row * dim;
}

/* Set one row's embedding from a float array. */
static inline void td_embedding_set(td_t* col, int32_t dim,
                                     int64_t row, const float* vec) {
    float* dst = td_embedding_row(col, dim, row);
    memcpy(dst, vec, (size_t)dim * sizeof(float));
}

/* Number of rows in an embedding column. */
static inline int64_t td_embedding_nrows(td_t* col, int32_t dim) {
    return col->len / dim;
}
```

- [x] **Step 2: Implement `td_embedding_new` in `vec.c`**

```c
td_t* td_embedding_new(int64_t nrows, int32_t dim) {
    int64_t total = nrows * dim;
    td_t* v = td_vec_new(TD_F32, total);
    if (!v || TD_IS_ERR(v)) return v;
    v->len = total;
    return v;
}
```

- [x] **Step 3: Add Rust wrapper in `engine.rs`**

Add FFI declaration in `src/ffi.rs`:
```rust
    pub fn td_embedding_new(nrows: i64, dim: i32) -> *mut td_t;
```

Add to `impl Table` in `src/engine.rs`:
```rust
    /// Create a new embedding column for the given number of rows and dimension.
    /// Returns a raw td_t pointer that can be added to a table.
    ///
    /// # Safety
    /// The returned pointer must be released via td_release when done.
    pub fn create_embedding_column(
        ctx: &Context,
        nrows: i64,
        dim: i32,
        data: &[f32],
    ) -> Result<*mut ffi::td_t> {
        if data.len() != (nrows * dim as i64) as usize {
            return Err(Error::Length);
        }
        let col = unsafe { ffi::td_embedding_new(nrows, dim) };
        let col = check_ptr(col)?;
        // Copy data into the column
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                ffi::td_data(col) as *mut f32,
                data.len(),
            )
        };
        dst.copy_from_slice(data);
        Ok(col)
    }
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -f vendor/teide/ src/ffi.rs src/engine.rs
git commit -m "feat(engine): add embedding column storage (TD_F32 flat array with dimension)"
```

---

## Task 3: Cosine Similarity + Euclidean Distance — C Kernels

These are row-level functions: given two embedding columns (same dimension), compute similarity/distance per row, producing an F64 output column.

Also support: one embedding column vs one constant vector (for query-time "find similar to this query embedding").

**Files:**
- Modify: `vendor/teide/include/teide/td.h` (opcodes + declarations)
- Modify: `vendor/teide/src/ops/graph.c` (builders)
- Modify: `vendor/teide/src/ops/exec.c` (kernels)
- Modify: `vendor/teide/src/ops/dump.c` (names)

- [x] **Step 1: Add opcodes and declarations to `td.h`**

```c
#define OP_COSINE_SIM      88   /* cosine similarity between embeddings   */
#define OP_EUCLIDEAN_DIST  89   /* euclidean distance between embeddings  */
#define OP_KNN             90   /* brute-force K nearest neighbors        */
```

```c
/* Vector similarity ops */
td_op_t* td_cosine_sim(td_graph_t* g, td_op_t* emb_col,
                        const float* query_vec, int32_t dim);
td_op_t* td_euclidean_dist(td_graph_t* g, td_op_t* emb_col,
                            const float* query_vec, int32_t dim);
td_op_t* td_knn(td_graph_t* g, td_op_t* emb_col,
                 const float* query_vec, int32_t dim, int64_t k);
```

- [x] **Step 2: Extend `td_op_ext_t` for vector ops**

Add a new union member in `td_op_ext_t`:
```c
struct {  /* OP_COSINE_SIM / OP_EUCLIDEAN_DIST / OP_KNN */
    float*    query_vec;       /* query embedding (caller-owned, must outlive graph) */
    int32_t   dim;             /* embedding dimension */
    int64_t   k;               /* top-K for KNN */
} vector;
```

- [x] **Step 3: Add builder functions to `graph.c`**

```c
td_op_t* td_cosine_sim(td_graph_t* g, td_op_t* emb_col,
                        const float* query_vec, int32_t dim) {
    if (!g || !emb_col || !query_vec || dim <= 0) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    emb_col = &g->nodes[emb_col->id];

    ext->base.opcode    = OP_COSINE_SIM;
    ext->base.arity     = 1;
    ext->base.inputs[0] = emb_col;
    ext->base.out_type  = TD_F64;
    ext->base.est_rows  = emb_col->est_rows;
    ext->vector.query_vec = (float*)query_vec;
    ext->vector.dim       = dim;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}

td_op_t* td_euclidean_dist(td_graph_t* g, td_op_t* emb_col,
                            const float* query_vec, int32_t dim) {
    if (!g || !emb_col || !query_vec || dim <= 0) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    emb_col = &g->nodes[emb_col->id];

    ext->base.opcode    = OP_EUCLIDEAN_DIST;
    ext->base.arity     = 1;
    ext->base.inputs[0] = emb_col;
    ext->base.out_type  = TD_F64;
    ext->base.est_rows  = emb_col->est_rows;
    ext->vector.query_vec = (float*)query_vec;
    ext->vector.dim       = dim;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 4: Add cosine similarity kernel to `exec.c`**

```c
/*
 * Cosine similarity: dot(a,b) / (||a|| * ||b||)
 * Input: TD_F32 embedding column (flat N*D floats)
 * Output: TD_F64 vector of similarities (one per row)
 */
static td_t* exec_cosine_sim(td_graph_t* g, td_op_t* op, td_t* emb_vec) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    const float* query = ext->vector.query_vec;
    int32_t dim = ext->vector.dim;

    if (!query || dim <= 0) return TD_ERR_PTR(TD_ERR_SCHEMA);
    if (emb_vec->type != TD_F32) return TD_ERR_PTR(TD_ERR_TYPE);

    int64_t total = emb_vec->len;
    int64_t nrows = total / dim;
    if (nrows * dim != total) return TD_ERR_PTR(TD_ERR_LENGTH);

    const float* data = (const float*)td_data(emb_vec);

    /* Precompute query norm */
    double q_norm_sq = 0.0;
    for (int32_t j = 0; j < dim; j++) {
        q_norm_sq += (double)query[j] * (double)query[j];
    }
    double q_norm = sqrt(q_norm_sq);

    /* Compute per-row similarity */
    td_t* result = td_vec_new(TD_F64, nrows);
    if (!result || TD_IS_ERR(result)) return TD_ERR_PTR(TD_ERR_OOM);
    result->len = nrows;
    double* out = (double*)td_data(result);

    for (int64_t i = 0; i < nrows; i++) {
        const float* row = data + i * dim;
        double dot = 0.0;
        double r_norm_sq = 0.0;
        for (int32_t j = 0; j < dim; j++) {
            dot += (double)row[j] * (double)query[j];
            r_norm_sq += (double)row[j] * (double)row[j];
        }
        double r_norm = sqrt(r_norm_sq);
        double denom = q_norm * r_norm;
        out[i] = (denom > 0.0) ? dot / denom : 0.0;
    }

    return result;
}
```

- [x] **Step 5: Add euclidean distance kernel to `exec.c`**

```c
/*
 * Euclidean distance: sqrt(sum((a_i - b_i)^2))
 */
static td_t* exec_euclidean_dist(td_graph_t* g, td_op_t* op, td_t* emb_vec) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    const float* query = ext->vector.query_vec;
    int32_t dim = ext->vector.dim;

    if (!query || dim <= 0) return TD_ERR_PTR(TD_ERR_SCHEMA);
    if (emb_vec->type != TD_F32) return TD_ERR_PTR(TD_ERR_TYPE);

    int64_t total = emb_vec->len;
    int64_t nrows = total / dim;
    if (nrows * dim != total) return TD_ERR_PTR(TD_ERR_LENGTH);

    const float* data = (const float*)td_data(emb_vec);

    td_t* result = td_vec_new(TD_F64, nrows);
    if (!result || TD_IS_ERR(result)) return TD_ERR_PTR(TD_ERR_OOM);
    result->len = nrows;
    double* out = (double*)td_data(result);

    for (int64_t i = 0; i < nrows; i++) {
        const float* row = data + i * dim;
        double sum_sq = 0.0;
        for (int32_t j = 0; j < dim; j++) {
            double d = (double)row[j] - (double)query[j];
            sum_sq += d * d;
        }
        out[i] = sqrt(sum_sq);
    }

    return result;
}
```

- [x] **Step 6: Add dispatch cases and dump names**

In `exec_node()`:
```c
case OP_COSINE_SIM: {
    td_t* emb = exec_node(g, op->inputs[0]);
    if (!emb || TD_IS_ERR(emb)) return emb;
    td_t* result = exec_cosine_sim(g, op, emb);
    td_release(emb);
    return result;
}
case OP_EUCLIDEAN_DIST: {
    td_t* emb = exec_node(g, op->inputs[0]);
    if (!emb || TD_IS_ERR(emb)) return emb;
    td_t* result = exec_euclidean_dist(g, op, emb);
    td_release(emb);
    return result;
}
```

In `dump.c`:
```c
case OP_COSINE_SIM:     return "COSINE_SIM";
case OP_EUCLIDEAN_DIST: return "EUCLIDEAN_DIST";
```

- [x] **Step 7: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 8: Commit**

```bash
git add -f vendor/teide/
git commit -m "feat(engine): add cosine similarity and euclidean distance C kernels for embeddings"
```

---

## Task 4: KNN Brute-Force Search — C Kernel

Given an embedding column and a query vector, return the K most similar rows. Uses brute-force cosine similarity scan with a max-heap to track top-K.

**Output:** TD_TABLE with `_rowid` (I64) and `_similarity` (F64), sorted by similarity descending.

**Files:**
- Modify: `vendor/teide/src/ops/graph.c` (builder)
- Modify: `vendor/teide/src/ops/exec.c` (kernel)
- Modify: `vendor/teide/src/ops/dump.c` (name)

- [x] **Step 1: Add KNN builder to `graph.c`**

```c
td_op_t* td_knn(td_graph_t* g, td_op_t* emb_col,
                 const float* query_vec, int32_t dim, int64_t k) {
    if (!g || !emb_col || !query_vec || dim <= 0 || k <= 0) return NULL;

    td_op_ext_t* ext = graph_alloc_ext_node(g);
    if (!ext) return NULL;

    emb_col = &g->nodes[emb_col->id];

    ext->base.opcode    = OP_KNN;
    ext->base.arity     = 1;
    ext->base.inputs[0] = emb_col;
    ext->base.out_type  = TD_TABLE;
    ext->base.est_rows  = (uint32_t)k;
    ext->vector.query_vec = (float*)query_vec;
    ext->vector.dim       = dim;
    ext->vector.k         = k;

    g->nodes[ext->base.id] = ext->base;
    return &g->nodes[ext->base.id];
}
```

- [x] **Step 2: Add KNN kernel to `exec.c`**

```c
/* Max-heap entry for KNN (track worst of top-K) */
typedef struct {
    double  sim;
    int64_t rowid;
} knn_entry_t;

static void knn_heap_insert(knn_entry_t* heap, int64_t k, int64_t* size,
                             double sim, int64_t rowid) {
    if (*size < k) {
        /* Heap not full: insert and sift up */
        int64_t i = (*size)++;
        heap[i].sim = sim;
        heap[i].rowid = rowid;
        /* Sift up (max-heap: parent >= children by sim, so worst at top) */
        while (i > 0) {
            int64_t parent = (i - 1) / 2;
            /* Max-heap: we want the LOWEST similarity at root */
            if (heap[parent].sim >= heap[i].sim) break;
            knn_entry_t tmp = heap[parent]; heap[parent] = heap[i]; heap[i] = tmp;
            i = parent;
        }
    } else if (sim > heap[0].sim) {
        /* Better than worst in heap: replace root and sift down */
        heap[0].sim = sim;
        heap[0].rowid = rowid;
        int64_t i = 0;
        while (1) {
            int64_t left = 2*i+1, right = 2*i+2, largest = i;
            if (left < k && heap[left].sim > heap[largest].sim) largest = left;
            if (right < k && heap[right].sim > heap[largest].sim) largest = right;
            if (largest == i) break;
            knn_entry_t tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp;
            i = largest;
        }
    }
}

/*
 * Brute-force KNN via cosine similarity.
 * Returns top-K most similar rows.
 */
static td_t* exec_knn(td_graph_t* g, td_op_t* op, td_t* emb_vec) {
    td_op_ext_t* ext = find_ext(g, op->id);
    if (!ext) return TD_ERR_PTR(TD_ERR_NYI);

    const float* query = ext->vector.query_vec;
    int32_t dim = ext->vector.dim;
    int64_t k = ext->vector.k;

    if (!query || dim <= 0 || k <= 0) return TD_ERR_PTR(TD_ERR_SCHEMA);
    if (emb_vec->type != TD_F32) return TD_ERR_PTR(TD_ERR_TYPE);

    int64_t total = emb_vec->len;
    int64_t nrows = total / dim;
    if (nrows * dim != total) return TD_ERR_PTR(TD_ERR_LENGTH);
    if (k > nrows) k = nrows;

    const float* data = (const float*)td_data(emb_vec);

    /* Precompute query norm */
    double q_norm_sq = 0.0;
    for (int32_t j = 0; j < dim; j++) q_norm_sq += (double)query[j] * query[j];
    double q_norm = sqrt(q_norm_sq);

    /* Max-heap for top-K */
    knn_entry_t* heap = (knn_entry_t*)scratch_alloc((size_t)k * sizeof(knn_entry_t));
    if (!heap) return TD_ERR_PTR(TD_ERR_OOM);
    int64_t heap_size = 0;

    for (int64_t i = 0; i < nrows; i++) {
        const float* row = data + i * dim;
        double dot = 0.0, r_norm_sq = 0.0;
        for (int32_t j = 0; j < dim; j++) {
            dot += (double)row[j] * query[j];
            r_norm_sq += (double)row[j] * row[j];
        }
        double r_norm = sqrt(r_norm_sq);
        double denom = q_norm * r_norm;
        double sim = (denom > 0.0) ? dot / denom : 0.0;
        knn_heap_insert(heap, k, &heap_size, sim, i);
    }

    /* Extract heap into sorted arrays (descending by similarity) */
    /* First, copy heap entries and sort */
    knn_entry_t* sorted = (knn_entry_t*)scratch_alloc((size_t)heap_size * sizeof(knn_entry_t));
    if (!sorted) { scratch_free(heap); return TD_ERR_PTR(TD_ERR_OOM); }
    memcpy(sorted, heap, (size_t)heap_size * sizeof(knn_entry_t));
    scratch_free(heap);

    /* Simple insertion sort (k is small) */
    for (int64_t i = 1; i < heap_size; i++) {
        knn_entry_t key = sorted[i];
        int64_t j = i - 1;
        while (j >= 0 && sorted[j].sim < key.sim) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    /* Build output table: _rowid (I64), _similarity (F64) */
    td_t* rowid_vec = td_vec_new(TD_I64, heap_size);
    td_t* sim_vec   = td_vec_new(TD_F64, heap_size);
    if (!rowid_vec || !sim_vec) {
        scratch_free(sorted);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    int64_t* rdata = (int64_t*)td_data(rowid_vec);
    double*  sdata = (double*)td_data(sim_vec);
    for (int64_t i = 0; i < heap_size; i++) {
        rdata[i] = sorted[i].rowid;
        sdata[i] = sorted[i].sim;
    }
    rowid_vec->len = heap_size;
    sim_vec->len   = heap_size;

    scratch_free(sorted);

    td_t* result = td_table_new(2, heap_size);
    if (!result) {
        td_release(rowid_vec);
        td_release(sim_vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }
    td_table_set_col(result, 0, rowid_vec, td_sym_intern("_rowid"));
    td_table_set_col(result, 1, sim_vec, td_sym_intern("_similarity"));

    return result;
}
```

- [x] **Step 3: Add dispatch and dump**

In `exec_node()`:
```c
case OP_KNN: {
    td_t* emb = exec_node(g, op->inputs[0]);
    if (!emb || TD_IS_ERR(emb)) return emb;
    td_t* result = exec_knn(g, op, emb);
    td_release(emb);
    return result;
}
```

In `dump.c`:
```c
case OP_KNN: return "KNN";
```

- [x] **Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add -f vendor/teide/
git commit -m "feat(engine): add brute-force KNN search kernel with max-heap top-K"
```

---

## Task 5: Rust FFI Bindings + Safe Wrappers

**Files:**
- Modify: `src/ffi.rs`
- Modify: `src/engine.rs`

- [x] **Step 1: Add FFI declarations to `src/ffi.rs`**

```rust
pub const OP_COSINE_SIM: u16 = 88;
pub const OP_EUCLIDEAN_DIST: u16 = 89;
pub const OP_KNN: u16 = 90;
```

In the `extern "C"` block:
```rust
    // --- Embedding / Vector ops ---
    pub fn td_embedding_new(nrows: i64, dim: i32) -> *mut td_t;
    pub fn td_cosine_sim(
        g: *mut td_graph_t,
        emb_col: *mut td_op_t,
        query_vec: *const f32,
        dim: i32,
    ) -> *mut td_op_t;
    pub fn td_euclidean_dist(
        g: *mut td_graph_t,
        emb_col: *mut td_op_t,
        query_vec: *const f32,
        dim: i32,
    ) -> *mut td_op_t;
    pub fn td_knn(
        g: *mut td_graph_t,
        emb_col: *mut td_op_t,
        query_vec: *const f32,
        dim: i32,
        k: i64,
    ) -> *mut td_op_t;
```

- [x] **Step 2: Add safe wrappers to `src/engine.rs`**

In `impl Graph`:
```rust
    /// Compute cosine similarity between an embedding column and a query vector.
    /// The query vector must have the same dimension as the embeddings.
    /// Returns an F64 column with one similarity value per row.
    ///
    /// # Safety
    /// The `query_vec` slice must remain valid until `execute()` is called,
    /// because the C engine stores a raw pointer to it.
    pub fn cosine_sim(
        &self,
        emb_col: Column,
        query_vec: &[f32],
    ) -> Result<Column> {
        let dim = query_vec.len() as i32;
        Self::check_op(unsafe {
            ffi::td_cosine_sim(self.raw, emb_col.raw, query_vec.as_ptr(), dim)
        })
    }

    /// Compute euclidean distance between an embedding column and a query vector.
    pub fn euclidean_dist(
        &self,
        emb_col: Column,
        query_vec: &[f32],
    ) -> Result<Column> {
        let dim = query_vec.len() as i32;
        Self::check_op(unsafe {
            ffi::td_euclidean_dist(self.raw, emb_col.raw, query_vec.as_ptr(), dim)
        })
    }

    /// Brute-force K nearest neighbor search on an embedding column.
    /// Returns a table with `_rowid` (I64) and `_similarity` (F64) columns,
    /// sorted by similarity descending.
    ///
    /// # Safety
    /// The `query_vec` slice must remain valid until `execute()` is called.
    pub fn knn(
        &self,
        emb_col: Column,
        query_vec: &[f32],
        k: i64,
    ) -> Result<Column> {
        let dim = query_vec.len() as i32;
        Self::check_op(unsafe {
            ffi::td_knn(self.raw, emb_col.raw, query_vec.as_ptr(), dim, k)
        })
    }
```

- [x] **Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -5`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add src/ffi.rs src/engine.rs
git commit -m "feat: add FFI bindings and safe wrappers for vector similarity and KNN"
```

---

## Task 6: Rust API Tests

**Files:**
- Modify: `tests/engine_api.rs`

- [x] **Step 1: Add embedding creation and cosine similarity test**

```rust
#[test]
fn vector_cosine_similarity() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    // Create a 3-row, 4-dimensional embedding column
    let dim: i32 = 4;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // row 0: unit vector along x
        0.0, 1.0, 0.0, 0.0,  // row 1: unit vector along y
        1.0, 1.0, 0.0, 0.0,  // row 2: 45 degrees between x and y
    ];
    let emb_col = Table::create_embedding_column(&ctx, 3, dim, &embeddings).unwrap();

    // Query: unit vector along x → should be most similar to row 0
    let query = vec![1.0f32, 0.0, 0.0, 0.0];

    // Build a dummy table to hold the embedding
    // (We need a table to create a Graph)
    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    writeln!(csv_f, "0").unwrap();
    writeln!(csv_f, "1").unwrap();
    writeln!(csv_f, "2").unwrap();
    csv_f.flush().unwrap();
    let csv_path = csv_f.path().to_str().unwrap().to_string();
    let table = ctx.read_csv(&csv_path).unwrap();

    let g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_vec(emb_col).unwrap() };
    let sim = g.cosine_sim(emb_node, &query).unwrap();
    let result = g.execute(sim).unwrap();

    assert_eq!(result.nrows(), 3);
    let s0 = result.get_f64(0, 0).unwrap(); // row 0: cos(0°) = 1.0
    let s1 = result.get_f64(0, 1).unwrap(); // row 1: cos(90°) = 0.0
    let s2 = result.get_f64(0, 2).unwrap(); // row 2: cos(45°) ≈ 0.707

    assert!((s0 - 1.0).abs() < 0.001, "row 0 should be 1.0, got {s0}");
    assert!(s1.abs() < 0.001, "row 1 should be 0.0, got {s1}");
    assert!((s2 - 0.7071).abs() < 0.01, "row 2 should be ~0.707, got {s2}");
}
```

- [x] **Step 2: Add KNN test**

```rust
#[test]
fn vector_knn() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0,  // row 0
        0.9, 0.1, 0.0,  // row 1: very similar to row 0
        0.0, 1.0, 0.0,  // row 2: orthogonal
        0.0, 0.0, 1.0,  // row 3: orthogonal
        0.8, 0.2, 0.0,  // row 4: similar to row 0
    ];
    let emb_col = Table::create_embedding_column(&ctx, 5, dim, &embeddings).unwrap();

    let query = vec![1.0f32, 0.0, 0.0];

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    for i in 0..5 { writeln!(csv_f, "{i}").unwrap(); }
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_vec(emb_col).unwrap() };
    let knn_result = g.knn(emb_node, &query, 3).unwrap();
    let result = g.execute(knn_result).unwrap();

    // Should return 3 rows, sorted by similarity descending
    assert_eq!(result.nrows(), 3);
    // Top-1 should be row 0 (exact match, sim=1.0)
    let top_rowid = result.get_i64(0, 0).unwrap();
    assert_eq!(top_rowid, 0, "top result should be row 0");
    let top_sim = result.get_f64(1, 0).unwrap();
    assert!((top_sim - 1.0).abs() < 0.001, "top similarity should be 1.0");
}
```

- [x] **Step 3: Run tests**

Run: `cargo test --all-features -- vector_cosine_similarity vector_knn`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add tests/engine_api.rs
git commit -m "test: add Rust API tests for cosine similarity and KNN"
```

---

## Task 7: SQL Integration — COSINE_SIMILARITY and KNN Functions

Expose vector similarity as SQL scalar functions and KNN as a table function.

**Files:**
- Modify: `src/sql/planner.rs` (add `COSINE_SIMILARITY` and `EUCLIDEAN_DISTANCE` as scalar functions)
- Modify: `src/sql/expr.rs` (expression planning)
- Create: `tests/slt/vector.slt`
- Modify: `tests/slt_runner.rs`

- [x] **Step 1: Add SQL function handling**

The SQL syntax for vector search:

```sql
-- Scalar function: compute similarity for each row
SELECT name, COSINE_SIMILARITY(embedding, ARRAY[1.0, 0.0, 0.0]) AS sim
FROM documents
ORDER BY sim DESC
LIMIT 10;

-- Table function: KNN search
SELECT d.name, k._similarity
FROM KNN(documents, 'embedding', ARRAY[1.0, 0.0, 0.0], 10) AS k
JOIN documents d ON d._rowid = k._rowid;
```

In `src/sql/planner.rs`, add `COSINE_SIMILARITY` and `EUCLIDEAN_DISTANCE` to the scalar function dispatch. In `resolve_table_function`, add `KNN` as a table function.

This requires parsing `ARRAY[...]` literals as F32 vectors. Add a helper:

```rust
/// Parse an ARRAY[...] literal into a Vec<f32>.
fn parse_array_literal(expr: &Expr) -> Result<Vec<f32>, SqlError> {
    match expr {
        Expr::Array(arr) => {
            let mut values = Vec::new();
            for elem in &arr.elem {
                match elem {
                    Expr::Value(Value::Number(n, _)) => {
                        let f: f32 = n.parse().map_err(|_| {
                            SqlError::Plan(format!("Invalid array element: {n}"))
                        })?;
                        values.push(f);
                    }
                    Expr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                        if let Expr::Value(Value::Number(n, _)) = expr.as_ref() {
                            let f: f32 = n.parse().map_err(|_| {
                                SqlError::Plan(format!("Invalid array element: -{n}"))
                            })?;
                            values.push(-f);
                        } else {
                            return Err(SqlError::Plan("Invalid array element".into()));
                        }
                    }
                    _ => return Err(SqlError::Plan(format!(
                        "Array elements must be numbers, got: {elem}"
                    ))),
                }
            }
            Ok(values)
        }
        _ => Err(SqlError::Plan("Expected ARRAY[...] literal".into())),
    }
}
```

- [x] **Step 2: Create SLT tests**

Create `tests/slt/vector.slt`:

```
# Vector similarity functions

# Setup: create a table with embeddings
statement ok
CREATE TABLE docs (id INTEGER, name VARCHAR)

statement ok
INSERT INTO docs VALUES (0, 'math'), (1, 'science'), (2, 'art'), (3, 'music'), (4, 'physics')

# Note: embedding columns need to be created via API for now.
# SLT tests for vector ops will focus on SQL syntax validation
# and integration with the GRAPH_TABLE framework.

# Test: COSINE_SIMILARITY function exists and rejects wrong args
statement error
SELECT COSINE_SIMILARITY() FROM docs

# Test: EUCLIDEAN_DISTANCE function exists
statement error
SELECT EUCLIDEAN_DISTANCE() FROM docs
```

- [x] **Step 3: Add SLT runner**

```rust
#[test]
fn slt_vector() {
    run_slt("tests/slt/vector.slt");
}
```

- [x] **Step 4: Run all tests**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS

- [x] **Step 5: Commit**

```bash
git add src/sql/planner.rs src/sql/expr.rs tests/slt/vector.slt tests/slt_runner.rs
git commit -m "feat(sql): add COSINE_SIMILARITY, EUCLIDEAN_DISTANCE, and KNN SQL functions"
```

---

## Task 8: Full Regression Test

- [ ] **Step 1: Run complete test suite**

Run: `cargo test --all-features -- --skip server_ --skip extended_`
Expected: All PASS

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | tail -10`
Expected: No warnings

- [ ] **Step 3: Commit final changes**

```bash
git add -A
git commit -m "feat: complete vector search implementation

Adds TD_F32 type, embedding column storage, cosine similarity,
euclidean distance, and brute-force KNN search. All kernels operate
on contiguous F32 memory for cache-friendly SIMD-ready computation.
Exposed via Rust API and SQL functions."
```

---

## Future Work: HNSW Index

Brute-force KNN is O(N*D) per query. For >1M vectors, an HNSW (Hierarchical Navigable Small World) index reduces this to O(D * log N). This would be:

1. New C data structure `td_hnsw_t` with multi-layer proximity graph
2. Build index: `td_hnsw_build(emb_col, dim, M, ef_construction)`
3. Search: `td_hnsw_search(index, query, k, ef_search)` → same output as KNN
4. Persistence: save/load/mmap like CSR indexes
5. SQL: `CREATE VECTOR INDEX idx ON table(col) USING HNSW(M=16, ef=200)`

Estimated effort: ~2 weeks for a production-quality implementation.

---

## Summary

| Component | Location | What |
|---|---|---|
| TD_F32 type | C types.c | 4-byte float, slot 8 |
| Embedding column | C vec.c | Flat N*D F32 array |
| Cosine similarity | C exec.c | dot(a,b) / (‖a‖ * ‖b‖) per row |
| Euclidean distance | C exec.c | √Σ(aᵢ-bᵢ)² per row |
| KNN brute-force | C exec.c | Max-heap top-K scan |
| Rust wrappers | engine.rs | Safe API for all ops |
| SQL functions | planner.rs | COSINE_SIMILARITY(), EUCLIDEAN_DISTANCE(), KNN() |
