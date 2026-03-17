/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#ifndef TD_HNSW_H
#define TD_HNSW_H

#include <teide/td.h>

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
    bool             owns_data;       /* true if loaded from disk (owns neighbor arrays etc.) */
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

/* --- Accessors --- */
int32_t td_hnsw_dim(const td_hnsw_t* idx);

/* --- Persistence --- */
td_err_t td_hnsw_save(const td_hnsw_t* idx, const char* dir);
td_hnsw_t* td_hnsw_load(const char* dir);
td_hnsw_t* td_hnsw_mmap(const char* dir);

#endif /* TD_HNSW_H */
