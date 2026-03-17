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

#include "hnsw.h"
#include "mem/sys.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

/* --------------------------------------------------------------------------
 * Distance function (cosine distance = 1 - cosine_similarity)
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * Random level assignment (HNSW paper, Section 3.1)
 * -------------------------------------------------------------------------- */

static _Thread_local uint32_t hnsw_rng_state = 42;

static uint32_t hnsw_rand(void) {
    /* xorshift32 — fast, deterministic, no global state collision */
    uint32_t x = hnsw_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    hnsw_rng_state = x;
    return x;
}

static int32_t hnsw_random_level(int32_t M) {
    double ml = 1.0 / log((double)M);
    double r = (double)hnsw_rand() / (double)UINT32_MAX;
    if (r < 1e-10) r = 1e-10;
    int32_t level = (int32_t)floor(-log(r) * ml);
    if (level >= HNSW_MAX_LAYERS) level = HNSW_MAX_LAYERS - 1;
    return level;
}

/* --------------------------------------------------------------------------
 * Candidate heap (min-heap by distance for beam search)
 * -------------------------------------------------------------------------- */

typedef struct {
    int64_t id;     /* global node id */
    double  dist;   /* cosine distance to query */
} hnsw_cand_t;

static void heap_sift_up(hnsw_cand_t* h, int64_t i) {
    while (i > 0) {
        int64_t p = (i - 1) / 2;
        if (h[p].dist <= h[i].dist) break;
        hnsw_cand_t tmp = h[p]; h[p] = h[i]; h[i] = tmp;
        i = p;
    }
}

static void heap_sift_down(hnsw_cand_t* h, int64_t n, int64_t i) {
    for (;;) {
        int64_t best = i;
        int64_t l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && h[l].dist < h[best].dist) best = l;
        if (r < n && h[r].dist < h[best].dist) best = r;
        if (best == i) break;
        hnsw_cand_t tmp = h[best]; h[best] = h[i]; h[i] = tmp;
        i = best;
    }
}

/* Max-heap: sift keeping largest at top */
static void maxheap_sift_up(hnsw_cand_t* h, int64_t i) {
    while (i > 0) {
        int64_t p = (i - 1) / 2;
        if (h[p].dist >= h[i].dist) break;
        hnsw_cand_t tmp = h[p]; h[p] = h[i]; h[i] = tmp;
        i = p;
    }
}

static void maxheap_sift_down(hnsw_cand_t* h, int64_t n, int64_t i) {
    for (;;) {
        int64_t best = i;
        int64_t l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && h[l].dist > h[best].dist) best = l;
        if (r < n && h[r].dist > h[best].dist) best = r;
        if (best == i) break;
        hnsw_cand_t tmp = h[best]; h[best] = h[i]; h[i] = tmp;
        i = best;
    }
}

/* --------------------------------------------------------------------------
 * Visited set (bitset)
 * -------------------------------------------------------------------------- */

typedef struct {
    uint8_t* bits;
    int64_t  n_nodes;
} hnsw_visited_t;

static hnsw_visited_t visited_new(int64_t n_nodes) {
    hnsw_visited_t v;
    v.n_nodes = n_nodes;
    size_t sz = ((size_t)n_nodes + 7) / 8;
    v.bits = (uint8_t*)td_sys_alloc(sz);
    if (v.bits) memset(v.bits, 0, sz);
    return v;
}

static void visited_free(hnsw_visited_t* v) {
    if (v->bits) td_sys_free(v->bits);
    v->bits = NULL;
}

static bool visited_test(const hnsw_visited_t* v, int64_t id) {
    if (id < 0 || id >= v->n_nodes) return true;
    return (v->bits[id / 8] >> (id % 8)) & 1;
}

static void visited_set(hnsw_visited_t* v, int64_t id) {
    if (id >= 0 && id < v->n_nodes)
        v->bits[id / 8] |= (uint8_t)(1 << (id % 8));
}

/* --------------------------------------------------------------------------
 * Layer helper: find index of global node id within a layer
 * -------------------------------------------------------------------------- */

static int64_t layer_local_idx(const td_hnsw_layer_t* layer, int64_t global_id) {
    /* For layer 0, all nodes are present: local == global */
    /* For higher layers, linear scan (small) or we could build a reverse map */
    for (int64_t i = 0; i < layer->n_nodes; i++) {
        if (layer->node_ids[i] == global_id) return i;
    }
    return -1;
}

/* Get neighbor list for a node in a layer (by global id) */
static int64_t* layer_neighbors(const td_hnsw_layer_t* layer, int64_t global_id,
                                  int64_t* out_M_max) {
    int64_t local = layer_local_idx(layer, global_id);
    if (local < 0) { *out_M_max = 0; return NULL; }
    *out_M_max = layer->M_max;
    return &layer->neighbors[local * layer->M_max];
}

/* Count actual (non -1) neighbors */
static int64_t count_neighbors(const int64_t* nb, int64_t M_max) {
    int64_t c = 0;
    for (int64_t i = 0; i < M_max; i++) {
        if (nb[i] < 0) break;
        c++;
    }
    return c;
}

/* Add a neighbor to a node's list (append if room) */
static bool add_neighbor(int64_t* nb, int64_t M_max, int64_t new_id) {
    for (int64_t i = 0; i < M_max; i++) {
        if (nb[i] < 0) { nb[i] = new_id; return true; }
        if (nb[i] == new_id) return true; /* already present */
    }
    return false; /* full */
}

/* --------------------------------------------------------------------------
 * Search layer: beam search on a single layer
 * Returns candidates sorted by distance (ascending).
 * -------------------------------------------------------------------------- */

static int64_t hnsw_search_layer(
    const td_hnsw_t* idx,
    const float* query,
    const int64_t* entry_points, int64_t n_entries,
    int32_t layer_idx,
    int32_t ef,
    hnsw_cand_t* results /* pre-allocated, ef entries */)
{
    const td_hnsw_layer_t* layer = &idx->layers[layer_idx];

    /* Visited set */
    hnsw_visited_t vis = visited_new(idx->n_nodes);
    if (!vis.bits) return 0;

    /* Min-heap: candidates to explore (sorted by distance, smallest first) */
    int64_t cand_cap = ef * 2 + n_entries + 1;
    hnsw_cand_t* candidates = (hnsw_cand_t*)td_sys_alloc((size_t)cand_cap * sizeof(hnsw_cand_t));
    if (!candidates) { visited_free(&vis); return 0; }
    int64_t cand_sz = 0;

    /* Result set (max-heap, largest distance on top — we keep at most ef) */
    int64_t res_sz = 0;

    /* Initialize with entry points */
    for (int64_t i = 0; i < n_entries; i++) {
        int64_t ep = entry_points[i];
        if (visited_test(&vis, ep)) continue;
        visited_set(&vis, ep);

        double d = hnsw_cosine_dist(query, idx->vectors + ep * idx->dim, idx->dim);

        /* Add to candidates (min-heap) */
        candidates[cand_sz] = (hnsw_cand_t){ ep, d };
        heap_sift_up(candidates, cand_sz);
        cand_sz++;

        /* Add to results (max-heap) */
        results[res_sz] = (hnsw_cand_t){ ep, d };
        maxheap_sift_up(results, res_sz);
        res_sz++;
    }

    /* Beam search */
    while (cand_sz > 0) {
        /* Pop closest candidate */
        hnsw_cand_t closest = candidates[0];
        candidates[0] = candidates[cand_sz - 1];
        cand_sz--;
        if (cand_sz > 0) heap_sift_down(candidates, cand_sz, 0);

        /* If closest is farther than the farthest result, stop */
        if (res_sz >= ef && closest.dist > results[0].dist) break;

        /* Explore neighbors */
        int64_t M_max;
        int64_t* nb = layer_neighbors(layer, closest.id, &M_max);
        if (!nb) continue;

        for (int64_t i = 0; i < M_max; i++) {
            int64_t nid = nb[i];
            if (nid < 0) break;
            if (visited_test(&vis, nid)) continue;
            visited_set(&vis, nid);

            double d = hnsw_cosine_dist(query, idx->vectors + nid * idx->dim, idx->dim);

            /* Add to result if room or closer than farthest */
            if (res_sz < ef) {
                results[res_sz] = (hnsw_cand_t){ nid, d };
                maxheap_sift_up(results, res_sz);
                res_sz++;

                /* Also add to candidates */
                if (cand_sz < cand_cap) {
                    candidates[cand_sz] = (hnsw_cand_t){ nid, d };
                    heap_sift_up(candidates, cand_sz);
                    cand_sz++;
                }
            } else if (d < results[0].dist) {
                /* Replace farthest result */
                results[0] = (hnsw_cand_t){ nid, d };
                maxheap_sift_down(results, res_sz, 0);

                /* Also add to candidates */
                if (cand_sz < cand_cap) {
                    candidates[cand_sz] = (hnsw_cand_t){ nid, d };
                    heap_sift_up(candidates, cand_sz);
                    cand_sz++;
                }
            }
        }
    }

    td_sys_free(candidates);
    visited_free(&vis);

    /* Sort results by distance ascending (insertion sort, ef is small) */
    for (int64_t i = 1; i < res_sz; i++) {
        hnsw_cand_t key = results[i];
        int64_t j = i - 1;
        while (j >= 0 && results[j].dist > key.dist) {
            results[j + 1] = results[j];
            j--;
        }
        results[j + 1] = key;
    }

    return res_sz;
}

/* --------------------------------------------------------------------------
 * Greedy closest: find single nearest neighbor in a layer (used during descent)
 * -------------------------------------------------------------------------- */

static int64_t hnsw_greedy_closest(const td_hnsw_t* idx, const float* query,
                                     int64_t ep, int32_t layer_idx) {
    const td_hnsw_layer_t* layer = &idx->layers[layer_idx];
    double best_dist = hnsw_cosine_dist(query, idx->vectors + ep * idx->dim, idx->dim);
    bool changed = true;

    while (changed) {
        changed = false;
        int64_t M_max;
        int64_t* nb = layer_neighbors(layer, ep, &M_max);
        if (!nb) break;

        for (int64_t i = 0; i < M_max; i++) {
            int64_t nid = nb[i];
            if (nid < 0) break;
            double d = hnsw_cosine_dist(query, idx->vectors + nid * idx->dim, idx->dim);
            if (d < best_dist) {
                best_dist = d;
                ep = nid;
                changed = true;
            }
        }
    }
    return ep;
}

/* --------------------------------------------------------------------------
 * Neighbor pruning: keep M closest neighbors (simple selection)
 * -------------------------------------------------------------------------- */

static void prune_neighbors(const td_hnsw_t* idx, int64_t node_id,
                              int64_t* nb, int64_t M_max, int64_t M_keep) {
    /* Count current neighbors */
    int64_t count = count_neighbors(nb, M_max);
    if (count <= M_keep) return;

    /* Compute distances from node to each neighbor */
    const float* vec = idx->vectors + node_id * idx->dim;
    hnsw_cand_t* ranked = (hnsw_cand_t*)td_sys_alloc((size_t)count * sizeof(hnsw_cand_t));
    if (!ranked) return;

    for (int64_t i = 0; i < count; i++) {
        ranked[i].id = nb[i];
        ranked[i].dist = hnsw_cosine_dist(vec, idx->vectors + nb[i] * idx->dim, idx->dim);
    }

    /* Sort by distance ascending */
    for (int64_t i = 1; i < count; i++) {
        hnsw_cand_t key = ranked[i];
        int64_t j = i - 1;
        while (j >= 0 && ranked[j].dist > key.dist) {
            ranked[j + 1] = ranked[j];
            j--;
        }
        ranked[j + 1] = key;
    }

    /* Keep M_keep closest */
    for (int64_t i = 0; i < M_max; i++) {
        nb[i] = (i < M_keep) ? ranked[i].id : -1;
    }

    td_sys_free(ranked);
}

/* --------------------------------------------------------------------------
 * HNSW Build (Algorithm 1 from HNSW paper)
 * -------------------------------------------------------------------------- */

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
    /* Copy vectors so the index owns its data — prevents use-after-free
     * if the caller frees the original buffer. */
    size_t vec_bytes = (size_t)n_nodes * (size_t)dim * sizeof(float);
    float* vec_copy = (float*)td_sys_alloc(vec_bytes);
    if (!vec_copy) { td_sys_free(idx); return NULL; }
    memcpy(vec_copy, vectors, vec_bytes);
    idx->vectors = vec_copy;
    idx->owns_data = true;

    /* Allocate node levels */
    idx->node_level = (int8_t*)td_sys_alloc((size_t)n_nodes * sizeof(int8_t));
    if (!idx->node_level) { td_hnsw_free(idx); return NULL; }

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
        size_t nb_size = (size_t)count * (size_t)layer->M_max * sizeof(int64_t);
        layer->neighbors = (int64_t*)td_sys_alloc(nb_size);
        layer->node_ids  = (int64_t*)td_sys_alloc((size_t)count * sizeof(int64_t));
        if (!layer->neighbors || !layer->node_ids) {
            td_hnsw_free(idx);
            return NULL;
        }

        /* Initialize neighbors to -1 (empty) */
        memset(layer->neighbors, 0xFF, nb_size);

        /* Fill node_ids mapping */
        int64_t j = 0;
        for (int64_t i = 0; i < n_nodes; i++) {
            if (idx->node_level[i] >= l) {
                layer->node_ids[j++] = i;
            }
        }
    }

    /* Temp buffer for search results during construction */
    int64_t max_ef = ef_construction > idx->M_max0 ? ef_construction : idx->M_max0;
    hnsw_cand_t* search_buf = (hnsw_cand_t*)td_sys_alloc((size_t)(max_ef + 1) * sizeof(hnsw_cand_t));
    if (!search_buf) { td_hnsw_free(idx); return NULL; }

    /* Insert nodes one by one */
    for (int64_t i = 1; i < n_nodes; i++) {
        const float* vec = vectors + i * dim;
        int32_t node_level = idx->node_level[i];

        /* Phase 1: Greedy descent from top layer to node_level+1 */
        int64_t ep = idx->entry_point;
        for (int32_t l = idx->n_layers - 1; l > node_level; l--) {
            ep = hnsw_greedy_closest(idx, vec, ep, l);
        }

        /* Phase 2: Insert into layers [node_level ... 0] */
        for (int32_t l = node_level; l >= 0; l--) {
            td_hnsw_layer_t* layer = &idx->layers[l];
            int64_t M_max_l = layer->M_max;
            int64_t M_keep = (l == 0) ? idx->M_max0 : M;

            /* Search for ef_construction nearest neighbors at this layer */
            int64_t n_found = hnsw_search_layer(idx, vec, &ep, 1, l,
                                                  ef_construction, search_buf);

            /* Connect node i to the M nearest found */
            int64_t local_i = layer_local_idx(layer, i);
            if (local_i < 0) continue;

            int64_t* my_nb = &layer->neighbors[local_i * M_max_l];
            int64_t n_connect = (n_found < M_keep) ? n_found : M_keep;
            for (int64_t j = 0; j < n_connect; j++) {
                my_nb[j] = search_buf[j].id;
            }

            /* Add bidirectional edges: each neighbor also gets i */
            for (int64_t j = 0; j < n_connect; j++) {
                int64_t nb_id = search_buf[j].id;
                int64_t nb_local = layer_local_idx(layer, nb_id);
                if (nb_local < 0) continue;

                int64_t* their_nb = &layer->neighbors[nb_local * M_max_l];
                if (!add_neighbor(their_nb, M_max_l, i)) {
                    /* Neighbor list full — prune to make room, then add i */
                    prune_neighbors(idx, nb_id, their_nb, M_max_l, M_keep);
                    add_neighbor(their_nb, M_max_l, i);
                }
            }

            /* Update ep for next lower layer */
            if (n_found > 0) ep = search_buf[0].id;
        }

        /* Update entry point if this node has higher level */
        if (node_level > idx->node_level[idx->entry_point]) {
            idx->entry_point = i;
        }
    }

    td_sys_free(search_buf);
    return idx;
}

/* --------------------------------------------------------------------------
 * Free
 * -------------------------------------------------------------------------- */

void td_hnsw_free(td_hnsw_t* idx) {
    if (!idx) return;
    for (int32_t l = 0; l < idx->n_layers; l++) {
        if (idx->layers[l].neighbors) td_sys_free(idx->layers[l].neighbors);
        if (idx->layers[l].node_ids) td_sys_free(idx->layers[l].node_ids);
    }
    if (idx->node_level) td_sys_free(idx->node_level);
    if (idx->owns_data && idx->vectors) td_sys_free((void*)idx->vectors);
    td_sys_free(idx);
}

/* --------------------------------------------------------------------------
 * Search: find K approximate nearest neighbors
 * -------------------------------------------------------------------------- */

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
        ep = hnsw_greedy_closest(idx, query, ep, l);
    }

    /* Phase 2: Beam search on layer 0 with ef_search width */
    hnsw_cand_t* results = (hnsw_cand_t*)td_sys_alloc(
        (size_t)ef_search * sizeof(hnsw_cand_t));
    if (!results) return 0;

    int64_t n_found = hnsw_search_layer(idx, query, &ep, 1, 0, ef_search, results);

    /* Extract top-K from results (already sorted by distance ascending) */
    int64_t result_count = (n_found < k) ? n_found : k;
    for (int64_t i = 0; i < result_count; i++) {
        out_ids[i]   = results[i].id;
        out_dists[i] = results[i].dist;
    }

    td_sys_free(results);
    return result_count;
}

/* --------------------------------------------------------------------------
 * Accessors
 * -------------------------------------------------------------------------- */

int32_t td_hnsw_dim(const td_hnsw_t* idx) {
    return idx ? idx->dim : 0;
}

/* --------------------------------------------------------------------------
 * Persistence: save/load/mmap
 *
 * File layout in directory:
 *   hnsw_header.bin  — fixed-size header
 *   hnsw_levels.bin  — node_level[n_nodes]
 *   hnsw_layer_N.bin — per-layer: neighbors + node_ids
 * -------------------------------------------------------------------------- */

typedef struct {
    int64_t n_nodes;
    int32_t dim;
    int32_t n_layers;
    int32_t M;
    int32_t M_max0;
    int32_t ef_construction;
    int32_t _pad;
    int64_t entry_point;
} hnsw_file_header_t;

td_err_t td_hnsw_save(const td_hnsw_t* idx, const char* dir) {
    if (!idx || !dir) return TD_ERR_IO;

    if (mkdir(dir, 0755) != 0 && errno != EEXIST) return TD_ERR_IO;

    char path[1024];
    FILE* f;

    /* Write header */
    snprintf(path, sizeof(path), "%s/hnsw_header.bin", dir);
    f = fopen(path, "wb");
    if (!f) return TD_ERR_IO;
    hnsw_file_header_t hdr = {
        .n_nodes = idx->n_nodes,
        .dim = idx->dim,
        .n_layers = idx->n_layers,
        .M = idx->M,
        .M_max0 = idx->M_max0,
        .ef_construction = idx->ef_construction,
        ._pad = 0,
        .entry_point = idx->entry_point
    };
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return TD_ERR_IO; }
    fclose(f);

    /* Write node levels */
    snprintf(path, sizeof(path), "%s/hnsw_levels.bin", dir);
    f = fopen(path, "wb");
    if (!f) return TD_ERR_IO;
    if (fwrite(idx->node_level, sizeof(int8_t), (size_t)idx->n_nodes, f) !=
        (size_t)idx->n_nodes) {
        fclose(f); return TD_ERR_IO;
    }
    fclose(f);

    /* Write each layer */
    for (int32_t l = 0; l < idx->n_layers; l++) {
        const td_hnsw_layer_t* layer = &idx->layers[l];
        snprintf(path, sizeof(path), "%s/hnsw_layer_%d.bin", dir, l);
        f = fopen(path, "wb");
        if (!f) return TD_ERR_IO;

        /* Write layer metadata: n_nodes, M_max */
        if (fwrite(&layer->n_nodes, sizeof(int64_t), 1, f) != 1) { fclose(f); return TD_ERR_IO; }
        if (fwrite(&layer->M_max, sizeof(int64_t), 1, f) != 1) { fclose(f); return TD_ERR_IO; }

        /* Write neighbors */
        size_t nb_count = (size_t)layer->n_nodes * (size_t)layer->M_max;
        if (nb_count > 0) {
            if (fwrite(layer->neighbors, sizeof(int64_t), nb_count, f) != nb_count) {
                fclose(f); return TD_ERR_IO;
            }
        }

        /* Write node_ids */
        if (layer->n_nodes > 0) {
            if (fwrite(layer->node_ids, sizeof(int64_t), (size_t)layer->n_nodes, f) !=
                (size_t)layer->n_nodes) {
                fclose(f); return TD_ERR_IO;
            }
        }

        fclose(f);
    }

    /* Write vectors */
    snprintf(path, sizeof(path), "%s/hnsw_vectors.bin", dir);
    f = fopen(path, "wb");
    if (!f) return TD_ERR_IO;
    size_t vec_count = (size_t)idx->n_nodes * (size_t)idx->dim;
    if (vec_count > 0) {
        if (fwrite(idx->vectors, sizeof(float), vec_count, f) != vec_count) {
            fclose(f); return TD_ERR_IO;
        }
    }
    fclose(f);

    return TD_OK;
}

static td_hnsw_t* hnsw_load_impl(const char* dir, bool use_mmap) {
    if (!dir) return NULL;
    (void)use_mmap; /* mmap optimization deferred — both paths read into memory */

    char path[1024];
    FILE* f;

    /* Read header */
    snprintf(path, sizeof(path), "%s/hnsw_header.bin", dir);
    f = fopen(path, "rb");
    if (!f) return NULL;
    hnsw_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    fclose(f);

    if (hdr.n_nodes <= 0 || hdr.dim <= 0 || hdr.n_layers <= 0 ||
        hdr.n_layers > HNSW_MAX_LAYERS ||
        hdr.M <= 0 || hdr.M_max0 <= 0 ||
        hdr.entry_point < 0 || hdr.entry_point >= hdr.n_nodes) return NULL;

    td_hnsw_t* idx = (td_hnsw_t*)td_sys_alloc(sizeof(td_hnsw_t));
    if (!idx) return NULL;
    memset(idx, 0, sizeof(td_hnsw_t));

    idx->n_nodes = hdr.n_nodes;
    idx->dim = hdr.dim;
    idx->n_layers = hdr.n_layers;
    idx->M = hdr.M;
    idx->M_max0 = hdr.M_max0;
    idx->ef_construction = hdr.ef_construction;
    idx->entry_point = hdr.entry_point;
    idx->vectors = NULL;
    idx->owns_data = true;

    /* Read node levels */
    snprintf(path, sizeof(path), "%s/hnsw_levels.bin", dir);
    f = fopen(path, "rb");
    if (!f) { td_hnsw_free(idx); return NULL; }
    idx->node_level = (int8_t*)td_sys_alloc((size_t)hdr.n_nodes * sizeof(int8_t));
    if (!idx->node_level) { fclose(f); td_hnsw_free(idx); return NULL; }
    if (fread(idx->node_level, sizeof(int8_t), (size_t)hdr.n_nodes, f) !=
        (size_t)hdr.n_nodes) {
        fclose(f); td_hnsw_free(idx); return NULL;
    }
    fclose(f);

    /* Read each layer */
    for (int32_t l = 0; l < hdr.n_layers; l++) {
        td_hnsw_layer_t* layer = &idx->layers[l];
        snprintf(path, sizeof(path), "%s/hnsw_layer_%d.bin", dir, l);
        f = fopen(path, "rb");
        if (!f) { td_hnsw_free(idx); return NULL; }

        /* Read layer metadata */
        if (fread(&layer->n_nodes, sizeof(int64_t), 1, f) != 1) { fclose(f); td_hnsw_free(idx); return NULL; }
        if (fread(&layer->M_max, sizeof(int64_t), 1, f) != 1) { fclose(f); td_hnsw_free(idx); return NULL; }

        /* Allocate and read neighbors */
        size_t nb_count = (size_t)layer->n_nodes * (size_t)layer->M_max;
        if (nb_count > 0) {
            layer->neighbors = (int64_t*)td_sys_alloc(nb_count * sizeof(int64_t));
            if (!layer->neighbors) { fclose(f); td_hnsw_free(idx); return NULL; }
            if (fread(layer->neighbors, sizeof(int64_t), nb_count, f) != nb_count) {
                fclose(f); td_hnsw_free(idx); return NULL;
            }
        }

        /* Allocate and read node_ids */
        if (layer->n_nodes > 0) {
            layer->node_ids = (int64_t*)td_sys_alloc((size_t)layer->n_nodes * sizeof(int64_t));
            if (!layer->node_ids) { fclose(f); td_hnsw_free(idx); return NULL; }
            if (fread(layer->node_ids, sizeof(int64_t), (size_t)layer->n_nodes, f) !=
                (size_t)layer->n_nodes) {
                fclose(f); td_hnsw_free(idx); return NULL;
            }
        }

        fclose(f);
    }

    /* Read vectors */
    snprintf(path, sizeof(path), "%s/hnsw_vectors.bin", dir);
    f = fopen(path, "rb");
    if (!f) { td_hnsw_free(idx); return NULL; }
    size_t vec_count = (size_t)hdr.n_nodes * (size_t)hdr.dim;
    if (vec_count > 0) {
        float* vecs = (float*)td_sys_alloc(vec_count * sizeof(float));
        if (!vecs) { fclose(f); td_hnsw_free(idx); return NULL; }
        if (fread(vecs, sizeof(float), vec_count, f) != vec_count) {
            fclose(f); td_sys_free(vecs); td_hnsw_free(idx); return NULL;
        }
        idx->vectors = vecs;
    }
    fclose(f);

    return idx;
}

td_hnsw_t* td_hnsw_load(const char* dir) {
    return hnsw_load_impl(dir, false);
}

td_hnsw_t* td_hnsw_mmap(const char* dir) {
    return hnsw_load_impl(dir, true);
}
