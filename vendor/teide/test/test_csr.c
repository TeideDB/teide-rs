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

#include "munit.h"
#include <teide/td.h>
#include "store/csr.h"
#include "ops/fvec.h"
#include <string.h>

/* --------------------------------------------------------------------------
 * Helper: create a simple graph with edges
 *
 *   0 -> 1, 0 -> 2, 1 -> 2, 1 -> 3, 2 -> 3, 3 -> 0
 *   (6 edges, 4 nodes — a cycle)
 * -------------------------------------------------------------------------- */

static td_t* make_edge_table(void) {
    int64_t src_data[] = {0, 0, 1, 1, 2, 3};
    int64_t dst_data[] = {1, 2, 2, 3, 3, 0};
    int64_t n = 6;

    td_t* src_vec = td_vec_from_raw(TD_I64, src_data, n);
    td_t* dst_vec = td_vec_from_raw(TD_I64, dst_data, n);

    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);

    td_t* tbl = td_table_new(2);
    tbl = td_table_add_col(tbl, src_sym, src_vec);
    tbl = td_table_add_col(tbl, dst_sym, dst_vec);

    td_release(src_vec);
    td_release(dst_vec);
    return tbl;
}

/* --------------------------------------------------------------------------
 * Test: CSR build from edges
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_build(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    munit_assert_ptr_not_null(edges);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Forward CSR: check degrees */
    munit_assert(rel->fwd.n_nodes == 4);
    munit_assert(rel->fwd.n_edges == 6);
    munit_assert(td_csr_degree(&rel->fwd, 0) == 2);  /* 0->1, 0->2 */
    munit_assert(td_csr_degree(&rel->fwd, 1) == 2);  /* 1->2, 1->3 */
    munit_assert(td_csr_degree(&rel->fwd, 2) == 1);  /* 2->3 */
    munit_assert(td_csr_degree(&rel->fwd, 3) == 1);  /* 3->0 */

    /* Check neighbors of node 0 */
    int64_t cnt;
    int64_t* nbrs = td_csr_neighbors(&rel->fwd, 0, &cnt);
    munit_assert(cnt == 2);
    /* Neighbors should be 1 and 2 (order may vary) */
    munit_assert(nbrs[0] == 1 || nbrs[0] == 2);
    munit_assert(nbrs[1] == 1 || nbrs[1] == 2);
    munit_assert(nbrs[0] != nbrs[1]);

    /* Reverse CSR */
    munit_assert(rel->rev.n_nodes == 4);
    munit_assert(rel->rev.n_edges == 6);
    munit_assert(td_csr_degree(&rel->rev, 0) == 1);  /* 3->0 */
    munit_assert(td_csr_degree(&rel->rev, 1) == 1);  /* 0->1 */
    munit_assert(td_csr_degree(&rel->rev, 2) == 2);  /* 0->2, 1->2 */
    munit_assert(td_csr_degree(&rel->rev, 3) == 2);  /* 1->3, 2->3 */

    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: CSR sorted (for LFTJ)
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_sorted(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);
    munit_assert_ptr_not_null(rel);
    munit_assert_true(rel->fwd.sorted);
    munit_assert_true(rel->rev.sorted);

    /* Check sorted adjacency list for node 0 (fwd) */
    int64_t cnt;
    int64_t* nbrs = td_csr_neighbors(&rel->fwd, 0, &cnt);
    munit_assert(cnt == 2);
    munit_assert(nbrs[0] == 1);
    munit_assert(nbrs[1] == 2);

    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_EXPAND (1-hop forward)
 * -------------------------------------------------------------------------- */

static MunitResult test_expand(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Expand from nodes {0, 1} forward */
    int64_t start_data[] = {0, 1};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 2);

    td_graph_t* g = td_graph_new(NULL);
    munit_assert_ptr_not_null(g);

    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    munit_assert_ptr_not_null(expand);

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Node 0 has 2 outgoing, node 1 has 2 outgoing = 4 total */
    munit_assert(td_table_nrows(result) == 4);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_EXPAND (reverse)
 * -------------------------------------------------------------------------- */

static MunitResult test_expand_reverse(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* Expand from node 3 reverse — should find nodes pointing TO 3: {1, 2} */
    int64_t start_data[] = {3};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 1);  /* direction=1: reverse */

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 2);  /* 1->3, 2->3 */

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_VAR_EXPAND (variable-length BFS)
 * -------------------------------------------------------------------------- */

static MunitResult test_var_expand(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    int64_t start_data[] = {0};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* var_exp = td_var_expand(g, src, rel, 0, 1, 3, false);

    td_t* result = td_execute(g, var_exp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* From node 0 with depth 1..3:
     * depth 1: 0->1, 0->2 (2 results)
     * depth 2: 1->3, 2->3 (but 3 visited only once) => 1 result
     * depth 3: 3->0 (but 0 already visited) => no results
     * Total reachable: nodes 1, 2, 3 at depths 1, 1, 2 = 3 results */
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_SHORTEST_PATH
 * -------------------------------------------------------------------------- */

static MunitResult test_shortest_path(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 3);
    td_op_t* sp = td_shortest_path(g, src, dst, rel, 10);

    td_t* result = td_execute(g, sp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Shortest path 0->3: 0->1->3 (length 3 nodes) or 0->2->3 */
    int64_t nrows = td_table_nrows(result);
    munit_assert(nrows == 3);  /* 3 nodes in path */

    /* First node should be 0, last should be 3 */
    int64_t node_sym = td_sym_intern("_node", 5);
    td_t* node_col = td_table_get_col(result, node_sym);
    munit_assert_ptr_not_null(node_col);
    int64_t* nodes = (int64_t*)td_data(node_col);
    munit_assert(nodes[0] == 0);
    munit_assert(nodes[nrows - 1] == 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_SHORTEST_PATH (no path)
 * -------------------------------------------------------------------------- */

static MunitResult test_shortest_path_no_path(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Build a graph with no path from 0 to 3: only 0->1 */
    int64_t src_data[] = {0};
    int64_t dst_data[] = {1};
    td_t* s = td_vec_from_raw(TD_I64, src_data, 1);
    td_t* d = td_vec_from_raw(TD_I64, dst_data, 1);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, s);
    edges = td_table_add_col(edges, dst_sym, d);
    td_release(s); td_release(d);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src_op = td_const_i64(g, 0);
    td_op_t* dst_op = td_const_i64(g, 3);
    td_op_t* sp = td_shortest_path(g, src_op, dst_op, rel, 10);

    td_t* result = td_execute(g, sp);
    munit_assert_true(TD_IS_ERR(result));
    munit_assert_int(TD_ERR_CODE(result), ==, TD_ERR_RANGE);

    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_WCO_JOIN (triangle enumeration)
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_join_triangle(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Complete graph K3: 0<->1, 1<->2, 0<->2 */
    int64_t src_data[] = {0, 0, 1, 1, 2, 2};
    int64_t dst_data[] = {1, 2, 0, 2, 0, 1};
    int64_t n = 6;

    td_t* sv = td_vec_from_raw(TD_I64, src_data, n);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, n);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    /* Build with sorted=true (required for LFTJ) */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 3, 3, true);
    munit_assert_ptr_not_null(rel);
    munit_assert_true(rel->fwd.sorted);

    /* Triangle pattern: 3 vars, 3 rels (all same rel for K3) */
    td_rel_t* rels[3] = {rel, rel, rel};

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 3, 3);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* K3 has directed triangles */
    munit_assert(td_table_nrows(result) >= 1);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Multi-table graph (td_graph_add_table + td_scan_table)
 * -------------------------------------------------------------------------- */

static MunitResult test_multi_table(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Table 1: persons */
    int64_t ages[] = {25, 30, 35, 40};
    td_t* age_vec = td_vec_from_raw(TD_I64, ages, 4);
    int64_t age_sym = td_sym_intern("age", 3);
    td_t* persons = td_table_new(1);
    persons = td_table_add_col(persons, age_sym, age_vec);
    td_release(age_vec);

    /* Table 2: tasks */
    int64_t priorities[] = {1, 2, 3};
    td_t* prio_vec = td_vec_from_raw(TD_I64, priorities, 3);
    int64_t prio_sym = td_sym_intern("priority", 8);
    td_t* tasks = td_table_new(1);
    tasks = td_table_add_col(tasks, prio_sym, prio_vec);
    td_release(prio_vec);

    td_graph_t* g = td_graph_new(persons);  /* primary table */
    uint16_t tasks_id = td_graph_add_table(g, tasks);
    munit_assert(tasks_id == 0);

    /* Scan from persons (primary) */
    td_op_t* age_scan = td_scan(g, "age");
    munit_assert_ptr_not_null(age_scan);

    /* Scan from tasks (registered table) */
    td_op_t* prio_scan = td_scan_table(g, tasks_id, "priority");
    munit_assert_ptr_not_null(prio_scan);

    /* Execute scans */
    td_t* age_result = td_execute(g, age_scan);
    munit_assert_false(TD_IS_ERR(age_result));
    munit_assert(age_result->len == 4);
    td_release(age_result);

    td_t* prio_result = td_execute(g, prio_scan);
    munit_assert_false(TD_IS_ERR(prio_result));
    munit_assert(prio_result->len == 3);
    td_release(prio_result);

    td_graph_free(g);
    td_release(persons);
    td_release(tasks);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_WCO_JOIN (chain pattern: 3 vars, 2 rels — general LFTJ)
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_join_chain(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph: 0->1, 0->2, 1->2, 1->3, 2->3 (directed, no back edges) */
    int64_t src_data[] = {0, 0, 1, 1, 2};
    int64_t dst_data[] = {1, 2, 2, 3, 3};
    int64_t n = 5;

    td_t* sv = td_vec_from_raw(TD_I64, src_data, n);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, n);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);
    munit_assert_ptr_not_null(rel);

    /* Chain pattern: a->b->c with n_vars=3, n_rels=2
     * rels[0]: a->b, rels[1]: b->c (fallback chain pattern in LFTJ) */
    td_rel_t* rels[2] = {rel, rel};

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 2, 3);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* 2-hop paths: (0,1,2), (0,1,3), (0,2,3), (1,2,3) = 4 paths */
    int64_t nrows = td_table_nrows(result);
    munit_assert(nrows == 4);

    /* Verify we have 3 columns: _v0, _v1, _v2 */
    munit_assert(td_table_ncols(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Factorized expand (degree counting)
 * -------------------------------------------------------------------------- */

static MunitResult test_expand_factorized(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Expand nodes {0, 1, 2} forward — factorized should give degree counts */
    int64_t start_data[] = {0, 1, 2};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 3);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    munit_assert_ptr_not_null(expand);

    /* Manually set factorized flag (normally done by optimizer) */
    td_op_ext_t* ext = NULL;
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == expand->id) {
            ext = g->ext_nodes[i];
            break;
        }
    }
    munit_assert_ptr_not_null(ext);
    ext->graph.factorized = 1;

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Should have 2 columns: _src and _count */
    munit_assert(td_table_ncols(result) == 2);

    int64_t cnt_sym = td_sym_intern("_count", 6);
    td_t* cnt_col = td_table_get_col(result, cnt_sym);
    munit_assert_ptr_not_null(cnt_col);

    /* Degrees: node 0=2, node 1=2, node 2=1 */
    int64_t* counts = (int64_t*)td_data(cnt_col);
    int64_t total_deg = 0;
    for (int64_t i = 0; i < cnt_col->len; i++)
        total_deg += counts[i];
    munit_assert(total_deg == 5);  /* 2 + 2 + 1 = 5 */

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: SIP expand (source-side selection skip)
 * -------------------------------------------------------------------------- */

static MunitResult test_sip_expand(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Create source-side selection: only allow node 0 (skip nodes 1, 2, 3) */
    td_t* src_sel = td_sel_new(4);
    munit_assert_ptr_not_null(src_sel);
    munit_assert_false(TD_IS_ERR(src_sel));
    uint64_t* sel_bits = td_sel_bits(src_sel);
    TD_SEL_BIT_SET(sel_bits, 0);  /* only node 0 passes */

    /* Expand from nodes {0, 1, 2} forward — but SIP should skip 1, 2 */
    int64_t start_data[] = {0, 1, 2};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 3);

    td_graph_t* g = td_graph_new(NULL);

    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);

    /* Attach SIP selection to the expand ext node */
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == expand->id) {
            g->ext_nodes[i]->graph.sip_sel = src_sel;
            break;
        }
    }

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Only node 0 should be expanded: degree 2 → 2 output rows */
    munit_assert(td_table_nrows(result) == 2);

    td_release(result);
    td_graph_free(g);
    td_release(src_sel);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: S-Join semijoin filter in exec_join
 * -------------------------------------------------------------------------- */

static MunitResult test_sjoin_filter(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Left table: id column with many rows, most not in right side */
    int64_t n_left = 100;
    td_t* left_ids = td_vec_new(TD_I64, n_left);
    munit_assert_ptr_not_null(left_ids);
    munit_assert_false(TD_IS_ERR(left_ids));
    left_ids->len = n_left;
    int64_t* lid = (int64_t*)td_data(left_ids);
    for (int64_t i = 0; i < n_left; i++) lid[i] = i;

    td_t* left_vals = td_vec_new(TD_I64, n_left);
    left_vals->len = n_left;
    int64_t* lv = (int64_t*)td_data(left_vals);
    for (int64_t i = 0; i < n_left; i++) lv[i] = i * 10;

    int64_t id_sym = td_sym_intern("id", 2);
    int64_t val_sym = td_sym_intern("val", 3);
    td_t* left_tbl = td_table_new(2);
    left_tbl = td_table_add_col(left_tbl, id_sym, left_ids);
    left_tbl = td_table_add_col(left_tbl, val_sym, left_vals);
    td_release(left_ids); td_release(left_vals);

    /* Right table: small, only ids 5, 10, 15 */
    int64_t rids[] = {5, 10, 15};
    td_t* right_ids = td_vec_from_raw(TD_I64, rids, 3);
    int64_t rvals[] = {500, 1000, 1500};
    td_t* right_vals = td_vec_from_raw(TD_I64, rvals, 3);

    int64_t rval_sym = td_sym_intern("rval", 4);
    td_t* right_tbl = td_table_new(2);
    right_tbl = td_table_add_col(right_tbl, id_sym, right_ids);
    right_tbl = td_table_add_col(right_tbl, rval_sym, right_vals);
    td_release(right_ids); td_release(right_vals);

    /* Inner join on id — should trigger S-Join (100 > 3*2) */
    td_graph_t* g = td_graph_new(left_tbl);
    uint16_t right_id = td_graph_add_table(g, right_tbl);

    /* Build join: table-producing ops for left/right, key scans for join keys */
    td_op_t* left_tbl_op  = td_const_table(g, left_tbl);
    td_op_t* right_tbl_op = td_const_table(g, right_tbl);
    td_op_t* left_scan    = td_scan(g, "id");
    td_op_t* right_scan   = td_scan_table(g, right_id, "id");

    td_op_t* left_keys[1]  = { left_scan };
    td_op_t* right_keys[1] = { right_scan };

    td_op_t* join = td_join(g, left_tbl_op, left_keys, right_tbl_op, right_keys, 1, 0);
    munit_assert_ptr_not_null(join);

    td_t* result = td_execute(g, join);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Should match exactly 3 rows (ids 5, 10, 15) */
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(left_tbl);
    td_release(right_tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: SIP bitmap auto-construction from optimizer hint
 * -------------------------------------------------------------------------- */

static MunitResult test_sip_auto_build(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Use >64 source nodes to trigger SIP auto-build */
    int64_t start_data[100];
    for (int i = 0; i < 100; i++) start_data[i] = i % 4;
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 100);

    /* --- Graph 1: baseline without SIP hint --- */
    td_graph_t* g1 = td_graph_new(NULL);
    td_op_t* src1 = td_const_vec(g1, start_vec);
    td_op_t* expand1 = td_expand(g1, src1, rel, 0);

    td_t* baseline = td_execute(g1, expand1);
    munit_assert_false(TD_IS_ERR(baseline));
    munit_assert_int(baseline->type, ==, TD_TABLE);
    int64_t baseline_rows = td_table_nrows(baseline);
    munit_assert(baseline_rows > 0);
    td_release(baseline);
    td_graph_free(g1);

    /* --- Graph 2: with SIP hint set before execution --- */
    td_graph_t* g2 = td_graph_new(NULL);
    td_op_t* src2 = td_const_vec(g2, start_vec);
    td_op_t* expand2 = td_expand(g2, src2, rel, 0);

    /* Set SIP hint flag in pad[2] to trigger auto-build */
    td_op_ext_t* ext = NULL;
    for (uint32_t i = 0; i < g2->ext_count; i++) {
        if (g2->ext_nodes[i] && g2->ext_nodes[i]->base.id == expand2->id) {
            ext = g2->ext_nodes[i];
            break;
        }
    }
    munit_assert_ptr_not_null(ext);
    ext->base.pad[2] = 1;  /* SIP hint flag */

    td_t* result = td_execute(g2, expand2);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* All 4 nodes have degree > 0 — SIP should pass all, same count */
    munit_assert(td_table_nrows(result) == baseline_rows);

    /* Verify sip_sel was auto-built */
    munit_assert_ptr_not_null(ext->graph.sip_sel);

    td_release(result);
    td_graph_free(g2);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Factorized EXPAND → GROUP pipeline (degree count shortcut)
 * -------------------------------------------------------------------------- */

static MunitResult test_factorized_group(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    /* Graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0
     * Out-degrees: 0=2, 1=2, 2=1, 3=1 */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    int64_t start_data[] = {0, 1, 2, 3};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 4);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);

    /* Set factorized flag on expand */
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == expand->id) {
            g->ext_nodes[i]->graph.factorized = 1;
            break;
        }
    }

    /* Build GROUP BY _src with COUNT(*) on the expand output */
    td_op_t* key_src = td_scan(g, "_src");
    td_op_t* agg_cnt = td_scan(g, "_count");
    td_op_t* keys[1] = { key_src };
    td_op_t* agg_ins[1] = { agg_cnt };
    uint16_t agg_ops[1] = { OP_COUNT };
    td_op_t* group = td_group(g, keys, 1, agg_ops, agg_ins, 1);
    munit_assert_ptr_not_null(group);

    td_t* result = td_execute(g, group);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Factorized expand produces 4 rows (one per node with deg > 0).
     * GROUP BY _src COUNT(*) on factorized output should return 4 groups
     * with counts = degrees [2, 2, 1, 1]. */
    munit_assert(td_table_nrows(result) == 4);

    /* Verify total count across groups == total degree (6 edges) */
    int64_t agg_sym = td_sym_intern("_agg", 4);
    td_t* agg_col = td_table_get_col(result, agg_sym);
    munit_assert_ptr_not_null(agg_col);
    int64_t* agg_data = (int64_t*)td_data(agg_col);
    int64_t total = 0;
    for (int64_t i = 0; i < agg_col->len; i++) total += agg_data[i];
    munit_assert(total == 6);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: ASP-Join (factorized left → filtered right build)
 * -------------------------------------------------------------------------- */

static MunitResult test_asp_join(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Left table: small factorized output with _src keys {2, 5, 8} */
    int64_t lids[] = {2, 5, 8};
    int64_t lcnts[] = {10, 20, 30};
    td_t* left_src = td_vec_from_raw(TD_I64, lids, 3);
    td_t* left_cnt = td_vec_from_raw(TD_I64, lcnts, 3);

    int64_t id_sym = td_sym_intern("id", 2);
    int64_t cnt_sym = td_sym_intern("_count", 6);
    td_t* left_tbl = td_table_new(2);
    left_tbl = td_table_add_col(left_tbl, id_sym, left_src);
    left_tbl = td_table_add_col(left_tbl, cnt_sym, left_cnt);
    td_release(left_src); td_release(left_cnt);

    /* Right table: large, ids 0..99 — most don't match left */
    int64_t n_right = 100;
    td_t* right_ids = td_vec_new(TD_I64, n_right);
    munit_assert_ptr_not_null(right_ids);
    munit_assert_false(TD_IS_ERR(right_ids));
    right_ids->len = n_right;
    int64_t* rid = (int64_t*)td_data(right_ids);
    for (int64_t i = 0; i < n_right; i++) rid[i] = i;

    int64_t val_sym = td_sym_intern("val", 3);
    td_t* right_vals = td_vec_new(TD_I64, n_right);
    right_vals->len = n_right;
    int64_t* rv = (int64_t*)td_data(right_vals);
    for (int64_t i = 0; i < n_right; i++) rv[i] = i * 100;

    td_t* right_tbl = td_table_new(2);
    right_tbl = td_table_add_col(right_tbl, id_sym, right_ids);
    right_tbl = td_table_add_col(right_tbl, val_sym, right_vals);
    td_release(right_ids); td_release(right_vals);

    /* Inner join: left (3 rows) × right (100 rows) on id
     * ASP-Join triggers when right_rows > left_rows * 2 and left has _count */
    td_graph_t* g = td_graph_new(left_tbl);
    uint16_t right_id = td_graph_add_table(g, right_tbl);

    td_op_t* left_tbl_op  = td_const_table(g, left_tbl);
    td_op_t* right_tbl_op = td_const_table(g, right_tbl);
    td_op_t* left_scan    = td_scan(g, "id");
    td_op_t* right_scan   = td_scan_table(g, right_id, "id");

    td_op_t* left_keys[1]  = { left_scan };
    td_op_t* right_keys[1] = { right_scan };

    td_op_t* join = td_join(g, left_tbl_op, left_keys, right_tbl_op, right_keys, 1, 0);
    munit_assert_ptr_not_null(join);

    td_t* result = td_execute(g, join);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Should match exactly 3 rows (ids 2, 5, 8) */
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(left_tbl);
    td_release(right_tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_EXPAND direction==2 (both forward + reverse)
 * -------------------------------------------------------------------------- */

static MunitResult test_expand_both(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Expand from node 1 in BOTH directions:
     * Fwd: 1->2, 1->3 (2 edges)
     * Rev: 0->1 (1 edge — nodes pointing TO 1)
     * Total: 3 */
    int64_t start_data[] = {1};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 2);  /* direction=2: both */

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_VAR_EXPAND reverse direction
 * -------------------------------------------------------------------------- */

static MunitResult test_var_expand_reverse(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* From node 3, reverse (follow incoming edges):
     * depth 1: 3<-1, 3<-2 => nodes 1, 2
     * depth 2: 1<-0, 2<-0 => node 0 (visited once)
     * Total reachable: 3 */
    int64_t start_data[] = {3};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* var_exp = td_var_expand(g, src, rel, 1, 1, 3, false);

    td_t* result = td_execute(g, var_exp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_VAR_EXPAND direction==2 (both)
 * -------------------------------------------------------------------------- */

static MunitResult test_var_expand_both(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* From node 0, both directions, depth 1..1:
     * Fwd: 0->1, 0->2 (nodes 1, 2)
     * Rev: 3->0 (node 3)
     * Total reachable at depth 1: 3 */
    int64_t start_data[] = {0};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* var_exp = td_var_expand(g, src, rel, 2, 1, 1, false);

    td_t* result = td_execute(g, var_exp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 3);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_SHORTEST_PATH reverse direction
 * -------------------------------------------------------------------------- */

static MunitResult test_shortest_path_reverse(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* Reverse shortest path from 0 to 3: follow incoming edges from 0.
     * Rev edges of 0: 3->0 (node 3). Path: 0 -> 3 (via reverse). Length 2 nodes. */
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 3);
    td_op_t* sp = td_shortest_path(g, src, dst, rel, 10);

    /* Set direction to reverse */
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == sp->id) {
            g->ext_nodes[i]->graph.direction = 1;
            break;
        }
    }

    td_t* result = td_execute(g, sp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    /* Reverse path 0->3: 0's rev neighbor is 3. Direct path of 2 nodes. */
    munit_assert(td_table_nrows(result) == 2);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: CSR save/load/mmap roundtrip
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_save_load(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);
    munit_assert_ptr_not_null(rel);

    /* Save */
    const char* dir = "/tmp/test_csr_save";
    td_err_t err = td_rel_save(rel, dir);
    munit_assert_int(err, ==, TD_OK);

    /* Load */
    td_rel_t* loaded = td_rel_load(dir);
    munit_assert_ptr_not_null(loaded);
    munit_assert(loaded->fwd.n_nodes == 4);
    munit_assert(loaded->fwd.n_edges == 6);
    munit_assert(loaded->rev.n_nodes == 4);
    munit_assert(loaded->rev.n_edges == 6);
    munit_assert_true(loaded->fwd.sorted);

    /* Verify neighbor data matches */
    int64_t cnt_orig, cnt_loaded;
    int64_t* nbrs_orig = td_csr_neighbors(&rel->fwd, 0, &cnt_orig);
    int64_t* nbrs_loaded = td_csr_neighbors(&loaded->fwd, 0, &cnt_loaded);
    munit_assert(cnt_orig == cnt_loaded);
    for (int64_t i = 0; i < cnt_orig; i++)
        munit_assert(nbrs_orig[i] == nbrs_loaded[i]);

    td_rel_free(loaded);

    /* Mmap */
    td_rel_t* mmapped = td_rel_mmap(dir);
    munit_assert_ptr_not_null(mmapped);
    munit_assert(mmapped->fwd.n_nodes == 4);
    munit_assert(mmapped->fwd.n_edges == 6);

    td_rel_free(mmapped);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: CSR with out-of-range node IDs (should silently skip)
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_out_of_range(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Edges: 0->1, 0->99 (99 is out of range for n_nodes=4) */
    int64_t src_data[] = {0, 0};
    int64_t dst_data[] = {1, 99};
    td_t* sv = td_vec_from_raw(TD_I64, src_data, 2);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, 2);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);

    /* Fwd CSR filters by src key: both src=0 are valid, so 2 fwd edges */
    munit_assert(rel->fwd.n_edges == 2);
    munit_assert(td_csr_degree(&rel->fwd, 0) == 2);

    /* Rev CSR filters by dst key: dst=1 is valid, dst=99 is out-of-range → 1 edge */
    munit_assert(rel->rev.n_edges == 1);
    munit_assert(td_csr_degree(&rel->rev, 1) == 1);

    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: CSR with empty edge table
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_empty(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* sv = td_vec_new(TD_I64, 1);
    sv->len = 0;
    td_t* dv = td_vec_new(TD_I64, 1);
    dv->len = 0;
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_ptr_not_null(rel);
    munit_assert(rel->fwd.n_edges == 0);
    munit_assert(rel->rev.n_edges == 0);
    munit_assert(rel->fwd.n_nodes == 4);

    /* All degrees should be 0 */
    for (int i = 0; i < 4; i++)
        munit_assert(td_csr_degree(&rel->fwd, i) == 0);

    /* Expand from empty graph should return 0 rows */
    int64_t start_data[] = {0, 1};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 2);
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert(td_table_nrows(result) == 0);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: CSR type validation (non-I64 columns rejected)
 * -------------------------------------------------------------------------- */

static MunitResult test_csr_type_validation(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Build edge table with F64 columns (should be rejected) */
    double src_data[] = {0.0, 1.0};
    double dst_data[] = {1.0, 2.0};
    td_t* sv = td_vec_from_raw(TD_F64, src_data, 2);
    td_t* dv = td_vec_from_raw(TD_F64, dst_data, 2);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 3, 3, false);
    munit_assert_ptr_equal(rel, NULL);  /* Should fail due to type check */

    /* Also test negative n_nodes */
    td_t* edges2 = make_edge_table();
    td_rel_t* rel2 = td_rel_from_edges(edges2, "src", "dst", -1, 4, false);
    munit_assert_ptr_equal(rel2, NULL);

    td_release(edges2);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Graph with self-loop edges
 * -------------------------------------------------------------------------- */

static MunitResult test_self_loop(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph with self-loops: 0->0, 0->1, 1->1 */
    int64_t src_data[] = {0, 0, 1};
    int64_t dst_data[] = {0, 1, 1};
    td_t* sv = td_vec_from_raw(TD_I64, src_data, 3);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, 3);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 2, 2, false);
    munit_assert_ptr_not_null(rel);
    munit_assert(rel->fwd.n_edges == 3);

    /* Expand from node 0: self-loop 0->0 + 0->1 = 2 results */
    int64_t start_data[] = {0};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert(td_table_nrows(result) == 2);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Expand with empty source vector
 * -------------------------------------------------------------------------- */

static MunitResult test_expand_empty_src(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* start_vec = td_vec_new(TD_I64, 1);
    start_vec->len = 0;

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert(td_table_nrows(result) == 0);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: td_rel_build (FK-based CSR construction)
 * -------------------------------------------------------------------------- */

static MunitResult test_rel_build(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Table with FK column "ref" pointing to target nodes 0..2 */
    int64_t refs[] = {2, 0, 1, 2};
    td_t* ref_vec = td_vec_from_raw(TD_I64, refs, 4);
    int64_t ref_sym = td_sym_intern("ref", 3);
    td_t* tbl = td_table_new(1);
    tbl = td_table_add_col(tbl, ref_sym, ref_vec);
    td_release(ref_vec);

    td_rel_t* rel = td_rel_build(tbl, "ref", 3, true);
    munit_assert_ptr_not_null(rel);

    /* Fwd: src=row index (0..3), dst=ref value */
    munit_assert(rel->fwd.n_nodes == 4);  /* 4 rows */
    munit_assert(rel->fwd.n_edges == 4);
    munit_assert(td_csr_degree(&rel->fwd, 0) == 1);  /* row 0 -> ref 2 */
    munit_assert(td_csr_degree(&rel->fwd, 1) == 1);  /* row 1 -> ref 0 */

    /* Rev: dst=ref target (0..2), src=row index */
    munit_assert(rel->rev.n_nodes == 3);  /* 3 target nodes */
    munit_assert(rel->rev.n_edges == 4);
    munit_assert(td_csr_degree(&rel->rev, 0) == 1);  /* target 0 <- row 1 */
    munit_assert(td_csr_degree(&rel->rev, 2) == 2);  /* target 2 <- rows 0,3 */

    munit_assert_true(rel->fwd.sorted);

    /* Error cases */
    munit_assert_ptr_equal(td_rel_build(NULL, "ref", 3, false), NULL);
    munit_assert_ptr_equal(td_rel_build(tbl, "nonexistent", 3, false), NULL);
    munit_assert_ptr_equal(td_rel_build(tbl, "ref", -1, false), NULL);

    td_rel_free(rel);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: OP_SHORTEST_PATH direction==2 (bidirectional BFS)
 * -------------------------------------------------------------------------- */

static MunitResult test_shortest_path_both(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph where path 0->3 requires bidirectional traversal:
     * Fwd: 0->1  Rev: 3->2->1
     * Combined path: 0 -fwd-> 1 <-rev- 2 <-rev- 3
     * Actually: with both, from 0 we see fwd(0)={1} and rev(0)={3},
     * so the shortest bidirectional path 0->3 is just 0->3 (1 hop via rev). */
    td_t* edges = make_edge_table();  /* 0->1,0->2,1->2,1->3,2->3,3->0 */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 3);
    td_op_t* sp = td_shortest_path(g, src, dst, rel, 10);

    /* Set direction to both */
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == sp->id) {
            g->ext_nodes[i]->graph.direction = 2;
            break;
        }
    }

    td_t* result = td_execute(g, sp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* With both fwd+rev from 0: fwd neighbors={1,2}, rev neighbors={3}.
     * BFS finds 3 at depth 1 via rev. Path: [0, 3] = 2 nodes. */
    munit_assert(td_table_nrows(result) == 2);

    int64_t node_sym = td_sym_intern("_node", 5);
    td_t* node_col = td_table_get_col(result, node_sym);
    int64_t* nodes = (int64_t*)td_data(node_col);
    munit_assert(nodes[0] == 0);
    munit_assert(nodes[1] == 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: var_expand with max_depth==0 (should return empty)
 * -------------------------------------------------------------------------- */

static MunitResult test_var_expand_depth0(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    int64_t start_data[] = {0};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* var_exp = td_var_expand(g, src, rel, 0, 1, 0, false);

    td_t* result = td_execute(g, var_exp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 0);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: WCO join with unsorted rels (should return error)
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_unsorted(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Build with sorted=false */
    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);
    munit_assert_false(rel->fwd.sorted);

    td_rel_t* rels[3] = {rel, rel, rel};
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 3, 3);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_true(TD_IS_ERR(result));

    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Expand with out-of-range source node IDs at execution time
 * -------------------------------------------------------------------------- */

static MunitResult test_expand_oob_src(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* Source vector has node 99 (out of range) mixed with valid nodes */
    int64_t start_data[] = {0, 99, 1};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 3);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 0);
    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));

    /* Node 0: 2 edges, node 99: skipped, node 1: 2 edges = 4 total */
    munit_assert(td_table_nrows(result) == 4);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: Triangle exact count verification
 * -------------------------------------------------------------------------- */

static MunitResult test_triangle_exact(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Complete bidirectional graph K3: 0<->1, 1<->2, 0<->2 */
    int64_t src_data[] = {0, 0, 1, 1, 2, 2};
    int64_t dst_data[] = {1, 2, 0, 2, 0, 1};
    td_t* sv = td_vec_from_raw(TD_I64, src_data, 6);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, 6);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 3, 3, true);
    td_rel_t* rels[3] = {rel, rel, rel};

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 3, 3);
    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* K3 triangle pattern with variable ordering 0<1<2:
     * The LFTJ enumerates a where fwd[a] intersects:
     *   var1 b: neighbors of a (via fwd)
     *   var2 c: fwd[b] intersect fwd[a] (c > b due to ordering)
     * For K3 bidirectional: each (a,b,c) with a<b<c in neighbor sets.
     * With all edges bidirectional, root iterates 0,1,2.
     * a=0: fwd[0]={1,2}, b candidates. b=1: fwd[1] intersect fwd[0] = {0,2} intersect {1,2} = {2}. c=2. Emit (0,1,2).
     * b=2: fwd[2] intersect fwd[0] = {0,1} intersect {1,2} = {1}. c=1. Emit (0,2,1).
     * a=1: fwd[1]={0,2}. b=0: fwd[0] intersect fwd[1] = {1,2} intersect {0,2} = {2}. Emit (1,0,2).
     * b=2: fwd[2] intersect fwd[1] = {0,1} intersect {0,2} = {0}. Emit (1,2,0).
     * a=2: fwd[2]={0,1}. b=0: fwd[0] intersect fwd[2] = {1,2} intersect {0,1} = {1}. Emit (2,0,1).
     * b=1: fwd[1] intersect fwd[2] = {0,2} intersect {0,1} = {0}. Emit (2,1,0).
     * Total: 6 directed triangles */
    munit_assert(td_table_nrows(result) == 6);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: LFTJ chain with 4 variables (a→b→c→d)
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_chain4(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph: 0->1, 0->2, 1->2, 1->3, 2->3 */
    int64_t src_data[] = {0, 0, 1, 1, 2};
    int64_t dst_data[] = {1, 2, 2, 3, 3};
    td_t* sv = td_vec_from_raw(TD_I64, src_data, 5);
    td_t* dv = td_vec_from_raw(TD_I64, dst_data, 5);
    int64_t src_sym = td_sym_intern("src", 3);
    int64_t dst_sym = td_sym_intern("dst", 3);
    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, src_sym, sv);
    edges = td_table_add_col(edges, dst_sym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);

    /* Chain pattern: a→b→c→d with n_vars=4, n_rels=3 */
    td_rel_t* rels[3] = {rel, rel, rel};
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 3, 4);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* 3-hop paths: (0,1,2,3), (0,1,3,?), (0,2,3,?)
     * 0→1→2→3: valid. 0→1→3→?: no edges from 3. 0→2→3→?: no edges from 3.
     * 1→2→3→?: no edges from 3.
     * Only valid 3-hop: (0,1,2,3) = 1 path */
    munit_assert(td_table_nrows(result) == 1);

    /* Verify columns */
    munit_assert(td_table_ncols(result) == 4);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: shortest_path src == dst (zero-hop)
 * -------------------------------------------------------------------------- */

static MunitResult test_shortest_path_self(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src_op = td_const_i64(g, 2);
    td_op_t* dst_op = td_const_i64(g, 2);
    td_op_t* sp = td_shortest_path(g, src_op, dst_op, rel, 10);
    munit_assert_ptr_not_null(sp);

    td_t* result = td_execute(g, sp);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* src == dst: should return single row with node=2, depth=0 */
    munit_assert(td_table_nrows(result) == 1);
    td_t* node_col = td_table_get_col(result, td_sym_intern("_node", 5));
    td_t* depth_col = td_table_get_col(result, td_sym_intern("_depth", 6));
    munit_assert_ptr_not_null(node_col);
    munit_assert_ptr_not_null(depth_col);
    munit_assert(((int64_t*)td_data(node_col))[0] == 2);
    munit_assert(((int64_t*)td_data(depth_col))[0] == 0);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: save/load verifies reverse CSR data
 * -------------------------------------------------------------------------- */

static MunitResult test_save_load_rev(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);
    munit_assert_ptr_not_null(rel);

    const char* dir = "/tmp/test_csr_rev";
    td_err_t err = td_rel_save(rel, dir);
    munit_assert_int(err, ==, TD_OK);

    td_rel_t* loaded = td_rel_load(dir);
    munit_assert_ptr_not_null(loaded);

    /* Verify reverse CSR matches original */
    munit_assert(loaded->rev.n_nodes == rel->rev.n_nodes);
    munit_assert(loaded->rev.n_edges == rel->rev.n_edges);
    for (int64_t i = 0; i < 4; i++) {
        munit_assert(td_csr_degree(&loaded->rev, i) == td_csr_degree(&rel->rev, i));
    }
    /* Verify sorted flags persisted */
    munit_assert(loaded->fwd.sorted == true);
    munit_assert(loaded->rev.sorted == true);

    td_rel_free(loaded);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: 4-clique WCO join (LFTJ with 4 vars, 6 rels)
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_4clique(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Complete graph K5: 5 nodes, all pairs connected (10 directed edges).
     * 4-cliques in K5: C(5,4) = 5 */
    int64_t src_d[20], dst_d[20];
    int k = 0;
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++) {
            src_d[k] = i; dst_d[k] = j; k++;
            src_d[k] = j; dst_d[k] = i; k++;
        }
    td_t* sv = td_vec_from_raw(TD_I64, src_d, k);
    td_t* dv = td_vec_from_raw(TD_I64, dst_d, k);
    int64_t ssym = td_sym_intern("src", 3);
    int64_t dsym = td_sym_intern("dst", 3);
    td_t* etbl = td_table_new(2);
    etbl = td_table_add_col(etbl, ssym, sv);
    etbl = td_table_add_col(etbl, dsym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(etbl, "src", "dst", 5, 5, true);
    munit_assert_ptr_not_null(rel);

    /* 4-clique: 6 rels = all pairs among 4 variables */
    td_rel_t* rels[6] = {rel, rel, rel, rel, rel, rel};
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 6, 4);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* K5 bidirectional: each node's fwd neighbors are all 4 others.
     * LFTJ enumerates all ordered 4-tuples of distinct nodes:
     * 5 * 4 * 3 * 2 = 120 permutations (= 5 cliques * 4! orderings). */
    munit_assert(td_table_nrows(result) == 120);
    munit_assert(td_table_ncols(result) == 4);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(etbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: fvec module — ftable create/materialize
 * -------------------------------------------------------------------------- */

static MunitResult test_fvec_materialize(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Create ftable with 2 columns: one flat, one unflat */
    td_ftable_t* ft = td_ftable_new(2);
    munit_assert_ptr_not_null(ft);
    munit_assert_int(ft->n_cols, ==, 2);

    /* Column 0: flat — single value replicated 5 times */
    int64_t vals0[] = {42, 99, 7};
    td_t* v0 = td_vec_from_raw(TD_I64, vals0, 3);
    ft->columns[0].vec = v0;
    ft->columns[0].cur_idx = 1;           /* value at index 1 = 99 */
    ft->columns[0].cardinality = 5;

    /* Column 1: unflat — full vector of 5 elements */
    int64_t vals1[] = {10, 20, 30, 40, 50};
    td_t* v1 = td_vec_from_raw(TD_I64, vals1, 5);
    ft->columns[1].vec = v1;
    ft->columns[1].cur_idx = -1;          /* unflat */
    ft->columns[1].cardinality = 5;

    /* Materialize */
    td_t* result = td_ftable_materialize(ft);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 5);
    munit_assert(td_table_ncols(result) == 2);

    /* Verify flat column: all 5 values should be 99 */
    td_t* c0 = td_table_get_col(result, td_sym_intern("_c0", 3));
    munit_assert_ptr_not_null(c0);
    int64_t* d0 = (int64_t*)td_data(c0);
    for (int64_t i = 0; i < 5; i++)
        munit_assert(d0[i] == 99);

    /* Verify unflat column: exact copy */
    td_t* c1 = td_table_get_col(result, td_sym_intern("_c1", 3));
    munit_assert_ptr_not_null(c1);
    int64_t* d1 = (int64_t*)td_data(c1);
    for (int64_t i = 0; i < 5; i++)
        munit_assert(d1[i] == vals1[i]);

    td_release(result);
    td_ftable_free(ft);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: fvec — empty ftable materialization
 * -------------------------------------------------------------------------- */

static MunitResult test_fvec_empty(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* 0 columns → error */
    td_t* err = td_ftable_materialize(NULL);
    munit_assert(TD_IS_ERR(err));

    /* ftable with columns but no vec set → produces table with 0 real cols */
    td_ftable_t* ft = td_ftable_new(1);
    munit_assert_ptr_not_null(ft);
    ft->columns[0].vec = NULL;
    td_t* result = td_ftable_materialize(ft);
    /* With no actual columns added, result should still be a valid table */
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    td_release(result);
    td_ftable_free(ft);

    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: fvec — ftable free with semijoin
 * -------------------------------------------------------------------------- */

static MunitResult test_fvec_semijoin(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_ftable_t* ft = td_ftable_new(1);
    munit_assert_ptr_not_null(ft);

    /* Set a semijoin bitmap */
    td_t* sel = td_sel_new(64);
    munit_assert_false(TD_IS_ERR(sel));
    ft->semijoin = sel;

    /* Set a column */
    int64_t vals[] = {1, 2, 3};
    ft->columns[0].vec = td_vec_from_raw(TD_I64, vals, 3);
    ft->columns[0].cur_idx = -1;
    ft->columns[0].cardinality = 3;

    td_t* result = td_ftable_materialize(ft);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert(td_table_nrows(result) == 3);
    td_release(result);

    /* Free should clean up semijoin bitmap too */
    td_ftable_free(ft);

    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: factorized expand with direction=1 (reverse)
 * -------------------------------------------------------------------------- */

static MunitResult test_factorized_reverse(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    /* Expand nodes {0, 1, 2} reverse — degree counts from rev CSR */
    int64_t start_data[] = {0, 1, 2};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 3);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    td_op_t* expand = td_expand(g, src, rel, 1);  /* direction=1 rev */
    munit_assert_ptr_not_null(expand);

    /* Set factorized flag */
    td_op_ext_t* ext = NULL;
    for (uint32_t i = 0; i < g->ext_count; i++) {
        if (g->ext_nodes[i] && g->ext_nodes[i]->base.id == expand->id) {
            ext = g->ext_nodes[i]; break;
        }
    }
    munit_assert_ptr_not_null(ext);
    ext->graph.factorized = 1;

    td_t* result = td_execute(g, expand);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);

    /* Reverse degrees: node 0 has 1 in-edge (3->0), node 1 has 1 (0->1),
     * node 2 has 2 in-edges (0->2, 1->2) */
    munit_assert(td_table_ncols(result) == 2);
    int64_t cnt_sym = td_sym_intern("_count", 6);
    td_t* cnt_col = td_table_get_col(result, cnt_sym);
    munit_assert_ptr_not_null(cnt_col);
    int64_t* counts = (int64_t*)td_data(cnt_col);
    int64_t nrows = td_table_nrows(result);
    /* All 3 nodes should have degree > 0, so all 3 appear */
    munit_assert(nrows == 3);
    /* Sum of degrees: 1 + 1 + 2 = 4 */
    int64_t sum = 0;
    for (int64_t i = 0; i < nrows; i++) sum += counts[i];
    munit_assert(sum == 4);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: LFTJ with disconnected/empty result
 * -------------------------------------------------------------------------- */

static MunitResult test_wco_empty_result(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph with no triangles: 0->1, 2->3 (two disconnected edges) */
    int64_t src_d[] = {0, 2};
    int64_t dst_d[] = {1, 3};
    td_t* sv = td_vec_from_raw(TD_I64, src_d, 2);
    td_t* dv = td_vec_from_raw(TD_I64, dst_d, 2);
    int64_t ssym = td_sym_intern("src", 3);
    int64_t dsym = td_sym_intern("dst", 3);
    td_t* etbl = td_table_new(2);
    etbl = td_table_add_col(etbl, ssym, sv);
    etbl = td_table_add_col(etbl, dsym, dv);
    td_release(sv); td_release(dv);

    td_rel_t* rel = td_rel_from_edges(etbl, "src", "dst", 4, 4, true);
    munit_assert_ptr_not_null(rel);

    /* Triangle pattern (n_vars=3, n_rels=3) — no triangles exist */
    td_rel_t* rels[3] = {rel, rel, rel};
    td_graph_t* g = td_graph_new(NULL);
    td_op_t* wco = td_wco_join(g, rels, 3, 3);
    munit_assert_ptr_not_null(wco);

    td_t* result = td_execute(g, wco);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    munit_assert(td_table_nrows(result) == 0);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(etbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: var_expand with min_depth > max_depth (should return empty)
 * -------------------------------------------------------------------------- */

static MunitResult test_var_expand_bad_range(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    int64_t start_data[] = {0};
    td_t* start_vec = td_vec_from_raw(TD_I64, start_data, 1);

    td_graph_t* g = td_graph_new(NULL);
    td_op_t* src = td_const_vec(g, start_vec);
    /* min_depth=5, max_depth=2 → no results possible */
    td_op_t* ve = td_var_expand(g, src, rel, 0, 5, 2, false);
    munit_assert_ptr_not_null(ve);

    td_t* result = td_execute(g, ve);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_TABLE);
    /* min > max means no depth qualifies → 0 rows */
    munit_assert(td_table_nrows(result) == 0);

    td_release(result);
    td_graph_free(g);
    td_release(start_vec);
    td_rel_free(rel);
    td_release(edges);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * test_degree_cent: in/out/total degree from CSR offsets
 * -------------------------------------------------------------------------- */
static MunitResult test_degree_cent(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

    /* Node 0: out=2, in=1, total=3 */
    munit_assert_int(out_d[0], ==, 2);
    munit_assert_int(in_d[0],  ==, 1);
    munit_assert_int(deg_d[0], ==, 3);
    /* Node 1: out=2, in=1, total=3 */
    munit_assert_int(out_d[1], ==, 2);
    munit_assert_int(in_d[1],  ==, 1);
    munit_assert_int(deg_d[1], ==, 3);
    /* Node 2: out=1, in=2, total=3 */
    munit_assert_int(out_d[2], ==, 1);
    munit_assert_int(in_d[2],  ==, 2);
    munit_assert_int(deg_d[2], ==, 3);
    /* Node 3: out=1, in=2, total=3 */
    munit_assert_int(out_d[3], ==, 1);
    munit_assert_int(in_d[3],  ==, 2);
    munit_assert_int(deg_d[3], ==, 3);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

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
    (void)td_sym_init();

    td_t* edges = make_dag_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

    /* Order values must be a valid permutation of [0..3] */
    uint8_t seen[4] = {0};
    for (int i = 0; i < 4; i++) {
        munit_assert_true(ord[i] >= 0 && ord[i] < 4);
        munit_assert_false(seen[ord[i]]);
        seen[ord[i]] = 1;
    }
    /* Node 0 must come before 1,2; node 3 must come after 1,2 */
    munit_assert_true(ord[0] < ord[1]);
    munit_assert_true(ord[0] < ord[2]);
    munit_assert_true(ord[3] > ord[1]);
    munit_assert_true(ord[3] > ord[2]);

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
    (void)td_sym_init();

    /* Cyclic graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0 */
    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

static MunitResult test_dfs(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

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
    munit_assert_int(nodes[0], ==, 0);
    munit_assert_int(depths[0], ==, 0);
    munit_assert_int(parents[0], ==, -1);

    /* All 4 nodes must be distinct and valid */
    uint8_t node_seen[4] = {0};
    for (int64_t i = 0; i < 4; i++) {
        munit_assert_true(nodes[i] >= 0 && nodes[i] < 4);
        munit_assert_false(node_seen[nodes[i]]);
        node_seen[nodes[i]] = 1;
        munit_assert_int(depths[i], >=, 0);
        /* Non-root nodes must have a valid parent */
        if (i > 0) {
            munit_assert_true(parents[i] >= 0 && parents[i] < 4);
        }
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
    (void)td_sym_init();

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

    /* Verify correct nodes and depths */
    int64_t node_sym   = td_sym_intern("_node", 5);
    int64_t depth_sym  = td_sym_intern("_depth", 6);
    td_t* node_col  = td_table_get_col(result, node_sym);
    td_t* depth_col = td_table_get_col(result, depth_sym);
    int64_t* ns = (int64_t*)td_data(node_col);
    int64_t* ds = (int64_t*)td_data(depth_col);
    uint8_t found[4] = {0};
    for (int64_t i = 0; i < 3; i++) {
        munit_assert_true(ds[i] <= 1);
        munit_assert_true(ns[i] >= 0 && ns[i] < 4);
        found[ns[i]] = 1;
    }
    /* Node 3 should not be reached at depth 1 */
    munit_assert_true(found[0] && found[1] && found[2]);
    munit_assert_false(found[3]);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(src_tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * test_cluster_coeff: clustering coefficient via triangle counting
 * -------------------------------------------------------------------------- */
static MunitResult test_cluster_coeff(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* Graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0
     * Treat as undirected: node 0 neighbors={1,2,3}, node 1 neighbors={0,2,3}, etc.
     * All 4 nodes form a near-clique. */
    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* cc = td_cluster_coeff(g, rel);
    munit_assert_ptr_not_null(cc);

    td_t* result = td_execute(g, cc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* Verify exact clustering coefficients.
     * Graph edges: 0->1, 0->2, 1->2, 1->3, 2->3, 3->0 (directed).
     * Undirected neighbors: 0={1,2,3}, 1={0,2,3}, 2={0,1,3}, 3={0,1,2}.
     * Formula: directed_fwd_edges_between_neighbors / (deg * (deg-1)).
     * Node 0 (deg=3, pairs=6): fwd edges among {1,2,3}: 1->2,1->3,2->3 = 3; coeff=3/6=0.5
     * Node 1 (deg=3, pairs=6): fwd edges among {0,2,3}: 0->2,2->3,3->0 = 3; coeff=3/6=0.5
     * Node 2 (deg=3, pairs=6): fwd edges among {0,1,3}: 0->1,1->3,3->0 = 3; coeff=3/6=0.5
     * Node 3 (deg=3, pairs=6): fwd edges among {0,1,2}: 0->1,0->2,1->2 = 3; coeff=3/6=0.5 */
    int64_t coeff_sym = td_sym_intern("_coefficient", 12);
    td_t* coeff_col = td_table_get_col(result, coeff_sym);
    munit_assert_ptr_not_null(coeff_col);
    double* coeffs = (double*)td_data(coeff_col);
    for (int i = 0; i < 4; i++) {
        munit_assert_double(coeffs[i], >=, 0.49);
        munit_assert_double(coeffs[i], <=, 0.51);
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

/* --------------------------------------------------------------------------
 * Test: Random walk
 * -------------------------------------------------------------------------- */

static MunitResult test_random_walk(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

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
    munit_assert_int(nodes[0], ==, 0);

    /* All nodes should be valid (0..3) and consecutive pairs must be edges.
     * Graph edges: 0->1, 0->2, 1->2, 1->3, 2->3, 3->0 */
    int edges_src[] = {0, 0, 1, 1, 2, 3};
    int edges_dst[] = {1, 2, 2, 3, 3, 0};
    int n_edges = 6;
    for (int i = 0; i < 11; i++) {
        munit_assert(nodes[i] >= 0);
        munit_assert(nodes[i] < 4);
        if (i > 0) {
            bool valid_edge = false;
            for (int e = 0; e < n_edges; e++) {
                if (edges_src[e] == nodes[i-1] && edges_dst[e] == nodes[i]) {
                    valid_edge = true;
                    break;
                }
            }
            munit_assert_true(valid_edge);
        }
    }

    /* Step column should be 0..10 */
    int64_t step_sym = td_sym_intern("_step", 5);
    td_t* step_col = td_table_get_col(result, step_sym);
    int64_t* steps = (int64_t*)td_data(step_col);
    for (int i = 0; i < 11; i++) {
        munit_assert_int(steps[i], ==, i);
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

/* --------------------------------------------------------------------------
 * Test: A* shortest path
 * -------------------------------------------------------------------------- */

/* Weighted graph with lat/lon node properties:
 * 5 nodes, 6 edges:
 *   0->1 (w=1.0), 0->2 (w=4.0), 1->3 (w=2.0), 2->3 (w=1.0), 3->4 (w=3.0), 1->4 (w=10.0)
 * Node coordinates: 0=(0,0), 1=(1,0), 2=(0,2), 3=(2,1), 4=(3,0) */
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
    double lat_arr[] = {0.0, 1.0, 0.0, 2.0, 3.0};
    double lon_arr[] = {0.0, 0.0, 2.0, 1.0, 0.0};
    td_t* nv = td_vec_from_raw(TD_I64, (int64_t[]){0,1,2,3,4}, 5);
    td_t* latv = td_vec_new(TD_F64, 5);
    memcpy(td_data(latv), lat_arr, sizeof(lat_arr));
    latv->len = 5;
    td_t* lonv = td_vec_new(TD_F64, 5);
    memcpy(td_data(lonv), lon_arr, sizeof(lon_arr));
    lonv->len = 5;

    td_t* np = td_table_new(3);
    np = td_table_add_col(np, td_sym_intern("_node", 5), nv); td_release(nv);
    np = td_table_add_col(np, td_sym_intern("lat", 3), latv); td_release(latv);
    np = td_table_add_col(np, td_sym_intern("lon", 3), lonv); td_release(lonv);
    *out_node_props = np;
}

static MunitResult test_astar(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges; td_rel_t* rel; td_t* node_props;
    make_astar_graph(&edges, &rel, &node_props);

    td_graph_t* g = td_graph_new(edges);
    td_op_t* src = td_const_i64(g, 0);
    td_op_t* dst = td_const_i64(g, 4);
    td_op_t* as = td_astar(g, src, dst, rel, "weight", "lat", "lon", node_props, 255);
    munit_assert_ptr_not_null(as);

    td_t* result = td_execute(g, as);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));

    /* Should find path to node 4 with dist=6.0 (0->1->3->4: 1+2+3) */
    int64_t nrows = td_table_nrows(result);
    munit_assert_int(nrows, >, 0);

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

static MunitResult test_k_shortest(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

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

    /* Should find at least 2 paths: 0->1->3->4 (6.0) and 0->2->3->4 (8.0) */
    int64_t nrows = td_table_nrows(result);
    munit_assert_int(nrows, >=, 2);

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

    /* Verify path_id 0 exists and ends at dst with dist ~6.0 */
    int64_t node_sym = td_sym_intern("_node", 5);
    td_t* node_col = td_table_get_col(result, node_sym);
    int64_t* nodes_arr = (int64_t*)td_data(node_col);
    int64_t* pids = (int64_t*)td_data(pid_col);

    /* Find last row of path 0 */
    int64_t last_p0 = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (pids[i] == 0) last_p0 = i;
    }
    munit_assert_int(nodes_arr[last_p0], ==, 4);
    munit_assert_double(dists[last_p0], >=, 5.99);
    munit_assert_double(dists[last_p0], <=, 6.01);

    td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(node_props);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

/* --------------------------------------------------------------------------
 * Test: betweenness centrality (Brandes, exact)
 * -------------------------------------------------------------------------- */

static MunitResult test_betweenness(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    /* DAG: 0->1, 0->2, 1->3, 2->3
     * Nodes 1 and 2 are bridges -- should have nonzero betweenness. */
    td_t* edges = make_dag_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

    /* Undirected K_{2,2} (0-1, 0-2, 1-3, 2-3): each node is the sole
     * intermediary for exactly one pair (e.g., node 0 mediates {1,2} via
     * 1-0-2, but sigma_{1,2}=2 since 1-3-2 also exists), giving C_B = 0.5.
     * By symmetry all four nodes have equal betweenness. */
    for (int i = 0; i < 4; i++) {
        munit_assert_double(cents[i] - 0.5, >=, -1e-9);
        munit_assert_double(cents[i] - 0.5, <=, 1e-9);
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

/* --------------------------------------------------------------------------
 * Test: betweenness centrality (Brandes, sampled)
 * -------------------------------------------------------------------------- */

static MunitResult test_betweenness_sampled(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* bc = td_betweenness(g, rel, 2);  /* sample 2 sources */
    td_t* result = td_execute(g, bc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 4);

    /* Verify centrality values are non-negative */
    int64_t cent_sym = td_sym_intern("_centrality", 11);
    td_t* cent_col = td_table_get_col(result, cent_sym);
    munit_assert_ptr_not_null(cent_col);
    double* cents = (double*)td_data(cent_col);
    for (int i = 0; i < 4; i++) {
        munit_assert_double(cents[i], >=, 0.0);
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

static MunitResult test_closeness(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();  /* 4-node cycle */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

static MunitResult test_closeness_sampled(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges = make_edge_table();  /* 4-node cycle */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, false);

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* cc = td_closeness(g, rel, 2);  /* sample 2 sources */
    td_t* result = td_execute(g, cc);
    munit_assert_ptr_not_null(result);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(td_table_nrows(result), ==, 2);

    /* Verify centrality values are positive */
    int64_t cent_sym = td_sym_intern("_centrality", 11);
    td_t* cent_col = td_table_get_col(result, cent_sym);
    munit_assert_ptr_not_null(cent_col);
    double* cents = (double*)td_data(cent_col);
    for (int i = 0; i < 2; i++) {
        munit_assert_double(cents[i], >, 0.0);
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

static MunitResult test_mst(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    (void)td_sym_init();

    td_t* edges; td_rel_t* rel; td_t* node_props;
    make_astar_graph(&edges, &rel, &node_props);
    /* 5 nodes, 6 edges: 0->1(1), 0->2(4), 1->3(2), 2->3(1), 3->4(3), 1->4(10)
     * MST (undirected) should pick: 0-1(1), 2-3(1), 1-3(2), 3-4(3) = total 7
     * (skip 0-2(4) and 1-4(10)) */

    td_t* tbl = td_table_new(1);
    td_t* dummy_vec = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    tbl = td_table_add_col(tbl, td_sym_intern("_dummy", 6), dummy_vec);
    td_release(dummy_vec);
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

/* --------------------------------------------------------------------------
 * Suite definition
 * -------------------------------------------------------------------------- */

static MunitTest csr_tests[] = {
    { "/build",            test_csr_build,            NULL, NULL, 0, NULL },
    { "/sorted",           test_csr_sorted,           NULL, NULL, 0, NULL },
    { "/expand",           test_expand,               NULL, NULL, 0, NULL },
    { "/expand_reverse",   test_expand_reverse,       NULL, NULL, 0, NULL },
    { "/var_expand",       test_var_expand,            NULL, NULL, 0, NULL },
    { "/shortest_path",    test_shortest_path,         NULL, NULL, 0, NULL },
    { "/shortest_path_no", test_shortest_path_no_path, NULL, NULL, 0, NULL },
    { "/wco_join",         test_wco_join_triangle,     NULL, NULL, 0, NULL },
    { "/wco_chain",        test_wco_join_chain,        NULL, NULL, 0, NULL },
    { "/factorized",       test_expand_factorized,     NULL, NULL, 0, NULL },
    { "/sip_expand",       test_sip_expand,            NULL, NULL, 0, NULL },
    { "/sip_auto",         test_sip_auto_build,        NULL, NULL, 0, NULL },
    { "/sjoin",            test_sjoin_filter,           NULL, NULL, 0, NULL },
    { "/asp_join",         test_asp_join,              NULL, NULL, 0, NULL },
    { "/fact_group",       test_factorized_group,      NULL, NULL, 0, NULL },
    { "/multi_table",      test_multi_table,           NULL, NULL, 0, NULL },
    { "/expand_both",      test_expand_both,           NULL, NULL, 0, NULL },
    { "/var_rev",          test_var_expand_reverse,    NULL, NULL, 0, NULL },
    { "/var_both",         test_var_expand_both,       NULL, NULL, 0, NULL },
    { "/sp_reverse",       test_shortest_path_reverse, NULL, NULL, 0, NULL },
    { "/save_load",        test_csr_save_load,         NULL, NULL, 0, NULL },
    { "/out_of_range",     test_csr_out_of_range,      NULL, NULL, 0, NULL },
    { "/empty",            test_csr_empty,             NULL, NULL, 0, NULL },
    { "/type_check",       test_csr_type_validation,   NULL, NULL, 0, NULL },
    { "/self_loop",        test_self_loop,             NULL, NULL, 0, NULL },
    { "/empty_src",        test_expand_empty_src,      NULL, NULL, 0, NULL },
    { "/rel_build",        test_rel_build,             NULL, NULL, 0, NULL },
    { "/sp_both",          test_shortest_path_both,    NULL, NULL, 0, NULL },
    { "/var_depth0",       test_var_expand_depth0,     NULL, NULL, 0, NULL },
    { "/wco_unsorted",     test_wco_unsorted,          NULL, NULL, 0, NULL },
    { "/expand_oob_src",   test_expand_oob_src,        NULL, NULL, 0, NULL },
    { "/triangle_exact",   test_triangle_exact,        NULL, NULL, 0, NULL },
    { "/wco_chain4",       test_wco_chain4,            NULL, NULL, 0, NULL },
    { "/sp_self",          test_shortest_path_self,    NULL, NULL, 0, NULL },
    { "/save_load_rev",    test_save_load_rev,         NULL, NULL, 0, NULL },
    { "/wco_4clique",      test_wco_4clique,           NULL, NULL, 0, NULL },
    { "/fvec_mat",         test_fvec_materialize,      NULL, NULL, 0, NULL },
    { "/fvec_empty",       test_fvec_empty,            NULL, NULL, 0, NULL },
    { "/fvec_semijoin",    test_fvec_semijoin,         NULL, NULL, 0, NULL },
    { "/fact_rev",         test_factorized_reverse,    NULL, NULL, 0, NULL },
    { "/wco_empty",        test_wco_empty_result,      NULL, NULL, 0, NULL },
    { "/var_bad_range",    test_var_expand_bad_range,   NULL, NULL, 0, NULL },
    { "/degree_cent",     test_degree_cent,            NULL, NULL, 0, NULL },
    { "/topsort",         test_topsort,                NULL, NULL, 0, NULL },
    { "/topsort_cycle",   test_topsort_cycle,          NULL, NULL, 0, NULL },
    { "/dfs",             test_dfs,                    NULL, NULL, 0, NULL },
    { "/dfs_max_depth",   test_dfs_max_depth,          NULL, NULL, 0, NULL },
    { "/cluster_coeff",  test_cluster_coeff,          NULL, NULL, 0, NULL },
    { "/random_walk",   test_random_walk,            NULL, NULL, 0, NULL },
    { "/astar",         test_astar,                  NULL, NULL, 0, NULL },
    { "/k_shortest",   test_k_shortest,             NULL, NULL, 0, NULL },
    { "/betweenness",  test_betweenness,            NULL, NULL, 0, NULL },
    { "/betweenness_s", test_betweenness_sampled,   NULL, NULL, 0, NULL },
    { "/closeness",    test_closeness,             NULL, NULL, 0, NULL },
    { "/closeness_s",  test_closeness_sampled,     NULL, NULL, 0, NULL },
    { "/mst",          test_mst,                   NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },  /* terminator */
};

MunitSuite test_csr_suite = {
    "/csr",          /* prefix */
    csr_tests,       /* tests */
    NULL,            /* suites */
    1,               /* iterations */
    0,               /* options */
};
