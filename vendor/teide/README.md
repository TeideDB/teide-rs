<p align="center">
  <img src="docs/logo.svg" alt="Teide" width="360">
</p>

<p align="center">
  Analytics and graph traversal in one fused pipeline.
</p>

<p align="center">
  <a href="https://github.com/TeideDB/teide/actions/workflows/ci.yml"><img src="https://github.com/TeideDB/teide/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="include/teide/td.h"><img src="https://img.shields.io/badge/header-td.h-informational" alt="Single Header"></a>
</p>

---

Teide is an embeddable columnar compute engine where analytics operations and
graph traversals live in the same operation DAG, pass through a 10-pass
optimizer, and execute as fused morsel-driven bytecode. Pure C17. Zero
dependencies. One header.

## Capabilities

|                              | Teide | DuckDB | Polars |
|------------------------------|:-----:|:------:|:------:|
| Native graph engine (CSR)    |   ✓   |        |        |
| Worst-case optimal joins     |   ✓   |        |        |
| Factorized execution         |   ✓   |        |        |
| SIP optimizer                |   ✓   |        |        |
| Embeddable (single header)   |   ✓   |        |        |
| Zero external dependencies   |   ✓   |        |        |
| Fused morsel pipelines       |   ✓   |   ✓    |   ✓    |
| 10-pass query optimizer      |   ✓   |   ✓    |        |
| COW ref counting             |   ✓   |        |   ✓    |
| Custom memory allocator      |   ✓   |   ✓    |        |
| Window functions & ASOF join |   ✓   |   ✓    |   ✓    |

Teide is not a SQL database. It is designed for workloads that mix analytics
with graph traversal in a single fused pipeline — without stitching tools
together.

## How It Works

<picture>
  <img src="docs/architecture.svg" alt="Architecture: User Code → Lazy DAG → Optimizer → Fused Morsel Executor → Result" width="520">
</picture>

**Build** — Construct a lazy DAG with 40+ operators: scans, filters, joins,
aggregations, window functions, graph traversals. Nothing executes yet.

**Optimize** — The DAG passes through 10 rewrite passes: type inference →
constant folding → sideways information passing → factorize → predicate
pushdown → filter reorder → projection pushdown → partition pruning → fusion →
dead code elimination.

**Execute** — Fused morsel-driven bytecode processes 1024-element chunks that
stay L1-resident. Radix-partitioned hash joins size partitions to fit L2.
Thread pool dispatches morsels in parallel.

## Memory Model

<picture>
  <img src="docs/memory.svg" alt="Memory Model: Heap with buddy allocator, slab cache, thread-local arenas, td_t block layout" width="600">
</picture>

Everything is a `td_t` — a 32-byte block header. Atoms, vectors, lists,
tables, selection bitmaps. Buddy allocator with slab cache handles ~90% of
allocations in O(1). Thread-local arenas enable lock-free allocation. COW ref
counting gives zero-copy slices and shared columns.

## Examples

### Filter + group + sum

```c
#include <teide/td.h>

int main(void) {
    td_heap_init();
    td_sym_init();

    td_t* trades = td_read_csv("trades.csv");

    /* Build the operation DAG — nothing executes yet */
    td_graph_t* g = td_graph_new(trades);

    /* Filter: keep only rows where flag == 0 */
    td_op_t* flag = td_scan(g, "flag");
    td_op_t* pred = td_eq(g, flag, td_const_i64(g, 0));

    td_op_t* region = td_filter(g, td_scan(g, "region"), pred);
    td_op_t* amount = td_filter(g, td_scan(g, "amount"), pred);

    /* Group by region, sum amounts */
    td_op_t* keys[]    = { region };
    uint16_t agg_ops[] = { OP_SUM };
    td_op_t* agg_ins[] = { amount };
    td_op_t* grp = td_group(g, keys, 1, agg_ops, agg_ins, 1);

    /* Optimize (10 passes) and execute */
    td_t* result = td_execute(g, td_optimize(g, grp));

    /* result:
     *   region | sum_amount
     *   -------|----------
     *        0 |    166583
     *        1 |    166742
     *        2 |    166900
     *        3 |    167058
     *        4 |    167217
     */

    if (result && !TD_IS_ERR(result)) td_release(result);
    td_graph_free(g);
    td_release(trades);
    td_sym_destroy();
    td_heap_destroy();
    return 0;
}
```

### Graph traversal: BFS from a start node

```c
#include <teide/td.h>

int main(void) {
    td_heap_init();
    td_sym_init();

    /* Build a directed graph: 0→1, 0→2, 1→2, 1→3, 2→3, 3→0 */
    td_t* src = td_vec_from_raw(TD_I64, (int64_t[]){0,0,1,1,2,3}, 6);
    td_t* dst = td_vec_from_raw(TD_I64, (int64_t[]){1,2,2,3,3,0}, 6);

    td_t* edges = td_table_new(2);
    edges = td_table_add_col(edges, td_sym_intern("src", 3), src);
    edges = td_table_add_col(edges, td_sym_intern("dst", 3), dst);
    td_release(src);
    td_release(dst);

    /* Double-indexed CSR (forward + reverse) */
    td_rel_t* rel = td_rel_from_edges(edges, "src", "dst", 4, 4, true);

    /* Start at node 0, BFS 1..3 hops forward */
    td_t* start = td_vec_from_raw(TD_I64, (int64_t[]){0}, 1);
    td_t* nodes = td_table_new(1);
    nodes = td_table_add_col(nodes, td_sym_intern("id", 2), start);
    td_release(start);

    td_graph_t* g = td_graph_new(nodes);
    td_op_t* reach = td_var_expand(g, td_scan(g, "id"), rel, 0, 1, 3, false);

    td_t* result = td_execute(g, td_optimize(g, reach));

    /* result (BFS from node 0, depth 1..3):
     *   src | dst | depth
     *   ----|-----|------
     *     0 |   1 |     1
     *     0 |   2 |     1
     *     0 |   3 |     2
     */

    if (result && !TD_IS_ERR(result)) td_release(result);
    td_graph_free(g);
    td_rel_free(rel);
    td_release(edges);
    td_release(nodes);
    td_sym_destroy();
    td_heap_destroy();
    return 0;
}
```

### Join two tables

```c
#include <teide/td.h>

int main(void) {
    td_heap_init();
    td_sym_init();

    td_t* orders = td_read_csv("orders.csv");
    td_t* custs  = td_read_csv("customers.csv");

    td_graph_t* g = td_graph_new(orders);

    td_op_t* lo = td_const_table(g, orders);
    td_op_t* ro = td_const_table(g, custs);

    /* Inner join on customer_id */
    td_op_t* lk[] = { td_scan(g, "customer_id") };
    td_op_t* rk[] = { td_scan(g, "customer_id") };
    td_op_t* joined = td_join(g, lo, lk, ro, rk, 1, 0);

    td_t* result = td_execute(g, td_optimize(g, joined));

    /* result:
     *   customer_id | amount | name
     *   ------------|--------|--------
     *             1 |    250 | Alice
     *             2 |    180 | Bob
     *             2 |    340 | Bob
     *             3 |    120 | Charlie
     */

    if (result && !TD_IS_ERR(result)) td_release(result);
    td_graph_free(g);
    td_release(orders);
    td_release(custs);
    td_sym_destroy();
    td_heap_destroy();
    return 0;
}
```

## Features

**Execution engine**
- Lazy DAG with 40+ operators — nothing runs until `td_execute`
- 10-pass optimizer with sideways information passing and graph-aware rewriting
- Fused morsel-driven bytecode — element-wise ops merged into single-pass chunks
- Radix-partitioned hash joins sized for L2 cache
- Thread pool with parallel morsel dispatch

**Graph engine**
- Double-indexed CSR storage (forward + reverse), mmap support
- 1-hop expand, variable-length BFS, shortest path
- Worst-case optimal joins via Leapfrog Triejoin (triangles, k-cliques)
- Factorized execution avoids materializing cross-products
- SIP propagates selection bitmaps backward through expand chains

**Data types & operations**
- Unified 32-byte `td_t` block header — atoms, vectors, lists, tables, bitmaps
- Dictionary-encoded symbols (8/16/32/64-bit adaptive-width indices)
- Variable-length strings with inline SSO and per-vector pool
- Window functions: ROW_NUMBER, RANK, DENSE_RANK, NTILE, SUM, AVG, LAG, LEAD, ...
- ASOF joins for time-series alignment
- Full null propagation across all string and arithmetic operations

**Memory & storage**
- Buddy allocator with slab cache — O(1) for ~90% of allocations
- Thread-local arenas, lock-free allocation, COW ref counting
- Columnar `.col` files, splayed tables, date-partitioned tables, mmap
- Arena allocator for bulk short-lived allocations

**I/O**
- CSV reader with type inference, configurable delimiters, null handling
- Zero external dependencies — pure C17, single public header

## API Overview

Single public header: [`include/teide/td.h`](include/teide/td.h)

| Category | Functions |
|-------------------|-------------------------------------------------------------------------|
| **Lifecycle** | `td_heap_init`, `td_heap_destroy`, `td_sym_init`, `td_sym_destroy` |
| **Memory** | `td_alloc`, `td_free`, `td_retain`, `td_release`, `td_cow` |
| **Atoms** | `td_bool`, `td_i64`, `td_f64`, `td_str`, `td_sym`, `td_date`, ... |
| **Vectors** | `td_vec_new`, `td_vec_append`, `td_vec_set`, `td_vec_get`, `td_vec_slice`, `td_vec_concat`, `td_vec_from_raw` |
| **Tables** | `td_table_new`, `td_table_add_col`, `td_table_get_col`, `td_table_ncols`, `td_table_nrows` |
| **DAG sources** | `td_graph_new`, `td_scan`, `td_const_i64`, `td_const_f64`, `td_const_str`, `td_const_table` |
| **Unary ops** | `td_neg`, `td_abs`, `td_not`, `td_sqrt_op`, `td_log_op`, `td_isnull`, `td_cast`, `td_upper`, `td_lower`, `td_trim_op` |
| **Binary ops** | `td_add`, `td_sub`, `td_mul`, `td_div`, `td_mod`, `td_eq`, `td_ne`, `td_lt`, `td_le`, `td_gt`, `td_ge`, `td_and`, `td_or`, `td_like` |
| **Aggregations** | `td_sum`, `td_count`, `td_avg`, `td_min_op`, `td_max_op`, `td_first`, `td_last`, `td_stddev`, `td_count_distinct` |
| **Structural** | `td_filter`, `td_sort_op`, `td_group`, `td_distinct`, `td_join`, `td_asof_join`, `td_select`, `td_head`, `td_tail` |
| **Window** | `td_window_op` (ROW_NUMBER, RANK, DENSE_RANK, NTILE, SUM, AVG, LAG, LEAD, ...) |
| **Graph** | `td_expand`, `td_var_expand`, `td_shortest_path`, `td_wco_join` |
| **CSR / Relations**| `td_rel_build`, `td_rel_from_edges`, `td_rel_save`, `td_rel_load`, `td_rel_mmap`, `td_rel_free` |
| **Optimizer** | `td_optimize`, `td_fuse_pass` |
| **Executor** | `td_execute` |
| **Storage** | `td_col_save`, `td_col_load`, `td_col_mmap`, `td_splay_save`, `td_splay_load`, `td_part_load` |
| **CSV** | `td_read_csv`, `td_read_csv_opts`, `td_write_csv` |
| **Parallelism** | `td_pool_init`, `td_pool_destroy`, `td_parallel_begin`, `td_parallel_end` |

## Performance

Key design choices that make Teide fast:

- **Morsel-fused execution** — element-wise ops fused into a single pass over 1024-element chunks, maximizing L1 cache residency
- **Radix-partitioned hash joins** — adaptive radix bits (2..14) size partitions to fit L2 cache
- **Buddy + slab allocator** — O(1) alloc/free for common sizes, no system allocator overhead
- **COW ref counting** — zero-copy slices and shared columns, copy only on mutation
- **Selection bitmaps** — `TD_SEL` segments skip entire morsels when all rows pass or all are filtered

Benchmarks: [teide-bench](https://github.com/TeideDB/teide-bench)

## Build

```bash
# Debug (ASan + UBSan)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release
cmake -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release

# Run all tests (270+ tests across 28 suites)
cd build && ctest --output-on-failure

# Run a single test suite
./build/test_teide --suite /vec
```

## Project Structure

```
include/teide/td.h         Single public header (all types, opcodes, API)
src/mem/                    Buddy allocator, slab cache, VM abstraction
src/core/                   Type system, atoms, strings, symbols
src/vec/                    Vector operations, morsel iterator
src/table/                  Table construction, column access, schema
src/store/                  Column files, splayed tables, partitions, CSR
src/ops/                    DAG construction, optimizer, executor, LFTJ
src/io/                     CSV reader/writer
test/                       270+ tests across 28 suites
bench/                      Benchmark harness
```

## License

[MIT](LICENSE)
