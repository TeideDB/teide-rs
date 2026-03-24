# Graph Algorithms Expansion — Design

## Goal

Add 10 graph algorithms to reach parity with CozoDB: degree centrality, topological sort, DFS, A*, Yen's k-shortest paths, clustering coefficients, random walk, betweenness centrality, closeness centrality, and MST (Kruskal).

## Opcodes

```c
#define OP_DEGREE_CENT     92
#define OP_TOPSORT         93
#define OP_DFS             94
#define OP_ASTAR           95
#define OP_K_SHORTEST      96
#define OP_CLUSTER_COEFF   97
#define OP_RANDOM_WALK     98
#define OP_BETWEENNESS     99
#define OP_CLOSENESS      100
#define OP_MST            101
```

## Parameters

Reuse existing `td_op_ext_t.graph` struct fields:
- `direction` — 0=fwd, 1=rev, 2=both
- `max_iter` — repurposed as `k` (k-shortest), `walk_length` (random walk), `sample_size` (betweenness/closeness)
- `max_depth` — A* and DFS depth limit
- `weight_col_sym` — A*, k-shortest, MST weight column

New field for A* coordinate columns:
```c
int64_t coord_col_syms[2];  /* A*: lat/lon property column names */
```

## DAG Constructors

```c
/* Batch 1 — trivial */
td_op_t* td_degree_cent(td_graph_t* g, td_rel_t* rel);
td_op_t* td_topsort(td_graph_t* g, td_rel_t* rel);
td_op_t* td_dfs(td_graph_t* g, td_op_t* src, td_rel_t* rel, uint8_t max_depth);

/* Batch 2 — medium */
td_op_t* td_astar(td_graph_t* g, td_op_t* src, td_op_t* dst,
                  td_rel_t* rel, const char* weight_col,
                  const char* lat_col, const char* lon_col, uint8_t max_depth);
td_op_t* td_k_shortest(td_graph_t* g, td_op_t* src, td_op_t* dst,
                       td_rel_t* rel, const char* weight_col, uint16_t k);
td_op_t* td_cluster_coeff(td_graph_t* g, td_rel_t* rel);
td_op_t* td_random_walk(td_graph_t* g, td_op_t* src, td_rel_t* rel,
                        uint16_t walk_length);

/* Batch 3 — heavy */
td_op_t* td_betweenness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size);
td_op_t* td_closeness(td_graph_t* g, td_rel_t* rel, uint16_t sample_size);
td_op_t* td_mst(td_graph_t* g, td_rel_t* rel, const char* weight_col);
```

## Result Tables

| Algorithm | Columns | Types |
|---|---|---|
| Degree centrality | `_node`, `_in_degree`, `_out_degree`, `_degree` | I64, I64, I64, I64 |
| Topological sort | `_node`, `_order` | I64, I64 |
| DFS | `_node`, `_depth`, `_parent` | I64, I64, I64 |
| A* | `_node`, `_dist`, `_depth` | I64, F64, I64 |
| Yen's k-shortest | `_path_id`, `_node`, `_dist` | I64, I64, F64 |
| Clustering coefficients | `_node`, `_coefficient` | I64, F64 |
| Random walk | `_step`, `_node` | I64, I64 |
| Betweenness centrality | `_node`, `_centrality` | I64, F64 |
| Closeness centrality | `_node`, `_centrality` | I64, F64 |
| MST (Kruskal) | `_src`, `_dst`, `_weight` | I64, I64, F64 |

## Algorithm Summaries

### Batch 1 — Trivial (O(n) or O(n+m))

**Degree centrality**: Single pass over CSR offsets. `in_deg[v] = rev_off[v+1] - rev_off[v]`, `out_deg[v] = fwd_off[v+1] - fwd_off[v]`, `degree = in + out`.

**Topological sort**: Kahn's algorithm — compute in-degrees, BFS from zero-degree nodes, decrement neighbors. Return error if cycle detected (result count < n).

**DFS**: Stack-based traversal from source node. Track depth and parent per visited node. Respect `max_depth` limit.

### Batch 2 — Medium

**A***: Dijkstra with heuristic. Read lat/lon from node property columns, compute Euclidean distance to target as h(v). Priority = g(v) + h(v). Reuse existing `dijk_entry_t` heap.

**Yen's k-shortest**: Dijkstra for first path. For i=1..k-1: try each spur node on previous path, mask used edges, Dijkstra from spur to target, combine root+spur. O(k·n·m·log(n)).

**Clustering coefficients**: For each node v with degree d, count triangles by checking neighbor-pair connectivity via binary search in sorted CSR targets. `coeff = 2·triangles / (d·(d-1))`.

**Random walk**: Start at source, randomly pick neighbor at each step using xorshift64 PRNG. Walk for `walk_length` steps.

### Batch 3 — Heavy (O(n·m))

**Betweenness centrality (Brandes)**: BFS from each source (or `sample_size` random sources), accumulate dependency scores backward through BFS tree. O(sample·m) approximate.

**Closeness centrality**: BFS from each source (or `sample_size` random sources), sum distances. `closeness[v] = (n-1) / sum_dist[v]`.

**MST (Kruskal)**: Collect all weighted edges, sort by weight. Union-find with path compression + union by rank. Add edge if endpoints in different components. Produces a forest for disconnected graphs.

## Memory Management

All algorithms use `td_scratch_arena_t` for temporary allocations:
```c
td_scratch_arena_t arena;
td_scratch_arena_init(&arena);
// ... td_scratch_arena_push(&arena, nbytes) ...
td_scratch_arena_reset(&arena);  // free all on every exit path
```

## Implementation Batching

Batch 1 (3 algorithms) → Batch 2 (4 algorithms) → Batch 3 (3 algorithms). Each batch gets its own implementation plan, tests, and commit cycle.
