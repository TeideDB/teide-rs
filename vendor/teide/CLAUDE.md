# CLAUDE.md

## What is Teide?

Pure C17 zero-dependency columnar dataframe library with native graph engine. Lazy fusion API → operation DAG → optimizer → fused morsel-driven execution. CSR edge indices, graph traversal opcodes, worst-case optimal joins, and sideways information passing — all in the same pipeline.

## Build & Test

```bash
# Debug (ASan + UBSan)
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build

# Release
cmake -B build_release -DCMAKE_BUILD_TYPE=Release && cmake --build build_release

# Run all tests
cd build && ctest --output-on-failure

# Run a single test suite
./build/test_teide --suite /vec
```

## Architecture

Core abstraction is `td_t` — a 32-byte block header. Every object (atom, vector, list, table) is a `td_t` with data following at byte 32.

**Memory**: buddy allocator with thread-local arenas, slab cache for small allocations, COW ref counting. Arena (bump) allocator (`td_arena_t`) for bulk short-lived allocations — blocks carry `TD_ATTR_ARENA` flag, making retain/release no-ops; entire arena freed at once.

**Execution pipeline**:
1. Build lazy DAG: `td_graph_new(df)` → `td_scan/td_add/td_filter/...` → `td_execute(g, root)`
2. Optimizer: type inference → constant fold → SIP → factorize → predicate pushdown → filter reorder → fusion → DCE
3. Fused executor: bytecode over register slots, morsel-by-morsel (1024 elements)

**Strings**: two representations — `TD_SYM` (dictionary-encoded symbol columns, integer indices into global intern table) and `TD_STR` (variable-length 16-byte `td_str_t` elements: strings <= 12 bytes stored inline, longer strings in a per-vector pool with 4-byte prefix for fast comparison rejection). All string opcodes (comparisons, STRLEN, UPPER/LOWER/TRIM, SUBSTR, REPLACE, CONCAT, IF) support both types. String transformation opcodes (STRLEN, UPPER/LOWER/TRIM, SUBSTR, REPLACE, CONCAT) propagate nulls: null input rows produce null output rows (CONCAT is null if any argument is null). Access via `td_str_vec_get()`; executor uses `str_resolve()` to get element array + pool pointer. Hash via `td_str_t_hash()`, compare via `td_str_t_cmp()`/`td_str_t_eq()`. During execution, `col_propagate_str_pool()` shares the source pool with the destination vector; both src and dst must be TD_STR.

**Graph engine**: CSR edge indices (`td_csr_t`, `td_rel_t`) alongside columnar tables.
- Storage: double-indexed CSR (forward + reverse), persisted as `.col` files, supports mmap
- Opcodes: `OP_EXPAND` (1-hop), `OP_VAR_EXPAND` (BFS), `OP_SHORTEST_PATH`, `OP_WCO_JOIN` (LFTJ)
- Factorized execution: `td_fvec_t` / `td_ftable_t` avoid materializing cross-products
- Optimizer: SIP pass propagates `TD_SEL` bitmaps backward through `OP_EXPAND` chains

## Code Conventions

- **Prefix**: all public symbols `td_`, internal functions `static`
- **Constants**: `TD_UPPER_SNAKE_CASE`
- **Types**: `td_name_t` (typedef'd structs)
- **Morsel-only processing**: all vector loops chunk through `td_morsel_t` (1024 elements)
- **Error returns**: `td_t*` functions use `TD_ERR_PTR()` / `TD_IS_ERR()`; other functions return `td_err_t`
- **COW cleanup**: after `td_cow()` returns a new copy, all error paths must release it (`if (vec != original) td_release(vec)`). Use `goto fail` pattern.
- **No external deps**: pure C17, single public header `include/teide/td.h`
- **No system allocator**: never use `malloc`/`calloc`/`realloc`/`free`. Use `td_alloc()`/`td_free()` for general allocation, `td_arena_alloc()` for bulk short-lived blocks. `td_sys_alloc`/`td_sys_free` reserved for allocator internals only.
- **SIMD first**: performance work must prefer SIMD approaches. Profile before optimizing, benchmark after.

## Key File Paths

```
include/teide/td.h         Single public header (all types, opcodes, API)
src/store/csr.{h,c}        CSR storage — build, save, load, mmap, free
src/ops/graph.c             DAG construction (td_expand, td_var_expand, etc.)
src/ops/exec.c              Fused morsel-driven executor (all opcodes)
src/ops/opt.c               Optimizer passes (type inference, SIP, factorize, predicate pushdown, filter reorder, fusion, DCE)
src/ops/lftj.{h,c}         Leapfrog Triejoin — iterator, search, enumeration
src/ops/fvec.{h,c}         Factorized vectors — td_fvec_t, td_ftable_t
src/store/fileio.{h,c}     Cross-platform file I/O — locking (flock/LockFileEx), fsync, atomic rename
src/table/sym.{h,c}        Global sym intern table — arena-backed string atoms, save/load, append-only persistence, file locking
src/mem/arena.{h,c}        Arena (bump) allocator — td_arena_t, bulk alloc for sym table
test/test_arena.c           Arena allocator tests (alloc, reset, destroy, sym integration)
test/test_csr.c             Graph engine tests (42 tests)
test/test_opt.c             Optimizer pass tests (filter reorder, predicate pushdown)
test/test_store.c           Storage tests (file I/O, sym persistence, col bounds validation)
test/test_sym.c             Sym table tests (save/load roundtrip, append-only, corruption)
test/test_str.c             TD_STR string vector tests (slice, concat, hash, comparisons)
test/test_exec.c            Executor tests (string ops, comparisons, conditionals, joins)
src/vec/vec.c               Vector operations — append, set, concat, slice, TD_STR string vectors with pool
src/io/csv.{h,c}           CSV loader — mmap, parallel parse, null handling, sym merge
bench/bench_csv*.c          CSV loading benchmarks (build with -DTEIDE_BENCH=ON)
```
