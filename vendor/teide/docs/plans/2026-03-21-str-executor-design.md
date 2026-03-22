# TD_STR Executor Integration — Full Pipeline Support

## Goal

Add TD_STR support to the executor, optimizer, and DAG construction — full operation parity with TD_SYM. Serialization and CSV loader deferred to follow-up.

## Comparison Operations (OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE)

DuckDB-style 16-byte comparison. Fast path compares the `td_str_t` struct directly (len + 12 bytes inline or prefix). Slow path resolves via pool only when both strings are pooled and prefix matches.

Operates directly on `td_str_t*` arrays — no scratch buffer materialization like TD_SYM. Morsel loop:

```c
td_str_t* a = (td_str_t*)td_data(lhs);
td_str_t* b = (td_str_t*)td_data(rhs);
const char* pool_a = lhs->str_pool ? td_data(lhs->str_pool) : NULL;
const char* pool_b = rhs->str_pool ? td_data(rhs->str_pool) : NULL;

for (int64_t j = 0; j < n; j++)
    dst[j] = td_str_t_eq(&a[j], pool_a, &b[j], pool_b);
```

Scalar comparisons (TD_ATOM_STR vs TD_STR column): build a temporary `td_str_t` on the stack from the atom's string data, compare against each element.

## String Operations (UPPER, LOWER, TRIM, STRLEN, SUBSTR, CONCAT, REPLACE)

All string ops on TD_STR inputs produce TD_STR outputs. Shared helper:

```c
static inline const char* str_elem(td_t* col, int64_t i, size_t* out_len) {
    td_str_t* elems = (td_str_t*)td_data(col);
    *out_len = elems[i].len;
    const char* pool = col->str_pool ? td_data(col->str_pool) : NULL;
    return td_str_t_ptr(&elems[i], pool);
}
```

- **STRLEN**: Read `elem->len` directly. No pool access. Output TD_I64.
- **UPPER/LOWER/TRIM**: Read via `str_elem()`, transform in stack buffer, `td_str_vec_append` to output.
- **SUBSTR**: Read source, compute slice, append substring. Short results naturally inline.
- **CONCAT**: Read from two inputs (TD_STR or TD_SYM), concatenate in stack buffer, append to output TD_STR.
- **REPLACE**: Read, apply replacement, append result.

Mixed inputs (one TD_SYM, one TD_STR): resolve SYM via `td_sym_lookup()`, STR via `str_elem()`.

## Hashing, Group-By, Sort

**Hash**: `td_str_t_hash()` — hash string content using wyhash. Inline strings hash from struct, pooled from pool.

```c
static inline uint64_t td_str_t_hash(const td_str_t* s, const char* pool) {
    if (s->len == 0) return WYHASH_EMPTY;
    const char* p = td_str_is_inline(s) ? s->data : pool + s->pool_off;
    return wyhash(p, s->len, 0);
}
```

**Group-By**: Hash each element, probe hash table to assign group IDs. Table stores `(hash, td_str_t, group_id)`. Collision resolved via `td_str_t_eq()`. Output is I64 group-ID vector.

**Sort**: Comparison-based sort using `td_str_t_cmp()`. Build index array, sort indices with comparator + pool access. TD_SYM keeps its radix sort path.

**Count Distinct**: Same hash table as group-by, count unique entries.

## Optimizer and DAG Construction

**Type inference** (`opt.c`):
- TD_STR is its own type class (not integer-class like TD_SYM)
- TD_STR + TD_STR → TD_STR
- TD_STR + TD_SYM → TD_STR (SYM resolved to string at exec time)
- String ops on TD_STR → TD_STR output

**DAG construction** (`graph.c`):
- `td_const_str()` stays TD_SYM — string literals are low-cardinality
- Mixed TD_SYM literal + TD_STR column handled at exec time

**OP_IF**: When branches have mixed TD_STR/TD_SYM, output is TD_STR. SYM values resolved to strings.

## Slice and Concat

**Slice**: Zero-copy view. Slice retains parent's pool. `td_str_vec_get` on a slice resolves pool offsets against the parent's pool.

**Concat**: New TD_STR with merged elements and pools. Pool offsets from the second vector rebased by `pool_a_size`. Inline elements copy directly.

## Scope

**In scope**: All executor opcodes, optimizer type inference, DAG type propagation, slice, concat.

**Deferred**: col.c save/load, CSV loader cardinality detection, mmap disk format.
