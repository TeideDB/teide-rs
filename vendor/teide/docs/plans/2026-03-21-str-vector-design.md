# TD_STR Vector — Inline String Columns with Pool Storage

## Motivation

Currently all string columns go through `TD_SYM` (global symbol interning). This is optimal for low-cardinality categorical data but adds unnecessary overhead for high-cardinality columns (emails, URLs, free text) where interning produces a huge symbol table with little deduplication benefit.

`TD_STR` adds a variable-length string vector type inspired by DuckDB's `string_t`, adapted to Teide's mmap-first architecture using pool offsets instead of raw pointers.

## Type Constants

```c
#define TD_STR       21
#define TD_ATOM_STR  (-TD_STR)   // -21, replaces old hardcoded -8
#define TD_ATOM_F32  (-TD_F32)   // -8, was incorrectly TD_ATOM_STR
```

The old `TD_ATOM_STR (-8)` is removed. All references updated to `-TD_STR (-21)`.

## In-Memory Element: td_str_t (16 bytes)

```c
typedef union {
    struct { uint32_t len; char data[12]; };            // inline: len <= 12
    struct { uint32_t len; char prefix[4];
             uint32_t pool_off; uint32_t _pad; };       // pooled: len > 12
} td_str_t;
```

- **Short strings (<=12 bytes)**: fully inline in the struct, no pool access
- **Long strings (>12 bytes)**: 4-byte prefix for fast comparison rejection, `pool_off` is byte offset into string pool

Distinguish inline vs pooled: check `len <= 12`.

## String Pool

Contiguous byte region storing long string data.

- **Arena-backed** for heap-allocated vectors (`td_arena_t`)
- **mmap'd** for disk-loaded vectors (zero deserialization)
- **Append-only**: new and updated strings appended at end
- **Dead tracking**: `uint32_t dead_bytes` on the vector
- **Compaction trigger**: `dead_bytes > pool_size / 2`
- **Compaction**: sequential scan — copy live strings to new pool, update all `pool_off` values, reset `dead_bytes`
- **Inline strings (<=12 bytes) bypass the pool entirely**

Pool is stored as a separate block (not part of the `td_t` allocation) — allows flexible growth during mutations and works for both heap and mmap backing.

## Disk Format (.col file)

```
td_t header (32 bytes)
td_str_t elements[count]      // 16 bytes x count
char pool[pool_size]           // packed string bytes for len > 12
```

mmap the file -> elements reference pool via offsets -> zero deserialization. The `td_str_t` array and pool are in a single contiguous file mapping. In-memory (heap) vectors use a separate arena for the pool.

## Null Handling

Same as all other vector types: `nullmap` bits in `td_t` header (up to 128 elements) or `ext_nullmap` for larger vectors. Null rows have zeroed `td_str_t` element, no pool space consumed.

## Mutations

1. **Append row**: if `len > 12`, bump-allocate into pool arena; write `td_str_t` element
2. **Update row**: append new string to pool (if long), overwrite `td_str_t` element, add old string's pool bytes to `dead_bytes` (if old string was pooled)
3. **Compact**: scan elements, copy live pooled strings to fresh pool, update `pool_off` values, reset `dead_bytes`

## Comparison Operations

```
Equality:
  1. Compare len (4 bytes) — reject if different
  2. If both inline: memcmp(data, data, 12) — one cache line, no indirection
  3. If pooled: compare prefix (4 bytes) — reject if different
  4. Fallback: memcmp via pool_base + pool_off

Ordering:
  1. Compare prefix bytes (min(len, 4) bytes)
  2. If equal: full memcmp of string data
  3. If equal content: shorter string is less
```

4-byte prefix enables fast rejection without touching the pool for ~95%+ of unequal comparisons on typical data.

## Hashing

- Inline strings: hash directly from `td_str_t.data`
- Pooled strings: hash from `pool_base + pool_off`

## Integration Points

- **CSV loader**: high-cardinality string columns detected during parse → emit `TD_STR` instead of `TD_SYM`
- **Executor**: `OP_EQ`, `OP_LT`, `OP_HASH`, `OP_STRLEN` etc. need `TD_STR` paths
- **Type inference**: optimizer pass must propagate `TD_STR` type
- **Serialization**: `.col` save/load using the disk format above
