# CSV Symbol Column Performance Design

**Date**: 2026-03-20
**Status**: Approved

## Problem

CSV loading is much slower than DuckDB on symbol columns with many unique strings. The bottleneck is the merge phase where per-worker local symbol tables are merged into the global table: FNV-1a hashing is slow, each string acquires/releases a spinlock, and duplicate strings across workers cause redundant global intern calls.

## Design

### 1. Switch Global Sym Table from FNV-1a to wyhash

Replace `fnv1a()` in `src/table/sym.c` with `(uint32_t)td_hash_bytes()` (wyhash, already available in `src/ops/hash.h`). This unifies the hash function with local symbol tables (~5-10x faster per-string).

Affects: `td_sym_intern()`, `td_sym_find()`, `td_sym_load()` (rehash on load). On-disk format unchanged — hashes aren't persisted, `td_sym_load()` rebuilds the hash table from the string array.

### 2. Cross-Worker Dedup Before Global Merge

Rewrite `merge_local_syms()` to add an intermediate dedup step:

**Phase 1 — Merge locals into combined table.** For each string column, create a fresh `local_sym_t`. Iterate all workers' local tables, call `local_sym_intern()` on the combined table. Build mapping: `worker_mapping[w][local_id] → combined_id`.

**Phase 2 — Insert combined into global.** Iterate the combined table once. Call `td_sym_intern_prehashed()` with the wyhash extracted from the combined table's bucket entries. Build mapping: `combined_mapping[combined_id] → global_id`.

**Phase 3 — Compose mappings.** Before fixup, compose: `final_mapping[w][lid] = combined_mapping[worker_mapping[w][lid]]`. Fixup loop stays identical — one lookup per row.

**Impact:** For 8 workers with 80% string overlap, global intern calls drop from ~800K to ~100K.

### 3. Unlocked Prehashed Intern

Add to `src/table/sym.c`:

```c
static int64_t sym_intern_prehashed(uint32_t hash, const char* str, size_t len)
```

Same as `td_sym_intern()` but no spinlock and accepts pre-computed hash. Exposed via `src/table/sym.h` as `td_sym_intern_prehashed()`.

**Caller contract:** Only call when no other thread is interning (after `td_pool_dispatch` returns). Already guaranteed — merge runs on main thread after workers finish.

## Change Surface

| Area | File | Change |
|------|------|--------|
| Global hash switch | `src/table/sym.c` | Replace `fnv1a()` with `(uint32_t)td_hash_bytes()` |
| Unlocked prehashed intern | `src/table/sym.c` | Add `sym_intern_prehashed()` |
| Public wrapper | `src/table/sym.h` | Add `td_sym_intern_prehashed()` declaration |
| Cross-worker dedup | `src/io/csv.c` | Rewrite merge phase: locals → combined → global |
| Mapping composition | `src/io/csv.c` | Compose worker→combined→global before fixup |
| Hash reuse | `src/io/csv.c` | Extract hash from combined buckets, pass to prehashed intern |

## Not Changed

- `local_sym_t` struct and `local_sym_intern()` — unchanged
- Parse functions (`csv_parse_fn`, `csv_parse_serial`) — untouched
- Fixup dispatch structure — same, mappings built differently
- On-disk sym format — unchanged

## Testing

Existing tests cover correctness:
- 11 CSV tests (including null handling) verify roundtrip
- Sym tests (`test_sym.c`) verify intern/persist/load (will exercise wyhash automatically)
- No new tests needed — behavior identical, only performance changes
