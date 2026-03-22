# CSV Symbol Column Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Speed up CSV loading of high-cardinality symbol columns by switching the global sym table to wyhash, deduplicating across workers before global merge, and eliminating lock overhead during merge.

**Architecture:** Three changes: (1) replace FNV-1a with wyhash in sym.c, (2) merge per-worker local tables into one combined local table before touching global, (3) add unlocked prehashed intern for the merge phase.

**Tech Stack:** Pure C17, no new dependencies. Reuses existing wyhash from `src/ops/hash.h` and `local_sym_t` from `src/io/csv.c`.

---

### Task 1: Switch Global Sym Table from FNV-1a to wyhash

**Files:**
- Modify: `src/table/sym.c:24-43,188,265`
- Modify: `src/table/sym.h`

**Step 1: Write test to verify sym roundtrip still works after hash change**

No new test needed — existing `test_sym.c` tests cover intern/find/persist/load roundtrip. The hash change is transparent to the API. We'll verify by running existing tests.

**Step 2: Replace FNV-1a with wyhash in sym.c**

In `src/table/sym.c`:

1. Add include after existing includes (line 27):
```c
#include "ops/hash.h"
```

2. Remove the `fnv1a()` function (lines 32-43, the function and its comment block).

3. Replace all `fnv1a(str, len)` calls with `(uint32_t)td_hash_bytes(str, len)`:
   - Line 188 in `td_sym_intern`: `uint32_t hash = fnv1a(str, len);` → `uint32_t hash = (uint32_t)td_hash_bytes(str, len);`
   - Line 265 in `td_sym_find`: `uint32_t hash = fnv1a(str, len);` → `uint32_t hash = (uint32_t)td_hash_bytes(str, len);`

4. Update the comment in `sym.h` (line 30) from "FNV-1a 32-bit hashing" to "wyhash (truncated to 32-bit)".

**Step 3: Build and run tests**

Run: `cmake --build build -j$(nproc) && cd build && ctest --output-on-failure`

Expected: ALL tests pass (sym tests verify roundtrip, CSV tests verify end-to-end).

**Step 4: Commit**

```bash
git add src/table/sym.c src/table/sym.h
git commit -m "perf: switch global sym table from FNV-1a to wyhash"
```

---

### Task 2: Add Unlocked Prehashed Intern

**Files:**
- Modify: `src/table/sym.c`
- Modify: `src/table/sym.h`

**Step 1: Add `td_sym_intern_prehashed()` to sym.c**

Add after `td_sym_intern()` (after line ~252):

```c
/* --------------------------------------------------------------------------
 * td_sym_intern_prehashed -- intern with pre-computed hash, no lock.
 *
 * CALLER CONTRACT: must only be called when no other thread is interning
 * (e.g., after td_pool_dispatch returns during CSV merge).
 * -------------------------------------------------------------------------- */

int64_t td_sym_intern_prehashed(uint32_t hash, const char* str, size_t len) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return -1;

    /* No lock — caller guarantees single-threaded access */

    uint32_t mask = g_sym.bucket_cap - 1;
    uint32_t slot = hash & mask;

    /* Probe for existing entry */
    for (;;) {
        uint64_t e = g_sym.buckets[slot];
        if (e == 0) break;

        uint32_t e_hash = (uint32_t)(e >> 32);
        if (e_hash == hash) {
            uint32_t e_id = (uint32_t)(e & 0xFFFFFFFF) - 1;
            td_t* existing = g_sym.strings[e_id];
            if (td_str_len(existing) == len &&
                memcmp(td_str_ptr(existing), str, len) == 0) {
                return (int64_t)e_id;
            }
        }
        slot = (slot + 1) & mask;
    }

    /* Growth check */
    if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 70) {
        if (!ht_grow()) {
            if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 95) {
                return -1;
            }
        }
    }

    /* Not found — create new entry */
    uint32_t new_id = g_sym.str_count;

    if (new_id >= g_sym.str_cap) {
        if (g_sym.str_cap >= UINT32_MAX / 2) return -1;
        uint32_t new_str_cap = g_sym.str_cap * 2;
        td_t** new_strings = (td_t**)td_sys_realloc(g_sym.strings,
                                                    (size_t)new_str_cap * sizeof(td_t*));
        if (!new_strings) return -1;
        g_sym.strings = new_strings;
        g_sym.str_cap = new_str_cap;
    }

    td_t* s = td_str(str, len);
    if (!s || TD_IS_ERR(s)) return -1;
    g_sym.strings[new_id] = s;
    g_sym.str_count++;

    ht_insert(g_sym.buckets, g_sym.bucket_cap, hash, new_id);

    return (int64_t)new_id;
}
```

**Step 2: Add declaration to sym.h**

Add before `#endif`:
```c
/* Intern with pre-computed wyhash, no lock.
 * Caller must guarantee single-threaded access. */
int64_t td_sym_intern_prehashed(uint32_t hash, const char* str, size_t len);
```

**Step 3: Build and run tests**

Run: `cmake --build build -j$(nproc) && cd build && ctest --output-on-failure`

Expected: All tests pass. The new function isn't called yet — just verifying it compiles.

**Step 4: Commit**

```bash
git add src/table/sym.c src/table/sym.h
git commit -m "feat: add td_sym_intern_prehashed() — unlocked intern with pre-computed hash"
```

---

### Task 3: Cross-Worker Dedup in Merge Phase

**Files:**
- Modify: `src/io/csv.c:908-994` (`merge_local_syms`)

**Step 1: Rewrite `merge_local_syms` to dedup across workers**

Replace the current merge loop (lines ~929-956) with three phases:

```c
static bool merge_local_syms(local_sym_t* local_syms, uint32_t n_workers,
                              int n_cols, const csv_type_t* col_types,
                              void** col_data, int64_t n_rows,
                              td_pool_t* pool, int64_t* col_max_ids,
                              uint8_t** col_nullmaps) {
    bool ok = true;
    for (int c = 0; c < n_cols; c++) {
        if (col_types[c] != CSV_TYPE_STR) continue;

        /* --- Phase 1: Merge all worker locals into one combined table --- */
        local_sym_t combined;
        memset(&combined, 0, sizeof(combined));
        local_sym_init(&combined);

        /* Per-worker mapping: local_id → combined_id */
        int64_t* worker_to_combined[n_workers]; /* VLA */
        td_t* wtc_hdrs[n_workers];
        uint32_t wtc_counts[n_workers];
        for (uint32_t w = 0; w < n_workers; w++) {
            worker_to_combined[w] = NULL;
            wtc_hdrs[w] = NULL;
            wtc_counts[w] = 0;
        }

        for (uint32_t w = 0; w < n_workers; w++) {
            local_sym_t* ls = &local_syms[(size_t)w * (size_t)n_cols + (size_t)c];
            if (ls->count == 0) continue;

            worker_to_combined[w] = (int64_t*)scratch_alloc(&wtc_hdrs[w],
                                        ls->count * sizeof(int64_t));
            if (!worker_to_combined[w]) continue;
            wtc_counts[w] = ls->count;

            for (uint32_t i = 0; i < ls->count; i++) {
                uint32_t cid = local_sym_intern(&combined,
                    ls->arena + ls->offsets[i], ls->lens[i]);
                worker_to_combined[w][i] = (int64_t)cid;
            }
        }

        /* --- Phase 2: Insert combined into global (unlocked, prehashed) --- */

        /* Pre-grow global table */
        uint32_t current_count = td_sym_count();
        td_sym_ensure_cap(current_count + combined.count);

        /* combined_id → global_id mapping */
        td_t* ctg_hdr = NULL;
        int64_t* combined_to_global = NULL;
        if (combined.count > 0) {
            combined_to_global = (int64_t*)scratch_alloc(&ctg_hdr,
                                    combined.count * sizeof(int64_t));
        }

        if (combined_to_global) {
            for (uint32_t i = 0; i < combined.count; i++) {
                /* Extract pre-computed hash from combined table's buckets.
                 * We need to find the bucket for id=i. Rather than searching,
                 * recompute from the string — wyhash is the same function. */
                uint32_t hash = (uint32_t)td_hash_bytes(
                    combined.arena + combined.offsets[i], combined.lens[i]);
                combined_to_global[i] = td_sym_intern_prehashed(hash,
                    combined.arena + combined.offsets[i], combined.lens[i]);
                if (combined_to_global[i] < 0) {
                    ok = false;
                    combined_to_global[i] = 0;
                }
            }
        }

        /* --- Phase 3: Compose mappings and build final worker→global --- */
        int64_t* mappings[n_workers]; /* VLA: final worker_lid → global_id */
        td_t* map_hdrs[n_workers];
        uint32_t mapping_counts[n_workers];
        for (uint32_t w = 0; w < n_workers; w++) {
            mappings[w] = NULL;
            map_hdrs[w] = NULL;
            mapping_counts[w] = 0;
        }

        for (uint32_t w = 0; w < n_workers; w++) {
            if (!worker_to_combined[w] || wtc_counts[w] == 0) continue;

            mappings[w] = (int64_t*)scratch_alloc(&map_hdrs[w],
                                        wtc_counts[w] * sizeof(int64_t));
            if (!mappings[w]) continue;
            mapping_counts[w] = wtc_counts[w];

            for (uint32_t i = 0; i < wtc_counts[w]; i++) {
                int64_t cid = worker_to_combined[w][i];
                mappings[w][i] = (combined_to_global && cid >= 0 &&
                                  (uint32_t)cid < combined.count)
                    ? combined_to_global[cid] : 0;
            }
        }

        /* Free intermediate mappings */
        for (uint32_t w = 0; w < n_workers; w++) scratch_free(wtc_hdrs[w]);
        scratch_free(ctg_hdr);
        local_sym_free(&combined);

        /* Track max global sym_id for adaptive width narrowing */
        int64_t max_id = 0;
        for (uint32_t w = 0; w < n_workers; w++) {
            for (uint32_t i = 0; i < mapping_counts[w]; i++) {
                if (mappings[w][i] > max_id) max_id = mappings[w][i];
            }
        }
        if (col_max_ids) col_max_ids[c] = max_id;

        /* Fix up column data: parallel unpack (wid, lid) → global sym_id */
        uint32_t* data = (uint32_t*)col_data[c];
        uint8_t* nm = col_nullmaps ? col_nullmaps[c] : NULL;
        if (pool && n_rows > 1024) {
            sym_fixup_ctx_t ctx = { .data = data, .mappings = mappings,
                                    .mapping_counts = mapping_counts,
                                    .n_workers = n_workers,
                                    .nullmap = nm };
            td_pool_dispatch(pool, sym_fixup_fn, &ctx, n_rows);
        } else {
            for (int64_t r = 0; r < n_rows; r++) {
                if (nm && (nm[r >> 3] & (1u << (r & 7)))) continue;
                uint32_t packed = data[r];
                uint32_t wid = UNPACK_WID(packed);
                if (wid >= n_workers) continue;
                uint32_t lid = UNPACK_LID(packed);
                if (mappings[wid] && lid < mapping_counts[wid])
                    data[r] = (uint32_t)mappings[wid][lid];
                else
                    data[r] = 0;
            }
        }

        for (uint32_t w = 0; w < n_workers; w++) scratch_free(map_hdrs[w]);
    }
    return ok;
}
```

**Step 2: Add `#include "ops/hash.h"` to csv.c if not already present**

Check if csv.c already includes hash.h (it may via another include). If not, add it near the top.

**Step 3: Build and run all tests**

Run: `cmake --build build -j$(nproc) && cd build && ctest --output-on-failure`

Expected: All tests pass — behavior identical, mappings just built differently.

**Step 4: Commit**

```bash
git add src/io/csv.c
git commit -m "perf: cross-worker dedup in CSV sym merge — locals→combined→global"
```

---

### Task 4: Regression Check

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `cd build && ctest --output-on-failure`

Expected: All tests pass with ASan/UBSan (debug build). No sanitizer warnings.

**Step 2: Verify sym persistence roundtrip**

The sym tests (`test_sym.c`) cover save/load roundtrip. Confirm they pass — this validates that the wyhash switch doesn't break persistence (hashes are rebuilt from strings on load).

**Step 3: Done — no commit needed**
