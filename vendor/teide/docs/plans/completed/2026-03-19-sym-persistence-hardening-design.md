# Sym Persistence Hardening Design

**Date**: 2026-03-19
**Status**: Done
**Goal**: Harden the kdb+-style global sym file model to prevent on-disk corruption, stale sym mismatches, and concurrent write races — without removing TD_SYM or sacrificing its performance advantages.

## Background

TD_SYM columns store compact integer indices (adaptive W8/W16/W32/W64) into a global intern table. This gives O(1) comparisons, small memory footprint, and cache-friendly joins. The problem is persistence: columns on disk contain raw indices that are meaningless without the exact matching sym file. Current code has six gaps:

1. `td_sym_save` rewrites the entire sym file — can overwrite newer entries from another writer
2. `td_splay_load` has no `sym_path` parameter — silently uses whatever's in memory
3. No bounds check on column load — out-of-range indices produce silent garbage
4. No crash safety — `td_sym_save` uses `fopen("wb")` without fsync or atomic rename
5. No file locking — concurrent writers can corrupt the sym file
6. No integrity validation between column files and the sym table they were written against

## Design

### A. Sym File as td_t Object

Replace the custom TSYM binary format with a regular td_t object — a TD_LIST of TD_ATOM_STR atoms. Saved and loaded via existing `td_col_save` / `td_col_load` infrastructure.

**Save**:
```c
td_err_t td_sym_save(const char* path) {
    // Lock file (exclusive)
    // Build TD_LIST of TD_ATOM_STR from g_sym.strings[0..str_count-1]
    // td_col_save(list, path_tmp)
    // fsync + rename for atomic write
    // Unlock
}
```

**Load**:
```c
td_err_t td_sym_load(const char* path) {
    // Lock file (shared)
    // td_col_load(path) → TD_LIST
    // Skip entries [0..g_sym.str_count-1] (already loaded)
    // td_sym_intern remaining entries
    // Unlock
}
```

Delete the custom TSYM format (magic 0x4D595354, length-prefixed entries). No migration — clean break.

### B. Append-Only Sym Writes

Symbol IDs are stable and monotonically increasing. New field in `sym_table_t`:

```c
uint32_t persisted_count;  // entries known to be on disk
```

On save:
1. If `persisted_count == str_count` → nothing to do
2. Build list of entries `[persisted_count .. str_count-1]`
3. If file exists: load existing list, append new entries, re-save
4. If file doesn't exist: save full list
5. Update `persisted_count = str_count` after successful write

Count header (the td_t list `len` field) is part of the 32-byte header written atomically. Crash mid-append of entries → old count still valid, partial trailing data ignored.

### C. Cross-Platform File Locking

New file: `src/store/fileio.h` / `src/store/fileio.c`

```c
#ifdef _WIN32
typedef HANDLE td_fd_t;
#else
typedef int td_fd_t;
#endif

td_fd_t  td_file_open(const char* path, int flags);
void     td_file_close(td_fd_t fd);
td_err_t td_file_lock_ex(td_fd_t fd);   // exclusive (write)
td_err_t td_file_lock_sh(td_fd_t fd);   // shared (read)
td_err_t td_file_unlock(td_fd_t fd);
td_err_t td_file_sync(td_fd_t fd);
td_err_t td_file_rename(const char* old_path, const char* new_path);
```

Platform implementations:
- **POSIX**: `flock()`, `fsync()`, `rename()`
- **Windows**: `LockFileEx()`, `FlushFileBuffers()`, `MoveFileEx(MOVEFILE_REPLACE_EXISTING)`

Used by sym save/load only. Column save/load continues using `FILE*` (no concurrent column writers).

### D. Bounds Validation on Column Load

Static helper in `col.c`, called after loading/mmapping a TD_SYM column:

```c
static td_err_t validate_sym_bounds(const void* data, int64_t len,
                                     uint8_t attrs, uint32_t sym_count) {
    // Width-dispatched scan for max index
    // W8:  scan uint8_t array
    // W16: scan uint16_t array
    // W32: scan uint32_t array
    // W64: scan int64_t array
    // If max_id >= sym_count → TD_ERR_CORRUPT
}
```

**Cost**: O(n) sequential scan over data already in cache (just loaded/memcpy'd). Negligible compared to I/O.

**Skip condition**: if `td_sym_count() == 0`, skip validation (allows raw column loads in tests without sym file). The mandatory-sym-before-data check (Section F) handles the higher-level enforcement.

### E. Sym Count in Column Metadata

On save, store current sym table size in column metadata via existing splay tree:

```c
// In td_col_save, for TD_SYM columns:
td_meta_set(col, TD_META_DICT, td_i64((int64_t)td_sym_count()));
```

On load, validate as O(1) fast-reject before the O(n) bounds scan:

```c
// In td_col_load, for TD_SYM columns:
td_t* meta = td_meta_get(col, TD_META_DICT);
if (meta && td_sym_count() > 0) {
    uint32_t needed = (uint32_t)meta->i64;
    if (td_sym_count() < needed) return TD_ERR_PTR(TD_ERR_CORRUPT);
}
```

### F. Mandatory Sym-Before-Data

**Signature change**:
```c
// Before:
td_t* td_splay_load(const char* dir);

// After:
td_t* td_splay_load(const char* dir, const char* sym_path);
```

Behavior:
1. If `sym_path` provided → `td_sym_load(sym_path)`, fail on error
2. Load all columns
3. Post-load scan: if any column is TD_SYM and `td_sym_count() == 0` → `TD_ERR_CORRUPT`

**`td_read_splayed`**: sym failure becomes fatal (currently silently ignored).

**`td_part_load`**: constructs `sym_path` internally, passes to `td_splay_load`.

## Files Changed

| File | Change |
|------|--------|
| `src/store/fileio.h` | **New** — cross-platform file lock/sync/rename API |
| `src/store/fileio.c` | **New** — POSIX + Windows implementations |
| `src/table/sym.c` | Rewrite save/load: td_t list format, file locking, append-only, `persisted_count` |
| `src/table/sym.h` | Add `persisted_count` to struct |
| `src/store/col.c` | Add `validate_sym_bounds()` in load/mmap, sym count metadata on save |
| `src/store/splay.c` | `td_splay_load` gains `sym_path`, fatal sym errors, post-load TD_SYM check |
| `src/store/part.c` | Update `td_splay_load` call site to pass `sym_path` |
| `include/teide/td.h` | `td_splay_load` signature, `td_file_*` declarations |
| `test/test_store.c` | New tests for all protection mechanisms |
| `test/test_sym.c` | Update for new save/load format |

## Implementation Order

1. `fileio.{h,c}` — no dependencies, can be tested standalone
2. `sym.c` rewrite — depends on fileio, delete TSYM format
3. `col.c` bounds validation + metadata — independent of sym rewrite
4. `splay.c` + `part.c` — depends on sym.c and col.c changes
5. `td.h` API update + test updates — final integration

## Tasks

### Task 1: Cross-platform file I/O abstraction (fileio.{h,c})

- [x] Create `src/store/fileio.h` with `td_fd_t`, `td_file_open`, `td_file_close`, `td_file_lock_ex`, `td_file_lock_sh`, `td_file_unlock`, `td_file_sync`, `td_file_rename`
- [x] Create `src/store/fileio.c` with POSIX implementation (flock, fsync, rename) and Windows stubs
- [x] Add `td_file_*` declarations to `include/teide/td.h`
- [x] Add fileio.c to CMake build
- [x] Write tests for file lock/sync/rename operations in `test/test_store.c`

### Task 2: Sym save/load rewrite (sym.c td_t list format + append-only + locking)

- [x] Add `persisted_count` field to `sym_table_t` in `src/table/sym.h`
- [x] Rewrite `td_sym_save` in `src/table/sym.c`: build TD_LIST of TD_ATOM_STR, use td_col_save, file locking, fsync + atomic rename, append-only logic
- [x] Rewrite `td_sym_load` in `src/table/sym.c`: td_col_load, skip already-loaded entries, file locking
- [x] Delete old TSYM format code (magic 0x4D595354, length-prefixed entries)
- [x] Update `test/test_sym.c` with save/load roundtrip, append-only, corrupt file tests

### Task 3: Column bounds validation + sym count metadata (col.c)

- [x] Add `validate_sym_bounds()` static helper in `src/store/col.c` — width-dispatched scan for max index
- [x] Call `validate_sym_bounds()` in `td_col_load` and `td_col_mmap` for TD_SYM columns
- [x] Store sym count metadata on save: `td_meta_set(col, TD_META_DICT, ...)` in `td_col_save` for TD_SYM columns
- [x] Validate sym count metadata on load: fast-reject if sym table too small
- [x] Write tests: out-of-range index rejection, sym count metadata mismatch

### Task 4: Mandatory sym-before-data (splay.c + part.c + td.h API update)

- [x] Change `td_splay_load` signature to `td_splay_load(const char* dir, const char* sym_path)`
- [x] Implement: if sym_path provided, call `td_sym_load`; post-load check for TD_SYM columns with empty sym table
- [x] Make sym failure fatal in `td_read_splayed`
- [x] Update `td_part_load` to construct sym_path and pass to `td_splay_load`
- [x] Update `include/teide/td.h` with new `td_splay_load` signature
- [x] Update all call sites and tests for new signature
- [x] Write integration tests: splay load with/without sym, TD_SYM columns + missing sym

## Test Plan

- Save sym as td_t list, load back, verify all strings match
- Append-only: save, intern more, save again, verify old IDs stable and new IDs present
- Corrupt sym file (truncate, bad data) → verify `TD_ERR_CORRUPT`
- Save TD_SYM column, load with wrong/missing sym → `TD_ERR_CORRUPT`
- Bounds validation: craft column with out-of-range index → rejection
- Sym count metadata: load column against sym table with fewer entries → rejection
- `td_splay_load` with NULL `sym_path` + TD_SYM columns + empty sym table → `TD_ERR_CORRUPT`
- File locking: fork + concurrent sym writes → no corruption
