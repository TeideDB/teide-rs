# Arena Allocator Design

**Date**: 2026-03-20
**Status**: Approved

## Problem

At 10M unique symbols, CSV loading is 5x slower than DuckDB. Profiling shows `td_alloc` (buddy allocator) at 22% of total time — 20M individual allocations for string atoms during sym merge. The buddy allocator is general-purpose and per-allocation overhead dominates at scale.

## Design

### General-Purpose Arena Allocator

A bump allocator backed by a linked list of fixed-size chunks. Returns `td_t*` with 32-byte aligned headers and `TD_ATTR_ARENA` flag. No individual frees — reset or destroy frees everything at once.

### API

```c
td_arena_t* td_arena_new(size_t chunk_size);
td_t*       td_arena_alloc(td_arena_t* arena, size_t nbytes);
void        td_arena_reset(td_arena_t* arena);
void        td_arena_destroy(td_arena_t* arena);
```

### Chunk Management

- Linked list of chunks, each allocated via `td_sys_alloc`
- Chunk size configurable at creation (e.g., 1MB for sym table, 64KB for morsels)
- Each chunk has a bump pointer; allocations never span chunks
- Oversized requests get their own dedicated chunk
- All returned pointers are 32-byte aligned (td_t requirement)

### td_release Integration

Early-out in `td_release()` and `td_retain()`:

```c
if (v->attrs & TD_ATTR_ARENA) return;  /* arena-owned, no-op */
```

`TD_ATTR_ARENA` = 0x80 (next free bit in attrs).

### Sym Table Integration

- Add `td_arena_t* arena` field to `sym_table_t`
- Initialize in `td_sym_init()` with `td_arena_new(1024 * 1024)` (1MB chunks)
- New `sym_str_arena()` — same as `td_str()` but uses `td_arena_alloc`
- Replace `td_str()` calls in `td_sym_intern` and `td_sym_intern_prehashed`
- `td_sym_destroy()` calls `td_arena_destroy()` — frees all strings at once

### Expected Impact

20M buddy allocator calls → ~150 `td_sys_alloc` calls (one per 1MB chunk). The 22% `td_alloc` cost in the sym merge path drops to near zero.

## Change Surface

| Area | File | Change |
|------|------|--------|
| Arena impl | `src/mem/arena.c` (new) | Core allocator |
| Arena header | `src/mem/arena.h` (new) | API declarations |
| Public attr flag | `include/teide/td.h` | `TD_ATTR_ARENA 0x80` |
| Release/retain | `src/mem/cow.c` | Early-out for arena blocks |
| Sym table | `src/table/sym.c` | Arena field, `sym_str_arena()`, destroy |
| Build | `CMakeLists.txt` | Add `src/mem/arena.c` |

## Not Changed

- `csv.c` — benefits automatically via sym table
- `exec.c` — morsel arena integration is future work
- Buddy allocator — unchanged
- `td_str()` — unchanged, arena has its own variant

## Testing

- Existing sym + CSV tests exercise arena path automatically
- New `test_arena_basic` — alloc, reset, re-alloc, destroy
- Benchmark to verify td_alloc cost drops at 10M scale
