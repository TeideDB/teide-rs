# Arena Allocator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a general-purpose arena (bump) allocator that returns `td_t*` blocks, and wire it into the sym table to eliminate 20M buddy allocator calls during high-cardinality CSV loading.

**Architecture:** Arena backed by linked list of `td_sys_alloc`'d chunks. Returns 32-byte aligned `td_t*` with `TD_ATTR_ARENA` flag. `td_release`/`td_retain` no-op for arena blocks. Sym table uses arena for string atoms instead of buddy allocator.

**Tech Stack:** Pure C17, no new dependencies. Uses `td_sys_alloc`/`td_sys_free` for backing memory.

---

### Task 1: Add `TD_ATTR_ARENA` Flag [x]

**Files:**
- Modify: `include/teide/td.h:171`

**Step 1: Add the flag constant**

After line 171 (`#define TD_ATTR_HAS_NULLS    0x40`), add:

```c
#define TD_ATTR_ARENA        0x80
```

**Step 2: Build and run tests — verify no breakage**

Run: `cmake --build build -j$(nproc) && cd build && ctest --output-on-failure`

Expected: All tests pass (flag is defined but not used yet).

**Step 3: Commit**

```bash
git add include/teide/td.h
git commit -m "feat: add TD_ATTR_ARENA flag (0x80) for arena-allocated blocks"
```

---

### Task 2: Add Arena Early-Out in `td_release` and `td_retain` [x]

**Files:**
- Modify: `src/mem/cow.c:31-47`

**Step 1: Write failing test**

Create `test/test_arena.c` with a minimal test that verifies arena-flagged blocks survive `td_release`:

```c
#include "munit.h"
#include <teide/td.h>
#include <string.h>

static MunitResult test_arena_release_noop(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    /* Allocate a block and mark it as arena-owned */
    td_t* v = td_alloc(0);
    munit_assert_ptr_not_null(v);
    v->attrs |= TD_ATTR_ARENA;

    /* td_release should be a no-op — block should not be freed */
    td_release(v);

    /* If td_release freed it, accessing v->attrs would be UB / ASan error.
     * Since it's a no-op, we can still read it. */
    munit_assert_true(v->attrs & TD_ATTR_ARENA);

    /* Clean up: remove flag and release properly */
    v->attrs &= (uint8_t)~TD_ATTR_ARENA;
    td_release(v);

    td_heap_destroy();
    return MUNIT_OK;
}

static MunitTest arena_tests[] = {
    { "/release_noop", test_arena_release_noop, NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL }
};

MunitSuite test_arena_suite = {
    "/arena", arena_tests, NULL, 1, 0
};
```

**Step 2: Register test suite in `test/test_main.c`**

Add `extern MunitSuite test_arena_suite;` after line 63 (after `test_audit_suite`).

Add `{ "/arena",   NULL, NULL, 0, 0 },` in the `child_suites[]` array before the `NULL` terminator (before line 94).

Add `child_suites[27] = test_arena_suite;` after line 133 (after `test_audit_suite` assignment).

**Step 3: Build and run — test should FAIL**

Run: `cmake --build build -j$(nproc) && ./build/test_teide --suite /arena`

Expected: FAIL — `td_release` frees the block, ASan detects use-after-free.

**Step 4: Add early-out in `td_retain` and `td_release`**

In `src/mem/cow.c`, modify `td_retain` (line 31):

```c
void td_retain(td_t* v) {
    if (!v || TD_IS_ERR(v)) return;
    if (v->attrs & TD_ATTR_ARENA) return;  /* arena-owned, no-op */
    atomic_fetch_add_explicit(&v->rc, 1, memory_order_relaxed);
}
```

Modify `td_release` (line 43):

```c
void td_release(td_t* v) {
    if (!v || TD_IS_ERR(v)) return;
    if (v->attrs & TD_ATTR_ARENA) return;  /* arena-owned, no-op */
    uint32_t prev = atomic_fetch_sub_explicit(&v->rc, 1, memory_order_acq_rel);
    if (prev == 1) td_free(v);
}
```

**Step 5: Build and run — test should PASS**

Run: `cmake --build build -j$(nproc) && ./build/test_teide --suite /arena`

Expected: PASS.

**Step 6: Run full test suite**

Run: `cd build && ctest --output-on-failure`

Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/mem/cow.c test/test_arena.c test/test_main.c
git commit -m "feat: td_release/td_retain no-op for TD_ATTR_ARENA blocks"
```

---

### Task 3: Implement Arena Allocator Core [x]

**Files:**
- Create: `src/mem/arena.h`
- Create: `src/mem/arena.c`
- Modify: `test/test_arena.c`

**Step 1: Write failing tests**

Add to `test/test_arena.c`:

```c
#include "mem/arena.h"

static MunitResult test_arena_alloc_basic(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    td_arena_t* arena = td_arena_new(4096);
    munit_assert_ptr_not_null(arena);

    /* Allocate a small block (header only, 0 data bytes) */
    td_t* v1 = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v1);
    munit_assert_false(TD_IS_ERR(v1));
    /* Must be 32-byte aligned */
    munit_assert_size((uintptr_t)v1 % 32, ==, 0);
    /* Must have arena flag */
    munit_assert_true(v1->attrs & TD_ATTR_ARENA);
    /* Must have rc=1 */
    munit_assert_uint32(v1->rc, ==, 1);

    /* Allocate a block with data */
    td_t* v2 = td_arena_alloc(arena, 100);
    munit_assert_ptr_not_null(v2);
    munit_assert_size((uintptr_t)v2 % 32, ==, 0);
    /* v2 should be different from v1 */
    munit_assert_ptr_not_equal(v1, v2);

    /* td_data(v2) should be writable for 100 bytes */
    memset(td_data(v2), 0xAB, 100);
    munit_assert_uint8(((uint8_t*)td_data(v2))[0], ==, 0xAB);
    munit_assert_uint8(((uint8_t*)td_data(v2))[99], ==, 0xAB);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_arena_grows_across_chunks(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    /* Tiny chunk size to force multiple chunks */
    td_arena_t* arena = td_arena_new(256);
    munit_assert_ptr_not_null(arena);

    /* Allocate many blocks — should span multiple chunks */
    for (int i = 0; i < 100; i++) {
        td_t* v = td_arena_alloc(arena, 64);
        munit_assert_ptr_not_null(v);
        munit_assert_false(TD_IS_ERR(v));
        munit_assert_true(v->attrs & TD_ATTR_ARENA);
    }

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_arena_reset(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    td_arena_t* arena = td_arena_new(4096);

    td_t* v1 = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v1);

    td_arena_reset(arena);

    /* After reset, next alloc may reuse the same memory */
    td_t* v2 = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v2);
    /* v2 should be at the start of the first chunk (same as v1 was) */
    munit_assert_ptr_equal(v1, v2);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_arena_oversize(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    /* Chunk size 256, but request 1024 bytes of data */
    td_arena_t* arena = td_arena_new(256);

    td_t* v = td_arena_alloc(arena, 1024);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_true(v->attrs & TD_ATTR_ARENA);
    memset(td_data(v), 0xCD, 1024);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}
```

Update the `arena_tests[]` array:

```c
static MunitTest arena_tests[] = {
    { "/release_noop",         test_arena_release_noop,         NULL, NULL, 0, NULL },
    { "/alloc_basic",          test_arena_alloc_basic,          NULL, NULL, 0, NULL },
    { "/grows_across_chunks",  test_arena_grows_across_chunks,  NULL, NULL, 0, NULL },
    { "/reset",                test_arena_reset,                NULL, NULL, 0, NULL },
    { "/oversize",             test_arena_oversize,             NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL }
};
```

**Step 2: Build — should fail (arena.h doesn't exist)**

Run: `cmake --build build -j$(nproc)`

Expected: Compilation error — `mem/arena.h` not found.

**Step 3: Create `src/mem/arena.h`**

```c
#ifndef TD_ARENA_H
#define TD_ARENA_H

#include <teide/td.h>

typedef struct td_arena td_arena_t;

/* Create arena with given chunk size (bytes). Chunks allocated via td_sys_alloc. */
td_arena_t* td_arena_new(size_t chunk_size);

/* Allocate td_t* block with nbytes of data space.
 * Returns 32-byte aligned td_t* with TD_ATTR_ARENA set, rc=1.
 * Returns NULL on OOM. */
td_t* td_arena_alloc(td_arena_t* arena, size_t nbytes);

/* Reset arena — rewind all chunks to zero. Memory retained for reuse. */
void td_arena_reset(td_arena_t* arena);

/* Destroy arena — free all backing memory. */
void td_arena_destroy(td_arena_t* arena);

#endif /* TD_ARENA_H */
```

**Step 4: Create `src/mem/arena.c`**

```c
#include "arena.h"
#include "sys.h"
#include <string.h>
#include <stdatomic.h>

/* 32-byte alignment for td_t */
#define ARENA_ALIGN 32
#define ARENA_ALIGN_UP(x) (((x) + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1))

/* Each chunk is a contiguous block of memory with a bump pointer. */
typedef struct td_arena_chunk {
    struct td_arena_chunk* next;
    size_t cap;    /* usable capacity (excluding this header) */
    size_t used;   /* bytes used so far */
    /* Data follows at aligned offset */
} td_arena_chunk_t;

/* Arena header */
struct td_arena {
    td_arena_chunk_t* head;       /* current chunk (allocate from here) */
    td_arena_chunk_t* chunks;     /* linked list of all chunks (for reset/destroy) */
    size_t            chunk_size; /* default chunk capacity */
};

/* Chunk data starts at aligned offset after the header */
static inline char* chunk_data(td_arena_chunk_t* c) {
    size_t hdr = ARENA_ALIGN_UP(sizeof(td_arena_chunk_t));
    return (char*)c + hdr;
}

static td_arena_chunk_t* arena_new_chunk(size_t min_cap) {
    size_t hdr = ARENA_ALIGN_UP(sizeof(td_arena_chunk_t));
    size_t total = hdr + min_cap;
    td_arena_chunk_t* c = (td_arena_chunk_t*)td_sys_alloc(total);
    if (!c) return NULL;
    c->next = NULL;
    c->cap = min_cap;
    c->used = 0;
    return c;
}

td_arena_t* td_arena_new(size_t chunk_size) {
    /* Minimum chunk size: at least one td_t block (32 header + some data) */
    if (chunk_size < 256) chunk_size = 256;
    /* Align chunk_size up to 32 bytes */
    chunk_size = ARENA_ALIGN_UP(chunk_size);

    td_arena_t* a = (td_arena_t*)td_sys_alloc(sizeof(td_arena_t));
    if (!a) return NULL;

    td_arena_chunk_t* first = arena_new_chunk(chunk_size);
    if (!first) {
        td_sys_free(a);
        return NULL;
    }

    a->head = first;
    a->chunks = first;
    a->chunk_size = chunk_size;
    return a;
}

td_t* td_arena_alloc(td_arena_t* arena, size_t nbytes) {
    /* Total size: 32-byte td_t header + data, aligned up to 32 */
    size_t block_size = ARENA_ALIGN_UP(32 + nbytes);

    td_arena_chunk_t* c = arena->head;

    /* Does current chunk have space? */
    if (c->used + block_size > c->cap) {
        /* Need a new chunk — at least chunk_size, or larger for oversize */
        size_t new_cap = arena->chunk_size;
        if (block_size > new_cap) new_cap = ARENA_ALIGN_UP(block_size);

        td_arena_chunk_t* nc = arena_new_chunk(new_cap);
        if (!nc) return NULL;

        /* Prepend to chunk list and set as head */
        nc->next = arena->chunks;
        arena->chunks = nc;
        arena->head = nc;
        c = nc;
    }

    /* Bump allocate */
    char* base = chunk_data(c);
    td_t* v = (td_t*)(base + c->used);
    c->used += block_size;

    /* Initialize header */
    memset(v, 0, 32);
    v->attrs = TD_ATTR_ARENA;
    atomic_store_explicit(&v->rc, 1, memory_order_relaxed);

    return v;
}

void td_arena_reset(td_arena_t* arena) {
    /* Free all chunks except the first one in the list.
     * Walk the chunk list to find the original (last in list). */
    td_arena_chunk_t* keep = NULL;
    td_arena_chunk_t* c = arena->chunks;
    while (c) {
        td_arena_chunk_t* next = c->next;
        if (next == NULL) {
            /* This is the last chunk (the first one allocated) — keep it */
            keep = c;
        } else {
            /* Not the last — free it unless it's the one we keep */
        }
        c = next;
    }

    /* Simpler approach: free all extra chunks, keep the largest or first */
    c = arena->chunks;
    td_arena_chunk_t* prev = NULL;
    td_arena_chunk_t* first_allocated = NULL;

    /* Find the tail (first allocated chunk) */
    while (c) {
        if (!c->next) { first_allocated = c; break; }
        c = c->next;
    }

    /* Free everything except first_allocated */
    c = arena->chunks;
    while (c != first_allocated) {
        td_arena_chunk_t* next = c->next;
        td_sys_free(c);
        c = next;
    }

    first_allocated->used = 0;
    first_allocated->next = NULL;
    arena->head = first_allocated;
    arena->chunks = first_allocated;
}

void td_arena_destroy(td_arena_t* arena) {
    if (!arena) return;
    td_arena_chunk_t* c = arena->chunks;
    while (c) {
        td_arena_chunk_t* next = c->next;
        td_sys_free(c);
        c = next;
    }
    td_sys_free(arena);
}
```

**Step 5: Build and run arena tests**

Run: `cmake --build build -j$(nproc) && ./build/test_teide --suite /arena`

Expected: All 5 arena tests PASS.

**Step 6: Run full test suite**

Run: `cd build && ctest --output-on-failure`

Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/mem/arena.h src/mem/arena.c test/test_arena.c
git commit -m "feat: arena allocator — bump allocator with linked chunks, reset, destroy"
```

---

### Task 4: Wire Arena Into Sym Table [x]

**Files:**
- Modify: `src/table/sym.c:24-31,41-53,75-99,105-118,226-230,289-294`
- Modify: `test/test_arena.c`

**Step 1: Write failing test — sym intern uses arena**

Add to `test/test_arena.c`:

```c
static MunitResult test_arena_sym_intern(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Intern many strings — should use arena, not buddy allocator */
    for (int i = 0; i < 10000; i++) {
        char buf[32];
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        int64_t id = td_sym_intern(buf, (size_t)len);
        munit_assert_int(id, >=, 0);
    }

    /* Verify strings are accessible */
    td_t* s = td_sym_str(0);
    munit_assert_ptr_not_null(s);
    munit_assert_true(s->attrs & TD_ATTR_ARENA);

    /* Verify roundtrip */
    int64_t id = td_sym_find("sym_999", 7);
    munit_assert_int(id, >=, 0);
    td_t* found = td_sym_str(id);
    munit_assert_int(td_str_len(found), ==, 7);
    munit_assert_memory_equal(7, td_str_ptr(found), "sym_999");

    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Add to `arena_tests[]`:
```c
{ "/sym_intern", test_arena_sym_intern, NULL, NULL, 0, NULL },
```

**Step 2: Build and run — test should FAIL**

Run: `cmake --build build -j$(nproc) && ./build/test_teide --suite /arena/sym_intern`

Expected: FAIL — `s->attrs & TD_ATTR_ARENA` is false (sym still uses buddy).

**Step 3: Add arena field to sym table and `sym_str_arena` function**

In `src/table/sym.c`:

Add include at line 26 (after `#include "mem/sys.h"`):
```c
#include "mem/arena.h"
```

Add arena field to `sym_table_t` (after line 52, before `} sym_table_t;`):
```c
    /* Arena for string atoms — avoids per-string buddy allocator calls */
    td_arena_t*  arena;
```

Add `sym_str_arena` function after `sym_unlock` (after line 69):

```c
/* Arena-backed td_str equivalent. Same logic as td_str() in atom.c
 * but allocates from the sym arena instead of the buddy allocator. */
static td_t* sym_str_arena(td_arena_t* arena, const char* s, size_t len) {
    if (len < 7) {
        /* SSO path: inline in header */
        td_t* v = td_arena_alloc(arena, 0);
        if (!v) return NULL;
        v->type = TD_ATOM_STR;
        v->slen = (uint8_t)len;
        if (len > 0) memcpy(v->sdata, s, len);
        v->sdata[len] = '\0';
        return v;
    }
    /* Long string: CHAR vector + header, both from arena */
    size_t data_size = len + 1;
    td_t* chars = td_arena_alloc(arena, data_size);
    if (!chars) return NULL;
    chars->type = TD_CHAR;
    chars->len = (int64_t)len;
    memcpy(td_data(chars), s, len);
    ((char*)td_data(chars))[len] = '\0';

    td_t* v = td_arena_alloc(arena, 0);
    if (!v) return NULL;
    v->type = TD_ATOM_STR;
    v->obj = chars;
    return v;
}
```

**Step 4: Initialize arena in `td_sym_init`**

In `td_sym_init` (after line 91, after strings allocation succeeds):
```c
    g_sym.arena = td_arena_new(1024 * 1024);  /* 1MB chunks */
    if (!g_sym.arena) {
        td_sys_free(g_sym.strings);
        td_sys_free(g_sym.buckets);
        g_sym.strings = NULL;
        g_sym.buckets = NULL;
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return;
    }
```

**Step 5: Replace `td_str` with `sym_str_arena` in both intern functions**

In `td_sym_intern` (line 228):
```c
    /* OLD: td_t* s = td_str(str, len); */
    td_t* s = sym_str_arena(g_sym.arena, str, len);
    if (!s) { sym_unlock(); return -1; }
```

In `td_sym_intern_prehashed` (line 291):
```c
    /* OLD: td_t* s = td_str(str, len); */
    td_t* s = sym_str_arena(g_sym.arena, str, len);
    if (!s) return -1;
```

**Step 6: Update `td_sym_destroy`**

Replace the release loop and cleanup (lines 105-118):

```c
void td_sym_destroy(void) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return;

    /* Arena-backed strings: td_release is a no-op (TD_ATTR_ARENA).
     * Destroy the arena to free all string atoms at once. */
    if (g_sym.arena) {
        td_arena_destroy(g_sym.arena);
        g_sym.arena = NULL;
    }

    td_sys_free(g_sym.strings);
    td_sys_free(g_sym.buckets);

    memset(&g_sym, 0, sizeof(g_sym));
    atomic_store_explicit(&g_sym_inited, false, memory_order_release);
}
```

**Step 7: Build and run — test should PASS**

Run: `cmake --build build -j$(nproc) && ./build/test_teide --suite /arena`

Expected: All 6 arena tests PASS.

**Step 8: Run full test suite**

Run: `cd build && ctest --output-on-failure`

Expected: All tests pass — existing sym, CSV, and other tests work correctly with arena-backed strings.

**Step 9: Commit**

```bash
git add src/table/sym.c test/test_arena.c
git commit -m "feat: wire arena allocator into sym table — 20M td_alloc calls eliminated"
```

---

### Task 5: Benchmark and Verify Performance [x]

**Files:** None (verification only)

**Step 1: Build release**

Run: `cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DTEIDE_BENCH=ON && cmake --build build_release -j$(nproc)`

**Step 2: Generate 10M-row test CSV**

```bash
python3 -c "
import csv, os
with open('/tmp/teide_10m.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id','val','sym'])
    for i in range(10000000):
        w.writerow([i, round(i*0.1, 2), f'sym_{i}'])
print(f'{os.path.getsize(\"/tmp/teide_10m.csv\")/1024/1024:.0f} MB')
"
```

**Step 3: Build bench binary and run Teide**

```bash
cat > /tmp/bench_arena.c << 'EOF'
#define _POSIX_C_SOURCE 199309L
#include <teide/td.h>
#include <stdio.h>
#include <time.h>
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
int main(void) {
    double best = 1e18;
    int64_t nrows = 0;
    for (int r = 0; r < 3; r++) {
        td_heap_init();
        td_sym_init();
        double start = now_ns();
        td_t* t = td_read_csv("/tmp/teide_10m.csv");
        double elapsed = now_ns() - start;
        if (t && !TD_IS_ERR(t)) { nrows = td_table_nrows(t); td_release(t); }
        td_sym_destroy();
        td_heap_destroy();
        if (elapsed < best) best = elapsed;
        printf("  run %d: %.0f ms\n", r+1, elapsed / 1e6);
    }
    printf("  Teide best: %.0f ms  (%lld rows)\n", best / 1e6, (long long)nrows);
    return 0;
}
EOF
gcc -O2 -o /tmp/bench_arena /tmp/bench_arena.c -I include -I src build_release/libteide.a -lm -lpthread
/tmp/bench_arena
```

**Step 4: Run DuckDB for comparison**

```bash
python3 -c "
import duckdb, time
con = duckdb.connect()
best = float('inf')
for _ in range(3):
    con.execute('DROP TABLE IF EXISTS t')
    t0 = time.perf_counter()
    con.execute(\"CREATE TABLE t AS SELECT * FROM read_csv('/tmp/teide_10m.csv')\")
    elapsed = time.perf_counter() - t0
    if elapsed < best: best = elapsed
    print(f'  run: {elapsed*1000:.0f} ms')
print(f'  DuckDB best: {best*1000:.0f} ms')
con.close()
"
```

**Step 5: Record results and compare**

Baseline (before arena): Teide 1280ms, DuckDB 252ms.

Record the new Teide time. The arena should eliminate the 22% `td_alloc` overhead (~280ms), bringing Teide closer to ~1000ms. If still significantly slower than DuckDB, the remaining bottleneck is in the parse loop (46%) which is a separate optimization.

**Step 6: Clean up**

```bash
rm /tmp/teide_10m.csv /tmp/bench_arena /tmp/bench_arena.c
```

**Step 7: Done — no commit needed (verification only)**

---

### Task 6: Run Full Regression Suite [x]

**Files:** None (verification only)

**Step 1: Run full debug test suite with ASan/UBSan**

Run: `cd build && ctest --output-on-failure`

Expected: All tests pass, no sanitizer warnings.

**Step 2: Done**
