# TD_STR Vector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `TD_STR` vector type with DuckDB-style 16-byte inline elements and arena-backed string pool for high-cardinality string columns.

**Architecture:** Each `td_str_t` element (16 bytes) inlines strings ≤12 bytes or stores a 4-byte prefix + pool offset for longer strings. The pool is a separate growable `td_t*` block (CHAR vector) pointed to from the header's bytes 8-15. Mutations are append-only to the pool with dead-byte tracking and compaction.

**Tech Stack:** Pure C17, no external deps. Uses `td_alloc()`/`td_free()`, `td_arena_t` for pool backing.

**Design doc:** `docs/plans/2026-03-21-str-vector-design.md`

---

### Task 1: Fix type constants — TD_STR, TD_ATOM_STR, TD_ATOM_F32 [x]

**Files:**
- Modify: `include/teide/td.h:124-165`
- Modify: `src/core/types.c:27-49`

**Step 1: Add TD_STR and td_str_t to td.h**

After the `TD_SYM` block (line 131), add:

```c
/* Variable-length string column (inline + pool) */
#define TD_STR       21

/* Number of types (positive range): must be > max type ID */
#define TD_TYPE_COUNT 22
```

Change `TD_TYPE_COUNT` from 21 to 22 (line 165).

Add the `td_str_t` typedef after the `td_write_sym()` function (around line 303):

```c
/* ===== Inline String Element (16 bytes) ===== */

typedef union {
    struct { uint32_t len; char     data[12]; };      /* inline: len <= 12 */
    struct { uint32_t len_; char    prefix[4];        /* pooled: len > 12  */
             uint32_t pool_off; uint32_t _pad; };
} td_str_t;

#define TD_STR_INLINE_MAX 12

static inline bool td_str_is_inline(const td_str_t* s) {
    return s->len <= TD_STR_INLINE_MAX;
}
```

**Step 2: Fix TD_ATOM_STR and add TD_ATOM_F32**

In the atom variants block (lines 149-162), change:

```c
#define TD_ATOM_F32        (-TD_F32)    /* was incorrectly TD_ATOM_STR */
#define TD_ATOM_STR        (-TD_STR)    /* -21, was hardcoded -8 */
```

Remove the old `#define TD_ATOM_STR (-8)` line.

**Step 3: Add TD_STR to type sizes table**

In `src/core/types.c`, the array currently has 21 entries (indices 0-20). Add entry for index 21:

```c
    /* [TD_SYM]       = 20 */ 8,
    /* [TD_STR]       = 21 */ 16,  /* sizeof(td_str_t) */
```

**Step 4: Add str_pool to td_t header union**

In `include/teide/td.h` inside the td_t union (line 227-231), add a new struct to the bytes 0-15 union:

```c
struct { union td_t* ext_nullmap;  union td_t* sym_dict; };
struct { union td_t* str_ext_null; union td_t* str_pool; };
```

**Step 5: Build and verify**

Run: `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build 2>&1 | head -50`
Expected: Build succeeds (no references to old -8 value since all use the TD_ATOM_STR macro)

**Step 6: Run all tests**

Run: `cd build && ctest --output-on-failure`
Expected: All existing tests pass (TD_ATOM_STR value changed but all comparisons use the macro)

**Step 7: Commit**

```bash
git add include/teide/td.h src/core/types.c
git commit -m "feat: add TD_STR type constant and td_str_t, fix TD_ATOM_STR to -TD_STR"
```

---

### Task 2: td_str_vec_new — create empty string vector with pool [x]

**Files:**
- Modify: `src/vec/vec.c:47-68`
- Modify: `include/teide/td.h` (add API declaration)
- Test: `test/test_str.c` (add new tests to existing suite)

**Step 1: Write failing tests**

Add to `test/test_str.c`:

```c
static MunitResult test_str_vec_new(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 10);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(v->type, ==, TD_STR);
    munit_assert_int64(td_len(v), ==, 0);
    /* Pool starts as NULL (no long strings yet) */
    munit_assert_null(v->str_pool);
    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_new_zero_cap(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 0);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(v->type, ==, TD_STR);
    td_release(v);
    return MUNIT_OK;
}
```

Register these in the `str_tests[]` array and rebuild.

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_vec_new(TD_STR, 10)` hits the type bounds check or returns wrong type

**Step 3: Implement td_vec_new for TD_STR**

In `src/vec/vec.c`, the existing `td_vec_new()` (line 47) already handles generic types via `td_elem_size()`. Since we added `td_type_sizes[21] = 16`, `td_vec_new(TD_STR, cap)` will allocate `cap * 16` bytes of data space. It should work with no changes to `td_vec_new()` itself.

However, we need to ensure the `str_pool` pointer is initialized to NULL. Add after line 65:

```c
    if (type == TD_STR) v->str_pool = NULL;
```

Also update `vec_capacity()` (line 35-41) — `td_sym_elem_size()` needs to handle TD_STR. Since TD_STR uses the standard size (16), `td_elem_size(TD_STR)` returns 16 which is correct. No change needed here.

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vec/vec.c test/test_str.c
git commit -m "feat: td_vec_new supports TD_STR — allocates td_str_t element array"
```

---

### Task 3: td_str_vec_append — inline strings (≤12 bytes) [x]

**Files:**
- Modify: `src/vec/vec.c`
- Modify: `include/teide/td.h` (add `td_str_vec_append` declaration)
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_vec_append_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Append a short string (fits in 12 bytes) */
    v = td_str_vec_append(v, "hello", 5);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int64(td_len(v), ==, 1);

    /* Verify element is inline */
    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint32(elems[0].len, ==, 5);
    munit_assert_memory_equal(5, elems[0].data, "hello");

    /* Pool should still be NULL (no long strings) */
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_append_inline_12(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Exactly 12 bytes — should still inline */
    v = td_str_vec_append(v, "123456789012", 12);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint32(elems[0].len, ==, 12);
    munit_assert_memory_equal(12, elems[0].data, "123456789012");
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_append_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    v = td_str_vec_append(v, "", 0);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint32(elems[0].len, ==, 0);
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_vec_append` not defined

**Step 3: Declare API in td.h**

Add to the Vector API section (around line 748):

```c
/* ===== String Vector API ===== */

td_t* td_str_vec_append(td_t* vec, const char* s, size_t len);
```

**Step 4: Implement inline path**

In `src/vec/vec.c`, add:

```c
td_t* td_str_vec_append(td_t* vec, const char* s, size_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);

    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    int64_t cap = vec_capacity(vec);
    if (vec->len >= cap) {
        size_t new_data_size = (size_t)(vec->len + 1) * sizeof(td_str_t);
        if (new_data_size < 32) new_data_size = 32;
        else {
            size_t s2 = 32;
            while (s2 < new_data_size) {
                if (s2 > SIZE_MAX / 2) return TD_ERR_PTR(TD_ERR_OOM);
                s2 *= 2;
            }
            new_data_size = s2;
        }
        td_t* nv = td_scratch_realloc(vec, new_data_size);
        if (!nv || TD_IS_ERR(nv)) return nv;
        vec = nv;
    }

    td_str_t* elem = &((td_str_t*)td_data(vec))[vec->len];
    memset(elem, 0, sizeof(td_str_t));
    elem->len = (uint32_t)len;

    if (len <= TD_STR_INLINE_MAX) {
        if (len > 0) memcpy(elem->data, s, len);
    } else {
        /* Pool path — implemented in Task 4 */
        return TD_ERR_PTR(TD_ERR_TYPE); /* placeholder */
    }

    vec->len++;
    return vec;
}
```

**Step 5: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS for inline tests

**Step 6: Commit**

```bash
git add include/teide/td.h src/vec/vec.c test/test_str.c
git commit -m "feat: td_str_vec_append — inline path for strings <= 12 bytes"
```

---

### Task 4: td_str_vec_append — pooled strings (>12 bytes) [x]

**Files:**
- Modify: `src/vec/vec.c`
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_vec_append_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* 13 bytes — must go to pool */
    const char* long_str = "hello world!!";
    v = td_str_vec_append(v, long_str, 13);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int64(td_len(v), ==, 1);

    /* Pool should be allocated */
    munit_assert_not_null(v->str_pool);

    /* Element should have prefix and pool offset */
    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint32(elems[0].len, ==, 13);
    munit_assert_memory_equal(4, elems[0].prefix, "hell");
    munit_assert_uint32(elems[0].pool_off, ==, 0);

    /* Verify pool contains the string */
    const char* pool_data = (const char*)td_data(v->str_pool);
    munit_assert_memory_equal(13, pool_data + 0, long_str);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_append_mixed(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Mix inline and pooled */
    v = td_str_vec_append(v, "short", 5);
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_append(v, "tiny", 4);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int64(td_len(v), ==, 3);

    td_str_t* elems = (td_str_t*)td_data(v);
    /* First: inline */
    munit_assert_uint32(elems[0].len, ==, 5);
    munit_assert_memory_equal(5, elems[0].data, "short");
    /* Second: pooled */
    munit_assert_uint32(elems[1].len, ==, 22);
    munit_assert_memory_equal(4, elems[1].prefix, "this");
    /* Third: inline */
    munit_assert_uint32(elems[2].len, ==, 4);
    munit_assert_memory_equal(4, elems[2].data, "tiny");

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — pooled path returns error

**Step 3: Implement pool allocation and pooled append**

Replace the pool placeholder in `td_str_vec_append` with:

```c
    } else {
        /* Pool path: allocate pool if needed, append string bytes */
        if (!vec->str_pool) {
            size_t init_pool = len < 256 ? 256 : len * 2;
            vec->str_pool = td_alloc(init_pool);
            if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) {
                vec->str_pool = NULL;
                return TD_ERR_PTR(TD_ERR_OOM);
            }
            vec->str_pool->type = TD_CHAR;
            vec->str_pool->len = 0;  /* used bytes in pool */
        }

        /* Grow pool if needed */
        int64_t pool_used = vec->str_pool->len;
        size_t pool_cap = ((size_t)1 << vec->str_pool->order) - 32;
        if ((size_t)pool_used + len > pool_cap) {
            size_t need = (size_t)pool_used + len;
            size_t new_cap = pool_cap;
            while (new_cap < need) {
                if (new_cap > SIZE_MAX / 2) return TD_ERR_PTR(TD_ERR_OOM);
                new_cap *= 2;
            }
            td_t* np = td_scratch_realloc(vec->str_pool, new_cap);
            if (!np || TD_IS_ERR(np)) return TD_ERR_PTR(TD_ERR_OOM);
            vec->str_pool = np;
        }

        /* Copy string into pool */
        char* pool_base = (char*)td_data(vec->str_pool);
        memcpy(pool_base + pool_used, s, len);

        /* Fill element: prefix + offset */
        memcpy(elem->prefix, s, 4);
        elem->pool_off = (uint32_t)pool_used;
        vec->str_pool->len = pool_used + (int64_t)len;
    }
```

**Step 4: Handle pool cleanup on vec release**

In `src/mem/heap.c` (where `td_release`/`td_free` handles child objects), add TD_STR pool release. Find the release logic and add:

```c
if (v->type == TD_STR && v->str_pool) {
    td_release(v->str_pool);
}
```

**Step 5: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 6: Run all tests to check no regressions**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 7: Commit**

```bash
git add src/vec/vec.c src/mem/heap.c test/test_str.c
git commit -m "feat: td_str_vec_append — pooled path for strings > 12 bytes"
```

---

### Task 5: td_str_vec_get — read string from vector [x]

**Files:**
- Modify: `src/vec/vec.c`
- Modify: `include/teide/td.h` (add declaration)
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_vec_get_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_not_null(ptr);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, ptr, "hello");

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_get_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "this is a long string!", 22);

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_not_null(ptr);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, ptr, "this is a long string!");

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_get_oob(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    size_t len;
    const char* ptr = td_str_vec_get(v, 1, &len);
    munit_assert_null(ptr);

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_vec_get` not defined

**Step 3: Declare and implement**

In `include/teide/td.h`:

```c
const char* td_str_vec_get(td_t* vec, int64_t idx, size_t* out_len);
```

In `src/vec/vec.c`:

```c
const char* td_str_vec_get(td_t* vec, int64_t idx, size_t* out_len) {
    if (!vec || TD_IS_ERR(vec) || vec->type != TD_STR) return NULL;
    if (idx < 0 || idx >= vec->len) return NULL;

    const td_str_t* elem = &((const td_str_t*)td_data(vec))[idx];
    *out_len = elem->len;

    if (elem->len == 0) return "";
    if (td_str_is_inline(elem)) return elem->data;

    /* Pooled: resolve via pool */
    if (!vec->str_pool) return NULL;
    return (const char*)td_data(vec->str_pool) + elem->pool_off;
}
```

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Commit**

```bash
git add include/teide/td.h src/vec/vec.c test/test_str.c
git commit -m "feat: td_str_vec_get — read inline or pooled strings by index"
```

---

### Task 6: td_str_vec_set — update string in-place with dead tracking [x]

**Files:**
- Modify: `src/vec/vec.c`
- Modify: `include/teide/td.h`
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_vec_set_inline_to_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    v = td_str_vec_set(v, 0, "world", 5);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, ptr, "world");

    /* No pool needed */
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_set_pooled_to_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "original long string!", 21);

    v = td_str_vec_set(v, 0, "replacement string!!", 20);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 20);
    munit_assert_memory_equal(20, ptr, "replacement string!!");

    /* Old 21 bytes are dead in pool */
    /* Pool used should be 21 (old dead) + 20 (new) = 41 */
    munit_assert_int64(v->str_pool->len, ==, 41);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_set_oob(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    td_t* r = td_str_vec_set(v, 5, "x", 1);
    munit_assert_true(TD_IS_ERR(r));

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_vec_set` not defined

**Step 3: Declare and implement**

In `include/teide/td.h`:

```c
td_t* td_str_vec_set(td_t* vec, int64_t idx, const char* s, size_t len);
```

In `src/vec/vec.c`:

```c
td_t* td_str_vec_set(td_t* vec, int64_t idx, const char* s, size_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (idx < 0 || idx >= vec->len) return TD_ERR_PTR(TD_ERR_RANGE);

    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    td_str_t* elem = &((td_str_t*)td_data(vec))[idx];

    /* Track dead bytes if old string was pooled */
    /* (dead_bytes tracking deferred to Task 7 — compact) */

    memset(elem, 0, sizeof(td_str_t));
    elem->len = (uint32_t)len;

    if (len <= TD_STR_INLINE_MAX) {
        if (len > 0) memcpy(elem->data, s, len);
    } else {
        /* Reuse pool append logic */
        if (!vec->str_pool) {
            size_t init_pool = len < 256 ? 256 : len * 2;
            vec->str_pool = td_alloc(init_pool);
            if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) {
                vec->str_pool = NULL;
                return TD_ERR_PTR(TD_ERR_OOM);
            }
            vec->str_pool->type = TD_CHAR;
            vec->str_pool->len = 0;
        }

        int64_t pool_used = vec->str_pool->len;
        size_t pool_cap = ((size_t)1 << vec->str_pool->order) - 32;
        if ((size_t)pool_used + len > pool_cap) {
            size_t need = (size_t)pool_used + len;
            size_t new_cap = pool_cap;
            while (new_cap < need) {
                if (new_cap > SIZE_MAX / 2) return TD_ERR_PTR(TD_ERR_OOM);
                new_cap *= 2;
            }
            td_t* np = td_scratch_realloc(vec->str_pool, new_cap);
            if (!np || TD_IS_ERR(np)) return TD_ERR_PTR(TD_ERR_OOM);
            vec->str_pool = np;
        }

        char* pool_base = (char*)td_data(vec->str_pool);
        memcpy(pool_base + pool_used, s, len);
        memcpy(elem->prefix, s, 4);
        elem->pool_off = (uint32_t)pool_used;
        vec->str_pool->len = pool_used + (int64_t)len;
    }

    return vec;
}
```

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Commit**

```bash
git add include/teide/td.h src/vec/vec.c test/test_str.c
git commit -m "feat: td_str_vec_set — in-place update with pool append"
```

---

### Task 7: td_str_vec_compact — reclaim dead pool space [x]

**Files:**
- Modify: `src/vec/vec.c`
- Modify: `include/teide/td.h`
- Test: `test/test_str.c`

**Step 1: Add dead_bytes tracking**

The `attrs` byte on the `td_t` header is only 8 bits — not enough for dead_bytes. Store dead_bytes in the pool block's `attrs` and first 3 bytes of the pool's nullmap as a uint32:

Actually simpler: use `vec->str_pool->i32` (the value union) since the pool is a CHAR vector and its value union is unused. Or, define a helper:

```c
/* Dead bytes stored in pool's nullmap[0..3] as uint32_t */
static inline uint32_t str_pool_dead(td_t* vec) {
    if (!vec->str_pool) return 0;
    uint32_t d;
    memcpy(&d, vec->str_pool->nullmap, 4);
    return d;
}
static inline void str_pool_add_dead(td_t* vec, uint32_t bytes) {
    uint32_t d = str_pool_dead(vec) + bytes;
    memcpy(vec->str_pool->nullmap, &d, 4);
}
```

**Step 2: Write failing test**

```c
static MunitResult test_str_vec_compact(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Fill with pooled strings */
    v = td_str_vec_append(v, "original string one!", 20);
    v = td_str_vec_append(v, "original string two!", 20);

    /* Overwrite first — creates 20 dead bytes */
    v = td_str_vec_set(v, 0, "replacement str one!", 20);

    int64_t pool_before = v->str_pool->len;
    munit_assert_int64(pool_before, ==, 60);  /* 20+20+20 */

    /* Compact */
    v = td_str_vec_compact(v);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    /* Pool should shrink: only 40 live bytes (string two + replacement) */
    munit_assert_int64(v->str_pool->len, ==, 40);

    /* Strings still readable */
    size_t len;
    const char* p0 = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 20);
    munit_assert_memory_equal(20, p0, "replacement str one!");

    const char* p1 = td_str_vec_get(v, 1, &len);
    munit_assert_size(len, ==, 20);
    munit_assert_memory_equal(20, p1, "original string two!");

    td_release(v);
    return MUNIT_OK;
}
```

**Step 3: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_vec_compact` not defined

**Step 4: Wire dead tracking into td_str_vec_set**

Update `td_str_vec_set` to call `str_pool_add_dead()` before overwriting a pooled element:

```c
    /* Track dead bytes if old string was pooled */
    if (!td_str_is_inline(elem) && elem->len > 0 && vec->str_pool) {
        str_pool_add_dead(vec, elem->len);
    }
```

**Step 5: Declare and implement compact**

In `include/teide/td.h`:

```c
td_t* td_str_vec_compact(td_t* vec);
```

In `src/vec/vec.c`:

```c
td_t* td_str_vec_compact(td_t* vec) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (!vec->str_pool || str_pool_dead(vec) == 0) return vec;

    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    /* Allocate new pool sized to live bytes only */
    int64_t pool_used = vec->str_pool->len;
    uint32_t dead = str_pool_dead(vec);
    size_t live_size = (size_t)(pool_used - dead);
    if (live_size == 0) {
        td_release(vec->str_pool);
        vec->str_pool = NULL;
        return vec;
    }

    td_t* new_pool = td_alloc(live_size);
    if (!new_pool || TD_IS_ERR(new_pool)) return vec;  /* keep old on OOM */
    new_pool->type = TD_CHAR;
    new_pool->len = 0;
    memset(new_pool->nullmap, 0, 16);  /* zero dead_bytes */

    char* old_base = (char*)td_data(vec->str_pool);
    char* new_base = (char*)td_data(new_pool);
    uint32_t write_off = 0;

    /* Scan elements, copy live pooled strings */
    td_str_t* elems = (td_str_t*)td_data(vec);
    for (int64_t i = 0; i < vec->len; i++) {
        if (td_str_is_inline(&elems[i]) || elems[i].len == 0) continue;

        uint32_t slen = elems[i].len;
        memcpy(new_base + write_off, old_base + elems[i].pool_off, slen);
        elems[i].pool_off = write_off;
        write_off += slen;
    }

    new_pool->len = (int64_t)write_off;
    td_release(vec->str_pool);
    vec->str_pool = new_pool;

    return vec;
}
```

**Step 6: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 7: Run all tests**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 8: Commit**

```bash
git add include/teide/td.h src/vec/vec.c test/test_str.c
git commit -m "feat: td_str_vec_compact — reclaim dead pool space"
```

---

### Task 8: td_str_t comparison and hashing helpers [x]

**Files:**
- Modify: `include/teide/td.h` (inline helpers)
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_t_eq_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "world", 5);

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_true(td_str_t_eq(&elems[0], &elems[1], NULL));
    munit_assert_false(td_str_t_eq(&elems[0], &elems[2], NULL));

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_t_eq_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "a]longer string here!", 21);
    v = td_str_vec_append(v, "a]longer string here!", 21);
    v = td_str_vec_append(v, "a]longer string nope!", 21);

    td_str_t* elems = (td_str_t*)td_data(v);
    const char* pool = (const char*)td_data(v->str_pool);
    munit_assert_true(td_str_t_eq(&elems[0], &elems[1], pool));
    /* Same prefix "a]lo" but different content */
    munit_assert_false(td_str_t_eq(&elems[0], &elems[2], pool));

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_t_cmp_order(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "apple", 5);
    v = td_str_vec_append(v, "banana", 6);
    v = td_str_vec_append(v, "apple", 5);

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_int(td_str_t_cmp(&elems[0], &elems[1], NULL), <, 0);
    munit_assert_int(td_str_t_cmp(&elems[1], &elems[0], NULL), >, 0);
    munit_assert_int(td_str_t_cmp(&elems[0], &elems[2], NULL), ==, 0);

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_t_eq`, `td_str_t_cmp` not defined

**Step 3: Implement inline helpers in td.h**

Add after the `td_str_is_inline` definition:

```c
/* Resolve string data pointer for a td_str_t element.
 * pool_base: base of string pool (NULL if all strings are inline) */
static inline const char* td_str_t_ptr(const td_str_t* s, const char* pool_base) {
    if (s->len == 0) return "";
    if (td_str_is_inline(s)) return s->data;
    return pool_base + s->pool_off;
}

/* Equality: fast reject on len, then prefix, then full compare */
static inline bool td_str_t_eq(const td_str_t* a, const td_str_t* b, const char* pool_base) {
    if (a->len != b->len) return false;
    if (a->len == 0) return true;
    if (td_str_is_inline(a)) {
        return memcmp(a->data, b->data, a->len) == 0;
    }
    /* Both pooled: check prefix first */
    if (memcmp(a->prefix, b->prefix, 4) != 0) return false;
    return memcmp(pool_base + a->pool_off, pool_base + b->pool_off, a->len) == 0;
}

/* Ordering: lexicographic, shorter string is less on prefix tie */
static inline int td_str_t_cmp(const td_str_t* a, const td_str_t* b, const char* pool_base) {
    const char* pa = td_str_t_ptr(a, pool_base);
    const char* pb = td_str_t_ptr(b, pool_base);
    uint32_t min_len = a->len < b->len ? a->len : b->len;
    int r = memcmp(pa, pb, min_len);
    if (r != 0) return r;
    return (a->len > b->len) - (a->len < b->len);
}
```

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Commit**

```bash
git add include/teide/td.h test/test_str.c
git commit -m "feat: td_str_t comparison helpers — eq with prefix rejection, lexicographic cmp"
```

---

### Task 9: Null support for TD_STR vectors [x]

**Files:**
- Modify: `src/vec/vec.c` (ensure null bitmap ops work with TD_STR)
- Test: `test/test_str.c`

**Step 1: Write failing tests**

```c
static MunitResult test_str_vec_null(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "", 0);  /* empty, not null */
    v = td_str_vec_append(v, "world", 5);
    munit_assert_int64(td_len(v), ==, 3);

    /* Mark row 1 as null */
    td_vec_set_null(v, 1, true);
    munit_assert_true(td_vec_is_null(v, 1));
    munit_assert_false(td_vec_is_null(v, 0));
    munit_assert_false(td_vec_is_null(v, 2));

    /* Non-null rows still readable */
    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, ptr, "hello");

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run test**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: Likely PASS already since null bitmap ops are type-agnostic. If not, fix the interaction.

**Step 3: Commit (if changes needed)**

```bash
git add test/test_str.c
git commit -m "test: null bitmap support for TD_STR vectors"
```

---

### Task 10: Grow-on-append stress test

**Files:**
- Test: `test/test_str.c`

**Step 1: Write stress test**

```c
static MunitResult test_str_vec_grow(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 2);  /* start small */

    /* Append 200 strings, mix of inline and pooled */
    for (int i = 0; i < 200; i++) {
        char buf[64];
        int n = snprintf(buf, sizeof(buf), "string-number-%04d", i);
        v = td_str_vec_append(v, buf, (size_t)n);
        munit_assert_not_null(v);
        munit_assert_false(TD_IS_ERR(v));
    }
    munit_assert_int64(td_len(v), ==, 200);

    /* Verify all readable */
    for (int i = 0; i < 200; i++) {
        char buf[64];
        int n = snprintf(buf, sizeof(buf), "string-number-%04d", i);
        size_t len;
        const char* ptr = td_str_vec_get(v, i, &len);
        munit_assert_not_null(ptr);
        munit_assert_size(len, ==, (size_t)n);
        munit_assert_memory_equal(len, ptr, buf);
    }

    td_release(v);
    return MUNIT_OK;
}
```

**Step 2: Run test**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS — exercises reallocation of both element array and pool

**Step 3: Commit**

```bash
git add test/test_str.c
git commit -m "test: TD_STR grow-on-append stress test — 200 strings with realloc"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Type constants: TD_STR=21, TD_ATOM_STR=-21, TD_ATOM_F32=-8 | td.h, types.c |
| 2 | `td_vec_new(TD_STR, cap)` — element array | vec.c |
| 3 | `td_str_vec_append` — inline path (≤12 bytes) | vec.c |
| 4 | `td_str_vec_append` — pooled path (>12 bytes) | vec.c, heap.c |
| 5 | `td_str_vec_get` — read by index | vec.c |
| 6 | `td_str_vec_set` — in-place update | vec.c |
| 7 | `td_str_vec_compact` — reclaim dead pool bytes | vec.c |
| 8 | `td_str_t_eq`, `td_str_t_cmp` — comparison helpers | td.h |
| 9 | Null bitmap integration | test only |
| 10 | Grow stress test | test only |
