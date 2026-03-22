# TD_STR Final Review Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 bugs from the final deep review: COW leak on OOM, null propagation in string ops, and missing dst type check in col_propagate_str_pool.

**Architecture:** Targeted fixes to vec.c (COW error paths) and exec.c (null checks in string op loops, dst type guard).

**Tech Stack:** Pure C17. Build: `cmake --build build`. Test: `./build/test_teide` and `cd build && ctest --output-on-failure`.

---

### Task 1: Fix COW'd vec leak on OOM in td_str_vec_append and td_str_vec_set

- [x] Implemented and committed

After `td_cow` returns a fresh copy (rc=1), any subsequent OOM returns `TD_ERR_PTR` without releasing the copy. The caller receives an error pointer and the cow'd vec is permanently leaked.

**Files:**
- Modify: `src/vec/vec.c:549-606` (td_str_vec_append)
- Modify: `src/vec/vec.c:666-730` (td_str_vec_set)
- Test: `test/test_str.c`

**Step 1: Write a test that exercises the COW path**

We can't easily trigger OOM, but we can test that COW + mutation works correctly and the original is preserved:

```c
static MunitResult test_str_vec_cow_append(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "this is a long string!", 22);

    /* Share the vector (rc=2) */
    td_retain(v);
    td_t* shared = v;

    /* Append to shared vec — triggers COW */
    v = td_str_vec_append(v, "new", 3);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int64(td_len(v), ==, 3);

    /* Original should still have 2 elements */
    munit_assert_int64(td_len(shared), ==, 2);

    /* Both should have correct data */
    size_t len;
    const char* p = td_str_vec_get(v, 2, &len);
    munit_assert_size(len, ==, 3);
    munit_assert_memory_equal(3, p, "new");

    const char* orig = td_str_vec_get(shared, 1, &len);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, orig, "this is a long string!");

    td_release(v);
    td_release(shared);
    return MUNIT_OK;
}

static MunitResult test_str_vec_cow_set(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "world", 5);

    /* Share */
    td_retain(v);
    td_t* shared = v;

    /* Set on shared vec — triggers COW */
    v = td_str_vec_set(v, 0, "changed", 7);
    munit_assert_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    /* Original preserved */
    size_t len;
    const char* orig = td_str_vec_get(shared, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, orig, "hello");

    const char* changed = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 7);
    munit_assert_memory_equal(7, changed, "changed");

    td_release(v);
    td_release(shared);
    return MUNIT_OK;
}
```

Register in `str_tests[]`.

**Step 2: Implement fix — release cow'd vec on error paths**

The cleanest pattern: after `td_cow` succeeds, every error return must go through a label that releases `vec`. Refactor both functions to use a `fail:` label.

In `td_str_vec_append`:

```c
td_t* td_str_vec_append(td_t* vec, const char* s, size_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (len > UINT32_MAX) return TD_ERR_PTR(TD_ERR_RANGE);

    td_t* original = vec;
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    if (!str_pool_cow(vec)) goto fail_oom;

    int64_t pool_off = 0;
    if (len > TD_STR_INLINE_MAX) {
        if (!vec->str_pool) {
            size_t init_pool = len < 256 ? 256 : len * 2;
            vec->str_pool = td_alloc(init_pool);
            if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) {
                vec->str_pool = NULL;
                goto fail_oom;
            }
            vec->str_pool->type = TD_CHAR;
            vec->str_pool->len = 0;
        }

        int64_t pool_used = vec->str_pool->len;
        size_t pool_cap = ((size_t)1 << vec->str_pool->order) - 32;
        if ((size_t)pool_used + len > pool_cap) {
            size_t need = (size_t)pool_used + len;
            size_t new_cap = pool_cap;
            if (new_cap == 0) new_cap = 256;
            while (new_cap < need) {
                if (new_cap > SIZE_MAX / 2) goto fail_oom;
                new_cap *= 2;
            }
            td_t* np = td_scratch_realloc(vec->str_pool, new_cap);
            if (!np || TD_IS_ERR(np)) goto fail_oom;
            vec->str_pool = np;
        }

        if ((uint64_t)pool_used > UINT32_MAX) goto fail_range;
        pool_off = pool_used;
    }

    /* Grow element array if needed */
    int64_t cap = vec_capacity(vec);
    if (vec->len >= cap) {
        size_t new_data_size = (size_t)(vec->len + 1) * sizeof(td_str_t);
        if (new_data_size < 32) new_data_size = 32;
        else {
            size_t s2 = 32;
            while (s2 < new_data_size) {
                if (s2 > SIZE_MAX / 2) goto fail_oom;
                s2 *= 2;
            }
            new_data_size = s2;
        }
        td_t* nv = td_scratch_realloc(vec, new_data_size);
        if (!nv || TD_IS_ERR(nv)) goto fail_oom;
        vec = nv;
    }

    /* Success path — write element */
    td_str_t* elem = &((td_str_t*)td_data(vec))[vec->len];
    memset(elem, 0, sizeof(td_str_t));
    elem->len = (uint32_t)len;

    if (len <= TD_STR_INLINE_MAX) {
        if (len > 0) memcpy(elem->data, s, len);
    } else {
        char* pool_base = (char*)td_data(vec->str_pool);
        memcpy(pool_base + pool_off, s, len);
        memcpy(elem->prefix, s, 4);
        elem->pool_off = (uint32_t)pool_off;
        vec->str_pool->len = pool_off + (int64_t)len;
    }

    vec->len++;
    return vec;

fail_oom:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_OOM);
fail_range:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_RANGE);
}
```

Apply the same `original` + `goto fail_oom/fail_range` pattern to `td_str_vec_set`.

**Important:** After `td_scratch_realloc(vec, ...)` succeeds and returns `nv`, the `original` comparison still works: if cow happened, `vec` was already different from `original` before realloc, and `nv` is the reallocated version of the cow'd copy. So `nv != original` is true and `td_release(nv)` would be correct on a subsequent failure. However, there are no failure paths after `td_scratch_realloc(vec, ...)` succeeds in append — the next code is the write + return. So this is safe.

**Step 3: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS (COW tests verify correct behavior, leak is fixed structurally)

**Step 4: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 5: Commit**

```bash
git add src/vec/vec.c test/test_str.c
git commit -m "fix: release COW'd vec on OOM in td_str_vec_append/set — prevents memory leak"
```

---

### Task 2: Null propagation in string ops (UPPER/LOWER/TRIM/STRLEN/SUBSTR/REPLACE/CONCAT)

- [x] Implemented and committed

String operations process null elements as real data instead of propagating NULL to the output.

**Files:**
- Modify: `src/ops/exec.c:10561-10602` (exec_string_unary)
- Modify: `src/ops/exec.c:10608-10632` (exec_strlen)
- Modify: `src/ops/exec.c:10635-10725` (exec_substr)
- Modify: `src/ops/exec.c:10728-10817` (exec_replace)
- Modify: `src/ops/exec.c:10820-10938` (exec_concat)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_upper_null(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* tbl = td_table_new(1, 3);
    td_table_set_name(tbl, 0, "name");

    td_t* col = td_vec_new(TD_STR, 3);
    col = td_str_vec_append(col, "hello", 5);
    col = td_str_vec_append(col, "world", 5);
    col = td_str_vec_append(col, "foo", 3);
    td_vec_set_null(col, 1, true);  /* row 1 is null */

    td_table_set_col(tbl, 0, col);
    td_release(col);

    td_graph_t* g = td_graph_new(tbl);
    td_op_t* name = td_scan(g, "name");
    td_op_t* up = td_upper(g, name);
    td_t* result = td_execute(g, up);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_STR);
    munit_assert_int(result->len, ==, 3);

    /* Row 0: "HELLO" */
    size_t len;
    const char* s0 = td_str_vec_get(result, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, s0, "HELLO");

    /* Row 1: null propagated */
    munit_assert_true(td_vec_is_null(result, 1));

    /* Row 2: "FOO" */
    const char* s2 = td_str_vec_get(result, 2, &len);
    munit_assert_size(len, ==, 3);
    munit_assert_memory_equal(3, s2, "FOO");

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_exec_str_strlen_null(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    td_t* tbl = td_table_new(1, 3);
    td_table_set_name(tbl, 0, "name");

    td_t* col = td_vec_new(TD_STR, 3);
    col = td_str_vec_append(col, "hello", 5);
    col = td_str_vec_append(col, "world", 5);
    col = td_str_vec_append(col, "foo", 3);
    td_vec_set_null(col, 1, true);

    td_table_set_col(tbl, 0, col);
    td_release(col);

    td_graph_t* g = td_graph_new(tbl);
    td_op_t* name = td_scan(g, "name");
    td_op_t* slen = td_strlen(g, name);
    td_t* result = td_execute(g, slen);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_I64);
    int64_t* d = (int64_t*)td_data(result);
    munit_assert_int64(d[0], ==, 5);
    munit_assert_true(td_vec_is_null(result, 1));  /* null propagated */
    munit_assert_int64(d[2], ==, 3);

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Register in exec test suite.

**Step 2: Run to verify failure**

Expected: FAIL — null row processed as real data, `td_vec_is_null(result, 1)` returns false

**Step 3: Implement null checks in each string op**

The pattern is the same for all ops. At the top of each row loop iteration, check if the input row is null and propagate:

**exec_string_unary** (line 10561):
```c
    for (int64_t i = 0; i < len; i++) {
        /* Propagate null */
        if (td_vec_is_null((td_t*)input, i)) {
            if (is_str) {
                result = td_str_vec_append(result, "", 0);
                if (TD_IS_ERR(result)) break;
                td_vec_set_null(result, result->len - 1, true);
            } else {
                sym_dst[i] = 0;
                td_vec_set_null(result, i, true);
            }
            continue;
        }
        /* ... existing code ... */
```

**exec_strlen** (line 10621):
```c
    for (int64_t i = 0; i < len; i++) {
        if (td_vec_is_null((td_t*)input, i)) {
            dst[i] = 0;
            td_vec_set_null(result, i, true);
            continue;
        }
        /* ... existing code ... */
```

**exec_substr** (line 10685): Same pattern — check `td_vec_is_null(input, i)`, append empty + set null for TD_STR, or set `dst[i] = 0` + null for TD_SYM.

**exec_replace** (line 10762): Same pattern.

**exec_concat** (line 10872): Check if ANY input arg at row `r` is null. If so, the concat result for that row is null:
```c
    for (int64_t r = 0; r < nrows; r++) {
        /* Check if any arg is null at this row */
        bool any_null = false;
        for (int a = 0; a < n_args; a++) {
            if (!td_is_atom(args[a]) && td_vec_is_null((td_t*)args[a], r < args[a]->len ? r : 0)) {
                any_null = true;
                break;
            }
        }
        if (any_null) {
            if (out_str) {
                result = td_str_vec_append(result, "", 0);
                if (TD_IS_ERR(result)) break;
                td_vec_set_null(result, result->len - 1, true);
            } else {
                dst[r] = 0;
                td_vec_set_null(result, r, true);
            }
            continue;
        }
        /* ... existing concat logic ... */
```

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 6: Commit**

```bash
git add src/ops/exec.c test/test_exec.c
git commit -m "fix: string ops propagate nulls — UPPER/LOWER/TRIM/STRLEN/SUBSTR/REPLACE/CONCAT skip null rows"
```

---

### Task 3: Add dst type check in col_propagate_str_pool

- [x] Implemented and committed

One-line defensive fix. If `dst` is not TD_STR, writing to `dst->str_pool` corrupts the `nullmap` union field.

**Files:**
- Modify: `src/ops/exec.c:159-167`

**Step 1: Implement fix**

Add `dst->type` check at line 160:

```c
static inline void col_propagate_str_pool(td_t* dst, const td_t* src) {
    if (src->type != TD_STR || dst->type != TD_STR) return;
    const td_t* owner = (src->attrs & TD_ATTR_SLICE) ? src->slice_parent : src;
    if (owner->str_pool) {
        if (dst->str_pool) td_release(dst->str_pool);
        td_retain(owner->str_pool);
        dst->str_pool = owner->str_pool;
    }
}
```

**Step 2: Run full suite**

Run: `cd build && cmake --build . && ctest --output-on-failure`
Expected: All pass

**Step 3: Commit**

```bash
git add src/ops/exec.c
git commit -m "fix: col_propagate_str_pool validates dst is TD_STR — prevents nullmap corruption"
```

---

## Summary

| Task | Bug | Severity | Effort |
|------|-----|----------|--------|
| 1 | COW'd vec leak on OOM in append/set | High | Medium — `original` + `goto fail` pattern |
| 2 | String ops don't propagate nulls | High | Medium — null check at top of each op loop |
| 3 | `col_propagate_str_pool` missing dst check | Low | One line |
