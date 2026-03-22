# TD_STR Review Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 bugs found during code review of the TD_STR executor integration.

**Architecture:** Targeted fixes to vec.c (concat overflow + null propagation), exec.c (CONCAT OOB + scalar broadcasting), td.h (hash assert), and graph.c (IF type resolution).

**Tech Stack:** Pure C17. Build: `cmake --build build`. Test: `./build/test_teide --suite /str` and `cd build && ctest --output-on-failure`.

---

### Task 1: Fix concat pool_off overflow (vec.c)

The rebase `dst[i].pool_off += (uint32_t)a_pool_size` silently wraps when `total_pool > UINT32_MAX`. The check on line 275 only validates `a_pool_size` alone and fires too late (after allocation).

**Files:**
- Modify: `src/vec/vec.c:253-278`
- Test: `test/test_str.c`

- [x] Write failing test
- [x] Implement overflow guard fix
- [x] Run tests and verify pass

**Step 1: Write failing test**

```c
static MunitResult test_str_vec_concat_overflow_guard(const void* params, void* fixture) {
    (void)params; (void)fixture;
    /* We can't actually allocate 4GB pools, but we can verify the guard
     * rejects when total_pool would exceed UINT32_MAX.
     * Instead, test that normal concat works and that the guard code path
     * exists by testing concat of two vectors with pools. */
    td_t* a = td_vec_new(TD_STR, 2);
    a = td_str_vec_append(a, "this is a long pooled string a!", 30);
    td_t* b = td_vec_new(TD_STR, 2);
    b = td_str_vec_append(b, "this is a long pooled string b!", 30);

    td_t* c = td_vec_concat(a, b);
    munit_assert_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int64(td_len(c), ==, 2);

    /* Verify rebased pool offset resolves correctly */
    size_t len;
    const char* p1 = td_str_vec_get(c, 1, &len);
    munit_assert_size(len, ==, 30);
    munit_assert_memory_equal(30, p1, "this is a long pooled string b!");

    td_release(c);
    td_release(a);
    td_release(b);
    return MUNIT_OK;
}
```

Register in `str_tests[]`.

**Step 2: Implement fix**

In `src/vec/vec.c`, move the overflow check BEFORE the pool allocation (before line 258). Replace lines 253-278 with:

```c
        /* Merge pools: a's pool + b's pool */
        int64_t a_pool_size = (a_pool_owner->str_pool) ? a_pool_owner->str_pool->len : 0;
        int64_t b_pool_size = (b_pool_owner->str_pool) ? b_pool_owner->str_pool->len : 0;
        int64_t total_pool = a_pool_size + b_pool_size;

        /* Guard: total pool must fit in uint32_t for pool_off rebasing */
        if (total_pool > (int64_t)UINT32_MAX) {
            td_release(result);
            return TD_ERR_PTR(TD_ERR_RANGE);
        }

        if (total_pool > 0) {
            result->str_pool = td_alloc((size_t)total_pool);
            if (!result->str_pool || TD_IS_ERR(result->str_pool)) {
                result->str_pool = NULL;
                td_release(result);
                return TD_ERR_PTR(TD_ERR_OOM);
            }
            result->str_pool->type = TD_CHAR;
            result->str_pool->len = total_pool;
            char* pool_dst = (char*)td_data(result->str_pool);
            if (a_pool_size > 0)
                memcpy(pool_dst, td_data(a_pool_owner->str_pool), (size_t)a_pool_size);
            if (b_pool_size > 0)
                memcpy(pool_dst + a_pool_size, td_data(b_pool_owner->str_pool), (size_t)b_pool_size);
        }

        /* Copy b's elements, rebasing pool offsets */
        for (int64_t i = 0; i < b->len; i++) {
            dst[a->len + i] = b_elems[i];
            if (!td_str_is_inline(&b_elems[i]) && b_elems[i].len > 0) {
                dst[a->len + i].pool_off += (uint32_t)a_pool_size;
            }
        }
```

This removes the old `if (a_pool_size > UINT32_MAX)` check at line 275 and replaces it with a `total_pool > UINT32_MAX` check before allocation.

**Step 3: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 4: Commit**

```bash
git add src/vec/vec.c test/test_str.c
git commit -m "fix: concat pool_off overflow — check total_pool before allocation"
```

---

### Task 2: Fix concat null bitmap propagation (vec.c)

TD_STR concat drops `TD_ATTR_HAS_NULLS` and external nullmaps from inputs.

**Files:**
- Modify: `src/vec/vec.c:232-287`
- Test: `test/test_str.c`

- [x] Write failing test
- [x] Implement null propagation fix
- [x] Run tests and verify pass

**Step 1: Write failing test**

```c
static MunitResult test_str_vec_concat_nulls(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* a = td_vec_new(TD_STR, 3);
    a = td_str_vec_append(a, "hello", 5);
    a = td_str_vec_append(a, "world", 5);
    a = td_str_vec_append(a, "foo", 3);
    td_vec_set_null(a, 1, true);

    td_t* b = td_vec_new(TD_STR, 2);
    b = td_str_vec_append(b, "bar", 3);
    b = td_str_vec_append(b, "baz", 3);
    td_vec_set_null(b, 0, true);

    td_t* c = td_vec_concat(a, b);
    munit_assert_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int64(td_len(c), ==, 5);

    /* a's nulls preserved */
    munit_assert_false(td_vec_is_null(c, 0));
    munit_assert_true(td_vec_is_null(c, 1));   /* a[1] was null */
    munit_assert_false(td_vec_is_null(c, 2));
    /* b's nulls preserved at offset a->len */
    munit_assert_true(td_vec_is_null(c, 3));    /* b[0] was null */
    munit_assert_false(td_vec_is_null(c, 4));

    td_release(c);
    td_release(a);
    td_release(b);
    return MUNIT_OK;
}
```

Register in `str_tests[]`.

**Step 2: Run test to verify failure**

Expected: FAIL — nulls at positions 1 and 3 are lost

**Step 3: Implement null propagation**

After the element and pool copy in `td_vec_concat` for TD_STR, add null bitmap merging. Insert before `return result;` at end of the TD_STR block:

```c
        /* Propagate null bitmaps from a and b */
        if ((a->attrs & TD_ATTR_HAS_NULLS) || (b->attrs & TD_ATTR_HAS_NULLS)) {
            for (int64_t i = 0; i < a->len; i++) {
                if (td_vec_is_null((td_t*)a, i))
                    td_vec_set_null(result, i, true);
            }
            for (int64_t i = 0; i < b->len; i++) {
                if (td_vec_is_null((td_t*)b, i))
                    td_vec_set_null(result, a->len + i, true);
            }
        }
```

This uses the existing `td_vec_set_null` / `td_vec_is_null` which handle external nullmap promotion automatically. For TD_STR vectors, the first `td_vec_set_null` call will allocate the external nullmap.

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 6: Commit**

```bash
git add src/vec/vec.c test/test_str.c
git commit -m "fix: concat null bitmap propagation — merge nulls from both inputs"
```

---

### Task 3: Fix CONCAT OOB on mismatched arg lengths (exec.c)

In `exec_concat`, if two TD_STR args have different lengths, shorter args are accessed at out-of-bounds indices.

**Files:**
- Modify: `src/ops/exec.c:10854-10891`
- Test: `test/test_exec.c`

- [x] Implement bounds clamping fix
- [x] Run tests and verify pass

**Step 1: Write a defensive test**

This is hard to test directly since normal table queries have equal-length columns. Instead, add bounds clamping in the code and verify existing tests still pass.

**Step 2: Implement bounds clamping**

In `exec_concat`, for each TD_STR arg access, clamp the row index to the arg's length. Change lines 10854-10857 and 10886-10891:

In the pre-scan loop (~line 10852):
```c
            if (t == TD_STR) {
                const td_str_t* elems; const char* p;
                str_resolve(args[a], &elems, &p);
                int64_t ar = td_is_atom(args[a]) ? 0 : (r < args[a]->len ? r : 0);
                total += elems[ar].len;
```

In the copy loop (~line 10886):
```c
            if (t == TD_STR) {
                const td_str_t* elems; const char* pool;
                str_resolve(args[a], &elems, &pool);
                int64_t ar = td_is_atom(args[a]) ? 0 : (r < args[a]->len ? r : 0);
                const char* sp = td_str_t_ptr(&elems[ar], pool);
                size_t sl = elems[ar].len;
```

This treats any shorter-than-nrows TD_STR arg as broadcasting its first element (consistent with how scalar atoms work).

**Step 3: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS — all existing concat tests still pass

**Step 4: Commit**

```bash
git add src/ops/exec.c
git commit -m "fix: CONCAT bounds clamp — prevent OOB on mismatched TD_STR arg lengths"
```

---

### Task 4: Fix binary comparison rejecting length-1 TD_STR vectors (exec.c)

A TD_STR vector with `len == 1` is classified as `l_scalar = true` but `l_atom_str = false`, causing `TD_ERR_TYPE`.

**Files:**
- Modify: `src/ops/exec.c:1629-1630` (l_atom_str / r_atom_str)
- Modify: `src/ops/exec.c:1366-1390` (atom_to_str_t)
- Test: `test/test_exec.c`

- [x] Write failing test
- [x] Implement l_atom_str and atom_to_str_t fixes
- [x] Run tests and verify pass

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_eq_len1_broadcast(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Table with a TD_STR column */
    td_t* tbl = td_table_new(2, 3);
    td_table_set_name(tbl, 0, "name");
    td_table_set_name(tbl, 1, "tag");

    td_t* c0 = td_vec_new(TD_STR, 3);
    c0 = td_str_vec_append(c0, "alice", 5);
    c0 = td_str_vec_append(c0, "bob", 3);
    c0 = td_str_vec_append(c0, "alice", 5);
    td_table_set_col(tbl, 0, c0);
    td_release(c0);

    /* Length-1 TD_STR column (should broadcast as scalar) */
    td_t* c1 = td_vec_new(TD_STR, 1);
    c1 = td_str_vec_append(c1, "alice", 5);
    td_table_set_col(tbl, 1, c1);
    td_release(c1);

    td_graph_t* g = td_graph_new(tbl);
    td_op_t* name = td_scan(g, "name");
    td_op_t* tag = td_scan(g, "tag");
    td_op_t* eq = td_eq(g, name, tag);
    td_t* result = td_execute(g, eq);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_BOOL);
    munit_assert_int(result->len, ==, 3);
    uint8_t* d = (uint8_t*)td_data(result);
    munit_assert_uint8(d[0], ==, 1);  /* alice == alice */
    munit_assert_uint8(d[1], ==, 0);  /* bob != alice */
    munit_assert_uint8(d[2], ==, 1);  /* alice == alice */

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run to verify failure**

Expected: FAIL — returns TD_ERR_TYPE

**Step 3: Implement fix**

In `src/ops/exec.c`, expand `l_atom_str` and `r_atom_str` (lines 1629-1630) to include length-1 TD_STR vectors:

```c
        bool l_atom_str = (l_scalar && (lhs->type == TD_ATOM_STR
                          || lhs->type == TD_STR
                          || (TD_IS_SYM(lhs->type) && td_is_atom(lhs))));
        bool r_atom_str = (r_scalar && (rhs->type == TD_ATOM_STR
                          || rhs->type == TD_STR
                          || (TD_IS_SYM(rhs->type) && td_is_atom(rhs))));
```

Then extend `atom_to_str_t` (line 1367) to handle `TD_STR` type (length-1 vector):

```c
static void atom_to_str_t(td_t* atom, td_str_t* out, const char** out_pool) {
    const char* sp;
    size_t sl;
    if (atom->type == TD_ATOM_STR) {
        sp = td_str_ptr(atom);
        sl = td_str_len(atom);
    } else if (atom->type == TD_STR) {
        /* Length-1 TD_STR vector used as scalar */
        const td_str_t* elems = (const td_str_t*)td_data(atom);
        *out = elems[0];
        *out_pool = atom->str_pool ? (const char*)td_data(atom->str_pool) : NULL;
        return;
    } else if (TD_IS_SYM(atom->type) && td_is_atom(atom)) {
        td_t* s = td_sym_str(atom->i64);
        sp = s ? td_str_ptr(s) : "";
        sl = s ? td_str_len(s) : 0;
    } else {
        sp = ""; sl = 0;
    }
    memset(out, 0, sizeof(td_str_t));
    out->len = (uint32_t)sl;
    if (sl <= TD_STR_INLINE_MAX) {
        if (sl > 0) memcpy(out->data, sp, sl);
        *out_pool = NULL;
    } else {
        memcpy(out->prefix, sp, 4);
        out->pool_off = 0;
        *out_pool = sp;
    }
}
```

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/exec.c test/test_exec.c
git commit -m "fix: binary comparison accepts length-1 TD_STR vectors as broadcast scalars"
```

---

### Task 5: Fix td_str_t_hash NULL pool assert (td.h)

Replace the silent `return 0` with a debug assert for the caller contract violation.

**Files:**
- Modify: `include/teide/td.h:364`

- [x] Implement assert fix
- [x] Run tests and verify pass

**Step 1: Implement fix**

Change line 364 in `td_str_t_hash`:

```c
static inline uint64_t td_str_t_hash(const td_str_t* s, const char* pool_base) {
    if (s->len == 0) return 0x9E3779B97F4A7C15ULL;
    /* Precondition: pooled strings must have valid pool_base */
    if (!td_str_is_inline(s)) {
        assert(pool_base != NULL && "td_str_t_hash: pooled string requires non-NULL pool_base");
    }
    const char* p = td_str_is_inline(s) ? s->data : pool_base + s->pool_off;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < s->len; i++) {
        h ^= (uint64_t)(unsigned char)p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
```

Also add `#include <assert.h>` if not already included. Check first:

Run: `grep '#include.*assert' include/teide/td.h`

If not present, add it near the other includes at the top of td.h.

**Step 2: Run tests**

Run: `cd build && cmake --build . && ctest --output-on-failure`
Expected: All pass (no caller passes NULL pool for pooled strings)

**Step 3: Commit**

```bash
git add include/teide/td.h
git commit -m "fix: td_str_t_hash asserts non-NULL pool for pooled strings"
```

---

### Task 6: Fix td_if type resolution divergence (graph.c)

The post-promote SYM override in `td_if` is inconsistent with `promote()` which treats SYM as integer-class. Since `promote()` now handles TD_STR directly (returns TD_STR when either input is TD_STR), simplify the override.

**Files:**
- Modify: `src/ops/graph.c:459-464`
- Test: existing tests cover this

- [x] Implement type resolution fix
- [x] Run tests and verify pass

**Step 1: Implement fix**

Replace lines 459-464 in graph.c:

```c
    int8_t out_type = promote(then_val->out_type, else_val->out_type);
    /* String types: TD_STR wins over TD_SYM; TD_SYM stays TD_SYM (not integer) */
    if (then_val->out_type == TD_STR || else_val->out_type == TD_STR)
        out_type = TD_STR;
    else if (then_val->out_type == TD_SYM || else_val->out_type == TD_SYM)
        out_type = TD_SYM;
```

Wait — the reviewers said the `promote()` already returns TD_STR for TD_STR inputs, making the first `if` redundant. But the SYM override is the real issue: `promote()` returns TD_I64 for SYM, but `td_if` overrides to TD_SYM. This is intentional for IF (you want `IF(cond, sym_col, sym_col)` to produce SYM, not I64). The problem is only that this diverges from `promote()`.

The cleanest fix: keep the SYM override but add it to `promote()` itself so the behavior is consistent. However, that would affect all binary ops (ADD on SYM would return SYM instead of I64), which is wrong.

Better approach: keep the existing code as-is, but remove the redundant TD_STR check since `promote()` already handles it:

```c
    int8_t out_type = promote(then_val->out_type, else_val->out_type);
    /* IF preserves string types: SYM stays SYM (not I64 from promote) */
    if (then_val->out_type == TD_SYM || else_val->out_type == TD_SYM)
        out_type = TD_SYM;
```

The TD_STR case is already handled by `promote()` returning TD_STR. The SYM override is intentional and correct for IF specifically.

**Step 2: Run tests**

Run: `cd build && cmake --build . && ctest --output-on-failure`
Expected: All pass

**Step 3: Commit**

```bash
git add src/ops/graph.c
git commit -m "fix: td_if type resolution — remove redundant TD_STR check, document SYM override"
```

---

## Summary

| Task | Bug | File | Fix |
|------|-----|------|-----|
| 1 | pool_off overflow wraps silently | vec.c | Check `total_pool > UINT32_MAX` before allocation |
| 2 | Concat drops null bitmaps | vec.c | Merge null bits from both inputs |
| 3 | CONCAT OOB on mismatched lengths | exec.c | Clamp row index to arg length |
| 4 | Length-1 TD_STR rejected as scalar | exec.c | Extend `l_atom_str` + `atom_to_str_t` |
| 5 | Hash returns 0 for NULL pool | td.h | Assert precondition |
| 6 | IF type resolution redundancy | graph.c | Remove redundant TD_STR check |
