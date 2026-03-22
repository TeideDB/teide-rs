# TD_STR Deep Review Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 bugs found during deep review: slice null redirect (cross-cutting), compact saturation overflow, atom_to_str_t OOB read.

**Architecture:** Targeted fixes to vec.c and exec.c with regression tests.

**Tech Stack:** Pure C17. Build: `cmake --build build`. Test: `./build/test_teide` and `cd build && ctest --output-on-failure`.

---

### Task 1: Fix td_vec_is_null — add slice redirect (cross-cutting, all vector types)

`td_vec_is_null` has no slice-aware path. Slices don't carry `TD_ATTR_HAS_NULLS` in their attrs (they only have `TD_ATTR_SLICE | width bits`). So `td_vec_is_null(slice, i)` always returns false, even when the parent has nulls. This affects concat null propagation and any other caller checking nulls on a slice.

**Files:**
- Modify: `src/vec/vec.c:782-801` (td_vec_is_null)
- Test: `test/test_str.c` (TD_STR slice with nulls)
- Test: `test/test_vec.c` (generic slice with nulls — confirms cross-type fix)

- [x] Step 1: Write failing tests

In `test/test_str.c`:

```c
static MunitResult test_str_vec_slice_null(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "world", 5);
    v = td_str_vec_append(v, "foo", 3);
    v = td_str_vec_append(v, "bar", 3);
    td_vec_set_null(v, 1, true);
    td_vec_set_null(v, 3, true);

    /* Slice [1..3) — includes null at parent index 1 */
    td_t* s = td_vec_slice(v, 1, 2);
    munit_assert_not_null(s);
    munit_assert_false(TD_IS_ERR(s));

    /* Slice index 0 = parent index 1 = null */
    munit_assert_true(td_vec_is_null(s, 0));
    /* Slice index 1 = parent index 2 = not null */
    munit_assert_false(td_vec_is_null(s, 1));

    td_release(s);
    td_release(v);
    return MUNIT_OK;
}
```

In `test/test_vec.c`, add a similar test for a generic type (e.g. TD_I64):

```c
static MunitResult test_vec_slice_null(const void* params, void* fixture) {
    (void)params; (void)fixture;
    int64_t vals[] = {10, 20, 30, 40};
    td_t* v = td_vec_from_raw(TD_I64, vals, 4);
    td_vec_set_null(v, 1, true);
    td_vec_set_null(v, 3, true);

    td_t* s = td_vec_slice(v, 1, 2);
    munit_assert_not_null(s);
    munit_assert_false(TD_IS_ERR(s));

    /* Slice index 0 = parent index 1 = null */
    munit_assert_true(td_vec_is_null(s, 0));
    /* Slice index 1 = parent index 2 = not null */
    munit_assert_false(td_vec_is_null(s, 1));

    td_release(s);
    td_release(v);
    return MUNIT_OK;
}
```

Register both in their respective test arrays.

- [x] Step 2: Run tests to verify they fail

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_vec_is_null(s, 0)` returns false

- [x] Step 3: Implement slice redirect in td_vec_is_null

In `src/vec/vec.c`, at the top of `td_vec_is_null` (line 782), add a slice redirect before the `TD_ATTR_HAS_NULLS` check:

```c
bool td_vec_is_null(td_t* vec, int64_t idx) {
    if (!vec || TD_IS_ERR(vec)) return false;
    if (idx < 0 || idx >= vec->len) return false;

    /* Slice: delegate to parent with adjusted index */
    if (vec->attrs & TD_ATTR_SLICE) {
        td_t* parent = vec->slice_parent;
        int64_t pidx = vec->slice_offset + idx;
        return td_vec_is_null(parent, pidx);
    }

    if (!(vec->attrs & TD_ATTR_HAS_NULLS)) return false;

    if (vec->attrs & TD_ATTR_NULLMAP_EXT) {
        td_t* ext = vec->ext_nullmap;
        int64_t byte_idx = idx / 8;
        if (byte_idx >= ext->len) return false;
        uint8_t* bits = (uint8_t*)td_data(ext);
        return (bits[byte_idx] >> (idx % 8)) & 1;
    }

    /* Inline nullmap — not available for TD_STR (bytes 0-15 hold str_pool) */
    if (vec->type == TD_STR) return false;
    if (idx >= 128) return false;
    int byte_idx = (int)(idx / 8);
    int bit_idx = (int)(idx % 8);
    return (vec->nullmap[byte_idx] >> bit_idx) & 1;
}
```

- [x] Step 4: Run tests to verify they pass

Run: `cd build && cmake --build . && ./test_teide --suite /str && ./test_teide --suite /vec`
Expected: PASS

- [x] Step 5: Run full suite

Run: `cd build && ctest --output-on-failure`
Expected: All pass

- [x] Step 6: Commit

```bash
git add src/vec/vec.c test/test_str.c test/test_vec.c
git commit -m "fix: td_vec_is_null redirects through slices to parent — fixes null loss on slice concat"
```

---

### Task 2: Fix td_str_vec_compact buffer overflow on dead counter saturation

When `str_pool_add_dead` saturates at `UINT32_MAX`, compaction computes `live_size = pool_used - UINT32_MAX`. The guard clamps `dead` to `pool_used`, producing `live_size = 0`. Then `td_alloc(0)` returns a minimum block (64 bytes capacity), but the compaction loop copies all live strings into it — heap buffer overflow.

**Files:**
- Modify: `src/vec/vec.c:733-760` (td_str_vec_compact)
- Test: `test/test_str.c`

- [x] Step 1: Write test for compact with saturated dead counter
- [x] Step 2: Implement fix — compute true live size by scanning elements
- [x] Step 3: Run tests
- [x] Step 4: Commit

---

### Task 3: Fix atom_to_str_t OOB read on empty TD_STR vector

A length-0 TD_STR vector classified as scalar reads `elems[0]` without bounds check.

**Files:**
- Modify: `src/ops/exec.c:1375-1380` (atom_to_str_t)
- Test: `test/test_exec.c`

- [x] Step 1: Write test

```c
static MunitResult test_exec_str_eq_empty_vec_scalar(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Table with a TD_STR column and a length-0 TD_STR "scalar" */
    td_t* tbl = td_table_new(2, 3);
    td_table_set_name(tbl, 0, "name");
    td_table_set_name(tbl, 1, "empty");

    td_t* c0 = td_vec_new(TD_STR, 3);
    c0 = td_str_vec_append(c0, "alice", 5);
    c0 = td_str_vec_append(c0, "bob", 3);
    c0 = td_str_vec_append(c0, "", 0);
    td_table_set_col(tbl, 0, c0);
    td_release(c0);

    /* Length-0 TD_STR vector — should broadcast as empty string */
    td_t* c1 = td_vec_new(TD_STR, 0);
    td_table_set_col(tbl, 1, c1);
    td_release(c1);

    td_graph_t* g = td_graph_new(tbl);
    td_op_t* name = td_scan(g, "name");
    td_op_t* empty = td_scan(g, "empty");
    td_op_t* eq = td_eq(g, name, empty);
    td_t* result = td_execute(g, eq);

    /* Should not crash — either returns error or produces result */
    if (!TD_IS_ERR(result)) {
        munit_assert_int(result->type, ==, TD_BOOL);
        td_release(result);
    }

    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

- [x] Step 2: Implement fix

In `src/ops/exec.c`, add a bounds check in `atom_to_str_t` for the TD_STR case (line 1375):

```c
    } else if (atom->type == TD_STR) {
        /* Length-1 TD_STR vector used as scalar */
        if (atom->len < 1) {
            memset(out, 0, sizeof(td_str_t));
            *out_pool = NULL;
            return;
        }
        const td_str_t* elems = (const td_str_t*)td_data(atom);
        *out = elems[0];
        *out_pool = atom->str_pool ? (const char*)td_data(atom->str_pool) : NULL;
        return;
```

- [x] Step 3: Run tests

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS — no crash on empty TD_STR scalar

- [x] Step 4: Run full suite

Run: `cd build && ctest --output-on-failure`
Expected: All pass

- [x] Step 5: Commit

```bash
git add src/ops/exec.c test/test_exec.c
git commit -m "fix: atom_to_str_t bounds check — empty TD_STR vector returns empty string instead of OOB read"
```

---

### Task 4: Add comment for unretained SYM atom pointer in atom_to_str_t

Not a bug today (sym table is append-only) but fragile. Document the lifetime dependency.

**Files:**
- Modify: `src/ops/exec.c:1381-1384`

**Step 1: Add comment**

```c
    } else if (TD_IS_SYM(atom->type) && td_is_atom(atom)) {
        /* SAFETY: td_sym_str returns a borrowed pointer into the append-only
         * sym table.  The pointer is valid for the lifetime of the sym table
         * (i.e., the entire query execution).  If the sym table ever gains
         * eviction, this must retain the returned atom. */
        td_t* s = td_sym_str(atom->i64);
        sp = s ? td_str_ptr(s) : "";
        sl = s ? td_str_len(s) : 0;
```

**Step 2: Commit**

```bash
git add src/ops/exec.c
git commit -m "docs: annotate unretained sym pointer lifetime in atom_to_str_t"
```

---

## Summary

| Task | Bug | Severity | Fix |
|------|-----|----------|-----|
| 1 | `td_vec_is_null` no slice redirect | Critical (all types) | Add slice redirect at top of function |
| 2 | Compact overflow on dead saturation | Important | Scan elements for true live size |
| 3 | `atom_to_str_t` OOB on empty TD_STR | Important | Bounds check `atom->len < 1` |
| 4 | Unretained SYM pointer | Documentation | Comment explaining lifetime |
