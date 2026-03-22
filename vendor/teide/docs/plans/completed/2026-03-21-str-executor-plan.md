# TD_STR Executor Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full TD_STR support to the executor, optimizer, DAG construction, and vec operations — parity with TD_SYM for all string operations.

**Architecture:** Each opcode that handles TD_SYM gets a parallel TD_STR branch. Comparisons use DuckDB-style 16-byte `td_str_t` comparison (len + prefix fast rejection, pool fallback). String ops read via `str_elem()` helper and produce TD_STR output via `td_str_vec_append`. Group-by/sort use hash-based and comparison-based approaches respectively.

**Tech Stack:** Pure C17, no external deps. Tests use munit framework. Build with `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`. Run tests with `./build/test_teide --suite /str` or `cd build && ctest --output-on-failure`.

**Design doc:** `docs/plans/2026-03-21-str-executor-design.md`

---

### Task 1: Slice and concat for TD_STR vectors [DONE]

Currently `td_vec_slice` and `td_vec_concat` reject TD_STR. These are needed before executor integration because the executor uses them internally.

**Files:**
- Modify: `src/vec/vec.c:189-217` (td_vec_slice)
- Modify: `src/vec/vec.c:223-284` (td_vec_concat)
- Test: `test/test_str.c`

**Step 1: Write failing tests**

Add to `test/test_str.c`:

```c
static MunitResult test_str_vec_slice(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_append(v, "world", 5);
    v = td_str_vec_append(v, "another long string!!", 21);

    td_t* s = td_vec_slice(v, 1, 2);
    munit_assert_not_null(s);
    munit_assert_false(TD_IS_ERR(s));
    munit_assert_int64(td_len(s), ==, 2);

    /* Read through slice — inline string */
    size_t len;
    const char* p0 = td_str_vec_get(s, 0, &len);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, p0, "this is a long string!");

    const char* p1 = td_str_vec_get(s, 1, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, p1, "world");

    td_release(s);
    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_concat_vecs(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* a = td_vec_new(TD_STR, 2);
    a = td_str_vec_append(a, "hello", 5);
    a = td_str_vec_append(a, "this is long string a!", 22);

    td_t* b = td_vec_new(TD_STR, 2);
    b = td_str_vec_append(b, "world", 5);
    b = td_str_vec_append(b, "this is long string b!", 22);

    td_t* c = td_vec_concat(a, b);
    munit_assert_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int64(td_len(c), ==, 4);

    size_t len;
    const char* p0 = td_str_vec_get(c, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, p0, "hello");

    const char* p1 = td_str_vec_get(c, 1, &len);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, p1, "this is long string a!");

    const char* p2 = td_str_vec_get(c, 2, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, p2, "world");

    const char* p3 = td_str_vec_get(c, 3, &len);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, p3, "this is long string b!");

    td_release(c);
    td_release(a);
    td_release(b);
    return MUNIT_OK;
}
```

Register in `str_tests[]` array.

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — slice/concat return TD_ERR_TYPE for TD_STR

**Step 3: Implement TD_STR slice**

In `src/vec/vec.c`, the `td_vec_slice` function (line 191) currently has `if (vec->type == TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);`. Remove this guard. The existing slice logic creates a header-only block with `TD_ATTR_SLICE` flag and stores parent + offset. This works for TD_STR because:
- Elements are fixed-size (16 bytes each) — offset arithmetic works
- Pool offsets in elements still resolve against parent's pool

But `td_str_vec_get` needs to handle slices. Update it to check `TD_ATTR_SLICE` and redirect to the parent:

```c
const char* td_str_vec_get(td_t* vec, int64_t idx, size_t* out_len) {
    if (out_len) *out_len = 0;
    if (!vec || TD_IS_ERR(vec) || vec->type != TD_STR) return NULL;
    if (idx < 0 || idx >= vec->len) return NULL;

    /* Slice: redirect to parent */
    td_t* data_owner = vec;
    int64_t data_idx = idx;
    if (vec->attrs & TD_ATTR_SLICE) {
        data_owner = vec->slice_parent;
        data_idx = vec->slice_offset + idx;
    }

    const td_str_t* elem = &((const td_str_t*)td_data(data_owner))[data_idx];
    if (out_len) *out_len = elem->len;

    if (elem->len == 0) return "";
    if (td_str_is_inline(elem)) return elem->data;
    if (!data_owner->str_pool) return NULL;
    return (const char*)td_data(data_owner->str_pool) + elem->pool_off;
}
```

Also remove the TD_STR guard from `td_vec_slice` (line 191).

**Step 4: Implement TD_STR concat**

Remove the TD_STR guard from `td_vec_concat` (line 228). Add a new branch after the existing concat logic:

```c
    if (a->type == TD_STR) {
        /* Concat TD_STR vectors: copy elements, merge pools */
        td_t* result = td_vec_new(TD_STR, total_len);
        if (!result || TD_IS_ERR(result)) return result;
        result->len = total_len;

        td_str_t* dst = (td_str_t*)td_data(result);
        const td_str_t* a_elems = (const td_str_t*)td_data(a);
        const td_str_t* b_elems = (const td_str_t*)td_data(b);

        /* Copy a's elements as-is */
        memcpy(dst, a_elems, (size_t)a->len * sizeof(td_str_t));

        /* Merge pools: a's pool + b's pool */
        int64_t a_pool_size = (a->str_pool) ? a->str_pool->len : 0;
        int64_t b_pool_size = (b->str_pool) ? b->str_pool->len : 0;
        int64_t total_pool = a_pool_size + b_pool_size;

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
                memcpy(pool_dst, td_data(a->str_pool), (size_t)a_pool_size);
            if (b_pool_size > 0)
                memcpy(pool_dst + a_pool_size, td_data(b->str_pool), (size_t)b_pool_size);
        }

        /* Copy b's elements, rebasing pool offsets */
        for (int64_t i = 0; i < b->len; i++) {
            dst[a->len + i] = b_elems[i];
            if (!td_str_is_inline(&b_elems[i]) && b_elems[i].len > 0) {
                dst[a->len + i].pool_off += (uint32_t)a_pool_size;
            }
        }

        return result;
    }
```

Insert this early in `td_vec_concat`, before the existing type-mismatch check works.

**Step 5: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 6: Run all tests**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 7: Commit**

```bash
git add src/vec/vec.c test/test_str.c
git commit -m "feat: TD_STR slice and concat — zero-copy slices, pool-merging concat"
```

---

### Task 2: td_str_t_hash and str_elem helper in td.h [DONE]

These shared helpers are needed by multiple executor tasks. Add them first.

**Files:**
- Modify: `include/teide/td.h` (add td_str_t_hash, str_resolve helper)
- Test: `test/test_str.c`

**Step 1: Write failing test**

```c
static MunitResult test_str_t_hash_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "world", 5);

    td_str_t* elems = (td_str_t*)td_data(v);
    uint64_t h0 = td_str_t_hash(&elems[0], NULL);
    uint64_t h1 = td_str_t_hash(&elems[1], NULL);
    uint64_t h2 = td_str_t_hash(&elems[2], NULL);

    /* Same strings → same hash */
    munit_assert_uint64(h0, ==, h1);
    /* Different strings → different hash (extremely likely) */
    munit_assert_uint64(h0, !=, h2);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_t_hash_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_append(v, "different long string!", 22);

    td_str_t* elems = (td_str_t*)td_data(v);
    const char* pool = (const char*)td_data(v->str_pool);
    uint64_t h0 = td_str_t_hash(&elems[0], pool);
    uint64_t h1 = td_str_t_hash(&elems[1], pool);
    uint64_t h2 = td_str_t_hash(&elems[2], pool);

    munit_assert_uint64(h0, ==, h1);
    munit_assert_uint64(h0, !=, h2);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_t_hash_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 2);
    v = td_str_vec_append(v, "", 0);
    v = td_str_vec_append(v, "", 0);

    td_str_t* elems = (td_str_t*)td_data(v);
    uint64_t h0 = td_str_t_hash(&elems[0], NULL);
    uint64_t h1 = td_str_t_hash(&elems[1], NULL);
    munit_assert_uint64(h0, ==, h1);

    td_release(v);
    return MUNIT_OK;
}
```

Register in `str_tests[]`.

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: FAIL — `td_str_t_hash` not defined

**Step 3: Implement td_str_t_hash**

In `include/teide/td.h`, after the `td_str_t_cmp` function, add:

```c
/* Hash a td_str_t element using wyhash */
static inline uint64_t td_str_t_hash(const td_str_t* s, const char* pool_base) {
    if (s->len == 0) return 0x9E3779B97F4A7C15ULL; /* golden ratio constant for empty */
    const char* p = td_str_is_inline(s) ? s->data : pool_base + s->pool_off;
    return td_wyhash(p, s->len, 0);
}
```

Note: Check that `td_wyhash` is declared in td.h (the sym table already uses wyhash). If the wyhash function has a different name, use that. Search for `wyhash` in the codebase to find the correct function name.

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /str`
Expected: PASS

**Step 5: Commit**

```bash
git add include/teide/td.h test/test_str.c
git commit -m "feat: td_str_t_hash — wyhash for inline and pooled strings"
```

---

### Task 3: Binary comparisons (OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE) for TD_STR [DONE]

The core DuckDB-style comparison path.

**Files:**
- Modify: `src/ops/exec.c:1365-1467` (binary_range / exec_binary)
- Modify: `src/ops/exec.c:1490-1579` (exec_binary — scalar resolution)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

Add a `make_str_table()` helper and test to `test/test_exec.c`:

```c
static td_t* make_str_table(void) {
    /* Build a table with TD_STR "name" column */
    td_t* tbl = td_table_new(1, 5);
    td_table_set_name(tbl, 0, "name");

    td_t* col = td_vec_new(TD_STR, 5);
    col = td_str_vec_append(col, "hello", 5);
    col = td_str_vec_append(col, "WORLD", 5);
    col = td_str_vec_append(col, "  foo  ", 7);
    col = td_str_vec_append(col, "bar_baz", 7);
    col = td_str_vec_append(col, "", 0);

    td_table_set_col(tbl, 0, col);
    td_release(col);
    return tbl;
}

static MunitResult test_exec_str_eq(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* lit = td_const_str(g, "hello");
    td_op_t* eq = td_eq(g, name, lit);

    td_t* result = td_execute(g, eq);
    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_BOOL);
    munit_assert_int(result->len, ==, 5);

    uint8_t* data_p = (uint8_t*)td_data(result);
    munit_assert_uint8(data_p[0], ==, 1);  /* "hello" == "hello" */
    munit_assert_uint8(data_p[1], ==, 0);  /* "WORLD" != "hello" */
    munit_assert_uint8(data_p[2], ==, 0);
    munit_assert_uint8(data_p[3], ==, 0);
    munit_assert_uint8(data_p[4], ==, 0);

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

Register in the exec test suite.

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: FAIL — TD_STR column not handled in binary comparison path

**Step 3: Implement TD_STR comparison in exec_binary**

The key change is in `exec_binary` (around line 1490). Before the existing scalar resolution logic, add a TD_STR fast path:

```c
    /* TD_STR comparison: operate directly on td_str_t arrays */
    bool l_is_str = (!l_scalar && lhs->type == TD_STR);
    bool r_is_str = (!r_scalar && rhs->type == TD_STR);
    bool l_scalar_str = (l_scalar && lhs->type == TD_ATOM_STR);
    bool r_scalar_str = (r_scalar && rhs->type == TD_ATOM_STR);

    if (l_is_str || r_is_str || (l_scalar_str && r_is_str) || (r_scalar_str && l_is_str)) {
        /* Build scalar td_str_t on stack if needed */
        td_str_t l_tmp = {0}, r_tmp = {0};
        if (l_scalar_str) {
            const char* sp = td_str_ptr(lhs);
            size_t sl = td_str_len(lhs);
            l_tmp.len = (uint32_t)sl;
            if (sl <= TD_STR_INLINE_MAX) {
                if (sl > 0) memcpy(l_tmp.data, sp, sl);
            } else {
                memcpy(l_tmp.prefix, sp, 4);
                /* Store pointer offset — use a side pointer */
            }
        }
        /* Similar for r_scalar_str */

        /* ... (see implementation note below) */
    }
```

**Implementation note:** The full TD_STR binary path is complex. The approach is:

1. In `exec_binary` (line ~1490), detect when either side is TD_STR or TD_ATOM_STR-vs-TD_STR
2. Build the morsel loop that calls `td_str_t_eq`/`td_str_t_cmp` per element
3. For scalar TD_ATOM_STR vs TD_STR column: resolve atom to string, build temp `td_str_t`, compare
4. For TD_SYM vs TD_STR mixed: resolve SYM elements to strings via `td_sym_str()`, build temp `td_str_t`s

The implementer should study the existing `binary_range` function (line 1365-1467) and add a separate `binary_range_str` function that operates on `td_str_t*` arrays instead of numeric pointers:

```c
static void binary_range_str(td_op_t* op, td_t* lhs, td_t* rhs, td_t* result,
                             bool l_scalar, bool r_scalar,
                             int64_t start, int64_t n) {
    uint8_t* dst = (uint8_t*)td_data(result) + start;
    uint16_t opc = op->opcode;

    const td_str_t* l_elems = l_scalar ? NULL : (const td_str_t*)td_data(lhs) + start;
    const td_str_t* r_elems = r_scalar ? NULL : (const td_str_t*)td_data(rhs) + start;
    const char* l_pool = (!l_scalar && lhs->str_pool) ? (const char*)td_data(lhs->str_pool) : NULL;
    const char* r_pool = (!r_scalar && rhs->str_pool) ? (const char*)td_data(rhs->str_pool) : NULL;

    /* For scalar side, build a single td_str_t */
    td_str_t l_scalar_str = {0}, r_scalar_str = {0};
    const char* l_scalar_pool = NULL;
    const char* r_scalar_pool = NULL;
    if (l_scalar) {
        /* Resolve lhs atom to td_str_t — see atom_to_str_t helper */
        atom_to_str_t(lhs, &l_scalar_str, &l_scalar_pool);
        l_elems = &l_scalar_str;
    }
    if (r_scalar) {
        atom_to_str_t(rhs, &r_scalar_str, &r_scalar_pool);
        r_elems = &r_scalar_str;
    }

    for (int64_t i = 0; i < n; i++) {
        const td_str_t* a = l_scalar ? l_elems : &l_elems[i];
        const td_str_t* b = r_scalar ? r_elems : &r_elems[i];
        const char* pa = l_scalar ? l_scalar_pool : l_pool;
        const char* pb = r_scalar ? r_scalar_pool : r_pool;

        switch (opc) {
            case OP_EQ: dst[i] = td_str_t_eq(a, pa, b, pb); break;
            case OP_NE: dst[i] = !td_str_t_eq(a, pa, b, pb); break;
            case OP_LT: dst[i] = td_str_t_cmp(a, pa, b, pb) < 0; break;
            case OP_LE: dst[i] = td_str_t_cmp(a, pa, b, pb) <= 0; break;
            case OP_GT: dst[i] = td_str_t_cmp(a, pa, b, pb) > 0; break;
            case OP_GE: dst[i] = td_str_t_cmp(a, pa, b, pb) >= 0; break;
            default: dst[i] = 0; break;
        }
    }
}
```

Also add a helper to convert atoms/SYM scalars to `td_str_t`:

```c
static void atom_to_str_t(td_t* atom, td_str_t* out, const char** out_pool) {
    const char* sp;
    size_t sl;
    if (atom->type == TD_ATOM_STR) {
        sp = td_str_ptr(atom);
        sl = td_str_len(atom);
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
        *out_pool = sp; /* point directly at atom's string data */
    }
}
```

Wire this into `exec_binary`: before the existing scalar resolution code, check if any side is TD_STR and divert to `binary_range_str`.

**Step 4: Run tests to verify they pass**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 6: Commit**

```bash
git add src/ops/exec.c test/test_exec.c
git commit -m "feat: TD_STR binary comparisons — DuckDB-style 16-byte fast path"
```

---

### Task 4: STRLEN for TD_STR [DONE]

Simplest string op — just reads `elem->len`.

**Files:**
- Modify: `src/ops/exec.c:10300-10320` (exec_strlen)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_strlen(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* slen = td_strlen(g, name);
    td_t* result = td_execute(g, slen);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_I64);
    munit_assert_int(result->len, ==, 5);
    int64_t* d = (int64_t*)td_data(result);
    munit_assert_int64(d[0], ==, 5);   /* "hello" */
    munit_assert_int64(d[1], ==, 5);   /* "WORLD" */
    munit_assert_int64(d[2], ==, 7);   /* "  foo  " */
    munit_assert_int64(d[3], ==, 7);   /* "bar_baz" */
    munit_assert_int64(d[4], ==, 0);   /* "" */

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run tests to verify they fail**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: FAIL — exec_strlen doesn't handle TD_STR input

**Step 3: Implement**

In `exec_strlen` (around line 10300), the current code calls `sym_elem()` which only works for TD_SYM. Add a TD_STR branch:

```c
    for (int64_t i = 0; i < len; i++) {
        if (input->type == TD_STR) {
            td_str_t* elems = (td_str_t*)td_data(input);
            dst[i] = (int64_t)elems[i].len;
        } else {
            const char* sp; size_t sl;
            sym_elem(input, i, &sp, &sl);
            dst[i] = (int64_t)sl;
        }
    }
```

Or more efficiently, hoist the type check outside the loop.

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/exec.c test/test_exec.c
git commit -m "feat: STRLEN for TD_STR — reads len directly from td_str_t element"
```

---

### Task 5: UPPER / LOWER / TRIM for TD_STR [DONE]

**Files:**
- Modify: `src/ops/exec.c:10260-10299` (exec_string_unary)
- Test: `test/test_exec.c`

**Step 1: Write failing tests**

```c
static MunitResult test_exec_str_upper(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* up = td_upper(g, name);
    td_t* result = td_execute(g, up);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_STR);
    munit_assert_int(result->len, ==, 5);

    size_t len;
    const char* s0 = td_str_vec_get(result, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, s0, "HELLO");

    const char* s4 = td_str_vec_get(result, 4, &len);
    munit_assert_size(len, ==, 0);

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_exec_str_trim(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* tr = td_trim_op(g, name);
    td_t* result = td_execute(g, tr);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_STR);

    size_t len;
    const char* s2 = td_str_vec_get(result, 2, &len);
    munit_assert_size(len, ==, 3);
    munit_assert_memory_equal(3, s2, "foo");

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run to verify failure**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: FAIL

**Step 3: Implement TD_STR path in exec_string_unary**

The current `exec_string_unary` creates a TD_SYM output and uses `td_sym_intern()`. For TD_STR input, create a TD_STR output and use `td_str_vec_append()`:

```c
static td_t* exec_string_unary(td_graph_t* g, td_op_t* op) {
    td_t* input = exec_node(g, op->inputs[0]);
    if (!input || TD_IS_ERR(input)) return input;

    int64_t len = input->len;
    bool is_str = (input->type == TD_STR);

    td_t* result;
    if (is_str) {
        result = td_vec_new(TD_STR, len);
    } else {
        result = td_vec_new(TD_SYM, len);
    }
    if (!result || TD_IS_ERR(result)) { td_release(input); return result; }

    uint16_t opc = op->opcode;
    for (int64_t i = 0; i < len; i++) {
        const char* sp; size_t sl;
        if (is_str) {
            td_str_t* elems = (td_str_t*)td_data(input);
            const char* pool = input->str_pool ? (const char*)td_data(input->str_pool) : NULL;
            sp = td_str_t_ptr(&elems[i], pool);
            sl = elems[i].len;
        } else {
            sym_elem(input, i, &sp, &sl);
        }

        char sbuf[8192];
        char* buf = sbuf;
        td_t* dyn_hdr = NULL;
        if (sl >= sizeof(sbuf)) {
            buf = (char*)scratch_alloc(&dyn_hdr, sl + 1);
            if (!buf) {
                if (is_str) { result = td_str_vec_append(result, "", 0); }
                else { ((int64_t*)td_data(result))[i] = td_sym_intern("", 0); result->len = i + 1; }
                continue;
            }
        }
        size_t out_len = sl;
        if (opc == OP_UPPER) {
            for (size_t j = 0; j < out_len; j++) buf[j] = (char)toupper((unsigned char)sp[j]);
        } else if (opc == OP_LOWER) {
            for (size_t j = 0; j < out_len; j++) buf[j] = (char)tolower((unsigned char)sp[j]);
        } else { /* OP_TRIM */
            size_t start = 0, end = sl;
            while (start < sl && isspace((unsigned char)sp[start])) start++;
            while (end > start && isspace((unsigned char)sp[end - 1])) end--;
            out_len = end - start;
            memcpy(buf, sp + start, out_len);
        }

        if (is_str) {
            result = td_str_vec_append(result, buf, out_len);
        } else {
            buf[out_len] = '\0';
            ((int64_t*)td_data(result))[i] = td_sym_intern(buf, out_len);
            result->len = i + 1;
        }
        scratch_free(dyn_hdr);
    }

    if (!is_str) result->len = len;
    td_release(input);
    return result;
}
```

**Step 4: Update DAG construction for TD_STR output type**

In `src/ops/graph.c`, the functions `td_upper()`, `td_lower()`, `td_trim_op()` hardcode `TD_SYM` output. They should propagate the input type. Change:

```c
td_op_t* td_upper(td_graph_t* g, td_op_t* a)   { return make_unary(g, OP_UPPER, a, a->out_type == TD_STR ? TD_STR : TD_SYM); }
td_op_t* td_lower(td_graph_t* g, td_op_t* a)   { return make_unary(g, OP_LOWER, a, a->out_type == TD_STR ? TD_STR : TD_SYM); }
td_op_t* td_trim_op(td_graph_t* g, td_op_t* a) { return make_unary(g, OP_TRIM, a, a->out_type == TD_STR ? TD_STR : TD_SYM); }
```

**Step 5: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 6: Commit**

```bash
git add src/ops/exec.c src/ops/graph.c test/test_exec.c
git commit -m "feat: UPPER/LOWER/TRIM for TD_STR — produces TD_STR output"
```

---

### Task 6: SUBSTR and REPLACE for TD_STR [DONE]

**Files:**
- Modify: `src/ops/exec.c` (exec_substr, exec_replace)
- Modify: `src/ops/graph.c` (output type propagation)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_substr(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* start = td_const_i64(g, 1);
    td_op_t* len_op = td_const_i64(g, 3);
    td_op_t* sub = td_substr(g, name, start, len_op);
    td_t* result = td_execute(g, sub);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_STR);
    munit_assert_int(result->len, ==, 5);

    size_t len;
    const char* s0 = td_str_vec_get(result, 0, &len);
    munit_assert_size(len, ==, 3);
    munit_assert_memory_equal(3, s0, "hel");

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run to verify failure**

Expected: FAIL

**Step 3: Implement TD_STR path in exec_substr and exec_replace**

Follow the same pattern as Task 5: check `input->type == TD_STR`, read via `td_str_t_ptr()`, write via `td_str_vec_append()`. Update graph.c to propagate TD_STR output type for `td_substr` and `td_replace`.

In `src/ops/graph.c`, change the output type for substr and replace:

```c
ext->base.out_type = (str->out_type == TD_STR) ? TD_STR : TD_SYM;
```

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/exec.c src/ops/graph.c test/test_exec.c
git commit -m "feat: SUBSTR/REPLACE for TD_STR — substring and replacement with TD_STR output"
```

---

### Task 7: CONCAT for TD_STR [DONE]

**Files:**
- Modify: `src/ops/exec.c:10486-10540` (exec_concat)
- Modify: `src/ops/graph.c` (concat output type)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_concat(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Table with two TD_STR columns */
    td_t* tbl = td_table_new(2, 3);
    td_table_set_name(tbl, 0, "first");
    td_table_set_name(tbl, 1, "last");

    td_t* c0 = td_vec_new(TD_STR, 3);
    c0 = td_str_vec_append(c0, "John", 4);
    c0 = td_str_vec_append(c0, "Jane", 4);
    c0 = td_str_vec_append(c0, "Bob", 3);
    td_table_set_col(tbl, 0, c0);
    td_release(c0);

    td_t* c1 = td_vec_new(TD_STR, 3);
    c1 = td_str_vec_append(c1, " Doe", 4);
    c1 = td_str_vec_append(c1, " Smith", 6);
    c1 = td_str_vec_append(c1, " Jr", 3);
    td_table_set_col(tbl, 1, c1);
    td_release(c1);

    td_graph_t* g = td_graph_new(tbl);
    td_op_t* first = td_scan(g, "first");
    td_op_t* last = td_scan(g, "last");
    td_op_t* cat = td_concat(g, first, last);
    td_t* result = td_execute(g, cat);

    munit_assert_false(TD_IS_ERR(result));
    munit_assert_int(result->type, ==, TD_STR);
    munit_assert_int(result->len, ==, 3);

    size_t len;
    const char* s0 = td_str_vec_get(result, 0, &len);
    munit_assert_size(len, ==, 8);
    munit_assert_memory_equal(8, s0, "John Doe");

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run to verify failure**

Expected: FAIL

**Step 3: Implement TD_STR concat in executor**

In `exec_concat`, detect if any arg is TD_STR. If so, output is TD_STR. Read each arg via the appropriate helper (TD_STR: `td_str_t_ptr()`, TD_SYM: `sym_elem()`, TD_ATOM_STR: `td_str_ptr()`). Concatenate in stack buffer, `td_str_vec_append()` to result.

Update graph.c `td_concat()` to set output type to TD_STR when any input is TD_STR.

**Step 4: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/exec.c src/ops/graph.c test/test_exec.c
git commit -m "feat: CONCAT for TD_STR — mixed TD_STR/TD_SYM inputs produce TD_STR"
```

---

### Task 8: OP_IF for TD_STR [DONE]

**Files:**
- Modify: `src/ops/exec.c:10041-10067` (exec_if TD_SYM branch)
- Modify: `src/ops/graph.c:458-460` (td_if output type)
- Test: `test/test_exec.c`

**Step 1: Write failing test**

```c
static MunitResult test_exec_str_if(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    td_op_t* name = td_scan(g, "name");
    td_op_t* lit = td_const_str(g, "hello");
    td_op_t* cond = td_eq(g, name, lit);
    td_op_t* then_v = td_const_str(g, "YES");
    td_op_t* else_v = td_const_str(g, "NO");
    td_op_t* if_op = td_if(g, cond, then_v, else_v);
    td_t* result = td_execute(g, if_op);

    munit_assert_false(TD_IS_ERR(result));
    /* Result could be TD_SYM (both branches are const_str which is SYM) */
    /* Just verify values are correct */
    munit_assert_int(result->len, ==, 5);

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Implement**

In `exec_if`, add a TD_STR branch after the existing TD_SYM branch. When `out_type == TD_STR`:
- Create output via `td_vec_new(TD_STR, len)`
- For each row, resolve the then/else value (could be TD_STR element, TD_SYM element, or TD_ATOM_STR scalar) and `td_str_vec_append()` the selected value

In graph.c, update `td_if` output type resolution:
```c
if (then_val->out_type == TD_STR || else_val->out_type == TD_STR)
    out_type = TD_STR;
else if (then_val->out_type == TD_SYM || else_val->out_type == TD_SYM)
    out_type = TD_SYM;
```

**Step 3: Run tests**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 4: Commit**

```bash
git add src/ops/exec.c src/ops/graph.c test/test_exec.c
git commit -m "feat: OP_IF for TD_STR — conditional with string branches"
```

---

### Task 9: Type inference for TD_STR in optimizer

**Files:**
- Modify: `src/ops/opt.c:48-58` (promote_type)
- Modify: `src/ops/graph.c:401-407` (promote)
- Test: `test/test_opt.c`

**Step 1: Update promote_type functions**

In both `src/ops/opt.c` (line 48) and `src/ops/graph.c` (line 402), the `promote_type`/`promote` functions treat TD_SYM as integer-class. TD_STR should NOT be promoted to I64. Add early returns:

```c
static int8_t promote_type(int8_t a, int8_t b) {
    /* TD_STR stays TD_STR — never promoted to numeric */
    if (a == TD_STR || b == TD_STR) return TD_STR;
    if (a == TD_F64 || b == TD_F64) return TD_F64;
    /* ... rest unchanged ... */
}
```

Same change in graph.c's `promote()`.

**Step 2: Run all tests**

Run: `cd build && ctest --output-on-failure`
Expected: All pass — existing TD_SYM behavior unchanged, TD_STR propagates correctly

**Step 3: Commit**

```bash
git add src/ops/opt.c src/ops/graph.c
git commit -m "feat: type inference treats TD_STR as its own class — no numeric promotion"
```

---

### Task 10: Full integration smoke test

End-to-end test combining multiple TD_STR operations in a single pipeline.

**Files:**
- Test: `test/test_exec.c`

**Step 1: Write integration test**

```c
static MunitResult test_exec_str_pipeline(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();
    td_t* tbl = make_str_table();
    td_graph_t* g = td_graph_new(tbl);

    /* Pipeline: UPPER(name) where STRLEN(name) > 3 */
    td_op_t* name = td_scan(g, "name");
    td_op_t* slen = td_strlen(g, name);
    td_op_t* three = td_const_i64(g, 3);
    td_op_t* gt = td_gt(g, slen, three);

    td_op_t* name2 = td_scan(g, "name");
    td_op_t* up = td_upper(g, name2);

    td_op_t* scan = td_scan_table(g);
    td_op_t* filtered = td_filter(g, scan, gt);
    td_t* result = td_execute(g, filtered);

    munit_assert_false(TD_IS_ERR(result));
    /* Should have rows where strlen > 3: "hello"(5), "WORLD"(5), "  foo  "(7), "bar_baz"(7) = 4 rows */
    munit_assert_int(td_table_nrows(result), ==, 4);

    td_release(result);
    td_graph_free(g);
    td_release(tbl);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Run to verify it passes**

Run: `cd build && cmake --build . && ./test_teide --suite /exec`
Expected: PASS

**Step 3: Run full suite**

Run: `cd build && ctest --output-on-failure`
Expected: All pass

**Step 4: Commit**

```bash
git add test/test_exec.c
git commit -m "test: TD_STR end-to-end pipeline — STRLEN filter + UPPER through executor"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Slice + concat for TD_STR vectors | vec.c |
| 2 | td_str_t_hash helper | td.h |
| 3 | Binary comparisons (EQ/NE/LT/LE/GT/GE) | exec.c |
| 4 | STRLEN | exec.c |
| 5 | UPPER / LOWER / TRIM | exec.c, graph.c |
| 6 | SUBSTR / REPLACE | exec.c, graph.c |
| 7 | CONCAT | exec.c, graph.c |
| 8 | OP_IF | exec.c, graph.c |
| 9 | Type inference | opt.c, graph.c |
| 10 | Integration smoke test | test_exec.c |

**Not included (deferred):** Group-by hash table, comparison-based sort, count distinct, col.c serialization, CSV loader. These can be follow-up tasks once the core executor paths are solid.
