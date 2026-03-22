# CSV Null Handling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the CSV loader produce NULL values (with proper validity bitmaps) for fields that cannot be parsed as the inferred column type, instead of silently coercing them to 0/false/empty.

**Architecture:** Add `bool* is_null` output parameters to all fast parsers, bulk-allocate nullmaps before parsing, set bits directly in the hot loop, and strip nullmaps from all-valid columns post-parse.

**Tech Stack:** Pure C17, no new dependencies. Uses existing `td_t` nullmap infrastructure (inline for ≤128 rows, external bitmap for >128).

---

### Task 1: Add Null-Aware Tests for Numeric Columns

**Files:**
- Modify: `test/test_csv.c`

**Step 1: Write failing test — empty i64 fields become NULL**

Add this test after the existing `test_csv_empty_table`:

```c
static MunitResult test_csv_null_i64(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Write CSV with empty fields in integer column */
    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "x\n10\n\n30\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));
    munit_assert_int(td_table_nrows(loaded), ==, 3);

    td_t* col = td_table_get_col_idx(loaded, 0);
    munit_assert_ptr_not_null(col);

    /* Row 0: valid 10 */
    munit_assert_false(td_vec_is_null(col, 0));
    munit_assert_int(((int64_t*)td_data(col))[0], ==, 10);
    /* Row 1: empty → NULL */
    munit_assert_true(td_vec_is_null(col, 1));
    /* Row 2: valid 30 */
    munit_assert_false(td_vec_is_null(col, 2));
    munit_assert_int(((int64_t*)td_data(col))[2], ==, 30);

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Write failing test — unparseable i64 fields become NULL**

```c
static MunitResult test_csv_null_i64_unparseable(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    /* Type inference sees integers in sample, but row 2 has "N/A" */
    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "x\n10\nN/A\n30\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));

    td_t* col = td_table_get_col_idx(loaded, 0);
    munit_assert_false(td_vec_is_null(col, 0));
    munit_assert_int(((int64_t*)td_data(col))[0], ==, 10);
    munit_assert_true(td_vec_is_null(col, 1));
    munit_assert_false(td_vec_is_null(col, 2));
    munit_assert_int(((int64_t*)td_data(col))[2], ==, 30);

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 3: Write failing test — empty f64 fields become NULL**

```c
static MunitResult test_csv_null_f64(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "x\n1.5\n\n3.5\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));

    td_t* col = td_table_get_col_idx(loaded, 0);
    munit_assert_false(td_vec_is_null(col, 0));
    munit_assert_double_equal(((double*)td_data(col))[0], 1.5, 6);
    munit_assert_true(td_vec_is_null(col, 1));
    munit_assert_false(td_vec_is_null(col, 2));
    munit_assert_double_equal(((double*)td_data(col))[2], 3.5, 6);

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 4: Register tests in `csv_tests[]` array**

Add to the `csv_tests[]` array before the `{ NULL, ... }` sentinel:

```c
{ "/null_i64",             test_csv_null_i64,             NULL, NULL, 0, NULL },
{ "/null_i64_unparseable", test_csv_null_i64_unparseable, NULL, NULL, 0, NULL },
{ "/null_f64",             test_csv_null_f64,             NULL, NULL, 0, NULL },
```

**Step 5: Build and run — confirm tests fail**

Run: `cd .worktrees/csv-null-handling && cmake --build build -j$(nproc) && ./build/test_teide --suite /csv`

Expected: The 3 new tests FAIL (null checks return false because no nullmap is set).

**Step 6: Commit**

```bash
git add test/test_csv.c
git commit -m "test: add failing tests for CSV null handling on i64/f64 columns"
```

---

### Task 2: Make Parsers Null-Aware

**Files:**
- Modify: `src/io/csv.c:481-649` (fast_i64, fast_f64, fast_date, fast_time, fast_timestamp)

**Step 1: Add `bool* is_null` to `fast_i64`**

Change signature at line 481 from:
```c
TD_INLINE int64_t fast_i64(const char* p, size_t len) {
```
to:
```c
TD_INLINE int64_t fast_i64(const char* p, size_t len, bool* is_null) {
```

Replace the empty check at line 482:
```c
if (TD_UNLIKELY(len == 0)) return 0;
```
with:
```c
if (TD_UNLIKELY(len == 0)) { *is_null = true; return 0; }
*is_null = false;
```

Add null signaling when digit scan doesn't consume entire field. After the accumulation loop (line 510 `p++; }`), before the 18-digit check, add:
```c
if (p != end) { *is_null = true; return 0; }
```

Also add null signaling in the two `strtoll` fallbacks (lines 501 and 520). Change each:
```c
return strtoll(tmp, NULL, 10);
```
to:
```c
char* endp;
int64_t v = strtoll(tmp, &endp, 10);
if (*endp != '\0') { *is_null = true; return 0; }
return v;
```

**Step 2: Add `bool* is_null` to `fast_f64`**

Change signature at line 540 from:
```c
TD_INLINE double fast_f64(const char* p, size_t len) {
```
to:
```c
TD_INLINE double fast_f64(const char* p, size_t len, bool* is_null) {
```

Replace the empty check at line 541:
```c
if (TD_UNLIKELY(len == 0)) return 0.0;
```
with:
```c
if (TD_UNLIKELY(len == 0)) { *is_null = true; return 0.0; }
*is_null = false;
```

After the exponent parsing section (line 636, after the closing `}`), before `return negative ? -val : val;`, add a check that the entire field was consumed:
```c
if (p != end) { *is_null = true; return 0.0; }
```

In the `strtod_fallback` label (line 640-648), change:
```c
return strtod(tmp, NULL);
```
to:
```c
char* endp;
double v = strtod(tmp, &endp, 10);
if (*endp != '\0') { *is_null = true; return 0.0; }
return v;
```

Note: `strtod` only takes 2 args. The correct call is:
```c
char* endp;
double v = strtod(tmp, &endp);
if (*endp != '\0') { *is_null = true; return 0.0; }
return v;
```

**Step 3: Add `bool* is_null` to `fast_date`**

Change signature at line 672 from:
```c
TD_INLINE int32_t fast_date(const char* p, size_t len) {
```
to:
```c
TD_INLINE int32_t fast_date(const char* p, size_t len, bool* is_null) {
```

Replace `if (TD_UNLIKELY(len < 10)) return 0;` with:
```c
if (TD_UNLIKELY(len < 10)) { *is_null = true; return 0; }
*is_null = false;
```

Replace `if (TD_UNLIKELY(m < 1 || m > 12 || d < 1 || d > 31)) return 0;` with:
```c
if (TD_UNLIKELY(m < 1 || m > 12 || d < 1 || d > 31)) { *is_null = true; return 0; }
```

**Step 4: Add `bool* is_null` to `fast_time`**

Change signature at line 682 from:
```c
TD_INLINE int32_t fast_time(const char* p, size_t len) {
```
to:
```c
TD_INLINE int32_t fast_time(const char* p, size_t len, bool* is_null) {
```

Replace `if (TD_UNLIKELY(len < 8)) return 0;` with:
```c
if (TD_UNLIKELY(len < 8)) { *is_null = true; return 0; }
*is_null = false;
```

Replace `if (TD_UNLIKELY(h > 23 || mi > 59 || s > 59)) return 0;` with:
```c
if (TD_UNLIKELY(h > 23 || mi > 59 || s > 59)) { *is_null = true; return 0; }
```

**Step 5: Add `bool* is_null` to `fast_timestamp`**

Change signature at line 725 from:
```c
TD_INLINE int64_t fast_timestamp(const char* p, size_t len) {
```
to:
```c
TD_INLINE int64_t fast_timestamp(const char* p, size_t len, bool* is_null) {
```

Replace `if (TD_UNLIKELY(len < 19)) return 0;` with:
```c
if (TD_UNLIKELY(len < 19)) { *is_null = true; return 0; }
*is_null = false;
```

Update the internal calls to `fast_date` and `fast_time_us`:
```c
int32_t days = fast_date(p, 10, is_null);
if (*is_null) return 0;
int64_t time_us = fast_time_us(p + 11, len - 11);
```

Note: `fast_time_us` is a separate helper used only inside `fast_timestamp`. It doesn't need its own `is_null` — the range checks already in `fast_date` and the length check in `fast_timestamp` cover the cases. But we should add the same pattern to `fast_time_us` for consistency. Add `bool* is_null` param and handle the `len < 8` and range checks the same way as `fast_time`.

Update `fast_timestamp` to:
```c
TD_INLINE int64_t fast_timestamp(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len < 19)) { *is_null = true; return 0; }
    *is_null = false;
    int32_t days = fast_date(p, 10, is_null);
    if (*is_null) return 0;
    bool time_null = false;
    int64_t time_us = fast_time_us(p + 11, len - 11, &time_null);
    if (time_null) { *is_null = true; return 0; }
    return (int64_t)days * 86400000000LL + time_us;
}
```

**Step 6: Extract `fast_bool`**

Add a new function before the serial parse function (before line 1045):

```c
TD_INLINE uint8_t fast_bool(const char* s, size_t len, bool* is_null) {
    if (len == 0) { *is_null = true; return 0; }
    *is_null = false;
    if ((len == 4 && (memcmp(s, "true", 4) == 0 || memcmp(s, "TRUE", 4) == 0)) ||
        (len == 1 && s[0] == '1'))
        return 1;
    if ((len == 5 && (memcmp(s, "false", 5) == 0 || memcmp(s, "FALSE", 5) == 0)) ||
        (len == 1 && s[0] == '0'))
        return 0;
    *is_null = true;
    return 0;
}
```

**Step 7: Build — confirm it compiles (callers will be updated in Task 3)**

This step won't compile yet because callers still pass the old argument count. That's fine — proceed to Task 3.

**Step 8: Commit**

```bash
git add src/io/csv.c
git commit -m "feat: add is_null output param to all CSV fast parsers"
```

---

### Task 3: Allocate Nullmaps and Wire Into Parse Loops

**Files:**
- Modify: `src/io/csv.c:1275-1398` (column allocation, parse dispatch, serial/parallel paths)

**Step 1: Add nullmap allocation after column vector creation (after line 1293)**

Insert after the column allocation loop (after `col_data[c] = td_data(col_vecs[c]);`):

```c
/* ---- 8b. Pre-allocate nullmaps for all columns ---- */
uint8_t* col_nullmaps[CSV_MAX_COLS];
bool col_had_null[CSV_MAX_COLS];
memset(col_had_null, 0, (size_t)ncols * sizeof(bool));

for (int c = 0; c < ncols; c++) {
    td_t* vec = col_vecs[c];
    if (n_rows <= 128) {
        vec->attrs |= TD_ATTR_HAS_NULLS;
        memset(vec->nullmap, 0, 16);
        col_nullmaps[c] = vec->nullmap;
    } else {
        size_t bmp_bytes = ((size_t)n_rows + 7) / 8;
        td_t* ext = td_vec_new(TD_U8, (int64_t)bmp_bytes);
        if (!ext || TD_IS_ERR(ext)) {
            for (int j = 0; j <= c; j++) td_release(col_vecs[j]);
            goto fail_offsets;
        }
        ext->len = (int64_t)bmp_bytes;
        memset(td_data(ext), 0, bmp_bytes);
        vec->ext_nullmap = ext;
        vec->attrs |= TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT;
        col_nullmaps[c] = (uint8_t*)td_data(ext);
    }
}
```

**Step 2: Update `csv_par_ctx_t` to include nullmaps**

Add to the struct (around line 942):
```c
uint8_t**         col_nullmaps; /* [n_cols] direct pointers to nullmap data */
bool*             worker_had_null; /* [n_workers * n_cols] */
```

**Step 3: Update `csv_parse_fn` (parallel path) — lines 955-1039**

In the parse function, get the nullmap pointers and worker-local null tracking:

After line 962 (my_syms assignment), add:
```c
bool* my_had_null = &ctx->worker_had_null[(size_t)worker_id * (size_t)ctx->n_cols];
```

Update the switch cases in the parse loop (lines 1006-1035):

```c
switch (ctx->col_types[c]) {
    case CSV_TYPE_BOOL: {
        bool is_null;
        uint8_t v = fast_bool(fld, flen, &is_null);
        ((uint8_t*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_I64: {
        bool is_null;
        int64_t v = fast_i64(fld, flen, &is_null);
        ((int64_t*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_F64: {
        bool is_null;
        double v = fast_f64(fld, flen, &is_null);
        ((double*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_DATE: {
        bool is_null;
        int32_t v = fast_date(fld, flen, &is_null);
        ((int32_t*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_TIME: {
        bool is_null;
        int32_t v = fast_time(fld, flen, &is_null);
        ((int32_t*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_TIMESTAMP: {
        bool is_null;
        int64_t v = fast_timestamp(fld, flen, &is_null);
        ((int64_t*)ctx->col_data[c])[row] = v;
        if (is_null) {
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        }
        break;
    }
    case CSV_TYPE_STR: {
        if (flen == 0) {
            ((uint32_t*)ctx->col_data[c])[row] = 0;
            ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
            my_had_null[c] = true;
        } else {
            uint32_t lid = local_sym_intern(&my_syms[c], fld, flen);
            if (TD_UNLIKELY(lid == UINT32_MAX)) { if (dyn_esc) td_sys_free(dyn_esc); continue; }
            ((uint32_t*)ctx->col_data[c])[row] = PACK_SYM(worker_id, lid);
        }
        break;
    }
    default:
        break;
}
```

Also update the row-boundary guard (lines 980-994) to set null bits for missing trailing columns:
```c
if (p >= row_end) {
    for (; c < ctx->n_cols; c++) {
        switch (ctx->col_types[c]) {
            case CSV_TYPE_BOOL: ((uint8_t*)ctx->col_data[c])[row] = 0; break;
            case CSV_TYPE_I64:  ((int64_t*)ctx->col_data[c])[row] = 0; break;
            case CSV_TYPE_F64:  ((double*)ctx->col_data[c])[row] = 0.0; break;
            case CSV_TYPE_DATE: ((int32_t*)ctx->col_data[c])[row] = 0; break;
            case CSV_TYPE_TIME: ((int32_t*)ctx->col_data[c])[row] = 0; break;
            case CSV_TYPE_TIMESTAMP: ((int64_t*)ctx->col_data[c])[row] = 0; break;
            case CSV_TYPE_STR:  ((uint32_t*)ctx->col_data[c])[row] = 0; break;
            default: break;
        }
        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
        my_had_null[c] = true;
    }
    break;
}
```

**Step 4: Update `csv_parse_serial` — same null-aware pattern**

Change the `csv_parse_serial` signature to accept nullmap pointers and had_null tracking:

```c
static void csv_parse_serial(const char* buf, size_t buf_size,
                              const int64_t* row_offsets, int64_t n_rows,
                              int n_cols, char delim,
                              const csv_type_t* col_types, void** col_data,
                              uint8_t** col_nullmaps, bool* col_had_null) {
```

Apply the same switch-case changes as the parallel path (using `col_nullmaps` and `col_had_null` directly instead of per-worker arrays).

For the STR case in serial path, the null check is:
```c
case CSV_TYPE_STR: {
    if (flen == 0) {
        ((uint32_t*)col_data[c])[row] = 0;
        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
        col_had_null[c] = true;
    } else {
        int64_t sym_id = td_sym_intern(fld, flen);
        if (sym_id < 0) sym_id = 0;
        ((uint32_t*)col_data[c])[row] = (uint32_t)sym_id;
    }
    break;
}
```

**Step 5: Wire nullmaps into dispatch call sites in `td_read_csv_opts`**

For the parallel path (around line 1354), add to `csv_par_ctx_t` init:
```c
.col_nullmaps = col_nullmaps,
.worker_had_null = worker_had_null_buf,
```

Before the parallel dispatch, allocate the per-worker null tracking:
```c
bool* worker_had_null_buf = NULL;
if (use_parallel) {
    size_t whn_sz = (size_t)n_workers * (size_t)ncols * sizeof(bool);
    worker_had_null_buf = (bool*)td_sys_alloc(whn_sz);
    if (worker_had_null_buf) memset(worker_had_null_buf, 0, whn_sz);
}
```

After the parallel dispatch completes (after merge_local_syms), OR the per-worker flags:
```c
if (worker_had_null_buf) {
    for (uint32_t w = 0; w < n_workers; w++) {
        for (int c = 0; c < ncols; c++) {
            if (worker_had_null_buf[(size_t)w * (size_t)ncols + (size_t)c])
                col_had_null[c] = true;
        }
    }
    td_sys_free(worker_had_null_buf);
}
```

For the serial path (around line 1387), pass the new params:
```c
csv_parse_serial(buf, file_size, row_offsets, n_rows,
                 ncols, delimiter, parse_types, col_data,
                 col_nullmaps, col_had_null);
```

**Step 6: Build and run tests**

Run: `cd .worktrees/csv-null-handling && cmake --build build -j$(nproc) && ./build/test_teide --suite /csv`

Expected: New null tests still FAIL — nullmaps are allocated but post-parse cleanup hasn't been added yet, and the tests should now see nulls set. Actually at this point the nullmaps ARE set, so the tests should PASS. Verify.

**Step 7: Commit**

```bash
git add src/io/csv.c
git commit -m "feat: allocate nullmaps and set null bits in CSV parse loops"
```

---

### Task 4: Post-Parse Cleanup and Sym Fixup Null Skipping

**Files:**
- Modify: `src/io/csv.c`

**Step 1: Add null cleanup after parse, before column narrowing (before line 1401)**

Insert after the parse block closes:
```c
/* ---- 9b. Strip nullmaps from all-valid columns ---- */
for (int c = 0; c < ncols; c++) {
    if (col_had_null[c]) continue;
    td_t* vec = col_vecs[c];
    if (vec->attrs & TD_ATTR_NULLMAP_EXT) {
        td_release(vec->ext_nullmap);
        vec->ext_nullmap = NULL;
    }
    vec->attrs &= (uint8_t)~(TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT);
    memset(vec->nullmap, 0, 16);
}
```

**Step 2: Update `sym_fixup_fn` to skip NULL elements**

In the sym fixup function (line 836-851), add null bitmap parameter to `sym_fixup_ctx_t`:

```c
typedef struct {
    uint32_t*  data;
    int64_t**  mappings;
    uint32_t*  mapping_counts;
    uint32_t   n_workers;
    uint8_t*   nullmap;      /* NULL if no nulls in this column */
} sym_fixup_ctx_t;
```

In `sym_fixup_fn`, skip null rows:
```c
static void sym_fixup_fn(void* arg, uint32_t worker_id, int64_t start, int64_t end) {
    (void)worker_id;
    sym_fixup_ctx_t* c = (sym_fixup_ctx_t*)arg;
    uint32_t* restrict data = c->data;
    int64_t** restrict mappings = c->mappings;
    uint32_t* restrict mcounts = c->mapping_counts;
    uint8_t*  restrict nm = c->nullmap;
    for (int64_t r = start; r < end; r++) {
        if (nm && (nm[r >> 3] & (1u << (r & 7)))) continue; /* NULL — skip */
        uint32_t packed = data[r];
        uint32_t wid = UNPACK_WID(packed);
        if (wid >= c->n_workers) continue;
        uint32_t lid = UNPACK_LID(packed);
        if (mappings[wid] && lid < mcounts[wid])
            data[r] = (uint32_t)mappings[wid][lid];
        else
            data[r] = 0;
    }
}
```

**Step 3: Pass nullmap when constructing `sym_fixup_ctx_t` in `merge_local_syms`**

Update `merge_local_syms` signature to accept `uint8_t** col_nullmaps`:

```c
static bool merge_local_syms(local_sym_t* local_syms, uint32_t n_workers,
                              int n_cols, const csv_type_t* col_types,
                              void** col_data, int64_t n_rows,
                              td_pool_t* pool, int64_t* col_max_ids,
                              uint8_t** col_nullmaps) {
```

Where `sym_fixup_ctx_t` is initialized in the merge function, set:
```c
.nullmap = col_nullmaps[c],
```

Update the call site in `td_read_csv_opts` to pass `col_nullmaps`.

**Step 4: Propagate nullmap through column narrowing (lines 1401-1421)**

When narrowing sym columns from W32 → W8/W16, transfer the nullmap to the new vector. After the narrowing copy loop, before `td_release(col_vecs[c])`:

```c
/* Transfer nullmap to narrowed vector */
if (col_vecs[c]->attrs & TD_ATTR_HAS_NULLS) {
    narrow->attrs |= (col_vecs[c]->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT));
    if (col_vecs[c]->attrs & TD_ATTR_NULLMAP_EXT) {
        narrow->ext_nullmap = col_vecs[c]->ext_nullmap;
        td_retain(narrow->ext_nullmap);
    } else {
        memcpy(narrow->nullmap, col_vecs[c]->nullmap, 16);
    }
}
```

**Step 5: Also update serial path max-sym-id scan to skip NULLs (lines 1390-1397)**

```c
for (int c = 0; c < ncols; c++) {
    if (parse_types[c] != CSV_TYPE_STR) continue;
    const uint32_t* d = (const uint32_t*)col_data[c];
    uint8_t* nm = col_nullmaps[c];
    uint32_t mx = 0;
    for (int64_t r = 0; r < n_rows; r++) {
        if (nm && (nm[r >> 3] & (1u << (r & 7)))) continue;
        if (d[r] > mx) mx = d[r];
    }
    sym_max_ids[c] = (int64_t)mx;
}
```

**Step 6: Build and run all tests**

Run: `cd .worktrees/csv-null-handling && cmake --build build -j$(nproc) && ./build/test_teide --suite /csv`

Expected: ALL tests pass including the 3 new null tests.

**Step 7: Commit**

```bash
git add src/io/csv.c
git commit -m "feat: post-parse nullmap cleanup, sym fixup null skipping, narrowing transfer"
```

---

### Task 5: Add Tests for Bool, Date, Time, Timestamp, and String Nulls

**Files:**
- Modify: `test/test_csv.c`

**Step 1: Write test — bool null handling**

```c
static MunitResult test_csv_null_bool(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "flag\ntrue\n\nfalse\nmaybe\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));

    td_t* col = td_table_get_col_idx(loaded, 0);
    munit_assert_false(td_vec_is_null(col, 0));
    munit_assert_uint8(((uint8_t*)td_data(col))[0], ==, 1);
    munit_assert_true(td_vec_is_null(col, 1));  /* empty */
    munit_assert_false(td_vec_is_null(col, 2));
    munit_assert_uint8(((uint8_t*)td_data(col))[2], ==, 0);
    munit_assert_true(td_vec_is_null(col, 3));  /* "maybe" not a bool */

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 2: Write test — string/symbol null handling**

```c
static MunitResult test_csv_null_sym(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "name\nalice\n\nbob\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));

    td_t* col = td_table_get_col_idx(loaded, 0);
    munit_assert_false(td_vec_is_null(col, 0));
    munit_assert_true(td_vec_is_null(col, 1));  /* empty → NULL */
    munit_assert_false(td_vec_is_null(col, 2));

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 3: Write test — no nulls means no nullmap overhead**

```c
static MunitResult test_csv_no_nulls_no_nullmap(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "x\n10\n20\n30\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));

    td_t* col = td_table_get_col_idx(loaded, 0);
    /* No nulls → HAS_NULLS flag should be stripped */
    munit_assert_false(col->attrs & TD_ATTR_HAS_NULLS);

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 4: Write test — multi-column with mixed nulls**

```c
static MunitResult test_csv_null_mixed_columns(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();
    td_sym_init();

    FILE* f = fopen(TMP_CSV, "w");
    fprintf(f, "id,val,name\n1,1.5,alice\n,2.5,\n3,,bob\n");
    fclose(f);

    td_t* loaded = td_read_csv(TMP_CSV);
    munit_assert_false(TD_IS_ERR(loaded));
    munit_assert_int(td_table_nrows(loaded), ==, 3);
    munit_assert_int(td_table_ncols(loaded), ==, 3);

    td_t* id_col = td_table_get_col_idx(loaded, 0);
    td_t* val_col = td_table_get_col_idx(loaded, 1);
    td_t* name_col = td_table_get_col_idx(loaded, 2);

    /* id column: 1, NULL, 3 */
    munit_assert_false(td_vec_is_null(id_col, 0));
    munit_assert_true(td_vec_is_null(id_col, 1));
    munit_assert_false(td_vec_is_null(id_col, 2));

    /* val column: 1.5, 2.5, NULL */
    munit_assert_false(td_vec_is_null(val_col, 0));
    munit_assert_false(td_vec_is_null(val_col, 1));
    munit_assert_true(td_vec_is_null(val_col, 2));

    /* name column: alice, NULL, bob */
    munit_assert_false(td_vec_is_null(name_col, 0));
    munit_assert_true(td_vec_is_null(name_col, 1));
    munit_assert_false(td_vec_is_null(name_col, 2));

    td_release(loaded);
    unlink(TMP_CSV);
    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}
```

**Step 5: Register all new tests in `csv_tests[]`**

```c
{ "/null_bool",             test_csv_null_bool,             NULL, NULL, 0, NULL },
{ "/null_sym",              test_csv_null_sym,              NULL, NULL, 0, NULL },
{ "/no_nulls_no_nullmap",   test_csv_no_nulls_no_nullmap,  NULL, NULL, 0, NULL },
{ "/null_mixed_columns",    test_csv_null_mixed_columns,    NULL, NULL, 0, NULL },
```

**Step 6: Build and run**

Run: `cd .worktrees/csv-null-handling && cmake --build build -j$(nproc) && ./build/test_teide --suite /csv`

Expected: ALL tests PASS.

**Step 7: Run full test suite for regressions**

Run: `cd .worktrees/csv-null-handling/build && ctest --output-on-failure`

Expected: All tests pass.

**Step 8: Commit**

```bash
git add test/test_csv.c
git commit -m "test: add null handling tests for bool, sym, mixed columns, and no-null cleanup"
```

---

### Task 6: Verify Existing Tests Still Pass (Regression Check)

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `cd .worktrees/csv-null-handling/build && ctest --output-on-failure`

Expected: ALL tests pass, including pre-existing CSV roundtrip tests. The roundtrip tests write valid data so no nulls should appear — verify the no-nulls cleanup path works.

**Step 2: Run with ASan/UBSan**

The debug build already has ASan/UBSan. Confirm no sanitizer warnings in the test output.

**Step 3: Done — no commit needed**
