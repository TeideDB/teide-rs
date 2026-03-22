/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "munit.h"
#include <teide/td.h>
#include <string.h>

/* ---- Setup / Teardown -------------------------------------------------- */

static void* str_setup(const void* params, void* user_data) {
    (void)params; (void)user_data;
    td_heap_init();
    return NULL;
}

static void str_teardown(void* fixture) {
    (void)fixture;
    td_heap_destroy();
}

/* ---- str_ptr SSO ------------------------------------------------------- */

static MunitResult test_str_ptr_sso(const void* params, void* fixture) {
    (void)params; (void)fixture;

    td_t* s = td_str("hello", 5);
    munit_assert_ptr_not_null(s);

    const char* p = td_str_ptr(s);
    munit_assert_ptr_not_null(p);
    munit_assert_memory_equal(5, p, "hello");

    td_release(s);
    return MUNIT_OK;
}

/* ---- str_ptr long ------------------------------------------------------ */

static MunitResult test_str_ptr_long(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* text = "this is a longer string";
    size_t len = strlen(text);
    td_t* s = td_str(text, len);
    munit_assert_ptr_not_null(s);

    const char* p = td_str_ptr(s);
    munit_assert_ptr_not_null(p);
    munit_assert_memory_equal(len, p, text);

    td_release(s);
    return MUNIT_OK;
}

/* ---- str_len ----------------------------------------------------------- */

static MunitResult test_str_len(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* SSO */
    td_t* s1 = td_str("abc", 3);
    munit_assert_size(td_str_len(s1), ==, 3);
    td_release(s1);

    /* Empty SSO */
    td_t* s2 = td_str("", 0);
    munit_assert_size(td_str_len(s2), ==, 0);
    td_release(s2);

    /* Long */
    const char* text = "a longer string for testing";
    size_t len = strlen(text);
    td_t* s3 = td_str(text, len);
    munit_assert_size(td_str_len(s3), ==, len);
    td_release(s3);

    return MUNIT_OK;
}

/* ---- str_cmp equal ----------------------------------------------------- */

static MunitResult test_str_cmp_equal(const void* params, void* fixture) {
    (void)params; (void)fixture;

    td_t* a = td_str("hello", 5);
    td_t* b = td_str("hello", 5);

    munit_assert_int(td_str_cmp(a, b), ==, 0);

    td_release(a);
    td_release(b);
    return MUNIT_OK;
}

/* ---- str_cmp different ------------------------------------------------- */

static MunitResult test_str_cmp_different(const void* params, void* fixture) {
    (void)params; (void)fixture;

    td_t* a = td_str("abc", 3);
    td_t* b = td_str("abd", 3);

    int cmp = td_str_cmp(a, b);
    munit_assert_int(cmp, <, 0);  /* 'c' < 'd' */

    int cmp2 = td_str_cmp(b, a);
    munit_assert_int(cmp2, >, 0);

    td_release(a);
    td_release(b);
    return MUNIT_OK;
}

/* ---- str_cmp prefix ---------------------------------------------------- */

static MunitResult test_str_cmp_prefix(const void* params, void* fixture) {
    (void)params; (void)fixture;

    td_t* a = td_str("abc", 3);
    td_t* b = td_str("abcde", 5);

    int cmp = td_str_cmp(a, b);
    munit_assert_int(cmp, <, 0);  /* shorter sorts first */

    int cmp2 = td_str_cmp(b, a);
    munit_assert_int(cmp2, >, 0);

    td_release(a);
    td_release(b);
    return MUNIT_OK;
}

/* ---- str_vec_new ------------------------------------------------------- */

static MunitResult test_str_vec_new(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 10);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(v->type, ==, TD_STR);
    munit_assert_int(td_len(v), ==, 0);
    /* Pool starts as NULL (no long strings yet) */
    munit_assert_null(v->str_pool);
    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_new_zero_cap(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 0);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(v->type, ==, TD_STR);
    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_append inline ---------------------------------------------- */

static MunitResult test_str_vec_append_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Append a short string (fits in 12 bytes) */
    v = td_str_vec_append(v, "hello", 5);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(td_len(v), ==, 1);

    /* Verify element is inline */
    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint(elems[0].len, ==, 5);
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
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint(elems[0].len, ==, 12);
    munit_assert_memory_equal(12, elems[0].data, "123456789012");
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_append_empty(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    v = td_str_vec_append(v, "", 0);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint(elems[0].len, ==, 0);
    munit_assert_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_append pooled ---------------------------------------------- */

static MunitResult test_str_vec_append_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* 13 bytes — must go to pool */
    const char* long_str = "hello world!!";
    v = td_str_vec_append(v, long_str, 13);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(td_len(v), ==, 1);

    /* Pool should be allocated */
    munit_assert_ptr_not_null(v->str_pool);

    /* Element should have prefix and pool offset */
    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_uint(elems[0].len, ==, 13);
    munit_assert_memory_equal(4, elems[0].prefix, "hell");
    munit_assert_uint(elems[0].pool_off, ==, 0);

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
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(td_len(v), ==, 3);

    td_str_t* elems = (td_str_t*)td_data(v);
    /* First: inline */
    munit_assert_uint(elems[0].len, ==, 5);
    munit_assert_memory_equal(5, elems[0].data, "short");
    /* Second: pooled */
    munit_assert_uint(elems[1].len, ==, 22);
    munit_assert_memory_equal(4, elems[1].prefix, "this");
    /* Third: inline */
    munit_assert_uint(elems[2].len, ==, 4);
    munit_assert_memory_equal(4, elems[2].data, "tiny");

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_get ------------------------------------------------------- */

static MunitResult test_str_vec_get_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_ptr_not_null(ptr);
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
    munit_assert_ptr_not_null(ptr);
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

/* ---- str_vec_set ------------------------------------------------------- */

static MunitResult test_str_vec_set_inline_to_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    v = td_str_vec_set(v, 0, "world", 5);
    munit_assert_ptr_not_null(v);
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
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 20);
    munit_assert_memory_equal(20, ptr, "replacement string!!");

    /* Old 21 bytes are dead in pool */
    /* Pool used should be 21 (old dead) + 20 (new) = 41 */
    munit_assert_int(v->str_pool->len, ==, 41);

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

static MunitResult test_str_vec_set_inline_to_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "short", 5);

    /* Replace inline with pooled */
    v = td_str_vec_set(v, 0, "this is now a long string!", 26);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 26);
    munit_assert_memory_equal(26, ptr, "this is now a long string!");
    munit_assert_ptr_not_null(v->str_pool);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_set_pooled_to_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "this is a long string!!", 23);

    /* Replace pooled with inline */
    v = td_str_vec_set(v, 0, "tiny", 4);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 4);
    munit_assert_memory_equal(4, ptr, "tiny");

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_get/set negative index ------------------------------------ */

static MunitResult test_str_vec_get_negative(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    size_t len = SIZE_MAX;
    const char* ptr = td_str_vec_get(v, -1, &len);
    munit_assert_null(ptr);
    /* len should be zeroed on error */
    munit_assert_size(len, ==, 0);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_set_negative(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);

    td_t* r = td_str_vec_set(v, -1, "x", 1);
    munit_assert_true(TD_IS_ERR(r));

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_compact --------------------------------------------------- */

static MunitResult test_str_vec_compact(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Fill with pooled strings */
    v = td_str_vec_append(v, "original string one!", 20);
    v = td_str_vec_append(v, "original string two!", 20);

    /* Overwrite first — creates 20 dead bytes */
    v = td_str_vec_set(v, 0, "replacement str one!", 20);

    munit_assert_int(v->str_pool->len, ==, 60);  /* 20+20+20 */

    /* Compact */
    v = td_str_vec_compact(v);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    /* Pool should shrink: only 40 live bytes (string two + replacement) */
    munit_assert_int(v->str_pool->len, ==, 40);

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

static MunitResult test_str_vec_compact_noop(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* No pooled strings — compact should be a no-op */
    v = td_str_vec_append(v, "short", 5);
    v = td_str_vec_compact(v);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_null(v->str_pool);

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, ptr, "short");

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_compact_all_dead(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Append pooled, then overwrite to inline — all pool bytes dead */
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_set(v, 0, "short", 5);

    v = td_str_vec_compact(v);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    /* Pool should be freed entirely */
    munit_assert_null(v->str_pool);

    size_t len;
    const char* ptr = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, ptr, "short");

    td_release(v);
    return MUNIT_OK;
}

/* ---- compact: all pool bytes dead after overwrite ----------------------- */

static MunitResult test_str_vec_compact_all_pool_dead(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Append two pooled strings */
    v = td_str_vec_append(v, "first long pooled str!", 22);
    v = td_str_vec_append(v, "second long pooled st!", 22);

    /* Overwrite both with inline — all 44 pool bytes become dead */
    v = td_str_vec_set(v, 0, "a", 1);
    v = td_str_vec_set(v, 1, "b", 1);

    /* Compact should free pool entirely */
    v = td_str_vec_compact(v);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_null(v->str_pool);

    /* Strings still readable */
    size_t len;
    const char* p0 = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 1);
    munit_assert_memory_equal(1, p0, "a");

    td_release(v);
    return MUNIT_OK;
}

/* ---- compact: saturated dead counter ----------------------------------- */

static MunitResult test_str_vec_compact_saturated_dead(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);

    /* Append one pooled string (>12 bytes so it goes to pool) */
    v = td_str_vec_append(v, "this is a pooled string!", 24);

    /* Force the dead-byte counter to UINT32_MAX to simulate saturation.
     * The counter is stored in the first 4 bytes of str_pool->nullmap. */
    munit_assert_ptr_not_null(v->str_pool);
    uint32_t saturated = UINT32_MAX;
    memcpy(v->str_pool->nullmap, &saturated, 4);

    /* Compact must scan elements for true live size instead of using
     * pool_used - dead (which would underflow). */
    v = td_str_vec_compact(v);
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));

    /* The pooled string must survive compact */
    munit_assert_ptr_not_null(v->str_pool);
    size_t len;
    const char* s = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 24);
    munit_assert_memory_equal(24, s, "this is a pooled string!");

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_t_eq inline --------------------------------------------------- */

static MunitResult test_str_t_eq_inline(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "world", 5);

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_true(td_str_t_eq(&elems[0], NULL, &elems[1], NULL));
    munit_assert_false(td_str_t_eq(&elems[0], NULL, &elems[2], NULL));

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_t_eq pooled --------------------------------------------------- */

static MunitResult test_str_t_eq_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "a]longer string here!", 21);
    v = td_str_vec_append(v, "a]longer string here!", 21);
    v = td_str_vec_append(v, "a]longer string nope!", 21);

    td_str_t* elems = (td_str_t*)td_data(v);
    const char* pool = (const char*)td_data(v->str_pool);
    munit_assert_true(td_str_t_eq(&elems[0], pool, &elems[1], pool));
    /* Same prefix "a]lo" but different content */
    munit_assert_false(td_str_t_eq(&elems[0], pool, &elems[2], pool));

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_t_cmp order --------------------------------------------------- */

static MunitResult test_str_t_cmp_order(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "apple", 5);
    v = td_str_vec_append(v, "banana", 6);
    v = td_str_vec_append(v, "apple", 5);

    td_str_t* elems = (td_str_t*)td_data(v);
    munit_assert_int(td_str_t_cmp(&elems[0], NULL, &elems[1], NULL), <, 0);
    munit_assert_int(td_str_t_cmp(&elems[1], NULL, &elems[0], NULL), >, 0);
    munit_assert_int(td_str_t_cmp(&elems[0], NULL, &elems[2], NULL), ==, 0);

    td_release(v);
    return MUNIT_OK;
}

/* ---- str_vec_null ------------------------------------------------------ */

static MunitResult test_str_vec_null(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "", 0);  /* empty, not null */
    v = td_str_vec_append(v, "world", 5);
    munit_assert_int(td_len(v), ==, 3);

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

/* ---- null with pooled strings ------------------------------------------ */

static MunitResult test_str_vec_null_pooled(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "this is a long string!!", 23);
    v = td_str_vec_append(v, "another long string!!!", 22);
    v = td_str_vec_append(v, "short", 5);

    /* Set null on row 1 — must not corrupt str_pool */
    td_vec_set_null(v, 1, true);
    munit_assert_true(td_vec_is_null(v, 1));
    munit_assert_false(td_vec_is_null(v, 0));
    munit_assert_false(td_vec_is_null(v, 2));

    /* Pool and strings must still be intact */
    munit_assert_ptr_not_null(v->str_pool);
    size_t len;
    const char* p0 = td_str_vec_get(v, 0, &len);
    munit_assert_size(len, ==, 23);
    munit_assert_memory_equal(23, p0, "this is a long string!!");

    const char* p2 = td_str_vec_get(v, 2, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, p2, "short");

    td_release(v);
    return MUNIT_OK;
}

/* ---- grow stress test -------------------------------------------------- */

static MunitResult test_str_vec_grow(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 2);  /* small initial capacity */

    /* Append 200 strings, mixing inline and pooled, forcing multiple reallocs */
    for (int i = 0; i < 200; i++) {
        char buf[64];
        int slen;
        if (i % 3 == 0) {
            /* Pooled: > 12 bytes */
            slen = snprintf(buf, sizeof(buf), "long-string-number-%04d!", i);
        } else {
            /* Inline: <= 12 bytes */
            slen = snprintf(buf, sizeof(buf), "s%03d", i);
        }
        v = td_str_vec_append(v, buf, (size_t)slen);
        munit_assert_ptr_not_null(v);
        munit_assert_false(TD_IS_ERR(v));
    }

    munit_assert_int(td_len(v), ==, 200);

    /* Verify all strings are readable */
    for (int i = 0; i < 200; i++) {
        char buf[64];
        int slen;
        if (i % 3 == 0) {
            slen = snprintf(buf, sizeof(buf), "long-string-number-%04d!", i);
        } else {
            slen = snprintf(buf, sizeof(buf), "s%03d", i);
        }
        size_t got_len;
        const char* ptr = td_str_vec_get(v, i, &got_len);
        munit_assert_ptr_not_null(ptr);
        munit_assert_size(got_len, ==, (size_t)slen);
        munit_assert_memory_equal((size_t)slen, ptr, buf);
    }

    td_release(v);
    return MUNIT_OK;
}

/* ---- Slice tests ------------------------------------------------------- */

static MunitResult test_str_vec_slice(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* v = td_vec_new(TD_STR, 4);
    v = td_str_vec_append(v, "hello", 5);
    v = td_str_vec_append(v, "this is a long string!", 22);
    v = td_str_vec_append(v, "world", 5);
    v = td_str_vec_append(v, "another long string!!", 21);

    td_t* s = td_vec_slice(v, 1, 2);
    munit_assert_ptr_not_null(s);
    munit_assert_false(TD_IS_ERR(s));
    munit_assert_int(td_len(s), ==, 2);

    /* Read through slice — pooled string */
    size_t len;
    const char* p0 = td_str_vec_get(s, 0, &len);
    munit_assert_size(len, ==, 22);
    munit_assert_memory_equal(22, p0, "this is a long string!");

    /* Read through slice — inline string */
    const char* p1 = td_str_vec_get(s, 1, &len);
    munit_assert_size(len, ==, 5);
    munit_assert_memory_equal(5, p1, "world");

    td_release(s);
    td_release(v);
    return MUNIT_OK;
}

/* ---- Concat tests ------------------------------------------------------ */

static MunitResult test_str_vec_concat_vecs(const void* params, void* fixture) {
    (void)params; (void)fixture;
    td_t* a = td_vec_new(TD_STR, 2);
    a = td_str_vec_append(a, "hello", 5);
    a = td_str_vec_append(a, "this is long string a!", 22);

    td_t* b = td_vec_new(TD_STR, 2);
    b = td_str_vec_append(b, "world", 5);
    b = td_str_vec_append(b, "this is long string b!", 22);

    td_t* c = td_vec_concat(a, b);
    munit_assert_ptr_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int(td_len(c), ==, 4);

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

/* ---- Suite definition -------------------------------------------------- */

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

    /* Same strings -> same hash */
    munit_assert(h0 == h1);
    /* Different strings -> different hash (extremely likely) */
    munit_assert(h0 != h2);

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

    munit_assert(h0 == h1);
    munit_assert(h0 != h2);

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
    munit_assert(h0 == h1);

    td_release(v);
    return MUNIT_OK;
}

static MunitResult test_str_vec_concat_pooled_rebase(const void* params, void* fixture) {
    (void)params; (void)fixture;
    /* Verify that concat of two pooled TD_STR vectors correctly rebases
     * pool offsets so that the second vector's strings resolve correctly. */
    td_t* a = td_vec_new(TD_STR, 2);
    a = td_str_vec_append(a, "this is a long pooled string a!", 30);
    td_t* b = td_vec_new(TD_STR, 2);
    b = td_str_vec_append(b, "this is a long pooled string b!", 30);

    td_t* c = td_vec_concat(a, b);
    munit_assert_ptr_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int(td_len(c), ==, 2);

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
    munit_assert_ptr_not_null(c);
    munit_assert_false(TD_IS_ERR(c));
    munit_assert_int(td_len(c), ==, 5);

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

/* ---- str_vec_slice_null ------------------------------------------------ */

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
    munit_assert_ptr_not_null(s);
    munit_assert_false(TD_IS_ERR(s));

    /* Slice index 0 = parent index 1 = null */
    munit_assert_true(td_vec_is_null(s, 0));
    /* Slice index 1 = parent index 2 = not null */
    munit_assert_false(td_vec_is_null(s, 1));

    td_release(s);
    td_release(v);
    return MUNIT_OK;
}

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
    munit_assert_ptr_not_null(v);
    munit_assert_false(TD_IS_ERR(v));
    munit_assert_int(td_len(v), ==, 3);

    /* Original should still have 2 elements */
    munit_assert_int(td_len(shared), ==, 2);

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
    munit_assert_ptr_not_null(v);
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

static MunitTest str_tests[] = {
    { "/ptr_sso",       test_str_ptr_sso,       str_setup, str_teardown, 0, NULL },
    { "/ptr_long",      test_str_ptr_long,       str_setup, str_teardown, 0, NULL },
    { "/len",           test_str_len,             str_setup, str_teardown, 0, NULL },
    { "/cmp_equal",     test_str_cmp_equal,       str_setup, str_teardown, 0, NULL },
    { "/cmp_different", test_str_cmp_different,   str_setup, str_teardown, 0, NULL },
    { "/cmp_prefix",    test_str_cmp_prefix,      str_setup, str_teardown, 0, NULL },
    { "/vec_new",       test_str_vec_new,         str_setup, str_teardown, 0, NULL },
    { "/vec_new_zero",  test_str_vec_new_zero_cap, str_setup, str_teardown, 0, NULL },
    { "/vec_append_inline",    test_str_vec_append_inline,    str_setup, str_teardown, 0, NULL },
    { "/vec_append_inline_12", test_str_vec_append_inline_12, str_setup, str_teardown, 0, NULL },
    { "/vec_append_empty",     test_str_vec_append_empty,     str_setup, str_teardown, 0, NULL },
    { "/vec_append_pooled",    test_str_vec_append_pooled,    str_setup, str_teardown, 0, NULL },
    { "/vec_append_mixed",     test_str_vec_append_mixed,     str_setup, str_teardown, 0, NULL },
    { "/vec_get_inline",       test_str_vec_get_inline,       str_setup, str_teardown, 0, NULL },
    { "/vec_get_pooled",       test_str_vec_get_pooled,       str_setup, str_teardown, 0, NULL },
    { "/vec_get_oob",          test_str_vec_get_oob,          str_setup, str_teardown, 0, NULL },
    { "/vec_set_i2i",          test_str_vec_set_inline_to_inline, str_setup, str_teardown, 0, NULL },
    { "/vec_set_p2p",          test_str_vec_set_pooled_to_pooled, str_setup, str_teardown, 0, NULL },
    { "/vec_set_oob",          test_str_vec_set_oob,          str_setup, str_teardown, 0, NULL },
    { "/vec_set_i2p",          test_str_vec_set_inline_to_pooled, str_setup, str_teardown, 0, NULL },
    { "/vec_set_p2i",          test_str_vec_set_pooled_to_inline, str_setup, str_teardown, 0, NULL },
    { "/vec_get_neg",          test_str_vec_get_negative,     str_setup, str_teardown, 0, NULL },
    { "/vec_set_neg",          test_str_vec_set_negative,     str_setup, str_teardown, 0, NULL },
    { "/vec_compact",          test_str_vec_compact,          str_setup, str_teardown, 0, NULL },
    { "/vec_compact_noop",     test_str_vec_compact_noop,     str_setup, str_teardown, 0, NULL },
    { "/vec_compact_all_dead", test_str_vec_compact_all_dead, str_setup, str_teardown, 0, NULL },
    { "/vec_compact_all_pool_dead", test_str_vec_compact_all_pool_dead, str_setup, str_teardown, 0, NULL },
    { "/vec_compact_saturated_dead", test_str_vec_compact_saturated_dead, str_setup, str_teardown, 0, NULL },
    { "/t_eq_inline",          test_str_t_eq_inline,          str_setup, str_teardown, 0, NULL },
    { "/t_eq_pooled",          test_str_t_eq_pooled,          str_setup, str_teardown, 0, NULL },
    { "/t_cmp_order",          test_str_t_cmp_order,          str_setup, str_teardown, 0, NULL },
    { "/vec_null",             test_str_vec_null,             str_setup, str_teardown, 0, NULL },
    { "/vec_null_pooled",      test_str_vec_null_pooled,      str_setup, str_teardown, 0, NULL },
    { "/vec_grow",             test_str_vec_grow,             str_setup, str_teardown, 0, NULL },
    { "/vec_slice",            test_str_vec_slice,            str_setup, str_teardown, 0, NULL },
    { "/vec_concat",           test_str_vec_concat_vecs,      str_setup, str_teardown, 0, NULL },
    { "/t_hash_inline",        test_str_t_hash_inline,        str_setup, str_teardown, 0, NULL },
    { "/t_hash_pooled",        test_str_t_hash_pooled,        str_setup, str_teardown, 0, NULL },
    { "/t_hash_empty",         test_str_t_hash_empty,         str_setup, str_teardown, 0, NULL },
    { "/vec_concat_pooled_rebase",  test_str_vec_concat_pooled_rebase, str_setup, str_teardown, 0, NULL },
    { "/vec_concat_nulls",    test_str_vec_concat_nulls,         str_setup, str_teardown, 0, NULL },
    { "/vec_slice_null",      test_str_vec_slice_null,           str_setup, str_teardown, 0, NULL },
    { "/vec_cow_append",      test_str_vec_cow_append,           str_setup, str_teardown, 0, NULL },
    { "/vec_cow_set",         test_str_vec_cow_set,              str_setup, str_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_str_suite = {
    "/str",
    str_tests,
    NULL,
    0,
    0,
};
