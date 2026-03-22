/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
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
#include "mem/arena.h"
#include <string.h>
#include <stdio.h>

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
    munit_assert_uint(v1->rc, ==, 1);

    /* Allocate a block with data */
    td_t* v2 = td_arena_alloc(arena, 100);
    munit_assert_ptr_not_null(v2);
    munit_assert_size((uintptr_t)v2 % 32, ==, 0);
    /* v2 should be different from v1 */
    munit_assert_true(v1 != v2);

    /* td_data(v2) should be writable for 100 bytes */
    memset(td_data(v2), 0xAB, 100);
    munit_assert_uint(((uint8_t*)td_data(v2))[0], ==, 0xAB);
    munit_assert_uint(((uint8_t*)td_data(v2))[99], ==, 0xAB);

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

    /* After reset, arena should be usable with a fresh chunk */
    td_t* v2 = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v2);
    munit_assert_true(v2->attrs & TD_ATTR_ARENA);

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

static MunitResult test_arena_destroy_null(const void* params, void* data) {
    (void)params; (void)data;
    /* Must not crash */
    td_arena_destroy(NULL);
    return MUNIT_OK;
}

static MunitResult test_arena_reset_multi_chunk(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    /* Tiny chunk to force multiple chunks */
    td_arena_t* arena = td_arena_new(256);
    munit_assert_ptr_not_null(arena);

    /* Allocate enough to span several chunks */
    for (int i = 0; i < 50; i++) {
        td_t* v = td_arena_alloc(arena, 64);
        munit_assert_ptr_not_null(v);
    }

    /* Reset should free extra chunks */
    td_arena_reset(arena);

    /* Arena should still be usable after reset */
    td_t* v = td_arena_alloc(arena, 64);
    munit_assert_ptr_not_null(v);
    munit_assert_true(v->attrs & TD_ATTR_ARENA);
    memset(td_data(v), 0xAB, 64);
    munit_assert_uint(((uint8_t*)td_data(v))[0], ==, 0xAB);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_arena_retain_noop(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    td_arena_t* arena = td_arena_new(4096);
    td_t* v = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v);
    munit_assert_uint(v->rc, ==, 1);

    /* td_retain should be a no-op for arena blocks */
    td_retain(v);
    munit_assert_uint(v->rc, ==, 1);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitResult test_arena_cow_noop(const void* params, void* data) {
    (void)params; (void)data;
    td_heap_init();

    td_arena_t* arena = td_arena_new(4096);
    td_t* v = td_arena_alloc(arena, 0);
    munit_assert_ptr_not_null(v);

    /* td_cow should return same pointer for arena blocks */
    td_t* cow_result = td_cow(v);
    munit_assert_ptr_equal(v, cow_result);

    td_arena_destroy(arena);
    td_heap_destroy();
    return MUNIT_OK;
}

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

    /* Verify long strings (>=7 bytes) use the arena CHAR vector path */
    const char* long_str = "this_is_a_long_symbol_name_for_testing";
    size_t long_len = strlen(long_str);
    int64_t long_id = td_sym_intern(long_str, long_len);
    munit_assert_int(long_id, >=, 0);
    td_t* long_s = td_sym_str(long_id);
    munit_assert_ptr_not_null(long_s);
    munit_assert_true(long_s->attrs & TD_ATTR_ARENA);
    munit_assert_int(td_str_len(long_s), ==, (int64_t)long_len);
    munit_assert_memory_equal(long_len, td_str_ptr(long_s), long_str);

    td_sym_destroy();
    td_heap_destroy();
    return MUNIT_OK;
}

static MunitTest arena_tests[] = {
    { "/release_noop",         test_arena_release_noop,         NULL, NULL, 0, NULL },
    { "/alloc_basic",          test_arena_alloc_basic,          NULL, NULL, 0, NULL },
    { "/grows_across_chunks",  test_arena_grows_across_chunks,  NULL, NULL, 0, NULL },
    { "/reset",                test_arena_reset,                NULL, NULL, 0, NULL },
    { "/oversize",             test_arena_oversize,             NULL, NULL, 0, NULL },
    { "/destroy_null",         test_arena_destroy_null,         NULL, NULL, 0, NULL },
    { "/reset_multi_chunk",    test_arena_reset_multi_chunk,    NULL, NULL, 0, NULL },
    { "/retain_noop",          test_arena_retain_noop,          NULL, NULL, 0, NULL },
    { "/cow_noop",             test_arena_cow_noop,             NULL, NULL, 0, NULL },
    { "/sym_intern",           test_arena_sym_intern,           NULL, NULL, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL }
};

MunitSuite test_arena_suite = {
    "/arena", arena_tests, NULL, 1, 0
};
