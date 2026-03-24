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
#include <stdio.h>

/* ---- Setup / Teardown -------------------------------------------------- */

static void* sym_setup(const void* params, void* user_data) {
    (void)params; (void)user_data;
    td_heap_init();
    (void)td_sym_init();
    return NULL;
}

static void sym_teardown(void* fixture) {
    (void)fixture;
    td_sym_destroy();
    td_heap_destroy();
}

/* ---- sym_init_destroy -------------------------------------------------- */

static MunitResult test_sym_init_destroy(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* After init, count should be 0 */
    munit_assert_uint(td_sym_count(), ==, 0);

    return MUNIT_OK;
}

/* ---- sym_intern_basic -------------------------------------------------- */

static MunitResult test_sym_intern_basic(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = td_sym_intern("hello", 5);
    munit_assert_int(id, >=, 0);
    munit_assert_uint(td_sym_count(), ==, 1);

    return MUNIT_OK;
}

/* ---- sym_intern_duplicate ---------------------------------------------- */

static MunitResult test_sym_intern_duplicate(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id1 = td_sym_intern("hello", 5);
    int64_t id2 = td_sym_intern("hello", 5);
    munit_assert_int(id1, ==, id2);
    munit_assert_uint(td_sym_count(), ==, 1);

    return MUNIT_OK;
}

/* ---- sym_find_existing ------------------------------------------------- */

static MunitResult test_sym_find_existing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = td_sym_intern("world", 5);
    int64_t found = td_sym_find("world", 5);
    munit_assert_int(found, ==, id);

    return MUNIT_OK;
}

/* ---- sym_find_missing -------------------------------------------------- */

static MunitResult test_sym_find_missing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t found = td_sym_find("nonexistent", 11);
    munit_assert_int(found, ==, -1);

    return MUNIT_OK;
}

/* ---- sym_str_roundtrip ------------------------------------------------- */

static MunitResult test_sym_str_roundtrip(const void* params, void* fixture) {
    (void)params; (void)fixture;

    int64_t id = td_sym_intern("roundtrip", 9);
    munit_assert_int(id, >=, 0);

    td_t* s = td_sym_str(id);
    munit_assert_ptr_not_null(s);
    munit_assert_size(td_str_len(s), ==, 9);
    munit_assert_memory_equal(9, td_str_ptr(s), "roundtrip");

    return MUNIT_OK;
}

/* ---- sym_count --------------------------------------------------------- */

static MunitResult test_sym_count(const void* params, void* fixture) {
    (void)params; (void)fixture;

    munit_assert_uint(td_sym_count(), ==, 0);

    td_sym_intern("a", 1);
    munit_assert_uint(td_sym_count(), ==, 1);

    td_sym_intern("b", 1);
    munit_assert_uint(td_sym_count(), ==, 2);

    td_sym_intern("c", 1);
    munit_assert_uint(td_sym_count(), ==, 3);

    /* Duplicate should not increase count */
    td_sym_intern("a", 1);
    munit_assert_uint(td_sym_count(), ==, 3);

    return MUNIT_OK;
}

/* ---- sym_many ---------------------------------------------------------- */

static MunitResult test_sym_many(const void* params, void* fixture) {
    (void)params; (void)fixture;

    /* Intern 1000 unique symbols */
    int64_t ids[1000];
    char buf[32];
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        ids[i] = td_sym_intern(buf, (size_t)len);
        munit_assert_int(ids[i], >=, 0);
    }

    munit_assert_uint(td_sym_count(), ==, 1000);

    /* Verify all are distinct IDs */
    for (int i = 0; i < 1000; i++) {
        for (int j = i + 1; j < 1000; j++) {
            munit_assert_int(ids[i], !=, ids[j]);
        }
    }

    /* Verify all are retrievable with correct strings */
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        td_t* s = td_sym_str(ids[i]);
        munit_assert_ptr_not_null(s);
        munit_assert_size(td_str_len(s), ==, (size_t)len);
        munit_assert_memory_equal((size_t)len, td_str_ptr(s), buf);
    }

    /* Re-interning should return same IDs */
    for (int i = 0; i < 1000; i++) {
        int len = snprintf(buf, sizeof(buf), "sym_%d", i);
        int64_t id2 = td_sym_intern(buf, (size_t)len);
        munit_assert_int(id2, ==, ids[i]);
    }

    munit_assert_uint(td_sym_count(), ==, 1000);

    return MUNIT_OK;
}

/* ---- sym_bulk: intern 100K symbols and verify none are lost ------------ */

static MunitResult test_sym_bulk(const void* params, void* fixture) {
    (void)params; (void)fixture;

    #define BULK_N 100000

    /* Pre-reserve capacity (tests td_sym_ensure_cap) */
    bool cap_ok = td_sym_ensure_cap(BULK_N);
    munit_assert_true(cap_ok);

    /* Intern 100K unique symbols */
    char buf[32];
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id = td_sym_intern(buf, (size_t)len);
        munit_assert_int(id, >=, 0);
    }

    munit_assert_uint(td_sym_count(), ==, BULK_N);

    /* Verify every symbol is retrievable with correct string */
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id = td_sym_find(buf, (size_t)len);
        munit_assert_int(id, >=, 0);
        td_t* s = td_sym_str(id);
        munit_assert_ptr_not_null(s);
        munit_assert_size(td_str_len(s), ==, (size_t)len);
        munit_assert_memory_equal((size_t)len, td_str_ptr(s), buf);
    }

    /* Re-interning must return same IDs (idempotent) */
    for (int i = 0; i < BULK_N; i++) {
        int len = snprintf(buf, sizeof(buf), "bulk_%06d", i);
        int64_t id1 = td_sym_find(buf, (size_t)len);
        int64_t id2 = td_sym_intern(buf, (size_t)len);
        munit_assert_int(id1, ==, id2);
    }

    munit_assert_uint(td_sym_count(), ==, BULK_N);

    #undef BULK_N
    return MUNIT_OK;
}

/* ---- sym_save_load_roundtrip ------------------------------------------- */

static MunitResult test_sym_save_load_roundtrip(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_roundtrip.sym";

    /* Intern some symbols */
    int64_t id_hello = td_sym_intern("hello", 5);
    int64_t id_world = td_sym_intern("world", 5);
    int64_t id_foo   = td_sym_intern("foo", 3);
    munit_assert_int(id_hello, >=, 0);
    munit_assert_int(id_world, >=, 0);
    munit_assert_int(id_foo, >=, 0);
    munit_assert_uint(td_sym_count(), ==, 3);

    /* Save */
    td_err_t err = td_sym_save(sym_path);
    munit_assert_int(err, ==, TD_OK);

    /* Destroy and re-init sym table */
    td_sym_destroy();
    (void)td_sym_init();
    munit_assert_uint(td_sym_count(), ==, 0);

    /* Load */
    err = td_sym_load(sym_path);
    munit_assert_int(err, ==, TD_OK);
    munit_assert_uint(td_sym_count(), ==, 3);

    /* Verify all strings match */
    td_t* s0 = td_sym_str(id_hello);
    munit_assert_ptr_not_null(s0);
    munit_assert_size(td_str_len(s0), ==, 5);
    munit_assert_memory_equal(5, td_str_ptr(s0), "hello");

    td_t* s1 = td_sym_str(id_world);
    munit_assert_ptr_not_null(s1);
    munit_assert_size(td_str_len(s1), ==, 5);
    munit_assert_memory_equal(5, td_str_ptr(s1), "world");

    td_t* s2 = td_sym_str(id_foo);
    munit_assert_ptr_not_null(s2);
    munit_assert_size(td_str_len(s2), ==, 3);
    munit_assert_memory_equal(3, td_str_ptr(s2), "foo");

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_save_append_only --------------------------------------------- */

static MunitResult test_sym_save_append_only(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_append.sym";

    /* Intern initial batch */
    int64_t id_a = td_sym_intern("alpha", 5);
    int64_t id_b = td_sym_intern("beta", 4);
    munit_assert_int(id_a, >=, 0);
    munit_assert_int(id_b, >=, 0);

    /* First save */
    td_err_t err = td_sym_save(sym_path);
    munit_assert_int(err, ==, TD_OK);

    /* Second save with no changes -> should be no-op */
    err = td_sym_save(sym_path);
    munit_assert_int(err, ==, TD_OK);

    /* Intern more symbols */
    int64_t id_c = td_sym_intern("gamma", 5);
    int64_t id_d = td_sym_intern("delta", 5);
    munit_assert_int(id_c, >=, 0);
    munit_assert_int(id_d, >=, 0);

    /* Save again (append-only: new entries added) */
    err = td_sym_save(sym_path);
    munit_assert_int(err, ==, TD_OK);

    /* Destroy and reload */
    td_sym_destroy();
    (void)td_sym_init();
    err = td_sym_load(sym_path);
    munit_assert_int(err, ==, TD_OK);
    munit_assert_uint(td_sym_count(), ==, 4);

    /* Verify old IDs are stable */
    td_t* sa = td_sym_str(id_a);
    munit_assert_ptr_not_null(sa);
    munit_assert_memory_equal(5, td_str_ptr(sa), "alpha");

    td_t* sb = td_sym_str(id_b);
    munit_assert_ptr_not_null(sb);
    munit_assert_memory_equal(4, td_str_ptr(sb), "beta");

    /* Verify new IDs are present */
    td_t* sc = td_sym_str(id_c);
    munit_assert_ptr_not_null(sc);
    munit_assert_memory_equal(5, td_str_ptr(sc), "gamma");

    td_t* sd = td_sym_str(id_d);
    munit_assert_ptr_not_null(sd);
    munit_assert_memory_equal(5, td_str_ptr(sd), "delta");

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_load_corrupt -------------------------------------------------- */

static MunitResult test_sym_load_corrupt(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_corrupt.sym";

    /* Write garbage data */
    FILE* f = fopen(sym_path, "wb");
    munit_assert_ptr_not_null(f);
    const char garbage[] = "this is not a valid sym file";
    fwrite(garbage, 1, sizeof(garbage), f);
    fclose(f);

    /* Load should fail with corrupt */
    td_err_t err = td_sym_load(sym_path);
    munit_assert_int(err, !=, TD_OK);

    /* Count unchanged */
    munit_assert_uint(td_sym_count(), ==, 0);

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_load_truncated ------------------------------------------------ */

static MunitResult test_sym_load_truncated(const void* params, void* fixture) {
    (void)params; (void)fixture;

    const char* sym_path = "/tmp/test_sym_trunc.sym";

    /* Intern and save valid sym file */
    td_sym_intern("abc", 3);
    td_sym_intern("def", 3);
    td_err_t err = td_sym_save(sym_path);
    munit_assert_int(err, ==, TD_OK);

    /* Truncate the file to 2 bytes */
    FILE* f = fopen(sym_path, "wb");
    munit_assert_ptr_not_null(f);
    fwrite("AB", 1, 2, f);
    fclose(f);

    /* Destroy and re-init */
    td_sym_destroy();
    (void)td_sym_init();

    /* Load should fail */
    err = td_sym_load(sym_path);
    munit_assert_int(err, !=, TD_OK);
    munit_assert_uint(td_sym_count(), ==, 0);

    /* Cleanup */
    remove(sym_path);
    char lk_path[4096];
    snprintf(lk_path, sizeof(lk_path), "%s.lk", sym_path);
    remove(lk_path);

    return MUNIT_OK;
}

/* ---- sym_load_missing_file --------------------------------------------- */

static MunitResult test_sym_load_missing(const void* params, void* fixture) {
    (void)params; (void)fixture;

    td_err_t err = td_sym_load("/tmp/nonexistent_sym_file_xyz.sym");
    munit_assert_int(err, !=, TD_OK);
    munit_assert_uint(td_sym_count(), ==, 0);

    return MUNIT_OK;
}

/* ---- Suite definition -------------------------------------------------- */

static MunitTest sym_tests[] = {
    { "/init_destroy",    test_sym_init_destroy,    sym_setup, sym_teardown, 0, NULL },
    { "/intern_basic",    test_sym_intern_basic,    sym_setup, sym_teardown, 0, NULL },
    { "/intern_duplicate", test_sym_intern_duplicate, sym_setup, sym_teardown, 0, NULL },
    { "/find_existing",   test_sym_find_existing,   sym_setup, sym_teardown, 0, NULL },
    { "/find_missing",    test_sym_find_missing,    sym_setup, sym_teardown, 0, NULL },
    { "/str_roundtrip",   test_sym_str_roundtrip,   sym_setup, sym_teardown, 0, NULL },
    { "/count",           test_sym_count,           sym_setup, sym_teardown, 0, NULL },
    { "/many",            test_sym_many,            sym_setup, sym_teardown, 0, NULL },
    { "/bulk",            test_sym_bulk,            sym_setup, sym_teardown, 0, NULL },
    { "/save_load_roundtrip", test_sym_save_load_roundtrip, sym_setup, sym_teardown, 0, NULL },
    { "/save_append_only",    test_sym_save_append_only,    sym_setup, sym_teardown, 0, NULL },
    { "/load_corrupt",        test_sym_load_corrupt,        sym_setup, sym_teardown, 0, NULL },
    { "/load_truncated",      test_sym_load_truncated,      sym_setup, sym_teardown, 0, NULL },
    { "/load_missing",        test_sym_load_missing,        sym_setup, sym_teardown, 0, NULL },
    { NULL, NULL, NULL, NULL, 0, NULL },
};

MunitSuite test_sym_suite = {
    "/sym",
    sym_tests,
    NULL,
    0,
    0,
};
