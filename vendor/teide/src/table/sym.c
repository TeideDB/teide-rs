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

#include "sym.h"
#include "mem/sys.h"
#include <string.h>
#include <stdatomic.h>

/* --------------------------------------------------------------------------
 * FNV-1a 32-bit hash
 * -------------------------------------------------------------------------- */

static uint32_t fnv1a(const char* data, size_t len) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 0x01000193u;
    }
    return h;
}

/* --------------------------------------------------------------------------
 * Symbol table structure (static global, sequential mode only).
 * NOT thread-safe: all interning must happen before td_parallel_begin().
 * -------------------------------------------------------------------------- */

#define SYM_INIT_CAP     256
#define SYM_LOAD_FACTOR  0.7

typedef struct {
    /* Hash table: each bucket stores (hash32 << 32) | (id + 1), 0 = empty */
    uint64_t*  buckets;
    uint32_t   bucket_cap;   /* always power of 2 */

    /* String array: strings[id] = td_t* string atom */
    td_t**     strings;
    uint32_t   str_count;
    uint32_t   str_cap;
} sym_table_t;

static sym_table_t g_sym;
static _Atomic(bool) g_sym_inited = false;

/* Spinlock protecting g_sym mutations in td_sym_intern */
static _Atomic(int) g_sym_lock = 0;
static inline void sym_lock(void) {
    while (atomic_exchange_explicit(&g_sym_lock, 1, memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
}
static inline void sym_unlock(void) {
    atomic_store_explicit(&g_sym_lock, 0, memory_order_release);
}

/* --------------------------------------------------------------------------
 * td_sym_init
 * -------------------------------------------------------------------------- */

void td_sym_init(void) {
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&g_sym_inited, &expected, true,
            memory_order_acq_rel, memory_order_acquire))
        return; /* already initialized by another thread */

    g_sym.bucket_cap = SYM_INIT_CAP;
    /* td_sys_alloc uses mmap(MAP_ANONYMOUS) which zero-initializes. */
    g_sym.buckets = (uint64_t*)td_sys_alloc(g_sym.bucket_cap * sizeof(uint64_t));
    if (!g_sym.buckets) {
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return;
    }

    g_sym.str_cap = SYM_INIT_CAP;
    g_sym.str_count = 0;
    g_sym.strings = (td_t**)td_sys_alloc(g_sym.str_cap * sizeof(td_t*));
    if (!g_sym.strings) {
        td_sys_free(g_sym.buckets);
        g_sym.buckets = NULL;
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return;
    }
    /* g_sym_inited already set to true by CAS above */
}

/* --------------------------------------------------------------------------
 * td_sym_destroy
 * -------------------------------------------------------------------------- */

void td_sym_destroy(void) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return;

    /* Release all interned string atoms */
    for (uint32_t i = 0; i < g_sym.str_count; i++) {
        if (g_sym.strings[i]) {
            td_release(g_sym.strings[i]);
        }
    }

    td_sys_free(g_sym.strings);
    td_sys_free(g_sym.buckets);

    memset(&g_sym, 0, sizeof(g_sym));
    atomic_store_explicit(&g_sym_inited, false, memory_order_release);
}

/* --------------------------------------------------------------------------
 * Hash table helpers
 * -------------------------------------------------------------------------- */

static void ht_insert(uint64_t* buckets, uint32_t cap, uint32_t hash, uint32_t id) {
    uint32_t mask = cap - 1;
    uint32_t slot = hash & mask;
    uint64_t entry = ((uint64_t)hash << 32) | ((uint64_t)(id + 1));

    for (;;) {
        if (buckets[slot] == 0) {
            buckets[slot] = entry;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

/* Grow hash table to new_cap (must be power of 2 and > current cap). */
static bool ht_grow_to(uint32_t new_cap) {
    uint64_t* new_buckets = (uint64_t*)td_sys_alloc((size_t)new_cap * sizeof(uint64_t));
    if (!new_buckets) return false;

    /* Re-insert all existing entries */
    for (uint32_t i = 0; i < g_sym.bucket_cap; i++) {
        uint64_t e = g_sym.buckets[i];
        if (e == 0) continue;
        uint32_t h = (uint32_t)(e >> 32);
        uint32_t id = (uint32_t)(e & 0xFFFFFFFF) - 1;
        ht_insert(new_buckets, new_cap, h, id);
    }

    td_sys_free(g_sym.buckets);
    g_sym.buckets = new_buckets;
    g_sym.bucket_cap = new_cap;
    return true;
}

static bool ht_grow(void) {
    /* Overflow guard: bucket_cap is always power of 2.
     * At 2^31, doubling overflows uint32_t. */
    if (g_sym.bucket_cap >= (UINT32_MAX / 2 + 1)) return false;
    return ht_grow_to(g_sym.bucket_cap * 2);
}

/* --------------------------------------------------------------------------
 * td_sym_intern
 * -------------------------------------------------------------------------- */

int64_t td_sym_intern(const char* str, size_t len) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return -1;

    sym_lock();

    uint32_t hash = fnv1a(str, len);
    uint32_t mask = g_sym.bucket_cap - 1;
    uint32_t slot = hash & mask;

    /* Probe for existing entry */
    for (;;) {
        uint64_t e = g_sym.buckets[slot];
        if (e == 0) break;  /* empty -- not found */

        uint32_t e_hash = (uint32_t)(e >> 32);
        if (e_hash == hash) {
            uint32_t e_id = (uint32_t)(e & 0xFFFFFFFF) - 1;
            td_t* existing = g_sym.strings[e_id];
            if (td_str_len(existing) == len &&
                memcmp(td_str_ptr(existing), str, len) == 0) {
                sym_unlock();
                return (int64_t)e_id;
            }
        }
        slot = (slot + 1) & mask;
    }

    /* Grow hash table if load factor exceeds threshold, or if critically
     * full.  Attempt grow before refusing insert.
     * Cast to uint64_t to prevent overflow when bucket_cap >= 2^26. */
    if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 70) {
        if (!ht_grow()) {
            /* If critically full even after failed grow, refuse insert
             * to prevent infinite probe loops. */
            if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 95) {
                sym_unlock();
                return -1;
            }
        }
    }

    /* Not found -- create new entry */
    uint32_t new_id = g_sym.str_count;

    /* Grow strings array if needed */
    if (new_id >= g_sym.str_cap) {
        if (g_sym.str_cap >= UINT32_MAX / 2) { sym_unlock(); return -1; }
        uint32_t new_str_cap = g_sym.str_cap * 2;
        td_t** new_strings = (td_t**)td_sys_realloc(g_sym.strings,
                                                    (size_t)new_str_cap * sizeof(td_t*));
        if (!new_strings) { sym_unlock(); return -1; }
        g_sym.strings = new_strings;
        g_sym.str_cap = new_str_cap;
    }

    /* Create string atom — td_str() returns with rc=1 which is the
     * sym table's owning reference. No additional retain needed. */
    td_t* s = td_str(str, len);
    if (!s || TD_IS_ERR(s)) { sym_unlock(); return -1; }
    g_sym.strings[new_id] = s;
    g_sym.str_count++;

    /* Insert into hash table.
     * Note: ht_insert probes from hash & mask to find an empty slot,
     * so it works correctly even if ht_grow changed the bucket array. */
    ht_insert(g_sym.buckets, g_sym.bucket_cap, hash, new_id);

    sym_unlock();
    return (int64_t)new_id;
}

/* --------------------------------------------------------------------------
 * td_sym_find
 * -------------------------------------------------------------------------- */

int64_t td_sym_find(const char* str, size_t len) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return -1;

    /* Lock required: concurrent td_sym_intern may trigger ht_grow which
     * frees and replaces g_sym.buckets -- reading without lock is UAF. */
    sym_lock();

    uint32_t hash = fnv1a(str, len);
    uint32_t mask = g_sym.bucket_cap - 1;
    uint32_t slot = hash & mask;

    for (;;) {
        uint64_t e = g_sym.buckets[slot];
        if (e == 0) { sym_unlock(); return -1; }  /* empty -- not found */

        uint32_t e_hash = (uint32_t)(e >> 32);
        if (e_hash == hash) {
            uint32_t e_id = (uint32_t)(e & 0xFFFFFFFF) - 1;
            td_t* existing = g_sym.strings[e_id];
            if (td_str_len(existing) == len &&
                memcmp(td_str_ptr(existing), str, len) == 0) {
                sym_unlock();
                return (int64_t)e_id;
            }
        }
        slot = (slot + 1) & mask;
    }
}

/* --------------------------------------------------------------------------
 * td_sym_str
 * -------------------------------------------------------------------------- */

/* Returned pointer is valid only while no concurrent td_sym_intern occurs.
 * Safe during read-only execution phase (after all interning is complete).
 * Caller must not store the pointer across sym table mutations (ht_grow
 * or strings realloc). */
td_t* td_sym_str(int64_t id) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return NULL;

    /* Lock required: concurrent td_sym_intern may realloc g_sym.strings. */
    sym_lock();
    if (id < 0 || (uint32_t)id >= g_sym.str_count) { sym_unlock(); return NULL; }
    td_t* s = g_sym.strings[id];
    sym_unlock();
    return s;
}

/* --------------------------------------------------------------------------
 * td_sym_count
 * -------------------------------------------------------------------------- */

uint32_t td_sym_count(void) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return 0;

    /* Lock required: concurrent td_sym_intern may modify str_count. */
    sym_lock();
    uint32_t count = g_sym.str_count;
    sym_unlock();
    return count;
}

/* --------------------------------------------------------------------------
 * td_sym_ensure_cap -- pre-grow hash table and strings array
 *
 * Ensures the symbol table can hold at least `needed` total symbols without
 * rehashing.  Call before bulk interning (e.g., CSV merge) to prevent
 * mid-insert OOM that silently drops symbols.
 * -------------------------------------------------------------------------- */

bool td_sym_ensure_cap(uint32_t needed) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return false;

    sym_lock();

    /* Grow strings array if needed */
    while (g_sym.str_cap < needed) {
        if (g_sym.str_cap >= UINT32_MAX / 2) { sym_unlock(); return false; }
        uint32_t new_str_cap = g_sym.str_cap * 2;
        if (new_str_cap < needed) { /* jump directly to needed */
            new_str_cap = needed;
            /* Round up to power of 2 */
            new_str_cap--;
            new_str_cap |= new_str_cap >> 1;
            new_str_cap |= new_str_cap >> 2;
            new_str_cap |= new_str_cap >> 4;
            new_str_cap |= new_str_cap >> 8;
            new_str_cap |= new_str_cap >> 16;
            new_str_cap++;
        }
        td_t** new_strings = (td_t**)td_sys_realloc(g_sym.strings,
                                                     (size_t)new_str_cap * sizeof(td_t*));
        if (!new_strings) { sym_unlock(); return false; }
        g_sym.strings = new_strings;
        g_sym.str_cap = new_str_cap;
    }

    /* Grow hash table so load factor stays below threshold after filling */
    uint32_t needed_buckets = (uint32_t)((double)needed / SYM_LOAD_FACTOR) + 1;
    /* Round up to power of 2 */
    needed_buckets--;
    needed_buckets |= needed_buckets >> 1;
    needed_buckets |= needed_buckets >> 2;
    needed_buckets |= needed_buckets >> 4;
    needed_buckets |= needed_buckets >> 8;
    needed_buckets |= needed_buckets >> 16;
    needed_buckets++;

    if (needed_buckets > g_sym.bucket_cap) {
        if (!ht_grow_to(needed_buckets)) { sym_unlock(); return false; }
    }

    sym_unlock();
    return true;
}

/* --------------------------------------------------------------------------
 * td_sym_save -- serialize symbol table to a binary file
 *
 * Format:
 *   [4B magic "TSYM"][4B count]
 *   For each symbol: [4B len][len bytes data]
 * -------------------------------------------------------------------------- */

#include <stdio.h>

td_err_t td_sym_save(const char* path) {
    if (!path) return TD_ERR_IO;
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return TD_ERR_IO;

    /* Hold the lock for the entire save to prevent concurrent td_sym_intern
     * from reallocating the strings array or mutating str_count mid-save. */
    sym_lock();
    uint32_t count = g_sym.str_count;
    td_t** strings = g_sym.strings;

    FILE* f = fopen(path, "wb");
    if (!f) { sym_unlock(); return TD_ERR_IO; }

    uint32_t magic = 0x4D595354;  /* "TSYM" little-endian */

    if (fwrite(&magic, 4, 1, f) != 1 ||
        fwrite(&count, 4, 1, f) != 1) {
        fclose(f);
        sym_unlock();
        return TD_ERR_IO;
    }

    for (uint32_t i = 0; i < count; i++) {
        td_t* s = strings[i];
        uint32_t len = (uint32_t)td_str_len(s);
        const char* data = td_str_ptr(s);

        if (fwrite(&len, 4, 1, f) != 1 ||
            (len > 0 && fwrite(data, 1, len, f) != len)) {
            fclose(f);
            sym_unlock();
            return TD_ERR_IO;
        }
    }

    fclose(f);
    sym_unlock();
    return TD_OK;
}

/* --------------------------------------------------------------------------
 * td_sym_load -- deserialize symbol table from a binary file
 *
 * Interns all symbols from the file into the global symbol table.
 * Must be called after td_sym_init(). Existing symbols are preserved
 * (td_sym_intern is idempotent for matching strings).
 * -------------------------------------------------------------------------- */

td_err_t td_sym_load(const char* path) {
    if (!path) return TD_ERR_IO;
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return TD_ERR_IO;

    FILE* f = fopen(path, "rb");
    if (!f) return TD_ERR_IO;

    uint32_t magic, count;
    if (fread(&magic, 4, 1, f) != 1 ||
        fread(&count, 4, 1, f) != 1) {
        fclose(f);
        return TD_ERR_CORRUPT;
    }

    if (magic != 0x4D595354) {
        fclose(f);
        return TD_ERR_CORRUPT;
    }

    /* Reject unreasonable count to prevent DoS from crafted files */
    if (count > 100000000u) { /* 100M max symbols */
        fclose(f);
        return TD_ERR_CORRUPT;
    }

    /* Read buffer -- reuse for all strings */
    char buf[4096];
    char* heap_buf = NULL;
    size_t heap_cap = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t len;
        if (fread(&len, 4, 1, f) != 1) {
            if (heap_buf) td_sys_free(heap_buf);
            fclose(f);
            return TD_ERR_CORRUPT;
        }

        if (len > 16 * 1024 * 1024) { /* 16MB max per symbol */
            if (heap_buf) td_sys_free(heap_buf);
            fclose(f);
            return TD_ERR_CORRUPT;
        }

        char* dst = buf;
        if (len > sizeof(buf)) {
            if (len > heap_cap) {
                char* nb = (char*)td_sys_realloc(heap_buf, len);
                if (!nb) {
                    if (heap_buf) td_sys_free(heap_buf);
                    fclose(f);
                    return TD_ERR_OOM;
                }
                heap_buf = nb;
                heap_cap = len;
            }
            dst = heap_buf;
        }

        if (len > 0 && fread(dst, 1, len, f) != len) {
            if (heap_buf) td_sys_free(heap_buf);
            fclose(f);
            return TD_ERR_CORRUPT;
        }

        int64_t id = td_sym_intern(dst, len);
        if (id < 0) {
            if (heap_buf) td_sys_free(heap_buf);
            fclose(f);
            return TD_ERR_OOM;
        }
    }

    if (heap_buf) td_sys_free(heap_buf);
    fclose(f);
    return TD_OK;
}
