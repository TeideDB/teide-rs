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
#include "store/fileio.h"
#include "mem/sys.h"
#include "mem/arena.h"
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <errno.h>
#include "ops/hash.h"

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

    /* Persistence: entries [0..persisted_count-1] are known on disk */
    uint32_t   persisted_count;

    /* Arena for string atoms — avoids per-string buddy allocator calls */
    td_arena_t*  arena;
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

/* Arena-backed td_str equivalent. Same logic as td_str() in atom.c
 * but allocates from the sym arena instead of the buddy allocator. */
static td_t* sym_str_arena(td_arena_t* arena, const char* s, size_t len) {
    if (len < 7) {
        /* SSO path: inline in header */
        td_t* v = td_arena_alloc(arena, 0);
        if (!v) return NULL;
        v->type = TD_ATOM_STR;
        v->slen = (uint8_t)len;
        if (len > 0) memcpy(v->sdata, s, len);
        v->sdata[len] = '\0';
        return v;
    }
    /* Long string: fused single allocation for CHAR vector + STR header.
     * Layout: [CHAR td_t header (32B) | string data (len+1) | padding | STR td_t header (32B)]
     * This halves arena_alloc calls for long strings. */
    size_t data_size = len + 1;
    size_t chars_block = ((32 + data_size) + 31) & ~(size_t)31;  /* align up to 32 */
    td_t* chars = td_arena_alloc(arena, chars_block + 32 - 32);  /* chars_block - 32 (header) + 32 (str header) */
    if (!chars) return NULL;
    chars->type = TD_CHAR;
    chars->len = (int64_t)len;
    memcpy(td_data(chars), s, len);
    ((char*)td_data(chars))[len] = '\0';

    /* STR header sits right after the CHAR block */
    td_t* v = (td_t*)((char*)chars + chars_block);
    memset(v, 0, 32);
    v->attrs = TD_ATTR_ARENA;
    atomic_store_explicit(&v->rc, 1, memory_order_relaxed);
    v->type = TD_ATOM_STR;
    v->obj = chars;
    return v;
}

/* --------------------------------------------------------------------------
 * td_sym_init
 * -------------------------------------------------------------------------- */

td_err_t td_sym_init(void) {
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&g_sym_inited, &expected, true,
            memory_order_acq_rel, memory_order_acquire))
        return TD_OK; /* already initialized by another thread */

    g_sym.bucket_cap = SYM_INIT_CAP;
    /* td_sys_alloc uses mmap(MAP_ANONYMOUS) which zero-initializes. */
    g_sym.buckets = (uint64_t*)td_sys_alloc(g_sym.bucket_cap * sizeof(uint64_t));
    if (!g_sym.buckets) {
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return TD_ERR_OOM;
    }

    g_sym.str_cap = SYM_INIT_CAP;
    g_sym.str_count = 0;
    g_sym.strings = (td_t**)td_sys_alloc(g_sym.str_cap * sizeof(td_t*));
    if (!g_sym.strings) {
        td_sys_free(g_sym.buckets);
        g_sym.buckets = NULL;
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return TD_ERR_OOM;
    }

    g_sym.arena = td_arena_new(1024 * 1024);  /* 1MB chunks */
    if (!g_sym.arena) {
        td_sys_free(g_sym.strings);
        td_sys_free(g_sym.buckets);
        g_sym.strings = NULL;
        g_sym.buckets = NULL;
        atomic_store_explicit(&g_sym_inited, false, memory_order_release);
        return TD_ERR_OOM;
    }
    /* g_sym_inited already set to true by CAS above */
    return TD_OK;
}

/* --------------------------------------------------------------------------
 * td_sym_destroy
 * -------------------------------------------------------------------------- */

void td_sym_destroy(void) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return;

    /* Arena-backed strings: td_release is a no-op (TD_ATTR_ARENA).
     * Destroy the arena to free all string atoms at once. */
    if (g_sym.arena) {
        td_arena_destroy(g_sym.arena);
        g_sym.arena = NULL;
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

    uint32_t hash = (uint32_t)td_hash_bytes(str, len);
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

    /* Create string atom from arena — avoids buddy allocator overhead.
     * Arena blocks have rc=1 and TD_ATTR_ARENA set. */
    td_t* s = sym_str_arena(g_sym.arena, str, len);
    if (!s) { sym_unlock(); return -1; }
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
 * td_sym_intern_prehashed -- intern with pre-computed hash, no lock.
 *
 * CALLER CONTRACT: must only be called when no other thread is interning
 * (e.g., after td_pool_dispatch returns during CSV merge).
 * -------------------------------------------------------------------------- */

int64_t td_sym_intern_prehashed(uint32_t hash, const char* str, size_t len) {
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return -1;

    uint32_t mask = g_sym.bucket_cap - 1;
    uint32_t slot = hash & mask;

    for (;;) {
        uint64_t e = g_sym.buckets[slot];
        if (e == 0) break;

        uint32_t e_hash = (uint32_t)(e >> 32);
        if (e_hash == hash) {
            uint32_t e_id = (uint32_t)(e & 0xFFFFFFFF) - 1;
            td_t* existing = g_sym.strings[e_id];
            if (td_str_len(existing) == len &&
                memcmp(td_str_ptr(existing), str, len) == 0) {
                return (int64_t)e_id;
            }
        }
        slot = (slot + 1) & mask;
    }

    if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 70) {
        if (!ht_grow()) {
            if ((uint64_t)g_sym.str_count * 100 >= (uint64_t)g_sym.bucket_cap * 95) {
                return -1;
            }
        }
    }

    uint32_t new_id = g_sym.str_count;

    if (new_id >= g_sym.str_cap) {
        if (g_sym.str_cap >= UINT32_MAX / 2) return -1;
        uint32_t new_str_cap = g_sym.str_cap * 2;
        td_t** new_strings = (td_t**)td_sys_realloc(g_sym.strings,
                                                    (size_t)new_str_cap * sizeof(td_t*));
        if (!new_strings) return -1;
        g_sym.strings = new_strings;
        g_sym.str_cap = new_str_cap;
    }

    td_t* s = sym_str_arena(g_sym.arena, str, len);
    if (!s) return -1;
    g_sym.strings[new_id] = s;
    g_sym.str_count++;

    ht_insert(g_sym.buckets, g_sym.bucket_cap, hash, new_id);

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

    uint32_t hash = (uint32_t)td_hash_bytes(str, len);
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
            if (new_str_cap == 0) { sym_unlock(); return false; }
        }
        td_t** new_strings = (td_t**)td_sys_realloc(g_sym.strings,
                                                     (size_t)new_str_cap * sizeof(td_t*));
        if (!new_strings) { sym_unlock(); return false; }
        g_sym.strings = new_strings;
        g_sym.str_cap = new_str_cap;
    }

    /* Grow hash table so load factor stays below threshold after filling */
    double raw_buckets = (double)needed / SYM_LOAD_FACTOR + 1.0;
    if (raw_buckets > (double)UINT32_MAX) { sym_unlock(); return false; }
    uint32_t needed_buckets = (uint32_t)raw_buckets;
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
 * td_sym_save -- serialize symbol table as TD_LIST of TD_ATOM_STR
 *
 * Uses td_col_save (STRL format), file locking for concurrent writers,
 * and fsync + atomic rename for crash safety.  Append-only: skips save
 * when persisted_count == str_count.
 * -------------------------------------------------------------------------- */

td_err_t td_sym_save(const char* path) {
    if (!path) return TD_ERR_IO;
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return TD_ERR_IO;

    /* Quick check: nothing new to persist? */
    sym_lock();
    if (g_sym.persisted_count == g_sym.str_count) {
        sym_unlock();
        return TD_OK;
    }
    sym_unlock();

    /* Build lock and temp paths */
    char lock_path[1024];
    char tmp_path[1024];
    if (snprintf(lock_path, sizeof(lock_path), "%s.lk", path) >= (int)sizeof(lock_path))
        return TD_ERR_IO;
    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) >= (int)sizeof(tmp_path))
        return TD_ERR_IO;

    /* Acquire cross-process exclusive lock */
    td_fd_t lock_fd = td_file_open(lock_path, TD_OPEN_READ | TD_OPEN_WRITE | TD_OPEN_CREATE);
    if (lock_fd == TD_FD_INVALID) return TD_ERR_IO;
    td_err_t err = td_file_lock_ex(lock_fd);
    if (err != TD_OK) { td_file_close(lock_fd); return err; }

    /* If file exists, load and merge (pick up entries from other writers).
     * Distinguish "file not found" (proceed with full save) from real I/O
     * errors (abort to avoid overwriting a file we couldn't read). */
    {
        td_t* existing = td_col_load(path);
        if (existing && !TD_IS_ERR(existing)) {
            if (existing->type != TD_LIST) {
                td_release(existing);
                td_file_unlock(lock_fd);
                td_file_close(lock_fd);
                return TD_ERR_CORRUPT;
            }
            /* Intern any new entries from disk (idempotent).
             * Verify each entry's in-memory ID matches its disk position:
             * if a local symbol already occupies a slot that disk expects,
             * the tables have diverged and merging would silently reorder
             * symbol IDs, corrupting previously written TD_SYM columns. */
            td_t** slots = (td_t**)td_data(existing);
            for (int64_t i = 0; i < existing->len; i++) {
                td_t* s = slots[i];
                if (!s || TD_IS_ERR(s) || s->type != TD_ATOM_STR) {
                    td_release(existing);
                    td_file_unlock(lock_fd);
                    td_file_close(lock_fd);
                    return TD_ERR_CORRUPT;
                }
                int64_t id = td_sym_intern(td_str_ptr(s), td_str_len(s));
                if (id < 0) {
                    td_release(existing);
                    td_file_unlock(lock_fd);
                    td_file_close(lock_fd);
                    return TD_ERR_OOM;
                }
                if (id != i) {
                    /* Divergent symbol tables: disk position i maps to
                     * in-memory id != i.  A local symbol occupies the
                     * slot, so merging would reorder IDs and corrupt
                     * any TD_SYM columns written by the other writer. */
                    td_release(existing);
                    td_file_unlock(lock_fd);
                    td_file_close(lock_fd);
                    return TD_ERR_CORRUPT;
                }
            }
            td_release(existing);
        } else {
            /* td_col_load failed — check if the file actually exists.
             * If it does, the failure is a real I/O/corruption error;
             * do not overwrite the file with a potentially incomplete
             * in-memory snapshot. */
            td_fd_t probe_fd = td_file_open(path, TD_OPEN_READ);
            if (probe_fd != TD_FD_INVALID) {
                /* File exists and is readable but td_col_load failed —
                 * corruption or format error; do not overwrite. */
                td_file_close(probe_fd);
                td_file_unlock(lock_fd);
                td_file_close(lock_fd);
                return TD_IS_ERR(existing) ? TD_ERR_CODE(existing) : TD_ERR_IO;
            }
            if (errno != ENOENT) {
                /* File may exist but we can't open it (EACCES, EMFILE,
                 * EIO, etc.) — do not overwrite, report I/O error. */
                td_file_unlock(lock_fd);
                td_file_close(lock_fd);
                return TD_ERR_IO;
            }
            /* File does not exist (ENOENT) — proceed with full save */
        }
    }

    /* Snapshot string pointers under sym_lock, then build list without it.
     * Strings are append-only and never freed, so pointers remain valid. */
    sym_lock();
    uint32_t count = g_sym.str_count;
    size_t snap_sz = count * sizeof(td_t*);
    td_t* snap_block = td_alloc(snap_sz);
    if (!snap_block) {
        sym_unlock();
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return TD_ERR_OOM;
    }
    td_t** snap = (td_t**)td_data(snap_block);
    memcpy(snap, g_sym.strings, snap_sz);
    sym_unlock();

    /* Build TD_LIST of TD_ATOM_STR from snapshot */
    td_t* list = td_list_new((int64_t)count);
    if (!list || TD_IS_ERR(list)) {
        td_free(snap_block);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return TD_ERR_OOM;
    }

    for (uint32_t i = 0; i < count; i++) {
        list = td_list_append(list, snap[i]);
        if (!list || TD_IS_ERR(list)) {
            td_free(snap_block);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_OOM;
        }
    }
    td_free(snap_block);

    /* Save to temp file via td_col_save (writes STRL format) */
    err = td_col_save(list, tmp_path);
    td_release(list);
    if (err != TD_OK) {
        remove(tmp_path);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return err;
    }

    /* Fsync temp file for durability */
    td_fd_t tmp_fd = td_file_open(tmp_path, TD_OPEN_READ | TD_OPEN_WRITE);
    if (tmp_fd == TD_FD_INVALID) {
        remove(tmp_path);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return TD_ERR_IO;
    }
    err = td_file_sync(tmp_fd);
    td_file_close(tmp_fd);
    if (err != TD_OK) {
        remove(tmp_path);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return err;
    }

    /* Atomic rename: tmp -> final path */
    err = td_file_rename(tmp_path, path);
    if (err != TD_OK) {
        remove(tmp_path);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return err;
    }

    /* Fsync parent directory so the new directory entry is durable.
     * Without this, a crash after rename can lose the new file. */
    err = td_file_sync_dir(path);
    if (err != TD_OK) {
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return err;
    }

    /* Update persisted count */
    sym_lock();
    g_sym.persisted_count = count;
    sym_unlock();

    td_file_unlock(lock_fd);
    td_file_close(lock_fd);
    return TD_OK;
}

/* --------------------------------------------------------------------------
 * td_sym_load -- load symbol table from TD_LIST file (STRL format)
 *
 * Uses td_col_load to read the list, then interns entries beyond what's
 * already in memory.  File locking prevents reading a partial write.
 * -------------------------------------------------------------------------- */

td_err_t td_sym_load(const char* path) {
    if (!path) return TD_ERR_IO;
    if (!atomic_load_explicit(&g_sym_inited, memory_order_acquire)) return TD_ERR_IO;

    /* Acquire cross-process shared lock.
     * Try read-only open first so that read-only users (snapshots, read-only
     * mounts) can load without write permission on the directory.  Fall back
     * to read-write+create if the lock file doesn't exist yet.  If both fail,
     * only proceed without locking on read-only filesystem (EROFS) — other
     * errors (EMFILE, ENFILE, EACCES on writable fs, etc.) are real failures
     * that would silently drop the shared-lock guarantee. */
    char lock_path[1024];
    if (snprintf(lock_path, sizeof(lock_path), "%s.lk", path) >= (int)sizeof(lock_path))
        return TD_ERR_IO;
    td_fd_t lock_fd = td_file_open(lock_path, TD_OPEN_READ);
    if (lock_fd == TD_FD_INVALID) {
        int saved_errno = errno;
        lock_fd = td_file_open(lock_path, TD_OPEN_READ | TD_OPEN_WRITE | TD_OPEN_CREATE);
        if (lock_fd == TD_FD_INVALID) {
            /* Only proceed unlocked on read-only filesystem (EROFS) where
             * concurrent writes are impossible.  All other failures are
             * real errors that should not be silently ignored. */
            if (saved_errno != EROFS && errno != EROFS)
                return TD_ERR_IO;
        }
    }
    if (lock_fd != TD_FD_INVALID) {
        td_err_t err = td_file_lock_sh(lock_fd);
        if (err != TD_OK) { td_file_close(lock_fd); return err; }
    }

    /* Load the sym file as a TD_LIST of TD_ATOM_STR */
    td_t* list = td_col_load(path);
    if (!list || TD_IS_ERR(list)) {
        td_err_t code = TD_IS_ERR(list) ? TD_ERR_CODE(list) : TD_ERR_IO;
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return code;
    }

    if (list->type != TD_LIST || list->len > UINT32_MAX) {
        td_release(list);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return TD_ERR_CORRUPT;
    }

    /* Validate existing entries match, then intern remaining.
     * Use persisted_count (not str_count) as the already-loaded prefix:
     * runtime code may td_sym_intern transient names that were never
     * persisted, and those must not participate in prefix validation
     * or affect the intern start offset. */
    sym_lock();
    uint32_t already = g_sym.persisted_count;
    sym_unlock();
    td_t** slots = (td_t**)td_data(list);

    /* Reject stale/truncated sym file: if disk has fewer entries than what
     * we previously loaded from disk, the file is outdated or truncated. */
    if (already > 0 && list->len < (int64_t)already) {
        td_release(list);
        td_file_unlock(lock_fd);
        td_file_close(lock_fd);
        return TD_ERR_CORRUPT;
    }

    /* Validate entries [0..already-1] match the persisted prefix */
    for (int64_t i = 0; i < (int64_t)already && i < list->len; i++) {
        td_t* s = slots[i];
        if (!s || TD_IS_ERR(s) || s->type != TD_ATOM_STR) {
            td_release(list);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_CORRUPT;
        }
        td_t* mem_s = td_sym_str(i);
        if (!mem_s || td_str_len(mem_s) != td_str_len(s) ||
            memcmp(td_str_ptr(mem_s), td_str_ptr(s), td_str_len(s)) != 0) {
            td_release(list);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_CORRUPT;
        }
    }

    /* Intern entries beyond what's already in memory.
     * Verify each entry's in-memory ID matches its disk position:
     * if transient runtime-interned symbols already occupy these
     * slots, the disk entries would get wrong IDs, causing TD_SYM
     * columns to resolve the wrong strings. */
    for (int64_t i = (int64_t)already; i < list->len; i++) {
        td_t* s = slots[i];
        if (!s || TD_IS_ERR(s) || s->type != TD_ATOM_STR) {
            td_release(list);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_CORRUPT;
        }
        int64_t id = td_sym_intern(td_str_ptr(s), td_str_len(s));
        if (id < 0) {
            td_release(list);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_OOM;
        }
        if (id != i) {
            /* ID mismatch: disk position i was assigned in-memory
             * id != i, meaning a transient symbol occupies the slot.
             * The sym table has diverged from disk; continuing would
             * cause TD_SYM columns to resolve wrong strings. */
            td_release(list);
            td_file_unlock(lock_fd);
            td_file_close(lock_fd);
            return TD_ERR_CORRUPT;
        }
    }

    /* Update persisted count to reflect what is actually on disk.
     * Use list->len (not str_count) because transient runtime-interned
     * symbols may exist beyond the persisted prefix. */
    sym_lock();
    g_sym.persisted_count = (uint32_t)list->len;
    sym_unlock();

    td_release(list);
    td_file_unlock(lock_fd);
    td_file_close(lock_fd);
    return TD_OK;
}
