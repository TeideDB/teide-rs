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

#include "arena.h"
#include "sys.h"
#include <string.h>

/* 32-byte alignment for td_t */
#define ARENA_ALIGN 32
#define ARENA_ALIGN_UP(x) (((x) + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1))

/* Each chunk is a contiguous block of memory with a bump pointer. */
typedef struct td_arena_chunk {
    struct td_arena_chunk* next;
    size_t cap;    /* usable capacity (excluding this header) */
    size_t used;   /* bytes used so far */
} td_arena_chunk_t;

/* Arena header */
struct td_arena {
    td_arena_chunk_t* chunks;     /* linked list of all chunks (head = current) */
    size_t            chunk_size; /* default chunk capacity */
};

/* Chunk data starts at aligned offset after the header */
static inline char* chunk_data(td_arena_chunk_t* c) {
    size_t hdr = ARENA_ALIGN_UP(sizeof(td_arena_chunk_t));
    return (char*)c + hdr;
}

static td_arena_chunk_t* arena_new_chunk(size_t min_cap) {
    size_t hdr = ARENA_ALIGN_UP(sizeof(td_arena_chunk_t));
    if (min_cap > SIZE_MAX - hdr) return NULL;
    size_t total = hdr + min_cap;
    td_arena_chunk_t* c = (td_arena_chunk_t*)td_sys_alloc(total);
    if (!c) return NULL;
    c->next = NULL;
    c->cap = min_cap;
    c->used = 0;
    return c;
}

td_arena_t* td_arena_new(size_t chunk_size) {
    if (chunk_size < 256) chunk_size = 256;
    chunk_size = ARENA_ALIGN_UP(chunk_size);

    td_arena_t* a = (td_arena_t*)td_sys_alloc(sizeof(td_arena_t));
    if (!a) return NULL;

    td_arena_chunk_t* first = arena_new_chunk(chunk_size);
    if (!first) {
        td_sys_free(a);
        return NULL;
    }

    a->chunks = first;
    a->chunk_size = chunk_size;
    return a;
}

td_t* td_arena_alloc(td_arena_t* arena, size_t nbytes) {
    if (!arena) return NULL;
    if (nbytes > SIZE_MAX - 32 - (ARENA_ALIGN - 1)) return NULL;
    size_t block_size = ARENA_ALIGN_UP(32 + nbytes);

    td_arena_chunk_t* c = arena->chunks;

    if (c->used + block_size > c->cap) {
        size_t new_cap = arena->chunk_size;
        if (block_size > new_cap) new_cap = ARENA_ALIGN_UP(block_size);

        td_arena_chunk_t* nc = arena_new_chunk(new_cap);
        if (!nc) return NULL;

        nc->next = arena->chunks;
        arena->chunks = nc;
        c = nc;
    }

    char* base = chunk_data(c);
    td_t* v = (td_t*)(base + c->used);
    c->used += block_size;

    memset(v, 0, 32);
    v->attrs = TD_ATTR_ARENA;
    v->rc = 1;

    return v;
}

void td_arena_reset(td_arena_t* arena) {
    if (!arena || !arena->chunks) return;

    /* Keep the head chunk (most recently allocated), free the rest */
    td_arena_chunk_t* keep = arena->chunks;
    td_arena_chunk_t* c = keep->next;
    while (c) {
        td_arena_chunk_t* next = c->next;
        td_sys_free(c);
        c = next;
    }
    keep->next = NULL;
    keep->used = 0;
    arena->chunks = keep;
}

void td_arena_destroy(td_arena_t* arena) {
    if (!arena) return;
    td_arena_chunk_t* c = arena->chunks;
    while (c) {
        td_arena_chunk_t* next = c->next;
        td_sys_free(c);
        c = next;
    }
    td_sys_free(arena);
}
