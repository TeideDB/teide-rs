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

#ifndef TD_ARENA_H
#define TD_ARENA_H

#include <teide/td.h>

typedef struct td_arena td_arena_t;

/* Create arena with given chunk size (bytes). Chunks allocated via td_sys_alloc. */
td_arena_t* td_arena_new(size_t chunk_size);

/* Allocate td_t* block with nbytes of data space.
 * Returns 32-byte aligned td_t* with TD_ATTR_ARENA set, rc=1.
 * Returns NULL on OOM. */
td_t* td_arena_alloc(td_arena_t* arena, size_t nbytes);

/* Reset arena — rewind all chunks to zero. Memory retained for reuse. */
void td_arena_reset(td_arena_t* arena);

/* Destroy arena — free all backing memory. */
void td_arena_destroy(td_arena_t* arena);

#endif /* TD_ARENA_H */
