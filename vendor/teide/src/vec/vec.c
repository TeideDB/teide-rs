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

#include "vec.h"
#include <string.h>

/* --------------------------------------------------------------------------
 * Capacity helpers
 *
 * A vector's capacity is determined by its buddy order:
 *   capacity = (2^order - 32) / elem_size
 * When len reaches capacity, realloc to next power-of-2 data size.
 * -------------------------------------------------------------------------- */

static int64_t vec_capacity(td_t* vec) {
    size_t block_size = (size_t)1 << vec->order;
    size_t data_space = block_size - 32;  /* 32B td_t header */
    uint8_t esz = td_sym_elem_size(vec->type, vec->attrs);
    if (esz == 0) return 0;
    return (int64_t)(data_space / esz);
}

/* --------------------------------------------------------------------------
 * td_vec_new
 * -------------------------------------------------------------------------- */

td_t* td_vec_new(int8_t type, int64_t capacity) {
    if (type <= 0 || type >= TD_TYPE_COUNT)
        return TD_ERR_PTR(TD_ERR_TYPE);
    if (type == TD_SYM)
        return td_sym_vec_new(TD_SYM_W64, capacity);  /* default: global sym IDs */
    if (capacity < 0) return TD_ERR_PTR(TD_ERR_RANGE);

    uint8_t esz = td_elem_size(type);
    size_t data_size = (size_t)capacity * esz;
    if (esz > 1 && data_size / esz != (size_t)capacity)
        return TD_ERR_PTR(TD_ERR_OOM);

    td_t* v = td_alloc(data_size);
    if (!v || TD_IS_ERR(v)) return v;

    v->type = type;
    v->len = 0;
    v->attrs = 0;
    memset(v->nullmap, 0, 16);
    if (type == TD_STR) v->str_pool = NULL;

    return v;
}

/* --------------------------------------------------------------------------
 * td_sym_vec_new — create a TD_SYM vector with adaptive index width
 *
 * sym_width: TD_SYM_W8, TD_SYM_W16, TD_SYM_W32, or TD_SYM_W64
 * capacity:  number of elements (rows)
 * -------------------------------------------------------------------------- */

td_t* td_sym_vec_new(uint8_t sym_width, int64_t capacity) {
    if ((sym_width & ~TD_SYM_W_MASK) != 0)
        return TD_ERR_PTR(TD_ERR_TYPE);
    if (capacity < 0) return TD_ERR_PTR(TD_ERR_RANGE);

    uint8_t esz = (uint8_t)TD_SYM_ELEM(sym_width);
    size_t data_size = (size_t)capacity * esz;
    if (esz > 1 && data_size / esz != (size_t)capacity)
        return TD_ERR_PTR(TD_ERR_OOM);

    td_t* v = td_alloc(data_size);
    if (!v || TD_IS_ERR(v)) return v;

    v->type = TD_SYM;
    v->len = 0;
    v->attrs = sym_width;  /* lower 2 bits encode width */
    memset(v->nullmap, 0, 16);

    return v;
}

/* --------------------------------------------------------------------------
 * td_vec_append
 * -------------------------------------------------------------------------- */

td_t* td_vec_append(td_t* vec, const void* elem) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type <= 0 || vec->type >= TD_TYPE_COUNT)
        return TD_ERR_PTR(TD_ERR_TYPE);
    if (vec->type == TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);

    /* COW: if shared, copy first */
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    uint8_t esz = td_sym_elem_size(vec->type, vec->attrs);
    int64_t cap = vec_capacity(vec);

    /* Grow if needed */
    if (vec->len >= cap) {
        size_t new_data_size = (size_t)(vec->len + 1) * esz;
        /* Round up to next power of 2 block */
        if (new_data_size < 32) new_data_size = 32;
        else {
            size_t s = 32;
            while (s < new_data_size) {
                if (s > SIZE_MAX / 2) return TD_ERR_PTR(TD_ERR_OOM);
                s *= 2;
            }
            new_data_size = s;
        }
        td_t* new_vec = td_scratch_realloc(vec, new_data_size);
        if (!new_vec || TD_IS_ERR(new_vec)) return new_vec;
        vec = new_vec;
    }

    /* Append element */
    char* dst = (char*)td_data(vec) + vec->len * esz;
    memcpy(dst, elem, esz);
    vec->len++;

    return vec;
}

/* --------------------------------------------------------------------------
 * td_vec_set
 * -------------------------------------------------------------------------- */

td_t* td_vec_set(td_t* vec, int64_t idx, const void* elem) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type == TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (idx < 0 || idx >= vec->len)
        return TD_ERR_PTR(TD_ERR_RANGE);

    /* COW: if shared, copy first */
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;

    uint8_t esz = td_sym_elem_size(vec->type, vec->attrs);
    char* dst = (char*)td_data(vec) + idx * esz;
    memcpy(dst, elem, esz);

    return vec;
}

/* --------------------------------------------------------------------------
 * td_vec_get
 * -------------------------------------------------------------------------- */

void* td_vec_get(td_t* vec, int64_t idx) {
    if (!vec || TD_IS_ERR(vec)) return NULL;
    if (vec->type == TD_STR) return NULL;

    /* Slice path: redirect to parent */
    if (vec->attrs & TD_ATTR_SLICE) {
        td_t* parent = vec->slice_parent;
        int64_t offset = vec->slice_offset;
        if (idx < 0 || idx >= vec->len) return NULL;
        uint8_t esz = td_sym_elem_size(parent->type, parent->attrs);
        return (char*)td_data(parent) + (offset + idx) * esz;
    }

    if (idx < 0 || idx >= vec->len) return NULL;
    uint8_t esz = td_sym_elem_size(vec->type, vec->attrs);
    return (char*)td_data(vec) + idx * esz;
}

/* --------------------------------------------------------------------------
 * td_vec_slice  (zero-copy view)
 * -------------------------------------------------------------------------- */

td_t* td_vec_slice(td_t* vec, int64_t offset, int64_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (offset < 0 || len < 0 || offset > vec->len || len > vec->len - offset)
        return TD_ERR_PTR(TD_ERR_RANGE);

    /* If input is already a slice, resolve to ultimate parent */
    td_t* parent = vec;
    int64_t parent_offset = offset;
    if (vec->attrs & TD_ATTR_SLICE) {
        parent = vec->slice_parent;
        parent_offset = vec->slice_offset + offset;
    }

    /* Allocate a header-only block for the slice view */
    td_t* s = td_alloc(0);
    if (!s || TD_IS_ERR(s)) return s;

    s->type = parent->type;
    s->attrs = TD_ATTR_SLICE | (parent->attrs & TD_SYM_W_MASK);
    s->len = len;
    s->slice_parent = parent;
    s->slice_offset = parent_offset;

    /* Retain the parent so it stays alive */
    td_retain(parent);

    return s;
}

/* --------------------------------------------------------------------------
 * td_vec_concat
 * -------------------------------------------------------------------------- */

td_t* td_vec_concat(td_t* a, td_t* b) {
    if (!a || TD_IS_ERR(a)) return a;
    if (!b || TD_IS_ERR(b)) return b;
    if (a->type != b->type)
        return TD_ERR_PTR(TD_ERR_TYPE);

    if (a->type == TD_STR) {
        int64_t total_len = a->len + b->len;
        if (total_len < a->len) return TD_ERR_PTR(TD_ERR_OOM);

        td_t* result = td_vec_new(TD_STR, total_len);
        if (!result || TD_IS_ERR(result)) return result;
        result->len = total_len;

        td_str_t* dst = (td_str_t*)td_data(result);

        /* Resolve a's data (may be a slice) */
        const td_str_t* a_elems = (a->attrs & TD_ATTR_SLICE)
            ? &((const td_str_t*)td_data(a->slice_parent))[a->slice_offset]
            : (const td_str_t*)td_data(a);
        td_t* a_pool_owner = (a->attrs & TD_ATTR_SLICE) ? a->slice_parent : a;

        /* Resolve b's data (may be a slice) */
        const td_str_t* b_elems = (b->attrs & TD_ATTR_SLICE)
            ? &((const td_str_t*)td_data(b->slice_parent))[b->slice_offset]
            : (const td_str_t*)td_data(b);
        td_t* b_pool_owner = (b->attrs & TD_ATTR_SLICE) ? b->slice_parent : b;

        /* Copy a's elements as-is */
        memcpy(dst, a_elems, (size_t)a->len * sizeof(td_str_t));

        /* Merge pools: a's pool + b's pool */
        int64_t a_pool_size = (a_pool_owner->str_pool) ? a_pool_owner->str_pool->len : 0;
        int64_t b_pool_size = (b_pool_owner->str_pool) ? b_pool_owner->str_pool->len : 0;
        int64_t total_pool = a_pool_size + b_pool_size;

        /* Guard: total pool must fit in uint32_t for pool_off rebasing */
        if (total_pool > (int64_t)UINT32_MAX) {
            td_release(result);
            return TD_ERR_PTR(TD_ERR_RANGE);
        }

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
                memcpy(pool_dst, td_data(a_pool_owner->str_pool), (size_t)a_pool_size);
            if (b_pool_size > 0)
                memcpy(pool_dst + a_pool_size, td_data(b_pool_owner->str_pool), (size_t)b_pool_size);
        }

        /* Copy b's elements, rebasing pool offsets */
        for (int64_t i = 0; i < b->len; i++) {
            dst[a->len + i] = b_elems[i];
            if (!td_str_is_inline(&b_elems[i]) && b_elems[i].len > 0) {
                dst[a->len + i].pool_off += (uint32_t)a_pool_size;
            }
        }

        /* Propagate null bitmaps from a and b.
         * Slices don't carry TD_ATTR_HAS_NULLS — check TD_ATTR_SLICE too. */
        if ((a->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_SLICE)) ||
            (b->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_SLICE))) {
            for (int64_t i = 0; i < a->len; i++) {
                if (td_vec_is_null((td_t*)a, i))
                    td_vec_set_null(result, i, true);
            }
            for (int64_t i = 0; i < b->len; i++) {
                if (td_vec_is_null((td_t*)b, i))
                    td_vec_set_null(result, a->len + i, true);
            }
        }

        return result;
    }

    uint8_t a_esz = td_sym_elem_size(a->type, a->attrs);
    uint8_t b_esz = td_sym_elem_size(b->type, b->attrs);
    /* Use the wider of the two widths for SYM columns — carry only width bits,
     * not flags like TD_ATTR_SLICE or TD_ATTR_HAS_NULLS from inputs. */
    uint8_t out_attrs = (a_esz >= b_esz) ? (a->attrs & TD_SYM_W_MASK) : (b->attrs & TD_SYM_W_MASK);
    uint8_t esz = (a_esz >= b_esz) ? a_esz : b_esz;

    int64_t total_len = a->len + b->len;
    if (total_len < a->len) return TD_ERR_PTR(TD_ERR_OOM); /* overflow */
    size_t data_size = (size_t)total_len * esz;
    if (esz > 1 && data_size / esz != (size_t)total_len)
        return TD_ERR_PTR(TD_ERR_OOM); /* multiplication overflow */

    td_t* result = td_alloc(data_size);
    if (!result || TD_IS_ERR(result)) return result;

    result->type = a->type;
    result->len = total_len;
    result->attrs = out_attrs;
    memset(result->nullmap, 0, 16);

    /* For SYM with mismatched widths, widen element-by-element */
    if (a->type == TD_SYM && a_esz != b_esz) {
        void* dst = td_data(result);
        for (int64_t i = 0; i < a->len; i++) {
            int64_t val = td_read_sym(td_data(a), i, a->type, a->attrs);
            td_write_sym(dst, i, (uint64_t)val, result->type, result->attrs);
        }
        for (int64_t i = 0; i < b->len; i++) {
            int64_t val = td_read_sym(td_data(b), i, b->type, b->attrs);
            td_write_sym(dst, a->len + i, (uint64_t)val, result->type, result->attrs);
        }
    } else {
        /* Same width: fast memcpy path */
        void* a_data = (a->attrs & TD_ATTR_SLICE) ?
            ((char*)td_data(a->slice_parent) + a->slice_offset * esz) :
            td_data(a);
        memcpy(td_data(result), a_data, (size_t)a->len * esz);

        void* b_data = (b->attrs & TD_ATTR_SLICE) ?
            ((char*)td_data(b->slice_parent) + b->slice_offset * esz) :
            td_data(b);
        memcpy((char*)td_data(result) + (size_t)a->len * esz, b_data,
               (size_t)b->len * esz);
    }

    /* Propagate null bitmaps from a and b.
     * Slices don't carry TD_ATTR_HAS_NULLS — check TD_ATTR_SLICE too. */
    if ((a->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_SLICE)) ||
        (b->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_SLICE))) {
        for (int64_t i = 0; i < a->len; i++) {
            if (td_vec_is_null((td_t*)a, i))
                td_vec_set_null(result, i, true);
        }
        for (int64_t i = 0; i < b->len; i++) {
            if (td_vec_is_null((td_t*)b, i))
                td_vec_set_null(result, a->len + i, true);
        }
    }

    /* LIST/TABLE columns hold child pointers — retain them */
    if (a->type == TD_LIST || a->type == TD_TABLE) {
        td_t** ptrs = (td_t**)td_data(result);
        for (int64_t i = 0; i < total_len; i++) {
            if (ptrs[i]) td_retain(ptrs[i]);
        }
    }

    return result;
}

/* --------------------------------------------------------------------------
 * td_vec_from_raw
 * -------------------------------------------------------------------------- */

td_t* td_vec_from_raw(int8_t type, const void* data, int64_t count) {
    if (type <= 0 || type >= TD_TYPE_COUNT)
        return TD_ERR_PTR(TD_ERR_TYPE);
    if (type == TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (count < 0) return TD_ERR_PTR(TD_ERR_RANGE);

    /* TD_SYM defaults to W64 (global sym IDs) */
    uint8_t sym_w = (type == TD_SYM) ? TD_SYM_W64 : 0;
    uint8_t esz = td_sym_elem_size(type, sym_w);
    size_t data_size = (size_t)count * esz;

    td_t* v = td_alloc(data_size);
    if (!v || TD_IS_ERR(v)) return v;

    v->type = type;
    v->len = count;
    v->attrs = sym_w;
    memset(v->nullmap, 0, 16);

    memcpy(td_data(v), data, data_size);

    /* LIST/TABLE elements are child pointers — retain them */
    if (type == TD_LIST || type == TD_TABLE) {
        td_t** ptrs = (td_t**)td_data(v);
        for (int64_t i = 0; i < count; i++) {
            if (ptrs[i]) td_retain(ptrs[i]);
        }
    }

    return v;
}

/* --------------------------------------------------------------------------
 * Null bitmap operations
 *
 * Inline: for vectors with <=128 elements, bits stored in nullmap[16] (128 bits).
 * External: for >128 elements, allocate a U8 vector bitmap via ext_nullmap.
 * -------------------------------------------------------------------------- */

void td_vec_set_null(td_t* vec, int64_t idx, bool is_null) {
    if (!vec || TD_IS_ERR(vec)) return;
    if (vec->attrs & TD_ATTR_SLICE) return; /* cannot set null on slice — COW first */
    if (idx < 0 || idx >= vec->len) return;

    /* Mark HAS_NULLS if setting a null (defer for TD_STR until ext alloc succeeds) */
    if (is_null && vec->type != TD_STR) vec->attrs |= TD_ATTR_HAS_NULLS;

    if (!(vec->attrs & TD_ATTR_NULLMAP_EXT)) {
        /* TD_STR uses bytes 8-15 for str_pool — must skip inline nullmap
         * and promote to external immediately to avoid aliasing corruption */
        if (vec->type != TD_STR && idx < 128) {
            /* Inline nullmap path (<=128 elements, non-STR types) */
            int byte_idx = (int)(idx / 8);
            int bit_idx = (int)(idx % 8);
            if (is_null)
                vec->nullmap[byte_idx] |= (uint8_t)(1u << bit_idx);
            else
                vec->nullmap[byte_idx] &= (uint8_t)~(1u << bit_idx);
            return;
        }
        /* Need to promote to external nullmap */
        int64_t bitmap_len = (vec->len + 7) / 8;
        td_t* ext = td_vec_new(TD_U8, bitmap_len);
        if (!ext || TD_IS_ERR(ext)) return;
        ext->len = bitmap_len;
        if (vec->type == TD_STR) {
            /* TD_STR: nullmap bytes contain str_ext_null/str_pool, not bits */
            memset(td_data(ext), 0, (size_t)bitmap_len);
        } else {
            /* Copy existing inline bits */
            memcpy(td_data(ext), vec->nullmap, 16);
            /* Zero remaining bytes */
            if (bitmap_len > 16)
                memset((char*)td_data(ext) + 16, 0, (size_t)(bitmap_len - 16));
        }
        vec->attrs |= TD_ATTR_NULLMAP_EXT;
        if (is_null) vec->attrs |= TD_ATTR_HAS_NULLS;
        vec->ext_nullmap = ext;
    }

    /* External nullmap path */
    td_t* ext = vec->ext_nullmap;
    /* Grow external bitmap if needed */
    int64_t needed_bytes = (idx / 8) + 1;
    if (needed_bytes > ext->len) {
        int64_t new_len = (vec->len + 7) / 8;
        if (new_len < needed_bytes) new_len = needed_bytes;
        size_t new_data_size = (size_t)new_len;
        int64_t old_len = ext->len;
        td_t* new_ext = td_scratch_realloc(ext, new_data_size);
        if (!new_ext || TD_IS_ERR(new_ext)) return;
        /* Zero new bytes */
        if (new_len > old_len)
            memset((char*)td_data(new_ext) + old_len, 0,
                   (size_t)(new_len - old_len));
        new_ext->len = new_len;
        vec->ext_nullmap = new_ext;
        ext = new_ext;
    }

    uint8_t* bits = (uint8_t*)td_data(ext);
    int byte_idx = (int)(idx / 8);
    int bit_idx = (int)(idx % 8);
    if (is_null)
        bits[byte_idx] |= (uint8_t)(1u << bit_idx);
    else
        bits[byte_idx] &= (uint8_t)~(1u << bit_idx);
}

/* --------------------------------------------------------------------------
 * str_pool_cow — ensure pool is privately owned after td_cow()
 *
 * After td_cow(), the copy shares the same str_pool as the original.
 * td_retain_owned_refs bumps pool rc, so direct mutation would corrupt
 * the original's pool data (or td_scratch_realloc would td_free a
 * shared block).  Deep-copy the pool when rc > 1.
 * -------------------------------------------------------------------------- */

static td_t* str_pool_cow(td_t* vec) {
    if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) return vec;
    uint32_t pool_rc = atomic_load_explicit(&vec->str_pool->rc, memory_order_acquire);
    if (pool_rc <= 1) return vec;

    size_t pool_data_size = ((size_t)1 << vec->str_pool->order) - 32;
    td_t* new_pool = td_alloc(pool_data_size);
    if (!new_pool || TD_IS_ERR(new_pool)) return NULL;

    size_t copy_bytes = (size_t)vec->str_pool->len;
    if (copy_bytes > pool_data_size) copy_bytes = pool_data_size;

    uint8_t saved_order = new_pool->order;
    uint8_t saved_mmod  = new_pool->mmod;
    memcpy(new_pool, vec->str_pool, 32 + copy_bytes);
    new_pool->order = saved_order;
    new_pool->mmod  = saved_mmod;
    atomic_store_explicit(&new_pool->rc, 1, memory_order_relaxed);

    td_release(vec->str_pool);
    vec->str_pool = new_pool;
    return vec;
}

/* --------------------------------------------------------------------------
 * String pool dead-byte tracking
 *
 * Dead bytes are stored as a uint32_t in the pool block's nullmap[0..3],
 * which is otherwise unused (the pool is a raw CHAR vector).
 * -------------------------------------------------------------------------- */

static inline uint32_t str_pool_dead(td_t* vec) {
    if (!vec->str_pool) return 0;
    uint32_t d;
    memcpy(&d, vec->str_pool->nullmap, 4);
    return d;
}

static inline void str_pool_add_dead(td_t* vec, uint32_t bytes) {
    uint32_t d = str_pool_dead(vec);
    d = (d > UINT32_MAX - bytes) ? UINT32_MAX : d + bytes;
    memcpy(vec->str_pool->nullmap, &d, 4);
}

/* --------------------------------------------------------------------------
 * td_str_vec_append — append a string to a TD_STR vector
 *
 * Strings <= 12 bytes are inlined in the td_str_t element.
 * Strings > 12 bytes store a 4-byte prefix + offset into a growable pool.
 * -------------------------------------------------------------------------- */

td_t* td_str_vec_append(td_t* vec, const char* s, size_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (len > UINT32_MAX) return TD_ERR_PTR(TD_ERR_RANGE);

    td_t* original = vec;
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (!str_pool_cow(vec)) goto fail_oom;

    int64_t pool_off = 0;
    if (len > TD_STR_INLINE_MAX) {
        if (!vec->str_pool) {
            size_t init_pool = len < 256 ? 256 : len * 2;
            vec->str_pool = td_alloc(init_pool);
            if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) {
                vec->str_pool = NULL;
                goto fail_oom;
            }
            vec->str_pool->type = TD_CHAR;
            vec->str_pool->len = 0;
        }

        int64_t pool_used = vec->str_pool->len;
        size_t pool_cap = ((size_t)1 << vec->str_pool->order) - 32;
        if ((size_t)pool_used + len > pool_cap) {
            size_t need = (size_t)pool_used + len;
            size_t new_cap = pool_cap;
            if (new_cap == 0) new_cap = 256;
            while (new_cap < need) {
                if (new_cap > SIZE_MAX / 2) goto fail_oom;
                new_cap *= 2;
            }
            td_t* np = td_scratch_realloc(vec->str_pool, new_cap);
            if (!np || TD_IS_ERR(np)) goto fail_oom;
            vec->str_pool = np;
        }

        if ((uint64_t)pool_used > UINT32_MAX) goto fail_range;
        pool_off = pool_used;
    }

    /* Grow element array if needed — pool is already ready */
    int64_t cap = vec_capacity(vec);
    if (vec->len >= cap) {
        size_t new_data_size = (size_t)(vec->len + 1) * sizeof(td_str_t);
        if (new_data_size < 32) new_data_size = 32;
        else {
            size_t s2 = 32;
            while (s2 < new_data_size) {
                if (s2 > SIZE_MAX / 2) goto fail_oom;
                s2 *= 2;
            }
            new_data_size = s2;
        }
        td_t* nv = td_scratch_realloc(vec, new_data_size);
        if (!nv || TD_IS_ERR(nv)) goto fail_oom;
        vec = nv;
    }

    td_str_t* elem = &((td_str_t*)td_data(vec))[vec->len];
    memset(elem, 0, sizeof(td_str_t));
    elem->len = (uint32_t)len;

    if (len <= TD_STR_INLINE_MAX) {
        if (len > 0) memcpy(elem->data, s, len);
    } else {
        /* Copy string into pool (already allocated above) */
        char* pool_base = (char*)td_data(vec->str_pool);
        memcpy(pool_base + pool_off, s, len);

        memcpy(elem->prefix, s, 4);
        elem->pool_off = (uint32_t)pool_off;
        vec->str_pool->len = pool_off + (int64_t)len;
    }

    vec->len++;
    return vec;

fail_oom:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_OOM);
fail_range:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_RANGE);
}

/* --------------------------------------------------------------------------
 * td_str_vec_get — read a string from a TD_STR vector by index
 *
 * Returns a pointer to the string data (inline or pool) and sets *out_len.
 * Returns NULL for invalid input or out-of-bounds index.
 * -------------------------------------------------------------------------- */

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

    /* Pooled: resolve via pool */
    if (!data_owner->str_pool) return NULL;
    return (const char*)td_data(data_owner->str_pool) + elem->pool_off;
}

/* --------------------------------------------------------------------------
 * td_str_vec_set — update string at index in a TD_STR vector
 *
 * Overwrites element at idx. Old pooled bytes become dead space (reclaimed
 * by td_str_vec_compact). New pooled strings are appended to the pool.
 * -------------------------------------------------------------------------- */

td_t* td_str_vec_set(td_t* vec, int64_t idx, const char* s, size_t len) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (idx < 0 || idx >= vec->len) return TD_ERR_PTR(TD_ERR_RANGE);
    if (len > UINT32_MAX) return TD_ERR_PTR(TD_ERR_RANGE);

    td_t* original = vec;
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (!str_pool_cow(vec)) goto fail_oom;

    td_str_t* elem = &((td_str_t*)td_data(vec))[idx];

    if (len <= TD_STR_INLINE_MAX) {
        /* Track dead bytes if old string was pooled */
        if (!td_str_is_inline(elem) && elem->len > 0 && vec->str_pool) {
            str_pool_add_dead(vec, elem->len);
        }
        memset(elem, 0, sizeof(td_str_t));
        elem->len = (uint32_t)len;
        if (len > 0) memcpy(elem->data, s, len);
    } else {
        if (!vec->str_pool) {
            size_t init_pool = len < 256 ? 256 : len * 2;
            vec->str_pool = td_alloc(init_pool);
            if (!vec->str_pool || TD_IS_ERR(vec->str_pool)) {
                vec->str_pool = NULL;
                goto fail_oom;
            }
            vec->str_pool->type = TD_CHAR;
            vec->str_pool->len = 0;
        }

        /* Grow pool if needed */
        int64_t pool_used = vec->str_pool->len;
        size_t pool_cap = ((size_t)1 << vec->str_pool->order) - 32;
        if ((size_t)pool_used + len > pool_cap) {
            size_t need = (size_t)pool_used + len;
            size_t new_cap = pool_cap;
            if (new_cap == 0) new_cap = 256;
            while (new_cap < need) {
                if (new_cap > SIZE_MAX / 2) goto fail_oom;
                new_cap *= 2;
            }
            td_t* np = td_scratch_realloc(vec->str_pool, new_cap);
            if (!np || TD_IS_ERR(np)) goto fail_oom;
            vec->str_pool = np;
        }

        if ((uint64_t)pool_used > UINT32_MAX) goto fail_range;

        /* Pool alloc succeeded — now safe to modify the element */
        if (!td_str_is_inline(elem) && elem->len > 0 && vec->str_pool) {
            str_pool_add_dead(vec, elem->len);
        }

        char* pool_base = (char*)td_data(vec->str_pool);
        memcpy(pool_base + pool_used, s, len);
        memset(elem, 0, sizeof(td_str_t));
        elem->len = (uint32_t)len;
        memcpy(elem->prefix, s, 4);
        elem->pool_off = (uint32_t)pool_used;
        vec->str_pool->len = pool_used + (int64_t)len;
    }

    return vec;

fail_oom:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_OOM);
fail_range:
    if (vec != original) td_release(vec);
    return TD_ERR_PTR(TD_ERR_RANGE);
}

/* --------------------------------------------------------------------------
 * td_str_vec_compact — reclaim dead pool space
 *
 * Allocates a fresh pool containing only live pooled strings, updates
 * element offsets, and releases the old pool.
 * -------------------------------------------------------------------------- */

td_t* td_str_vec_compact(td_t* vec) {
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (vec->type != TD_STR) return TD_ERR_PTR(TD_ERR_TYPE);
    if (!vec->str_pool || str_pool_dead(vec) == 0) return vec;

    td_t* original = vec;
    vec = td_cow(vec);
    if (!vec || TD_IS_ERR(vec)) return vec;
    if (!str_pool_cow(vec)) {
        if (vec != original) td_release(vec);
        return TD_ERR_PTR(TD_ERR_OOM);
    }

    /* Compute true live size by scanning elements — avoids overflow when
     * the dead-byte counter (uint32_t) has saturated at UINT32_MAX. */
    td_str_t* elems = (td_str_t*)td_data(vec);
    size_t live_size = 0;
    for (int64_t i = 0; i < vec->len; i++) {
        if (td_vec_is_null(vec, i) || td_str_is_inline(&elems[i]) || elems[i].len == 0) continue;
        live_size += elems[i].len;
    }

    if (live_size == 0) {
        td_release(vec->str_pool);
        vec->str_pool = NULL;
        return vec;
    }

    td_t* new_pool = td_alloc(live_size);
    if (!new_pool || TD_IS_ERR(new_pool)) return vec;
    new_pool->type = TD_CHAR;
    new_pool->len = 0;
    memset(new_pool->nullmap, 0, 16);

    char* old_base = (char*)td_data(vec->str_pool);
    char* new_base = (char*)td_data(new_pool);
    uint32_t write_off = 0;

    for (int64_t i = 0; i < vec->len; i++) {
        if (td_vec_is_null(vec, i) || td_str_is_inline(&elems[i]) || elems[i].len == 0) continue;

        uint32_t slen = elems[i].len;
        memcpy(new_base + write_off, old_base + elems[i].pool_off, slen);
        elems[i].pool_off = write_off;
        write_off += slen;
    }

    new_pool->len = (int64_t)write_off;
    td_release(vec->str_pool);
    vec->str_pool = new_pool;

    return vec;
}

/* --------------------------------------------------------------------------
 * td_embedding_new — create a flat F32 vector for N*D embedding storage
 * -------------------------------------------------------------------------- */

td_t* td_embedding_new(int64_t nrows, int32_t dim) {
    int64_t total = nrows * (int64_t)dim;
    td_t* v = td_vec_new(TD_F32, total);
    if (!v || TD_IS_ERR(v)) return v;
    v->len = total;
    return v;
}

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
