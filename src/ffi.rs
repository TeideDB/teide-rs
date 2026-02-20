//   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
//   All rights reserved.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

// Raw FFI bindings to the Teide C17 engine
// Hand-written from include/teide/td.h
#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_void};
use std::sync::atomic::AtomicU32;

// ===== Type Constants =====

pub const TD_LIST: i8 = 0;
pub const TD_BOOL: i8 = 1;
pub const TD_U8: i8 = 2;
pub const TD_CHAR: i8 = 3;
pub const TD_I16: i8 = 4;
pub const TD_I32: i8 = 5;
pub const TD_I64: i8 = 6;
pub const TD_F64: i8 = 7;
pub const TD_DATE: i8 = 9;
pub const TD_TIME: i8 = 10;
pub const TD_TIMESTAMP: i8 = 11;
pub const TD_GUID: i8 = 12;
pub const TD_TABLE: i8 = 13;
pub const TD_SEL: i8 = 16;
pub const TD_SYM: i8 = 20;

pub const TD_TYPE_COUNT: usize = 21;

// Parted types
pub const TD_PARTED_BASE: i8 = 32;
pub const TD_MAPCOMMON: i8 = 64;

// MAPCOMMON inferred sub-types (stored in attrs field)
pub const TD_MC_SYM: u8 = 0;
pub const TD_MC_DATE: u8 = 1;
pub const TD_MC_I64: u8 = 2;

#[inline]
pub fn td_is_parted(t: i8) -> bool {
    (TD_PARTED_BASE..TD_MAPCOMMON).contains(&t)
}

#[inline]
pub fn td_parted_basetype(t: i8) -> i8 {
    t - TD_PARTED_BASE
}

// Atom variants (negative type tags)
pub const TD_ATOM_BOOL: i8 = -TD_BOOL;
pub const TD_ATOM_U8: i8 = -TD_U8;
pub const TD_ATOM_CHAR: i8 = -TD_CHAR;
pub const TD_ATOM_I16: i8 = -TD_I16;
pub const TD_ATOM_I32: i8 = -TD_I32;
pub const TD_ATOM_I64: i8 = -TD_I64;
pub const TD_ATOM_F64: i8 = -TD_F64;
pub const TD_ATOM_STR: i8 = -8;
pub const TD_ATOM_DATE: i8 = -TD_DATE;
pub const TD_ATOM_TIME: i8 = -TD_TIME;
pub const TD_ATOM_TIMESTAMP: i8 = -TD_TIMESTAMP;
pub const TD_ATOM_GUID: i8 = -TD_GUID;
pub const TD_ATOM_SYM: i8 = -TD_SYM;

// ===== Symbol Width Constants =====

pub const TD_SYM_W_MASK: u8 = 0x03;
pub const TD_SYM_W8: u8 = 0x00;
pub const TD_SYM_W16: u8 = 0x01;
pub const TD_SYM_W32: u8 = 0x02;
pub const TD_SYM_W64: u8 = 0x03;

#[inline]
pub fn td_is_sym(t: i8) -> bool {
    t == TD_SYM
}

/// Read a sym index at the correct width.
/// # Safety
/// `data` must point to valid column data, `row` must be in bounds.
#[inline]
pub unsafe fn read_sym(data: *const u8, row: usize, _t: i8, attrs: u8) -> i64 {
    match attrs & TD_SYM_W_MASK {
        TD_SYM_W8 => (unsafe { *data.add(row) }) as i64,
        TD_SYM_W16 => (unsafe { *(data as *const u16).add(row) }) as i64,
        TD_SYM_W32 => (unsafe { *(data as *const u32).add(row) }) as i64,
        _ => unsafe { *(data as *const i64).add(row) },
    }
}

// ===== Attribute Flags =====

pub const TD_ATTR_SLICE: u8 = 0x10;
pub const TD_ATTR_NULLMAP_EXT: u8 = 0x20;
pub const TD_ATTR_HAS_NULLS: u8 = 0x40;

// ===== Morsel Constants =====

pub const TD_MORSEL_ELEMS: i64 = 1024;

// ===== Slab Cache Constants =====

pub const TD_SLAB_CACHE_SIZE: usize = 64;
pub const TD_SLAB_ORDERS: usize = 5;

// ===== Buddy Allocator Constants =====

pub const TD_ORDER_MIN: u32 = 5;
pub const TD_ORDER_MAX: u32 = 30;

// ===== Parallel Threshold =====

pub const TD_PARALLEL_THRESHOLD: i64 = 64 * TD_MORSEL_ELEMS;
pub const TD_DISPATCH_MORSELS: u32 = 8;

// ===== Opcode Constants =====

// Sources
pub const OP_SCAN: u16 = 1;
pub const OP_CONST: u16 = 2;

// Unary element-wise (fuseable)
pub const OP_NEG: u16 = 10;
pub const OP_ABS: u16 = 11;
pub const OP_NOT: u16 = 12;
pub const OP_SQRT: u16 = 13;
pub const OP_LOG: u16 = 14;
pub const OP_EXP: u16 = 15;
pub const OP_CEIL: u16 = 16;
pub const OP_FLOOR: u16 = 17;
pub const OP_ISNULL: u16 = 18;
pub const OP_CAST: u16 = 19;

// Binary element-wise (fuseable)
pub const OP_ADD: u16 = 20;
pub const OP_SUB: u16 = 21;
pub const OP_MUL: u16 = 22;
pub const OP_DIV: u16 = 23;
pub const OP_MOD: u16 = 24;
pub const OP_EQ: u16 = 25;
pub const OP_NE: u16 = 26;
pub const OP_LT: u16 = 27;
pub const OP_LE: u16 = 28;
pub const OP_GT: u16 = 29;
pub const OP_GE: u16 = 30;
pub const OP_AND: u16 = 31;
pub const OP_OR: u16 = 32;
pub const OP_MIN2: u16 = 33;
pub const OP_MAX2: u16 = 34;
pub const OP_IF: u16 = 35;
pub const OP_LIKE: u16 = 36;
pub const OP_UPPER: u16 = 37;
pub const OP_LOWER: u16 = 38;
pub const OP_STRLEN: u16 = 39;
pub const OP_SUBSTR: u16 = 40;
pub const OP_REPLACE: u16 = 41;
pub const OP_TRIM: u16 = 42;
pub const OP_CONCAT: u16 = 43;
pub const OP_EXTRACT: u16 = 45;
pub const OP_DATE_TRUNC: u16 = 46;

// EXTRACT / DATE_TRUNC field identifiers
pub const TD_EXTRACT_YEAR: i64 = 0;
pub const TD_EXTRACT_MONTH: i64 = 1;
pub const TD_EXTRACT_DAY: i64 = 2;
pub const TD_EXTRACT_HOUR: i64 = 3;
pub const TD_EXTRACT_MINUTE: i64 = 4;
pub const TD_EXTRACT_SECOND: i64 = 5;
pub const TD_EXTRACT_DOW: i64 = 6;
pub const TD_EXTRACT_DOY: i64 = 7;
pub const TD_EXTRACT_EPOCH: i64 = 8;

// Reductions (pipeline breakers)
pub const OP_SUM: u16 = 50;
pub const OP_PROD: u16 = 51;
pub const OP_MIN: u16 = 52;
pub const OP_MAX: u16 = 53;
pub const OP_COUNT: u16 = 54;
pub const OP_AVG: u16 = 55;
pub const OP_FIRST: u16 = 56;
pub const OP_LAST: u16 = 57;
pub const OP_COUNT_DISTINCT: u16 = 58;
pub const OP_STDDEV: u16 = 59;

// Structural (pipeline breakers)
pub const OP_FILTER: u16 = 60;
pub const OP_SORT: u16 = 61;
pub const OP_GROUP: u16 = 62;
pub const OP_JOIN: u16 = 63;
pub const OP_WINDOW_JOIN: u16 = 64;
pub const OP_PROJECT: u16 = 65;
pub const OP_SELECT: u16 = 66;
pub const OP_HEAD: u16 = 67;
pub const OP_TAIL: u16 = 68;

// Misc
pub const OP_ALIAS: u16 = 70;
pub const OP_MATERIALIZE: u16 = 71;
pub const OP_WINDOW: u16 = 72;

// Statistical aggregates
pub const OP_STDDEV_POP: u16 = 73;
pub const OP_VAR: u16 = 74;
pub const OP_VAR_POP: u16 = 75;
pub const OP_ILIKE: u16 = 76;

// Window function kinds
pub const TD_WIN_ROW_NUMBER: u8 = 0;
pub const TD_WIN_RANK: u8 = 1;
pub const TD_WIN_DENSE_RANK: u8 = 2;
pub const TD_WIN_NTILE: u8 = 3;
pub const TD_WIN_SUM: u8 = 4;
pub const TD_WIN_AVG: u8 = 5;
pub const TD_WIN_MIN: u8 = 6;
pub const TD_WIN_MAX: u8 = 7;
pub const TD_WIN_COUNT: u8 = 8;
pub const TD_WIN_LAG: u8 = 9;
pub const TD_WIN_LEAD: u8 = 10;
pub const TD_WIN_FIRST_VALUE: u8 = 11;
pub const TD_WIN_LAST_VALUE: u8 = 12;
pub const TD_WIN_NTH_VALUE: u8 = 13;

// Window frame type
pub const TD_FRAME_ROWS: u8 = 0;
pub const TD_FRAME_RANGE: u8 = 1;

// Window frame bounds
pub const TD_BOUND_UNBOUNDED_PRECEDING: u8 = 0;
pub const TD_BOUND_N_PRECEDING: u8 = 1;
pub const TD_BOUND_CURRENT_ROW: u8 = 2;
pub const TD_BOUND_N_FOLLOWING: u8 = 3;
pub const TD_BOUND_UNBOUNDED_FOLLOWING: u8 = 4;

// Op flags
pub const OP_FLAG_FUSED: u8 = 0x01;
pub const OP_FLAG_DEAD: u8 = 0x02;

// ===== Error Handling =====

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum td_err_t {
    TD_OK = 0,
    TD_ERR_OOM = 1,
    TD_ERR_TYPE = 2,
    TD_ERR_RANGE = 3,
    TD_ERR_LENGTH = 4,
    TD_ERR_RANK = 5,
    TD_ERR_DOMAIN = 6,
    TD_ERR_NYI = 7,
    TD_ERR_IO = 8,
    TD_ERR_SCHEMA = 9,
    TD_ERR_CORRUPT = 10,
    TD_ERR_CANCEL = 11,
}

/// Equivalent to C macro: `TD_ERR_PTR(e)` — cast error code to pointer.
///
/// # Safety
/// The returned pointer is an encoded error sentinel and must never be dereferenced.
#[inline]
pub unsafe fn td_err_ptr(e: td_err_t) -> *mut td_t {
    e as usize as *mut td_t
}

/// Equivalent to C macro: `TD_IS_ERR(p)` — true if pointer is an error sentinel.
#[inline]
pub fn td_is_err(p: *const td_t) -> bool {
    (p as usize) < 32
}

/// Equivalent to C macro: `TD_ERR_CODE(p)` — extract error code from error pointer.
#[inline]
pub fn td_err_code(p: *const td_t) -> td_err_t {
    match p as usize as u32 {
        0 => td_err_t::TD_OK,
        1 => td_err_t::TD_ERR_OOM,
        2 => td_err_t::TD_ERR_TYPE,
        3 => td_err_t::TD_ERR_RANGE,
        4 => td_err_t::TD_ERR_LENGTH,
        5 => td_err_t::TD_ERR_RANK,
        6 => td_err_t::TD_ERR_DOMAIN,
        7 => td_err_t::TD_ERR_NYI,
        8 => td_err_t::TD_ERR_IO,
        9 => td_err_t::TD_ERR_SCHEMA,
        10 => td_err_t::TD_ERR_CORRUPT,
        11 => td_err_t::TD_ERR_CANCEL,
        _ => td_err_t::TD_ERR_CORRUPT,
    }
}

// ===== Core Type: td_t (32-byte block header) =====

/// Bytes 0-15 union: nullmap / slice / ext_nullmap
#[repr(C)]
#[derive(Copy, Clone)]
pub union td_t_head {
    pub nullmap: [u8; 16],
    pub slice: td_t_slice,
    pub ext: td_t_ext_nullmap,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_t_slice {
    pub slice_parent: *mut td_t,
    pub slice_offset: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_t_ext_nullmap {
    pub ext_nullmap: *mut td_t,
    pub _reserved: i64,
}

/// Bytes 24-31 value union
#[repr(C)]
#[derive(Copy, Clone)]
pub union td_t_val {
    pub b8: u8,
    pub u8_: u8,
    pub c8: c_char,
    pub i16_: i16,
    pub i32_: i32,
    pub u32_: u32,
    pub i64_: i64,
    pub f64_: c_double,
    pub obj: *mut td_t,
    pub sso: td_t_sso,
    pub len: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_t_sso {
    pub slen: u8,
    pub sdata: [c_char; 7],
}

/// The 32-byte block header. Layout must match C exactly.
///
/// Not `Copy`/`Clone` because the `rc` field is `AtomicU32` (matching the C
/// `_Atomic(uint32_t)`). This is fine — Rust code only ever handles `td_t`
/// through `*mut td_t` pointers, never by value.
#[repr(C)]
pub struct td_t {
    /// Bytes 0-15
    pub head: td_t_head,
    /// Byte 16
    pub mmod: u8,
    /// Byte 17
    pub order: u8,
    /// Byte 18
    pub type_: i8,
    /// Byte 19
    pub attrs: u8,
    /// Bytes 20-23: reference count.
    /// In C this is `_Atomic(uint32_t)`. We use `AtomicU32` to match the C
    /// type exactly. AtomicU32 has same size/alignment as u32 on all practical
    /// targets. Validated by the size assertion at the bottom of this file.
    /// All atomic operations on `rc` go through the C FFI (`td_retain`,
    /// `td_release`); Rust never manipulates this field directly.
    pub rc: AtomicU32,
    /// Bytes 24-31
    pub val: td_t_val,
}

// ===== Inline Accessors =====

/// Get the type tag of a td_t.
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_type(v: *const td_t) -> i8 {
    (*v).type_
}

/// True if the value is an atom (negative type).
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_is_atom(v: *const td_t) -> bool {
    (*v).type_ < 0
}

/// True if the value is a vector (positive type).
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_is_vec(v: *const td_t) -> bool {
    (*v).type_ > 0
}

/// Get the length (for vectors).
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_len(v: *const td_t) -> i64 {
    (*v).val.len
}

/// Get the attrs field of a td_t.
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_attrs(v: *const td_t) -> u8 {
    (*v).attrs
}

/// Get pointer to data payload (byte 32 onward).
///
/// # Safety
/// `v` must be a valid non-null pointer to a live `td_t`.
#[inline]
pub unsafe fn td_data(v: *mut td_t) -> *mut c_void {
    (v as *mut u8).add(32) as *mut c_void
}

/// Element size for a given type tag.
///
/// Returns 0 if `t` is out of range (instead of panicking), which is safe
/// because callers already treat 0 as an error indicator.
///
/// # Safety
/// Caller must ensure the C runtime is initialized so `td_type_sizes` is valid.
#[inline]
pub unsafe fn td_elem_size(t: i8) -> u8 {
    if (t as usize) >= TD_TYPE_COUNT {
        return 0;
    }
    td_type_sizes[t as usize]
}

// ===== Operation Node =====

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_op_t {
    pub opcode: u16,
    pub arity: u8,
    pub flags: u8,
    pub out_type: i8,
    pub pad: [u8; 3],
    pub id: u32,
    pub est_rows: u32,
    pub inputs: [*mut td_op_t; 2],
}

// Extended op node — opaque (104 bytes, complex unions).
// Size validated by compile-time assertion at bottom of ffi.rs (line ~840).
#[repr(C)]
pub struct td_op_ext_t {
    _opaque: [u8; 104],
}

// ===== Operation Graph =====

#[repr(C)]
pub struct td_graph_t {
    pub nodes: *mut td_op_t,
    pub node_count: u32,
    pub node_cap: u32,
    pub table: *mut td_t,
    pub ext_nodes: *mut *mut td_op_ext_t,
    pub ext_count: u32,
    pub ext_cap: u32,
    pub selection: *mut td_t,
}

// ===== Morsel Iterator =====

#[repr(C)]
pub struct td_morsel_t {
    pub vec: *mut td_t,
    pub offset: i64,
    pub len: i64,
    pub elem_size: u32,
    pub morsel_len: i64,
    pub morsel_ptr: *mut c_void,
    pub null_bits: *mut u8,
}

// ===== Executor Pipeline =====

#[repr(C)]
pub struct td_pipe_t {
    pub op: *mut td_op_t,
    pub inputs: [*mut td_pipe_t; 2],
    pub state: td_morsel_t,
    pub materialized: *mut td_t,
    pub spill_fd: c_int,
}

// ===== Memory Statistics =====

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct td_mem_stats_t {
    pub alloc_count: usize,
    pub free_count: usize,
    pub bytes_allocated: usize,
    pub peak_bytes: usize,
    pub slab_hits: usize,
    pub direct_count: usize,
    pub direct_bytes: usize,
    pub sys_current: usize,
    pub sys_peak: usize,
}

// ===== Opaque Forward Declarations =====

#[repr(C)]
pub struct td_heap_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct td_sym_table_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct td_sym_map_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct td_pool_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct td_task_t {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct td_dispatch_t {
    _opaque: [u8; 0],
}

// ===== Thread Types =====

#[cfg(target_os = "windows")]
pub type td_thread_t = *mut c_void;
#[cfg(not(target_os = "windows"))]
pub type td_thread_t = std::os::raw::c_ulong;

pub type td_thread_fn = Option<unsafe extern "C" fn(arg: *mut c_void)>;

// ===== Extern: type sizes table =====

extern "C" {
    pub static td_type_sizes: [u8; TD_TYPE_COUNT];
}

// ===== Extern "C" Function Declarations =====

extern "C" {
    // --- Error ---
    pub fn td_err_str(e: td_err_t) -> *const c_char;

    // --- Platform API ---
    pub fn td_vm_alloc(size: usize) -> *mut c_void;
    pub fn td_vm_free(ptr: *mut c_void, size: usize);
    pub fn td_vm_map_file(path: *const c_char, out_size: *mut usize) -> *mut c_void;
    pub fn td_vm_unmap_file(ptr: *mut c_void, size: usize);
    pub fn td_vm_advise_seq(ptr: *mut c_void, size: usize);
    pub fn td_vm_release(ptr: *mut c_void, size: usize);

    // --- Threading API ---
    pub fn td_thread_create(t: *mut td_thread_t, f: td_thread_fn, arg: *mut c_void) -> td_err_t;
    pub fn td_thread_join(t: td_thread_t) -> td_err_t;
    pub fn td_thread_count() -> u32;
    pub fn td_parallel_begin();
    pub fn td_parallel_end();
    pub fn td_heap_gc();

    // --- Memory Allocator API ---
    pub fn td_alloc(data_size: usize) -> *mut td_t;
    pub fn td_free(v: *mut td_t);
    pub fn td_alloc_copy(v: *mut td_t) -> *mut td_t;
    pub fn td_scratch_alloc(data_size: usize) -> *mut td_t;
    pub fn td_scratch_realloc(v: *mut td_t, new_data_size: usize) -> *mut td_t;
    pub fn td_heap_init();
    pub fn td_heap_destroy();
    pub fn td_heap_merge(src: *mut td_heap_t);
    pub fn td_mem_stats(out: *mut td_mem_stats_t);

    // --- COW / Ref Counting API ---
    pub fn td_retain(v: *mut td_t);
    pub fn td_release(v: *mut td_t);
    pub fn td_cow(v: *mut td_t) -> *mut td_t;

    // --- Atom Constructors ---
    pub fn td_bool(val: bool) -> *mut td_t;
    pub fn td_u8(val: u8) -> *mut td_t;
    pub fn td_char(val: c_char) -> *mut td_t;
    pub fn td_i16(val: i16) -> *mut td_t;
    pub fn td_i32(val: i32) -> *mut td_t;
    pub fn td_i64(val: i64) -> *mut td_t;
    pub fn td_f64(val: c_double) -> *mut td_t;
    pub fn td_str(s: *const c_char, len: usize) -> *mut td_t;
    pub fn td_sym(id: i64) -> *mut td_t;
    pub fn td_date(val: i64) -> *mut td_t;
    pub fn td_time(val: i64) -> *mut td_t;
    pub fn td_timestamp(val: i64) -> *mut td_t;
    pub fn td_guid(bytes: *const u8) -> *mut td_t;

    // --- Vector API ---
    pub fn td_vec_new(type_: i8, capacity: i64) -> *mut td_t;
    pub fn td_sym_vec_new(sym_width: u8, capacity: i64) -> *mut td_t;
    pub fn td_vec_append(vec: *mut td_t, elem: *const c_void) -> *mut td_t;
    pub fn td_vec_set(vec: *mut td_t, idx: i64, elem: *const c_void) -> *mut td_t;
    pub fn td_vec_get(vec: *mut td_t, idx: i64) -> *mut c_void;
    pub fn td_vec_slice(vec: *mut td_t, offset: i64, len: i64) -> *mut td_t;
    pub fn td_vec_concat(a: *mut td_t, b: *mut td_t) -> *mut td_t;
    pub fn td_vec_from_raw(type_: i8, data: *const c_void, count: i64) -> *mut td_t;
    pub fn td_vec_set_null(vec: *mut td_t, idx: i64, is_null: bool);
    pub fn td_vec_is_null(vec: *mut td_t, idx: i64) -> bool;

    // --- String API ---
    pub fn td_str_ptr(s: *mut td_t) -> *const c_char;
    pub fn td_str_len(s: *mut td_t) -> usize;
    pub fn td_str_cmp(a: *mut td_t, b: *mut td_t) -> c_int;

    // --- List API ---
    pub fn td_list_new(capacity: i64) -> *mut td_t;
    pub fn td_list_append(list: *mut td_t, item: *mut td_t) -> *mut td_t;
    pub fn td_list_get(list: *mut td_t, idx: i64) -> *mut td_t;
    pub fn td_list_set(list: *mut td_t, idx: i64, item: *mut td_t) -> *mut td_t;

    // --- Symbol Intern Table API ---
    pub fn td_sym_init();
    pub fn td_sym_destroy();
    pub fn td_sym_intern(s: *const c_char, len: usize) -> i64;
    pub fn td_sym_find(s: *const c_char, len: usize) -> i64;
    pub fn td_sym_str(id: i64) -> *mut td_t;
    pub fn td_sym_count() -> u32;

    // --- Table API ---
    pub fn td_table_new(ncols: i64) -> *mut td_t;
    pub fn td_table_add_col(tbl: *mut td_t, name_id: i64, col_vec: *mut td_t) -> *mut td_t;
    pub fn td_table_get_col(tbl: *mut td_t, name_id: i64) -> *mut td_t;
    pub fn td_table_get_col_idx(tbl: *mut td_t, idx: i64) -> *mut td_t;
    pub fn td_table_col_name(tbl: *mut td_t, idx: i64) -> i64;
    pub fn td_table_ncols(tbl: *mut td_t) -> i64;
    pub fn td_table_nrows(tbl: *mut td_t) -> i64;
    pub fn td_table_schema(tbl: *mut td_t) -> *mut td_t;

    // --- Morsel Iterator API ---
    pub fn td_morsel_init(m: *mut td_morsel_t, vec: *mut td_t);
    pub fn td_morsel_init_range(m: *mut td_morsel_t, vec: *mut td_t, start: i64, end: i64);
    pub fn td_morsel_next(m: *mut td_morsel_t) -> bool;

    // --- Operation Graph API ---
    pub fn td_graph_new(tbl: *mut td_t) -> *mut td_graph_t;
    pub fn td_graph_free(g: *mut td_graph_t);

    // Source ops
    pub fn td_scan(g: *mut td_graph_t, col_name: *const c_char) -> *mut td_op_t;
    pub fn td_const_f64(g: *mut td_graph_t, val: c_double) -> *mut td_op_t;
    pub fn td_const_i64(g: *mut td_graph_t, val: i64) -> *mut td_op_t;
    pub fn td_const_bool(g: *mut td_graph_t, val: bool) -> *mut td_op_t;
    pub fn td_const_str(g: *mut td_graph_t, s: *const c_char) -> *mut td_op_t;
    pub fn td_const_vec(g: *mut td_graph_t, vec: *mut td_t) -> *mut td_op_t;
    pub fn td_const_table(g: *mut td_graph_t, table: *mut td_t) -> *mut td_op_t;

    // Unary element-wise ops
    pub fn td_neg(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_abs(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_not(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_sqrt_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_log_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_exp_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_ceil_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_floor_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_isnull(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_cast(g: *mut td_graph_t, a: *mut td_op_t, target_type: i8) -> *mut td_op_t;

    // Binary element-wise ops
    pub fn td_add(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_sub(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_mul(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_div(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_mod(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_eq(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_ne(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_lt(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_le(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_gt(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_ge(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_and(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_or(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_min2(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_max2(g: *mut td_graph_t, a: *mut td_op_t, b: *mut td_op_t) -> *mut td_op_t;
    pub fn td_if(
        g: *mut td_graph_t,
        cond: *mut td_op_t,
        then_val: *mut td_op_t,
        else_val: *mut td_op_t,
    ) -> *mut td_op_t;
    pub fn td_like(g: *mut td_graph_t, input: *mut td_op_t, pattern: *mut td_op_t) -> *mut td_op_t;
    pub fn td_ilike(g: *mut td_graph_t, input: *mut td_op_t, pattern: *mut td_op_t)
        -> *mut td_op_t;
    pub fn td_upper(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_lower(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_strlen(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_substr(
        g: *mut td_graph_t,
        str: *mut td_op_t,
        start: *mut td_op_t,
        len: *mut td_op_t,
    ) -> *mut td_op_t;
    pub fn td_replace(
        g: *mut td_graph_t,
        str: *mut td_op_t,
        from: *mut td_op_t,
        to: *mut td_op_t,
    ) -> *mut td_op_t;
    pub fn td_trim_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_concat(g: *mut td_graph_t, args: *mut *mut td_op_t, n: c_int) -> *mut td_op_t;
    pub fn td_extract(g: *mut td_graph_t, col: *mut td_op_t, field: i64) -> *mut td_op_t;
    pub fn td_date_trunc(g: *mut td_graph_t, col: *mut td_op_t, field: i64) -> *mut td_op_t;

    // Reduction ops
    pub fn td_sum(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_prod(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_min_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_max_op(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_count(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_avg(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_first(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_last(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;
    pub fn td_count_distinct(g: *mut td_graph_t, a: *mut td_op_t) -> *mut td_op_t;

    // Structural ops
    pub fn td_filter(
        g: *mut td_graph_t,
        input: *mut td_op_t,
        predicate: *mut td_op_t,
    ) -> *mut td_op_t;

    pub fn td_sort_op(
        g: *mut td_graph_t,
        df_node: *mut td_op_t,
        keys: *mut *mut td_op_t,
        descs: *mut u8,
        nulls_first: *mut u8,
        n_cols: u8,
    ) -> *mut td_op_t;

    pub fn td_group(
        g: *mut td_graph_t,
        keys: *mut *mut td_op_t,
        n_keys: u8,
        agg_ops: *mut u16,
        agg_ins: *mut *mut td_op_t,
        n_aggs: u8,
    ) -> *mut td_op_t;

    pub fn td_distinct(g: *mut td_graph_t, keys: *mut *mut td_op_t, n_keys: u8) -> *mut td_op_t;

    pub fn td_join(
        g: *mut td_graph_t,
        left_df: *mut td_op_t,
        left_keys: *mut *mut td_op_t,
        right_df: *mut td_op_t,
        right_keys: *mut *mut td_op_t,
        n_keys: u8,
        join_type: u8,
    ) -> *mut td_op_t;

    pub fn td_window_join(
        g: *mut td_graph_t,
        left_df: *mut td_op_t,
        right_df: *mut td_op_t,
        time_key: *mut td_op_t,
        sym_key: *mut td_op_t,
        window_lo: i64,
        window_hi: i64,
        agg_ops: *mut u16,
        agg_ins: *mut *mut td_op_t,
        n_aggs: u8,
    ) -> *mut td_op_t;
    pub fn td_window_op(
        g: *mut td_graph_t,
        df_node: *mut td_op_t,
        part_keys: *mut *mut td_op_t,
        n_part: u8,
        order_keys: *mut *mut td_op_t,
        order_descs: *mut u8,
        n_order: u8,
        func_kinds: *mut u8,
        func_inputs: *mut *mut td_op_t,
        func_params: *mut i64,
        n_funcs: u8,
        frame_type: u8,
        frame_start: u8,
        frame_end: u8,
        frame_start_n: i64,
        frame_end_n: i64,
    ) -> *mut td_op_t;

    pub fn td_project(
        g: *mut td_graph_t,
        input: *mut td_op_t,
        cols: *mut *mut td_op_t,
        n_cols: u8,
    ) -> *mut td_op_t;

    pub fn td_select(
        g: *mut td_graph_t,
        input: *mut td_op_t,
        cols: *mut *mut td_op_t,
        n_cols: u8,
    ) -> *mut td_op_t;

    pub fn td_head(g: *mut td_graph_t, input: *mut td_op_t, n: i64) -> *mut td_op_t;
    pub fn td_tail(g: *mut td_graph_t, input: *mut td_op_t, n: i64) -> *mut td_op_t;
    pub fn td_alias(g: *mut td_graph_t, input: *mut td_op_t, name: *const c_char) -> *mut td_op_t;
    pub fn td_materialize(g: *mut td_graph_t, input: *mut td_op_t) -> *mut td_op_t;

    // --- Optimizer API ---
    pub fn td_optimize(g: *mut td_graph_t, root: *mut td_op_t) -> *mut td_op_t;
    pub fn td_fuse_pass(g: *mut td_graph_t, root: *mut td_op_t);

    // --- Executor API ---
    pub fn td_execute(g: *mut td_graph_t, root: *mut td_op_t) -> *mut td_t;

    // --- Storage API ---
    pub fn td_col_save(vec: *mut td_t, path: *const c_char) -> td_err_t;
    pub fn td_col_load(path: *const c_char) -> *mut td_t;
    pub fn td_splay_save(tbl: *mut td_t, dir: *const c_char, sym_path: *const c_char) -> td_err_t;
    pub fn td_splay_load(dir: *const c_char) -> *mut td_t;
    pub fn td_read_splayed(dir: *const c_char, sym_path: *const c_char) -> *mut td_t;
    pub fn td_part_load(db_root: *const c_char, table_name: *const c_char) -> *mut td_t;
    pub fn td_read_parted(db_root: *const c_char, table_name: *const c_char) -> *mut td_t;
    pub fn td_meta_save_d(schema: *mut td_t, path: *const c_char) -> td_err_t;
    pub fn td_meta_load_d(path: *const c_char) -> *mut td_t;

    // --- Symbol Persistence ---
    pub fn td_sym_save(path: *const c_char) -> td_err_t;
    pub fn td_sym_load(path: *const c_char) -> td_err_t;

    // --- CSV API ---
    pub fn td_read_csv(path: *const c_char) -> *mut td_t;
    pub fn td_read_csv_opts(
        path: *const c_char,
        delimiter: c_char,
        header: bool,
        col_types: *const i8,
        n_types: i32,
    ) -> *mut td_t;
    pub fn td_write_csv(table: *mut td_t, path: *const c_char) -> td_err_t;

    // --- Pool / Parallel API ---
    pub fn td_pool_init(n_workers: u32) -> td_err_t;
    pub fn td_pool_destroy();
    pub fn td_cancel();
}

// ===== Compile-time layout assertions =====

const _: () = {
    assert!(std::mem::size_of::<td_t>() == 32);
    assert!(std::mem::size_of::<td_op_t>() == 32);
    assert!(std::mem::size_of::<td_op_ext_t>() == 104);
    assert!(std::mem::size_of::<td_graph_t>() == 48);
};
