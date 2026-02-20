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

//! Safe Rust wrappers for the Teide C17 columnar table engine.
//!
//! Provides idiomatic, safe types around the raw FFI layer.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::ffi;

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/// Error values returned by the Teide engine or by wrapper-level input/runtime checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Oom,
    Type,
    Range,
    Length,
    Rank,
    Domain,
    Nyi,
    Io,
    Schema,
    Corrupt,
    Cancel,
    InvalidInput,
    NullPointer,
    EngineNotInitialized,
    RuntimeUnavailable,
}

impl Error {
    fn from_code(code: ffi::td_err_t) -> Self {
        match code {
            ffi::td_err_t::TD_ERR_OOM => Error::Oom,
            ffi::td_err_t::TD_ERR_TYPE => Error::Type,
            ffi::td_err_t::TD_ERR_RANGE => Error::Range,
            ffi::td_err_t::TD_ERR_LENGTH => Error::Length,
            ffi::td_err_t::TD_ERR_RANK => Error::Rank,
            ffi::td_err_t::TD_ERR_DOMAIN => Error::Domain,
            ffi::td_err_t::TD_ERR_NYI => Error::Nyi,
            ffi::td_err_t::TD_ERR_IO => Error::Io,
            ffi::td_err_t::TD_ERR_SCHEMA => Error::Schema,
            ffi::td_err_t::TD_ERR_CORRUPT => Error::Corrupt,
            ffi::td_err_t::TD_ERR_CANCEL => Error::Cancel,
            _ => Error::Corrupt,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Error::Oom => "out of memory",
            Error::Type => "type error",
            Error::Range => "range error",
            Error::Length => "length error",
            Error::Rank => "rank error",
            Error::Domain => "domain error",
            Error::Nyi => "not yet implemented",
            Error::Io => "I/O error",
            Error::Schema => "schema error",
            Error::Corrupt => "corrupt data",
            Error::Cancel => "query cancelled",
            Error::InvalidInput => "invalid input",
            Error::NullPointer => "null pointer",
            Error::EngineNotInitialized => "engine not initialized",
            Error::RuntimeUnavailable => "engine runtime is not available",
        };
        f.write_str(s)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Convert usize to u8, returning Error::Length on overflow.
fn to_u8(n: usize) -> Result<u8> {
    u8::try_from(n).map_err(|_| Error::Length)
}

/// Aggregation operation variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Sum,
    Avg,
    Min,
    Max,
    Count,
    First,
    Last,
    CountDistinct,
    Stddev,
    StddevPop,
    Var,
    VarPop,
}

impl AggOp {
    fn to_opcode(self) -> u16 {
        match self {
            AggOp::Sum => ffi::OP_SUM,
            AggOp::Avg => ffi::OP_AVG,
            AggOp::Min => ffi::OP_MIN,
            AggOp::Max => ffi::OP_MAX,
            AggOp::Count => ffi::OP_COUNT,
            AggOp::First => ffi::OP_FIRST,
            AggOp::Last => ffi::OP_LAST,
            AggOp::CountDistinct => ffi::OP_COUNT_DISTINCT,
            AggOp::Stddev => ffi::OP_STDDEV,
            AggOp::StddevPop => ffi::OP_STDDEV_POP,
            AggOp::Var => ffi::OP_VAR,
            AggOp::VarPop => ffi::OP_VAR_POP,
        }
    }
}

/// Window function variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Ntile(i64),
    Sum,
    Avg,
    Min,
    Max,
    Count,
    Lag(i64),
    Lead(i64),
    FirstValue,
    LastValue,
    NthValue(i64),
}

impl WindowFunc {
    /// Returns the TD_WIN_* kind code for the C API.
    pub fn kind_code(&self) -> u8 {
        match self {
            WindowFunc::RowNumber => ffi::TD_WIN_ROW_NUMBER,
            WindowFunc::Rank => ffi::TD_WIN_RANK,
            WindowFunc::DenseRank => ffi::TD_WIN_DENSE_RANK,
            WindowFunc::Ntile(_) => ffi::TD_WIN_NTILE,
            WindowFunc::Sum => ffi::TD_WIN_SUM,
            WindowFunc::Avg => ffi::TD_WIN_AVG,
            WindowFunc::Min => ffi::TD_WIN_MIN,
            WindowFunc::Max => ffi::TD_WIN_MAX,
            WindowFunc::Count => ffi::TD_WIN_COUNT,
            WindowFunc::Lag(_) => ffi::TD_WIN_LAG,
            WindowFunc::Lead(_) => ffi::TD_WIN_LEAD,
            WindowFunc::FirstValue => ffi::TD_WIN_FIRST_VALUE,
            WindowFunc::LastValue => ffi::TD_WIN_LAST_VALUE,
            WindowFunc::NthValue(_) => ffi::TD_WIN_NTH_VALUE,
        }
    }

    /// Returns the parameter value (offset, ntile count, etc.), or 0 if none.
    pub fn param(&self) -> i64 {
        match self {
            WindowFunc::Ntile(n)
            | WindowFunc::Lag(n)
            | WindowFunc::Lead(n)
            | WindowFunc::NthValue(n) => *n,
            _ => 0,
        }
    }
}

/// Window frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Rows,
    Range,
}

impl FrameType {
    fn to_code(self) -> u8 {
        match self {
            FrameType::Rows => ffi::TD_FRAME_ROWS,
            FrameType::Range => ffi::TD_FRAME_RANGE,
        }
    }
}

/// Window frame bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding(i64),
    CurrentRow,
    Following(i64),
    UnboundedFollowing,
}

impl FrameBound {
    fn to_code(self) -> u8 {
        match self {
            FrameBound::UnboundedPreceding => ffi::TD_BOUND_UNBOUNDED_PRECEDING,
            FrameBound::Preceding(_) => ffi::TD_BOUND_N_PRECEDING,
            FrameBound::CurrentRow => ffi::TD_BOUND_CURRENT_ROW,
            FrameBound::Following(_) => ffi::TD_BOUND_N_FOLLOWING,
            FrameBound::UnboundedFollowing => ffi::TD_BOUND_UNBOUNDED_FOLLOWING,
        }
    }

    fn to_n(self) -> i64 {
        match self {
            FrameBound::Preceding(n) | FrameBound::Following(n) => n,
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: check a td_t* return for error sentinel
// ---------------------------------------------------------------------------

fn check_ptr(ptr: *mut ffi::td_t) -> Result<*mut ffi::td_t> {
    if ptr.is_null() {
        return Err(Error::Oom);
    }
    if ffi::td_is_err(ptr) {
        Err(Error::from_code(ffi::td_err_code(ptr)))
    } else {
        Ok(ptr)
    }
}

struct EngineGuard {
    pool_inited: bool,
    sym_inited: bool,
    heap_inited: bool,
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        // Teardown order matters and is correct:
        // 1. td_pool_destroy() — sets the shutdown flag, signals all workers,
        //    then calls td_thread_join() on each worker thread. This is fully
        //    synchronous: all workers have exited (and destroyed their own
        //    thread-local heaps) before this call returns.
        // 2. td_sym_destroy() — tears down the global symbol table. Safe
        //    because no worker threads are running.
        // 3. td_heap_destroy() — tears down the main thread's heap.
        //    Must come last because sym_destroy may reference heap memory.
        unsafe {
            if self.pool_inited {
                ffi::td_pool_destroy();
            }
            if self.sym_inited {
                ffi::td_sym_destroy();
            }
            if self.heap_inited {
                ffi::td_heap_destroy();
            }
        }
    }
}

fn engine_slot() -> &'static Mutex<Weak<EngineGuard>> {
    static ENGINE_SLOT: OnceLock<Mutex<Weak<EngineGuard>>> = OnceLock::new();
    ENGINE_SLOT.get_or_init(|| Mutex::new(Weak::new()))
}

fn acquire_engine_guard() -> Result<Arc<EngineGuard>> {
    let mut lock = engine_slot().lock().map_err(|_| Error::Corrupt)?;
    if let Some(existing) = lock.upgrade() {
        return Ok(existing);
    }

    // Hold the lock across init to prevent double-initialization race
    unsafe {
        ffi::td_heap_init();
    }
    let mut guard = EngineGuard {
        pool_inited: false,
        sym_inited: false,
        heap_inited: true,
    };
    unsafe {
        ffi::td_sym_init();
    }
    guard.sym_inited = true;
    let err = unsafe { ffi::td_pool_init(0) };
    if err != ffi::td_err_t::TD_OK {
        // guard will drop, cleaning up heap + sym that were already initialized
        drop(guard);
        return Err(Error::from_code(err));
    }
    guard.pool_inited = true;

    let guard = Arc::new(guard);
    *lock = Arc::downgrade(&guard);
    Ok(guard)
}

fn acquire_existing_engine_guard() -> Result<Arc<EngineGuard>> {
    let lock = engine_slot().lock().map_err(|_| Error::Corrupt)?;
    lock.upgrade().ok_or(Error::RuntimeUnavailable)
}

/// Cancel any currently running query.
///
/// Safe to call from any thread (e.g. a signal handler or a separate
/// cancellation thread). The next morsel boundary in the executor will
/// observe the flag and return `Error::Cancel`.
///
/// This calls td_cancel() directly without acquiring the engine guard,
/// because td_cancel() only sets an atomic flag and is safe to call
/// from any thread without holding the guard. Acquiring the guard here
/// would risk triggering EngineGuard drop from the wrong thread.
pub fn cancel() {
    unsafe {
        ffi::td_cancel();
    }
}

// ---------------------------------------------------------------------------
// Context — manages global engine state (heap, sym table, thread pool)
// ---------------------------------------------------------------------------

/// Intern a string into the global symbol table. Returns a stable i64 ID.
///
/// Returns `Error::EngineNotInitialized` if no `Context` exists.
pub fn sym_intern(s: &str) -> Result<i64> {
    let _guard = acquire_existing_engine_guard().map_err(|_| Error::EngineNotInitialized)?;
    // SAFETY: td_sym_intern takes (const char*, size_t len) and uses the length
    // parameter, not NUL termination. Rust &str bytes are valid for the duration
    // of this call.
    Ok(unsafe { ffi::td_sym_intern(s.as_ptr() as *const std::ffi::c_char, s.len()) })
}

/// Engine context. Initializes the heap allocator, symbol table, and thread
/// pool on construction. Tears them down in reverse order on drop.
///
/// Only one `Context` should exist at a time. The C engine uses global state.
pub struct Context {
    engine: Arc<EngineGuard>,
    /// Cache of opened parted tables, keyed by "db_root/table_name" path.
    /// Avoids re-opening (mmap + sym_load) on every query.
    parted_cache: RefCell<HashMap<String, Table>>,
    // *mut () makes Context !Send + !Sync (C engine uses thread-local heaps)
    _not_send_sync: PhantomData<*mut ()>,
}

impl Context {
    /// Create a new engine context. Calls `td_heap_init`, `td_sym_init`,
    /// `td_pool_init(0)` (auto-detect thread count).
    pub fn new() -> Result<Self> {
        let engine = acquire_engine_guard()?;
        Ok(Context {
            engine,
            parted_cache: RefCell::new(HashMap::new()),
            _not_send_sync: PhantomData,
        })
    }

    /// Read a CSV file into a `Table`.
    pub fn read_csv(&self, path: &str) -> Result<Table> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidInput)?;
        let ptr = unsafe { ffi::td_read_csv(c_path.as_ptr()) };
        let ptr = check_ptr(ptr)?;
        Ok(Table {
            raw: ptr,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        })
    }

    /// Read a CSV file with custom options.
    ///
    /// Pass `col_types: None` to auto-infer types from a sample.
    /// Pass `col_types: Some(&[TD_I64, TD_F64, TD_SYM, ...])` to specify exact types.
    pub fn read_csv_opts(
        &self,
        path: &str,
        delimiter: char,
        header: bool,
        col_types: Option<&[i8]>,
    ) -> Result<Table> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidInput)?;
        let (types_ptr, n_types) = match col_types {
            Some(t) => (t.as_ptr(), t.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let ptr = unsafe {
            ffi::td_read_csv_opts(
                c_path.as_ptr(),
                delimiter as std::os::raw::c_char,
                header,
                types_ptr,
                n_types,
            )
        };
        let ptr = check_ptr(ptr)?;
        Ok(Table {
            raw: ptr,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        })
    }

    /// Open a splayed table from disk (zero-copy mmap).
    ///
    /// `dir` is the splayed table directory containing `.d` + column files.
    /// `sym_path` is the path to the shared sym file (or None to skip).
    pub fn read_splayed(&self, dir: &str, sym_path: Option<&str>) -> Result<Table> {
        let c_dir = CString::new(dir).map_err(|_| Error::InvalidInput)?;
        let c_sym = match sym_path {
            Some(s) => Some(CString::new(s).map_err(|_| Error::InvalidInput)?),
            None => None,
        };
        let sym_ptr = c_sym.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
        let ptr = unsafe { ffi::td_read_splayed(c_dir.as_ptr(), sym_ptr) };
        let ptr = check_ptr(ptr)?;
        Ok(Table {
            raw: ptr,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        })
    }

    /// Open a partitioned table from disk (zero-copy mmap).
    ///
    /// `db_root` is the database directory (e.g. `/tmp/teide_db`),
    /// `table_name` is the table name within each date partition.
    /// The symfile at `db_root/sym` is loaded automatically.
    /// Results are cached — subsequent calls with the same path return a
    /// clone_ref of the cached table (no re-open, no sym_load).
    pub fn read_parted(&self, db_root: &str, table_name: &str) -> Result<Table> {
        let cache_key = format!("{db_root}/{table_name}");

        // Return cached table if available
        if let Some(cached) = self.parted_cache.borrow().get(&cache_key) {
            return Ok(cached.clone_ref());
        }

        let c_root = CString::new(db_root).map_err(|_| Error::InvalidInput)?;
        let c_name = CString::new(table_name).map_err(|_| Error::InvalidInput)?;
        let ptr = unsafe { ffi::td_read_parted(c_root.as_ptr(), c_name.as_ptr()) };
        let ptr = check_ptr(ptr)?;
        let table = Table {
            raw: ptr,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        };

        // Cache for future queries
        self.parted_cache
            .borrow_mut()
            .insert(cache_key, table.clone_ref());
        Ok(table)
    }

    /// Create a new operation graph bound to a table.
    pub fn graph<'a>(&self, table: &'a Table) -> Result<Graph<'a>> {
        let raw = unsafe { ffi::td_graph_new(table.raw) };
        // td_graph_new returns null on failure (OOM). It returns a
        // td_graph_t*, not a td_t*, so td_is_err is not applicable.
        if raw.is_null() {
            return Err(Error::Oom);
        }
        Ok(Graph {
            raw,
            engine: table.engine.clone(),
            _table: PhantomData,
            _pinned: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Memory statistics
// ---------------------------------------------------------------------------

/// Snapshot of engine memory usage.
#[derive(Debug, Clone, Copy)]
pub struct MemStats {
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

/// Return a snapshot of engine memory usage.
pub fn mem_stats() -> MemStats {
    let mut raw = ffi::td_mem_stats_t::default();
    unsafe { ffi::td_mem_stats(&mut raw) };
    MemStats {
        alloc_count: raw.alloc_count,
        free_count: raw.free_count,
        bytes_allocated: raw.bytes_allocated,
        peak_bytes: raw.peak_bytes,
        slab_hits: raw.slab_hits,
        direct_count: raw.direct_count,
        direct_bytes: raw.direct_bytes,
        sys_current: raw.sys_current,
        sys_peak: raw.sys_peak,
    }
}

// ---------------------------------------------------------------------------
// Table — RAII wrapper around td_t* (type=TD_TABLE)
// ---------------------------------------------------------------------------

/// A columnar table backed by the Teide engine.
pub struct Table {
    raw: *mut ffi::td_t,
    engine: Arc<EngineGuard>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Table {
    /// Wrap a raw pointer as a Table (takes ownership — will call td_release on drop).
    /// The pointer must already be retained.
    ///
    /// # Safety
    /// `raw` must be a valid Teide `td_t*` table pointer obtained from the same
    /// initialized engine runtime.
    ///
    /// Returns `Error::RuntimeUnavailable` if no active engine runtime exists.
    pub unsafe fn from_raw(raw: *mut ffi::td_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::Corrupt);
        }
        let engine = acquire_existing_engine_guard()?;
        Ok(Table {
            raw,
            engine,
            _not_send_sync: PhantomData,
        })
    }

    /// Create a shared reference to this table by incrementing the C ref count.
    /// Both the original and the clone will call `td_release` on drop.
    pub fn clone_ref(&self) -> Self {
        unsafe {
            ffi::td_retain(self.raw);
        }
        Table {
            raw: self.raw,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        }
    }

    /// Raw pointer access (for interop).
    pub fn as_raw(&self) -> *mut ffi::td_t {
        self.raw
    }

    /// Number of rows.
    pub fn nrows(&self) -> i64 {
        unsafe { ffi::td_table_nrows(self.raw) }
    }

    /// Number of columns.
    pub fn ncols(&self) -> i64 {
        unsafe { ffi::td_table_ncols(self.raw) }
    }

    /// Symbol ID of the column name at `idx`.
    pub fn col_name(&self, idx: i64) -> i64 {
        unsafe { ffi::td_table_col_name(self.raw, idx) }
    }

    /// Resolve the column name at `idx` to an owned `String`.
    pub fn col_name_str(&self, idx: usize) -> String {
        let sym_id = self.col_name(idx as i64);
        let atom = unsafe { ffi::td_sym_str(sym_id) };
        if atom.is_null() {
            return String::new();
        }
        unsafe {
            let ptr = ffi::td_str_ptr(atom);
            let len = ffi::td_str_len(atom);
            let slice = std::slice::from_raw_parts(ptr as *const u8, len);
            std::str::from_utf8(slice).unwrap_or("").to_owned()
        }
    }

    /// Add a column to this table. The table pointer may change (COW semantics).
    /// `name` is a pre-interned symbol ID.
    ///
    /// # Safety
    /// `col` must be a valid, non-null Teide vector pointer from the current
    /// engine runtime. The caller retains ownership; this function does NOT
    /// release `col`.
    pub unsafe fn add_column_raw(&mut self, name_id: i64, col: *mut ffi::td_t) -> Result<()> {
        if col.is_null() {
            return Err(Error::NullPointer);
        }
        let next = unsafe { ffi::td_table_add_col(self.raw, name_id, col) };
        if next.is_null() {
            return Err(Error::Oom);
        }
        if ffi::td_is_err(next) {
            return Err(Error::from_code(ffi::td_err_code(next)));
        }
        self.raw = next;
        Ok(())
    }

    /// Create a new table with the same column data but renamed columns.
    /// `names` must have exactly `ncols()` entries.
    ///
    /// # Ownership
    /// `td_table_add_col` retains each column vector internally. If an error
    /// occurs mid-construction, releasing `new_raw` cascades to all columns
    /// already added, so no manual per-column release is needed on the error path.
    pub fn with_column_names(&self, names: &[String]) -> Result<Self> {
        let nc = self.ncols() as usize;
        if names.len() != nc {
            return Err(Error::Length);
        }
        unsafe {
            let mut new_raw = ffi::td_table_new(nc as i64);
            if new_raw.is_null() {
                return Err(Error::Oom);
            }
            if ffi::td_is_err(new_raw) {
                return Err(Error::from_code(ffi::td_err_code(new_raw)));
            }
            for (i, name) in names.iter().enumerate().take(nc) {
                let col = ffi::td_table_get_col_idx(self.raw, i as i64);
                if col.is_null() {
                    ffi::td_release(new_raw);
                    return Err(Error::Corrupt);
                }
                if ffi::td_is_err(col) {
                    let err = Error::from_code(ffi::td_err_code(col));
                    ffi::td_release(new_raw);
                    return Err(err);
                }
                let name_id = sym_intern(name)?;
                ffi::td_retain(col);
                let next_raw = ffi::td_table_add_col(new_raw, name_id, col);
                if next_raw.is_null() {
                    ffi::td_release(col);
                    ffi::td_release(new_raw);
                    return Err(Error::Oom);
                }
                if ffi::td_is_err(next_raw) {
                    let err = Error::from_code(ffi::td_err_code(next_raw));
                    ffi::td_release(col);
                    ffi::td_release(new_raw);
                    return Err(err);
                }
                new_raw = next_raw;
            }
            Table::from_raw(new_raw)
        }
    }

    /// Build a new table by picking columns by name in the given order.
    /// Columns are shared (ref-counted), so no data is copied.
    pub fn pick_columns(&self, names: &[&str]) -> Result<Self> {
        unsafe {
            let mut tbl = ffi::td_table_new(names.len() as i64);
            if tbl.is_null() || ffi::td_is_err(tbl) {
                return Err(Error::Oom);
            }
            for &name in names {
                let name_id = sym_intern(name)?;
                let col = ffi::td_table_get_col(self.raw, name_id);
                if col.is_null() || ffi::td_is_err(col) {
                    ffi::td_release(tbl);
                    return Err(Error::Schema);
                }
                ffi::td_retain(col);
                let next = ffi::td_table_add_col(tbl, name_id, col);
                if next.is_null() || ffi::td_is_err(next) {
                    ffi::td_release(col);
                    ffi::td_release(tbl);
                    return Err(Error::Oom);
                }
                tbl = next;
            }
            Table::from_raw(tbl)
        }
    }

    /// Write this table to a CSV file.
    pub fn write_csv(&self, path: &str) -> Result<()> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidInput)?;
        let err = unsafe { ffi::td_write_csv(self.raw, c_path.as_ptr()) };
        if err != ffi::td_err_t::TD_OK {
            Err(Error::from_code(err))
        } else {
            Ok(())
        }
    }

    /// Raw column vector at index (returns None for out-of-range or error).
    pub fn get_col_idx(&self, idx: i64) -> Option<*mut ffi::td_t> {
        let p = unsafe { ffi::td_table_get_col_idx(self.raw, idx) };
        if p.is_null() || ffi::td_is_err(p) {
            None
        } else {
            Some(p)
        }
    }

    /// Type tag of the column at `idx`.
    /// For parted columns, returns the base type (e.g. TD_I64 not TD_PARTED_BASE+TD_I64).
    /// For MAPCOMMON columns, returns the key_values vector type (DATE/I64/SYM).
    pub fn col_type(&self, idx: usize) -> i8 {
        match self.get_col_idx(idx as i64) {
            Some(col) => {
                let t = unsafe { ffi::td_type(col) };
                if t == ffi::TD_MAPCOMMON {
                    unsafe { Self::mapcommon_kv_type(col) }
                } else if ffi::td_is_parted(t) {
                    ffi::td_parted_basetype(t)
                } else {
                    t
                }
            }
            None => 0,
        }
    }

    /// True if column at `idx` is a MAPCOMMON virtual partition column.
    pub fn is_mapcommon(&self, idx: usize) -> bool {
        match self.get_col_idx(idx as i64) {
            Some(col) => {
                let t = unsafe { ffi::td_type(col) };
                t == ffi::TD_MAPCOMMON
            }
            None => false,
        }
    }

    /// Inferred sub-type of a MAPCOMMON column (TD_MC_SYM/DATE/I64).
    pub fn mapcommon_inferred_type(&self, idx: usize) -> u8 {
        match self.get_col_idx(idx as i64) {
            Some(col) => unsafe { (*col).attrs },
            None => ffi::TD_MC_SYM,
        }
    }

    /// Get the type of key_values inside a MAPCOMMON column.
    unsafe fn mapcommon_kv_type(mc: *mut ffi::td_t) -> i8 {
        let ptrs = unsafe { ffi::td_data(mc) as *const *mut ffi::td_t };
        let kv = unsafe { *ptrs.add(0) };
        unsafe { ffi::td_type(kv) }
    }

    /// Resolve a logical row in a TD_MAPCOMMON column to its partition index.
    /// MAPCOMMON stores [key_values, row_counts] as two td_t pointers.
    unsafe fn resolve_mapcommon_part(vec: *mut ffi::td_t, row: usize) -> Option<usize> {
        let ptrs = unsafe { ffi::td_data(vec) as *const *mut ffi::td_t };
        let rc_vec = unsafe { *ptrs.add(1) }; // row_counts
        let n_parts = unsafe { ffi::td_len(rc_vec) } as usize;
        let counts = unsafe { ffi::td_data(rc_vec) as *const i64 };
        let mut offset = 0usize;
        for p in 0..n_parts {
            let cnt = unsafe { *counts.add(p) } as usize;
            if row < offset + cnt {
                return Some(p);
            }
            offset += cnt;
        }
        None
    }

    /// Read an i64 value from a MAPCOMMON column (handles DATE/I64/SYM key types).
    unsafe fn read_mapcommon_i64(vec: *mut ffi::td_t, row: usize) -> Option<i64> {
        let p = unsafe { Self::resolve_mapcommon_part(vec, row)? };
        let ptrs = unsafe { ffi::td_data(vec) as *const *mut ffi::td_t };
        let kv = unsafe { *ptrs.add(0) };
        let kv_type = unsafe { ffi::td_type(kv) };
        let data = unsafe { ffi::td_data(kv) };
        match kv_type {
            ffi::TD_DATE | ffi::TD_I32 | ffi::TD_TIME => {
                let ptr = data as *const i32;
                Some(unsafe { *ptr.add(p) } as i64)
            }
            _ => {
                // TD_I64, TD_SYM, TD_TIMESTAMP
                let ptr = data as *const i64;
                Some(unsafe { *ptr.add(p) })
            }
        }
    }

    /// Resolve a logical row in a TD_PARTED column to (segment_data_ptr, local_row).
    /// Returns None if the row is out of range.
    unsafe fn resolve_parted_row(
        vec: *mut ffi::td_t,
        row: usize,
    ) -> Option<(*mut ffi::td_t, usize)> {
        let n_segs = unsafe { ffi::td_len(vec) } as usize;
        let segs = unsafe { ffi::td_data(vec) as *const *mut ffi::td_t };
        let mut offset = 0usize;
        for s in 0..n_segs {
            let seg = unsafe { *segs.add(s) };
            if seg.is_null() {
                continue;
            }
            let seg_len = unsafe { ffi::td_len(seg) } as usize;
            if row < offset + seg_len {
                return Some((seg, row - offset));
            }
            offset += seg_len;
        }
        None
    }

    /// Read an i64 value from column `col`, row `row`.
    pub fn get_i64(&self, col: usize, row: usize) -> Option<i64> {
        let vec = self.get_col_idx(col as i64)?;
        let t = unsafe { ffi::td_type(vec) };

        // Handle MAPCOMMON: return partition key value as i64
        if t == ffi::TD_MAPCOMMON {
            return unsafe { Self::read_mapcommon_i64(vec, row) };
        }

        // Handle parted columns: resolve to segment
        if ffi::td_is_parted(t) {
            let (seg, local_row) = unsafe { Self::resolve_parted_row(vec, row)? };
            let base_t = ffi::td_parted_basetype(t);
            return unsafe { Self::read_i64_from_vec(seg, base_t, local_row) };
        }

        let len = unsafe { ffi::td_len(vec) } as usize;
        if row >= len {
            return None;
        }
        unsafe { Self::read_i64_from_vec(vec, t, row) }
    }

    /// Read an i64 from a flat (non-parted) vector at the given row.
    unsafe fn read_i64_from_vec(vec: *mut ffi::td_t, t: i8, row: usize) -> Option<i64> {
        let data = unsafe { ffi::td_data(vec) };
        match t {
            ffi::TD_I64 | ffi::TD_TIME | ffi::TD_TIMESTAMP => {
                let p = data as *const i64;
                Some(unsafe { *p.add(row) })
            }
            ffi::TD_BOOL => {
                let p = data as *const u8;
                Some(unsafe { *p.add(row) } as i64)
            }
            ffi::TD_I32 | ffi::TD_DATE => {
                let p = data as *const i32;
                Some(unsafe { *p.add(row) } as i64)
            }
            ffi::TD_SYM => {
                let attrs = unsafe { ffi::td_attrs(vec) };
                Some(unsafe { ffi::read_sym(data as *const u8, row, t, attrs) })
            }
            _ => None,
        }
    }

    /// Read an f64 value from column `col`, row `row`.
    pub fn get_f64(&self, col: usize, row: usize) -> Option<f64> {
        let vec = self.get_col_idx(col as i64)?;
        let t = unsafe { ffi::td_type(vec) };

        // MAPCOMMON columns are partition key symbols, not floats
        if t == ffi::TD_MAPCOMMON {
            return None;
        }

        // Handle parted columns
        if ffi::td_is_parted(t) {
            let base_t = ffi::td_parted_basetype(t);
            if base_t != ffi::TD_F64 {
                return None;
            }
            let (seg, local_row) = unsafe { Self::resolve_parted_row(vec, row)? };
            let data = unsafe { ffi::td_data(seg) as *const f64 };
            return Some(unsafe { *data.add(local_row) });
        }

        let len = unsafe { ffi::td_len(vec) } as usize;
        if row >= len {
            return None;
        }
        if t != ffi::TD_F64 {
            return None;
        }
        unsafe {
            let data = ffi::td_data(vec) as *const f64;
            Some(*data.add(row))
        }
    }

    /// Read a string value from a SYM or MAPCOMMON column at `col`, `row`.
    pub fn get_str(&self, col: usize, row: usize) -> Option<String> {
        let vec = self.get_col_idx(col as i64)?;
        let t = unsafe { ffi::td_type(vec) };

        // Handle MAPCOMMON: resolve partition → typed value → string
        if t == ffi::TD_MAPCOMMON {
            let kv_type = unsafe { Self::mapcommon_kv_type(vec) };
            let val = unsafe { Self::read_mapcommon_i64(vec, row)? };
            return match kv_type {
                ffi::TD_SYM => {
                    let atom = unsafe { ffi::td_sym_str(val) };
                    if atom.is_null() {
                        return None;
                    }
                    unsafe {
                        let ptr = ffi::td_str_ptr(atom);
                        let slen = ffi::td_str_len(atom);
                        let slice = std::slice::from_raw_parts(ptr as *const u8, slen);
                        std::str::from_utf8(slice).ok().map(|s| s.to_owned())
                    }
                }
                ffi::TD_DATE => Some(Self::format_date(val as i32)),
                _ => Some(format!("{val}")),
            };
        }

        // Handle parted columns
        if ffi::td_is_parted(t) {
            let base_t = ffi::td_parted_basetype(t);
            let (seg, local_row) = unsafe { Self::resolve_parted_row(vec, row)? };
            return unsafe { Self::read_str_from_vec(seg, base_t, local_row) };
        }

        let len = unsafe { ffi::td_len(vec) } as usize;
        if row >= len {
            return None;
        }
        unsafe { Self::read_str_from_vec(vec, t, row) }
    }

    /// Read a string from a flat (non-parted) SYM vector.
    /// Format days since 2000-01-01 as "YYYY-MM-DD" string.
    /// Inverse of Hinnant civil_from_days algorithm.
    pub fn format_date(days_since_2000: i32) -> String {
        let z = days_since_2000 as i64 + 10957 + 719468; // shift to 0000-03-01 epoch
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!("{y:04}-{m:02}-{d:02}")
    }

    unsafe fn read_str_from_vec(vec: *mut ffi::td_t, t: i8, row: usize) -> Option<String> {
        let sym_id = match t {
            ffi::TD_SYM => {
                let data = unsafe { ffi::td_data(vec) as *const u8 };
                let attrs = unsafe { ffi::td_attrs(vec) };
                unsafe { ffi::read_sym(data, row, t, attrs) }
            }
            _ => return None,
        };
        let atom = unsafe { ffi::td_sym_str(sym_id) };
        if atom.is_null() {
            return None;
        }
        unsafe {
            let ptr = ffi::td_str_ptr(atom);
            let slen = ffi::td_str_len(atom);
            let slice = std::slice::from_raw_parts(ptr as *const u8, slen);
            std::str::from_utf8(slice).ok().map(|s| s.to_owned())
        }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if !self.raw.is_null() && !ffi::td_is_err(self.raw) {
            unsafe {
                ffi::td_release(self.raw);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Column — thin wrapper around *mut td_op_t (graph-owned, no Drop)
// ---------------------------------------------------------------------------

/// A reference to an operation node in a `Graph`. Does not own memory --
/// the `Graph` owns all nodes.
///
/// Raw pointer to a graph operation node. **WARNING**: This is only valid
/// while the parent `Graph` is alive. Do not store Columns beyond the
/// Graph's lifetime.
///
/// SAFETY: Column holds a raw pointer into Graph's node array.
/// It must not be used after the Graph is dropped or with a different Graph.
/// The current API enforces this by requiring &self on Graph methods that
/// accept Column arguments.
#[derive(Clone, Copy)]
pub struct Column {
    raw: *mut ffi::td_op_t,
}

impl Column {
    /// Raw pointer access.
    pub fn as_raw(&self) -> *mut ffi::td_op_t {
        self.raw
    }
}

// ---------------------------------------------------------------------------
// Graph<'a> — operation graph bound to a Table's lifetime
// ---------------------------------------------------------------------------

/// A lazy operation graph. Operations build a DAG that is optimized and
/// executed when `execute()` is called.
pub struct Graph<'a> {
    raw: *mut ffi::td_graph_t,
    engine: Arc<EngineGuard>,
    // Ties lifetime to the Table AND makes Graph !Send + !Sync via *mut ()
    _table: PhantomData<(&'a Table, *mut ())>,
    // Pin arrays passed to C functions that store pointers (td_group, td_sort_op,
    // td_window_op, td_join). These must live until the graph is dropped/executed.
    //
    // SAFETY: Vec::as_mut_ptr() returns a stable pointer to the Vec's heap buffer.
    // Wrapping each Vec in a Box<dyn Any> and pushing to _pinned does NOT
    // invalidate the pointer because: (a) Box::new moves the Vec *value* (three
    // words: ptr/len/cap) to the heap, but the data buffer the Vec points to is
    // already on the heap and does not move; (b) pushing Box<dyn Any> into
    // _pinned may reallocate the outer Vec<Box<...>>, but that only moves the
    // Box pointers, not the inner data buffers. Therefore the raw pointers
    // stored by C remain valid for the lifetime of the Graph.
    _pinned: Vec<Box<dyn std::any::Any>>,
}

impl Graph<'_> {
    /// Raw pointer access.
    pub fn as_raw(&self) -> *mut ffi::td_graph_t {
        self.raw
    }

    // ---- Helper: check an op pointer for null ------------------------------

    fn check_op(raw: *mut ffi::td_op_t) -> Result<Column> {
        if raw.is_null() {
            return Err(Error::NullPointer); // graph operation failed (null result)
        }
        Ok(Column { raw })
    }

    // ---- Source ops -------------------------------------------------------

    /// Scan a column by name from the bound table.
    /// Returns `Error::InvalidInput` when `col_name` contains interior NUL bytes.
    pub fn scan(&self, col_name: &str) -> Result<Column> {
        // SAFETY: C function copies/interns the string immediately. CString is
        // valid for the duration of the FFI call.
        let c_name = CString::new(col_name).map_err(|_| Error::InvalidInput)?;
        let raw = unsafe { ffi::td_scan(self.raw, c_name.as_ptr()) };
        Self::check_op(raw)
    }

    /// Create a constant f64 node.
    pub fn const_f64(&self, val: f64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_const_f64(self.raw, val) })
    }

    /// Create a constant i64 node.
    pub fn const_i64(&self, val: i64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_const_i64(self.raw, val) })
    }

    /// Create a constant bool node.
    pub fn const_bool(&self, val: bool) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_const_bool(self.raw, val) })
    }

    /// Create a constant string node.
    /// Returns `Error::InvalidInput` when `val` contains interior NUL bytes.
    pub fn const_str(&self, val: &str) -> Result<Column> {
        // SAFETY: C function copies/interns the string immediately. CString is
        // valid for the duration of the FFI call.
        let c_val = CString::new(val).map_err(|_| Error::InvalidInput)?;
        Self::check_op(unsafe { ffi::td_const_str(self.raw, c_val.as_ptr()) })
    }

    /// Create a constant table node referencing a table.
    pub fn const_table(&self, table: &Table) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_const_table(self.raw, table.raw) })
    }

    // ---- Binary element-wise ops ------------------------------------------

    pub fn add(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_add(self.raw, a.raw, b.raw) })
    }

    pub fn sub(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_sub(self.raw, a.raw, b.raw) })
    }

    pub fn mul(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_mul(self.raw, a.raw, b.raw) })
    }

    pub fn div(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_div(self.raw, a.raw, b.raw) })
    }

    pub fn modulo(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_mod(self.raw, a.raw, b.raw) })
    }

    pub fn eq(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_eq(self.raw, a.raw, b.raw) })
    }

    pub fn ne(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_ne(self.raw, a.raw, b.raw) })
    }

    pub fn lt(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_lt(self.raw, a.raw, b.raw) })
    }

    pub fn le(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_le(self.raw, a.raw, b.raw) })
    }

    pub fn gt(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_gt(self.raw, a.raw, b.raw) })
    }

    pub fn ge(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_ge(self.raw, a.raw, b.raw) })
    }

    pub fn and(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_and(self.raw, a.raw, b.raw) })
    }

    pub fn or(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_or(self.raw, a.raw, b.raw) })
    }

    pub fn min2(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_min2(self.raw, a.raw, b.raw) })
    }

    pub fn max2(&self, a: Column, b: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_max2(self.raw, a.raw, b.raw) })
    }

    pub fn if_then_else(&self, cond: Column, then_val: Column, else_val: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_if(self.raw, cond.raw, then_val.raw, else_val.raw) })
    }

    pub fn like(&self, input: Column, pattern: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_like(self.raw, input.raw, pattern.raw) })
    }

    pub fn ilike(&self, input: Column, pattern: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_ilike(self.raw, input.raw, pattern.raw) })
    }

    // ---- String ops -------------------------------------------------------

    pub fn upper(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_upper(self.raw, a.raw) })
    }

    pub fn lower(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_lower(self.raw, a.raw) })
    }

    pub fn strlen(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_strlen(self.raw, a.raw) })
    }

    pub fn trim(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_trim_op(self.raw, a.raw) })
    }

    pub fn substr(&self, s: Column, start: Column, len: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_substr(self.raw, s.raw, start.raw, len.raw) })
    }

    pub fn replace(&self, s: Column, from: Column, to: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_replace(self.raw, s.raw, from.raw, to.raw) })
    }

    pub fn concat(&self, args: &[Column]) -> Result<Column> {
        let mut ptrs: Vec<*mut ffi::td_op_t> = args.iter().map(|c| c.raw).collect();
        Self::check_op(unsafe {
            ffi::td_concat(self.raw, ptrs.as_mut_ptr(), args.len() as std::ffi::c_int)
        })
    }

    // ---- Unary ops --------------------------------------------------------

    pub fn not(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_not(self.raw, a.raw) })
    }

    pub fn neg(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_neg(self.raw, a.raw) })
    }

    pub fn abs(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_abs(self.raw, a.raw) })
    }

    pub fn sqrt(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_sqrt_op(self.raw, a.raw) })
    }

    pub fn log(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_log_op(self.raw, a.raw) })
    }

    pub fn exp(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_exp_op(self.raw, a.raw) })
    }

    pub fn ceil(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_ceil_op(self.raw, a.raw) })
    }

    pub fn floor(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_floor_op(self.raw, a.raw) })
    }

    pub fn isnull(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_isnull(self.raw, a.raw) })
    }

    pub fn cast(&self, a: Column, target_type: i8) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_cast(self.raw, a.raw, target_type) })
    }

    // ---- Date/time extraction ---------------------------------------------

    pub fn extract(&self, col: Column, field: i64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_extract(self.raw, col.raw, field) })
    }

    pub fn date_trunc(&self, col: Column, field: i64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_date_trunc(self.raw, col.raw, field) })
    }

    // ---- Reduction ops ----------------------------------------------------

    pub fn sum(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_sum(self.raw, a.raw) })
    }

    pub fn avg(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_avg(self.raw, a.raw) })
    }

    pub fn min_op(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_min_op(self.raw, a.raw) })
    }

    pub fn max_op(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_max_op(self.raw, a.raw) })
    }

    pub fn count(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_count(self.raw, a.raw) })
    }

    pub fn first(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_first(self.raw, a.raw) })
    }

    pub fn last(&self, a: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_last(self.raw, a.raw) })
    }

    // ---- Structural ops ---------------------------------------------------

    /// Group-by aggregation.
    pub fn group_by(
        &mut self,
        keys: &[Column],
        agg_ops: &[AggOp],
        agg_inputs: &[Column],
    ) -> Result<Column> {
        if agg_ops.len() != agg_inputs.len() {
            return Err(Error::Length);
        }

        let mut key_ptrs: Vec<*mut ffi::td_op_t> = keys.iter().map(|c| c.raw).collect();
        let mut ops: Vec<u16> = agg_ops.iter().map(|op| op.to_opcode()).collect();
        let mut input_ptrs: Vec<*mut ffi::td_op_t> = agg_inputs.iter().map(|c| c.raw).collect();

        let raw = unsafe {
            ffi::td_group(
                self.raw,
                key_ptrs.as_mut_ptr(),
                to_u8(keys.len())?,
                ops.as_mut_ptr(),
                input_ptrs.as_mut_ptr(),
                to_u8(agg_ops.len())?,
            )
        };
        // Pin arrays — td_group stores pointers to them
        self._pinned.push(Box::new(key_ptrs));
        self._pinned.push(Box::new(ops));
        self._pinned.push(Box::new(input_ptrs));
        Self::check_op(raw)
    }

    /// Distinct — GROUP BY with 0 aggregates, returns unique key combinations.
    pub fn distinct(&mut self, keys: &[Column]) -> Result<Column> {
        let mut key_ptrs: Vec<*mut ffi::td_op_t> = keys.iter().map(|c| c.raw).collect();
        let raw = unsafe { ffi::td_distinct(self.raw, key_ptrs.as_mut_ptr(), to_u8(keys.len())?) };
        self._pinned.push(Box::new(key_ptrs));
        Self::check_op(raw)
    }

    /// Hash join.
    pub fn join(
        &mut self,
        left_table: Column,
        left_keys: &[Column],
        right_table: Column,
        right_keys: &[Column],
        join_type: u8,
    ) -> Result<Column> {
        if left_keys.len() != right_keys.len() {
            return Err(Error::Length);
        }
        let mut lk: Vec<*mut ffi::td_op_t> = left_keys.iter().map(|c| c.raw).collect();
        let mut rk: Vec<*mut ffi::td_op_t> = right_keys.iter().map(|c| c.raw).collect();
        let raw = unsafe {
            ffi::td_join(
                self.raw,
                left_table.raw,
                lk.as_mut_ptr(),
                right_table.raw,
                rk.as_mut_ptr(),
                to_u8(left_keys.len())?,
                join_type,
            )
        };
        self._pinned.push(Box::new(lk));
        self._pinned.push(Box::new(rk));
        Self::check_op(raw)
    }

    /// Multi-column sort.
    pub fn sort(
        &mut self,
        table_node: Column,
        keys: &[Column],
        descs: &[bool],
        nulls_first: Option<&[bool]>,
    ) -> Result<Column> {
        if keys.len() != descs.len() {
            return Err(Error::Length);
        }

        let mut key_ptrs: Vec<*mut ffi::td_op_t> = keys.iter().map(|c| c.raw).collect();
        let mut desc_u8: Vec<u8> = descs.iter().map(|&d| d as u8).collect();

        let nf_ptr = if let Some(nf) = nulls_first {
            if keys.len() != nf.len() {
                return Err(Error::Length);
            }
            let mut nf_u8: Vec<u8> = nf.iter().map(|&n| n as u8).collect();
            let ptr = nf_u8.as_mut_ptr();
            self._pinned.push(Box::new(nf_u8));
            ptr
        } else {
            std::ptr::null_mut()
        };

        let raw = unsafe {
            ffi::td_sort_op(
                self.raw,
                table_node.raw,
                key_ptrs.as_mut_ptr(),
                desc_u8.as_mut_ptr(),
                nf_ptr,
                to_u8(keys.len())?,
            )
        };
        // Pin arrays — td_sort_op stores pointers to them
        self._pinned.push(Box::new(key_ptrs));
        self._pinned.push(Box::new(desc_u8));
        Self::check_op(raw)
    }

    /// Window function computation.
    ///
    /// Produces a table with all original columns plus one new column per
    /// window function. Partitions by `part_keys`, orders within each
    /// partition by `order_keys` (with `order_descs` flags), and computes
    /// each function described by `funcs` and `func_inputs`.
    #[allow(clippy::too_many_arguments)]
    pub fn window_op(
        &mut self,
        table_node: Column,
        part_keys: &[Column],
        order_keys: &[Column],
        order_descs: &[bool],
        funcs: &[WindowFunc],
        func_inputs: &[Column],
        frame_type: FrameType,
        frame_start: FrameBound,
        frame_end: FrameBound,
    ) -> Result<Column> {
        if order_keys.len() != order_descs.len() {
            return Err(Error::Length);
        }
        if funcs.len() != func_inputs.len() {
            return Err(Error::Length);
        }

        let mut pk_ptrs: Vec<*mut ffi::td_op_t> = part_keys.iter().map(|c| c.raw).collect();
        let mut ok_ptrs: Vec<*mut ffi::td_op_t> = order_keys.iter().map(|c| c.raw).collect();
        let mut od_u8: Vec<u8> = order_descs.iter().map(|&d| d as u8).collect();
        let mut kinds: Vec<u8> = funcs.iter().map(|f| f.kind_code()).collect();
        let mut fi_ptrs: Vec<*mut ffi::td_op_t> = func_inputs.iter().map(|c| c.raw).collect();
        let mut params: Vec<i64> = funcs.iter().map(|f| f.param()).collect();

        let raw = unsafe {
            ffi::td_window_op(
                self.raw,
                table_node.raw,
                pk_ptrs.as_mut_ptr(),
                to_u8(part_keys.len())?,
                ok_ptrs.as_mut_ptr(),
                od_u8.as_mut_ptr(),
                to_u8(order_keys.len())?,
                kinds.as_mut_ptr(),
                fi_ptrs.as_mut_ptr(),
                params.as_mut_ptr(),
                to_u8(funcs.len())?,
                frame_type.to_code(),
                frame_start.to_code(),
                frame_end.to_code(),
                frame_start.to_n(),
                frame_end.to_n(),
            )
        };
        // Pin all arrays — td_window_op stores pointers to them
        self._pinned.push(Box::new(pk_ptrs));
        self._pinned.push(Box::new(ok_ptrs));
        self._pinned.push(Box::new(od_u8));
        self._pinned.push(Box::new(kinds));
        self._pinned.push(Box::new(fi_ptrs));
        self._pinned.push(Box::new(params));
        Self::check_op(raw)
    }

    /// Project (select) specific columns from a table node.
    pub fn project(&self, input: Column, cols: &[Column]) -> Result<Column> {
        let mut col_ptrs: Vec<*mut ffi::td_op_t> = cols.iter().map(|c| c.raw).collect();
        Self::check_op(unsafe {
            ffi::td_project(
                self.raw,
                input.raw,
                col_ptrs.as_mut_ptr(),
                to_u8(cols.len())?,
            )
        })
    }

    /// Select specific columns from a table node (alias for project).
    pub fn select(&self, input: Column, cols: &[Column]) -> Result<Column> {
        let mut col_ptrs: Vec<*mut ffi::td_op_t> = cols.iter().map(|c| c.raw).collect();
        Self::check_op(unsafe {
            ffi::td_select(
                self.raw,
                input.raw,
                col_ptrs.as_mut_ptr(),
                to_u8(cols.len())?,
            )
        })
    }

    /// Filter rows by a boolean predicate column.
    pub fn filter(&self, input: Column, predicate: Column) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_filter(self.raw, input.raw, predicate.raw) })
    }

    /// Take the first `n` rows.
    pub fn head(&self, input: Column, n: i64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_head(self.raw, input.raw, n) })
    }

    /// Take the last `n` rows.
    pub fn tail(&self, input: Column, n: i64) -> Result<Column> {
        Self::check_op(unsafe { ffi::td_tail(self.raw, input.raw, n) })
    }

    /// Rename/alias a column.
    /// Returns `Error::InvalidInput` when `name` contains interior NUL bytes.
    pub fn alias(&self, input: Column, name: &str) -> Result<Column> {
        // SAFETY: C function copies/interns the string immediately. CString is
        // valid for the duration of the FFI call.
        let c_name = CString::new(name).map_err(|_| Error::InvalidInput)?;
        Self::check_op(unsafe { ffi::td_alias(self.raw, input.raw, c_name.as_ptr()) })
    }

    // ---- Execute ----------------------------------------------------------

    /// Optimize the DAG and execute it, returning a result `Table`.
    pub fn execute(&self, root: Column) -> Result<Table> {
        let optimized = unsafe { ffi::td_optimize(self.raw, root.raw) };
        if optimized.is_null() {
            return Err(Error::Oom);
        }
        let result = unsafe { ffi::td_execute(self.raw, optimized) };
        // GC: reclaim fully-free pools on the main thread after execution.
        unsafe { ffi::td_heap_gc() };
        let result = check_ptr(result)?;
        // td_execute returns a freshly allocated td_t* with rc=1 (caller owns it).
        // No td_retain needed — Table::drop will call td_release to free it.
        Ok(Table {
            raw: result,
            engine: self.engine.clone(),
            _not_send_sync: PhantomData,
        })
    }

    /// Execute a graph node and return the raw result (vector or table).
    /// Caller is responsible for releasing the result via `td_release`.
    ///
    /// td_execute returns with rc=1 (caller owns it), so no extra retain
    /// is needed. The caller must call `td_release` exactly once when done.
    pub fn execute_raw(&self, root: Column) -> Result<*mut ffi::td_t> {
        let optimized = unsafe { ffi::td_optimize(self.raw, root.raw) };
        if optimized.is_null() {
            return Err(Error::Oom);
        }
        let result = unsafe { ffi::td_execute(self.raw, optimized) };
        let result = check_ptr(result)?;
        Ok(result)
    }

    /// Inject a pre-computed vector as a constant node in the graph.
    ///
    /// # Safety
    /// `vec` must be a valid, non-null Teide vector pointer from the current
    /// engine runtime.
    pub unsafe fn const_vec(&self, vec: *mut ffi::td_t) -> Result<Column> {
        if vec.is_null() {
            return Err(Error::NullPointer);
        }
        Self::check_op(unsafe { ffi::td_const_vec(self.raw, vec) })
    }

    /// Set a boolean filter mask for group-by pushdown.
    /// Rows where mask[r]==0 are skipped in scan loops.
    ///
    /// # Safety
    /// `sel` must be a valid TD_SEL selection allocated by the same engine runtime.
    ///
    /// # Ownership
    /// The selection is retained here via `td_retain`. When the graph is freed by
    /// `td_graph_free`, the C engine releases the selection pointer. The
    /// retain/release sequence is therefore: caller creates sel (rc=1) ->
    /// set_selection retains (rc=2) -> caller may release their ref (rc=1)
    /// -> graph free releases (rc=0, freed). If an error occurs before graph
    /// execution, td_graph_free still handles the release.
    pub unsafe fn set_selection(&mut self, sel: *mut ffi::td_t) {
        unsafe {
            ffi::td_retain(sel);
            (*self.raw).selection = sel;
        }
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::td_graph_free(self.raw);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Re-exports for downstream crates
// ---------------------------------------------------------------------------

pub use ffi::td_graph_t;
pub use ffi::td_op_t;
pub use ffi::td_t;

/// Low-level FFI access for downstream crates (e.g., teide-db).
pub mod raw {
    pub use super::ffi::{td_t, td_type_sizes, td_vec_new};

    /// Read the logical type tag from a raw value pointer.
    ///
    /// # Safety
    /// `v` must be a valid non-null `td_t*`.
    #[inline]
    pub unsafe fn td_type(v: *mut td_t) -> i8 {
        unsafe { super::ffi::td_type(v as *const td_t) }
    }

    /// Read vector length from a raw value pointer.
    ///
    /// # Safety
    /// `v` must be a valid non-null `td_t*`.
    #[inline]
    pub unsafe fn td_len(v: *mut td_t) -> i64 {
        unsafe { super::ffi::td_len(v as *const td_t) }
    }

    /// Return raw data pointer for a vector.
    ///
    /// # Safety
    /// `v` must be a valid non-null vector `td_t*` from the current runtime.
    #[inline]
    pub unsafe fn td_data(v: *mut td_t) -> *mut u8 {
        unsafe { super::ffi::td_data(v) as *mut u8 }
    }

    /// Override vector length metadata in-place.
    ///
    /// # Safety
    /// `v` must be a valid non-null vector `td_t*`, and `len` must not exceed
    /// the allocated element capacity for that vector.
    #[inline]
    pub unsafe fn td_set_len(v: *mut td_t, len: i64) {
        unsafe {
            (*v).val.len = len;
        }
    }
}

/// Low-level helper: get column by symbol ID from a raw table pointer.
/// Returns null if not found. Caller must NOT release the result.
///
/// # Safety
/// `tbl` must be a valid table pointer from the current runtime.
pub unsafe fn ffi_table_get_col(tbl: *mut ffi::td_t, name_id: i64) -> *mut ffi::td_t {
    unsafe { ffi::td_table_get_col(tbl, name_id) }
}

/// Low-level helper: create new table.
///
/// # Safety
/// The engine runtime must be initialized and alive.
pub unsafe fn ffi_table_new(ncols: i64) -> *mut ffi::td_t {
    unsafe { ffi::td_table_new(ncols) }
}

/// Low-level helper: add column to table.
///
/// # Safety
/// `tbl` and `col` must be valid pointers from the same engine runtime.
pub unsafe fn ffi_table_add_col(
    tbl: *mut ffi::td_t,
    name_id: i64,
    col: *mut ffi::td_t,
) -> *mut ffi::td_t {
    unsafe { ffi::td_table_add_col(tbl, name_id, col) }
}

/// Low-level helper: concatenate two vectors.
///
/// # Safety
/// `a` and `b` must be valid vector pointers from the same engine runtime.
pub unsafe fn ffi_vec_concat(a: *mut ffi::td_t, b: *mut ffi::td_t) -> *mut ffi::td_t {
    unsafe { ffi::td_vec_concat(a, b) }
}

/// Low-level helper: release a td_t pointer.
///
/// # Safety
/// `v` must be a valid pointer whose lifetime is managed by Teide refcounting.
pub unsafe fn ffi_release(v: *mut ffi::td_t) {
    unsafe { ffi::td_release(v) }
}

/// Low-level helper: retain a td_t pointer.
///
/// # Safety
/// `v` must be a valid pointer whose lifetime is managed by Teide refcounting.
pub unsafe fn ffi_retain(v: *mut ffi::td_t) {
    unsafe { ffi::td_retain(v) }
}

/// Check if a raw pointer is an error sentinel.
pub fn ffi_is_err(p: *mut ffi::td_t) -> bool {
    ffi::td_is_err(p)
}

/// Decode a raw engine error pointer to `Error`.
pub fn ffi_error_from_ptr(p: *mut ffi::td_t) -> Option<Error> {
    if ffi::td_is_err(p) {
        Some(Error::from_code(ffi::td_err_code(p)))
    } else {
        None
    }
}

// Re-export EXTRACT field constants
pub mod extract_field {
    pub const YEAR: i64 = super::ffi::TD_EXTRACT_YEAR;
    pub const MONTH: i64 = super::ffi::TD_EXTRACT_MONTH;
    pub const DAY: i64 = super::ffi::TD_EXTRACT_DAY;
    pub const HOUR: i64 = super::ffi::TD_EXTRACT_HOUR;
    pub const MINUTE: i64 = super::ffi::TD_EXTRACT_MINUTE;
    pub const SECOND: i64 = super::ffi::TD_EXTRACT_SECOND;
    pub const DOW: i64 = super::ffi::TD_EXTRACT_DOW;
    pub const DOY: i64 = super::ffi::TD_EXTRACT_DOY;
    pub const EPOCH: i64 = super::ffi::TD_EXTRACT_EPOCH;
}

// Re-export window function constants
pub mod window_func {
    pub const ROW_NUMBER: u8 = super::ffi::TD_WIN_ROW_NUMBER;
    pub const RANK: u8 = super::ffi::TD_WIN_RANK;
    pub const DENSE_RANK: u8 = super::ffi::TD_WIN_DENSE_RANK;
    pub const NTILE: u8 = super::ffi::TD_WIN_NTILE;
    pub const SUM: u8 = super::ffi::TD_WIN_SUM;
    pub const AVG: u8 = super::ffi::TD_WIN_AVG;
    pub const MIN: u8 = super::ffi::TD_WIN_MIN;
    pub const MAX: u8 = super::ffi::TD_WIN_MAX;
    pub const COUNT: u8 = super::ffi::TD_WIN_COUNT;
    pub const LAG: u8 = super::ffi::TD_WIN_LAG;
    pub const LEAD: u8 = super::ffi::TD_WIN_LEAD;
    pub const FIRST_VALUE: u8 = super::ffi::TD_WIN_FIRST_VALUE;
    pub const LAST_VALUE: u8 = super::ffi::TD_WIN_LAST_VALUE;
    pub const NTH_VALUE: u8 = super::ffi::TD_WIN_NTH_VALUE;

    pub const FRAME_ROWS: u8 = super::ffi::TD_FRAME_ROWS;
    pub const FRAME_RANGE: u8 = super::ffi::TD_FRAME_RANGE;

    pub const BOUND_UNBOUNDED_PRECEDING: u8 = super::ffi::TD_BOUND_UNBOUNDED_PRECEDING;
    pub const BOUND_PRECEDING: u8 = super::ffi::TD_BOUND_N_PRECEDING;
    pub const BOUND_CURRENT_ROW: u8 = super::ffi::TD_BOUND_CURRENT_ROW;
    pub const BOUND_FOLLOWING: u8 = super::ffi::TD_BOUND_N_FOLLOWING;
    pub const BOUND_UNBOUNDED_FOLLOWING: u8 = super::ffi::TD_BOUND_UNBOUNDED_FOLLOWING;
}

// Re-export type constants
pub mod types {
    pub const BOOL: i8 = super::ffi::TD_BOOL;
    pub const I32: i8 = super::ffi::TD_I32;
    pub const I64: i8 = super::ffi::TD_I64;
    pub const F64: i8 = super::ffi::TD_F64;
    pub const DATE: i8 = super::ffi::TD_DATE;
    pub const TIME: i8 = super::ffi::TD_TIME;
    pub const TIMESTAMP: i8 = super::ffi::TD_TIMESTAMP;
    pub const TABLE: i8 = super::ffi::TD_TABLE;
    pub const SYM: i8 = super::ffi::TD_SYM;
}
