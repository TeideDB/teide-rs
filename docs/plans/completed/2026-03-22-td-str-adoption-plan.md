# TD_STR Adoption Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update teide-rs to support the new `TD_STR` variable-length string type from the upstream C engine, making it the default SQL string type.

**Architecture:** Sync the vendored C source, update FFI bindings, add `TD_STR` read/write support in the engine layer, change SQL type mappings from `TD_SYM` to `TD_STR`, and update all display/formatting paths.

**Tech Stack:** Rust FFI, C17, sqlparser (DuckDbDialect), SLT test framework

---

### Task 1: Sync Vendor C Source

**Files:**
- Replace: `vendor/teide/` (entire directory tree)

**Step 1: Copy the upstream C engine into vendor**

```bash
rm -rf vendor/teide
cp -r /home/hetoku/data/work/teidedb/teide vendor/teide
```

This brings in:
- `TD_STR` type (type constant 21) with inline+pool string storage
- `td_str_vec_append/get/set/compact` functions
- `td_sym_ensure_cap` function
- Fixed sym save/load format (file locking, atomic rename)
- New `src/mem/arena.c` (arena allocator)
- New `src/store/fileio.c` (cross-platform file I/O)
- Updated `td_splay_load` signature (added `sym_path` parameter)
- String op executor support (UPPER/LOWER/TRIM/CONCAT etc. on TD_STR)

**Step 2: Verify it compiles**

Run: `cargo build --all-features 2>&1 | head -50`
Expected: Linker errors about missing Rust declarations (since FFI hasn't been updated yet), but the C compilation should succeed. If you see C compile errors, investigate first.

**Step 3: Commit**

```bash
git add vendor/teide
git commit -m "chore: sync vendor/teide with upstream (TD_STR, arena, fileio)"
```

---

### Task 2: Update FFI Constants

**Files:**
- Modify: `src/ffi.rs:46-48` (type constants)
- Modify: `src/ffi.rs:77` (TD_ATOM_STR fix)
- Modify: `src/ffi.rs:110-115` (attribute flags)
- Modify: `src/ffi.rs:248` (opcode constants)

**Step 1: Add TD_STR constant and fix TD_ATOM_STR**

In `src/ffi.rs`, after line 46 (`pub const TD_SYM: i8 = 20;`), add:

```rust
pub const TD_STR: i8 = 21;
```

Change line 48 from:
```rust
pub const TD_TYPE_COUNT: usize = 21;
```
to:
```rust
pub const TD_TYPE_COUNT: usize = 22;
```

Change line 77 from:
```rust
pub const TD_ATOM_STR: i8 = -8;
```
to:
```rust
pub const TD_ATOM_STR: i8 = -TD_STR;
```

**Step 2: Add TD_ATTR_ARENA constant**

In `src/ffi.rs`, after line 114 (`pub const TD_ATTR_HAS_NULLS: u8 = 0x40;`), add:

```rust
pub const TD_ATTR_ARENA: u8 = 0x80;
```

**Step 3: Add OP_LOCAL_CLUSTERING_COEFF opcode**

In `src/ffi.rs`, after line 248 (`pub const OP_HNSW_KNN: u16 = 91;`), add:

```rust
pub const OP_LOCAL_CLUSTERING_COEFF: u16 = 92;
```

**Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | head -20`
Expected: Still linker errors (functions not yet declared), but no Rust compile errors from constants.

---

### Task 3: Update FFI Struct and Function Declarations

**Files:**
- Modify: `src/ffi.rs:337-358` (td_t_head union, td_t_ext_nullmap)
- Modify: `src/ffi.rs:682-703` (add string vector API, sym_ensure_cap)
- Modify: `src/ffi.rs:988-1001` (fix td_splay_load, add file I/O)

**Step 1: Update td_t_head union for str_pool**

The upstream `td_t` struct has three union variants in bytes 0-15:
1. `nullmap[16]` — null bitmask
2. `slice_parent + slice_offset` — slice metadata
3. `ext_nullmap + sym_dict` — external nullmap + symbol dictionary
4. **NEW:** `str_ext_null + str_pool` — string external nullmap + string pool

In `src/ffi.rs`, the `td_t_ext_nullmap` struct at line 354 currently has `_reserved: i64`. The upstream splits this into two variants. Since Rust unions can hold multiple struct variants, add a new variant to `td_t_head`:

After `td_t_ext_nullmap` (line 358), add a new struct:

```rust
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_t_str_pool {
    pub str_ext_null: *mut td_t,
    pub str_pool: *mut td_t,
}
```

Update `td_t_head` union (line 340) to add:

```rust
pub str: td_t_str_pool,
```

Also update `td_t_ext_nullmap` to rename `_reserved` to `sym_dict` to match upstream:

```rust
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct td_t_ext_nullmap {
    pub ext_nullmap: *mut td_t,
    pub sym_dict: *mut td_t,
}
```

**Step 2: Add string vector API functions**

In `src/ffi.rs`, after line 683 (`pub fn td_vec_is_null(...)`) and before the String API section, add:

```rust
    // --- String Vector API ---
    pub fn td_str_vec_append(vec: *mut td_t, s: *const c_char, len: usize) -> *mut td_t;
    pub fn td_str_vec_get(vec: *mut td_t, idx: i64, out_len: *mut usize) -> *const c_char;
    pub fn td_str_vec_set(vec: *mut td_t, idx: i64, s: *const c_char, len: usize) -> *mut td_t;
    pub fn td_str_vec_compact(vec: *mut td_t) -> *mut td_t;
```

**Step 3: Add td_sym_ensure_cap**

In `src/ffi.rs`, after line 702 (`pub fn td_sym_count() -> u32;`), add:

```rust
    pub fn td_sym_ensure_cap(needed: u32) -> bool;
```

**Step 4: Fix td_splay_load signature**

In `src/ffi.rs`, change line 992 from:
```rust
    pub fn td_splay_load(dir: *const c_char) -> *mut td_t;
```
to:
```rust
    pub fn td_splay_load(dir: *const c_char, sym_path: *const c_char) -> *mut td_t;
```

**Step 5: Add file I/O type and functions**

After the symbol persistence section (after line 1001), add:

```rust
    // --- File I/O API ---
    pub fn td_file_open(path: *const c_char, flags: c_int) -> c_int;
    pub fn td_file_close(fd: c_int);
    pub fn td_file_lock_ex(fd: c_int) -> td_err_t;
    pub fn td_file_lock_sh(fd: c_int) -> td_err_t;
    pub fn td_file_unlock(fd: c_int) -> td_err_t;
    pub fn td_file_sync(fd: c_int) -> td_err_t;
    pub fn td_file_sync_dir(path: *const c_char) -> td_err_t;
    pub fn td_file_rename(old_path: *const c_char, new_path: *const c_char) -> td_err_t;
```

Note: On Linux, `td_fd_t` is `int`, so `c_int` is correct.

**Step 6: Fix any callers of td_splay_load**

Search for `td_splay_load` callers in Rust code:

Run: `grep -rn "td_splay_load" src/`

If there are callers, update them to pass the `sym_path` argument. Likely in `engine.rs`. If the caller doesn't use a sym path, pass `std::ptr::null()`.

**Step 7: Verify it compiles and links**

Run: `cargo build --all-features 2>&1 | tail -20`
Expected: Clean compile. If linker errors remain, the function names don't match upstream — check spelling against `vendor/teide/include/teide/td.h`.

**Step 8: Commit**

```bash
git add src/ffi.rs
git commit -m "feat(ffi): add TD_STR type, string vector API, fix TD_ATOM_STR and splay_load signature"
```

---

### Task 4: Update Engine Layer

**Files:**
- Modify: `src/engine.rs:1097-1116` (read_str_from_vec — add TD_STR branch)
- Modify: `src/engine.rs:2700-2713` (types module — add STR constant)

**Step 1: Add STR to the types module**

In `src/engine.rs`, in the `types` module (line ~2712), after `pub const SYM: i8 = super::ffi::TD_SYM;`, add:

```rust
    pub const STR: i8 = super::ffi::TD_STR;
```

**Step 2: Add TD_STR branch to read_str_from_vec**

In `src/engine.rs`, the function `read_str_from_vec` at line 1097 currently only handles `TD_SYM`. Add a `TD_STR` branch:

Change:
```rust
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
```

To:
```rust
    unsafe fn read_str_from_vec(vec: *mut ffi::td_t, t: i8, row: usize) -> Option<String> {
        match t {
            ffi::TD_STR => {
                let mut out_len: usize = 0;
                let ptr = unsafe { ffi::td_str_vec_get(vec, row as i64, &mut out_len) };
                if ptr.is_null() || out_len == 0 {
                    // Check for null element
                    if unsafe { ffi::td_vec_is_null(vec, row as i64) } {
                        return None;
                    }
                    return Some(String::new());
                }
                unsafe {
                    let slice = std::slice::from_raw_parts(ptr as *const u8, out_len);
                    std::str::from_utf8(slice).ok().map(|s| s.to_owned())
                }
            }
            ffi::TD_SYM => {
                let data = unsafe { ffi::td_data(vec) as *const u8 };
                let attrs = unsafe { ffi::td_attrs(vec) };
                let sym_id = unsafe { ffi::read_sym(data, row, t, attrs) };
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
            _ => None,
        }
    }
```

**Step 3: Update get_str to handle TD_STR in parted columns**

In `src/engine.rs`, the `get_str` method at line ~1032 passes `base_t` to `read_str_from_vec` for parted columns. Since TD_STR is a new base type that could appear in parted columns, this works automatically — no change needed.

**Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

**Step 5: Commit**

```bash
git add src/engine.rs
git commit -m "feat(engine): add TD_STR support to read_str_from_vec and types module"
```

---

### Task 5: Update SQL Type Mappings

**Files:**
- Modify: `src/sql/planner.rs:641-668` (sql_type_to_td)
- Modify: `src/sql/expr.rs:885-903` (map_sql_type)

**Step 1: Change sql_type_to_td to map strings to TD_STR**

In `src/sql/planner.rs`, change lines 657-661 from:

```rust
        DataType::Varchar(_)
        | DataType::Text
        | DataType::Char(_)
        | DataType::CharVarying(_)
        | DataType::String(_) => Ok(ffi::TD_SYM),
```

to:

```rust
        DataType::Varchar(_)
        | DataType::Text
        | DataType::Char(_)
        | DataType::CharVarying(_)
        | DataType::String(_) => Ok(ffi::TD_STR),
```

Also add a `SYMBOL` type mapping. DuckDbDialect doesn't have a native `SYMBOL` keyword, so we need to handle it as a custom type. Add before the `_ =>` fallback arm (line 665):

```rust
        DataType::Custom(name, _) if name.0.len() == 1
            && name.0[0].value.eq_ignore_ascii_case("symbol") => Ok(ffi::TD_SYM),
```

**Step 2: Change map_sql_type to map strings to TD_STR**

In `src/sql/expr.rs`, change line 895 from:

```rust
        DataType::Varchar(_) | DataType::Text | DataType::String(_) => Ok(crate::types::SYM),
```

to:

```rust
        DataType::Varchar(_) | DataType::Text | DataType::String(_) => Ok(crate::types::STR),
```

**Step 3: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

**Step 4: Commit**

```bash
git add src/sql/planner.rs src/sql/expr.rs
git commit -m "feat(sql): map VARCHAR/TEXT/STRING to TD_STR, add SYMBOL keyword for TD_SYM"
```

---

### Task 6: Update INSERT Planner for TD_STR

**Files:**
- Modify: `src/sql/planner.rs:1280-1286` (append_value_to_vec TD_SYM branch)

**Step 1: Add TD_STR branch to append_value_to_vec**

In `src/sql/planner.rs`, the `append_value_to_vec` function has a `TD_SYM` arm at line 1280. Add a new arm for `TD_STR` right before it:

```rust
        ffi::TD_STR => {
            let s = eval_str_literal(expr)
                .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?;
            let cstr = std::ffi::CString::new(s.as_str())
                .map_err(|_| SqlError::Plan(format!("column '{}': string contains null byte", col_names[col_idx])))?;
            let next = unsafe { ffi::td_str_vec_append(vec, cstr.as_ptr(), s.len()) };
            check_vec_append(next)
        }
```

The existing `TD_SYM` arm stays for explicit `SYMBOL` columns.

**Step 2: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

**Step 3: Commit**

```bash
git add src/sql/planner.rs
git commit -m "feat(sql): add TD_STR branch to INSERT VALUES planner"
```

---

### Task 7: Update Broadcast and TD_ATOM_STR Handling

**Files:**
- Modify: `src/sql/planner.rs:3822-3831` (empty table broadcast)
- Modify: `src/sql/planner.rs:3920-4020` (broadcast_const_column)
- Modify: `src/sql/planner.rs:4095-4130` (atom-to-vector conversion)

**Step 1: Update empty table broadcast for TD_ATOM_STR**

The `TD_ATOM_STR` constant changed from `-8` to `-21` (`-TD_STR`). The existing logic at lines 3822-3824 maps `TD_ATOM_STR → TD_SYM`. With the new constant, this represents a `TD_STR` atom, so it should map to `TD_STR`:

Change lines 3822-3824 from:
```rust
            let vec_type = if col_type < 0 {
                if col_type == crate::ffi::TD_ATOM_STR {
                    crate::ffi::TD_SYM
```
to:
```rust
            let vec_type = if col_type < 0 {
                if col_type == crate::ffi::TD_ATOM_STR {
                    crate::ffi::TD_STR
```

**Step 2: Update broadcast_const_column for TD_ATOM_STR**

At line 3925-3926, change:
```rust
        if col_type == crate::ffi::TD_ATOM_STR {
            (crate::ffi::TD_SYM, true)
```
to:
```rust
        if col_type == crate::ffi::TD_ATOM_STR {
            (crate::ffi::TD_STR, false)
```

Note: `is_sym` is now `false` because TD_STR is not a SYM column. The broadcast logic for TD_STR needs to use `td_str_vec_append` in a loop instead of the SYM fill path.

After the existing `is_sym` branch (which handles SYM columns with adaptive width), add a TD_STR branch. This is the `if is_sym { ... }` block that starts around line 3941. After that block's closing brace, before the generic `else` block, add:

```rust
    } else if vec_type == crate::ffi::TD_STR {
        // TD_STR broadcast: read source string, append N times
        let (src_ptr, src_len) = if col_type < 0 {
            // Atom: read SSO string
            let ptr = unsafe { crate::ffi::td_str_ptr(col) };
            let slen = unsafe { crate::ffi::td_str_len(col) };
            (ptr, slen)
        } else {
            // 1-element vector: read via td_str_vec_get
            let mut out_len: usize = 0;
            let ptr = unsafe { crate::ffi::td_str_vec_get(col, 0, &mut out_len) };
            (ptr, out_len)
        };
        let mut vec = unsafe { crate::ffi::td_vec_new(crate::ffi::TD_STR, 0) };
        if vec.is_null() || crate::ffi_is_err(vec) {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        for _ in 0..target_len {
            let next = unsafe { crate::ffi::td_str_vec_append(vec, src_ptr, src_len) };
            if next.is_null() || crate::ffi_is_err(next) {
                unsafe { crate::ffi_release(vec) };
                return Err(SqlError::Engine(crate::Error::Oom));
            }
            vec = next;
        }
        unsafe { crate::ffi::td_set_len(vec, target_len) };
        Ok(vec)
```

**Step 3: Update atom-to-vector conversion (lines ~4098-4130)**

The code at line 4098 currently converts `TD_ATOM_STR` atoms by interning into `TD_SYM`. Now it should create a `TD_STR` vector instead:

Change lines 4098-4130 (the `TD_ATOM_STR` special case) from creating a SYM vector to creating a STR vector:

```rust
            // Special case: TD_ATOM_STR (-21) maps to TD_STR (21).
            let vec_type = if col_type == crate::ffi::TD_ATOM_STR {
                crate::ffi::TD_STR
            } else {
                -col_type
            };
            let vec = if col_type == crate::ffi::TD_ATOM_STR {
                // String atom: read the inline string, create a 1-element TD_STR vector.
                let ptr = unsafe { crate::ffi::td_str_ptr(col) };
                let slen = unsafe { crate::ffi::td_str_len(col) };
                let v = unsafe { crate::ffi::td_vec_new(crate::ffi::TD_STR, 0) };
                if v.is_null() || crate::ffi_is_err(v) {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                let next = unsafe { crate::ffi::td_str_vec_append(v, ptr, slen) };
                if next.is_null() || crate::ffi_is_err(next) {
                    unsafe { crate::ffi_release(v) };
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                next
            } else {
```

The rest of the else branch (generic atom → 1-element vector) stays as-is.

**Step 4: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

**Step 5: Commit**

```bash
git add src/sql/planner.rs
git commit -m "feat(sql): update broadcast and atom conversion for TD_STR"
```

---

### Task 8: Update col_vec_new and col_elem_size Helpers

**Files:**
- Modify: `src/sql/planner.rs:4484-4506` (col_vec_new, col_elem_size)

**Step 1: Update col_vec_new for TD_STR**

The `col_vec_new` function at line 4486 already falls through to `td_vec_new(col_type, capacity)` for non-SYM types. Since `td_vec_new(TD_STR, capacity)` works correctly in the C engine, no change is needed here.

**Step 2: Update col_elem_size for TD_STR**

The `col_elem_size` function at line 4497 returns element size for copying. For TD_STR, each element is a 16-byte `td_str_t`. This function is used for raw memory copies during UPDATE/DELETE.

After the `TD_SYM` check, add a `TD_STR` case:

```rust
    if col_type == crate::ffi::TD_STR {
        return 16; // sizeof(td_str_t)
    }
```

However, raw `memcpy` of `td_str_t` elements between vectors won't correctly handle pool references. The UPDATE/DELETE paths that use `col_elem_size` may need to use `td_str_vec_get` + `td_str_vec_set` instead of raw copies for `TD_STR` columns. Check the callers:

Run: `grep -n "col_elem_size" src/sql/planner.rs`

If callers do raw `ptr::copy_nonoverlapping`, they need a TD_STR-specific path using the string vector API. If the callers only use it for size validation, 16 is correct.

**Step 3: Verify and commit**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

```bash
git add src/sql/planner.rs
git commit -m "feat(sql): handle TD_STR element size in col_elem_size"
```

---

### Task 9: Update Server and PGQ

**Files:**
- Modify: `src/server/types.rs:31-44` (teide_to_pg_type)
- Modify: `src/server/types.rs:53-110` (format_cell)
- Modify: `src/sql/pgq.rs:4130` (convert_text_to_list_columns)

**Step 1: Add TD_STR to teide_to_pg_type**

In `src/server/types.rs`, add a new arm at line 42, before `ffi::TD_SYM => Type::VARCHAR`:

```rust
        ffi::TD_STR => Type::VARCHAR,
```

**Step 2: Add TD_STR to format_cell**

In `src/server/types.rs`, add a new arm at line 107, before `ffi::TD_SYM => table.get_str(col, row)`:

```rust
        ffi::TD_STR => table.get_str(col, row),
```

**Step 3: Update convert_text_to_list_columns in PGQ**

In `src/sql/pgq.rs`, line 4161, `table.get_str(ci, row)` already works for both TD_SYM and TD_STR (since we updated `read_str_from_vec`). No change needed here. The function reads string values from text columns to parse list syntax — it works regardless of the underlying string type.

**Step 4: Check the PGQ value reading (lines 1834, 1941, 2218)**

These all use `table.get_str()` which now handles both TD_SYM and TD_STR. No changes needed.

**Step 5: Verify it compiles**

Run: `cargo build --all-features 2>&1 | tail -10`
Expected: Clean compile.

**Step 6: Commit**

```bash
git add src/server/types.rs src/sql/pgq.rs
git commit -m "feat(server): add TD_STR support to PgWire type mapping and format_cell"
```

---

### Task 10: Run Existing Tests and Fix Breakage

**Files:**
- Potentially modify: any file where tests expose issues

**Step 1: Run the full test suite**

Run: `cargo test --all-features 2>&1 | tail -40`

Expected: Some tests may fail because:
1. `VARCHAR`/`TEXT` columns are now `TD_STR` instead of `TD_SYM`
2. String operations may behave differently on `TD_STR` vs `TD_SYM`
3. The `td_splay_load` caller may need updating

**Step 2: Analyze failures**

For each failure:
- If it's a type mismatch (code expects TD_SYM but gets TD_STR), update the expectation
- If it's a runtime crash, investigate the C engine's handling of TD_STR for that operation
- If it's a display/formatting issue, check `read_str_from_vec` and `format_cell`

**Step 3: Fix each failure**

Apply minimal fixes. Common patterns:
- `assert_eq!(col_type, TD_SYM)` → `assert_eq!(col_type, TD_STR)`
- Engine API tests that create `VARCHAR` tables will now have `TD_STR` columns
- SLT tests that check string output should pass unchanged (strings display the same regardless of storage type)

**Step 4: Run tests again until clean**

Run: `cargo test --all-features 2>&1 | tail -20`
Expected: All pass.

**Step 5: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test failures from TD_STR migration"
```

---

### Task 11: Add New SLT Tests for TD_STR

**Files:**
- Create: `tests/slt/str.slt`

**Step 1: Write the TD_STR test file**

Create `tests/slt/str.slt`:

```
# TD_STR variable-length string column tests

# --- Setup: create table with VARCHAR (now TD_STR) ---

statement ok
CREATE TABLE str_test (id INTEGER, name VARCHAR, bio TEXT)

statement ok
INSERT INTO str_test VALUES (1, 'alice', 'short'), (2, 'bob', 'a medium length biography'), (3, 'carol', 'this is a longer biography that exceeds twelve bytes for pool storage')

# --- Basic SELECT ---

query IT
SELECT id, name FROM str_test ORDER BY id
----
1 alice
2 bob
3 carol

# --- String functions ---

query T
SELECT UPPER(name) FROM str_test WHERE id = 1
----
ALICE

query T
SELECT LOWER(name) FROM str_test WHERE id = 2
----
bob

query I
SELECT LENGTH(name) FROM str_test WHERE id = 1
----
5

query T
SELECT TRIM(name) FROM str_test WHERE id = 1
----
alice

query T
SELECT SUBSTR(bio, 1, 5) FROM str_test WHERE id = 1
----
short

query T
SELECT REPLACE(name, 'alice', 'ALICE') FROM str_test WHERE id = 1
----
ALICE

# --- String concatenation ---

query T
SELECT name || ' - ' || bio FROM str_test WHERE id = 1
----
alice - short

# --- WHERE on string columns ---

query IT
SELECT id, name FROM str_test WHERE name = 'bob'
----
2 bob

query IT
SELECT id, name FROM str_test WHERE name LIKE 'a%'
----
1 alice

query IT
SELECT id, name FROM str_test WHERE name ILIKE 'A%'
----
1 alice

# --- ORDER BY string ---

query T
SELECT name FROM str_test ORDER BY name
----
alice
bob
carol

# --- COUNT DISTINCT on strings ---

query I
SELECT COUNT(DISTINCT name) FROM str_test
----
3

# --- NULL strings ---

statement ok
INSERT INTO str_test VALUES (4, NULL, NULL)

query IT
SELECT id, name FROM str_test WHERE name IS NULL
----
4 NULL

query I
SELECT COUNT(*) FROM str_test WHERE name IS NOT NULL
----
3

# --- SYMBOL type (explicit dictionary encoding) ---

statement ok
CREATE TABLE sym_test (id INTEGER, category SYMBOL)

statement ok
INSERT INTO sym_test VALUES (1, 'cat_a'), (2, 'cat_b'), (3, 'cat_a')

query IT
SELECT id, category FROM sym_test ORDER BY id
----
1 cat_a
2 cat_b
3 cat_a

query I
SELECT COUNT(DISTINCT category) FROM sym_test
----
2

# --- CAST to VARCHAR (TD_STR) ---

query T
SELECT CAST(42 AS VARCHAR)
----
42
```

**Step 2: Run the new test**

Run: `cargo test --all-features -- slt 2>&1 | tail -20`
Expected: All SLT tests pass including the new str.slt.

**Step 3: Commit**

```bash
git add tests/slt/str.slt
git commit -m "test: add SLT tests for TD_STR columns and SYMBOL DDL keyword"
```

---

### Task 12: Update Documentation

**Files:**
- Modify: `CLAUDE.md` (architecture section)

**Step 1: Update CLAUDE.md type system section**

In `CLAUDE.md`, update the Type System section to mention `TD_STR`:

Add after `TD_SYM=20`:
```
TD_STR=21
```

Update the SQL Layer section to note that `VARCHAR`/`TEXT`/`STRING` now maps to `TD_STR` (variable-length strings), while `SYMBOL` maps to `TD_SYM` (dictionary-encoded).

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for TD_STR type and SYMBOL keyword"
```

---

### Task 13: Final Verification

**Step 1: Full test suite**

Run: `cargo test --all-features`
Expected: All tests pass.

**Step 2: Build all targets**

Run: `cargo build --all-features`
Expected: Clean build.

**Step 3: Run benchmarks (if teide-bench is available)**

Run: `cd ../teide-bench && cargo bench --all-features 2>&1 | tail -20`
Expected: No regressions.

**Step 4: Verify SLT tests specifically**

Run: `cargo test --all-features -- slt`
Expected: All SLT files pass.

**Step 5: Quick smoke test with the REPL**

Run: `echo "CREATE TABLE t (id INT, name VARCHAR); INSERT INTO t VALUES (1, 'hello'); SELECT * FROM t;" | cargo run --features cli`
Expected: Shows table with id=1, name=hello.
