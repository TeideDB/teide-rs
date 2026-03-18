# SQL Bugfix & Documentation Overhaul

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 core SQL bugs (DATE literal parsing, SELECT without FROM, CAST date strings) and update all documentation to accurately reflect actual behavior.

**Architecture:** Bugs are in the SQL planner layer (`src/sql/planner.rs` and `src/sql/expr.rs`). Each fix adds date/time string parsing functions, constant-expression evaluation without a FROM clause, and string-to-date conversion in CAST. Documentation updates span ~15 HTML files in `docs/`.

**Tech Stack:** Rust, sqlparser crate, SLT (SQL logic test) framework

---

## Part 1: Bug Fixes

### Task 1: Add date/time string literal parsing helpers

These helpers convert SQL date/time string literals to Teide's internal integer representations.

**Files:**
- Modify: `src/sql/planner.rs:1327-1380` (after existing `eval_*_literal` functions)

- [x] Step 1: Write the failing test
- [x] Step 2: Run test to verify it fails
- [x] Step 3: Write the implementation
- [x] Step 4: Run tests to verify they pass
- [x] Step 5: Commit

**Step 1: Write the failing test**

Create SLT tests for date literal INSERT:

Add to `tests/slt/temporal.slt` (append at end):

```
# === DATE/TIME/TIMESTAMP string literal INSERT ===

statement ok
CREATE TABLE date_lit (d DATE, t TIME, ts TIMESTAMP)

statement ok
INSERT INTO date_lit VALUES ('2000-01-01', '00:00:00', '2000-01-01 00:00:00')

query T
SELECT d FROM date_lit WHERE d = 0
----
2000-01-01

statement ok
INSERT INTO date_lit VALUES ('2025-01-01', '12:00:00', '2025-01-01 12:01:00')

query T
SELECT d FROM date_lit WHERE d = 9132
----
2025-01-01

query T
SELECT t FROM date_lit WHERE t = 43200000
----
12:00:00

statement ok
CREATE TABLE date_lit2 (id INTEGER, name VARCHAR, created_at DATE)

statement ok
INSERT INTO date_lit2 VALUES (1, 'Alice', '2024-01-15'), (2, 'Bob', '2024-03-22')

query IT
SELECT id, created_at FROM date_lit2 ORDER BY id
----
1 2024-01-15
2 2024-03-22
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- slt`
Expected: FAIL with "expected integer literal"

**Step 3: Write the implementation**

In `src/sql/planner.rs`, add these parsing helpers after `eval_str_literal` (after line 1380):

```rust
/// Parse a date string "YYYY-MM-DD" to days since 2000-01-01.
/// Uses the Hinnant civil_from_days algorithm (inverse of Table::format_date).
fn parse_date_str(s: &str) -> Result<i32, String> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("invalid date literal '{s}', expected YYYY-MM-DD"));
    }
    let y: i64 = parts[0]
        .parse()
        .map_err(|_| format!("invalid year in date '{s}'"))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid month in date '{s}'"))?;
    let d: u32 = parts[2]
        .parse()
        .map_err(|_| format!("invalid day in date '{s}'"))?;
    if m < 1 || m > 12 || d < 1 || d > 31 {
        return Err(format!("date out of range: '{s}'"));
    }
    // Hinnant days_from_civil: compute days since 1970-01-01, then adjust to 2000-01-01
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (m as u64) + 2) / 5 + (d as u64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146097 + doe as i64 - 719468; // days since 1970-01-01
    Ok((days_since_epoch - 10957) as i32) // adjust to 2000-01-01 epoch
}

/// Parse a time string "HH:MM:SS" or "HH:MM:SS.mmm" to milliseconds since midnight.
fn parse_time_str(s: &str) -> Result<i32, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("invalid time literal '{s}', expected HH:MM:SS"));
    }
    let h: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid hour in time '{s}'"))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid minute in time '{s}'"))?;
    // seconds may have fractional part
    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let s_val: u32 = sec_parts[0]
        .parse()
        .map_err(|e| format!("invalid second in time '{s}': {e}"))?;
    let ms = if sec_parts.len() > 1 {
        let frac = sec_parts[1];
        // Pad or truncate to 3 digits for milliseconds
        let padded = format!("{:0<3}", frac);
        padded[..3]
            .parse::<u32>()
            .map_err(|e| format!("invalid fractional seconds in time '{s}': {e}"))?
    } else {
        0
    };
    if h > 23 || m > 59 || s_val > 59 {
        return Err(format!("time out of range: '{s}'"));
    }
    Ok((h * 3_600_000 + m * 60_000 + s_val * 1_000 + ms) as i32)
}

/// Parse a timestamp string "YYYY-MM-DD HH:MM:SS[.ffffff]" to microseconds since 2000-01-01.
fn parse_timestamp_str(s: &str) -> Result<i64, String> {
    // Split on space or 'T' separator
    let sep_pos = s.find(' ').or_else(|| s.find('T'));
    let (date_part, time_part) = match sep_pos {
        Some(pos) => (&s[..pos], &s[pos + 1..]),
        None => return Err(format!("invalid timestamp '{s}', expected YYYY-MM-DD HH:MM:SS")),
    };
    let days = parse_date_str(date_part)?;
    // Parse time with microsecond precision
    let time_parts: Vec<&str> = time_part.split(':').collect();
    if time_parts.len() != 3 {
        return Err(format!("invalid time in timestamp '{s}'"));
    }
    let h: i64 = time_parts[0]
        .parse()
        .map_err(|_| format!("invalid hour in timestamp '{s}'"))?;
    let m: i64 = time_parts[1]
        .parse()
        .map_err(|_| format!("invalid minute in timestamp '{s}'"))?;
    let sec_parts: Vec<&str> = time_parts[2].split('.').collect();
    let secs: i64 = sec_parts[0]
        .parse()
        .map_err(|_| format!("invalid second in timestamp '{s}'"))?;
    let us = if sec_parts.len() > 1 {
        let frac = sec_parts[1];
        let padded = format!("{:0<6}", frac);
        padded[..6]
            .parse::<i64>()
            .map_err(|e| format!("invalid fractional seconds in timestamp '{s}': {e}"))?
    } else {
        0
    };
    let day_us = days as i64 * 86_400_000_000;
    let time_us = h * 3_600_000_000 + m * 60_000_000 + secs * 1_000_000 + us;
    Ok(day_us + time_us)
}
```

Then modify `append_value_to_vec` in the same file to try string literal parsing for DATE/TIME/TIMESTAMP columns. Replace the `ffi::TD_DATE | ffi::TD_TIME` match arm (lines 1284-1291) and `ffi::TD_TIMESTAMP` arm (lines 1293-1299):

```rust
        ffi::TD_DATE | ffi::TD_TIME => {
            // Try integer literal first, then date/time string literal
            let v32 = if let Ok(val) = eval_i64_literal(expr) {
                val as i32
            } else if let Ok(s) = eval_str_literal(expr) {
                if typ == ffi::TD_DATE {
                    parse_date_str(&s)
                        .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
                } else {
                    parse_time_str(&s)
                        .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
                }
            } else {
                return Err(SqlError::Plan(format!(
                    "column '{}': expected integer or date/time string literal, got {expr}",
                    col_names[col_idx]
                )));
            };
            let next =
                unsafe { ffi::td_vec_append(vec, &v32 as *const i32 as *const c_void) };
            check_vec_append(next)
        }

        ffi::TD_TIMESTAMP => {
            let val = if let Ok(v) = eval_i64_literal(expr) {
                v
            } else if let Ok(s) = eval_str_literal(expr) {
                parse_timestamp_str(&s)
                    .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
            } else {
                return Err(SqlError::Plan(format!(
                    "column '{}': expected integer or timestamp string literal, got {expr}",
                    col_names[col_idx]
                )));
            };
            let next =
                unsafe { ffi::td_vec_append(vec, &val as *const i64 as *const c_void) };
            check_vec_append(next)
        }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sql/planner.rs tests/slt/temporal.slt
git commit -m "fix: parse date/time/timestamp string literals in INSERT VALUES"
```

---

### Task 2: Support SELECT without FROM (constant expressions)

Allow `SELECT 1, 'hello', 3.14, true;` by synthesizing a single-row dummy table.

**Files:**
- Modify: `src/sql/planner.rs:2355-2363` (the `resolve_from` function)

- [x] Step 1: Write the failing test
- [x] Step 2: Run test to verify it fails
- [x] Step 3: Write the implementation
- [x] Step 4: Run tests to verify they pass
- [x] Step 5: Commit

**Step 1: Write the failing test**

Create `tests/slt/constant_select.slt`:

```
# SELECT without FROM clause — constant expressions

query I
SELECT 1
----
1

query T
SELECT 'hello'
----
hello

query R
SELECT 3.14
----
3.14

query B
SELECT true
----
true

query ITR
SELECT 1, 'hello', 3.14
----
1 hello 3.14

query I
SELECT 1 + 2
----
3

query I
SELECT CAST(42 AS BIGINT)
----
42

query R
SELECT 42.0
----
42.0

query T
SELECT 'foo' || 'bar'
----
foobar
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- slt`
Expected: FAIL with "Missing FROM clause"

**Step 3: Write the implementation**

In `src/sql/planner.rs`, modify the `resolve_from` function. Replace lines 2361-2363:

```rust
    if from.is_empty() {
        // Constant SELECT (no FROM clause): create a single-row dummy table
        // with one column so that constant expressions evaluate once.
        let dummy = Table::single_row(&ctx)?;
        let schema = HashMap::new();
        let emb_dims = HashMap::new();
        return Ok((dummy, schema, emb_dims));
    }
```

Then add a `Table::single_row` method in `src/engine.rs`:

```rust
    /// Create a single-row dummy table with one integer column (value 0).
    /// Used for constant expression evaluation (SELECT without FROM).
    pub fn single_row(ctx: &Context) -> Result<Table, Error> {
        let g = ctx.graph_empty()?;
        let one = g.const_i64(1)?;
        let result = g.execute(one)?;
        Ok(result)
    }
```

Wait — we need a table, not a column. The approach should be different. Let me use the raw table builder:

In `src/sql/planner.rs`, modify `resolve_from` (replace lines 2361-2363):

```rust
    if from.is_empty() {
        // Constant SELECT (no FROM): synthesize a 1-row table so expressions evaluate once.
        let mut builder = RawTableBuilder::new(1)?;
        let vec = unsafe { crate::raw::td_vec_new(crate::ffi::TD_I64, 1) };
        if vec.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        let zero: i64 = 0;
        let vec = unsafe {
            crate::ffi::td_vec_append(vec, &zero as *const i64 as *const std::ffi::c_void)
        };
        if vec.is_null() || crate::ffi_is_err(vec) {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        let name_id = crate::sym_intern("_dummy")?;
        let res = builder.add_col(name_id, vec);
        unsafe { crate::ffi_release(vec) };
        res?;
        let table = builder.finish()?;
        let schema = HashMap::new(); // no user-visible columns
        let emb_dims = HashMap::new();
        return Ok((table, schema, emb_dims));
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sql/planner.rs tests/slt/constant_select.slt
git commit -m "feat: support SELECT without FROM clause (constant expressions)"
```

---

### Task 3: Support CAST of string literals to DATE/TIME/TIMESTAMP

When `CAST('2024-01-15' AS DATE)` is used in a SELECT expression, the string should be parsed to the internal integer representation.

**Files:**
- Modify: `src/sql/expr.rs:160-173` (the CAST handler in `plan_expr`)

- [x] Step 1: Write the failing test
- [x] Step 2: Run test to verify it fails
- [x] Step 3: Write the implementation
- [x] Step 4: Run tests to verify they pass
- [x] Step 5: Commit

**Step 1: Write the failing test**

Add to `tests/slt/temporal.slt`:

```
# === CAST string to DATE/TIME/TIMESTAMP ===

statement ok
CREATE TABLE cast_test (id INTEGER)

statement ok
INSERT INTO cast_test VALUES (1)

query T
SELECT CAST('2025-01-01' AS DATE) FROM cast_test
----
2025-01-01

query T
SELECT CAST('12:30:00' AS TIME) FROM cast_test
----
12:30:00

query T
SELECT CAST('2025-06-15 09:30:00' AS TIMESTAMP) FROM cast_test
----
2025-06-15 09:30:00
```

**Step 2: Run test to verify it fails**

Run: `cargo test --all-features -- slt`
Expected: FAIL (CAST will cast string to date type but won't parse the string content)

**Step 3: Write the implementation**

In `src/sql/expr.rs`, modify the CAST handler (lines 160-173). When the inner expression is a string literal and the target is a temporal type, parse the string and emit a constant:

```rust
        Expr::Cast {
            expr: inner,
            data_type,
            kind,
            ..
        } => {
            if *kind == CastKind::TryCast {
                return Err(SqlError::Plan("TRY_CAST not supported".into()));
            }
            let target = map_sql_type(data_type)?;
            // Special case: string literal → temporal type: parse at plan time
            if let Expr::Value(Value::SingleQuotedString(s)) = inner.as_ref() {
                match target {
                    crate::types::DATE => {
                        let days = crate::sql::planner::parse_date_str(s)
                            .map_err(|e| SqlError::Plan(e))?;
                        return Ok(g.const_i32(days)?);
                    }
                    crate::types::TIME => {
                        let ms = crate::sql::planner::parse_time_str(s)
                            .map_err(|e| SqlError::Plan(e))?;
                        return Ok(g.const_i32(ms)?);
                    }
                    crate::types::TIMESTAMP => {
                        let us = crate::sql::planner::parse_timestamp_str(s)
                            .map_err(|e| SqlError::Plan(e))?;
                        return Ok(g.const_i64(us)?);
                    }
                    _ => {}
                }
            }
            let e = plan_expr(g, inner, schema)?;
            Ok(g.cast(e, target)?)
        }
```

This requires making `parse_date_str`, `parse_time_str`, and `parse_timestamp_str` public in planner.rs:

```rust
pub fn parse_date_str(...) -> ...
pub fn parse_time_str(...) -> ...
pub fn parse_timestamp_str(...) -> ...
```

And checking that `Graph::const_i32` exists (or adding it).

**Step 4: Run tests to verify they pass**

Run: `cargo test --all-features -- slt`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sql/planner.rs src/sql/expr.rs tests/slt/temporal.slt
git commit -m "feat: CAST string literals to DATE/TIME/TIMESTAMP"
```

---

## Part 2: Documentation Overhaul

### Task 4: Update data-types.html

**Files:**
- Modify: `docs/sql/data-types.html`

- [x] Step 1: Make the edits to `docs/sql/data-types.html`
- [x] Step 2: Commit

**Changes:**
- Add SMALLINT (I16) to numeric types table
- Fix TIME storage: should say "32-bit" (ms as i32), not "64-bit"
- Add "Literal Formats" section showing:
  - DATE: `'YYYY-MM-DD'` in INSERT, or integer (days since 2000-01-01)
  - TIME: `'HH:MM:SS[.mmm]'` in INSERT, or integer (milliseconds since midnight)
  - TIMESTAMP: `'YYYY-MM-DD HH:MM:SS[.ffffff]'` in INSERT, or integer (microseconds since 2000-01-01)
- Update CAST section with temporal CAST examples:
  ```sql
  SELECT CAST('2025-01-15' AS DATE);
  SELECT CAST('12:30:00' AS TIME);
  SELECT CAST('2025-01-15 09:30:00' AS TIMESTAMP);
  ```
- Document that `REAL` maps to F64 (not F32) — avoids confusion
- Add note: NULL is represented as NaN for float columns, 0 for integers, empty string for VARCHAR

**Step 1: Make the edits to `docs/sql/data-types.html`**

**Step 2: Commit**

```bash
git add docs/sql/data-types.html
git commit -m "docs: update data-types page with temporal literals, SMALLINT, CAST examples"
```

---

### Task 5: Update dml.html

**Files:**
- Modify: `docs/sql/dml.html`

**Changes:**
- Add temporal literal examples in INSERT:
  ```sql
  CREATE TABLE events (id INTEGER, name VARCHAR, event_date DATE, event_time TIME);
  INSERT INTO events VALUES (1, 'Launch', '2025-03-15', '09:00:00');
  ```
- Add UPDATE syntax and examples (currently completely missing from DML page):
  ```sql
  UPDATE t SET val = 100.0 WHERE id = 1;
  UPDATE t SET name = 'updated' WHERE val > 50.0;
  ```
- Add DELETE syntax and examples (currently completely missing):
  ```sql
  DELETE FROM t WHERE id = 3;
  DELETE FROM t;  -- truncates
  ```
- Add column-list INSERT example:
  ```sql
  INSERT INTO t (name, id) VALUES ('alice', 1);
  ```
- Document embedding column restrictions:
  - INSERT VALUES not supported for high-dimensional (dim > 1) embedding columns
  - INSERT SELECT is supported for embedding columns

**Step 1: Make the edits to `docs/sql/dml.html`**

**Step 2: Commit**

```bash
git add docs/sql/dml.html
git commit -m "docs: add UPDATE, DELETE, temporal literals, column-list INSERT to DML page"
```

---

### Task 6: Update expressions.html

**Files:**
- Modify: `docs/sql/expressions.html`

**Changes:**
- Add BETWEEN, IN, LIKE, ILIKE examples (currently missing)
- Add IS NULL / IS NOT NULL examples
- Add temporal CAST examples
- Document that `SELECT 1 + 2` works without FROM (after Task 2)
- Add string concatenation with `||` operator
- Document LEAST, GREATEST functions

**Step 1: Make the edits**

**Step 2: Commit**

```bash
git add docs/sql/expressions.html
git commit -m "docs: add BETWEEN, IN, LIKE, IS NULL, constant SELECT to expressions page"
```

---

### Task 7: Update queries.html

**Files:**
- Modify: `docs/sql/queries.html`

**Changes:**
- Add "Constant Expressions" section showing SELECT without FROM:
  ```sql
  SELECT 1, 'hello', 3.14;
  SELECT 1 + 2;
  SELECT CAST('2025-01-15' AS DATE);
  ```
- Add column aliasing with AS
- Add ORDER BY with expressions example

**Step 1: Make the edits**

**Step 2: Commit**

```bash
git add docs/sql/queries.html
git commit -m "docs: add constant SELECT and alias examples to queries page"
```

---

### Task 8: Update functions.html

**Files:**
- Modify: `docs/sql/functions.html`

**Changes:**
- Verify all documented functions actually work; fix any that don't
- Add DATE_DIFF / DATEDIFF function documentation
- Add EXTRACT with all supported fields (YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, DOW, EPOCH)
- Add DATE_TRUNC with all supported units (year, month, day, hour, minute, second)
- Add CURRENT_DATE and NOW() / CURRENT_TIMESTAMP
- Organize into clear categories: Math, String, Date/Time, Null-handling

**Step 1: Read and audit `docs/sql/functions.html`**

**Step 2: Make the edits**

**Step 3: Commit**

```bash
git add docs/sql/functions.html
git commit -m "docs: comprehensive update to scalar functions reference"
```

---

### Task 9: Update tutorial/index.html

**Files:**
- Modify: `docs/tutorial/index.html`

**Changes:**
- Add a "Step: Working with Dates" section showing:
  ```sql
  CREATE TABLE events (id INTEGER, name VARCHAR, event_date DATE);
  INSERT INTO events VALUES (1, 'Launch', '2025-03-15');
  SELECT name, event_date FROM events;
  SELECT name, EXTRACT(YEAR FROM event_date) FROM events;
  ```
- Add a brief "Constant Expressions" example
- Verify all existing examples compile and run correctly against the engine

**Step 1: Make the edits**

**Step 2: Commit**

```bash
git add docs/tutorial/index.html
git commit -m "docs: add date handling and constant expressions to tutorial"
```

---

### Task 10: Audit remaining doc pages

**Files:**
- Read and audit: `docs/sql/filtering.html`, `docs/sql/aggregation.html`, `docs/sql/joins.html`, `docs/sql/window-functions.html`, `docs/sql/sorting.html`, `docs/sql/set-operations.html`, `docs/sql/subqueries.html`, `docs/sql/ddl.html`, `docs/sql/table-functions.html`
- Read and audit: `docs/sql/vector-search.html`, `docs/sql/pgq.html`, `docs/sql/graph-algorithms.html`
- Read and audit: `docs/getting-started/index.html`, `docs/api/index.html`, `docs/cli/index.html`, `docs/server/index.html`

**For each page:**
1. Read the page
2. Cross-reference every code example against the actual planner/expr code
3. Fix any examples that would fail
4. Add missing features that are implemented but undocumented
5. Remove or mark any features that are documented but not implemented

**Key items to check:**
- `vector-search.html`: Document DML restrictions on embedding tables
- `pgq.html`: Verify all MATCH pattern examples work
- `graph-algorithms.html`: Verify CLUSTERING_COEFFICIENT alias documented
- `ddl.html`: Add CREATE OR REPLACE TABLE, IF NOT EXISTS, column types complete

**Step 1: Read each page and cross-reference**

**Step 2: Make targeted edits**

**Step 3: Commit per page or group**

```bash
git add docs/
git commit -m "docs: audit and fix all SQL reference pages"
```

---

### Task 11: Run full test suite and verify

**Step 1: Run all tests**

```bash
cargo test --all-features
```

Expected: ALL PASS

**Step 2: Run benchmarks**

```bash
cd ../teide-bench && cargo bench --all-features
```

Expected: No regressions

**Step 3: Final commit if needed**

---

## Implementation Notes

### Date epoch
- Teide uses **2000-01-01** as epoch for DATE (days) and TIMESTAMP (microseconds)
- TIME is milliseconds since midnight (stored as i32)
- The `format_date` function in `engine.rs:1048` uses the Hinnant algorithm; the inverse parsing must match exactly

### Dummy table for SELECT without FROM
- A 1-row, 1-column table with a `_dummy` column ensures constant expressions evaluate exactly once
- The `_dummy` column is not in the schema map, so `SELECT *` won't include it — but `SELECT *` with empty FROM isn't meaningful anyway
- Must handle `SELECT *` gracefully: either return empty result or the constant columns

### CAST string-to-temporal
- Only handles string **literals** at plan time (not column values at runtime)
- For column-to-date casting, the C engine's `g.cast()` handles it if the types are compatible
- `const_i32` may need to be added to Graph API if it doesn't exist

### Public vs private parse functions
- `parse_date_str`, `parse_time_str`, `parse_timestamp_str` must be `pub(crate)` so `expr.rs` can call them
