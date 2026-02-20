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

//! Integration tests for the Teide Graph API.
//!
//! Uses a small inline CSV (~20 rows) to exercise all major graph operations:
//! scan, filter, arithmetic, group-by, sort, join, head/tail, project, alias,
//! string ops, cast, and error handling.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;

use teide::{types, AggOp, Context, FrameBound, FrameType, Table, WindowFunc};

// The C engine uses global state — serialize all tests.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

const CSV_HEADER: &str = "id1,id2,id3,id4,id5,id6,v1,v2,v3";
const CSV_ROWS: &[&str] = &[
    "id001,id001,id0000000001,1,10,100,1,2,1.5",
    "id001,id001,id0000000002,2,20,200,2,3,2.5",
    "id001,id002,id0000000003,3,30,300,3,4,3.5",
    "id001,id002,id0000000004,1,10,100,4,5,4.5",
    "id002,id001,id0000000005,2,20,200,5,6,5.5",
    "id002,id001,id0000000006,3,30,300,6,7,6.5",
    "id002,id002,id0000000007,1,10,100,7,8,7.5",
    "id002,id002,id0000000008,2,20,200,8,9,8.5",
    "id003,id001,id0000000009,3,30,300,9,10,9.5",
    "id003,id001,id0000000010,1,10,100,10,11,10.5",
    "id003,id002,id0000000011,2,20,200,1,2,11.5",
    "id003,id002,id0000000012,3,30,300,2,3,12.5",
    "id004,id001,id0000000013,1,10,100,3,4,1.5",
    "id004,id001,id0000000014,2,20,200,4,5,2.5",
    "id004,id002,id0000000015,3,30,300,5,6,3.5",
    "id004,id002,id0000000016,1,10,100,6,7,4.5",
    "id005,id001,id0000000017,2,20,200,7,8,5.5",
    "id005,id001,id0000000018,3,30,300,8,9,6.5",
    "id005,id002,id0000000019,1,10,100,9,10,7.5",
    "id005,id002,id0000000020,2,20,200,10,11,8.5",
];

fn create_test_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{CSV_HEADER}").unwrap();
    for row in CSV_ROWS {
        writeln!(f, "{row}").unwrap();
    }
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

fn create_join_right_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "id1,x1").unwrap();
    writeln!(f, "id001,100").unwrap();
    writeln!(f, "id002,200").unwrap();
    writeln!(f, "id003,300").unwrap();
    writeln!(f, "id006,400").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

/// Collect column 0 (string) → column 1 (i64) into a map.
fn collect_str_i64(table: &Table) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    for row in 0..table.nrows() as usize {
        let key = table.get_str(0, row).unwrap().to_string();
        let val = table.get_i64(1, row).unwrap();
        map.insert(key, val);
    }
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn csv_read() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    assert_eq!(table.nrows(), 20);
    assert_eq!(table.ncols(), 9);

    // Column names
    assert_eq!(table.col_name_str(0), "id1");
    assert_eq!(table.col_name_str(1), "id2");
    assert_eq!(table.col_name_str(2), "id3");
    assert_eq!(table.col_name_str(3), "id4");
    assert_eq!(table.col_name_str(6), "v1");
    assert_eq!(table.col_name_str(8), "v3");

    // Column types: id1-id3 are SYM (CSV now produces TD_SYM), id4-id6/v1-v2 are I64, v3 is F64
    assert_eq!(table.col_type(0), teide::types::SYM);
    assert_eq!(table.col_type(3), teide::types::I64);
    assert_eq!(table.col_type(6), teide::types::I64);
    assert_eq!(table.col_type(8), teide::types::F64);
}

#[test]
fn scan_and_filter() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let three = g.const_i64(3).unwrap();
    let pred = g.gt(v1, three).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();

    // v1 values > 3: 4,5,6,7,8,9,10 appear twice each = 14 rows
    assert_eq!(result.nrows(), 14);
    assert_eq!(result.ncols(), 9);
}

#[test]
fn arithmetic() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let v2 = g.scan("v2").unwrap();
    let sum_col = g.add(v1, v2).unwrap();
    let aliased = g.alias(sum_col, "v1_plus_v2").unwrap();
    let result_tbl = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(result_tbl).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 1);

    // Row 0: v1=1, v2=2 → 3
    assert_eq!(result.get_i64(0, 0).unwrap(), 3);
    // Row 4: v1=5, v2=6 → 11
    assert_eq!(result.get_i64(0, 4).unwrap(), 11);
    // Row 19: v1=10, v2=11 → 21
    assert_eq!(result.get_i64(0, 19).unwrap(), 21);
}

#[test]
fn group_by_sum() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let v1 = g.scan("v1").unwrap();
    let grp = g.group_by(&[id1], &[AggOp::Sum], &[v1]).unwrap();
    let result = g.execute(grp).unwrap();

    assert_eq!(result.nrows(), 5);
    assert_eq!(result.ncols(), 2);

    let sums = collect_str_i64(&result);
    assert_eq!(sums["id001"], 10);
    assert_eq!(sums["id002"], 26);
    assert_eq!(sums["id003"], 22);
    assert_eq!(sums["id004"], 18);
    assert_eq!(sums["id005"], 34);
}

#[test]
fn group_by_multi_agg() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let v1a = g.scan("v1").unwrap();
    let v3a = g.scan("v3").unwrap();
    let v1b = g.scan("v1").unwrap();
    let grp = g
        .group_by(
            &[id1],
            &[AggOp::Sum, AggOp::Avg, AggOp::Count],
            &[v1a, v3a, v1b],
        )
        .unwrap();
    let result = g.execute(grp).unwrap();

    // 5 groups, 4 columns (id1 + 3 aggs)
    assert_eq!(result.nrows(), 5);
    assert_eq!(result.ncols(), 4);

    // Verify SUM(v1) via column 1
    let mut sum_map = HashMap::new();
    for row in 0..5 {
        let key = result.get_str(0, row).unwrap().to_string();
        let sum_val = result.get_i64(1, row).unwrap();
        sum_map.insert(key, sum_val);
    }
    assert_eq!(sum_map["id001"], 10);
    assert_eq!(sum_map["id005"], 34);

    // Verify AVG(v3) via column 2 — each group has 4 rows
    let mut avg_map = HashMap::new();
    for row in 0..5 {
        let key = result.get_str(0, row).unwrap().to_string();
        let avg_val = result.get_f64(2, row).unwrap();
        avg_map.insert(key, avg_val);
    }
    assert!((avg_map["id001"] - 3.0).abs() < 1e-10);
    assert!((avg_map["id002"] - 7.0).abs() < 1e-10);
    assert!((avg_map["id003"] - 11.0).abs() < 1e-10);

    // Verify COUNT(v1) via column 3 — all groups have 4 rows
    for row in 0..5 {
        assert_eq!(result.get_i64(3, row).unwrap(), 4);
    }
}

#[test]
fn group_by_multi_key() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let id4 = g.scan("id4").unwrap();
    let v1 = g.scan("v1").unwrap();
    let grp = g.group_by(&[id1, id4], &[AggOp::Sum], &[v1]).unwrap();
    let result = g.execute(grp).unwrap();

    // 15 distinct (id1, id4) groups
    assert_eq!(result.nrows(), 15);
    assert_eq!(result.ncols(), 3);
}

#[test]
fn sort_single_asc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id4 = g.scan("id4").unwrap();
    let sorted = g.sort(tbl, &[id4], &[false], None).unwrap();
    let result = g.execute(sorted).unwrap();

    assert_eq!(result.nrows(), 20);
    // Verify ascending: id4 col is index 3
    let mut prev = i64::MIN;
    for row in 0..20 {
        let val = result.get_i64(3, row).unwrap();
        assert!(val >= prev, "row {row}: {val} < {prev}");
        prev = val;
    }
}

#[test]
fn sort_single_desc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v3 = g.scan("v3").unwrap();
    let sorted = g.sort(tbl, &[v3], &[true], None).unwrap();
    let result = g.execute(sorted).unwrap();

    assert_eq!(result.nrows(), 20);
    // Verify descending: v3 col is index 8
    let mut prev = f64::MAX;
    for row in 0..20 {
        let val = result.get_f64(8, row).unwrap();
        assert!(val <= prev, "row {row}: {val} > {prev}");
        prev = val;
    }
}

#[test]
fn sort_multi_key() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let id4 = g.scan("id4").unwrap();
    let sorted = g.sort(tbl, &[id1, id4], &[false, false], None).unwrap();
    let result = g.execute(sorted).unwrap();

    assert_eq!(result.nrows(), 20);
    // Verify: id1 (col 0) is non-decreasing; within same id1, id4 (col 3) is non-decreasing
    for row in 1..20 {
        let prev_id1 = result.get_str(0, row - 1).unwrap();
        let cur_id1 = result.get_str(0, row).unwrap();
        assert!(cur_id1 >= prev_id1, "id1 not sorted at row {row}");
        if cur_id1 == prev_id1 {
            let prev_id4 = result.get_i64(3, row - 1).unwrap();
            let cur_id4 = result.get_i64(3, row).unwrap();
            assert!(
                cur_id4 >= prev_id4,
                "id4 not sorted within same id1 at row {row}"
            );
        }
    }
}

#[test]
fn sort_with_limit() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id4 = g.scan("id4").unwrap();
    let sorted = g.sort(tbl, &[id4], &[false], None).unwrap();
    let limited = g.head(sorted, 3).unwrap();
    let result = g.execute(limited).unwrap();

    assert_eq!(result.nrows(), 3);
    // First 3 rows should all have id4=1 (smallest value)
    for row in 0..3 {
        assert_eq!(result.get_i64(3, row).unwrap(), 1);
    }
}

#[test]
fn join_inner() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_left_file, left_path) = create_test_csv();
    let (_right_file, right_path) = create_join_right_csv();
    let ctx = Context::new().unwrap();
    let left = ctx.read_csv(&left_path).unwrap();
    let right = ctx.read_csv(&right_path).unwrap();

    assert_eq!(right.col_name_str(0), "id1");

    let mut g = ctx.graph(&left).unwrap();
    let left_table = g.const_table(&left).unwrap();
    let right_table = g.const_table(&right).unwrap();
    let left_id1 = g.scan("id1").unwrap();
    let right_id1_vec = right.get_col_idx(0).unwrap();
    let right_id1 = unsafe { g.const_vec(right_id1_vec).unwrap() };

    let joined = g
        .join(left_table, &[left_id1], right_table, &[right_id1], 0)
        .unwrap();
    let result = g.execute(joined).unwrap();

    // id001(4) + id002(4) + id003(4) = 12 matched rows
    assert_eq!(result.nrows(), 12);
    // 9 left cols + 2 right cols = 11
    assert_eq!(result.ncols(), 11);
}

#[test]
fn join_left() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_left_file, left_path) = create_test_csv();
    let (_right_file, right_path) = create_join_right_csv();
    let ctx = Context::new().unwrap();
    let left = ctx.read_csv(&left_path).unwrap();
    let right = ctx.read_csv(&right_path).unwrap();

    let mut g = ctx.graph(&left).unwrap();
    let left_table = g.const_table(&left).unwrap();
    let right_table = g.const_table(&right).unwrap();
    let left_id1 = g.scan("id1").unwrap();
    let right_id1_vec = right.get_col_idx(0).unwrap();
    let right_id1 = unsafe { g.const_vec(right_id1_vec).unwrap() };

    let joined = g
        .join(left_table, &[left_id1], right_table, &[right_id1], 1)
        .unwrap();
    let result = g.execute(joined).unwrap();

    // All 20 left rows preserved
    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 11);
}

#[test]
fn head_tail() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    // head(5)
    let tbl = g.const_table(&table).unwrap();
    let h = g.head(tbl, 5).unwrap();
    let result = g.execute(h).unwrap();
    assert_eq!(result.nrows(), 5);
    assert_eq!(result.ncols(), 9);
    // First row should have v1=1
    assert_eq!(result.get_i64(6, 0).unwrap(), 1);

    // tail(5) — need a fresh graph
    drop(result);
    let g2 = ctx.graph(&table).unwrap();
    let df2 = g2.const_table(&table).unwrap();
    let t = g2.tail(df2, 5).unwrap();
    let result2 = g2.execute(t).unwrap();
    assert_eq!(result2.nrows(), 5);
    // Last row of original has v1=10
    assert_eq!(result2.get_i64(6, 4).unwrap(), 10);
}

#[test]
fn project() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let v1 = g.scan("v1").unwrap();
    let projected = g.select(tbl, &[id1, v1]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);
    assert_eq!(result.col_name_str(0), "id1");
    assert_eq!(result.col_name_str(1), "v1");
}

#[test]
fn alias_column() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let aliased = g.alias(v1, "my_v1").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.ncols(), 1);
    // The alias passes through the value correctly
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    assert_eq!(result.get_i64(0, 4).unwrap(), 5);
}

#[test]
fn string_ops() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let upper_col = g.upper(id1).unwrap();
    let upper_aliased = g.alias(upper_col, "upper_id1").unwrap();

    let id1b = g.scan("id1").unwrap();
    let len_col = g.strlen(id1b).unwrap();
    let len_aliased = g.alias(len_col, "len_id1").unwrap();

    let id1c = g.scan("id1").unwrap();
    let lower_col = g.lower(id1c).unwrap();
    let lower_aliased = g.alias(lower_col, "lower_id1").unwrap();

    let projected = g
        .select(tbl, &[upper_aliased, len_aliased, lower_aliased])
        .unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 3);

    // UPPER("id001") = "ID001"
    assert_eq!(result.get_str(0, 0).unwrap(), "ID001");
    // LENGTH("id001") = 5
    assert_eq!(result.get_i64(1, 0).unwrap(), 5);
    // LOWER("id001") = "id001" (already lowercase)
    assert_eq!(result.get_str(2, 0).unwrap(), "id001");
}

#[test]
fn concat_many_args() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let mut args = Vec::new();
    args.push(g.scan("id1").unwrap());
    for _ in 0..19 {
        args.push(g.const_str("x").unwrap());
    }

    let concat_col = g.concat(&args).unwrap();
    let aliased = g.alias(concat_col, "concat_many").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 1);
    let expected = format!("{}{}", "id001", "x".repeat(19));
    assert_eq!(result.get_str(0, 0).unwrap(), expected);
}

#[test]
fn cast_i64_to_f64() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let casted = g.cast(v1, teide::types::F64).unwrap();
    let aliased = g.alias(casted, "v1_f64").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.ncols(), 1);
    assert_eq!(result.col_type(0), teide::types::F64);
    assert!((result.get_f64(0, 0).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.get_f64(0, 4).unwrap() - 5.0).abs() < 1e-10);
}

#[test]
fn error_handling_bad_csv() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let ctx = Context::new().unwrap();
    let result = ctx.read_csv("/nonexistent/path/to/file.csv");
    match result {
        Err(teide::Error::Io) => {} // expected
        Err(other) => panic!("expected Io error, got: {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// Window function tests (Graph API)
// ---------------------------------------------------------------------------

#[test]
fn window_row_number() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 ASC)
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let part_key = g.scan("id1").unwrap();
    let order_key = g.scan("v1").unwrap();
    let dummy_input = g.scan("v1").unwrap();

    let win = g
        .window_op(
            tbl,
            &[part_key],
            &[order_key],
            &[false], // ASC
            &[WindowFunc::RowNumber],
            &[dummy_input],
            FrameType::Rows,
            FrameBound::UnboundedPreceding,
            FrameBound::UnboundedFollowing,
        )
        .unwrap();
    let result = g.execute(win).unwrap();

    // 20 rows, original 9 cols + 1 window col = 10
    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 10);

    // Each id1 partition has 4 rows → row_number should be 1-4 within each
    // Collect window col values grouped by id1
    let mut parts: HashMap<String, Vec<i64>> = HashMap::new();
    for r in 0..result.nrows() as usize {
        let key = result.get_str(0, r).unwrap().to_string();
        let rn = result.get_i64(9, r).unwrap();
        parts.entry(key).or_default().push(rn);
    }
    for vals in parts.values() {
        let mut sorted = vals.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![1, 2, 3, 4],
            "ROW_NUMBER should be 1..4 per partition"
        );
    }
}

#[test]
fn window_rank_with_ties() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // RANK() OVER (PARTITION BY id2 ORDER BY id4 ASC)
    // id2 has 2 distinct values, id4 has values 1-3 (with ties)
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let part_key = g.scan("id2").unwrap();
    let order_key = g.scan("id4").unwrap();
    let dummy = g.scan("id4").unwrap();

    let win = g
        .window_op(
            tbl,
            &[part_key],
            &[order_key],
            &[false], // ASC
            &[WindowFunc::Rank, WindowFunc::DenseRank],
            &[dummy, dummy],
            FrameType::Rows,
            FrameBound::UnboundedPreceding,
            FrameBound::UnboundedFollowing,
        )
        .unwrap();
    let result = g.execute(win).unwrap();

    // 20 rows, 9 + 2 window cols = 11
    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 11);

    // Verify RANK values are >= 1 and DENSE_RANK values are >= 1
    for r in 0..result.nrows() as usize {
        let rank = result.get_i64(9, r).unwrap();
        let dense_rank = result.get_i64(10, r).unwrap();
        assert!(rank >= 1, "RANK should be >= 1, got {rank}");
        assert!(
            dense_rank >= 1,
            "DENSE_RANK should be >= 1, got {dense_rank}"
        );
        assert!(dense_rank <= rank, "DENSE_RANK <= RANK");
    }
}

#[test]
fn window_running_sum() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // SUM(v1) OVER (ORDER BY v1 ASC ROWS UNBOUNDED PRECEDING TO CURRENT ROW)
    // No partition → entire table is one partition
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let order_key = g.scan("v1").unwrap();
    let sum_input = g.scan("v1").unwrap();

    let win = g
        .window_op(
            tbl,
            &[], // no partition
            &[order_key],
            &[false], // ASC
            &[WindowFunc::Sum],
            &[sum_input],
            FrameType::Rows,
            FrameBound::UnboundedPreceding,
            FrameBound::CurrentRow,
        )
        .unwrap();
    let result = g.execute(win).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 10);

    // The running sum should be monotonically non-decreasing
    // since v1 values sorted ASC are all positive
    // SUM(I64) → I64 result
    let mut sums: Vec<i64> = Vec::new();
    for r in 0..result.nrows() as usize {
        let s = result.get_i64(9, r).unwrap();
        sums.push(s);
    }
    sums.sort();
    let mut prev = 0i64;
    for s in &sums {
        assert!(*s >= prev, "running sum should be non-decreasing");
        prev = *s;
    }

    // Total sum of v1 = 1+2+3+4+5+6+7+8+9+10 + 1+2+3+4+5+6+7+8+9+10 = 110
    // The maximum running sum should equal the total
    let max_sum = *sums.last().unwrap();
    assert_eq!(
        max_sum, 110,
        "total running sum should be 110, got {max_sum}"
    );
}

// ---------------------------------------------------------------------------
// Date / Time / Timestamp CSV parsing
// ---------------------------------------------------------------------------

fn create_datetime_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "date,time,timestamp,value").unwrap();
    writeln!(f, "2024-01-15,09:30:00,2024-01-15T09:30:00,100").unwrap();
    writeln!(
        f,
        "2024-06-30,14:15:30.500000,2024-06-30 14:15:30.500000,200"
    )
    .unwrap();
    writeln!(f, "1970-01-01,00:00:00,1970-01-01T00:00:00,300").unwrap();
    writeln!(f, "2000-03-01,23:59:59,2000-03-01T23:59:59,400").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

#[test]
fn csv_date_time_auto_infer() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_datetime_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    assert_eq!(table.nrows(), 4);
    assert_eq!(table.ncols(), 4);

    // Auto-inferred types
    assert_eq!(table.col_type(0), types::DATE);
    assert_eq!(table.col_type(1), types::TIME);
    assert_eq!(table.col_type(2), types::TIMESTAMP);
    assert_eq!(table.col_type(3), types::I64);

    // DATE: 1970-01-01 → 0 days since epoch
    assert_eq!(table.get_i64(0, 2).unwrap(), 0);
    // DATE: 2024-01-15 → 19737 days since epoch
    // (53 years * 365 + 13 leap days + 14 days = 19372 + 365 = 19737)
    let d0 = table.get_i64(0, 0).unwrap();
    assert!(
        d0 > 19700 && d0 < 19800,
        "2024-01-15 should be ~19737 days, got {d0}"
    );

    // TIME: 00:00:00 → 0 microseconds
    assert_eq!(table.get_i64(1, 2).unwrap(), 0);
    // TIME: 09:30:00 → 9*3600 + 30*60 = 34200 seconds = 34200000000 µs
    assert_eq!(table.get_i64(1, 0).unwrap(), 34_200_000_000);
    // TIME: 14:15:30.500000 → 51330.5 seconds = 51330500000 µs
    assert_eq!(table.get_i64(1, 1).unwrap(), 51_330_500_000);

    // TIMESTAMP: 1970-01-01T00:00:00 → 0 µs since epoch
    assert_eq!(table.get_i64(2, 2).unwrap(), 0);
    // TIMESTAMP: 2024-01-15T09:30:00 → d0 * 86400000000 + 34200000000
    let ts0 = table.get_i64(2, 0).unwrap();
    let expected_ts = d0 * 86_400_000_000 + 34_200_000_000;
    assert_eq!(
        ts0, expected_ts,
        "timestamp should be days*86400M + time_us"
    );
}

#[test]
fn csv_explicit_date_types() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_datetime_csv();
    let ctx = Context::new().unwrap();

    // Force explicit types: DATE, TIME, TIMESTAMP, I64
    let col_types = [types::DATE, types::TIME, types::TIMESTAMP, types::I64];
    let table = ctx
        .read_csv_opts(&path, ',', true, Some(&col_types))
        .unwrap();

    assert_eq!(table.col_type(0), types::DATE);
    assert_eq!(table.col_type(1), types::TIME);
    assert_eq!(table.col_type(2), types::TIMESTAMP);
    assert_eq!(table.col_type(3), types::I64);

    // Epoch date/time should be zero
    assert_eq!(table.get_i64(0, 2).unwrap(), 0);
    assert_eq!(table.get_i64(1, 2).unwrap(), 0);
    assert_eq!(table.get_i64(2, 2).unwrap(), 0);
}

#[test]
fn cancel_resets_between_queries() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // Assumption: td_execute() resets the cancel flag at entry, before any
    // morsel dispatch. This test relies on that C-side implementation detail.
    // If td_execute changes to check the flag before resetting it, this test
    // would fail and need updating.
    teide::cancel();

    let g = ctx.graph(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let sum = g.sum(v1).unwrap();
    let result = g.execute(sum);
    // The query succeeds because td_execute resets the flag first.
    assert!(result.is_ok());

    // Verify next query also works after cancel was consumed
    let g2 = ctx.graph(&table).unwrap();
    let v1b = g2.scan("v1").unwrap();
    let sum2 = g2.sum(v1b).unwrap();
    let result2 = g2.execute(sum2);
    assert!(result2.is_ok());
}
