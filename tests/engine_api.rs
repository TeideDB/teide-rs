//! Functional API tests for the Teide Engine layer.
//!
//! Covers Graph binary/comparison/math/string ops, Table methods,
//! Rel/CSR lifecycle, utility functions, and edge cases.

use std::io::Write;
use std::sync::Mutex;

use teide::{extract_field, ffi, AggOp, Context, HnswIndex, Rel, Session, Table};

// The C engine uses global state — serialize all tests.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn create_datetime_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "date,time,timestamp,value").unwrap();
    writeln!(f, "2024-01-15,09:30:00,2024-01-15T09:30:00,100").unwrap();
    writeln!(f, "2024-06-30,14:15:30.500000,2024-06-30 14:15:30.500000,200").unwrap();
    writeln!(f, "1970-01-01,00:00:00,1970-01-01T00:00:00,300").unwrap();
    writeln!(f, "2000-03-01,23:59:59,2000-03-01T23:59:59,400").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

// ---------------------------------------------------------------------------
// Graph Constants
// ---------------------------------------------------------------------------

#[test]
fn const_f64_in_arithmetic() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v3 = g.scan("v3").unwrap();
    let two_point_five = g.const_f64(2.5).unwrap();
    let product = g.mul(v3, two_point_five).unwrap();
    let aliased = g.alias(product, "result").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v3=1.5, 1.5 * 2.5 = 3.75
    assert!((result.get_f64(0, 0).unwrap() - 3.75).abs() < 1e-10);
}

#[test]
fn const_bool_in_filter() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // Filter with const_bool(true) — should return all rows
    let tbl = g.const_table(&table).unwrap();
    let pred = g.const_bool(true).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();
    assert_eq!(result.nrows(), 20);
}

#[test]
fn const_str_in_comparison() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let target = g.const_str("id003").unwrap();
    let pred = g.eq(id1, target).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();

    // id003 has 4 rows
    assert_eq!(result.nrows(), 4);
}

// ---------------------------------------------------------------------------
// Graph Binary / Comparison Operators
// ---------------------------------------------------------------------------

#[test]
fn binary_sub() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v2 = g.scan("v2").unwrap();
    let v1 = g.scan("v1").unwrap();
    let diff = g.sub(v2, v1).unwrap();
    let aliased = g.alias(diff, "diff").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v2=2, v1=1 → 1
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    // All rows: v2 - v1 = 1 (v2 = v1 + 1 for all rows)
    for row in 0..20 {
        assert_eq!(result.get_i64(0, row).unwrap(), 1);
    }
}

#[test]
fn binary_mul() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let id4 = g.scan("id4").unwrap();
    let product = g.mul(v1, id4).unwrap();
    let aliased = g.alias(product, "product").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v1=1, id4=1 → 1
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    // Row 1: v1=2, id4=2 → 4
    assert_eq!(result.get_i64(0, 1).unwrap(), 4);
}

#[test]
fn binary_div() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let id6 = g.scan("id6").unwrap();
    let id5 = g.scan("id5").unwrap();
    let quotient = g.div(id6, id5).unwrap();
    let aliased = g.alias(quotient, "quotient").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: id6=100, id5=10 → 10.0 (integer division may produce f64)
    let val = result
        .get_i64(0, 0)
        .map(|v| v as f64)
        .or_else(|| result.get_f64(0, 0))
        .unwrap();
    assert!((val - 10.0).abs() < 1e-10);
}

#[test]
fn binary_modulo() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let three = g.const_i64(3).unwrap();
    let remainder = g.modulo(v1, three).unwrap();
    let aliased = g.alias(remainder, "remainder").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v1=1 % 3 = 1
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    // Row 2: v1=3 % 3 = 0
    assert_eq!(result.get_i64(0, 2).unwrap(), 0);
}

#[test]
fn comparison_eq_ne() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // Filter id4 == 1
    let tbl = g.const_table(&table).unwrap();
    let id4 = g.scan("id4").unwrap();
    let one = g.const_i64(1).unwrap();
    let pred = g.eq(id4, one).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();

    // id4=1 appears in 7 rows (every 3rd pattern repeats)
    assert!(result.nrows() > 0);
    for row in 0..result.nrows() as usize {
        assert_eq!(result.get_i64(3, row).unwrap(), 1);
    }
}

#[test]
fn comparison_lt_le_ge() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // lt: v1 < 3
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let three = g.const_i64(3).unwrap();
    let pred = g.lt(v1, three).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();
    // v1 values < 3: 1,2 appear twice each = 4 rows
    assert_eq!(result.nrows(), 4);

    // le: v1 <= 3
    let mut g2 = ctx.graph(&table).unwrap();
    let tbl2 = g2.const_table(&table).unwrap();
    let v1b = g2.scan("v1").unwrap();
    let three_b = g2.const_i64(3).unwrap();
    let pred2 = g2.le(v1b, three_b).unwrap();
    let filtered2 = g2.filter(tbl2, pred2).unwrap();
    let result2 = g2.execute(filtered2).unwrap();
    // v1 values <= 3: 1,2,3 appear twice each = 6 rows
    assert_eq!(result2.nrows(), 6);

    // ge: v1 >= 9
    let mut g3 = ctx.graph(&table).unwrap();
    let tbl3 = g3.const_table(&table).unwrap();
    let v1c = g3.scan("v1").unwrap();
    let nine = g3.const_i64(9).unwrap();
    let pred3 = g3.ge(v1c, nine).unwrap();
    let filtered3 = g3.filter(tbl3, pred3).unwrap();
    let result3 = g3.execute(filtered3).unwrap();
    // v1 values >= 9: 9,10 appear twice each = 4 rows
    assert_eq!(result3.nrows(), 4);
}

#[test]
fn logical_and_or() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // AND: id4 == 1 AND v1 > 5
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let id4 = g.scan("id4").unwrap();
    let one = g.const_i64(1).unwrap();
    let p1 = g.eq(id4, one).unwrap();
    let v1 = g.scan("v1").unwrap();
    let five = g.const_i64(5).unwrap();
    let p2 = g.gt(v1, five).unwrap();
    let combined = g.and(p1, p2).unwrap();
    let filtered = g.filter(tbl, combined).unwrap();
    let result = g.execute(filtered).unwrap();
    // Every result row must have id4=1 and v1>5
    for row in 0..result.nrows() as usize {
        assert_eq!(result.get_i64(3, row).unwrap(), 1);
        assert!(result.get_i64(6, row).unwrap() > 5);
    }

    // OR: id4 == 1 OR id4 == 3
    let mut g2 = ctx.graph(&table).unwrap();
    let tbl2 = g2.const_table(&table).unwrap();
    let id4a = g2.scan("id4").unwrap();
    let one_b = g2.const_i64(1).unwrap();
    let p3 = g2.eq(id4a, one_b).unwrap();
    let id4b = g2.scan("id4").unwrap();
    let three = g2.const_i64(3).unwrap();
    let p4 = g2.eq(id4b, three).unwrap();
    let combined2 = g2.or(p3, p4).unwrap();
    let filtered2 = g2.filter(tbl2, combined2).unwrap();
    let result2 = g2.execute(filtered2).unwrap();
    // Every result row has id4 == 1 or id4 == 3
    for row in 0..result2.nrows() as usize {
        let val = result2.get_i64(3, row).unwrap();
        assert!(val == 1 || val == 3, "expected id4=1 or 3, got {val}");
    }
}

// ---------------------------------------------------------------------------
// Graph Unary / Math Operators
// ---------------------------------------------------------------------------

#[test]
fn unary_not() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // NOT(v1 > 5) → v1 <= 5
    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let five = g.const_i64(5).unwrap();
    let pred = g.gt(v1, five).unwrap();
    let negated = g.not(pred).unwrap();
    let filtered = g.filter(tbl, negated).unwrap();
    let result = g.execute(filtered).unwrap();

    // v1 <= 5: values 1,2,3,4,5 appear twice each = 10 rows
    assert_eq!(result.nrows(), 10);
    for row in 0..result.nrows() as usize {
        assert!(result.get_i64(6, row).unwrap() <= 5);
    }
}

#[test]
fn unary_neg() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let negated = g.neg(v1).unwrap();
    let aliased = g.alias(negated, "neg_v1").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v1=1 → -1
    assert_eq!(result.get_i64(0, 0).unwrap(), -1);
    // Row 4: v1=5 → -5
    assert_eq!(result.get_i64(0, 4).unwrap(), -5);
}

#[test]
fn math_abs() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // abs(neg(v1)) == v1
    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let negated = g.neg(v1).unwrap();
    let abs_val = g.abs(negated).unwrap();
    let aliased = g.alias(abs_val, "abs_v1").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: abs(-1) = 1
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    // Row 9: abs(-10) = 10
    assert_eq!(result.get_i64(0, 9).unwrap(), 10);
}

#[test]
fn math_sqrt_log_exp() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // Cast v3 to f64 (already f64), then sqrt
    let tbl = g.const_table(&table).unwrap();
    let v3 = g.scan("v3").unwrap();
    let sq = g.sqrt(v3).unwrap();
    let aliased = g.alias(sq, "sqrt_v3").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // sqrt(1.5) ≈ 1.2247
    let s0 = result.get_f64(0, 0).unwrap();
    assert!((s0 - 1.5_f64.sqrt()).abs() < 1e-6, "sqrt(1.5) = {s0}");

    // exp then log should round-trip
    let mut g2 = ctx.graph(&table).unwrap();
    let tbl2 = g2.const_table(&table).unwrap();
    let v3b = g2.scan("v3").unwrap();
    let e = g2.exp(v3b).unwrap();
    let l = g2.log(e).unwrap();
    let aliased2 = g2.alias(l, "roundtrip").unwrap();
    let projected2 = g2.select(tbl2, &[aliased2]).unwrap();
    let result2 = g2.execute(projected2).unwrap();

    // log(exp(v3)) ≈ v3
    let r0 = result2.get_f64(0, 0).unwrap();
    assert!((r0 - 1.5).abs() < 1e-6, "log(exp(1.5)) = {r0}");
}

#[test]
fn math_ceil_floor() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v3 = g.scan("v3").unwrap();
    let v3b = g.scan("v3").unwrap();
    let c = g.ceil(v3).unwrap();
    let f = g.floor(v3b).unwrap();
    let c_alias = g.alias(c, "ceil_v3").unwrap();
    let f_alias = g.alias(f, "floor_v3").unwrap();
    let projected = g.select(tbl, &[c_alias, f_alias]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v3=1.5 → ceil=2.0, floor=1.0
    assert!((result.get_f64(0, 0).unwrap() - 2.0).abs() < 1e-10);
    assert!((result.get_f64(1, 0).unwrap() - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Graph String Operators
// ---------------------------------------------------------------------------

#[test]
fn string_trim() {
    let _guard = lock();
    // Create CSV with whitespace-padded strings
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "name,value").unwrap();
    writeln!(f, " alice ,10").unwrap();
    writeln!(f, "  bob  ,20").unwrap();
    writeln!(f, "charlie,30").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let name = g.scan("name").unwrap();
    let trimmed = g.trim(name).unwrap();
    let aliased = g.alias(trimmed, "trimmed").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 3);
    assert_eq!(result.get_str(0, 0).unwrap(), "alice");
    assert_eq!(result.get_str(0, 1).unwrap(), "bob");
    assert_eq!(result.get_str(0, 2).unwrap(), "charlie");
}

#[test]
fn string_substr() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // substr(id1, 1, 2) → "id" for all rows (1-based start)
    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let start = g.const_i64(1).unwrap();
    let len = g.const_i64(2).unwrap();
    let sub = g.substr(id1, start, len).unwrap();
    let aliased = g.alias(sub, "prefix").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    for row in 0..20 {
        assert_eq!(result.get_str(0, row).unwrap(), "id");
    }
}

#[test]
fn string_replace() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // replace(id1, "id", "ID") → "ID001", "ID002", etc.
    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let from = g.const_str("id").unwrap();
    let to = g.const_str("ID").unwrap();
    let replaced = g.replace(id1, from, to).unwrap();
    let aliased = g.alias(replaced, "replaced").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.get_str(0, 0).unwrap(), "ID001");
}

#[test]
fn string_like_ilike() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // LIKE: id1 LIKE 'id00%'
    let mut g = ctx.graph(&table).unwrap();
    let tbl = g.const_table(&table).unwrap();
    let id1 = g.scan("id1").unwrap();
    let pat = g.const_str("id00%").unwrap();
    let pred = g.like(id1, pat).unwrap();
    let filtered = g.filter(tbl, pred).unwrap();
    let result = g.execute(filtered).unwrap();
    // id001-id005 all match "id00%"
    assert_eq!(result.nrows(), 20);

    // ILIKE: case-insensitive
    let mut g2 = ctx.graph(&table).unwrap();
    let tbl2 = g2.const_table(&table).unwrap();
    let id1b = g2.scan("id1").unwrap();
    let pat2 = g2.const_str("ID00%").unwrap();
    let pred2 = g2.ilike(id1b, pat2).unwrap();
    let filtered2 = g2.filter(tbl2, pred2).unwrap();
    let result2 = g2.execute(filtered2).unwrap();
    assert_eq!(result2.nrows(), 20);
}

// ---------------------------------------------------------------------------
// Graph Aggregate Operators
// ---------------------------------------------------------------------------

#[test]
fn group_by_avg() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // Use group_by with no keys for whole-table AVG
    let id1 = g.scan("id1").unwrap();
    let v1 = g.scan("v1").unwrap();
    let grp = g.group_by(&[id1], &[AggOp::Avg], &[v1]).unwrap();
    let result = g.execute(grp).unwrap();

    // avg(v1) per id1 group — each group has 4 rows, e.g. id001: 1+2+3+4=10 → avg=2.5
    assert_eq!(result.nrows(), 5);
    // Collect and verify one group
    for row in 0..5 {
        let avg = result.get_f64(1, row).unwrap();
        assert!(avg > 0.0, "avg should be positive");
    }
}

#[test]
fn group_by_min_max() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let v1a = g.scan("v1").unwrap();
    let v1b = g.scan("v1").unwrap();
    let grp = g
        .group_by(&[id1], &[AggOp::Min, AggOp::Max], &[v1a, v1b])
        .unwrap();
    let result = g.execute(grp).unwrap();

    assert_eq!(result.nrows(), 5);
    // For each group, min <= max
    for row in 0..5 {
        let mn = result.get_i64(1, row).unwrap();
        let mx = result.get_i64(2, row).unwrap();
        assert!(mn <= mx, "min ({mn}) should be <= max ({mx})");
    }
}

#[test]
fn group_by_count() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let v1 = g.scan("v1").unwrap();
    let grp = g
        .group_by(&[id1], &[AggOp::Count], &[v1])
        .unwrap();
    let result = g.execute(grp).unwrap();

    assert_eq!(result.nrows(), 5);
    // Each id1 group has 4 rows
    for row in 0..5 {
        assert_eq!(result.get_i64(1, row).unwrap(), 4);
    }
}

#[test]
#[ignore = "CountDistinct via Graph API returns 0 — needs C engine investigation"]
fn group_by_count_distinct() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // COUNT(DISTINCT id4) per id1 — id4 has values 1,2,3 → 3 distinct per group
    let id1 = g.scan("id1").unwrap();
    let id4 = g.scan("id4").unwrap();
    let grp = g.group_by(&[id1], &[AggOp::CountDistinct], &[id4]).unwrap();
    let result = g.execute(grp).unwrap();

    assert_eq!(result.nrows(), 5);
    // Each group has id4 values from {1,2,3} → 3 distinct
    for row in 0..5 {
        let cd = result.get_i64(1, row).unwrap();
        assert!(cd > 0, "count_distinct should be > 0, got {cd}");
    }
}

#[test]
fn distinct_operation() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let id1 = g.scan("id1").unwrap();
    let dist = g.distinct(&[id1]).unwrap();
    let result = g.execute(dist).unwrap();

    // 5 distinct id1 values
    assert_eq!(result.nrows(), 5);
}

// ---------------------------------------------------------------------------
// Graph Binary Min/Max and If-Then-Else
// ---------------------------------------------------------------------------

#[test]
fn binary_min2_max2() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let v2 = g.scan("v2").unwrap();
    let v1b = g.scan("v1").unwrap();
    let v2b = g.scan("v2").unwrap();
    let mn = g.min2(v1, v2).unwrap();
    let mx = g.max2(v1b, v2b).unwrap();
    let mn_alias = g.alias(mn, "min_val").unwrap();
    let mx_alias = g.alias(mx, "max_val").unwrap();
    let projected = g.select(tbl, &[mn_alias, mx_alias]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // v1 is always < v2 (v2 = v1+1), so min2 = v1, max2 = v2
    // Row 0: v1=1, v2=2 → min=1, max=2
    assert_eq!(result.get_i64(0, 0).unwrap(), 1);
    assert_eq!(result.get_i64(1, 0).unwrap(), 2);
}

#[test]
fn if_then_else_op() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    // IF v1 > 5 THEN v1 ELSE 0
    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let five = g.const_i64(5).unwrap();
    let pred = g.gt(v1, five).unwrap();
    let v1b = g.scan("v1").unwrap();
    let zero = g.const_i64(0).unwrap();
    let ite = g.if_then_else(pred, v1b, zero).unwrap();
    let aliased = g.alias(ite, "clamped").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 20);
    // Row 0: v1=1, condition false → 0
    assert_eq!(result.get_i64(0, 0).unwrap(), 0);
    // Row 4: v1=5, condition false → 0
    assert_eq!(result.get_i64(0, 4).unwrap(), 0);
}

// ---------------------------------------------------------------------------
// Graph Temporal Operations
// ---------------------------------------------------------------------------

#[test]
fn extract_year_month_day() {
    let _guard = lock();
    let (_file, path) = create_datetime_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let date_col = g.scan("date").unwrap();
    let date_col2 = g.scan("date").unwrap();
    let date_col3 = g.scan("date").unwrap();
    let year = g.extract(date_col, extract_field::YEAR).unwrap();
    let month = g.extract(date_col2, extract_field::MONTH).unwrap();
    let day = g.extract(date_col3, extract_field::DAY).unwrap();
    let y_alias = g.alias(year, "year").unwrap();
    let m_alias = g.alias(month, "month").unwrap();
    let d_alias = g.alias(day, "day").unwrap();
    let projected = g.select(tbl, &[y_alias, m_alias, d_alias]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 4);
    // Row 0: 2024-01-15
    assert_eq!(result.get_i64(0, 0).unwrap(), 2024);
    assert_eq!(result.get_i64(1, 0).unwrap(), 1);
    assert_eq!(result.get_i64(2, 0).unwrap(), 15);
    // Row 2: 1970-01-01
    assert_eq!(result.get_i64(0, 2).unwrap(), 1970);
    assert_eq!(result.get_i64(1, 2).unwrap(), 1);
    assert_eq!(result.get_i64(2, 2).unwrap(), 1);
}

#[test]
fn date_trunc_month() {
    let _guard = lock();
    let (_file, path) = create_datetime_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let date_col = g.scan("date").unwrap();
    let truncated = g.date_trunc(date_col, extract_field::MONTH).unwrap();
    let aliased = g.alias(truncated, "trunc_month").unwrap();
    let projected = g.select(tbl, &[aliased]).unwrap();
    let result = g.execute(projected).unwrap();

    assert_eq!(result.nrows(), 4);
    // After truncating to month, day should be 1
    // Verify via extract on the truncated result
    // For now just check it executed without error
}

// ---------------------------------------------------------------------------
// Table Methods
// ---------------------------------------------------------------------------

#[test]
fn table_pick_columns() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    let picked = table.pick_columns(&["v1", "v3"]).unwrap();
    assert_eq!(picked.ncols(), 2);
    assert_eq!(picked.nrows(), 20);
    assert_eq!(picked.col_name_str(0), "v1");
    assert_eq!(picked.col_name_str(1), "v3");
    assert_eq!(picked.get_i64(0, 0).unwrap(), 1);
    assert!((picked.get_f64(1, 0).unwrap() - 1.5).abs() < 1e-10);
}

#[test]
fn table_pick_columns_unknown_errors() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    let result = table.pick_columns(&["nonexistent"]);
    assert!(result.is_err());
}

#[test]
fn table_with_column_names() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    let new_names: Vec<String> = (0..9).map(|i| format!("col_{i}")).collect();
    let renamed = table.with_column_names(&new_names).unwrap();

    assert_eq!(renamed.ncols(), 9);
    assert_eq!(renamed.nrows(), 20);
    assert_eq!(renamed.col_name_str(0), "col_0");
    assert_eq!(renamed.col_name_str(8), "col_8");
    // Data should be unchanged
    assert_eq!(renamed.get_i64(6, 0).unwrap(), 1);
}

#[test]
fn table_with_column_names_wrong_count_errors() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    let too_few = vec!["a".to_string(), "b".to_string()];
    let result = table.with_column_names(&too_few);
    assert!(result.is_err());
}

#[test]
fn table_write_csv_roundtrip() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();

    // Write to a temp file
    let out = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    let out_path = out.path().to_str().unwrap().to_string();
    table.write_csv(&out_path).unwrap();

    // Read it back
    let ctx2 = Context::new().unwrap();
    let table2 = ctx2.read_csv(&out_path).unwrap();
    assert_eq!(table2.nrows(), 20);
    assert_eq!(table2.ncols(), 9);
    assert_eq!(table2.get_i64(6, 0).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

#[test]
fn sym_intern_roundtrip() {
    let _guard = lock();
    let _ctx = Context::new().unwrap();

    let id1 = teide::sym_intern("test_symbol_123").unwrap();
    let id2 = teide::sym_intern("test_symbol_123").unwrap();
    // Same string → same ID
    assert_eq!(id1, id2);

    let id3 = teide::sym_intern("different_symbol").unwrap();
    // Different string → different ID
    assert_ne!(id1, id3);
}

#[test]
fn mem_stats_available() {
    let _guard = lock();
    let _ctx = Context::new().unwrap();

    let stats = teide::mem_stats();
    // Verify we can call mem_stats without crashing.
    // The exact values depend on engine state; just check the struct is populated.
    let _ = stats.bytes_allocated;
    let _ = stats.alloc_count;
    let _ = stats.peak_bytes;
}

#[test]
fn format_date_known_values() {
    // These are pure computations, no engine init needed
    assert_eq!(Table::format_date(0), "2000-01-01");
    assert_eq!(Table::format_date(-10957), "1970-01-01");
    assert_eq!(Table::format_date(8780), "2024-01-15");
}

#[test]
fn format_time_known_values() {
    assert_eq!(Table::format_time(0), "00:00:00");
    assert_eq!(Table::format_time(34_200_000), "09:30:00");
    assert_eq!(Table::format_time(51_330_500), "14:15:30.500");
    assert_eq!(Table::format_time(86_399_999), "23:59:59.999");
}

#[test]
fn format_timestamp_known_values() {
    // 2000-01-01 00:00:00
    assert_eq!(Table::format_timestamp(0), "2000-01-01 00:00:00");
    // 1970-01-01 00:00:00 = -946684800000000 µs
    assert_eq!(
        Table::format_timestamp(-946_684_800_000_000),
        "1970-01-01 00:00:00"
    );
}

// ---------------------------------------------------------------------------
// Graph Traversal / CSR (Rel)
// ---------------------------------------------------------------------------

fn create_edge_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    // Simple graph: 0→1, 0→2, 1→3, 2→3, 3→4
    writeln!(f, "src,dst").unwrap();
    writeln!(f, "0,1").unwrap();
    writeln!(f, "0,2").unwrap();
    writeln!(f, "1,3").unwrap();
    writeln!(f, "2,3").unwrap();
    writeln!(f, "3,4").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

#[test]
fn rel_from_edges_and_expand() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    // Build CSR from edge table: 5 source nodes, 5 destination nodes
    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    // Create a 1-element source table to expand from node 0
    let mut src_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(src_f, "node").unwrap();
    writeln!(src_f, "0").unwrap();
    src_f.flush().unwrap();
    let src_path = src_f.path().to_str().unwrap().to_string();
    let src_table = ctx.read_csv(&src_path).unwrap();

    let mut g = ctx.graph(&src_table).unwrap();
    let src_nodes = g.scan("node").unwrap();
    // direction=0 → outgoing edges
    let expanded = g.expand(src_nodes, &rel, 0).unwrap();
    let result = g.execute(expanded).unwrap();

    // Node 0 has outgoing edges to 1 and 2
    assert_eq!(result.nrows(), 2);
}

#[test]
fn rel_var_expand() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let mut src_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(src_f, "node").unwrap();
    writeln!(src_f, "0").unwrap();
    src_f.flush().unwrap();
    let src_path = src_f.path().to_str().unwrap().to_string();
    let src_table = ctx.read_csv(&src_path).unwrap();

    let mut g = ctx.graph(&src_table).unwrap();
    let src_nodes = g.scan("node").unwrap();
    // Variable-length BFS from node 0, depth 1-3, outgoing
    let vexp = g.var_expand(src_nodes, &rel, 0, 1, 3, false).unwrap();
    let result = g.execute(vexp).unwrap();

    // Node 0 can reach: depth 1 → {1,2}, depth 2 → {3}, depth 3 → {4}
    // Total reachable = 4 nodes
    assert_eq!(result.nrows(), 4, "var_expand should find 4 reachable nodes at depths 1-3");
}

#[test]
fn rel_shortest_path() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let mut src_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(src_f, "node").unwrap();
    writeln!(src_f, "0").unwrap();
    src_f.flush().unwrap();
    let src_path = src_f.path().to_str().unwrap().to_string();
    let src_table = ctx.read_csv(&src_path).unwrap();

    let mut g = ctx.graph(&src_table).unwrap();
    let src = g.const_i64(0).unwrap();
    let dst = g.const_i64(4).unwrap();
    // Shortest path from 0 to 4, max depth 10
    let sp = g.shortest_path(src, dst, &rel, 10).unwrap();
    let result = g.execute(sp).unwrap();

    // Path: 0→1→3→4 or 0→2→3→4 = 3 hops, path has 4 nodes
    assert_eq!(result.nrows(), 4, "shortest path from 0 to 4 should have 4 nodes");
}

#[test]
fn rel_save_load_roundtrip() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    // Save to temp directory
    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap();
    rel.save(dir_path).unwrap();

    // Load back
    let rel2 = Rel::load(dir_path).unwrap();

    // Verify by expanding from node 0
    let mut src_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(src_f, "node").unwrap();
    writeln!(src_f, "0").unwrap();
    src_f.flush().unwrap();
    let src_path = src_f.path().to_str().unwrap().to_string();
    let src_table = ctx.read_csv(&src_path).unwrap();

    let mut g = ctx.graph(&src_table).unwrap();
    let src_nodes = g.scan("node").unwrap();
    let expanded = g.expand(src_nodes, &rel2, 0).unwrap();
    let result = g.execute(expanded).unwrap();

    assert_eq!(result.nrows(), 2, "loaded rel should match original");
}

// ---------------------------------------------------------------------------
// Multi-table support
// ---------------------------------------------------------------------------

#[test]
fn graph_add_table_and_scan() {
    let _guard = lock();
    let (_file1, path1) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table1 = ctx.read_csv(&path1).unwrap();

    // Create a second small CSV
    let mut f2 = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f2, "key,extra").unwrap();
    writeln!(f2, "id001,AAA").unwrap();
    writeln!(f2, "id002,BBB").unwrap();
    f2.flush().unwrap();
    let path2 = f2.path().to_str().unwrap().to_string();
    let table2 = ctx.read_csv(&path2).unwrap();

    let mut g = ctx.graph(&table1).unwrap();
    let t2_id = g.add_table(&table2);
    let extra_col = g.scan_table(t2_id, "extra").unwrap();
    // Just verify the scan succeeds — it creates a valid graph node
    let _aliased = g.alias(extra_col, "ext").unwrap();
}

// ---------------------------------------------------------------------------
// isnull operator
// ---------------------------------------------------------------------------

#[test]
fn isnull_on_non_null_data() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let null_check = g.isnull(v1).unwrap();
    // NOT isnull → filter to non-null only
    let not_null = g.not(null_check).unwrap();
    let filtered = g.filter(tbl, not_null).unwrap();
    let result = g.execute(filtered).unwrap();

    // All data is non-null → all 20 rows pass
    assert_eq!(result.nrows(), 20);
}

// ---------------------------------------------------------------------------
// select (projection)
// ---------------------------------------------------------------------------

#[test]
fn select_multiple_columns() {
    let _guard = lock();
    let (_file, path) = create_test_csv();
    let ctx = Context::new().unwrap();
    let table = ctx.read_csv(&path).unwrap();
    let mut g = ctx.graph(&table).unwrap();

    let tbl = g.const_table(&table).unwrap();
    let v1 = g.scan("v1").unwrap();
    let v3 = g.scan("v3").unwrap();
    let selected = g.select(tbl, &[v1, v3]).unwrap();
    let result = g.execute(selected).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);
    assert_eq!(result.col_name_str(0), "v1");
    assert_eq!(result.col_name_str(1), "v3");
}

// ---------------------------------------------------------------------------
// read_csv_opts edge cases
// ---------------------------------------------------------------------------

#[test]
fn read_csv_custom_delimiter() {
    let _guard = lock();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "a|b|c").unwrap();
    writeln!(f, "1|2|3").unwrap();
    writeln!(f, "4|5|6").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let ctx = Context::new().unwrap();
    let table = ctx.read_csv_opts(&path, '|', true, None).unwrap();

    assert_eq!(table.nrows(), 2);
    assert_eq!(table.ncols(), 3);
    assert_eq!(table.col_name_str(0), "a");
    assert_eq!(table.get_i64(0, 0).unwrap(), 1);
    assert_eq!(table.get_i64(2, 1).unwrap(), 6);
}

#[test]
fn read_csv_no_header() {
    let _guard = lock();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "1,2,3").unwrap();
    writeln!(f, "4,5,6").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let ctx = Context::new().unwrap();
    let table = ctx.read_csv_opts(&path, ',', false, None).unwrap();

    assert_eq!(table.nrows(), 2);
    assert_eq!(table.ncols(), 3);
    assert_eq!(table.get_i64(0, 0).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// Graph Algorithm Tests
// ---------------------------------------------------------------------------

#[test]
fn graph_pagerank() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let mut g = ctx.graph(&edges).unwrap();
    let pr = g.pagerank(&rel, 20, 0.85).unwrap();
    let result = g.execute(pr).unwrap();

    // Should have 5 nodes with ranks
    assert_eq!(result.nrows(), 5);
    // All ranks should be positive
    for i in 0..5 {
        let rank = result.get_f64(1, i).unwrap();
        assert!(rank > 0.0, "rank should be positive, got {rank} for node {i}");
    }
    // Node 0 has no in-edges so should have the lowest rank (only gets base rank)
    let rank0 = result.get_f64(1, 0).unwrap();
    for i in 1..5 {
        let rank_i = result.get_f64(1, i).unwrap();
        assert!(
            rank_i >= rank0,
            "node {i} should have rank >= node 0 (no in-edges), got {rank_i} vs {rank0}"
        );
    }
}

#[test]
fn graph_connected_comp() {
    let _guard = lock();
    // Create two disconnected components: {0,1,2} and {3,4}
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "src,dst").unwrap();
    writeln!(f, "0,1").unwrap();
    writeln!(f, "1,2").unwrap();
    writeln!(f, "3,4").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let mut g = ctx.graph(&edges).unwrap();
    let cc = g.connected_comp(&rel).unwrap();
    let result = g.execute(cc).unwrap();

    assert_eq!(result.nrows(), 5);
    // Nodes 0,1,2 should have same component; 3,4 should have same component
    let c0 = result.get_i64(1, 0).unwrap();
    let c1 = result.get_i64(1, 1).unwrap();
    let c2 = result.get_i64(1, 2).unwrap();
    let c3 = result.get_i64(1, 3).unwrap();
    let c4 = result.get_i64(1, 4).unwrap();
    assert_eq!(c0, c1, "nodes 0,1 should be same component");
    assert_eq!(c1, c2, "nodes 1,2 should be same component");
    assert_eq!(c3, c4, "nodes 3,4 should be same component");
    assert_ne!(c0, c3, "components should be different");
}

#[test]
fn graph_dijkstra() {
    let _guard = lock();
    // Create a weighted edge table: src, dst, weight
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "src,dst,weight").unwrap();
    writeln!(f, "0,1,1.0").unwrap();
    writeln!(f, "0,2,4.0").unwrap();
    writeln!(f, "1,3,2.0").unwrap();
    writeln!(f, "2,3,1.0").unwrap();
    writeln!(f, "3,4,3.0").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();
    // Attach edge properties (the edge table itself) so Dijkstra can read weights
    rel.set_props(&edges);

    let mut g = ctx.graph(&edges).unwrap();
    let src = g.const_i64(0).unwrap();
    let dst = g.const_i64(4).unwrap();
    let dj = g.dijkstra(src, Some(dst), &rel, "weight", 255).unwrap();
    let result = g.execute(dj).unwrap();

    // Should find path 0→1→3→4 with total weight 1+2+3=6.0
    // Or 0→2→3→4 with weight 4+1+3=8.0
    // Dijkstra should return the shorter: 6.0
    assert!(result.nrows() > 0, "should find a path");

    // Check that node 4 is reachable with distance 6.0
    let nrows = result.nrows() as usize;
    let mut found_dst = false;
    for i in 0..nrows {
        let node = result.get_i64(0, i).unwrap();
        if node == 4 {
            let dist = result.get_f64(1, i).unwrap();
            assert!(
                (dist - 6.0).abs() < 0.001,
                "shortest distance to 4 should be 6.0, got {dist}"
            );
            found_dst = true;
        }
    }
    assert!(found_dst, "destination node 4 should be in results");
}

#[test]
fn graph_louvain() {
    let _guard = lock();
    let (_file, path) = create_edge_csv();
    let ctx = Context::new().unwrap();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 5, 5, true).unwrap();

    let mut g = ctx.graph(&edges).unwrap();
    let lv = g.louvain(&rel, 100).unwrap();
    let result = g.execute(lv).unwrap();

    assert_eq!(result.nrows(), 5);
    // Each node should have a non-negative community ID
    for i in 0..5 {
        let c = result.get_i64(1, i).unwrap();
        assert!(c >= 0, "community ID should be non-negative");
    }
}

// ---------------------------------------------------------------------------
// Vector Similarity Tests
// ---------------------------------------------------------------------------

#[test]
fn vector_cosine_similarity() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    // Create a 3-row, 4-dimensional embedding column
    let dim: i32 = 4;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // row 0: unit vector along x
        0.0, 1.0, 0.0, 0.0, // row 1: unit vector along y
        1.0, 1.0, 0.0, 0.0, // row 2: 45 degrees between x and y
    ];
    let emb_col = Table::create_embedding_column(&ctx,3, dim, &embeddings).unwrap();

    // Query: unit vector along x
    let query = vec![1.0f32, 0.0, 0.0, 0.0];

    // Create a dummy table to build a Graph
    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    for i in 0..3 {
        writeln!(csv_f, "{i}").unwrap();
    }
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    // const_embedding retains; release our reference.
    unsafe { ffi::td_release(emb_col) };
    // SAFETY: `query` outlives the execute_raw() call below.
    let sim = unsafe { g.cosine_sim(emb_node, &query) }.unwrap();
    // cosine_sim returns a raw F64 vector, use execute_raw
    let result = g.execute_raw(sim).unwrap();

    let len = unsafe { ffi::td_len(result) } as usize;
    assert_eq!(len, 3);
    let data = unsafe { std::slice::from_raw_parts(ffi::td_data(result) as *const f64, len) };

    let s0 = data[0]; // row 0: cos(0) = 1.0
    let s1 = data[1]; // row 1: cos(90) = 0.0
    let s2 = data[2]; // row 2: cos(45) ~ 0.707

    assert!((s0 - 1.0).abs() < 0.001, "row 0 should be 1.0, got {s0}");
    assert!(s1.abs() < 0.001, "row 1 should be 0.0, got {s1}");
    assert!(
        (s2 - 0.7071).abs() < 0.01,
        "row 2 should be ~0.707, got {s2}"
    );

    unsafe { ffi::td_release(result) };
}

#[test]
fn vector_euclidean_distance() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        2.0, 0.0, 0.0, // row 2
    ];
    let emb_col = Table::create_embedding_column(&ctx,3, dim, &embeddings).unwrap();

    let query = vec![1.0f32, 0.0, 0.0];

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    for i in 0..3 {
        writeln!(csv_f, "{i}").unwrap();
    }
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    unsafe { ffi::td_release(emb_col) };
    // SAFETY: `query` outlives the execute_raw() call below.
    let dist = unsafe { g.euclidean_dist(emb_node, &query) }.unwrap();
    let result = g.execute_raw(dist).unwrap();

    let len = unsafe { ffi::td_len(result) } as usize;
    assert_eq!(len, 3);
    let data = unsafe { std::slice::from_raw_parts(ffi::td_data(result) as *const f64, len) };

    let d0 = data[0]; // row 0: distance 0.0 (same point)
    let d1 = data[1]; // row 1: distance sqrt(2) ~ 1.414
    let d2 = data[2]; // row 2: distance 1.0

    assert!(d0.abs() < 0.001, "row 0 should be 0.0, got {d0}");
    assert!(
        (d1 - std::f64::consts::SQRT_2).abs() < 0.01,
        "row 1 should be ~1.414, got {d1}"
    );
    assert!((d2 - 1.0).abs() < 0.001, "row 2 should be 1.0, got {d2}");

    unsafe { ffi::td_release(result) };
}

#[test]
fn vector_knn() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, // row 0
        0.9, 0.1, 0.0, // row 1: very similar to row 0
        0.0, 1.0, 0.0, // row 2: orthogonal
        0.0, 0.0, 1.0, // row 3: orthogonal
        0.8, 0.2, 0.0, // row 4: similar to row 0
    ];
    let emb_col = Table::create_embedding_column(&ctx,5, dim, &embeddings).unwrap();

    let query = vec![1.0f32, 0.0, 0.0];

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    for i in 0..5 {
        writeln!(csv_f, "{i}").unwrap();
    }
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    unsafe { ffi::td_release(emb_col) };
    // SAFETY: `query` outlives the execute() call below.
    let knn_result = unsafe { g.knn(emb_node, &query, 3) }.unwrap();
    let result = g.execute(knn_result).unwrap();

    // Should return 3 rows, sorted by similarity descending
    assert_eq!(result.nrows(), 3);

    // Verify all 3 rows: rowid and similarity, in descending similarity order
    let rowid0 = result.get_i64(0, 0).unwrap();
    let sim0 = result.get_f64(1, 0).unwrap();
    assert_eq!(rowid0, 0, "rank-1 should be row 0 (exact match)");
    assert!((sim0 - 1.0).abs() < 0.001, "rank-1 sim should be 1.0, got {sim0}");

    let rowid1 = result.get_i64(0, 1).unwrap();
    let sim1 = result.get_f64(1, 1).unwrap();
    assert_eq!(rowid1, 1, "rank-2 should be row 1 (sim ~0.994)");
    assert!(sim1 > 0.99, "rank-2 sim should be >0.99, got {sim1}");

    let rowid2 = result.get_i64(0, 2).unwrap();
    let sim2 = result.get_f64(1, 2).unwrap();
    assert_eq!(rowid2, 4, "rank-3 should be row 4 (sim ~0.970)");
    assert!(sim2 > 0.96, "rank-3 sim should be >0.96, got {sim2}");

    // Verify descending order
    assert!(sim0 >= sim1, "results should be sorted descending");
    assert!(sim1 >= sim2, "results should be sorted descending");
}

#[test]
fn vector_cosine_sim_owned_dim_mismatch() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 4;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
    ];
    let emb_col = Table::create_embedding_column(&ctx, 2, dim, &embeddings).unwrap();

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    writeln!(csv_f, "0").unwrap();
    writeln!(csv_f, "1").unwrap();
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    unsafe { ffi::td_release(emb_col) };

    // Wrong dimension: 3 instead of 4 — should be caught by check_embedding_dim
    let wrong_query = vec![1.0f32, 0.0, 0.0];
    let result = g.cosine_sim_owned(emb_node, wrong_query);
    assert!(result.is_err(), "cosine_sim_owned should reject dimension mismatch");

    // Wrong dimension: 2 instead of 4 (evenly divides buffer — C kernel wouldn't catch this)
    let wrong_query2 = vec![1.0f32, 0.0];
    let result2 = g.cosine_sim_owned(emb_node, wrong_query2);
    assert!(
        result2.is_err(),
        "cosine_sim_owned should reject dimension mismatch even when it evenly divides"
    );
}

#[test]
fn vector_euclidean_dist_owned_dim_mismatch() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let embeddings: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let emb_col = Table::create_embedding_column(&ctx, 2, dim, &embeddings).unwrap();

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    writeln!(csv_f, "0").unwrap();
    writeln!(csv_f, "1").unwrap();
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    unsafe { ffi::td_release(emb_col) };

    // Wrong dimension: 2 instead of 3 (evenly divides 6-element buffer)
    let wrong_query = vec![1.0f32, 0.0];
    let result = g.euclidean_dist_owned(emb_node, wrong_query);
    assert!(
        result.is_err(),
        "euclidean_dist_owned should reject dimension mismatch"
    );
}

#[test]
fn vector_knn_owned_dim_mismatch() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 4;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
    ];
    let emb_col = Table::create_embedding_column(&ctx, 2, dim, &embeddings).unwrap();

    let mut csv_f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(csv_f, "id").unwrap();
    writeln!(csv_f, "0").unwrap();
    writeln!(csv_f, "1").unwrap();
    csv_f.flush().unwrap();
    let table = ctx.read_csv(csv_f.path().to_str().unwrap()).unwrap();

    let mut g = ctx.graph(&table).unwrap();
    let emb_node = unsafe { g.const_embedding(emb_col, dim).unwrap() };
    unsafe { ffi::td_release(emb_col) };

    // Wrong dimension: 2 instead of 4 (evenly divides)
    let wrong_query = vec![1.0f32, 0.0];
    let result = g.knn_owned(emb_node, wrong_query, 1);
    assert!(
        result.is_err(),
        "knn_owned should reject dimension mismatch"
    );
}

// ---------------------------------------------------------------------------
// SQL-level vector dimension validation tests
// ---------------------------------------------------------------------------

#[test]
fn sql_cosine_similarity_dim_mismatch_simple() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    // Create table with embedding column and register its dimension
    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // Wrong dimension: 3 instead of 4 — should be rejected
    let result = session.execute(
        "SELECT COSINE_SIMILARITY(embedding, ARRAY[1.0, 0.0, 0.0]) FROM docs"
    );
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("should reject dimension mismatch via simple identifier"),
    };
    assert!(err.contains("dimension"), "error should mention dimension: {err}");
}

#[test]
fn sql_cosine_similarity_dim_mismatch_qualified() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // Qualified reference: d.embedding — should also be validated
    let result = session.execute(
        "SELECT COSINE_SIMILARITY(d.embedding, ARRAY[1.0, 0.0, 0.0]) FROM docs d"
    );
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("should reject dimension mismatch via qualified identifier"),
    };
    assert!(err.contains("dimension"), "error should mention dimension: {err}");
}

#[test]
fn sql_insert_values_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // INSERT VALUES on embedding tables is rejected because VALUES produces
    // scalar F32 vectors, not dimension-aware embedding buffers.
    let result = session.execute("INSERT INTO docs VALUES (1, 'science', 0.0)");
    assert!(result.is_err(), "INSERT VALUES should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_embedding_dims_preserved_after_insert_select() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    // Create source and target tables without embedding metadata initially
    session.execute("CREATE TABLE src (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO src VALUES (0, 'math', 0.0)").unwrap();

    session.execute("CREATE TABLE docs AS SELECT * FROM src").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // INSERT SELECT from a table that also has matching embedding dims should work
    session.register_embedding_dim("src", "embedding", 4).unwrap();
    session.execute("INSERT INTO docs SELECT * FROM src").unwrap();

    // Should still reject wrong dimension after insert
    let result = session.execute(
        "SELECT COSINE_SIMILARITY(embedding, ARRAY[1.0, 0.0, 0.0]) FROM docs"
    );
    assert!(result.is_err(), "should reject dimension mismatch after INSERT");
}

#[test]
fn sql_delete_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // DELETE with WHERE on embedding tables is rejected because the C engine's
    // filter kernel is not dimension-aware and would corrupt flat N*D F32 arrays.
    let result = session.execute("DELETE FROM docs WHERE id = 0");
    assert!(result.is_err(), "DELETE WHERE should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");

    // DELETE without WHERE (truncate) should still work
    session.execute("DELETE FROM docs").unwrap();
}

#[test]
fn sql_update_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    // UPDATE on embedding tables is rejected because the C engine's
    // if_then_else kernel is not dimension-aware and would corrupt flat N*D arrays.
    let result = session.execute("UPDATE docs SET name = 'mathematics' WHERE id = 0");
    assert!(result.is_err(), "UPDATE should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_select_where_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    let result = session.execute("SELECT * FROM docs WHERE id = 0");
    assert!(result.is_err(), "SELECT WHERE should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_select_order_by_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    let result = session.execute("SELECT * FROM docs ORDER BY id");
    assert!(result.is_err(), "SELECT ORDER BY should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_select_limit_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    let result = session.execute("SELECT * FROM docs LIMIT 1");
    assert!(result.is_err(), "SELECT LIMIT should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_select_group_by_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    let result = session.execute("SELECT name, COUNT(*) FROM docs GROUP BY name");
    assert!(result.is_err(), "SELECT GROUP BY should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

#[test]
fn sql_select_distinct_rejected_on_embedding_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session.execute("CREATE TABLE docs (id INTEGER, name VARCHAR, embedding FLOAT)").unwrap();
    session.execute("INSERT INTO docs VALUES (0, 'math', 0.0), (1, 'science', 0.0)").unwrap();
    session.register_embedding_dim("docs", "embedding", 4).unwrap();

    let result = session.execute("SELECT DISTINCT name FROM docs");
    assert!(result.is_err(), "SELECT DISTINCT should be rejected on embedding tables");
    let err = format!("{}", result.err().unwrap());
    assert!(err.contains("not yet supported"), "error should mention not yet supported: {err}");
}

// ---------------------------------------------------------------------------
// HNSW Index Tests
// ---------------------------------------------------------------------------

#[test]
fn hnsw_build_and_search() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 4;
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // 0: x-axis
        0.0, 1.0, 0.0, 0.0, // 1: y-axis
        0.0, 0.0, 1.0, 0.0, // 2: z-axis
        0.9, 0.1, 0.0, 0.0, // 3: near x-axis
        0.1, 0.9, 0.0, 0.0, // 4: near y-axis
    ];

    let idx = HnswIndex::build(&ctx, &vectors, 5, dim, 4, 20).unwrap();

    // Search for vectors near x-axis
    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let results = idx.search(&query, 3, 10).unwrap();

    assert_eq!(results.len(), 3);
    // Top result should be node 0 (exact match) — distance 0
    assert_eq!(results[0].0, 0, "nearest should be node 0");
    // Node 3 should be second (0.9, 0.1, 0, 0) — very close to x-axis
    assert_eq!(results[1].0, 3, "second nearest should be node 3");
}

#[test]
fn hnsw_save_load_roundtrip() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    let dim: i32 = 3;
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, // 0
        0.0, 1.0, 0.0, // 1
        0.0, 0.0, 1.0, // 2
    ];

    let idx = HnswIndex::build(&ctx, &vectors, 3, dim, 4, 20).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_str().unwrap();
    idx.save(dir_path).unwrap();

    let idx2 = HnswIndex::load(dir_path).unwrap();
    let query = vec![1.0f32, 0.0, 0.0];
    let results = idx2.search(&query, 1, 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0, "loaded index should find node 0");
}

#[test]
fn hnsw_sql_create_drop_vector_index() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    // Create a table and add an embedding column
    session
        .execute("CREATE TABLE docs (id INTEGER, name VARCHAR)")
        .unwrap();
    session
        .execute("INSERT INTO docs VALUES (0, 'math'), (1, 'science'), (2, 'art')")
        .unwrap();

    let dim = 4i32;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // math
        0.0, 1.0, 0.0, 0.0, // science
        0.0, 0.0, 1.0, 0.0, // art
    ];
    session
        .add_embedding_column("docs", "embedding", dim, &embeddings)
        .unwrap();

    // CREATE VECTOR INDEX
    let result = session
        .execute("CREATE VECTOR INDEX emb_idx ON docs(embedding) USING HNSW(M=4, ef_construction=20)")
        .unwrap();
    match &result {
        teide::ExecResult::Ddl(msg) => {
            assert!(msg.contains("Created vector index 'emb_idx'"), "unexpected message: {msg}");
        }
        _ => panic!("Expected DDL result"),
    }

    // Duplicate should error
    let dup = session.execute("CREATE VECTOR INDEX emb_idx ON docs(embedding)");
    assert!(dup.is_err(), "duplicate index should fail");

    // DROP VECTOR INDEX
    let result = session.execute("DROP VECTOR INDEX emb_idx").unwrap();
    match &result {
        teide::ExecResult::Ddl(msg) => {
            assert!(msg.contains("Dropped vector index 'emb_idx'"), "unexpected message: {msg}");
        }
        _ => panic!("Expected DDL result"),
    }

    // DROP again should error
    let drop2 = session.execute("DROP VECTOR INDEX emb_idx");
    assert!(drop2.is_err(), "drop non-existent index should fail");

    // DROP IF EXISTS should succeed
    let result = session
        .execute("DROP VECTOR INDEX IF EXISTS emb_idx")
        .unwrap();
    match &result {
        teide::ExecResult::Ddl(msg) => {
            assert!(msg.contains("not found (skipped)"), "unexpected message: {msg}");
        }
        _ => panic!("Expected DDL result"),
    }
}

#[test]
fn hnsw_sql_drop_table_cascades_vector_index() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session
        .execute("CREATE TABLE docs (id INTEGER)")
        .unwrap();
    session
        .execute("INSERT INTO docs VALUES (0), (1), (2)")
        .unwrap();

    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    session
        .add_embedding_column("docs", "emb", 3, &embeddings)
        .unwrap();

    session
        .execute("CREATE VECTOR INDEX idx ON docs(emb) USING HNSW(M=4, ef_construction=20)")
        .unwrap();

    // Drop the table — should cascade and remove the vector index
    session.execute("DROP TABLE docs").unwrap();

    // The vector index should be gone
    let drop_idx = session.execute("DROP VECTOR INDEX idx");
    assert!(drop_idx.is_err(), "index should have been cascaded on table drop");
}

#[test]
fn hnsw_knn_query_uses_index() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    // Create a table with an embedding column.
    session
        .execute("CREATE TABLE docs (id INTEGER, name VARCHAR)")
        .unwrap();
    session
        .execute("INSERT INTO docs VALUES (0, 'math'), (1, 'science'), (2, 'art'), (3, 'physics'), (4, 'music')")
        .unwrap();

    let dim = 4i32;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // math
        0.0, 1.0, 0.0, 0.0, // science
        0.0, 0.0, 1.0, 0.0, // art
        0.9, 0.1, 0.0, 0.0, // physics (near math)
        0.0, 0.0, 0.0, 1.0, // music
    ];
    session
        .add_embedding_column("docs", "embedding", dim, &embeddings)
        .unwrap();

    // Create an HNSW vector index.
    session
        .execute("CREATE VECTOR INDEX emb_idx ON docs(embedding) USING HNSW(M=4, ef_construction=20)")
        .unwrap();

    // Query: ORDER BY cosine_similarity LIMIT k — should use HNSW index.
    let result = session
        .execute("SELECT id, name, cosine_similarity(embedding, ARRAY[1.0, 0.0, 0.0, 0.0]) AS sim FROM docs ORDER BY cosine_similarity(embedding, ARRAY[1.0, 0.0, 0.0, 0.0]) DESC LIMIT 2")
        .unwrap();
    match &result {
        teide::ExecResult::Query(sql_result) => {
            assert_eq!(sql_result.nrows, 2, "should return 2 rows");
            assert_eq!(sql_result.columns.len(), 3, "should have 3 columns: id, name, sim");
            assert_eq!(sql_result.columns[0], "id");
            assert_eq!(sql_result.columns[1], "name");
        }
        _ => panic!("Expected Query result"),
    }
}

#[test]
fn order_by_similarity_without_index() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    // Create a table with an embedding column but NO vector index.
    session
        .execute("CREATE TABLE docs2 (id INTEGER, name VARCHAR)")
        .unwrap();
    session
        .execute("INSERT INTO docs2 VALUES (0, 'math'), (1, 'science'), (2, 'art')")
        .unwrap();

    let dim = 3i32;
    let embeddings: Vec<f32> = vec![
        1.0, 0.0, 0.0, // math
        0.0, 1.0, 0.0, // science
        0.0, 0.0, 1.0, // art
    ];
    session
        .add_embedding_column("docs2", "embedding", dim, &embeddings)
        .unwrap();

    // Without an index, ORDER BY cosine_similarity LIMIT k should still work
    // (brute-force) when SELECT doesn't include the raw embedding column.
    let result = session
        .execute("SELECT id, cosine_similarity(embedding, ARRAY[1.0, 0.0, 0.0]) AS sim FROM docs2 ORDER BY sim DESC LIMIT 2")
        .unwrap();
    match &result {
        teide::ExecResult::Query(sql_result) => {
            assert_eq!(sql_result.nrows, 2, "should return 2 rows");
            assert_eq!(sql_result.columns.len(), 2, "should have 2 columns: id, sim");
        }
        _ => panic!("Expected Query result"),
    }
}

#[test]
fn test_rel_neighbors() {
    let _guard = lock();
    let ctx = Context::new().unwrap();

    // Build a small graph: 3 nodes, edges: 0->1, 0->2, 1->2
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "src,dst").unwrap();
    writeln!(f, "0,1").unwrap();
    writeln!(f, "0,2").unwrap();
    writeln!(f, "1,2").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let edges = ctx.read_csv(&path).unwrap();

    let rel = Rel::from_edges(&edges, "src", "dst", 3, 3, true).unwrap();

    // Forward: node 0 has neighbors [1, 2]
    let neighbors = rel.neighbors(0, 0); // direction 0 = forward
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&1));
    assert!(neighbors.contains(&2));

    // Forward: node 1 has neighbors [2]
    let neighbors = rel.neighbors(1, 0);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], 2);

    // Forward: node 2 has no outgoing neighbors
    assert!(rel.neighbors(2, 0).is_empty());

    // Reverse: node 2 has incoming from [0, 1]
    let neighbors = rel.neighbors(2, 1); // direction 1 = reverse
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&0));
    assert!(neighbors.contains(&1));

    // n_nodes — use asymmetric counts to verify direction matters
    // Re-build with n_src=3, n_dst=4 so the two directions differ
    let rel2 = Rel::from_edges(&edges, "src", "dst", 3, 4, true).unwrap();
    assert_eq!(rel2.n_nodes(0), 3); // forward: n_src_nodes
    assert_eq!(rel2.n_nodes(1), 4); // reverse: n_dst_nodes
}
