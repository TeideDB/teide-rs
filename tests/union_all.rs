//! Synthetic tests for Table::union_all and Graph::union_all.
//!
//! Each test uses temp CSV files to build two same-schema tables,
//! exercises the union, and verifies row counts + column values.

use std::io::Write as _;
use std::sync::Mutex;
use teide::Context;

// The C engine uses global state — serialize all tests.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_csv(rows: &[&str]) -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    for row in rows {
        writeln!(f, "{row}").unwrap();
    }
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

// ---------------------------------------------------------------------------
// Table::union_all — basic row concatenation
// ---------------------------------------------------------------------------

#[test]
fn union_all_row_count() {
    let _guard = lock();

    let (_f1, p1) = write_csv(&[
        "id,value",
        "1,100",
        "2,200",
        "3,300",
    ]);
    let (_f2, p2) = write_csv(&[
        "id,value",
        "4,400",
        "5,500",
    ]);

    let ctx = Context::new().unwrap();
    let t1 = ctx.read_csv(&p1).unwrap();
    let t2 = ctx.read_csv(&p2).unwrap();

    assert_eq!(t1.nrows(), 3);
    assert_eq!(t2.nrows(), 2);

    let result = t1.union_all(&t2).unwrap();
    assert_eq!(result.nrows(), 5, "union of 3+2 rows should yield 5");
}

#[test]
fn union_all_column_values() {
    let _guard = lock();

    let (_f1, p1) = write_csv(&[
        "id,value",
        "10,1",
        "20,2",
        "30,3",
    ]);
    let (_f2, p2) = write_csv(&[
        "id,value",
        "40,4",
        "50,5",
    ]);

    let ctx = Context::new().unwrap();
    let t1 = ctx.read_csv(&p1).unwrap();
    let t2 = ctx.read_csv(&p2).unwrap();

    let result = t1.union_all(&t2).unwrap();
    assert_eq!(result.nrows(), 5);

    // Column 0 = id: expect [10,20,30,40,50]
    let expected_ids: Vec<i64> = vec![10, 20, 30, 40, 50];
    for (row, &expected) in expected_ids.iter().enumerate() {
        let actual = result.get_i64(0, row).unwrap();
        assert_eq!(actual, expected, "id mismatch at row {row}");
    }

    // Column 1 = value: expect [1,2,3,4,5]
    let expected_vals: Vec<i64> = vec![1, 2, 3, 4, 5];
    for (row, &expected) in expected_vals.iter().enumerate() {
        let actual = result.get_i64(1, row).unwrap();
        assert_eq!(actual, expected, "value mismatch at row {row}");
    }
}

// ---------------------------------------------------------------------------
// Table::union_all — edge cases
// ---------------------------------------------------------------------------

#[test]
fn union_all_with_empty_left() {
    let _guard = lock();

    let (_f1, p1) = write_csv(&["id,value"]);   // header only, 0 rows
    let (_f2, p2) = write_csv(&["id,value", "7,70", "8,80"]);

    let ctx = Context::new().unwrap();
    let empty = ctx.read_csv(&p1).unwrap();
    let t2    = ctx.read_csv(&p2).unwrap();

    let result = empty.union_all(&t2).unwrap();
    assert_eq!(result.nrows(), 2, "empty UNION ALL t2 should equal t2");
}

#[test]
fn union_all_with_empty_right() {
    let _guard = lock();

    let (_f1, p1) = write_csv(&["id,value", "1,10", "2,20"]);
    let (_f2, p2) = write_csv(&["id,value"]);   // 0 rows

    let ctx = Context::new().unwrap();
    let t1    = ctx.read_csv(&p1).unwrap();
    let empty = ctx.read_csv(&p2).unwrap();

    let result = t1.union_all(&empty).unwrap();
    assert_eq!(result.nrows(), 2, "t1 UNION ALL empty should equal t1");
}

#[test]
fn union_all_no_deduplication() {
    let _guard = lock();

    // Same rows on both sides — UNION ALL keeps duplicates
    let (_f1, p1) = write_csv(&["id,value", "1,10", "2,20"]);
    let (_f2, p2) = write_csv(&["id,value", "1,10", "2,20"]);

    let ctx = Context::new().unwrap();
    let t1 = ctx.read_csv(&p1).unwrap();
    let t2 = ctx.read_csv(&p2).unwrap();

    let result = t1.union_all(&t2).unwrap();
    assert_eq!(result.nrows(), 4, "UNION ALL must not deduplicate");
}

// ---------------------------------------------------------------------------
// Table::union_all — larger schema (graph-like: src/dst edges)
// ---------------------------------------------------------------------------

#[test]
fn union_all_edge_tables() {
    let _guard = lock();

    // Simulates combining two edge CSVs for a Datalog fixpoint step
    let (_f1, p1) = write_csv(&[
        "src,dst",
        "0,1",
        "1,2",
        "2,3",
    ]);
    let (_f2, p2) = write_csv(&[
        "src,dst",
        "0,2",  // derived: 0→1→2
        "1,3",  // derived: 1→2→3
    ]);

    let ctx = Context::new().unwrap();
    let base    = ctx.read_csv(&p1).unwrap();
    let derived = ctx.read_csv(&p2).unwrap();

    let all_edges = base.union_all(&derived).unwrap();
    assert_eq!(all_edges.nrows(), 5, "3 base + 2 derived edges = 5");

    // Verify derived rows appear at the end (rows 3 and 4)
    assert_eq!(all_edges.get_i64(0, 3).unwrap(), 0); // src=0
    assert_eq!(all_edges.get_i64(1, 3).unwrap(), 2); // dst=2
    assert_eq!(all_edges.get_i64(0, 4).unwrap(), 1); // src=1
    assert_eq!(all_edges.get_i64(1, 4).unwrap(), 3); // dst=3
}

// ---------------------------------------------------------------------------
// Graph::union_all — low-level Column API
// ---------------------------------------------------------------------------

#[test]
fn graph_union_all_column_api() {
    let _guard = lock();

    let (_f1, p1) = write_csv(&["id,value", "1,10", "2,20", "3,30"]);
    let (_f2, p2) = write_csv(&["id,value", "4,40", "5,50"]);

    let ctx = Context::new().unwrap();
    let t1 = ctx.read_csv(&p1).unwrap();
    let t2 = ctx.read_csv(&p2).unwrap();

    // Use Graph-level union_all (Column API)
    let g = ctx.graph(&t1).unwrap();
    let left  = g.const_table(&t1).unwrap();
    let right = g.const_table(&t2).unwrap();
    let unioned = g.union_all(left, right).unwrap();
    let result = g.execute(unioned).unwrap();

    assert_eq!(result.nrows(), 5, "Graph::union_all 3+2 = 5 rows");
}
