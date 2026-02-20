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

//! SQL integration tests for teide-db.
//!
//! Small-data tests use a ~20-row inline CSV for deterministic assertions.
//! Benchmarks have been moved to `benches/` (criterion).

use std::io::Write;
use std::sync::Mutex;

use teide::sql::{ExecResult, Session, SqlResult};

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

/// Create a session with the test CSV registered as table "csv".
fn setup_session() -> (Session, tempfile::NamedTempFile) {
    let (file, path) = create_test_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE csv AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();
    (session, file)
}

/// Extract SqlResult from ExecResult::Query, panicking otherwise.
fn unwrap_query(result: ExecResult) -> SqlResult {
    match result {
        ExecResult::Query(r) => r,
        ExecResult::Ddl(msg) => panic!("expected Query, got Ddl: {msg}"),
    }
}

// ---------------------------------------------------------------------------
// Small-data SQL correctness tests
// ---------------------------------------------------------------------------

#[test]
fn select_star() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT * FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 9);
}

#[test]
fn select_columns() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT id1, v1 FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 2);
    assert_eq!(r.columns[0], "id1");
    assert_eq!(r.columns[1], "v1");
}

#[test]
fn select_literal_scalar_rejected() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let err = match session.execute("SELECT 'x' as s FROM csv") {
        Ok(_) => panic!("expected scalar literal projection to be rejected"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("scalar type"));
}

#[test]
fn where_clause() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT * FROM csv WHERE v1 > 3").unwrap());
    assert_eq!(r.table.nrows(), 14);
}

#[test]
fn group_by_sum() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, SUM(v1) as s FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 2);

    // Sorted by id1: id001=10, id002=26, id003=22, id004=18, id005=34
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 10); // id001
    assert_eq!(r.table.get_i64(1, 1).unwrap(), 26); // id002
    assert_eq!(r.table.get_i64(1, 2).unwrap(), 22); // id003
    assert_eq!(r.table.get_i64(1, 3).unwrap(), 18); // id004
    assert_eq!(r.table.get_i64(1, 4).unwrap(), 34); // id005
}

#[test]
fn group_by_avg() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, AVG(v3) as avg_v3 FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);

    assert!((r.table.get_f64(1, 0).unwrap() - 3.0).abs() < 1e-10); // id001
    assert!((r.table.get_f64(1, 1).unwrap() - 7.0).abs() < 1e-10); // id002
    assert!((r.table.get_f64(1, 2).unwrap() - 11.0).abs() < 1e-10); // id003
}

#[test]
fn group_by_multi() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, id4, SUM(v1) as s, COUNT(*) as cnt FROM csv GROUP BY id1, id4")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 15);
    assert_eq!(r.columns.len(), 4);
}

#[test]
fn having() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, SUM(v1) as s FROM csv GROUP BY id1 HAVING SUM(v1) > 10")
            .unwrap(),
    );
    // id001=10 excluded, id002=26, id003=22, id004=18, id005=34 pass
    assert_eq!(r.table.nrows(), 4);
}

#[test]
fn order_by_asc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT * FROM csv ORDER BY id4").unwrap());
    assert_eq!(r.table.nrows(), 20);

    // id4 is the first column in the sorted output or at its original position
    // Find id4 column index
    let id4_idx = r.columns.iter().position(|c| c == "id4").unwrap();
    let mut prev = i64::MIN;
    for row in 0..20 {
        let val = r.table.get_i64(id4_idx, row).unwrap();
        assert!(val >= prev, "id4 not ascending at row {row}");
        prev = val;
    }
}

#[test]
fn order_by_desc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv ORDER BY v3 DESC")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);

    let v3_idx = r.columns.iter().position(|c| c == "v3").unwrap();
    let mut prev = f64::MAX;
    for row in 0..20 {
        let val = r.table.get_f64(v3_idx, row).unwrap();
        assert!(val <= prev, "v3 not descending at row {row}");
        prev = val;
    }
}

#[test]
fn order_by_limit() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv ORDER BY id4 LIMIT 3")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 3);

    // All 3 rows should have id4=1
    let id4_idx = r.columns.iter().position(|c| c == "id4").unwrap();
    for row in 0..3 {
        assert_eq!(r.table.get_i64(id4_idx, row).unwrap(), 1);
    }
}

#[test]
fn order_by_offset_limit() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv ORDER BY id4 LIMIT 3 OFFSET 2")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 3);
}

#[test]
fn order_by_non_projected_column() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1 FROM csv ORDER BY id4 DESC, id1 ASC LIMIT 1")
            .unwrap(),
    );
    assert_eq!(r.columns.len(), 1);
    assert_eq!(r.table.ncols(), 1);
    assert_eq!(r.table.nrows(), 1);
    assert_eq!(r.table.get_str(0, 0).unwrap(), "id001");
}

#[test]
fn distinct() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT DISTINCT id1 FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 1);
}

#[test]
fn distinct_parenthesized_identifier() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT DISTINCT(id1) FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 1);
}

#[test]
fn select_unknown_projection_column_errors() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let mut session = Session::new().unwrap();
    session.execute("CREATE TABLE t (a INTEGER)").unwrap();
    session.execute("INSERT INTO t VALUES (1)").unwrap();

    let err = match session.execute("SELECT b FROM t") {
        Ok(_) => panic!("expected unknown projection column to error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("Column 'b' not found"));
}

#[test]
fn projection_reorder_is_respected() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let mut session = Session::new().unwrap();
    session
        .execute("CREATE TABLE t (a INTEGER, b INTEGER)")
        .unwrap();
    session.execute("INSERT INTO t VALUES (1, 2)").unwrap();

    let r = unwrap_query(session.execute("SELECT b, a FROM t").unwrap());
    assert_eq!(r.columns.len(), 2);
    assert_eq!(r.columns[0], "b");
    assert_eq!(r.columns[1], "a");
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 2);
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 1);
}

#[test]
fn count_star() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(session.execute("SELECT COUNT(*) FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 1);
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 20);
}

#[test]
fn nested_agg() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, SUM(v1) as s, MIN(v2) as mn, MAX(v3) as mx \
                 FROM csv GROUP BY id1 ORDER BY id1",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 4);

    // id001: SUM(v1)=10, MIN(v2)=2, MAX(v3)=4.5
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 10);
    assert_eq!(r.table.get_i64(2, 0).unwrap(), 2);
    assert!((r.table.get_f64(3, 0).unwrap() - 4.5).abs() < 1e-10);
}

#[test]
fn case_when() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT CASE WHEN v1 > 3 THEN 'high' ELSE 'low' END as label FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 1);
}

#[test]
fn like_filter() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv WHERE id1 LIKE 'id001'")
            .unwrap(),
    );
    // Exact match: 4 rows where id1='id001'
    assert_eq!(r.table.nrows(), 4);
}

#[test]
fn ilike_filter() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv WHERE id1 ILIKE 'ID001'")
            .unwrap(),
    );
    // Case-insensitive exact match: still 4 rows where id1='id001'
    assert_eq!(r.table.nrows(), 4);
}

#[test]
fn join_inner_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_left_file, left_path) = create_test_csv();
    let (_right_file, right_path) = create_join_right_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE x AS SELECT * FROM read_csv('{left_path}')"
        ))
        .unwrap();
    session
        .execute(&format!(
            "CREATE TABLE y AS SELECT * FROM read_csv('{right_path}')"
        ))
        .unwrap();

    let r = unwrap_query(
        session
            .execute("SELECT * FROM x INNER JOIN y ON x.id1 = y.id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 12);
}

#[test]
fn join_left_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_left_file, left_path) = create_test_csv();
    let (_right_file, right_path) = create_join_right_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE x AS SELECT * FROM read_csv('{left_path}')"
        ))
        .unwrap();
    session
        .execute(&format!(
            "CREATE TABLE y AS SELECT * FROM read_csv('{right_path}')"
        ))
        .unwrap();

    let r = unwrap_query(
        session
            .execute("SELECT * FROM x LEFT JOIN y ON x.id1 = y.id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
}

#[test]
fn join_full_outer_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_left_file, left_path) = create_test_csv();
    let (_right_file, right_path) = create_join_right_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE x AS SELECT * FROM read_csv('{left_path}')"
        ))
        .unwrap();
    session
        .execute(&format!(
            "CREATE TABLE y AS SELECT * FROM read_csv('{right_path}')"
        ))
        .unwrap();

    // Left table (csv): id001(4), id002(4), id003(4), id004(4), id005(4) = 20 rows
    // Right table: id001, id002, id003, id006 = 4 rows
    // FULL OUTER on id1: 12 matched + 8 unmatched left (id004,id005) + 1 unmatched right (id006) = 21
    let r = unwrap_query(
        session
            .execute("SELECT * FROM x FULL OUTER JOIN y ON x.id1 = y.id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 21);

    // Verify the result has columns from both sides
    // Left table has 9 cols; right table has 2 cols (id1 = key, skipped; x1 = non-key)
    // Total = 9 left + 1 right (x1) = 10
    assert_eq!(r.columns.len(), 10);
}

#[test]
fn cte() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // Use HAVING inside the CTE to filter (avoids column-name propagation issue)
    let r = unwrap_query(
        session
            .execute(
                "WITH t AS (SELECT id1, SUM(v1) as s FROM csv GROUP BY id1 HAVING SUM(v1) > 20) \
                 SELECT * FROM t",
            )
            .unwrap(),
    );
    // id002=26, id003=22, id005=34 pass; id001=10, id004=18 fail
    assert_eq!(r.table.nrows(), 3);
}

#[test]
fn union_all() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1 FROM csv WHERE v1 = 1 \
                 UNION ALL \
                 SELECT id1 FROM csv WHERE v1 = 2",
            )
            .unwrap(),
    );
    // v1=1 → 2 rows (rows 0,10); v1=2 → 2 rows (rows 1,11); total = 4
    assert_eq!(r.table.nrows(), 4);
}

#[test]
fn union_distinct() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1 FROM csv WHERE v1 = 1 \
                 UNION \
                 SELECT id1 FROM csv WHERE v1 = 2",
            )
            .unwrap(),
    );
    // v1=1 → id001,id003; v1=2 → id001,id003; UNION deduplicates to 2
    assert_eq!(r.table.nrows(), 2);
}

#[test]
fn union_all_literal_column_from_table() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let err = match session.execute(
        "SELECT 'x' AS s FROM csv \
         UNION ALL \
         SELECT 'x' AS s FROM csv",
    ) {
        Ok(_) => panic!("expected UNION ALL literal-column rejection"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("scalar type"));
}

#[test]
fn intersect_all_literal_column_from_table() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let err = match session.execute(
        "SELECT 'x' AS s FROM csv \
         INTERSECT ALL \
         SELECT 'x' AS s FROM csv",
    ) {
        Ok(_) => panic!("expected INTERSECT ALL literal-column rejection"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("scalar type"));
}

#[test]
fn except_all_literal_column_from_table() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let err = match session.execute(
        "SELECT 'x' AS s FROM csv \
         EXCEPT ALL \
         SELECT 'x' AS s FROM csv",
    ) {
        Ok(_) => panic!("expected EXCEPT ALL literal-column rejection"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("scalar type"));
}

#[test]
fn intersect_all_distinct_keys() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1 FROM csv WHERE id1 = 'id001' \
                 INTERSECT ALL \
                 SELECT id1 FROM csv WHERE id1 = 'id002'",
            )
            .unwrap(),
    );
    // No shared key values between the two inputs.
    assert_eq!(r.table.nrows(), 0);
}

#[test]
fn except_all_distinct_keys() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1 FROM csv WHERE id1 = 'id001' \
                 EXCEPT ALL \
                 SELECT id1 FROM csv WHERE id1 = 'id002'",
            )
            .unwrap(),
    );
    // Left side has four id001 rows; right side does not remove any of them.
    assert_eq!(r.table.nrows(), 4);
    for row in 0..r.table.nrows() as usize {
        assert_eq!(r.table.get_str(0, row).unwrap(), "id001");
    }
}

#[test]
fn subquery() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // Filter inside the subquery via HAVING (avoids alias-in-WHERE limitation)
    let r = unwrap_query(
        session
            .execute(
                "SELECT * FROM (SELECT id1, SUM(v1) as s FROM csv GROUP BY id1 HAVING SUM(v1) > 20) sub",
            )
            .unwrap(),
    );
    // id002=26, id003=22, id005=34 pass
    assert_eq!(r.table.nrows(), 3);
}

#[test]
fn string_functions() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT UPPER(id1) as u, LENGTH(id1) as l FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 2);

    // First row: UPPER('id001') = 'ID001', LENGTH('id001') = 5
    assert_eq!(r.table.get_str(0, 0).unwrap(), "ID001");
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 5);
}

#[test]
fn math_expressions() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT v1 + v2 as s, v3 * 2.0 as d FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 2);

    // Row 0: v1=1 + v2=2 = 3; v3=1.5 * 2.0 = 3.0
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 3);
    assert!((r.table.get_f64(1, 0).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn scalar_subexpr_broadcast() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT v1 + 2 * 1 as x, v1 + 2 as y FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 2);
    for i in 0..r.table.nrows() as usize {
        assert_eq!(
            r.table.get_i64(0, i).unwrap(),
            r.table.get_i64(1, i).unwrap()
        );
    }

    let agg = unwrap_query(
        session
            .execute("SELECT SUM(v1 + 2 * 1) as a, SUM(v1 + 2) as b, SUM(v1 - 2 * 1) as d, SUM(v1 - 2) as e, SUM(2 * 1) as c FROM csv")
            .unwrap(),
    );
    assert_eq!(agg.table.nrows(), 1);
    assert_eq!(
        agg.table.get_i64(0, 0).unwrap(),
        agg.table.get_i64(1, 0).unwrap()
    );
    assert_eq!(
        agg.table.get_i64(2, 0).unwrap(),
        agg.table.get_i64(3, 0).unwrap()
    );
    assert_eq!(agg.table.get_i64(4, 0).unwrap(), 40);

    let linear = unwrap_query(
        session
            .execute("SELECT SUM((v1 + 1) * 2) as a, SUM(v1) * 2 + COUNT(*) * 2 as b, AVG(v1 + v2 + 1) as c, AVG(v1 + v2) + 1 as d FROM csv")
            .unwrap(),
    );
    assert_eq!(linear.table.nrows(), 1);
    assert_eq!(
        linear.table.get_i64(0, 0).unwrap(),
        linear.table.get_i64(1, 0).unwrap()
    );
    let c = linear.table.get_f64(2, 0).unwrap();
    let d = linear.table.get_f64(3, 0).unwrap();
    assert!(
        (c - d).abs() < 1e-10,
        "expected AVG linear forms to match: {c} vs {d}"
    );
}

#[test]
fn create_table_as() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    let ddl = session
        .execute("CREATE TABLE t AS SELECT id1, SUM(v1) as s FROM csv GROUP BY id1")
        .unwrap();
    match ddl {
        ExecResult::Ddl(msg) => assert!(msg.contains("Created table"), "unexpected: {msg}"),
        _ => panic!("expected Ddl result"),
    }

    let r = unwrap_query(session.execute("SELECT * FROM t ORDER BY id1").unwrap());
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 2);
}

#[test]
fn multi_sort_keys() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv ORDER BY id1, id4 DESC LIMIT 5")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);

    // First 4 rows should be id001 (sorted first), id4 DESC within
    let id1_idx = r.columns.iter().position(|c| c == "id1").unwrap();
    assert_eq!(r.table.get_str(id1_idx, 0).unwrap(), "id001");
    assert_eq!(r.table.get_str(id1_idx, 3).unwrap(), "id001");
}

#[test]
fn error_table_not_found() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let mut session = Session::new().unwrap();
    let result = session.execute("SELECT * FROM nonexistent");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Window function SQL tests
// ---------------------------------------------------------------------------

#[test]
fn window_row_number_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, v1, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1) as rn FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 3);
    assert_eq!(r.columns[2], "rn");

    // Each id1 partition (5 groups × 4 rows) → rn ∈ {1,2,3,4}
    for row in 0..r.table.nrows() as usize {
        let rn = r.table.get_i64(2, row).unwrap();
        assert!((1..=4).contains(&rn), "ROW_NUMBER should be 1..4, got {rn}");
    }
}

#[test]
fn window_rank_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, id4, RANK() OVER (PARTITION BY id1 ORDER BY id4) as rnk FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns[2], "rnk");
    for row in 0..r.table.nrows() as usize {
        let rnk = r.table.get_i64(2, row).unwrap();
        assert!(rnk >= 1, "RANK should be >= 1, got {rnk}");
    }
}

#[test]
fn window_dense_rank_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, id4, DENSE_RANK() OVER (PARTITION BY id1 ORDER BY id4) as dr FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns[2], "dr");
    for row in 0..r.table.nrows() as usize {
        let dr = r.table.get_i64(2, row).unwrap();
        assert!(
            (1..=3).contains(&dr),
            "DENSE_RANK should be 1..3 (3 distinct id4 values), got {dr}"
        );
    }
}

#[test]
fn window_running_sum_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, v1, SUM(v1) OVER (PARTITION BY id1 ORDER BY v1 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_sum FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns[2], "running_sum");
}

#[test]
fn window_lag_lead_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT v1, LAG(v1) OVER (ORDER BY v1) as lag_v1, LEAD(v1) OVER (ORDER BY v1) as lead_v1 FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 3);
    assert_eq!(r.columns[1], "lag_v1");
    assert_eq!(r.columns[2], "lead_v1");
}

#[test]
fn window_no_partition_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // Entire table is one partition
    let r = unwrap_query(
        session
            .execute("SELECT v1, ROW_NUMBER() OVER (ORDER BY v1) as rn FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    // rn should range from 1 to 20
    let mut rns: Vec<i64> = (0..20).map(|i| r.table.get_i64(1, i).unwrap()).collect();
    rns.sort();
    assert_eq!(rns, (1..=20).collect::<Vec<_>>());
}

#[test]
fn window_mixed_specs_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, id3, \
                 ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY id3) as rn_part, \
                 ROW_NUMBER() OVER (ORDER BY id3) as rn_all \
                 FROM csv ORDER BY id3",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 4);

    let mut per_id1_counts: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();
    for row in 0..r.table.nrows() as usize {
        let id1 = r.table.get_str(0, row).unwrap().to_string();
        let rn_part = r.table.get_i64(2, row).unwrap();
        let rn_all = r.table.get_i64(3, row).unwrap();

        // ORDER BY id3 over full table: row_number must be strictly 1..20.
        assert_eq!(
            rn_all,
            row as i64 + 1,
            "global row number mismatch at row {row}"
        );

        // PARTITION BY id1: row_number must increment per id1 partition.
        let next = per_id1_counts.entry(id1).or_insert(0);
        *next += 1;
        assert_eq!(rn_part, *next, "partition row number mismatch at row {row}");
    }
}

#[test]
fn window_full_partition_avg_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // No ORDER BY → whole partition frame, AVG over entire partition
    let r = unwrap_query(
        session
            .execute("SELECT id1, v1, AVG(v1) OVER (PARTITION BY id1) as avg_v1 FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns[2], "avg_v1");

    // All rows in same partition should have same avg
    // id001: v1 = 1,2,3,4 → avg = 2.5
    for row in 0..r.table.nrows() as usize {
        let id1 = r.table.get_str(0, row).unwrap().to_string();
        let avg = r.table.get_f64(2, row).unwrap();
        if id1 == "id001" {
            assert!(
                (avg - 2.5).abs() < 1e-10,
                "id001 avg should be 2.5, got {avg}"
            );
        }
    }
}

#[test]
fn window_count_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, COUNT(v1) OVER (PARTITION BY id1) as cnt FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    assert_eq!(r.columns.len(), 2);
    // Each id1 partition has 4 rows → count should be 4
    for row in 0..r.table.nrows() as usize {
        let cnt = r.table.get_i64(1, row).unwrap();
        assert_eq!(cnt, 4, "COUNT per partition should be 4, got {cnt}");
    }
}

#[test]
fn window_ntile_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT v1, NTILE(4) OVER (ORDER BY v1) as tile FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    // NTILE(4) over 20 rows → 5 per tile, tiles 1-4
    for row in 0..r.table.nrows() as usize {
        let tile = r.table.get_i64(1, row).unwrap();
        assert!(
            (1..=4).contains(&tile),
            "NTILE(4) should be 1..4, got {tile}"
        );
    }
}

#[test]
fn window_first_last_value_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, v1, FIRST_VALUE(v1) OVER (PARTITION BY id1 ORDER BY v1) as fv FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    // FIRST_VALUE within each partition ordered by v1 → the min v1 in partition
    // id001: v1={1,2,3,4} → first_value = 1
    for row in 0..r.table.nrows() as usize {
        let id1 = r.table.get_str(0, row).unwrap().to_string();
        let fv = r.table.get_i64(2, row).unwrap();
        if id1 == "id001" {
            assert_eq!(fv, 1, "id001 FIRST_VALUE should be 1, got {fv}");
        }
    }
}

#[test]
fn window_expr_with_wildcard_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // SELECT *, ROW_NUMBER() OVER (...) <= 2 should produce original cols + bool column
    let r = unwrap_query(
        session
            .execute(
                "SELECT *, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) <= 2 FROM csv",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    // 9 original columns + 1 bool expression column = 10
    assert_eq!(
        r.columns.len(),
        10,
        "expected 10 columns, got {:?}",
        r.columns
    );
    // Last column should be bool type (type code 1)
    let last_col = r.columns.len() - 1;
    assert_eq!(r.table.col_type(last_col), 1, "last column should be bool");
    // id001 has 4 rows; top-2 by DESC v1 (v1=4,3) → true; v1=2,1 → false
    let mut true_count = 0;
    for row in 0..r.table.nrows() as usize {
        let id1 = r.table.get_str(0, row).unwrap().to_string();
        let flag = r.table.get_i64(last_col, row).unwrap();
        if id1 == "id001" && flag != 0 {
            true_count += 1;
        }
    }
    assert_eq!(
        true_count, 2,
        "id001 should have exactly 2 rows with <= 2 = true"
    );
}

#[test]
fn cte_column_alias_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // CTE with aliased columns should preserve aliases for the outer query
    let r = unwrap_query(
        session
            .execute(
                "WITH w AS (SELECT id1, v1 + v2 AS total FROM csv) \
                 SELECT id1, total FROM w WHERE total > 10 ORDER BY total DESC LIMIT 5",
            )
            .unwrap(),
    );
    assert_eq!(r.columns.len(), 2);
    assert_eq!(r.columns[0], "id1");
    assert_eq!(r.columns[1], "total");
    // Verify values are correct (v1 + v2 > 10 means large rows)
    for row in 0..r.table.nrows() as usize {
        let total = r.table.get_i64(1, row).unwrap();
        assert!(total > 10, "total should be > 10, got {}", total);
    }

    // CTE with window function alias
    let r2 = unwrap_query(
        session
            .execute(
                "WITH w AS (SELECT id1, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) AS rn FROM csv) \
                 SELECT * FROM w WHERE rn <= 2 ORDER BY id1, rn",
            )
            .unwrap(),
    );
    assert_eq!(r2.columns.len(), 2);
    assert_eq!(r2.columns[1], "rn");
    // 5 partitions (id001..id005) × 2 rows each = 10 rows
    assert_eq!(
        r2.table.nrows(),
        10,
        "expected 10 rows from 5 partitions × top-2"
    );
    for row in 0..r2.table.nrows() as usize {
        let rn = r2.table.get_i64(1, row).unwrap();
        assert!(rn <= 2, "rn should be <= 2, got {}", rn);
    }
}

#[test]
fn subquery_predicate_pushdown_sql() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    // Subquery without alias should work (aliasless derived table)
    let r = unwrap_query(
        session
            .execute("SELECT * FROM (SELECT id1, v1 FROM csv) sub WHERE id1 = 'id001'")
            .unwrap(),
    );
    assert_eq!(r.columns.len(), 2);
    assert_eq!(r.table.nrows(), 4, "id001 has 4 rows");
    for row in 0..r.table.nrows() as usize {
        assert_eq!(r.table.get_str(0, row).unwrap(), "id001");
    }

    // Predicate pushdown: equality on PARTITION BY key pushed into subquery
    // The result must be identical whether or not pushdown happens
    let r2 = unwrap_query(
        session
            .execute(
                "SELECT * FROM (\
                     SELECT *, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) <= 2 AS top2 \
                     FROM csv\
                 ) sub WHERE id1 = 'id001'",
            )
            .unwrap(),
    );
    // id001 has 4 rows; top2 = true for the 2 with highest v1 (v1=4, v1=3)
    assert_eq!(r2.table.nrows(), 4, "id001 has 4 rows after pushdown");
    let mut true_count = 0;
    for row in 0..r2.table.nrows() as usize {
        assert_eq!(r2.table.get_str(0, row).unwrap(), "id001");
        let top2 = r2.table.get_i64(r2.columns.len() - 1, row).unwrap();
        if top2 != 0 {
            true_count += 1;
        }
    }
    assert_eq!(true_count, 2, "exactly 2 rows should have top2=true");

    // Non-partition-key predicates stay in outer WHERE (not pushed)
    let r3 = unwrap_query(
        session
            .execute(
                "SELECT * FROM (\
                     SELECT *, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) AS rn \
                     FROM csv\
                 ) sub WHERE id1 = 'id001' AND rn <= 2",
            )
            .unwrap(),
    );
    assert_eq!(r3.table.nrows(), 2, "top-2 rows of id001 partition");
    for row in 0..r3.table.nrows() as usize {
        assert_eq!(r3.table.get_str(0, row).unwrap(), "id001");
        let rn = r3.table.get_i64(r3.columns.len() - 1, row).unwrap();
        assert!(rn <= 2, "rn should be <= 2, got {}", rn);
    }

    // Aliasless subquery (no AS sub) should also work
    let r4 = unwrap_query(
        session
            .execute(
                "SELECT * FROM (\
                     SELECT *, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) <= 2 AS top2 \
                     FROM csv\
                 ) WHERE id1 = 'id002'",
            )
            .unwrap(),
    );
    assert_eq!(r4.table.nrows(), 4, "id002 has 4 rows");
    for row in 0..r4.table.nrows() as usize {
        assert_eq!(r4.table.get_str(0, row).unwrap(), "id002");
    }
}

// ---------------------------------------------------------------------------
// Aggregate FILTER clause
// ---------------------------------------------------------------------------

#[test]
fn agg_filter_sum() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    // SUM(v1) FILTER (WHERE id1 = 'id001')
    // id001 rows have v1 = 1,2,3,4 => SUM = 10
    let r = unwrap_query(
        session
            .execute("SELECT SUM(v1) FILTER (WHERE id1 = 'id001') FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    let filtered_sum = r.table.get_f64(0, 0).unwrap();
    assert!(
        (filtered_sum - 10.0).abs() < 1e-9,
        "expected 10, got {filtered_sum}"
    );
}

#[test]
fn agg_filter_count() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    // COUNT(v1) FILTER (WHERE id1 = 'id001') should be 4
    // Internally rewritten to SUM(IF(cond, 1.0, 0.0)), result is f64
    let r = unwrap_query(
        session
            .execute("SELECT COUNT(v1) FILTER (WHERE id1 = 'id001') FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    let cnt = r.table.get_f64(0, 0).unwrap();
    assert!((cnt - 4.0).abs() < 1e-9, "expected 4, got {cnt}");
}

#[test]
fn agg_filter_with_group_by() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    // GROUP BY id2 with FILTER: SUM(v1) FILTER (WHERE id4 = 1)
    // id2=id001 (10 rows): rows where id4=1 → v1=1,10,3 => 14
    // id2=id002 (10 rows): rows where id4=1 → v1=4,7,6,9 => 26
    let r = unwrap_query(
        session
            .execute(
                "SELECT id2, SUM(v1) FILTER (WHERE id4 = 1) FROM csv GROUP BY id2 ORDER BY id2",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 2);
    // id001 group: filtered sum of v1 where id4=1
    let sum0 = r.table.get_f64(1, 0).unwrap();
    let sum1 = r.table.get_f64(1, 1).unwrap();
    // Verify both groups sum to expected totals (order may vary)
    let pair = if sum0 < sum1 {
        (sum0, sum1)
    } else {
        (sum1, sum0)
    };
    assert!(
        (pair.0 - 14.0).abs() < 1e-9 && (pair.1 - 26.0).abs() < 1e-9,
        "expected (14, 26), got ({}, {})",
        pair.0,
        pair.1
    );
}

#[test]
fn agg_filter_mixed_filtered_and_unfiltered() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();

    // Mix filtered and unfiltered aggs in the same query
    let r = unwrap_query(
        session
            .execute("SELECT SUM(v1), SUM(v1) FILTER (WHERE id1 = 'id001') FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    // Total SUM(v1) = 1+2+3+4+5+6+7+8+9+10+1+2+3+4+5+6+7+8+9+10 = 110
    // SUM(v1) is I64 (no filter → raw column sum)
    let total = r
        .table
        .get_i64(0, 0)
        .map(|v| v as f64)
        .or_else(|| r.table.get_f64(0, 0))
        .unwrap();
    // Filtered SUM is always F64 (because of CAST in the filter rewrite)
    let filtered = r.table.get_f64(1, 0).unwrap();
    assert!(
        (total - 110.0).abs() < 1e-9,
        "total SUM expected 110, got {total}"
    );
    assert!(
        (filtered - 10.0).abs() < 1e-9,
        "filtered SUM expected 10, got {filtered}"
    );
}

// ---------------------------------------------------------------------------
// NULLS FIRST / NULLS LAST with radix sort (>64 rows to trigger radix path)
// ---------------------------------------------------------------------------

/// Create a CSV with 100 rows: columns "id" (i64) and "val" (f64), val = id * 1.0.
fn create_nulls_sort_csv() -> (tempfile::NamedTempFile, String) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "id,val").unwrap();
    for i in 1..=100 {
        writeln!(f, "{},{}.0", i, i).unwrap();
    }
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

#[test]
fn nulls_first_asc_radix() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (file, path) = create_nulls_sort_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();

    // NULLIF(val, 50.0) → NULL when val=50, else val.  ORDER BY ASC NULLS FIRST.
    // The row with val=50 should appear first (NULL sorts first).
    let r = unwrap_query(
        session
            .execute("SELECT id, val FROM t ORDER BY NULLIF(val, 50.0) ASC NULLS FIRST")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 100);
    let val_idx = r.columns.iter().position(|c| c == "val").unwrap();
    let first_val = r.table.get_f64(val_idx, 0).unwrap();
    assert!(
        (first_val - 50.0).abs() < 1e-9,
        "ASC NULLS FIRST: expected val=50 at row 0, got {first_val}"
    );
    // Second row should be val=1.0 (smallest non-null)
    let second_val = r.table.get_f64(val_idx, 1).unwrap();
    assert!(
        (second_val - 1.0).abs() < 1e-9,
        "ASC NULLS FIRST: expected val=1 at row 1, got {second_val}"
    );
    drop(file);
}

#[test]
fn nulls_last_asc_radix() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (file, path) = create_nulls_sort_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();

    // ASC NULLS LAST (default behavior): NULL row should be at the end.
    let r = unwrap_query(
        session
            .execute("SELECT id, val FROM t ORDER BY NULLIF(val, 50.0) ASC NULLS LAST")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 100);
    let val_idx = r.columns.iter().position(|c| c == "val").unwrap();
    let last_val = r.table.get_f64(val_idx, 99).unwrap();
    assert!(
        (last_val - 50.0).abs() < 1e-9,
        "ASC NULLS LAST: expected val=50 at last row, got {last_val}"
    );
    // First row should be val=1.0
    let first_val = r.table.get_f64(val_idx, 0).unwrap();
    assert!(
        (first_val - 1.0).abs() < 1e-9,
        "ASC NULLS LAST: expected val=1 at row 0, got {first_val}"
    );
    drop(file);
}

#[test]
fn nulls_first_desc_radix() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (file, path) = create_nulls_sort_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();

    // DESC NULLS FIRST (default for DESC): NULL row should be first.
    let r = unwrap_query(
        session
            .execute("SELECT id, val FROM t ORDER BY NULLIF(val, 50.0) DESC NULLS FIRST")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 100);
    let val_idx = r.columns.iter().position(|c| c == "val").unwrap();
    let first_val = r.table.get_f64(val_idx, 0).unwrap();
    assert!(
        (first_val - 50.0).abs() < 1e-9,
        "DESC NULLS FIRST: expected val=50 at row 0, got {first_val}"
    );
    // Second row should be val=100.0 (largest non-null)
    let second_val = r.table.get_f64(val_idx, 1).unwrap();
    assert!(
        (second_val - 100.0).abs() < 1e-9,
        "DESC NULLS FIRST: expected val=100 at row 1, got {second_val}"
    );
    drop(file);
}

#[test]
fn nulls_last_desc_radix() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (file, path) = create_nulls_sort_csv();
    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();

    // DESC NULLS LAST: NULL row should be at the end.
    let r = unwrap_query(
        session
            .execute("SELECT id, val FROM t ORDER BY NULLIF(val, 50.0) DESC NULLS LAST")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 100);
    let val_idx = r.columns.iter().position(|c| c == "val").unwrap();
    let last_val = r.table.get_f64(val_idx, 99).unwrap();
    assert!(
        (last_val - 50.0).abs() < 1e-9,
        "DESC NULLS LAST: expected val=50 at last row, got {last_val}"
    );
    // First row should be val=100.0 (largest non-null)
    let first_val = r.table.get_f64(val_idx, 0).unwrap();
    assert!(
        (first_val - 100.0).abs() < 1e-9,
        "DESC NULLS LAST: expected val=100 at row 0, got {first_val}"
    );
    drop(file);
}

// ---------------------------------------------------------------------------
// STDDEV / VARIANCE aggregate tests
// ---------------------------------------------------------------------------

#[test]
fn group_by_stddev() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // id001 v1 = [1,2,3,4], mean=2.5, var_pop=1.25, stddev_samp = sqrt(5/3) ~ 1.2909944
    let r = unwrap_query(
        session
            .execute("SELECT id1, STDDEV(v1) as sd FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let sd_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = (5.0_f64 / 3.0).sqrt(); // sqrt(var_samp) for [1,2,3,4]
    assert!(
        (sd_id001 - expected).abs() < 1e-10,
        "STDDEV(v1) for id001: expected {expected}, got {sd_id001}"
    );
}

#[test]
fn group_by_stddev_pop() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, STDDEV_POP(v1) as sd FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let sd_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = 1.25_f64.sqrt(); // sqrt(var_pop) for [1,2,3,4]
    assert!(
        (sd_id001 - expected).abs() < 1e-10,
        "STDDEV_POP(v1) for id001: expected {expected}, got {sd_id001}"
    );
}

#[test]
fn group_by_variance() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // VARIANCE is an alias for VAR_SAMP
    let r = unwrap_query(
        session
            .execute("SELECT id1, VARIANCE(v1) as vr FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let var_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = 5.0_f64 / 3.0; // var_samp for [1,2,3,4]
    assert!(
        (var_id001 - expected).abs() < 1e-10,
        "VARIANCE(v1) for id001: expected {expected}, got {var_id001}"
    );
}

#[test]
fn group_by_var_pop() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT id1, VAR_POP(v1) as vr FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let var_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = 1.25_f64; // var_pop for [1,2,3,4]
    assert!(
        (var_id001 - expected).abs() < 1e-10,
        "VAR_POP(v1) for id001: expected {expected}, got {var_id001}"
    );
}

#[test]
fn scalar_stddev() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // Scalar aggregate (no GROUP BY): STDDEV over all 20 rows
    // v1 = [1..10, 1..10], n=20, sum=110, mean=5.5
    // sum_sq = 2*(1+4+9+16+25+36+49+64+81+100) = 770
    // var_pop = 770/20 - 30.25 = 8.25
    // var_samp = 8.25 * 20/19 = 165/19
    let r = unwrap_query(session.execute("SELECT STDDEV(v1) as sd FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 1);
    let sd = r.table.get_f64(0, 0).unwrap();
    let expected = (165.0_f64 / 19.0).sqrt();
    assert!(
        (sd - expected).abs() < 1e-10,
        "scalar STDDEV(v1): expected {expected}, got {sd}"
    );
}

#[test]
fn scalar_var_pop() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    let r = unwrap_query(
        session
            .execute("SELECT VAR_POP(v1) as vr FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    let vr = r.table.get_f64(0, 0).unwrap();
    let expected = 8.25_f64; // var_pop for [1..10, 1..10]
    assert!(
        (vr - expected).abs() < 1e-10,
        "scalar VAR_POP(v1): expected {expected}, got {vr}"
    );
}

#[test]
fn stddev_with_f64_input() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // Test STDDEV on f64 column (v3) with GROUP BY
    let r = unwrap_query(
        session
            .execute("SELECT id1, STDDEV_POP(v3) as sd FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    // id001 v3 = [1.5, 2.5, 3.5, 4.5], mean=3.0, var_pop = avg(sq) - mean^2
    // sum_sq = 2.25+6.25+12.25+20.25=41.0, var_pop = 41/4 - 9 = 1.25
    let sd_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = 1.25_f64.sqrt();
    assert!(
        (sd_id001 - expected).abs() < 1e-10,
        "STDDEV_POP(v3) for id001: expected {expected}, got {sd_id001}"
    );
}

#[test]
fn stddev_samp_alias() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // STDDEV_SAMP should behave identically to STDDEV
    let r = unwrap_query(
        session
            .execute("SELECT id1, STDDEV_SAMP(v1) as sd FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let sd_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = (5.0_f64 / 3.0).sqrt();
    assert!(
        (sd_id001 - expected).abs() < 1e-10,
        "STDDEV_SAMP(v1) for id001: expected {expected}, got {sd_id001}"
    );
}

#[test]
fn var_samp_alias() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // VAR_SAMP should behave identically to VARIANCE
    let r = unwrap_query(
        session
            .execute("SELECT id1, VAR_SAMP(v1) as vr FROM csv GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    let var_id001 = r.table.get_f64(1, 0).unwrap();
    let expected = 5.0_f64 / 3.0;
    assert!(
        (var_id001 - expected).abs() < 1e-10,
        "VAR_SAMP(v1) for id001: expected {expected}, got {var_id001}"
    );
}

#[test]
fn stddev_single_element_group_returns_nan() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // STDDEV (sample) with count=1 should return NaN (NULL), not 0
    // id3 has unique values → each group has exactly 1 row
    let r = unwrap_query(
        session
            .execute("SELECT id3, STDDEV(v1) as sd FROM csv GROUP BY id3 ORDER BY id3")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    let sd = r.table.get_f64(1, 0).unwrap();
    assert!(
        sd.is_nan(),
        "STDDEV of single-element group should be NaN, got {sd}"
    );
}

#[test]
fn variance_single_element_group_returns_nan() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (mut session, _f) = setup_session();
    // VARIANCE (sample) with count=1 should return NaN (NULL), not 0
    let r = unwrap_query(
        session
            .execute("SELECT id3, VARIANCE(v1) as vr FROM csv GROUP BY id3 ORDER BY id3")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
    let vr = r.table.get_f64(1, 0).unwrap();
    assert!(
        vr.is_nan(),
        "VARIANCE of single-element group should be NaN, got {vr}"
    );
}

// ---------------------------------------------------------------------------
// Parted table SQL tests
// ---------------------------------------------------------------------------

/// Helper: create a small 2-partition parted DB in a tempdir.
/// Returns (tempdir, db_root_path).
fn create_parted_db() -> (tempfile::TempDir, String) {
    use std::ffi::CString;

    let dir = tempfile::TempDir::new().unwrap();
    let db_root = dir.path().to_str().unwrap().to_string();

    // Create test CSV on disk
    let (file, csv_path) = create_test_csv();

    // Read CSV into a table
    let ctx = teide::Context::new().unwrap();
    let full_table = ctx.read_csv(&csv_path).unwrap();
    let nrows = full_table.nrows();
    let ncols = full_table.ncols();
    assert!(nrows == 20);

    let half = nrows / 2; // 10 rows per partition

    // Create 2 partitions: 2024.01.01 and 2024.01.02
    for part in 0..2i64 {
        let start = part * half;
        let prows = if part == 1 { nrows - start } else { half };
        let date_str = format!("2024.01.{:02}", part + 1);
        let part_dir = format!("{db_root}/{date_str}/data");
        std::fs::create_dir_all(&part_dir).unwrap();

        unsafe {
            let mut sub_tbl = teide::ffi::td_table_new(ncols);
            for c in 0..ncols {
                let col = teide::ffi::td_table_get_col_idx(full_table.as_raw(), c);
                let name_id = teide::ffi::td_table_col_name(full_table.as_raw(), c);
                let sliced = teide::ffi::td_vec_slice(col, start, prows);
                assert!(!sliced.is_null());
                sub_tbl = teide::ffi::td_table_add_col(sub_tbl, name_id, sliced);
                teide::ffi::td_release(sliced);
            }

            let c_dir = CString::new(part_dir.as_str()).unwrap();
            let sym_path = format!("{db_root}/sym");
            let c_sym = CString::new(sym_path.as_str()).unwrap();
            let err = teide::ffi::td_splay_save(sub_tbl, c_dir.as_ptr(), c_sym.as_ptr());
            assert_eq!(
                err,
                teide::ffi::td_err_t::TD_OK,
                "splay_save failed for partition {part}"
            );
            teide::ffi::td_release(sub_tbl);
        }
    }

    // Save shared symfile
    unsafe {
        let sym_path = format!("{db_root}/sym");
        let c_sym = CString::new(sym_path.as_str()).unwrap();
        let err = teide::ffi::td_sym_save(c_sym.as_ptr());
        assert_eq!(err, teide::ffi::td_err_t::TD_OK, "sym_save failed");
    }

    drop(full_table);
    drop(ctx);
    drop(file);
    (dir, db_root)
}

#[test]
fn parted_sql_groupby() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    // Query the parted table via path syntax: 'db_root/table_name'
    let r = unwrap_query(
        session
            .execute(&format!(
                "SELECT id1, SUM(v1) as s FROM read_parted('{db_root}', 'data') GROUP BY id1 ORDER BY id1"
            ))
            .unwrap(),
    );

    // 5 distinct id1 groups (id001..id005), 20 rows total
    assert_eq!(r.table.nrows(), 5, "expected 5 groups");

    // Verify first group: id001 has v1 = 1+2+3+4 = 10
    let sum0 = r
        .table
        .get_i64(1, 0)
        .or_else(|| r.table.get_f64(1, 0).map(|f| f as i64));
    assert_eq!(sum0, Some(10), "SUM(v1) for id001 should be 10");
}

#[test]
fn parted_sql_count() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    let r = unwrap_query(
        session
            .execute(&format!(
                "SELECT COUNT(*) as cnt FROM read_parted('{db_root}', 'data')"
            ))
            .unwrap(),
    );

    assert_eq!(r.table.nrows(), 1);
    let cnt = r.table.get_i64(0, 0).unwrap();
    assert_eq!(cnt, 20, "parted table should have 20 rows total");
}

#[test]
fn parted_sql_create_table_as_groupby() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    // Register a derived table from a GROUP BY on the parted table
    session
        .execute(&format!(
            "CREATE TABLE agg AS SELECT id1, SUM(v1) as s FROM read_parted('{db_root}', 'data') GROUP BY id1"
        ))
        .unwrap();

    // Query the registered (flat) table
    let r = unwrap_query(
        session
            .execute("SELECT id1, s FROM agg ORDER BY id1")
            .unwrap(),
    );

    assert_eq!(
        r.table.nrows(),
        5,
        "expected 5 groups from registered agg table"
    );
}

#[test]
fn parted_sql_multi_key_groupby() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    // Multi-key GROUP BY on parted table
    let r = unwrap_query(
        session
            .execute(&format!(
                "SELECT id1, id2, SUM(v1) as s FROM read_parted('{db_root}', 'data') GROUP BY id1, id2 ORDER BY id1, id2"
            ))
            .unwrap(),
    );

    // 5 id1 values × 2 id2 values = 10 groups
    assert_eq!(r.table.nrows(), 10, "expected 10 groups for (id1, id2)");
}

#[test]
fn parted_sql_select_star() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    // SELECT * on a parted table — should see all 20 rows
    let r = unwrap_query(
        session
            .execute(&format!("SELECT * FROM read_parted('{db_root}', 'data')"))
            .unwrap(),
    );

    assert_eq!(r.table.nrows(), 20, "SELECT * should return all 20 rows");
    // 9 data columns (no MAPCOMMON in result since we didn't add Date)
    assert!(r.table.ncols() >= 9, "expected at least 9 columns");

    // Verify row access works across partition boundaries
    // First partition has rows 0-9, second has rows 10-19
    let v1_row0 = r.table.get_i64(6, 0); // v1 col, first row
    let v1_row10 = r.table.get_i64(6, 10); // v1 col, first row of 2nd partition
    assert!(v1_row0.is_some(), "should read v1 from partition 0");
    assert!(v1_row10.is_some(), "should read v1 from partition 1");

    // id1 is SYM column — verify string access works across partitions
    let id1_row0 = r.table.get_str(0, 0);
    let id1_row19 = r.table.get_str(0, 19);
    assert!(id1_row0.is_some(), "should read id1 from first row");
    assert!(id1_row19.is_some(), "should read id1 from last row");
}

#[test]
fn parted_sql_create_table_select_star() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let (_dir, db_root) = create_parted_db();

    let mut session = Session::new().unwrap();

    // CREATE TABLE AS SELECT * from parted table
    session
        .execute(&format!(
            "CREATE TABLE flat AS SELECT * FROM read_parted('{db_root}', 'data')"
        ))
        .unwrap();

    // Query the registered table — should work with GROUP BY
    let r = unwrap_query(
        session
            .execute("SELECT id1, SUM(v1) as s FROM flat GROUP BY id1 ORDER BY id1")
            .unwrap(),
    );

    assert_eq!(
        r.table.nrows(),
        5,
        "expected 5 groups from registered flat table"
    );
}
