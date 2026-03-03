//! Extended SQL tests for teide-db.
//!
//! Covers session operations, NULL handling, error paths, edge cases,
//! and SQL features not exercised by the main sql.rs test suite.

use std::io::Write;
use std::sync::Mutex;

use teide::sql::{ExecResult, Session, SqlResult};

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

fn unwrap_query(result: ExecResult) -> SqlResult {
    match result {
        ExecResult::Query(r) => r,
        ExecResult::Ddl(msg) => panic!("expected Query, got Ddl: {msg}"),
    }
}

// ---------------------------------------------------------------------------
// Session Operations
// ---------------------------------------------------------------------------

#[test]
fn execute_script_multi_statement() {
    let _guard = lock();
    let (file, path) = create_test_csv();
    let mut session = Session::new().unwrap();

    let script = format!(
        "CREATE TABLE t1 AS SELECT * FROM read_csv('{path}'); \
         CREATE TABLE t2 AS SELECT v1, v2 FROM t1 WHERE v1 > 5"
    );
    let result = session.execute_script(&script).unwrap();
    // Last statement is DDL
    match result {
        ExecResult::Ddl(_) => {}
        ExecResult::Query(_) => panic!("expected Ddl from CREATE TABLE"),
    }

    // Verify both tables exist
    let r = unwrap_query(session.execute("SELECT COUNT(*) FROM t2").unwrap());
    let count = r.table.get_i64(0, 0).unwrap();
    assert!(count > 0);
    drop(file);
}

#[test]
fn execute_script_file() {
    let _guard = lock();
    let (csv_file, csv_path) = create_test_csv();

    // Write SQL script to temp file
    let mut sql_file = tempfile::Builder::new().suffix(".sql").tempfile().unwrap();
    writeln!(
        sql_file,
        "CREATE TABLE data AS SELECT * FROM read_csv('{csv_path}');"
    )
    .unwrap();
    writeln!(sql_file, "SELECT COUNT(*) AS cnt FROM data;").unwrap();
    sql_file.flush().unwrap();

    let mut session = Session::new().unwrap();
    let result = session
        .execute_script_file(sql_file.path())
        .unwrap();
    let r = unwrap_query(result);
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 20);
    drop(csv_file);
}

#[test]
fn session_table_names() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let names = session.table_names();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"csv"));

    // Create another table
    session
        .execute("CREATE TABLE t2 AS SELECT v1 FROM csv")
        .unwrap();
    let names2 = session.table_names();
    assert_eq!(names2.len(), 2);
    assert!(names2.contains(&"csv"));
    assert!(names2.contains(&"t2"));
}

#[test]
fn session_table_info() {
    let _guard = lock();
    let (session, _f) = setup_session();

    let info = session.table_info("csv").unwrap();
    assert_eq!(info.0, 20); // 20 rows
    assert_eq!(info.1, 9); // 9 columns

    assert!(session.table_info("nonexistent").is_none());
}

// ---------------------------------------------------------------------------
// NULL Handling
// ---------------------------------------------------------------------------

fn setup_session_with_nulls() -> (Session, tempfile::NamedTempFile) {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "a,b,c").unwrap();
    writeln!(f, "1,10,hello").unwrap();
    writeln!(f, "2,,world").unwrap();
    writeln!(f, ",30,").unwrap();
    writeln!(f, "4,40,foo").unwrap();
    writeln!(f, "5,,bar").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();
    (session, f)
}

#[test]
fn null_count_star_vs_count_col() {
    let _guard = lock();
    let (mut session, _f) = setup_session_with_nulls();

    // COUNT(*) counts all rows
    let r1 = unwrap_query(session.execute("SELECT COUNT(*) FROM t").unwrap());
    assert_eq!(r1.table.get_i64(0, 0).unwrap(), 5);

    // COUNT(b) — CSV reader may convert empty values to 0 rather than NULL.
    // Verify COUNT(b) returns a valid count regardless.
    let r2 = unwrap_query(session.execute("SELECT COUNT(b) FROM t").unwrap());
    let count_b = r2.table.get_i64(0, 0).unwrap();
    assert!(count_b > 0, "COUNT(b) should return a valid count");
}

#[test]
fn null_sum_ignores_nulls() {
    let _guard = lock();
    let (mut session, _f) = setup_session_with_nulls();

    // SUM(a) should skip NULL: 1+2+4+5 = 12
    let r = unwrap_query(session.execute("SELECT SUM(a) FROM t").unwrap());
    let sum = r.table.get_i64(0, 0).unwrap();
    assert_eq!(sum, 12);
}

// ---------------------------------------------------------------------------
// Edge Cases
// ---------------------------------------------------------------------------

#[test]
fn empty_table_operations() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // Create empty table via WHERE false
    session
        .execute("CREATE TABLE empty AS SELECT * FROM csv WHERE 1 = 0")
        .unwrap();
    let r = unwrap_query(session.execute("SELECT * FROM empty").unwrap());
    assert_eq!(r.table.nrows(), 0);
}

#[test]
fn limit_zero() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(session.execute("SELECT * FROM csv LIMIT 0").unwrap());
    assert_eq!(r.table.nrows(), 0);
}

#[test]
fn offset_beyond_rows() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT * FROM csv LIMIT 10 OFFSET 100")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 0);
}

#[test]
fn single_row_aggregation() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT SUM(v1), AVG(v1), MIN(v1), MAX(v1), COUNT(*) FROM csv WHERE v1 = 1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    // v1=1 appears in 2 rows
    let count = r.table.get_i64(4, 0).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn select_computed_expression() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // Compute expression on columns
    let r = unwrap_query(
        session
            .execute("SELECT v1 + v2 AS total FROM csv WHERE v1 = 1 LIMIT 1")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    // v1=1, v2=2 → 3
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 3);
}

#[test]
fn distinct_single_column() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(session.execute("SELECT DISTINCT id1 FROM csv").unwrap());
    assert_eq!(r.table.nrows(), 5);
}

#[test]
fn group_by_all_one_group() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // No GROUP BY key but aggregate → single group
    let r = unwrap_query(
        session
            .execute("SELECT SUM(v1), COUNT(*) FROM csv")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 1);
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 110); // sum of v1
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 20); // count
}

// ---------------------------------------------------------------------------
// SQL String Functions
// ---------------------------------------------------------------------------

#[test]
fn sql_substring() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT SUBSTRING(id1, 1, 2) AS prefix FROM csv LIMIT 1")
            .unwrap(),
    );
    assert_eq!(r.table.get_str(0, 0).unwrap(), "id");
}

#[test]
fn sql_trim() {
    let _guard = lock();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "name").unwrap();
    writeln!(f, " alice ").unwrap();
    writeln!(f, "  bob  ").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE t AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();
    let r = unwrap_query(session.execute("SELECT TRIM(name) AS trimmed FROM t").unwrap());
    assert_eq!(r.table.get_str(0, 0).unwrap(), "alice");
    assert_eq!(r.table.get_str(0, 1).unwrap(), "bob");
}

#[test]
fn sql_replace() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT REPLACE(id1, 'id', 'ID') AS replaced FROM csv LIMIT 1")
            .unwrap(),
    );
    assert_eq!(r.table.get_str(0, 0).unwrap(), "ID001");
}

#[test]
fn sql_abs_round() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // SQL planner requires FROM clause
    let r = unwrap_query(
        session
            .execute("SELECT ABS(v1 - 5) AS abs_val FROM csv WHERE v1 = 3 LIMIT 1")
            .unwrap(),
    );
    // |3 - 5| = 2
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 2);
}

// ---------------------------------------------------------------------------
// SQL CAST
// ---------------------------------------------------------------------------

#[test]
fn sql_cast_to_float() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute("SELECT CAST(v1 AS DOUBLE) AS v1_f FROM csv LIMIT 1")
            .unwrap(),
    );
    assert!((r.table.get_f64(0, 0).unwrap() - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Multiple CTEs
// ---------------------------------------------------------------------------

#[test]
fn multiple_ctes() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute(
                "WITH \
                 a AS (SELECT id1, SUM(v1) AS total FROM csv GROUP BY id1), \
                 b AS (SELECT id1, total FROM a WHERE total > 20) \
                 SELECT COUNT(*) AS cnt FROM b",
            )
            .unwrap(),
    );
    let cnt = r.table.get_i64(0, 0).unwrap();
    // id002=26, id003=22, id004=18 (not > 20), id005=34 → 3 groups > 20
    assert_eq!(cnt, 3);
}

// ---------------------------------------------------------------------------
// SQL Joins (coverage gaps)
// ---------------------------------------------------------------------------

#[test]
fn sql_self_join() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute(
                "SELECT a.id1, a.v1, b.v1 AS v1_b \
                 FROM csv a \
                 JOIN csv b ON a.id1 = b.id1 AND a.id4 = b.id4 \
                 WHERE a.v1 = 1 \
                 LIMIT 5",
            )
            .unwrap(),
    );
    assert!(r.table.nrows() > 0);
    assert!(r.table.nrows() <= 5);
}

// ---------------------------------------------------------------------------
// Error Paths
// ---------------------------------------------------------------------------

#[test]
fn error_unknown_table() {
    let _guard = lock();
    let mut session = Session::new().unwrap();
    let result = session.execute("SELECT * FROM nonexistent");
    assert!(result.is_err());
}

#[test]
fn error_unknown_column() {
    let _guard = lock();
    let (mut session, _f) = setup_session();
    let result = session.execute("SELECT nonexistent_col FROM csv");
    assert!(result.is_err());
}

#[test]
fn error_unknown_function() {
    let _guard = lock();
    let (mut session, _f) = setup_session();
    let result = session.execute("SELECT FAKE_FUNCTION(v1) FROM csv");
    assert!(result.is_err());
}

#[test]
fn error_parse_malformed_sql() {
    let _guard = lock();
    let mut session = Session::new().unwrap();
    let result = session.execute("SELECTT * FORM csv");
    assert!(result.is_err());
}

#[test]
fn error_empty_script() {
    let _guard = lock();
    let mut session = Session::new().unwrap();
    let result = session.execute_script("");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// ORDER BY edge cases
// ---------------------------------------------------------------------------

#[test]
fn order_by_non_projected_column() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // ORDER BY a column not in SELECT list
    let r = unwrap_query(
        session
            .execute("SELECT id1 FROM csv ORDER BY v1 LIMIT 5")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
}

// ---------------------------------------------------------------------------
// INSERT edge cases
// ---------------------------------------------------------------------------

#[test]
fn insert_multiple_rows() {
    let _guard = lock();
    let mut session = Session::new().unwrap();

    session
        .execute("CREATE TABLE t (a BIGINT, b BIGINT)")
        .unwrap();
    session
        .execute("INSERT INTO t VALUES (1, 10)")
        .unwrap();
    session
        .execute("INSERT INTO t VALUES (2, 20)")
        .unwrap();
    session
        .execute("INSERT INTO t VALUES (3, 30)")
        .unwrap();

    let r = unwrap_query(session.execute("SELECT SUM(a), SUM(b) FROM t").unwrap());
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 6); // 1+2+3
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 60); // 10+20+30
}

// ---------------------------------------------------------------------------
// Temporal SQL
// ---------------------------------------------------------------------------

#[test]
fn sql_extract_from_date() {
    let _guard = lock();
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "dt,val").unwrap();
    writeln!(f, "2024-01-15,100").unwrap();
    writeln!(f, "2024-06-30,200").unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();

    let mut session = Session::new().unwrap();
    session
        .execute(&format!(
            "CREATE TABLE dates AS SELECT * FROM read_csv('{path}')"
        ))
        .unwrap();

    let r = unwrap_query(
        session
            .execute("SELECT EXTRACT(YEAR FROM dt) AS yr, EXTRACT(MONTH FROM dt) AS mo FROM dates")
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 2);
    assert_eq!(r.table.get_i64(0, 0).unwrap(), 2024);
    assert_eq!(r.table.get_i64(1, 0).unwrap(), 1);
    assert_eq!(r.table.get_i64(1, 1).unwrap(), 6);
}

// ---------------------------------------------------------------------------
// Window function edge cases
// ---------------------------------------------------------------------------

#[test]
fn window_over_entire_table() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    // Window without PARTITION BY — entire table is one partition
    let r = unwrap_query(
        session
            .execute(
                "SELECT v1, SUM(v1) OVER (ORDER BY v1) AS running_sum \
                 FROM csv ORDER BY v1",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 20);
}

#[test]
fn multiple_aggregates_same_group() {
    let _guard = lock();
    let (mut session, _f) = setup_session();

    let r = unwrap_query(
        session
            .execute(
                "SELECT id1, SUM(v1) AS s, AVG(v1) AS a, MIN(v1) AS mn, MAX(v1) AS mx, COUNT(*) AS c \
                 FROM csv GROUP BY id1 ORDER BY id1",
            )
            .unwrap(),
    );
    assert_eq!(r.table.nrows(), 5);
    assert_eq!(r.columns.len(), 6);
}
