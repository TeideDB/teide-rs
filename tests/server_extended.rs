//! Extended integration tests for the PostgreSQL wire protocol server.
//!
//! Covers multi-query sessions, catalog queries, result edge cases,
//! error responses, and concurrent connections.

#![cfg(feature = "server")]
#![allow(clippy::await_holding_lock)]

use std::io::Write;
use std::process::{Child, Command};
use std::sync::Mutex;

use tokio_postgres::{NoTls, SimpleQueryMessage};

// The C engine uses global state — serialize all tests.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that kills the server child process on drop.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

/// Create a temp CSV with varied data types for comprehensive testing.
fn create_extended_csv() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(f, "id,name,value,score").unwrap();
    writeln!(f, "1,alice,10,1.5").unwrap();
    writeln!(f, "2,bob,20,2.5").unwrap();
    writeln!(f, "3,alice,30,3.5").unwrap();
    writeln!(f, "4,charlie,40,4.5").unwrap();
    writeln!(f, "5,bob,50,5.5").unwrap();
    writeln!(f, "6,alice,60,6.5").unwrap();
    writeln!(f, "7,dave,70,7.5").unwrap();
    writeln!(f, "8,charlie,80,8.5").unwrap();
    writeln!(f, "9,eve,90,9.5").unwrap();
    writeln!(f, "10,bob,100,10.5").unwrap();
    f.flush().unwrap();
    f
}

/// Start the server on a given port, load the test CSV, return a guard.
fn start_server(port: u16, csv_path: &str) -> ServerGuard {
    let binary = env!("CARGO_BIN_EXE_teide-server");
    let child = Command::new(binary)
        .arg("--port")
        .arg(port.to_string())
        .arg("--load")
        .arg(format!("t={csv_path}"))
        .spawn()
        .expect("failed to start teide-server");
    ServerGuard(child)
}

/// Connect to the test server.
async fn connect(port: u16) -> tokio_postgres::Client {
    let connstr = format!("host=127.0.0.1 port={port} user=test dbname=teide");
    let (client, connection) = tokio_postgres::connect(&connstr, NoTls)
        .await
        .expect("failed to connect");

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {e}");
        }
    });

    client
}

/// Extract data rows from simple_query results.
fn extract_rows(messages: &[SimpleQueryMessage]) -> Vec<&tokio_postgres::SimpleQueryRow> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-Query Sessions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_multi_query_same_connection() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15450, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15450).await;

    // First query
    let msgs1 = client.simple_query("SELECT COUNT(*) FROM t").await.unwrap();
    let rows1 = extract_rows(&msgs1);
    assert_eq!(rows1.len(), 1);
    let count: &str = rows1[0].get(0).unwrap();
    assert_eq!(count, "10");

    // Second query on same connection
    let msgs2 = client
        .simple_query("SELECT name, SUM(value) AS total FROM t GROUP BY name ORDER BY total DESC")
        .await
        .unwrap();
    let rows2 = extract_rows(&msgs2);
    assert!(rows2.len() > 0, "GROUP BY should return rows");

    // Third query — ordering
    let msgs3 = client
        .simple_query("SELECT id, name FROM t ORDER BY id LIMIT 3")
        .await
        .unwrap();
    let rows3 = extract_rows(&msgs3);
    assert_eq!(rows3.len(), 3);
}

#[tokio::test]
async fn server_create_and_query_in_session() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15451, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15451).await;

    // Create a derived table
    client
        .simple_query("CREATE TABLE big_values AS SELECT * FROM t WHERE value > 50")
        .await
        .unwrap();

    // Query the derived table
    let msgs = client
        .simple_query("SELECT COUNT(*) FROM big_values")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 1);
    let count: &str = rows[0].get(0).unwrap();
    // Values > 50: 60, 70, 80, 90, 100 → 5 rows
    assert_eq!(count, "5");
}

// ---------------------------------------------------------------------------
// Result Edge Cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_empty_result_set() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15452, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15452).await;

    // Query that returns no rows
    let msgs = client
        .simple_query("SELECT * FROM t WHERE value > 9999")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn server_all_column_types() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15453, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15453).await;

    // Query returns int, string, and float columns
    let msgs = client
        .simple_query("SELECT id, name, value, score FROM t ORDER BY id LIMIT 1")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 1);
    let id: &str = rows[0].get(0).unwrap();
    let name: &str = rows[0].get(1).unwrap();
    let value: &str = rows[0].get(2).unwrap();
    let score: &str = rows[0].get(3).unwrap();
    assert_eq!(id, "1");
    assert_eq!(name, "alice");
    assert_eq!(value, "10");
    assert_eq!(score, "1.5");
}

#[tokio::test]
async fn server_large_result_set() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Create a larger CSV
    let mut f = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(f, "id,value").unwrap();
    for i in 0..1000 {
        writeln!(f, "{i},{}", i * 10).unwrap();
    }
    f.flush().unwrap();

    let _server = start_server(15454, f.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15454).await;

    let msgs = client
        .simple_query("SELECT COUNT(*) FROM t")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 1);
    let count: &str = rows[0].get(0).unwrap();
    assert_eq!(count, "1000");
}

#[tokio::test]
async fn server_single_column_result() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15455, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15455).await;

    let msgs = client
        .simple_query("SELECT DISTINCT name FROM t ORDER BY name")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 5); // alice, bob, charlie, dave, eve
}

// ---------------------------------------------------------------------------
// Error Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_error_parse() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15456, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15456).await;

    // Malformed SQL
    let err = client
        .simple_query("SELECTT * FORM t")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "parse error should produce non-empty message"
    );
}

#[tokio::test]
async fn server_error_unknown_column() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15457, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15457).await;

    let err = client
        .simple_query("SELECT nonexistent_col FROM t")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "unknown column error should produce non-empty message"
    );
}

#[tokio::test]
async fn server_error_then_recover() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15458, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15458).await;

    // First query: error
    let _err = client
        .simple_query("SELECT * FROM nonexistent")
        .await
        .unwrap_err();

    // Second query: should succeed (connection recovered)
    let msgs = client
        .simple_query("SELECT COUNT(*) FROM t")
        .await
        .unwrap();
    let rows = extract_rows(&msgs);
    assert_eq!(rows.len(), 1);
    let count: &str = rows[0].get(0).unwrap();
    assert_eq!(count, "10");
}

// ---------------------------------------------------------------------------
// Concurrent Connections
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_concurrent_connections() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15459, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Open two connections
    let client1 = connect(15459).await;
    let client2 = connect(15459).await;

    // Run queries on both
    let (r1, r2) = tokio::join!(
        client1.simple_query("SELECT COUNT(*) FROM t"),
        client2.simple_query("SELECT SUM(value) FROM t"),
    );

    let msgs1 = r1.unwrap();
    let msgs2 = r2.unwrap();
    let rows1 = extract_rows(&msgs1);
    let rows2 = extract_rows(&msgs2);

    let count: &str = rows1[0].get(0).unwrap();
    assert_eq!(count, "10");

    let sum: &str = rows2[0].get(0).unwrap();
    assert_eq!(sum, "550"); // 10+20+30+...+100
}

// ---------------------------------------------------------------------------
// Extended Protocol Coverage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extended_aggregate_functions() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15460, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15460).await;

    let rows = client
        .query(
            "SELECT MIN(value) AS mn, MAX(value) AS mx, AVG(value) AS av FROM t",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    let mn: &str = rows[0].get(0);
    let mx: &str = rows[0].get(1);
    assert_eq!(mn, "10");
    assert_eq!(mx, "100");
}

#[tokio::test]
async fn extended_multi_query_session() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15461, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15461).await;

    // First query via extended protocol
    let rows1 = client
        .query("SELECT COUNT(*) FROM t", &[])
        .await
        .unwrap();
    let count: &str = rows1[0].get(0);
    assert_eq!(count, "10");

    // Second query via extended protocol
    let rows2 = client
        .query("SELECT DISTINCT name FROM t ORDER BY name", &[])
        .await
        .unwrap();
    assert_eq!(rows2.len(), 5);
}

#[tokio::test]
async fn extended_where_filter() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15462, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15462).await;

    let rows = client
        .query("SELECT id, name FROM t WHERE value > 50 ORDER BY id", &[])
        .await
        .unwrap();
    assert_eq!(rows.len(), 5);
    let first_id: &str = rows[0].get(0);
    assert_eq!(first_id, "6");
}

// ---------------------------------------------------------------------------
// DDL via server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_drop_table() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_extended_csv();
    let _server = start_server(15463, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15463).await;

    // Create derived table
    client
        .simple_query("CREATE TABLE t2 AS SELECT * FROM t LIMIT 3")
        .await
        .unwrap();

    // Verify it exists
    let msgs = client.simple_query("SELECT COUNT(*) FROM t2").await.unwrap();
    let rows = extract_rows(&msgs);
    let count: &str = rows[0].get(0).unwrap();
    assert_eq!(count, "3");

    // Drop it
    client.simple_query("DROP TABLE t2").await.unwrap();

    // Verify it's gone
    let err = client
        .simple_query("SELECT * FROM t2")
        .await
        .unwrap_err();
    assert!(!err.to_string().is_empty());
}
