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

//! Integration tests for the PostgreSQL wire protocol server.
//!
//! Starts teide-server as a child process, connects via tokio-postgres,
//! and runs queries over both simple and extended query protocols.

#![cfg(feature = "server")]
#![allow(clippy::await_holding_lock)] // Intentional: serializes tests for C engine global state

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

/// Create a temp CSV file with small test data and return the path.
fn create_test_csv() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(f, "id,name,value").unwrap();
    writeln!(f, "1,alice,10").unwrap();
    writeln!(f, "2,bob,20").unwrap();
    writeln!(f, "3,alice,30").unwrap();
    writeln!(f, "4,charlie,40").unwrap();
    writeln!(f, "5,bob,50").unwrap();
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

/// Extract data rows from simple_query results, skipping CommandComplete messages.
fn extract_rows(messages: &[SimpleQueryMessage]) -> Vec<&tokio_postgres::SimpleQueryRow> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn server_version_query() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15433, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15433).await;
    let messages = client.simple_query("SELECT version()").await.unwrap();
    let rows = extract_rows(&messages);
    assert_eq!(rows.len(), 1);
    let version: &str = rows[0].get(0).unwrap();
    assert!(
        version.contains("Teide"),
        "version should contain 'Teide': {version}"
    );
}

#[tokio::test]
async fn server_select_count() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15434, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15434).await;
    let messages = client.simple_query("SELECT COUNT(*) FROM t").await.unwrap();
    let rows = extract_rows(&messages);
    assert_eq!(rows.len(), 1);
    let count: &str = rows[0].get(0).unwrap();
    assert_eq!(count, "5");
}

#[tokio::test]
async fn server_group_by() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15435, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15435).await;
    let messages = client
        .simple_query("SELECT name, SUM(value) AS total FROM t GROUP BY name ORDER BY total DESC")
        .await
        .unwrap();
    let rows = extract_rows(&messages);

    assert_eq!(rows.len(), 3);
    let name0: &str = rows[0].get(0).unwrap();
    let total0: &str = rows[0].get(1).unwrap();
    assert_eq!(name0, "bob");
    assert_eq!(total0, "70");
}

#[tokio::test]
async fn server_catalog_tables() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15436, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15436).await;
    let messages = client
        .simple_query("SELECT table_name FROM information_schema.tables")
        .await
        .unwrap();
    let rows = extract_rows(&messages);
    assert_eq!(rows.len(), 1);
    // Catalog handler returns all 4 columns: table_catalog(0), table_schema(1),
    // table_name(2), table_type(3) — regardless of SELECT list.
    let name: &str = rows[0].get(2).unwrap();
    assert_eq!(name, "t");
}

#[tokio::test]
async fn server_error_handling() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15437, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15437).await;
    let err = client
        .simple_query("SELECT * FROM nonexistent")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("nonexistent")
            || msg.to_lowercase().contains("error")
            || msg.contains("XX000"),
        "error should indicate failure: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Extended Query Protocol tests (JDBC/DBeaver path)
//
// tokio_postgres::Client::query() uses Parse/Bind/Execute (extended protocol).
// These tests verify that the ExtendedQueryHandler works correctly.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extended_select_count() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15440, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15440).await;
    // query() uses extended protocol (Parse/Bind/Execute)
    let rows = client.query("SELECT COUNT(*) FROM t", &[]).await.unwrap();
    assert_eq!(rows.len(), 1);
    // Extended protocol returns text-format values since we return NoData for Describe
    let count: &str = rows[0].get(0);
    assert_eq!(count, "5");
}

#[tokio::test]
async fn extended_group_by() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15441, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15441).await;
    let rows = client
        .query(
            "SELECT name, SUM(value) AS total FROM t GROUP BY name ORDER BY total DESC",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 3);
    let name0: &str = rows[0].get(0);
    let total0: &str = rows[0].get(1);
    assert_eq!(name0, "bob");
    assert_eq!(total0, "70");
}

#[tokio::test]
async fn extended_catalog_set() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15442, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15442).await;
    // JDBC sends SET during handshake via extended protocol
    client
        .execute("SET extra_float_digits = 3", &[])
        .await
        .unwrap();
    client
        .execute("SET application_name = 'DBeaver'", &[])
        .await
        .unwrap();
    // If we get here without error, the extended protocol handled SET correctly
}

#[tokio::test]
async fn extended_select_constant() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15443, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15443).await;
    // Health-check ping via extended protocol
    let rows = client.query("SELECT 1", &[]).await.unwrap();
    assert_eq!(rows.len(), 1);
    let val: &str = rows[0].get(0);
    assert_eq!(val, "1");
}

#[tokio::test]
async fn extended_error_handling() {
    let _lock = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let csv = create_test_csv();
    let _server = start_server(15444, csv.path().to_str().unwrap());
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let client = connect(15444).await;
    let err = client
        .query("SELECT * FROM nonexistent", &[])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("nonexistent")
            || msg.to_lowercase().contains("error")
            || msg.contains("XX000"),
        "extended protocol error should indicate failure: {msg}"
    );
}
