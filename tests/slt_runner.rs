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

//! sqllogictest harness for Teide SQL.

use std::sync::Mutex;

use teide::sql::{ExecResult, Session};

// The C engine uses global state — serialize all tests.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

struct TeideDb {
    session: Session,
}

// SAFETY: The C engine uses global state with thread-local arenas. All SLT tests
// are serialized by ENGINE_LOCK, so no concurrent access occurs. The `Send` bound
// is required by sqllogictest's Runner/AsyncDB but we never actually send across threads.
unsafe impl Send for TeideDb {}

#[derive(Debug)]
struct TeideError(String);

impl std::fmt::Display for TeideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TeideError {}

impl sqllogictest::DB for TeideDb {
    type Error = TeideError;
    type ColumnType = sqllogictest::DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<sqllogictest::DBOutput<Self::ColumnType>, Self::Error> {
        match self.session.execute(sql) {
            Ok(ExecResult::Ddl(_)) => Ok(sqllogictest::DBOutput::StatementComplete(0)),
            Ok(ExecResult::Query(r)) => {
                let types: Vec<_> = (0..r.columns.len())
                    .map(|c| match r.table.col_type(c) {
                        4..=6 => sqllogictest::DefaultColumnType::Integer,
                        7 => sqllogictest::DefaultColumnType::FloatingPoint,
                        _ => sqllogictest::DefaultColumnType::Text,
                    })
                    .collect();
                let rows: Vec<Vec<String>> = (0..r.table.nrows() as usize)
                    .map(|row| {
                        (0..r.columns.len())
                            .map(|col| format_cell(&r.table, col, row))
                            .collect()
                    })
                    .collect();
                Ok(sqllogictest::DBOutput::Rows { types, rows })
            }
            Err(e) => Err(TeideError(e.to_string())),
        }
    }
}

fn format_cell(table: &teide::Table, col: usize, row: usize) -> String {
    let typ = table.col_type(col);
    match typ {
        4..=6 => match table.get_i64(col, row) {
            Some(v) => format!("{v}"),
            None => "NULL".to_string(),
        },
        7 => match table.get_f64(col, row) {
            Some(v) => {
                if v.is_nan() {
                    return "NULL".to_string();
                }
                let s = format!("{v:.6}");
                let s = s.trim_end_matches('0');
                if s.ends_with('.') {
                    format!("{s}0")
                } else {
                    s.to_string()
                }
            }
            None => "NULL".to_string(),
        },
        1 => match table.get_i64(col, row) {
            Some(v) => {
                if v != 0 {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            None => "NULL".to_string(),
        },
        15 | 20 => match table.get_str(col, row) {
            Some(s) => s.to_string(),
            None => "NULL".to_string(),
        },
        _ => "NULL".to_string(),
    }
}

fn run_slt(path: &str) {
    let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut runner = sqllogictest::Runner::new(|| async {
        let session = Session::new().map_err(|e| TeideError(e.to_string()))?;
        Ok(TeideDb { session })
    });
    runner
        .run_file(path)
        .unwrap_or_else(|e| panic!("SLT test failed for {path}: {e}"));
}

#[test]
fn slt_insert() {
    run_slt("tests/slt/insert.slt");
}

#[test]
fn slt_basic() {
    run_slt("tests/slt/basic.slt");
}

#[test]
fn slt_aggregate() {
    run_slt("tests/slt/aggregate.slt");
}

#[test]
fn slt_join() {
    run_slt("tests/slt/join.slt");
}

#[test]
fn slt_window() {
    run_slt("tests/slt/window.slt");
}

#[test]
fn slt_set_ops() {
    run_slt("tests/slt/set_ops.slt");
}

#[test]
fn slt_subquery() {
    run_slt("tests/slt/subquery.slt");
}

#[test]
fn slt_functions() {
    run_slt("tests/slt/functions.slt");
}

#[test]
fn slt_types() {
    run_slt("tests/slt/types.slt");
}

#[test]
fn slt_ddl() {
    run_slt("tests/slt/ddl.slt");
}

#[test]
fn slt_sort() {
    run_slt("tests/slt/sort.slt");
}

#[test]
fn slt_where() {
    run_slt("tests/slt/where.slt");
}

#[test]
fn slt_agg_filter() {
    run_slt("tests/slt/agg_filter.slt");
}

#[test]
fn slt_nulls_order() {
    run_slt("tests/slt/nulls_order.slt");
}

#[test]
fn slt_stats() {
    run_slt("tests/slt/stats.slt");
}

#[test]
fn slt_regressions() {
    run_slt("tests/slt/regressions.slt");
}
