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

//! Type mapping between Teide column types and PostgreSQL wire types,
//! plus text-protocol cell formatting.

use pgwire::api::Type;

use crate::ffi;

/// Map a Teide type tag (from `Table::col_type()`) to the corresponding
/// PostgreSQL wire-protocol `Type`.
pub fn teide_to_pg_type(td_type: i8) -> Type {
    match td_type {
        ffi::TD_BOOL => Type::BOOL,
        ffi::TD_I16 => Type::INT2,
        ffi::TD_I32 | ffi::TD_DATE => Type::INT4,
        ffi::TD_I64 | ffi::TD_TIME | ffi::TD_TIMESTAMP => Type::INT8,
        ffi::TD_F64 => Type::FLOAT8,
        ffi::TD_SYM => Type::VARCHAR,
        _ => Type::VARCHAR,
    }
}

/// Format a single cell value as a text-protocol string for the PG wire.
/// Returns `None` for NULL values.
pub fn format_cell(table: &crate::Table, col: usize, row: usize) -> Option<String> {
    let typ = table.col_type(col);
    match typ {
        ffi::TD_BOOL => {
            let v = table.get_i64(col, row)?;
            Some(if v != 0 {
                "t".to_string()
            } else {
                "f".to_string()
            })
        }
        ffi::TD_I16
        | ffi::TD_I32
        | ffi::TD_I64
        | ffi::TD_DATE
        | ffi::TD_TIME
        | ffi::TD_TIMESTAMP => {
            let v = table.get_i64(col, row)?;
            Some(v.to_string())
        }
        ffi::TD_F64 => {
            let v = table.get_f64(col, row)?;
            // Use enough precision to round-trip, but trim trailing zeros
            let s = format!("{v:.15}");
            let s = s.trim_end_matches('0');
            if s.ends_with('.') {
                Some(format!("{s}0"))
            } else {
                Some(s.to_string())
            }
        }
        ffi::TD_SYM => table.get_str(col, row),
        _ => Some("<unsupported>".to_string()),
    }
}
