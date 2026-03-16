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
        ffi::TD_I32 => Type::INT4,
        ffi::TD_DATE => Type::DATE,
        ffi::TD_TIME => Type::TIME,
        ffi::TD_I64 => Type::INT8,
        ffi::TD_TIMESTAMP => Type::TIMESTAMP,
        ffi::TD_F64 => Type::FLOAT8,
        ffi::TD_F32 => Type::FLOAT4,
        ffi::TD_SYM => Type::VARCHAR,
        _ => Type::VARCHAR,
    }
}

/// Format a single cell value as a text-protocol string for the PG wire.
/// Returns `None` for NULL values.
///
/// `emb_dim` is the embedding dimension for this column (0 = not an
/// embedding).  For `dim > 1` the logical `row` is expanded into the
/// `dim` consecutive f32 values starting at physical index `row * dim`.
pub fn format_cell(table: &crate::Table, col: usize, row: usize, emb_dim: i32) -> Option<String> {
    let typ = table.col_type(col);
    match typ {
        ffi::TD_F32 if emb_dim > 1 => {
            let d = emb_dim as usize;
            let base = row * d;
            let mut parts = Vec::with_capacity(d);
            for i in 0..d {
                match table.get_f32(col, base + i) {
                    Some(v) => parts.push(format!("{v}")),
                    None => parts.push("NULL".to_string()),
                }
            }
            Some(format!("[{}]", parts.join(", ")))
        }
        ffi::TD_F32 => {
            let v = table.get_f32(col, row)?;
            Some(format!("{v}"))
        }
        ffi::TD_BOOL => {
            let v = table.get_i64(col, row)?;
            Some(if v != 0 {
                "t".to_string()
            } else {
                "f".to_string()
            })
        }
        ffi::TD_I16 | ffi::TD_I32 | ffi::TD_I64 => {
            let v = table.get_i64(col, row)?;
            Some(v.to_string())
        }
        ffi::TD_DATE => {
            let v = table.get_i64(col, row)?;
            Some(crate::Table::format_date(v as i32))
        }
        ffi::TD_TIME => {
            let v = table.get_i64(col, row)?;
            Some(crate::Table::format_time(v as i32))
        }
        ffi::TD_TIMESTAMP => {
            let v = table.get_i64(col, row)?;
            Some(crate::Table::format_timestamp(v))
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
