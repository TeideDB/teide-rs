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

//! Encode query results into pgwire `QueryResponse` messages.

use std::sync::Arc;

use futures::stream;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo, QueryResponse};
use pgwire::api::Type;
use pgwire::error::PgWireResult;

use super::handler::WireResult;
use super::types::teide_to_pg_type;

/// Encode a pre-serialized `WireResult` (from the engine thread) into a
/// pgwire `QueryResponse` suitable for the text protocol.
///
/// All cell values have already been formatted as strings by the engine
/// thread, so this is purely a wire-encoding step — no C engine access.
///
/// When `all_text` is true, all column types are mapped to VARCHAR.
/// This is needed for the extended query protocol because clients like
/// tokio-postgres always request binary format and use type-specific
/// binary deserializers (`i64::from_be_bytes`, etc). Since we only
/// have pre-formatted text strings, mapping to VARCHAR makes the binary
/// encoding identical to text (raw UTF-8 bytes), which works correctly.
pub fn encode_wire_result(wr: &WireResult, all_text: bool) -> PgWireResult<QueryResponse> {
    let schema = Arc::new(
        wr.columns
            .iter()
            .map(|(name, td_type)| {
                let pg_type = if all_text {
                    Type::VARCHAR
                } else {
                    teide_to_pg_type(*td_type)
                };
                FieldInfo::new(name.clone(), None, None, pg_type, FieldFormat::Text)
            })
            .collect::<Vec<_>>(),
    );

    let ncols = schema.len();
    let mut rows = Vec::with_capacity(wr.rows.len());
    let mut encoder = DataRowEncoder::new(schema.clone());
    for row in &wr.rows {
        for c in 0..ncols {
            let cell = row.get(c).cloned().unwrap_or(None);
            encoder.encode_field(&cell)?;
        }
        rows.push(Ok(encoder.take_row()));
    }

    let row_stream = stream::iter(rows);
    Ok(QueryResponse::new(schema, row_stream))
}
