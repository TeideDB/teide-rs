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

use nu_ansi_term::{Color, Style};
use reedline::{Highlighter, StyledText};

const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "BY",
    "ORDER",
    "LIMIT",
    "AS",
    "ON",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "CROSS",
    "HAVING",
    "DISTINCT",
    "UNION",
    "ALL",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "INDEX",
    "VIEW",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "IN",
    "BETWEEN",
    "LIKE",
    "IS",
    "NULL",
    "EXISTS",
    "ASC",
    "DESC",
    "OFFSET",
    "FETCH",
    "WITH",
    "RECURSIVE",
    "EXCEPT",
    "INTERSECT",
    "OVER",
    "PARTITION",
    "WINDOW",
    "ROWS",
    "RANGE",
    "UNBOUNDED",
    "PRECEDING",
    "FOLLOWING",
    "CURRENT",
    "ROW",
];

const AGG_FUNCTIONS: &[&str] = &[
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "COUNT",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "NTILE",
    "LAG",
    "LEAD",
];

const OPERATORS: &[&str] = &["AND", "OR", "NOT"];

fn keyword_style() -> Style {
    Style::new().bold().fg(Color::Blue)
}
fn function_style() -> Style {
    Style::new().bold().fg(Color::Cyan)
}
fn string_style() -> Style {
    Style::new().fg(Color::Yellow)
}
fn number_style() -> Style {
    Style::new().fg(Color::Magenta)
}
fn operator_style() -> Style {
    Style::new().bold().fg(Color::Blue)
}
fn dot_cmd_style() -> Style {
    Style::new().fg(Color::Cyan)
}

pub struct SqlHighlighter;

impl Highlighter for SqlHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();

        // Dot commands — style the whole line
        if line.starts_with('.') {
            styled.push((dot_cmd_style(), line.to_string()));
            return styled;
        }

        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            let b = bytes[i];

            // Single-quoted string literal
            if b == b'\'' {
                let start = i;
                i += 1;
                while i < len && bytes[i] != b'\'' {
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
                styled.push((string_style(), line[start..i].to_string()));
                continue;
            }

            // Number literal
            if b.is_ascii_digit() || (b == b'-' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
                let start = i;
                if b == b'-' {
                    i += 1;
                }
                while i < len && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                // Only color as number if not followed by letter (otherwise part of identifier)
                if i < len && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
                    styled.push((Style::default(), line[start..i].to_string()));
                } else {
                    styled.push((number_style(), line[start..i].to_string()));
                }
                continue;
            }

            // Word (identifier or keyword)
            if b.is_ascii_alphabetic() || b == b'_' {
                let start = i;
                while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &line[start..i];
                let upper = word.to_ascii_uppercase();

                if AGG_FUNCTIONS.contains(&upper.as_str()) {
                    styled.push((function_style(), word.to_string()));
                } else if OPERATORS.contains(&upper.as_str()) {
                    styled.push((operator_style(), word.to_string()));
                } else if SQL_KEYWORDS.contains(&upper.as_str()) {
                    styled.push((keyword_style(), word.to_string()));
                } else {
                    styled.push((Style::default(), word.to_string()));
                }
                continue;
            }

            // Comparison/assignment operators
            if b == b'=' || b == b'<' || b == b'>' || b == b'!' {
                let start = i;
                i += 1;
                if i < len && bytes[i] == b'=' {
                    i += 1;
                }
                styled.push((operator_style(), line[start..i].to_string()));
                continue;
            }

            // Everything else (whitespace, parens, commas, semicolons)
            styled.push((Style::default(), (b as char).to_string()));
            i += 1;
        }

        styled
    }
}
