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

use reedline::{ValidationResult, Validator};

pub struct SqlValidator;

impl Validator for SqlValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        let trimmed = line.trim();

        // Empty input or dot commands are always complete
        if trimmed.is_empty() || trimmed.starts_with('.') {
            return ValidationResult::Complete;
        }

        // Unbalanced parentheses → incomplete (skip parens inside quotes)
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut in_dquote = false;
        for ch in trimmed.chars() {
            match ch {
                '\'' if !in_dquote => in_string = !in_string,
                '"' if !in_string => in_dquote = !in_dquote,
                '(' if !in_string && !in_dquote => depth += 1,
                ')' if !in_string && !in_dquote => depth -= 1,
                _ => {}
            }
        }
        if depth > 0 {
            return ValidationResult::Incomplete;
        }

        // SQL must end with semicolon
        if !trimmed.ends_with(';') {
            return ValidationResult::Incomplete;
        }

        ValidationResult::Complete
    }
}
