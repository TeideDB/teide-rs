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

use std::sync::{Arc, Mutex, MutexGuard};

use reedline::{Completer, Span, Suggestion};

// ---------------------------------------------------------------------------
// SQL keyword / function lists
// ---------------------------------------------------------------------------

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

const AGG_FUNCTIONS: &[(&str, &str)] = &[
    ("SUM", "SUM(col) \u{2192} sum of values"),
    ("AVG", "AVG(col) \u{2192} average"),
    ("MIN", "MIN(col) \u{2192} minimum"),
    ("MAX", "MAX(col) \u{2192} maximum"),
    ("COUNT", "COUNT(col) \u{2192} row count"),
    ("ROW_NUMBER", "ROW_NUMBER() OVER(...)"),
    ("RANK", "RANK() OVER(...)"),
    ("DENSE_RANK", "DENSE_RANK() OVER(...)"),
    ("NTILE", "NTILE(n) OVER(...)"),
    ("LAG", "LAG(col, offset) OVER(...)"),
    ("LEAD", "LEAD(col, offset) OVER(...)"),
];

const DOT_COMMANDS: &[(&str, &str)] = &[
    (".mode", "Set output format: table|csv|json"),
    (".tables", "List stored tables"),
    (".timer", "Show query time: on|off"),
    (".mem", "Show memory usage"),
    (".help", "Show available commands"),
    (".quit", "Exit the REPL"),
    (".exit", "Exit the REPL"),
];

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/// Subsequence fuzzy match. Returns (score, matched_indices) or None.
fn fuzzy_match(pattern: &str, candidate: &str) -> Option<(i32, Vec<usize>)> {
    let pat: Vec<char> = pattern.to_lowercase().chars().collect();
    let cand: Vec<char> = candidate.to_lowercase().chars().collect();
    let mut pi = 0;
    let mut indices = Vec::new();
    let mut score = 0i32;
    let mut prev_match = false;

    for (ci, &ch) in cand.iter().enumerate() {
        if pi < pat.len() && ch == pat[pi] {
            indices.push(ci);
            score += if prev_match { 3 } else { 1 }; // consecutive bonus
            if ci == 0 {
                score += 5;
            } // prefix bonus
            prev_match = true;
            pi += 1;
        } else {
            prev_match = false;
        }
    }

    if pi == pat.len() {
        Some((score, indices))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Completion context
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum CompletionContext {
    Table,
    Column,
    DotCommand,
    General,
}

fn detect_context(before: &str) -> CompletionContext {
    let upper = before.to_ascii_uppercase();

    if upper.trim_start().starts_with('.') {
        return CompletionContext::DotCommand;
    }

    let table_kws = ["FROM ", "JOIN "];
    let col_kws = [
        "SELECT ",
        "WHERE ",
        "BY ",
        "HAVING ",
        "ON ",
        "SET ",
        "ORDER BY ",
    ];

    let mut last_table: Option<usize> = None;
    let mut last_col: Option<usize> = None;

    for kw in &table_kws {
        if let Some(pos) = upper.rfind(kw) {
            last_table = Some(last_table.map_or(pos, |prev: usize| prev.max(pos)));
        }
    }
    for kw in &col_kws {
        if let Some(pos) = upper.rfind(kw) {
            last_col = Some(last_col.map_or(pos, |prev: usize| prev.max(pos)));
        }
    }

    match (last_table, last_col) {
        (Some(t), Some(c)) if t > c => CompletionContext::Table,
        (Some(_), None) => CompletionContext::Table,
        (_, Some(_)) => CompletionContext::Column,
        _ => CompletionContext::General,
    }
}

// ---------------------------------------------------------------------------
// Shared completion state (updated by REPL loop, read by completer)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub type_name: String,
    pub table_name: String,
}

#[derive(Clone)]
pub struct TableInfo {
    pub name: String,
    pub nrows: usize,
    pub ncols: usize,
}

#[derive(Default)]
pub struct CompletionState {
    pub columns: Vec<ColumnInfo>,
    pub tables: Vec<TableInfo>,
}

fn lock_completion_state(state: &Arc<Mutex<CompletionState>>) -> MutexGuard<'_, CompletionState> {
    match state.lock() {
        Ok(guard) => guard,
        // Intentionally recovers from poisoned mutex to keep completions working.
        // The completion state is best-effort metadata; stale data is acceptable.
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Handle held by the REPL loop to update completion metadata.
#[derive(Clone)]
pub struct CompletionUpdater {
    state: Arc<Mutex<CompletionState>>,
}

impl CompletionUpdater {
    pub fn set_columns(&self, columns: Vec<ColumnInfo>) {
        lock_completion_state(&self.state).columns = columns;
    }

    pub fn set_tables(&self, tables: Vec<TableInfo>) {
        lock_completion_state(&self.state).tables = tables;
    }
}

// ---------------------------------------------------------------------------
// SqlCompleter
// ---------------------------------------------------------------------------

pub struct SqlCompleter {
    state: Arc<Mutex<CompletionState>>,
    csv_files: Vec<String>,
}

impl SqlCompleter {
    pub fn new() -> (Self, CompletionUpdater) {
        let csv_files = std::fs::read_dir(".")
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".csv") {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        let state = Arc::new(Mutex::new(CompletionState::default()));
        let updater = CompletionUpdater {
            state: Arc::clone(&state),
        };
        (SqlCompleter { state, csv_files }, updater)
    }
}

impl Completer for SqlCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let before = &line[..pos];

        // Find the start of the current word
        let word_start = before
            .rfind(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == '.')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = &before[word_start..];

        let context = detect_context(before);

        // For dot commands, word_start is at the '.'
        if let CompletionContext::DotCommand = context {
            let dot_start = before.rfind('.').unwrap_or(0);
            let dot_prefix = &before[dot_start..];
            return complete_dot_commands(dot_prefix, dot_start, pos);
        }

        // Auto-popup threshold: need 1+ chars
        if prefix.is_empty() {
            return vec![];
        }

        let span = Span::new(word_start, pos);
        let mut candidates: Vec<(i32, Suggestion)> = Vec::new();

        // Read shared state
        let st = lock_completion_state(&self.state);

        match context {
            CompletionContext::Table => {
                for t in &st.tables {
                    if let Some((score, indices)) = fuzzy_match(prefix, &t.name) {
                        candidates.push((
                            score,
                            Suggestion {
                                value: t.name.clone(),
                                description: Some(format!(
                                    "table \u{2014} {} rows, {} cols",
                                    t.nrows, t.ncols
                                )),
                                style: None,
                                extra: None,
                                span,
                                append_whitespace: true,
                                match_indices: Some(indices),
                            },
                        ));
                    }
                }
                for f in &self.csv_files {
                    let display = format!("'{f}'");
                    if let Some((score, indices)) = fuzzy_match(prefix, f) {
                        candidates.push((
                            score,
                            Suggestion {
                                value: display,
                                description: Some("file".to_string()),
                                style: None,
                                extra: None,
                                span,
                                append_whitespace: true,
                                match_indices: Some(indices),
                            },
                        ));
                    }
                }
            }
            CompletionContext::Column => {
                for col in &st.columns {
                    if let Some((score, indices)) = fuzzy_match(prefix, &col.name) {
                        let desc = if col.table_name.is_empty() {
                            col.type_name.clone()
                        } else {
                            format!("{} \u{2014} {}", col.type_name, col.table_name)
                        };
                        candidates.push((
                            score,
                            Suggestion {
                                value: col.name.clone(),
                                description: Some(desc),
                                style: None,
                                extra: None,
                                span,
                                append_whitespace: false,
                                match_indices: Some(indices),
                            },
                        ));
                    }
                }
                add_functions(prefix, span, &mut candidates);
                add_keywords(prefix, span, &mut candidates);
            }
            CompletionContext::General => {
                for col in &st.columns {
                    if let Some((score, indices)) = fuzzy_match(prefix, &col.name) {
                        candidates.push((
                            score,
                            Suggestion {
                                value: col.name.clone(),
                                description: Some(col.type_name.clone()),
                                style: None,
                                extra: None,
                                span,
                                append_whitespace: false,
                                match_indices: Some(indices),
                            },
                        ));
                    }
                }
                add_functions(prefix, span, &mut candidates);
                add_keywords(prefix, span, &mut candidates);
                for t in &st.tables {
                    if let Some((score, indices)) = fuzzy_match(prefix, &t.name) {
                        candidates.push((
                            score,
                            Suggestion {
                                value: t.name.clone(),
                                description: Some(format!("table \u{2014} {} rows", t.nrows)),
                                style: None,
                                extra: None,
                                span,
                                append_whitespace: true,
                                match_indices: Some(indices),
                            },
                        ));
                    }
                }
            }
            CompletionContext::DotCommand => {}
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.0.cmp(&a.0));
        candidates.into_iter().map(|(_, s)| s).collect()
    }
}

fn complete_dot_commands(prefix: &str, dot_start: usize, pos: usize) -> Vec<Suggestion> {
    let span = Span::new(dot_start, pos);
    let mut results = Vec::new();
    for &(cmd, desc) in DOT_COMMANDS {
        if let Some((_, indices)) = fuzzy_match(prefix, cmd) {
            results.push(Suggestion {
                value: cmd.to_string(),
                description: Some(desc.to_string()),
                style: None,
                extra: None,
                span,
                append_whitespace: true,
                match_indices: Some(indices),
            });
        }
    }
    results
}

fn add_functions(prefix: &str, span: Span, candidates: &mut Vec<(i32, Suggestion)>) {
    for &(name, desc) in AGG_FUNCTIONS {
        if let Some((score, indices)) = fuzzy_match(prefix, name) {
            candidates.push((
                score,
                Suggestion {
                    value: name.to_string(),
                    description: Some(desc.to_string()),
                    style: None,
                    extra: None,
                    span,
                    append_whitespace: false,
                    match_indices: Some(indices),
                },
            ));
        }
    }
}

fn add_keywords(prefix: &str, span: Span, candidates: &mut Vec<(i32, Suggestion)>) {
    for &kw in SQL_KEYWORDS {
        if let Some((score, indices)) = fuzzy_match(prefix, kw) {
            candidates.push((
                score,
                Suggestion {
                    value: kw.to_string(),
                    description: Some("keyword".to_string()),
                    style: None,
                    extra: None,
                    span,
                    append_whitespace: true,
                    match_indices: Some(indices),
                },
            ));
        }
    }
}
