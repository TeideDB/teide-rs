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

mod completer;
mod highlighter;
mod prompt;
mod theme;
mod validator;

use clap::Parser;
use std::path::PathBuf;

use reedline::{
    default_emacs_keybindings, DescriptionMode, Emacs, FileBackedHistory, IdeMenu, KeyCode,
    KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal,
};

use completer::{ColumnInfo, CompletionUpdater, SqlCompleter, TableInfo};
use highlighter::SqlHighlighter;
use prompt::SqlPrompt;
use validator::SqlValidator;

#[derive(Parser)]
#[command(name = "teide", version, about = "Fast SQL engine powered by Teide")]
struct Args {
    /// SQL query or CSV file path
    input: Option<String>,
    /// Execute SQL from file
    #[arg(short, long)]
    file: Option<PathBuf>,
    /// Execute SQL init script before entering REPL
    #[arg(short, long)]
    init: Option<PathBuf>,
    /// Show query execution time
    #[arg(short, long)]
    timer: bool,
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Table,
    Csv,
    Json,
}

fn main() {
    let args = Args::parse();

    // Non-interactive: execute single query (with optional --init)
    if let Some(ref input) = args.input {
        if !input.ends_with(".csv") {
            if let Some(ref init) = args.init {
                let mut session = make_session();
                run_init_script(&mut session, init);
                run_session_query(&mut session, input);
            } else {
                run_single_query(input);
            }
            return;
        }
    }

    // Non-interactive: execute SQL from file (with optional --init)
    if let Some(ref file) = args.file {
        run_sql_file(file, args.init.as_deref(), args.timer);
        return;
    }

    // Interactive REPL
    run_repl(args.input.as_deref(), args.init.as_deref());
}

fn make_session() -> teide::sql::Session {
    match teide::sql::Session::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "{}Error: failed to init Teide engine: {e}{}",
                theme::ERROR,
                theme::R
            );
            std::process::exit(1);
        }
    }
}

fn run_init_script(session: &mut teide::sql::Session, path: &PathBuf) {
    if let Err(e) = session.execute_script_file(path.as_path()) {
        eprintln!(
            "{}Error in init script {}: {e}{}",
            theme::ERROR,
            path.display(),
            theme::R
        );
        std::process::exit(1);
    }
}

fn run_single_query(sql: &str) {
    let ctx = match teide::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!(
                "{}Error: failed to init Teide engine: {e}{}",
                theme::ERROR,
                theme::R
            );
            std::process::exit(1);
        }
    };
    match teide::sql::execute_sql(&ctx, sql) {
        Ok(result) => print_result(&result, &OutputFormat::Table),
        Err(e) => {
            eprintln!("{}Error: {e}{}", theme::ERROR, theme::R);
            std::process::exit(1);
        }
    }
}

fn run_session_query(session: &mut teide::sql::Session, sql: &str) {
    match session.execute(sql) {
        Ok(teide::sql::ExecResult::Query(result)) => print_result(&result, &OutputFormat::Table),
        Ok(teide::sql::ExecResult::Ddl(msg)) => println!("{}{msg}{}", theme::SUCCESS, theme::R),
        Err(e) => {
            eprintln!("{}Error: {e}{}", theme::ERROR, theme::R);
            std::process::exit(1);
        }
    }
}

fn run_sql_file(path: &PathBuf, init: Option<&std::path::Path>, timer: bool) {
    let mut session = make_session();
    if let Some(init_path) = init {
        if let Err(e) = session.execute_script_file(init_path) {
            eprintln!(
                "{}Error in init script {}: {e}{}",
                theme::ERROR,
                init_path.display(),
                theme::R
            );
            std::process::exit(1);
        }
    }
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {e}", path.display());
        std::process::exit(1);
    });
    let dialect = sqlparser::dialect::DuckDbDialect {};
    let stmts = match sqlparser::parser::Parser::parse_sql(&dialect, &contents) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "{}Error parsing {}: {e}{}",
                theme::ERROR,
                path.display(),
                theme::R
            );
            std::process::exit(1);
        }
    };
    for stmt in &stmts {
        let sql = stmt.to_string();
        let sql = sql.trim();
        if sql.is_empty() {
            continue;
        }
        let start = std::time::Instant::now();
        match session.execute(sql) {
            Ok(teide::sql::ExecResult::Query(result)) => {
                print_result(&result, &OutputFormat::Table);
                if timer {
                    let elapsed = start.elapsed();
                    println!("{}Run Time: {elapsed:.3?}{}", theme::FOOTER, theme::R);
                }
            }
            Ok(teide::sql::ExecResult::Ddl(msg)) => {
                println!("{}{msg}{}", theme::SUCCESS, theme::R);
                if timer {
                    let elapsed = start.elapsed();
                    println!("{}Run Time: {elapsed:.3?}{}", theme::FOOTER, theme::R);
                }
            }
            Err(e) => eprintln!("{}Error: {e}{}", theme::ERROR, theme::R),
        }
    }
}

fn run_repl(preload_csv: Option<&str>, init: Option<&std::path::Path>) {
    let mut session = make_session();

    print_banner();

    if let Some(init_path) = init {
        if let Err(e) = session.execute_script_file(init_path) {
            eprintln!(
                "{}Error in init script {}: {e}{}",
                theme::ERROR,
                init_path.display(),
                theme::R
            );
            return;
        }
    }

    if let Some(csv_path) = preload_csv {
        let sql = format!("CREATE TABLE t AS SELECT * FROM '{csv_path}'");
        match session.execute(&sql) {
            Ok(teide::sql::ExecResult::Ddl(msg)) => println!("{msg}"),
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error loading {csv_path}: {e}");
                return;
            }
        }
    }

    // --- Components ---
    let (completer, comp_updater) = SqlCompleter::new();

    let hinter = reedline::DefaultHinter::default()
        .with_style(nu_ansi_term::Style::new().fg(nu_ansi_term::Color::DarkGray));

    // --- IdeMenu with description panel ---
    let ide_menu = IdeMenu::default()
        .with_name("completion_menu")
        .with_min_completion_width(20)
        .with_max_completion_width(60)
        .with_max_completion_height(10)
        .with_padding(1)
        .with_description_mode(DescriptionMode::PreferRight)
        .with_min_description_width(20)
        .with_max_description_width(40)
        .with_default_border();

    // --- Keybindings ---
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );
    // Shift+Tab to go backwards through menu
    keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::BackTab,
        ReedlineEvent::MenuPrevious,
    );

    // --- History ---
    let history_path = dirs_or_home().join(".teide_history");
    let history = match FileBackedHistory::with_file(1000, history_path.clone()) {
        Ok(history) => Some(history),
        Err(e) => {
            eprintln!(
                "{}Warning: history disabled ({}): {}{}",
                theme::ERROR,
                history_path.display(),
                e,
                theme::R
            );
            None
        }
    };

    // --- Assemble editor ---
    let editor = Reedline::create()
        .with_completer(Box::new(completer))
        .with_highlighter(Box::new(SqlHighlighter))
        .with_validator(Box::new(SqlValidator))
        .with_hinter(Box::new(hinter))
        .with_menu(ReedlineMenu::EngineCompleter(Box::new(ide_menu)))
        .with_edit_mode(Box::new(Emacs::new(keybindings)));
    let mut editor = if let Some(history) = history {
        editor.with_history(Box::new(history))
    } else {
        editor
    };

    let prompt = SqlPrompt;
    let mut format = OutputFormat::Table;
    let mut show_timer = false;

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Dot commands
                if trimmed.starts_with('.') {
                    handle_dot_command(trimmed, &mut format, &mut show_timer, &session);
                    continue;
                }

                // SQL statement (Validator ensured it ends with ';')
                let sql = trimmed.trim_end_matches(';').trim();
                if sql.is_empty() {
                    continue;
                }

                let start = std::time::Instant::now();
                match session.execute(sql) {
                    Ok(teide::sql::ExecResult::Query(result)) => {
                        let elapsed = start.elapsed();
                        update_columns(&comp_updater, &result);
                        print_result(&result, &format);
                        if show_timer {
                            eprintln!(
                                "{}Run Time: {:.3}s{}",
                                theme::TIMER,
                                elapsed.as_secs_f64(),
                                theme::R
                            );
                        }
                    }
                    Ok(teide::sql::ExecResult::Ddl(msg)) => {
                        let elapsed = start.elapsed();
                        println!("{}{msg}{}", theme::SUCCESS, theme::R);
                        update_tables(&comp_updater, &session);
                        if show_timer {
                            eprintln!(
                                "{}Run Time: {:.3}s{}",
                                theme::TIMER,
                                elapsed.as_secs_f64(),
                                theme::R
                            );
                        }
                    }
                    Err(e) => eprintln!("{}Error: {e}{}", theme::ERROR, theme::R),
                }
            }
            Ok(Signal::CtrlD) => break,
            Ok(Signal::CtrlC) => continue,
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Completer updates (via shared Arc<Mutex<>> state)
// ---------------------------------------------------------------------------

fn update_columns(updater: &CompletionUpdater, result: &teide::sql::SqlResult) {
    let table = &result.table;
    let ncols = table.ncols() as usize;
    let mut columns = Vec::with_capacity(ncols);
    for i in 0..ncols {
        columns.push(ColumnInfo {
            name: if i < result.columns.len() {
                result.columns[i].clone()
            } else {
                table.col_name_str(i).to_string()
            },
            type_name: type_name(table.col_type(i)).to_string(),
            table_name: String::new(),
        });
    }
    updater.set_columns(columns);
}

fn update_tables(updater: &CompletionUpdater, session: &teide::sql::Session) {
    let names = session.table_names();
    let tables: Vec<TableInfo> = names
        .iter()
        .filter_map(|name| {
            session.table_info(name).map(|(nrows, ncols)| TableInfo {
                name: name.to_string(),
                nrows: nrows as usize,
                ncols,
            })
        })
        .collect();
    updater.set_tables(tables);
}

// ---------------------------------------------------------------------------
// Column index resolution
// ---------------------------------------------------------------------------

fn resolve_col_indices(result: &teide::sql::SqlResult) -> Vec<usize> {
    let table = &result.table;
    let ncols = table.ncols() as usize;

    let mut name_to_idx: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for i in 0..ncols {
        let name = table.col_name_str(i).to_lowercase();
        if !name.is_empty() {
            name_to_idx.insert(name, i);
        }
    }

    let mut indices = Vec::new();
    for (pos, col_name) in result.columns.iter().enumerate() {
        if let Some(&idx) = name_to_idx.get(&col_name.to_lowercase()) {
            indices.push(idx);
        } else if pos < ncols {
            indices.push(pos);
        }
    }

    if indices.is_empty() {
        indices = (0..ncols).collect();
    }

    indices
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_result(result: &teide::sql::SqlResult, format: &OutputFormat) {
    match format {
        OutputFormat::Table => print_table(result),
        OutputFormat::Csv => print_csv(result),
        OutputFormat::Json => print_json(result),
    }
}

const HEAD_ROWS: usize = 20;
const TAIL_ROWS: usize = 20;

fn type_name(typ: i8) -> &'static str {
    match typ {
        1 => "bool",
        4 => "i16",
        5 => "i32",
        6 => "i64",
        7 => "f64",
        9 => "date",
        10 => "time",
        11 => "timestamp",
        20 => "sym",
        15 => "enum",
        _ => "?",
    }
}

fn print_table(result: &teide::sql::SqlResult) {
    use std::fmt::Write;
    use std::io::Write as IoWrite;
    use theme::*;
    let mut out = String::with_capacity(4096);

    let table = &result.table;
    let nrows = table.nrows() as usize;
    let col_indices = resolve_col_indices(result);
    let ncols = col_indices.len();

    if ncols == 0 {
        println!("{FOOTER}(empty result){R}");
        return;
    }

    let col_names: Vec<String> = col_indices
        .iter()
        .enumerate()
        .map(|(pos, &idx)| {
            if pos < result.columns.len() {
                result.columns[pos].clone()
            } else {
                table.col_name_str(idx).to_string()
            }
        })
        .collect();

    let col_types: Vec<&str> = col_indices
        .iter()
        .map(|&idx| type_name(table.col_type(idx)))
        .collect();

    let is_right: Vec<bool> = col_indices
        .iter()
        .map(|&idx| matches!(table.col_type(idx), 1 | 4 | 5 | 6 | 7))
        .collect();

    let show_dots = nrows > HEAD_ROWS + TAIL_ROWS;
    let head_n = if show_dots { HEAD_ROWS } else { nrows };
    let tail_n = if show_dots { TAIL_ROWS } else { 0 };
    let shown = head_n + tail_n;

    let mut cells: Vec<Vec<String>> = Vec::with_capacity(shown + 3);
    let mut is_null: Vec<Vec<bool>> = Vec::with_capacity(shown + 3);
    for r in 0..head_n {
        let mut row = Vec::with_capacity(ncols);
        let mut nulls = Vec::with_capacity(ncols);
        for &c in &col_indices {
            let val = format_cell(table, c, r);
            nulls.push(val == "NULL");
            row.push(val);
        }
        cells.push(row);
        is_null.push(nulls);
    }
    if show_dots {
        cells.push(
            (0..ncols)
                .map(|_| "\u{00b7}\u{00b7}\u{00b7}".to_string())
                .collect(),
        );
        is_null.push(vec![false; ncols]);
        for r in (nrows - tail_n)..nrows {
            let mut row = Vec::with_capacity(ncols);
            let mut nulls = Vec::with_capacity(ncols);
            for &c in &col_indices {
                let val = format_cell(table, c, r);
                nulls.push(val == "NULL");
                row.push(val);
            }
            cells.push(row);
            is_null.push(nulls);
        }
    }

    let footer_left = if show_dots {
        format!("{nrows} rows ({shown} shown)")
    } else {
        format!("{nrows} rows")
    };
    let footer_right = format!("{ncols} columns");
    let footer_min = footer_left.len() + footer_right.len() + 3;

    let mut w: Vec<usize> = (0..ncols)
        .map(|c| {
            let mut max = col_names[c].len().max(col_types[c].len());
            for row in &cells {
                max = max.max(row[c].len());
            }
            max
        })
        .collect();

    let mut inner_width: usize = w.iter().map(|x| x + 2).sum::<usize>() + ncols - 1;

    if inner_width < footer_min {
        let extra = footer_min - inner_width;
        w[ncols - 1] += extra;
        inner_width += extra;
    }
    let mut buf = String::with_capacity(inner_width * 2);

    macro_rules! hline {
        ($left:expr, $mid:expr, $right:expr) => {{
            buf.clear();
            buf.push_str(BORDER);
            buf.push($left);
            for c in 0..ncols {
                if c > 0 {
                    buf.push($mid);
                }
                for _ in 0..w[c] + 2 {
                    buf.push('\u{2500}');
                }
            }
            buf.push($right);
            buf.push_str(R);
            let _ = writeln!(out, "{buf}");
        }};
    }

    hline!('\u{250c}', '\u{252c}', '\u{2510}');

    buf.clear();
    for c in 0..ncols {
        buf.push_str(BORDER);
        buf.push('\u{2502}');
        buf.push_str(R);
        let _ = write!(
            buf,
            " {BOLD}{HEADER}{:^width$}{R} ",
            col_names[c],
            width = w[c]
        );
    }
    buf.push_str(BORDER);
    buf.push('\u{2502}');
    buf.push_str(R);
    let _ = writeln!(out, "{buf}");

    buf.clear();
    for c in 0..ncols {
        buf.push_str(BORDER);
        buf.push('\u{2502}');
        buf.push_str(R);
        let _ = write!(buf, " {TYPE_DIM}{:^width$}{R} ", col_types[c], width = w[c]);
    }
    buf.push_str(BORDER);
    buf.push('\u{2502}');
    buf.push_str(R);
    let _ = writeln!(out, "{buf}");

    hline!('\u{251c}', '\u{253c}', '\u{2524}');

    let dots_idx = if show_dots { Some(head_n) } else { None };
    for (ri, row) in cells.iter().enumerate() {
        buf.clear();
        let is_dots = dots_idx == Some(ri);
        for c in 0..ncols {
            buf.push_str(BORDER);
            buf.push('\u{2502}');
            buf.push_str(R);
            if is_dots {
                let _ = write!(buf, " {FOOTER}{:^width$}{R} ", row[c], width = w[c]);
            } else if is_null[ri][c] {
                let _ = write!(
                    buf,
                    " {ITALIC}{NULL_CLR}{:>width$}{R} ",
                    row[c],
                    width = w[c]
                );
            } else if is_right[c] {
                let _ = write!(buf, " {TEXT}{:>width$}{R} ", row[c], width = w[c]);
            } else {
                let _ = write!(buf, " {TEXT}{:<width$}{R} ", row[c], width = w[c]);
            }
        }
        buf.push_str(BORDER);
        buf.push('\u{2502}');
        buf.push_str(R);
        let _ = writeln!(out, "{buf}");
    }

    hline!('\u{251c}', '\u{2534}', '\u{2524}');

    let pad = inner_width - footer_left.len() - footer_right.len() - 2;
    buf.clear();
    let _ = write!(
        buf,
        "{BORDER}\u{2502}{R} {FOOTER}{footer_left}{:pad$}{footer_right}{R} {BORDER}\u{2502}{R}",
        ""
    );
    let _ = writeln!(out, "{buf}");

    buf.clear();
    buf.push_str(BORDER);
    buf.push('\u{2514}');
    for _ in 0..inner_width {
        buf.push('\u{2500}');
    }
    buf.push('\u{2518}');
    buf.push_str(R);
    let _ = writeln!(out, "{buf}");

    // Single write to stdout — avoids per-line flush overhead on terminals
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
}

fn print_csv(result: &teide::sql::SqlResult) {
    let table = &result.table;
    let col_indices = resolve_col_indices(result);

    let headers: Vec<String> = col_indices
        .iter()
        .enumerate()
        .map(|(pos, &_idx)| {
            if pos < result.columns.len() {
                result.columns[pos].clone()
            } else {
                table.col_name_str(_idx).to_string()
            }
        })
        .collect();
    println!("{}", headers.join(","));

    for row in 0..table.nrows() as usize {
        let cells: Vec<String> = col_indices
            .iter()
            .map(|&col| format_cell(table, col, row))
            .collect();
        println!("{}", cells.join(","));
    }
}

fn print_json(result: &teide::sql::SqlResult) {
    let table = &result.table;
    let nrows = table.nrows() as usize;
    let col_indices = resolve_col_indices(result);

    let headers: Vec<String> = col_indices
        .iter()
        .enumerate()
        .map(|(pos, &_idx)| {
            if pos < result.columns.len() {
                result.columns[pos].clone()
            } else {
                table.col_name_str(_idx).to_string()
            }
        })
        .collect();

    println!("[");
    for row in 0..nrows {
        let pairs: Vec<String> = col_indices
            .iter()
            .enumerate()
            .map(|(i, &col)| format!("\"{}\": {}", headers[i], format_json_value(table, col, row)))
            .collect();
        let comma = if row + 1 < nrows { "," } else { "" };
        println!("  {{{}}}{comma}", pairs.join(", "));
    }
    println!("]");
}

fn format_cell(table: &teide::Table, col: usize, row: usize) -> String {
    let typ = table.col_type(col);
    match typ {
        9 => match table.get_i64(col, row) {
            Some(d) => teide::Table::format_date(d as i32),
            None => "NULL".to_string(),
        },
        4..=6 => match table.get_i64(col, row) {
            Some(v) => format!("{v}"),
            None => "NULL".to_string(),
        },
        7 => match table.get_f64(col, row) {
            Some(v) => {
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
        _ => "<unsupported>".to_string(),
    }
}

fn format_json_value(table: &teide::Table, col: usize, row: usize) -> String {
    let typ = table.col_type(col);
    match typ {
        9 => match table.get_i64(col, row) {
            Some(d) => format!("\"{}\"", teide::Table::format_date(d as i32)),
            None => "null".to_string(),
        },
        4..=6 => match table.get_i64(col, row) {
            Some(v) => format!("{v}"),
            None => "null".to_string(),
        },
        7 => match table.get_f64(col, row) {
            Some(v) => format!("{v}"),
            None => "null".to_string(),
        },
        1 => match table.get_i64(col, row) {
            Some(v) => {
                if v != 0 {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            None => "null".to_string(),
        },
        15 | 20 => match table.get_str(col, row) {
            Some(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        },
        _ => "null".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Dot commands
// ---------------------------------------------------------------------------

fn handle_dot_command(
    cmd: &str,
    format: &mut OutputFormat,
    timer: &mut bool,
    session: &teide::sql::Session,
) {
    use theme::*;

    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts[0] {
        ".mode" => {
            if parts.len() < 2 {
                println!("{FOOTER}Usage: .mode table|csv|json{R}");
                return;
            }
            match parts[1] {
                "table" => {
                    *format = OutputFormat::Table;
                    println!("{SUCCESS}Output mode: table{R}");
                }
                "csv" => {
                    *format = OutputFormat::Csv;
                    println!("{SUCCESS}Output mode: csv{R}");
                }
                "json" => {
                    *format = OutputFormat::Json;
                    println!("{SUCCESS}Output mode: json{R}");
                }
                _ => println!("{ERROR}Unknown mode. Use: table, csv, json{R}"),
            }
        }
        ".timer" => {
            if parts.len() < 2 {
                println!("{FOOTER}Usage: .timer on|off{R}");
                return;
            }
            match parts[1] {
                "on" => {
                    *timer = true;
                    println!("{SUCCESS}Timer: on{R}");
                }
                "off" => {
                    *timer = false;
                    println!("{SUCCESS}Timer: off{R}");
                }
                _ => println!("{FOOTER}Usage: .timer on|off{R}"),
            }
        }
        ".tables" => {
            let names = session.table_names();
            if names.is_empty() {
                println!("{FOOTER}No stored tables.{R}");
            } else {
                let mut sorted = names;
                sorted.sort();
                for name in sorted {
                    if let Some((nrows, ncols)) = session.table_info(name) {
                        println!("  {HEADER}{name:20}{R} {FOOTER}{nrows} rows, {ncols} cols{R}");
                    }
                }
            }
        }
        ".mem" => {
            let s = teide::mem_stats();
            let arena_cur = s.bytes_allocated + s.direct_bytes;
            let total_cur = arena_cur + s.sys_current;
            println!("{HEADER}Memory usage:{R}");
            println!("  {NORD7}Arena{R}    {FOOTER}{:>10}{R}  (peak {})  [{} allocs, {} frees, {} slab hits]",
                fmt_bytes(arena_cur), fmt_bytes(s.peak_bytes),
                s.alloc_count, s.free_count, s.slab_hits);
            println!(
                "  {NORD7}Direct{R}   {FOOTER}{:>10}{R}  [{} active mmaps]",
                fmt_bytes(s.direct_bytes),
                s.direct_count
            );
            println!(
                "  {NORD7}System{R}   {FOOTER}{:>10}{R}  (peak {})",
                fmt_bytes(s.sys_current),
                fmt_bytes(s.sys_peak)
            );
            println!(
                "  {NORD7}Total{R}    {FOOTER}{:>10}{R}",
                fmt_bytes(total_cur)
            );
        }
        ".help" => print_help(),
        // process::exit skips destructors; acceptable since OS reclaims all resources.
        ".quit" => std::process::exit(0),
        _ => println!(
            "{ERROR}Unknown command: {}. Type .help for commands.{R}",
            parts[0]
        ),
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn fmt_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ---------------------------------------------------------------------------
// Banner & help
// ---------------------------------------------------------------------------

fn print_banner() {
    use theme::*;

    let ver = env!("CARGO_PKG_VERSION");
    let hash = env!("GIT_HASH");
    let arch = std::env::consts::ARCH;
    let tag = format!("v{}  \u{b7}  {}  \u{b7}  {}", ver, hash, arch);
    let help = "type .help for commands";
    let tag_w = tag.chars().count();
    let help_w = help.chars().count();
    let w = tag_w.max(help_w);
    let fill = w.saturating_sub(11);
    println!(
        "{BAN_BORDER}\u{256d}\u{2500} {BOLD}{BAN_TITLE}Teide SQL{R}{BAN_BORDER} \u{2500}{}\u{256e}{R}",
        "\u{2500}".repeat(fill)
    );
    println!(
        "{BAN_BORDER}\u{2502}{R} {BAN_INFO}{}{}{R} {BAN_BORDER}\u{2502}{R}",
        tag,
        " ".repeat(w - tag_w)
    );
    println!(
        "{BAN_BORDER}\u{2502}{R} {BAN_HELP}{}{}{R} {BAN_BORDER}\u{2502}{R}",
        help,
        " ".repeat(w - help_w)
    );
    println!(
        "{BAN_BORDER}\u{2570}{}\u{256f}{R}",
        "\u{2500}".repeat(w + 2)
    );
    println!();
}

fn print_help() {
    use theme::*;

    println!("{BOLD}{HEADER}Commands:{R}");
    println!("  {NORD7}.mode table|csv|json{R}  {NORD3}Set output format{R}");
    println!("  {NORD7}.tables{R}               {NORD3}List stored tables{R}");
    println!("  {NORD7}.timer on|off{R}         {NORD3}Show query execution time{R}");
    println!("  {NORD7}.mem{R}                  {NORD3}Show memory usage{R}");
    println!("  {NORD7}.help{R}                 {NORD3}Show this help{R}");
    println!("  {NORD7}.quit{R}                 {NORD3}Exit{R}");
    println!();
    println!("{BOLD}{HEADER}SQL:{R}");
    println!("  {TEXT}SELECT id1, SUM(v1) FROM 'data.csv' GROUP BY id1;{R}");
    println!("  {TEXT}CREATE TABLE t AS SELECT * FROM 'data.csv' WHERE id1 = 'id016';{R}");
    println!("  {TEXT}SELECT * FROM t GROUP BY id1;{R}");
    println!("  {TEXT}DROP TABLE t;{R}");
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
