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

use std::path::PathBuf;

use reedline::{
    default_emacs_keybindings, Emacs, FileBackedHistory, Prompt, PromptEditMode,
    PromptHistorySearch, PromptHistorySearchStatus, Reedline, Signal,
};

use teide::ffi;
use teide::td::value::Value;
use teide::td::vm::Vm;

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

struct TdPrompt;

impl Prompt for TdPrompt {
    fn render_prompt_left(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("td)")
    }

    fn render_prompt_right(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(" ")
    }

    fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("  )")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> std::borrow::Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        std::borrow::Cow::Owned(format!("({}reverse-search: {}) ", prefix, history_search.term))
    }
}

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------

fn format_value(val: &Value) -> String {
    match val {
        Value::Nil => "(nil)".to_string(),
        Value::Str(s) => format!("\"{}\"", s),
        Value::Lambda(..) => "{lambda}".to_string(),
        Value::Dict { keys, vals } => format_dict(*keys, *vals),
        Value::Td(ptr) => {
            if ptr.is_null() || ffi::td_is_err(*ptr as *const ffi::td_t) {
                return "(nil)".to_string();
            }
            let type_tag = unsafe { ffi::td_type(*ptr as *const ffi::td_t) };
            if type_tag < 0 {
                // Atom
                format_atom(*ptr, type_tag)
            } else if type_tag == ffi::TD_TABLE {
                format_table(*ptr)
            } else if type_tag > 0 {
                // Vector
                format_vector(*ptr, type_tag)
            } else {
                // List (type 0)
                format_list(*ptr)
            }
        }
    }
}

fn format_atom(ptr: *mut ffi::td_t, type_tag: i8) -> String {
    unsafe {
        match type_tag {
            ffi::TD_ATOM_I64 => format!("{}", (*ptr).val.i64_),
            ffi::TD_ATOM_I32 => format!("{}", (*ptr).val.i32_),
            ffi::TD_ATOM_I16 => format!("{}", (*ptr).val.i16_),
            ffi::TD_ATOM_F64 => {
                let v = (*ptr).val.f64_;
                let s = format!("{:.6}", v);
                let s = s.trim_end_matches('0');
                if s.ends_with('.') {
                    format!("{}0", s)
                } else {
                    s.to_string()
                }
            }
            ffi::TD_ATOM_BOOL => {
                if (*ptr).val.b8 != 0 {
                    "1b".to_string()
                } else {
                    "0b".to_string()
                }
            }
            ffi::TD_ATOM_SYM => {
                let sym_id = (*ptr).val.i64_;
                let s_ptr = ffi::td_sym_str(sym_id);
                if s_ptr.is_null() || ffi::td_is_err(s_ptr as *const ffi::td_t) {
                    format!("`sym#{}", sym_id)
                } else {
                    let cstr = ffi::td_str_ptr(s_ptr);
                    if cstr.is_null() {
                        format!("`sym#{}", sym_id)
                    } else {
                        let rs = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                        format!("`{}", rs)
                    }
                }
            }
            ffi::TD_ATOM_STR => {
                let cstr = ffi::td_str_ptr(ptr);
                if cstr.is_null() {
                    "\"\"".to_string()
                } else {
                    let len = ffi::td_str_len(ptr);
                    let slice = std::slice::from_raw_parts(cstr as *const u8, len);
                    let s = String::from_utf8_lossy(slice);
                    format!("\"{}\"", s)
                }
            }
            _ => format!("<atom type={}>", type_tag),
        }
    }
}

fn format_vector(ptr: *mut ffi::td_t, type_tag: i8) -> String {
    unsafe {
        let len = ffi::td_len(ptr as *const ffi::td_t) as usize;
        if len == 0 {
            return match type_tag {
                ffi::TD_I64 => "`i64$()".to_string(),
                ffi::TD_F64 => "`f64$()".to_string(),
                ffi::TD_BOOL => "`bool$()".to_string(),
                ffi::TD_SYM => "`sym$()".to_string(),
                _ => format!("`type{}$()", type_tag),
            };
        }

        let data = ffi::td_data(ptr);
        let mut parts: Vec<String> = Vec::with_capacity(len.min(100));
        let show = len.min(100);

        match type_tag {
            ffi::TD_I64 => {
                let slice = std::slice::from_raw_parts(data as *const i64, len);
                for &v in &slice[..show] {
                    parts.push(format!("{}", v));
                }
            }
            ffi::TD_I32 => {
                let slice = std::slice::from_raw_parts(data as *const i32, len);
                for &v in &slice[..show] {
                    parts.push(format!("{}", v));
                }
            }
            ffi::TD_I16 => {
                let slice = std::slice::from_raw_parts(data as *const i16, len);
                for &v in &slice[..show] {
                    parts.push(format!("{}", v));
                }
            }
            ffi::TD_F64 => {
                let slice = std::slice::from_raw_parts(data as *const f64, len);
                for &v in &slice[..show] {
                    let s = format!("{:.6}", v);
                    let s = s.trim_end_matches('0');
                    if s.ends_with('.') {
                        parts.push(format!("{}0", s));
                    } else {
                        parts.push(s.to_string());
                    }
                }
            }
            ffi::TD_BOOL => {
                let slice = std::slice::from_raw_parts(data as *const u8, len);
                for &v in &slice[..show] {
                    parts.push(if v != 0 {
                        "1b".to_string()
                    } else {
                        "0b".to_string()
                    });
                }
            }
            ffi::TD_SYM => {
                let attrs = ffi::td_attrs(ptr as *const ffi::td_t);
                let raw_data = data as *const u8;
                for i in 0..show {
                    let sym_id = ffi::read_sym(raw_data, i, type_tag, attrs);
                    let s_ptr = ffi::td_sym_str(sym_id);
                    if s_ptr.is_null() || ffi::td_is_err(s_ptr as *const ffi::td_t) {
                        parts.push(format!("`sym#{}", sym_id));
                    } else {
                        let cstr = ffi::td_str_ptr(s_ptr);
                        if cstr.is_null() {
                            parts.push(format!("`sym#{}", sym_id));
                        } else {
                            let rs = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                            parts.push(format!("`{}", rs));
                        }
                    }
                }
            }
            _ => {
                for _ in 0..show {
                    parts.push("?".to_string());
                }
            }
        }

        if len > 100 {
            parts.push(format!("..({} more)", len - 100));
        }

        parts.join(" ")
    }
}

fn format_table(ptr: *mut ffi::td_t) -> String {
    unsafe {
        let ncols = ffi::td_table_ncols(ptr) as usize;
        let nrows = ffi::td_table_nrows(ptr) as usize;

        if ncols == 0 {
            return format!("(empty table: {} rows, {} cols)", nrows, ncols);
        }

        // Collect column names and data
        let mut col_names: Vec<String> = Vec::with_capacity(ncols);
        let mut col_data: Vec<Vec<String>> = Vec::with_capacity(ncols);

        let show_rows = nrows.min(20);

        for c in 0..ncols {
            // Column name
            let name_id = ffi::td_table_col_name(ptr, c as i64);
            let name_ptr = ffi::td_sym_str(name_id);
            let name = if name_ptr.is_null() || ffi::td_is_err(name_ptr as *const ffi::td_t) {
                format!("c{}", c)
            } else {
                let cstr = ffi::td_str_ptr(name_ptr);
                if cstr.is_null() {
                    format!("c{}", c)
                } else {
                    std::ffi::CStr::from_ptr(cstr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            col_names.push(name);

            // Column values
            let col_ptr = ffi::td_table_get_col_idx(ptr, c as i64);
            let mut cells: Vec<String> = Vec::with_capacity(show_rows);
            if col_ptr.is_null() || ffi::td_is_err(col_ptr as *const ffi::td_t) {
                for _ in 0..show_rows {
                    cells.push("NULL".to_string());
                }
            } else {
                let col_type = ffi::td_type(col_ptr as *const ffi::td_t);
                let data = ffi::td_data(col_ptr);
                for r in 0..show_rows {
                    cells.push(format_table_cell(col_ptr, col_type, data, r));
                }
            }
            col_data.push(cells);
        }

        // Compute column widths
        let mut widths: Vec<usize> = col_names.iter().map(|n| n.len()).collect();
        for (c, cells) in col_data.iter().enumerate() {
            for cell in cells {
                widths[c] = widths[c].max(cell.len());
            }
        }

        let mut out = String::new();

        // Header
        let header: Vec<String> = col_names
            .iter()
            .enumerate()
            .map(|(c, name)| format!("{:width$}", name, width = widths[c]))
            .collect();
        out.push_str(&header.join(" | "));
        out.push('\n');

        // Separator
        let sep: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
        out.push_str(&sep.join("-+-"));
        out.push('\n');

        // Rows
        for r in 0..show_rows {
            let row: Vec<String> = (0..ncols)
                .map(|c| format!("{:width$}", col_data[c][r], width = widths[c]))
                .collect();
            out.push_str(&row.join(" | "));
            out.push('\n');
        }

        if nrows > show_rows {
            out.push_str(&format!("...({} more rows)\n", nrows - show_rows));
        }

        out.push_str(&format!("{} rows x {} cols", nrows, ncols));
        out
    }
}

fn format_table_cell(
    col_ptr: *mut ffi::td_t,
    col_type: i8,
    data: *mut std::os::raw::c_void,
    row: usize,
) -> String {
    unsafe {
        match col_type {
            ffi::TD_I64 => {
                let slice = std::slice::from_raw_parts(data as *const i64, row + 1);
                format!("{}", slice[row])
            }
            ffi::TD_I32 => {
                let slice = std::slice::from_raw_parts(data as *const i32, row + 1);
                format!("{}", slice[row])
            }
            ffi::TD_I16 => {
                let slice = std::slice::from_raw_parts(data as *const i16, row + 1);
                format!("{}", slice[row])
            }
            ffi::TD_F64 => {
                let slice = std::slice::from_raw_parts(data as *const f64, row + 1);
                let v = slice[row];
                let s = format!("{:.6}", v);
                let s = s.trim_end_matches('0');
                if s.ends_with('.') {
                    format!("{}0", s)
                } else {
                    s.to_string()
                }
            }
            ffi::TD_BOOL => {
                let slice = std::slice::from_raw_parts(data as *const u8, row + 1);
                if slice[row] != 0 {
                    "1b".to_string()
                } else {
                    "0b".to_string()
                }
            }
            ffi::TD_SYM => {
                let attrs = ffi::td_attrs(col_ptr as *const ffi::td_t);
                let sym_id = ffi::read_sym(data as *const u8, row, col_type, attrs);
                let s_ptr = ffi::td_sym_str(sym_id);
                if s_ptr.is_null() || ffi::td_is_err(s_ptr as *const ffi::td_t) {
                    format!("sym#{}", sym_id)
                } else {
                    let cstr = ffi::td_str_ptr(s_ptr);
                    if cstr.is_null() {
                        format!("sym#{}", sym_id)
                    } else {
                        std::ffi::CStr::from_ptr(cstr)
                            .to_string_lossy()
                            .into_owned()
                    }
                }
            }
            _ => "?".to_string(),
        }
    }
}

fn format_dict(keys: *mut ffi::td_t, vals: *mut ffi::td_t) -> String {
    if keys.is_null() || vals.is_null() {
        return "(empty dict)".to_string();
    }

    unsafe {
        let key_val = Value::Td(keys);
        let val_val = Value::Td(vals);

        // Retain so the temporary Value wrappers don't double-free
        ffi::td_retain(keys);
        ffi::td_retain(vals);

        let key_len = ffi::td_len(keys as *const ffi::td_t) as usize;
        let val_type = ffi::td_type(vals as *const ffi::td_t);

        let mut out = String::new();
        let show = key_len.min(20);

        // Format keys
        let key_type = ffi::td_type(keys as *const ffi::td_t);
        let key_data = ffi::td_data(keys);
        let val_data = ffi::td_data(vals);

        for i in 0..show {
            let k = format_table_cell(keys, key_type, key_data, i);
            let v = if val_type == ffi::TD_LIST {
                // List of values — get each item
                let item = ffi::td_list_get(vals, i as i64);
                if item.is_null() || ffi::td_is_err(item as *const ffi::td_t) {
                    "(nil)".to_string()
                } else {
                    let item_type = ffi::td_type(item as *const ffi::td_t);
                    if item_type < 0 {
                        format_atom(item, item_type)
                    } else if item_type > 0 {
                        format_vector(item, item_type)
                    } else {
                        "?".to_string()
                    }
                }
            } else {
                format_table_cell(vals, val_type, val_data, i)
            };
            out.push_str(&format!("{} | {}\n", k, v));
        }

        if key_len > show {
            out.push_str(&format!("...({} more entries)\n", key_len - show));
        }

        // Drop explicitly to release our retained refs
        drop(key_val);
        drop(val_val);

        out.trim_end().to_string()
    }
}

fn format_list(ptr: *mut ffi::td_t) -> String {
    unsafe {
        let len = ffi::td_len(ptr as *const ffi::td_t) as usize;
        if len == 0 {
            return "()".to_string();
        }

        let show = len.min(20);
        let mut parts: Vec<String> = Vec::with_capacity(show);
        for i in 0..show {
            let item = ffi::td_list_get(ptr, i as i64);
            if item.is_null() || ffi::td_is_err(item as *const ffi::td_t) {
                parts.push("(nil)".to_string());
            } else {
                let item_type = ffi::td_type(item as *const ffi::td_t);
                if item_type < 0 {
                    parts.push(format_atom(item, item_type));
                } else if item_type > 0 {
                    parts.push(format!("({})", format_vector(item, item_type)));
                } else {
                    parts.push("(list)".to_string());
                }
            }
        }

        if len > show {
            parts.push(format!("..({} more)", len - show));
        }

        format!("({})", parts.join("; "))
    }
}

// ---------------------------------------------------------------------------
// Banner & help
// ---------------------------------------------------------------------------

fn print_banner() {
    let ver = env!("CARGO_PKG_VERSION");
    let hash = env!("GIT_HASH");
    let arch = std::env::consts::ARCH;
    println!("Td {} ({} {})", ver, hash, arch);
    println!("Type \\h for help, \\\\ to quit\n");
}

fn print_help() {
    println!("Td REPL commands:");
    println!("  \\\\    quit");
    println!("  \\h    show this help");
    println!("  \\t    toggle timer");
    println!();
    println!("Examples:");
    println!("  1 + 2");
    println!("  til 10");
    println!("  {{x+y}} [3;4]");
    println!("  (+/) til 10");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Initialize the Teide engine
    let _ctx = match teide::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Error: failed to init Teide engine: {e}");
            std::process::exit(1);
        }
    };

    let mut vm = Vm::new();
    let mut show_timer = false;

    // Non-interactive: read from stdin pipe
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(input) => {
                    let trimmed = input.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed == "\\t" || trimmed == "\\timer" {
                        show_timer = !show_timer;
                        continue;
                    }
                    let start = std::time::Instant::now();
                    match vm.eval(trimmed) {
                        Ok(val) => {
                            println!("{}", format_value(&val));
                            if show_timer {
                                let elapsed = start.elapsed();
                                eprintln!("  {elapsed:.3?}");
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading stdin: {e}");
                    std::process::exit(1);
                }
            }
        }
        return;
    }

    // Interactive REPL
    print_banner();

    // History
    let history_path = dirs_or_home().join(".td_history");
    let history = match FileBackedHistory::with_file(1000, history_path.clone()) {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("Warning: history disabled ({}): {}", history_path.display(), e);
            None
        }
    };

    // Keybindings
    let keybindings = default_emacs_keybindings();

    // Assemble editor
    let editor = Reedline::create()
        .with_edit_mode(Box::new(Emacs::new(keybindings)));
    let mut editor = if let Some(history) = history {
        editor.with_history(Box::new(history))
    } else {
        editor
    };

    let prompt = TdPrompt;

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Backslash commands
                if trimmed.starts_with('\\') {
                    match trimmed.split_whitespace().next().unwrap_or("") {
                        "\\\\" | "\\quit" | "\\q" => break,
                        "\\h" | "\\help" => { print_help(); continue; }
                        "\\t" | "\\timer" => {
                            show_timer = !show_timer;
                            println!("timer {}", if show_timer { "on" } else { "off" });
                            continue;
                        }
                        cmd => {
                            eprintln!("unknown command: {cmd}");
                            continue;
                        }
                    }
                }

                let start = std::time::Instant::now();
                match vm.eval(trimmed) {
                    Ok(val) => {
                        println!("{}", format_value(&val));
                        if show_timer {
                            let elapsed = start.elapsed();
                            eprintln!("  {elapsed:.3?}");
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                        if show_timer {
                            let elapsed = start.elapsed();
                            eprintln!("  {elapsed:.3?}");
                        }
                    }
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

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
