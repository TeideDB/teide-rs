// src/sql/pgq_parser.rs
// Pre-parser for SQL/PGQ syntax.
//
// sqlparser 0.53 has no SQL/PGQ support, so we intercept PGQ statements
// before they reach the SQL parser and handle them directly.

use super::SqlError;
use super::pgq::ColumnVisibility;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter to generate unique temp table names for GRAPH_TABLE results.
/// Combined with a process-level random nonce to avoid collisions with
/// user-defined table names.
static PGQ_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Random nonce generated once per process to make temp table names unpredictable
/// and avoid collisions with user tables.
fn temp_nonce() -> u64 {
    use std::sync::OnceLock;
    static NONCE: OnceLock<u64> = OnceLock::new();
    *NONCE.get_or_init(|| {
        // Use time + pid as a simple source of entropy without pulling in rand
        let pid = std::process::id() as u64;
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        // Mix bits so similar timestamps don't produce similar nonces
        pid.wrapping_mul(6364136223846793005).wrapping_add(time)
    })
}

// ---------------------------------------------------------------------------
// PGQ statement types (parsed from raw SQL)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct ParsedVertexTable {
    pub table_name: String,
    pub label: Option<String>,
    pub key_column: Option<String>,
    pub visibility: ColumnVisibility,
}

#[derive(Debug)]
pub(crate) struct ParsedEdgeTable {
    pub table_name: String,
    pub src_col: String,
    pub src_ref_table: String,
    pub src_ref_col: String,
    pub dst_col: String,
    pub dst_ref_table: String,
    pub dst_ref_col: String,
    pub label: Option<String>,
    pub visibility: ColumnVisibility,
}

#[derive(Debug)]
pub(crate) struct CreatePropertyGraph {
    pub name: String,
    pub vertex_tables: Vec<ParsedVertexTable>,
    pub edge_tables: Vec<ParsedEdgeTable>,
    pub or_replace: bool,
    pub if_not_exists: bool,
}

#[derive(Debug)]
pub(crate) struct CreateVectorIndex {
    pub name: String,
    pub table_name: String,
    pub column_name: String,
    pub m: Option<i32>,
    pub ef_construction: Option<i32>,
}

#[derive(Debug)]
pub(crate) enum PgqStatement {
    CreatePropertyGraph(CreatePropertyGraph),
    DropPropertyGraph { name: String, if_exists: bool },
    DescribePropertyGraph(String),
    CreateVectorIndex(CreateVectorIndex),
    DropVectorIndex { name: String, if_exists: bool },
}

// ---------------------------------------------------------------------------
// Pre-parser: detect and extract PGQ statements from raw SQL
// ---------------------------------------------------------------------------

/// Strip leading SQL comments (single-line `--` and block `/* */`) and whitespace.
/// Returns the remaining SQL text after all leading comments are removed.
fn strip_leading_comments(sql: &str) -> &str {
    let mut s = sql.trim_start();
    loop {
        if s.starts_with("--") {
            // Single-line comment: skip to end of line
            match s.find('\n') {
                Some(pos) => s = s[pos + 1..].trim_start(),
                None => return "", // entire string is a comment
            }
        } else if s.starts_with("/*") {
            // Block comment: handle nested /* */ to match tokenizer behavior
            let mut depth = 1u32;
            let mut i = 2;
            let bytes = s.as_bytes();
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                return ""; // unterminated block comment
            }
            s = s[i..].trim_start();
        } else {
            break;
        }
    }
    s
}

/// Check if a SQL string is a PGQ statement and parse it.
/// Returns None if the SQL is not a PGQ statement (should be passed to sqlparser).
pub(crate) fn try_parse_pgq(sql: &str) -> Result<Option<PgqStatement>, SqlError> {
    let stripped = strip_leading_comments(sql);
    let upper = stripped.to_uppercase();

    if upper.starts_with("CREATE PROPERTY GRAPH")
        || upper.starts_with("CREATE OR REPLACE PROPERTY GRAPH")
    {
        // Pass the full original SQL (trimmed) to the parser so line positions are preserved
        return Ok(Some(parse_create_property_graph(stripped)?));
    }
    if upper.starts_with("DROP PROPERTY GRAPH") {
        return Ok(Some(parse_drop_property_graph(stripped)?));
    }
    if upper.starts_with("DESCRIBE PROPERTY GRAPH") {
        return Ok(Some(parse_describe_property_graph(stripped)?));
    }
    if upper.starts_with("CREATE VECTOR INDEX") {
        return Ok(Some(parse_create_vector_index(stripped)?));
    }
    if upper.starts_with("DROP VECTOR INDEX") {
        return Ok(Some(parse_drop_vector_index(stripped)?));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Token-based parser helpers
// ---------------------------------------------------------------------------

struct Tokens {
    tokens: Vec<String>,
    pos: usize,
}

impl Tokens {
    fn new(sql: &str) -> Self {
        let tokens = tokenize(sql);
        Tokens { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Result<String, SqlError> {
        if self.pos >= self.tokens.len() {
            return Err(SqlError::Parse("Unexpected end of input".into()));
        }
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        Ok(tok)
    }

    fn expect(&mut self, expected: &str) -> Result<(), SqlError> {
        let tok = self.next()?;
        if tok.to_uppercase() != expected.to_uppercase() {
            return Err(SqlError::Parse(format!(
                "Expected '{expected}', got '{tok}'"
            )));
        }
        Ok(())
    }

    /// Return an error if there are unconsumed tokens (ignoring a trailing
    /// semicolon). This prevents trailing garbage from being silently dropped.
    fn expect_end(&self) -> Result<(), SqlError> {
        let remaining: Vec<&str> = self.tokens[self.pos..]
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s != ";")
            .collect();
        if !remaining.is_empty() {
            return Err(SqlError::Parse(format!(
                "Unexpected trailing tokens: {}",
                remaining.join(" ")
            )));
        }
        Ok(())
    }

}

/// Parse an optional PROPERTIES clause after LABEL / KEY in a vertex or edge
/// table definition.  Handles:
///   PROPERTIES (col1, col2, ...)
///   PROPERTIES ARE ALL COLUMNS EXCEPT (col1, col2, ...)
///   NO PROPERTIES
/// Returns `ColumnVisibility::All` when none of the above are present.
fn parse_properties(t: &mut Tokens) -> Result<ColumnVisibility, SqlError> {
    // Check for "NO PROPERTIES"
    if t.peek().map(|s| s.eq_ignore_ascii_case("NO")) == Some(true) {
        let saved = t.pos;
        t.next()?; // consume NO
        if t.peek().map(|s| s.eq_ignore_ascii_case("PROPERTIES")) == Some(true) {
            t.next()?; // consume PROPERTIES
            return Ok(ColumnVisibility::None);
        }
        // Not "NO PROPERTIES" — rewind
        t.pos = saved;
        return Ok(ColumnVisibility::All);
    }

    if t.peek().map(|s| s.eq_ignore_ascii_case("PROPERTIES")) != Some(true) {
        return Ok(ColumnVisibility::All);
    }
    t.next()?; // consume PROPERTIES

    // Check for "ARE ALL COLUMNS EXCEPT (...)"
    if t.peek().map(|s| s.eq_ignore_ascii_case("ARE")) == Some(true) {
        t.next()?; // consume ARE
        t.expect("ALL")?;
        t.expect("COLUMNS")?;
        t.expect("EXCEPT")?;
        t.expect("(")?;
        let mut cols = HashSet::new();
        loop {
            let col = t.next()?.to_lowercase();
            cols.insert(col);
            if t.peek() == Some(")") {
                break;
            }
            t.expect(",")?;
        }
        t.expect(")")?;
        return Ok(ColumnVisibility::AllExcept(cols));
    }

    // PROPERTIES (col1, col2, ...)
    t.expect("(")?;
    let mut cols = HashSet::new();
    loop {
        let col = t.next()?.to_lowercase();
        cols.insert(col);
        if t.peek() == Some(")") {
            break;
        }
        t.expect(",")?;
    }
    t.expect(")")?;
    Ok(ColumnVisibility::Only(cols))
}

/// Tokenize SQL into words and punctuation, respecting parentheses and commas.
fn tokenize(sql: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = sql.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            // -- line comments: skip to end of line
            '-' if {
                let mut peek2 = chars.clone();
                peek2.next();
                peek2.peek() == Some(&'-')
            } => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                // Skip until end of line
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }
            // /* */ block comments: skip entirely (supports nesting)
            '/' if {
                let mut peek2 = chars.clone();
                peek2.next();
                peek2.peek() == Some(&'*')
            } => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                chars.next(); // consume '/'
                chars.next(); // consume '*'
                let mut depth = 1u32;
                while depth > 0 {
                    match chars.next() {
                        Some('/') if chars.peek() == Some(&'*') => {
                            chars.next();
                            depth += 1;
                        }
                        Some('*') if chars.peek() == Some(&'/') => {
                            chars.next();
                            depth -= 1;
                        }
                        Some(_) => {}
                        None => break,
                    }
                }
            }
            '\'' => {
                // Keep single-quoted strings as one token (e.g. 'Alice')
                // Handles SQL-style escaped quotes: 'O''Brien' -> 'O''Brien'
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                let mut s = String::new();
                s.push('\'');
                chars.next(); // consume opening quote
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\'' {
                        // Check for escaped quote ('')
                        if chars.peek() == Some(&'\'') {
                            s.push('\'');
                            s.push('\'');
                            chars.next(); // consume second quote
                        } else {
                            s.push('\'');
                            break;
                        }
                    } else {
                        s.push(c);
                    }
                }
                tokens.push(s);
            }
            '"' => {
                // Double-quoted identifier: keep as one token (e.g. "My Column")
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                let mut s = String::new();
                chars.next(); // consume opening quote
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '"' {
                        // Check for escaped double quote ("")
                        if chars.peek() == Some(&'"') {
                            s.push('"');
                            chars.next(); // consume second quote
                        } else {
                            break;
                        }
                    } else {
                        s.push(c);
                    }
                }
                // Store unquoted content - callers compare case-insensitively
                tokens.push(s);
            }
            '(' | ')' | ',' | ';' | '[' | ']' | '{' | '}' | ':' | '=' | '-' | '<' | '>' | '+' | '*' | '/' | '.' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
                chars.next();
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                chars.next();
            }
            _ => {
                current.push(ch);
                chars.next();
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    // Remove trailing semicolons
    while tokens.last().map(|t| t.as_str()) == Some(";") {
        tokens.pop();
    }
    tokens
}

// ---------------------------------------------------------------------------
// CREATE PROPERTY GRAPH
// ---------------------------------------------------------------------------

fn parse_create_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect("CREATE")?;

    // Check for OR REPLACE
    let or_replace = if t.peek().map(|s| s.to_uppercase()) == Some("OR".into()) {
        t.next()?; // consume OR
        t.expect("REPLACE")?;
        true
    } else {
        false
    };

    t.expect("PROPERTY")?;
    t.expect("GRAPH")?;

    // Check for IF NOT EXISTS
    let if_not_exists = if t.peek().map(|s| s.to_uppercase()) == Some("IF".into()) {
        t.next()?; // consume IF
        t.expect("NOT")?;
        t.expect("EXISTS")?;
        true
    } else {
        false
    };

    if or_replace && if_not_exists {
        return Err(SqlError::Parse(
            "Cannot specify both OR REPLACE and IF NOT EXISTS".into(),
        ));
    }

    let name = t.next()?.to_lowercase();

    t.expect("VERTEX")?;
    t.expect("TABLES")?;
    t.expect("(")?;
    let vertex_tables = parse_vertex_tables(&mut t)?;
    t.expect(")")?;

    let mut edge_tables = Vec::new();
    if t.peek().map(|s| s.to_uppercase()) == Some("EDGE".into()) {
        t.expect("EDGE")?;
        t.expect("TABLES")?;
        t.expect("(")?;
        edge_tables = parse_edge_tables(&mut t)?;
        t.expect(")")?;
    }

    t.expect_end()?;

    Ok(PgqStatement::CreatePropertyGraph(CreatePropertyGraph {
        name,
        vertex_tables,
        edge_tables,
        or_replace,
        if_not_exists,
    }))
}

fn parse_vertex_tables(t: &mut Tokens) -> Result<Vec<ParsedVertexTable>, SqlError> {
    let mut tables = Vec::new();
    loop {
        let table_name = t.next()?.to_lowercase();
        let mut label = None;
        let mut key_column = None;
        // KEY and LABEL can appear in either order
        for _ in 0..2 {
            match t.peek().map(|s| s.to_uppercase()).as_deref() {
                Some("LABEL") => {
                    t.next()?;
                    label = Some(t.next()?);
                }
                Some("KEY") => {
                    t.next()?;
                    t.expect("(")?;
                    key_column = Some(t.next()?.to_lowercase());
                    t.expect(")")?;
                }
                _ => break,
            }
        }
        // Parse optional PROPERTIES / NO PROPERTIES / PROPERTIES ARE ALL COLUMNS EXCEPT
        let visibility = parse_properties(t)?;
        tables.push(ParsedVertexTable { table_name, label, key_column, visibility });
        if t.peek() == Some(",") {
            t.next()?; // consume comma
        } else {
            break;
        }
    }
    Ok(tables)
}

fn parse_edge_tables(t: &mut Tokens) -> Result<Vec<ParsedEdgeTable>, SqlError> {
    let mut tables = Vec::new();
    loop {
        let table_name = t.next()?.to_lowercase();

        // SOURCE KEY (<col>) REFERENCES <table> (<col>)
        t.expect("SOURCE")?;
        t.expect("KEY")?;
        t.expect("(")?;
        let src_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect("REFERENCES")?;
        let src_ref_table = t.next()?.to_lowercase();
        t.expect("(")?;
        let src_ref_col = t.next()?.to_lowercase();
        t.expect(")")?;

        // DESTINATION KEY (<col>) REFERENCES <table> (<col>)
        t.expect("DESTINATION")?;
        t.expect("KEY")?;
        t.expect("(")?;
        let dst_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect("REFERENCES")?;
        let dst_ref_table = t.next()?.to_lowercase();
        t.expect("(")?;
        let dst_ref_col = t.next()?.to_lowercase();
        t.expect(")")?;

        // LABEL <name>
        let mut label = None;
        if t.peek().map(|s| s.to_uppercase()) == Some("LABEL".into()) {
            t.next()?; // consume LABEL
            label = Some(t.next()?);
        }

        // Parse optional PROPERTIES / NO PROPERTIES / PROPERTIES ARE ALL COLUMNS EXCEPT
        let visibility = parse_properties(t)?;

        tables.push(ParsedEdgeTable {
            table_name,
            src_col,
            src_ref_table,
            src_ref_col,
            dst_col,
            dst_ref_table,
            dst_ref_col,
            label,
            visibility,
        });

        if t.peek() == Some(",") {
            t.next()?; // consume comma
        } else {
            break;
        }
    }
    Ok(tables)
}

// ---------------------------------------------------------------------------
// GRAPH_TABLE rewriting
// ---------------------------------------------------------------------------

use super::pgq::{
    ColumnEntry, EdgePattern, GraphTableExpr, MatchClause, MatchDirection,
    NodePattern, PathMode, PathPattern, PathQuantifier,
};

/// Scan SQL for GRAPH_TABLE(...) in FROM clauses and extract them.
/// Returns the rewritten SQL and a list of (temp_name, GraphTableExpr) pairs
/// (with GRAPH_TABLE replaced by a unique placeholder).
///
/// Properly skips single-quoted strings, double-quoted identifiers,
/// line comments (`--`), and block comments (`/* */`).
pub(crate) fn extract_graph_tables(
    sql: &str,
) -> Result<(String, Vec<(String, GraphTableExpr)>), SqlError> {
    let mut result = String::with_capacity(sql.len());
    let mut exprs = Vec::new();
    let bytes = sql.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        match bytes[i] {
            // Single-quoted string literal
            b'\'' => {
                let start = i;
                i += 1;
                while i < len {
                    if bytes[i] == b'\'' {
                        i += 1;
                        if i < len && bytes[i] == b'\'' {
                            i += 1; // escaped quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
                result.push_str(&sql[start..i]);
            }
            // Double-quoted identifier
            b'"' => {
                let start = i;
                i += 1;
                while i < len {
                    if bytes[i] == b'"' {
                        i += 1;
                        if i < len && bytes[i] == b'"' {
                            i += 1; // escaped quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
                result.push_str(&sql[start..i]);
            }
            // Line comment
            b'-' if i + 1 < len && bytes[i + 1] == b'-' => {
                let start = i;
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                if i < len {
                    i += 1; // skip newline
                }
                result.push_str(&sql[start..i]);
            }
            // Block comment (supports nested /* */)
            b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                let start = i;
                i += 2;
                let mut depth = 1u32;
                while depth > 0 && i < len {
                    if i + 1 < len && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if i + 1 < len && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                result.push_str(&sql[start..i]);
            }
            _ => {
                // Check for GRAPH_TABLE keyword
                if i + 11 <= len {
                    if let Some(slice) = sql.get(i..i + 11) {
                        if slice.eq_ignore_ascii_case("GRAPH_TABLE") {
                            let before_ok = i == 0
                                || (!bytes[i - 1].is_ascii_alphanumeric()
                                    && bytes[i - 1] != b'_');
                            let after_pos = i + 11;
                            let after_ok = after_pos >= len
                                || (!bytes[after_pos].is_ascii_alphanumeric()
                                    && bytes[after_pos] != b'_');

                            if before_ok && after_ok {
                                // Only treat as GRAPH_TABLE if followed by '(' (skip comments)
                                let rest = &sql[after_pos..];
                                let trimmed = super::skip_whitespace_and_comments(rest);
                                if trimmed.starts_with('(') {
                                    let paren_start =
                                        after_pos + (rest.len() - trimmed.len());

                                    let inner_end = find_matching_paren(sql, paren_start)?;
                                    let inner = &sql[paren_start + 1..inner_end];

                                    let expr = parse_graph_table_inner(inner)?;
                                    let uid = PGQ_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
                                    let temp_name = format!("__pgq_tmp_{:x}_{uid}", temp_nonce());
                                    exprs.push((temp_name.clone(), expr));

                                    result.push_str(&temp_name);
                                    i = inner_end + 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
                // Advance by UTF-8 char width to avoid landing on continuation bytes
                if let Some(ch) = sql[i..].chars().next() {
                    result.push(ch);
                    i += ch.len_utf8();
                } else {
                    i += 1;
                }
            }
        }
    }

    Ok((result, exprs))
}

/// Find the matching closing parenthesis for an opening one at `start`.
/// Handles single-quoted strings, double-quoted identifiers,
/// `--` line comments, and `/* */` block comments.
fn find_matching_paren(sql: &str, start: usize) -> Result<usize, SqlError> {
    let bytes = sql.as_bytes();
    let len = bytes.len();
    let mut depth = 0i32;
    let mut i = start;
    while i < len {
        let b = bytes[i];
        match b {
            b'\'' => {
                // Single-quoted string: skip to closing quote (handle '' escapes)
                i += 1;
                while i < len {
                    if bytes[i] == b'\'' {
                        i += 1;
                        if i < len && bytes[i] == b'\'' {
                            i += 1; // escaped quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'"' => {
                // Double-quoted identifier: skip to closing double quote
                i += 1;
                while i < len {
                    if bytes[i] == b'"' {
                        i += 1;
                        if i < len && bytes[i] == b'"' {
                            i += 1; // escaped double quote
                        } else {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            b'-' if i + 1 < len && bytes[i + 1] == b'-' => {
                // Line comment: skip to end of line
                i += 2;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                if i < len {
                    i += 1; // skip the newline
                }
            }
            b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                // Block comment: skip to closing */
                i += 2;
                let mut comment_depth = 1;
                while i + 1 < len && comment_depth > 0 {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        comment_depth += 1;
                        i += 2;
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        comment_depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i);
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    Err(SqlError::Parse("Unmatched parenthesis in GRAPH_TABLE".into()))
}

/// Parse the content inside GRAPH_TABLE(...).
/// Expected: <graph_name> MATCH <pattern> COLUMNS (<col_list>)
fn parse_graph_table_inner(inner: &str) -> Result<GraphTableExpr, SqlError> {
    let mut t = Tokens::new(inner);

    let graph_name = t.next()?.to_lowercase();

    // Parse optional path_var = ANY SHORTEST before MATCH
    let mut path_variable = None;
    let mut mode = PathMode::Walk;

    t.expect("MATCH")?;

    // Check for: p = ANY SHORTEST or just pattern
    let checkpoint = t.pos;
    if let Ok(maybe_var) = t.next() {
        if t.peek() == Some("=") {
            path_variable = Some(maybe_var.to_lowercase());
            t.next()?; // consume '='
            if t.peek().map(|s| s.to_uppercase()) == Some("ANY".into()) {
                t.next()?; // consume ANY
                t.expect("SHORTEST")?;
                mode = PathMode::AnyShortest;
            }
        } else {
            // Not a var assignment, rewind
            t.pos = checkpoint;
        }
    } else {
        t.pos = checkpoint;
    }

    // Parse patterns (comma-separated path patterns)
    let patterns = parse_match_patterns(&mut t)?;

    // Parse COLUMNS clause
    t.expect("COLUMNS")?;
    t.expect("(")?;
    let columns = parse_columns_clause(&mut t)?;
    t.expect(")")?;

    t.expect_end()?;

    Ok(GraphTableExpr {
        graph_name,
        match_clause: MatchClause {
            path_variable,
            mode,
            patterns,
        },
        columns,
    })
}

/// Parse comma-separated path patterns.
fn parse_match_patterns(t: &mut Tokens) -> Result<Vec<PathPattern>, SqlError> {
    let mut patterns = Vec::new();
    patterns.push(parse_single_path(t)?);

    while t.peek() == Some(",") {
        let saved = t.pos;
        t.next()?; // consume comma
        if t.peek().map(|s| s.to_uppercase()) == Some("COLUMNS".into()) {
            t.pos = saved;
            break;
        }
        if t.peek() == Some("(") {
            patterns.push(parse_single_path(t)?);
        } else {
            t.pos = saved;
            break;
        }
    }

    Ok(patterns)
}

/// Parse a single path: (node)-[edge]->(node)...
fn parse_single_path(t: &mut Tokens) -> Result<PathPattern, SqlError> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    nodes.push(parse_node_pattern(t)?);

    while let Some("-") | Some("<") = t.peek() {
        let (edge, node) = parse_edge_and_node(t)?;
        edges.push(edge);
        nodes.push(node);
    }

    Ok(PathPattern { nodes, edges })
}

/// Parse a node pattern: (var:Label WHERE condition)
fn parse_node_pattern(t: &mut Tokens) -> Result<NodePattern, SqlError> {
    t.expect("(")?;

    let mut variable = None;
    let mut label = None;
    let mut filter = None;

    if t.peek() != Some(")") {
        let first = t.next()?;
        if first == ":" {
            // :Label (no variable)
            label = Some(t.next()?.to_lowercase());
        } else if t.peek() == Some(":") {
            // var:Label
            variable = Some(first.to_lowercase());
            t.next()?; // consume ':'
            if t.peek() != Some(")")
                && t.peek().map(|s| s.to_uppercase()) != Some("WHERE".into())
            {
                label = Some(t.next()?.to_lowercase());
            }
        } else {
            // Just a variable name
            variable = Some(first.to_lowercase());
        }

        // Check for WHERE
        if t.peek().map(|s| s.to_uppercase()) == Some("WHERE".into()) {
            t.next()?; // consume WHERE
            let mut filter_tokens = Vec::new();
            let mut depth = 0;
            loop {
                match t.peek() {
                    Some(")") if depth == 0 => break,
                    Some("(") => {
                        depth += 1;
                        filter_tokens.push(t.next()?);
                    }
                    Some(")") => {
                        depth -= 1;
                        filter_tokens.push(t.next()?);
                    }
                    Some(_) => filter_tokens.push(t.next()?),
                    None => {
                        return Err(SqlError::Parse(
                            "Unexpected end in node pattern".into(),
                        ))
                    }
                }
            }
            if !filter_tokens.is_empty() {
                filter = Some(filter_tokens.join(" "));
            }
        }
    }

    t.expect(")")?;

    Ok(NodePattern {
        variable,
        label,
        filter,
    })
}

/// Parse an edge pattern and the following node: -[e:Label]->(node)
fn parse_edge_and_node(
    t: &mut Tokens,
) -> Result<(EdgePattern, NodePattern), SqlError> {
    let first = t.next()?;
    let starts_reverse = first == "<";
    if starts_reverse {
        t.expect("-")?;
    }

    let mut variable = None;
    let mut label = None;
    let mut quantifier = PathQuantifier::One;
    let mut cost_expr: Option<String> = None;

    if t.peek() == Some("[") {
        t.next()?; // consume '['

        if t.peek() != Some("]") {
            let first_tok = t.next()?;
            if first_tok == ":" {
                label = Some(t.next()?.to_lowercase());
            } else if t.peek() == Some(":") {
                variable = Some(first_tok.to_lowercase());
                t.next()?; // consume ':'
                if t.peek() != Some("]") {
                    label = Some(t.next()?.to_lowercase());
                }
            } else {
                variable = Some(first_tok.to_lowercase());
            }
        }

        // Parse optional COST expression inside edge brackets: [r:Label COST r.weight]
        if t.peek().map(|s| s.eq_ignore_ascii_case("COST")) == Some(true) {
            t.next()?; // consume COST
            let mut expr_tokens = Vec::new();
            // Collect tokens until ']'
            while t.peek() != Some("]") && t.peek().is_some() {
                expr_tokens.push(t.next()?);
            }
            if expr_tokens.is_empty() {
                return Err(SqlError::Parse("COST keyword requires an expression".into()));
            }
            cost_expr = Some(expr_tokens.join(""));
        }

        t.expect("]")?;
    }

    // Parse direction suffix: -> or -
    t.expect("-")?;
    let direction = if t.peek() == Some(">") {
        t.next()?; // consume '>'
        if starts_reverse {
            MatchDirection::Bidirectional
        } else {
            MatchDirection::Forward
        }
    } else if starts_reverse {
        MatchDirection::Reverse
    } else {
        MatchDirection::Undirected
    };

    // Parse optional quantifier: +, *, {min,max}
    match t.peek() {
        Some("+") => {
            t.next()?;
            quantifier = PathQuantifier::Plus;
        }
        Some("*") => {
            t.next()?;
            quantifier = PathQuantifier::Star;
        }
        Some("{") => {
            t.next()?; // consume '{'
            let first_tok = t.next()?;
            if first_tok == "," {
                // {,max} shorthand: min defaults to 0
                let max_str = t.next()?;
                let max: u8 = max_str
                    .parse()
                    .map_err(|_| SqlError::Parse(format!("Invalid max depth: {max_str}")))?;
                t.expect("}")?;
                quantifier = PathQuantifier::Range { min: 0, max };
            } else {
                let min: u8 = first_tok
                    .parse()
                    .map_err(|_| SqlError::Parse(format!("Invalid min depth: {first_tok}")))?;
                t.expect(",")?;
                if t.peek() == Some("}") {
                    // {min,} shorthand: max defaults to 255
                    t.next()?; // consume '}'
                    quantifier = PathQuantifier::Range { min, max: 255 };
                } else {
                    let max_str = t.next()?;
                    let max: u8 = max_str
                        .parse()
                        .map_err(|_| SqlError::Parse(format!("Invalid max depth: {max_str}")))?;
                    t.expect("}")?;
                    if min > max {
                        return Err(SqlError::Parse(format!(
                            "Invalid path quantifier: min ({min}) > max ({max})"
                        )));
                    }
                    quantifier = PathQuantifier::Range { min, max };
                }
            }
        }
        _ => {}
    }

    let node = parse_node_pattern(t)?;

    Ok((
        EdgePattern {
            variable,
            label,
            direction,
            quantifier,
            cost_expr,
        },
        node,
    ))
}

/// Parse COLUMNS clause entries: expr [AS alias], ...
fn parse_columns_clause(t: &mut Tokens) -> Result<Vec<ColumnEntry>, SqlError> {
    let mut entries = Vec::new();
    loop {
        if t.peek() == Some(")") {
            break;
        }

        let mut expr_tokens = Vec::new();
        let mut depth = 0;
        loop {
            match t.peek() {
                Some(")") if depth == 0 => break,
                Some(",") if depth == 0 => break,
                Some(tok) if tok.to_uppercase() == "AS" && depth == 0 => break,
                Some("(") => {
                    depth += 1;
                    expr_tokens.push(t.next()?);
                }
                Some(")") => {
                    depth -= 1;
                    expr_tokens.push(t.next()?);
                }
                Some(_) => expr_tokens.push(t.next()?),
                None => return Err(SqlError::Parse("Unexpected end in COLUMNS".into())),
            }
        }

        let expr = expr_tokens.join(" ");
        let mut alias = None;

        if t.peek().map(|s| s.to_uppercase()) == Some("AS".into()) {
            t.next()?; // consume AS
            alias = Some(t.next()?.to_lowercase());
        }

        entries.push(ColumnEntry { expr, alias });

        if t.peek() == Some(",") {
            t.next()?; // consume comma
        }
    }

    Ok(entries)
}

// ---------------------------------------------------------------------------
// DROP PROPERTY GRAPH
// ---------------------------------------------------------------------------

fn parse_drop_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect("DROP")?;
    t.expect("PROPERTY")?;
    t.expect("GRAPH")?;
    let mut if_exists = false;
    let name_or_if = t.next()?;
    let name = if name_or_if.to_uppercase() == "IF" {
        t.expect("EXISTS")?;
        if_exists = true;
        t.next()?.to_lowercase()
    } else {
        name_or_if.to_lowercase()
    };
    t.expect_end()?;
    Ok(PgqStatement::DropPropertyGraph { name, if_exists })
}

// ---------------------------------------------------------------------------
// DESCRIBE PROPERTY GRAPH
// Syntax: DESCRIBE PROPERTY GRAPH <name>
// ---------------------------------------------------------------------------

fn parse_describe_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect("DESCRIBE")?;
    t.expect("PROPERTY")?;
    t.expect("GRAPH")?;
    let name = t.next()?.to_lowercase();
    t.expect_end()?;
    Ok(PgqStatement::DescribePropertyGraph(name))
}

// ---------------------------------------------------------------------------
// CREATE VECTOR INDEX
// Syntax: CREATE VECTOR INDEX <name> ON <table>(<column>) [USING HNSW(M=16, ef_construction=200)]
// ---------------------------------------------------------------------------

fn parse_create_vector_index(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect("CREATE")?;
    t.expect("VECTOR")?;
    t.expect("INDEX")?;
    let name = t.next()?.to_lowercase();
    t.expect("ON")?;
    let table_name = t.next()?.to_lowercase();
    t.expect("(")?;
    let column_name = t.next()?.to_lowercase();
    t.expect(")")?;

    let mut m = None;
    let mut ef_construction = None;

    // Parse optional USING HNSW(M=16, ef_construction=200)
    if t.peek().map(|s| s.to_uppercase()) == Some("USING".into()) {
        t.next()?; // consume USING
        t.expect("HNSW")?;
        t.expect("(")?;
        loop {
            if t.peek() == Some(")") {
                break;
            }
            let param = t.next()?.to_uppercase();
            t.expect("=")?;
            let value_str = t.next()?;
            let value: i32 = value_str
                .parse()
                .map_err(|_| SqlError::Parse(format!("Invalid HNSW parameter value: {value_str}")))?;
            match param.as_str() {
                "M" => m = Some(value),
                "EF_CONSTRUCTION" => ef_construction = Some(value),
                _ => {
                    return Err(SqlError::Parse(format!(
                        "Unknown HNSW parameter: {param}"
                    )));
                }
            }
            if t.peek() == Some(",") {
                t.next()?; // consume comma
            }
        }
        t.expect(")")?;
    }

    t.expect_end()?;
    Ok(PgqStatement::CreateVectorIndex(CreateVectorIndex {
        name,
        table_name,
        column_name,
        m,
        ef_construction,
    }))
}

// ---------------------------------------------------------------------------
// DROP VECTOR INDEX
// Syntax: DROP VECTOR INDEX [IF EXISTS] <name>
// ---------------------------------------------------------------------------

fn parse_drop_vector_index(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect("DROP")?;
    t.expect("VECTOR")?;
    t.expect("INDEX")?;
    let mut if_exists = false;
    let name_or_if = t.next()?;
    let name = if name_or_if.to_uppercase() == "IF" {
        t.expect("EXISTS")?;
        if_exists = true;
        t.next()?.to_lowercase()
    } else {
        name_or_if.to_lowercase()
    };
    t.expect_end()?;
    Ok(PgqStatement::DropVectorIndex { name, if_exists })
}
