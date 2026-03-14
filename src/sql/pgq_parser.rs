// src/sql/pgq_parser.rs
// Pre-parser for SQL/PGQ syntax.
//
// sqlparser 0.53 has no SQL/PGQ support, so we intercept PGQ statements
// before they reach the SQL parser and handle them directly.

use super::SqlError;

// ---------------------------------------------------------------------------
// PGQ statement types (parsed from raw SQL)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct ParsedVertexTable {
    pub table_name: String,
    pub label: Option<String>,
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
}

#[derive(Debug)]
pub(crate) struct CreatePropertyGraph {
    pub name: String,
    pub vertex_tables: Vec<ParsedVertexTable>,
    pub edge_tables: Vec<ParsedEdgeTable>,
}

#[derive(Debug)]
pub(crate) enum PgqStatement {
    CreatePropertyGraph(CreatePropertyGraph),
    DropPropertyGraph { name: String, if_exists: bool },
}

// ---------------------------------------------------------------------------
// Pre-parser: detect and extract PGQ statements from raw SQL
// ---------------------------------------------------------------------------

/// Check if a SQL string is a PGQ statement and parse it.
/// Returns None if the SQL is not a PGQ statement (should be passed to sqlparser).
pub(crate) fn try_parse_pgq(sql: &str) -> Result<Option<PgqStatement>, SqlError> {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();

    if upper.starts_with("CREATE PROPERTY GRAPH") {
        return Ok(Some(parse_create_property_graph(trimmed)?));
    }
    if upper.starts_with("DROP PROPERTY GRAPH") {
        return Ok(Some(parse_drop_property_graph(trimmed)?));
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

}

/// Tokenize SQL into words and punctuation, respecting parentheses and commas.
fn tokenize(sql: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = sql.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
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
            '(' | ')' | ',' | ';' | '[' | ']' | '{' | '}' | ':' | '=' | '-' | '<' | '>' | '+' | '*' | '.' => {
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
    t.expect("PROPERTY")?;
    t.expect("GRAPH")?;
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

    Ok(PgqStatement::CreatePropertyGraph(CreatePropertyGraph {
        name,
        vertex_tables,
        edge_tables,
    }))
}

fn parse_vertex_tables(t: &mut Tokens) -> Result<Vec<ParsedVertexTable>, SqlError> {
    let mut tables = Vec::new();
    loop {
        let table_name = t.next()?.to_lowercase();
        let mut label = None;
        if t.peek().map(|s| s.to_uppercase()) == Some("LABEL".into()) {
            t.next()?; // consume LABEL
            label = Some(t.next()?);
        }
        tables.push(ParsedVertexTable { table_name, label });
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

        tables.push(ParsedEdgeTable {
            table_name,
            src_col,
            src_ref_table,
            src_ref_col,
            dst_col,
            dst_ref_table,
            dst_ref_col,
            label,
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
/// Returns a list of extracted GraphTableExpr and the rewritten SQL
/// (with GRAPH_TABLE replaced by a placeholder like `__pgq_result_0`).
pub(crate) fn extract_graph_tables(
    sql: &str,
) -> Result<(String, Vec<GraphTableExpr>), SqlError> {
    let upper = sql.to_uppercase();
    let mut result = String::with_capacity(sql.len());
    let mut exprs = Vec::new();
    let mut pos = 0;
    let bytes = sql.as_bytes();

    while pos < bytes.len() {
        if let Some(gt_pos) = upper[pos..].find("GRAPH_TABLE") {
            let abs_pos = pos + gt_pos;

            // Check that GRAPH_TABLE is not inside a string literal
            let in_str = sql[..abs_pos].chars().filter(|&c| c == '\'').count() % 2 != 0;
            if in_str {
                result.push_str(&sql[pos..abs_pos + "GRAPH_TABLE".len()]);
                pos = abs_pos + "GRAPH_TABLE".len();
                continue;
            }

            // Check word boundaries to avoid matching inside identifiers
            let before_ok = abs_pos == 0
                || !sql.as_bytes()[abs_pos - 1].is_ascii_alphanumeric()
                    && sql.as_bytes()[abs_pos - 1] != b'_';
            let after_pos = abs_pos + "GRAPH_TABLE".len();
            let after_ok = after_pos >= sql.len()
                || !sql.as_bytes()[after_pos].is_ascii_alphanumeric()
                    && sql.as_bytes()[after_pos] != b'_';
            if !before_ok || !after_ok {
                result.push_str(&sql[pos..after_pos]);
                pos = after_pos;
                continue;
            }

            result.push_str(&sql[pos..abs_pos]);

            let after_gt = abs_pos + "GRAPH_TABLE".len();
            let paren_start = sql[after_gt..]
                .find('(')
                .ok_or_else(|| SqlError::Parse("Expected '(' after GRAPH_TABLE".into()))?
                + after_gt;

            let inner_end = find_matching_paren(sql, paren_start)?;
            let inner = &sql[paren_start + 1..inner_end];

            let expr = parse_graph_table_inner(inner)?;
            let idx = exprs.len();
            exprs.push(expr);

            result.push_str(&format!("__pgq_result_{idx}"));

            pos = inner_end + 1;
        } else {
            result.push_str(&sql[pos..]);
            break;
        }
    }

    Ok((result, exprs))
}

/// Find the matching closing parenthesis for an opening one at `start`.
fn find_matching_paren(sql: &str, start: usize) -> Result<usize, SqlError> {
    let mut depth = 0;
    let mut in_string = false;
    for (i, ch) in sql[start..].char_indices() {
        match ch {
            '\'' if !in_string => in_string = true,
            '\'' if in_string => in_string = false,
            '(' if !in_string => depth += 1,
            ')' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Ok(start + i);
                }
            }
            _ => {}
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

        t.expect("]")?;
    }

    // Parse direction suffix: -> or -
    t.expect("-")?;
    let direction = if t.peek() == Some(">") {
        t.next()?; // consume '>'
        if starts_reverse {
            return Err(SqlError::Parse("Invalid edge direction: <-..->".into()));
        }
        MatchDirection::Forward
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
            let min_str = t.next()?;
            let min: u8 = min_str
                .parse()
                .map_err(|_| SqlError::Parse(format!("Invalid min depth: {min_str}")))?;
            t.expect(",")?;
            let max_str = t.next()?;
            let max: u8 = max_str
                .parse()
                .map_err(|_| SqlError::Parse(format!("Invalid max depth: {max_str}")))?;
            t.expect("}")?;
            quantifier = PathQuantifier::Range { min, max };
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
    Ok(PgqStatement::DropPropertyGraph { name, if_exists })
}
