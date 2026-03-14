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

    fn expect_upper(&mut self, expected: &str) -> Result<(), SqlError> {
        self.expect(expected)
    }

    #[allow(dead_code)]
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }
}

/// Tokenize SQL into words and punctuation, respecting parentheses and commas.
fn tokenize(sql: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = sql.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '(' | ')' | ',' | ';' => {
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
    t.expect_upper("CREATE")?;
    t.expect_upper("PROPERTY")?;
    t.expect_upper("GRAPH")?;
    let name = t.next()?.to_lowercase();

    t.expect_upper("VERTEX")?;
    t.expect_upper("TABLES")?;
    t.expect("(")?;
    let vertex_tables = parse_vertex_tables(&mut t)?;
    t.expect(")")?;

    let mut edge_tables = Vec::new();
    if t.peek().map(|s| s.to_uppercase()) == Some("EDGE".into()) {
        t.expect_upper("EDGE")?;
        t.expect_upper("TABLES")?;
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
        t.expect_upper("SOURCE")?;
        t.expect_upper("KEY")?;
        t.expect("(")?;
        let src_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect_upper("REFERENCES")?;
        let src_ref_table = t.next()?.to_lowercase();
        t.expect("(")?;
        let src_ref_col = t.next()?.to_lowercase();
        t.expect(")")?;

        // DESTINATION KEY (<col>) REFERENCES <table> (<col>)
        t.expect_upper("DESTINATION")?;
        t.expect_upper("KEY")?;
        t.expect("(")?;
        let dst_col = t.next()?.to_lowercase();
        t.expect(")")?;
        t.expect_upper("REFERENCES")?;
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
// DROP PROPERTY GRAPH
// ---------------------------------------------------------------------------

fn parse_drop_property_graph(sql: &str) -> Result<PgqStatement, SqlError> {
    let mut t = Tokens::new(sql);
    t.expect_upper("DROP")?;
    t.expect_upper("PROPERTY")?;
    t.expect_upper("GRAPH")?;
    let mut if_exists = false;
    let name_or_if = t.next()?;
    let name = if name_or_if.to_uppercase() == "IF" {
        t.expect_upper("EXISTS")?;
        if_exists = true;
        t.next()?.to_lowercase()
    } else {
        name_or_if.to_lowercase()
    };
    Ok(PgqStatement::DropPropertyGraph { name, if_exists })
}
