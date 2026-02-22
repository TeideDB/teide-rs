/// Tokens for the Td language lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Int(i64),
    Float(f64),
    Bool(bool),
    Sym(String),
    Str(String),
    Ident(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Bang,
    Hash,

    // Delimiters
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,

    // Punctuation
    Colon,
    Semi,
    Comma,

    // Keywords
    Select,
    Update,
    Delete,
    From,
    Where,
    By,

    // Verb / Adverb
    Verb(String),
    Adverb(String),

    // Special
    Dollar,
    Eof,
}
