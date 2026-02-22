use crate::td::token::Token;

/// A lexer that tokenizes Td source code into a stream of `Token`s.
///
/// Implements `Iterator<Item = Token>`. After all real tokens are emitted,
/// yields a single `Token::Eof` and then `None` on subsequent calls.
pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    done: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer {
            input: src.as_bytes(),
            pos: 0,
            done: false,
        }
    }

    // ---- helpers ----

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.input.get(self.pos + offset).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Returns true if the previous non-whitespace token position indicates
    /// that a `-` should be interpreted as a negative sign rather than a minus
    /// operator. This is the case at the start of input or after an operator or
    /// delimiter.
    fn neg_context(&self) -> bool {
        // Walk backwards through whitespace to find the previous significant byte.
        let mut i = self.pos;
        if i == 0 {
            return true;
        }
        i -= 1;
        while i > 0 && matches!(self.input[i], b' ' | b'\t' | b'\r' | b'\n') {
            i -= 1;
        }
        let prev = self.input[i];
        matches!(
            prev,
            b'+' | b'-'
                | b'*'
                | b'/'
                | b'%'
                | b'='
                | b'<'
                | b'>'
                | b'!'
                | b'#'
                | b'('
                | b'['
                | b'{'
                | b':'
                | b';'
                | b','
        )
    }

    fn read_number(&mut self, negative: bool) -> Token {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }

        // Check for bool: single digit followed by 'b' with no further alphanumeric.
        let digit_len = self.pos - start;
        if digit_len == 1 && self.peek() == Some(b'b') {
            let after_b = self.peek_at(1);
            if after_b.is_none() || !after_b.unwrap().is_ascii_alphanumeric() {
                let digit = self.input[start] - b'0';
                self.pos += 1; // consume 'b'
                return Token::Bool(digit != 0);
            }
        }

        // Check for float: digits followed by '.' then more digits.
        if self.peek() == Some(b'.') {
            if let Some(next) = self.peek_at(1) {
                if next.is_ascii_digit() {
                    self.pos += 1; // consume '.'
                    while let Some(b) = self.peek() {
                        if b.is_ascii_digit() {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
                    let mut val: f64 = s.parse().unwrap();
                    if negative {
                        val = -val;
                    }
                    return Token::Float(val);
                }
            }
        }

        let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
        let mut val: i64 = s.parse().unwrap();
        if negative {
            val = -val;
        }
        Token::Int(val)
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.input[start..self.pos])
            .unwrap()
            .to_string()
    }

    fn read_string(&mut self) -> Token {
        // Opening quote already consumed.
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'"' {
                let s = std::str::from_utf8(&self.input[start..self.pos])
                    .unwrap()
                    .to_string();
                self.pos += 1; // consume closing quote
                return Token::Str(s);
            }
            self.pos += 1;
        }
        // Unterminated string — return what we have.
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .unwrap()
            .to_string();
        Token::Str(s)
    }

    fn read_symbol(&mut self) -> Token {
        // Backtick already consumed.
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .unwrap()
            .to_string();
        Token::Sym(s)
    }

    fn classify_word(word: &str) -> Token {
        match word {
            // Query keywords
            "select" => Token::Select,
            "update" => Token::Update,
            "delete" => Token::Delete,
            "from" => Token::From,
            "by" => Token::By,

            // Adverbs
            "over" | "scan" | "each" => Token::Adverb(word.to_string()),

            // Verbs (where is always a verb; parser disambiguates)
            "sum" | "avg" | "count" | "min" | "max" | "first" | "last" | "where" | "til"
            | "neg" | "abs" | "asc" | "desc" | "distinct" | "read" | "load" | "save"
            | "cols" | "meta" | "key" | "value" => Token::Verb(word.to_string()),

            _ => Token::Ident(word.to_string()),
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let b = match self.peek() {
            None => return Token::Eof,
            Some(b) => b,
        };

        match b {
            // Minus / negative number
            b'-' => {
                if self.neg_context() {
                    if let Some(next) = self.peek_at(1) {
                        if next.is_ascii_digit() {
                            self.pos += 1; // consume '-'
                            return self.read_number(true);
                        }
                    }
                }
                self.pos += 1;
                Token::Minus
            }

            // Digits
            b'0'..=b'9' => self.read_number(false),

            // Identifiers and keywords
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let word = self.read_ident();
                Self::classify_word(&word)
            }

            // String literal
            b'"' => {
                self.pos += 1; // consume opening quote
                self.read_string()
            }

            // Symbol
            b'`' => {
                self.pos += 1; // consume backtick
                self.read_symbol()
            }

            // Comment or slash
            b'/' => {
                // Slash followed by space or end-of-line → line comment
                let next = self.peek_at(1);
                if next.is_none() || next == Some(b' ') || next == Some(b'\n') || next == Some(b'\r') {
                    // Skip to end of line
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            self.pos += 1;
                            break;
                        }
                        self.pos += 1;
                    }
                    // Recurse to get next real token
                    return self.next_token();
                }
                self.pos += 1;
                Token::Slash
            }

            // Two-character operators
            b'>' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            b'<' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Token::Le
                } else if self.peek() == Some(b'>') {
                    self.pos += 1;
                    Token::Ne
                } else {
                    Token::Lt
                }
            }
            b'!' => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Token::Ne
                } else {
                    Token::Bang
                }
            }

            // Single-character tokens
            b'+' => { self.pos += 1; Token::Plus }
            b'*' => { self.pos += 1; Token::Star }
            b'%' => { self.pos += 1; Token::Percent }
            b'=' => { self.pos += 1; Token::Eq }
            b'#' => { self.pos += 1; Token::Hash }
            b'(' => { self.pos += 1; Token::LParen }
            b')' => { self.pos += 1; Token::RParen }
            b'[' => { self.pos += 1; Token::LBrack }
            b']' => { self.pos += 1; Token::RBrack }
            b'{' => { self.pos += 1; Token::LBrace }
            b'}' => { self.pos += 1; Token::RBrace }
            b':' => { self.pos += 1; Token::Colon }
            b';' => { self.pos += 1; Token::Semi }
            b',' => { self.pos += 1; Token::Comma }
            b'$' => { self.pos += 1; Token::Dollar }

            // Unknown character — skip it
            _ => {
                self.pos += 1;
                self.next_token()
            }
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        if self.done {
            return None;
        }
        let tok = self.next_token();
        if tok == Token::Eof {
            self.done = true;
        }
        Some(tok)
    }
}
