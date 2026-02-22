use crate::td::ast::*;
use crate::td::error::{TdError, TdResult};
use crate::td::lexer::Lexer;
use crate::td::token::Token;

/// Parser state: holds a vector of tokens and a cursor position.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    // ---- helpers ----

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> TdResult<()> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(TdError::Parse(format!(
                "expected {:?}, got {:?}",
                expected, tok
            )))
        }
    }

    fn at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn at_expr_end(&self) -> bool {
        matches!(
            self.peek(),
            Token::Eof | Token::Semi | Token::RParen | Token::RBrack | Token::RBrace
        )
    }

    // ---- main entry point ----

    /// Parse the full token stream, handling semicolons as block separators.
    fn parse_program(&mut self) -> TdResult<Expr> {
        let mut exprs = Vec::new();
        if !self.at_end() {
            exprs.push(self.parse_expr()?);
        }
        while matches!(self.peek(), Token::Semi) {
            self.advance(); // consume ';'
            if self.at_end() {
                break;
            }
            exprs.push(self.parse_expr()?);
        }
        match exprs.len() {
            0 => Ok(Expr::Nil),
            1 => Ok(exprs.into_iter().next().unwrap()),
            _ => Ok(Expr::Block(exprs)),
        }
    }

    /// Parse a single expression (right-to-left evaluation).
    ///
    /// This parses an atom (the LHS), then checks if a binary operator or
    /// assignment follows. If so, it recursively parses the entire RHS.
    fn parse_expr(&mut self) -> TdResult<Expr> {
        let lhs = self.parse_atom()?;

        // Check for assignment: if lhs is Ident and next is Colon
        if let Expr::Ident(ref name) = lhs {
            if matches!(self.peek(), Token::Colon) {
                let name = name.clone();
                self.advance(); // consume ':'
                let rhs = self.parse_expr()?;
                return Ok(Expr::Assign(name, Box::new(rhs)));
            }
        }

        // Check for indexing: lhs followed by `[`
        let lhs = self.parse_postfix(lhs)?;

        // Check for binary operator
        if let Some(op) = self.peek_binop() {
            self.advance(); // consume the operator token
            let rhs = self.parse_expr()?; // right-to-left: parse full RHS
            return Ok(Expr::BinOp(op, Box::new(lhs), Box::new(rhs)));
        }

        // Check for `!` (dict constructor) - treated as infix
        if matches!(self.peek(), Token::Bang) {
            self.advance(); // consume '!'
            let rhs = self.parse_expr()?;
            return Ok(Expr::Dict(Box::new(lhs), Box::new(rhs)));
        }

        // Check for `where` used as infix verb: `a where cond`
        // Parsed as: Index(a, Verb(Where, cond))
        if matches!(self.peek(), Token::Verb(ref v) if v == "where") {
            self.advance(); // consume 'where'
            let cond = self.parse_expr()?;
            let where_expr = Expr::Verb(Verb::Where, vec![cond]);
            return Ok(Expr::Index(Box::new(lhs), vec![where_expr]));
        }

        // Check for adverb after the LHS (which could be an operator-like expression)
        // e.g., `+ over 1 2 3`
        // This is handled in parse_atom for operator tokens.

        Ok(lhs)
    }

    /// Parse postfix operations (indexing with `[...]`).
    fn parse_postfix(&mut self, mut expr: Expr) -> TdResult<Expr> {
        while matches!(self.peek(), Token::LBrack) {
            self.advance(); // consume '['
            let mut args = Vec::new();
            if !matches!(self.peek(), Token::RBrack) {
                args.push(self.parse_expr()?);
                while matches!(self.peek(), Token::Semi) {
                    self.advance(); // consume ';'
                    args.push(self.parse_expr()?);
                }
            }
            self.expect(&Token::RBrack)?;
            expr = Expr::Index(Box::new(expr), args);
        }
        Ok(expr)
    }

    /// Parse an atom: the smallest meaningful unit.
    fn parse_atom(&mut self) -> TdResult<Expr> {
        match self.peek().clone() {
            Token::Int(_) => self.parse_int_or_vector(),
            Token::Float(_) => self.parse_float_or_vector(),
            Token::Bool(b) => {
                self.advance();
                Ok(Expr::Bool(b))
            }
            Token::Sym(_) => self.parse_sym_or_vector(),
            Token::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Token::Ident(_) => {
                let tok = self.advance();
                if let Token::Ident(name) = tok {
                    Ok(Expr::Ident(name))
                } else {
                    unreachable!()
                }
            }
            Token::Verb(ref name) => {
                let verb = Self::parse_verb_name(name)?;
                self.advance();
                // Check for adverb after verb: `count each (...)`
                if let Token::Adverb(ref adverb_name) = self.peek().clone() {
                    let adverb = Self::parse_adverb_name(adverb_name)?;
                    self.advance(); // consume adverb
                    let arg = self.parse_expr()?;
                    Ok(Expr::Adverb(
                        adverb,
                        Box::new(Expr::Verb(verb, vec![])),
                        Box::new(arg),
                    ))
                } else if self.at_expr_end() {
                    // Verb with no argument (used as a value/reference)
                    Ok(Expr::Verb(verb, vec![]))
                } else {
                    let arg = self.parse_expr()?;
                    Ok(Expr::Verb(verb, vec![arg]))
                }
            }
            Token::LParen => {
                self.advance(); // consume '('
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBrace => self.parse_lambda(),
            Token::Dollar => self.parse_cond(),
            Token::Minus => {
                // Unary negation
                self.advance();
                let operand = self.parse_atom()?;
                Ok(Expr::UnaryOp(UnaryOp::Neg, Box::new(operand)))
            }
            // Operators that could be used with adverbs: `+ over 1 2 3`
            ref tok if Self::is_operator_token(tok) => {
                let op = self.peek_binop().ok_or_else(|| {
                    TdError::Parse(format!("unexpected operator {:?}", self.peek()))
                })?;
                self.advance(); // consume operator

                // Check for adverb
                if let Token::Adverb(ref adverb_name) = self.peek().clone() {
                    let adverb = Self::parse_adverb_name(adverb_name)?;
                    self.advance(); // consume adverb
                    let arg = self.parse_expr()?;
                    Ok(Expr::Adverb(
                        adverb,
                        Box::new(Expr::BinOp(
                            op,
                            Box::new(Expr::Nil),
                            Box::new(Expr::Nil),
                        )),
                        Box::new(arg),
                    ))
                } else {
                    Err(TdError::Parse(format!(
                        "unexpected operator {:?} without LHS",
                        op
                    )))
                }
            }
            _ => Err(TdError::Parse(format!(
                "unexpected token: {:?}",
                self.peek()
            ))),
        }
    }

    /// Parse one or more consecutive integer literals into a single Int or Vector.
    fn parse_int_or_vector(&mut self) -> TdResult<Expr> {
        let mut values = Vec::new();
        while let Token::Int(n) = self.peek().clone() {
            self.advance();
            values.push(Expr::Int(n));
            // Stop forming a vector if next token is an operator, semicolon, etc.
            // Only continue if next token is also an Int
            if !matches!(self.peek(), Token::Int(_)) {
                break;
            }
        }
        if values.len() == 1 {
            Ok(values.into_iter().next().unwrap())
        } else {
            Ok(Expr::Vector(values))
        }
    }

    /// Parse one or more consecutive float literals into a single Float or Vector.
    fn parse_float_or_vector(&mut self) -> TdResult<Expr> {
        let mut values = Vec::new();
        while let Token::Float(f) = self.peek().clone() {
            self.advance();
            values.push(Expr::Float(f));
            if !matches!(self.peek(), Token::Float(_)) {
                break;
            }
        }
        if values.len() == 1 {
            Ok(values.into_iter().next().unwrap())
        } else {
            Ok(Expr::Vector(values))
        }
    }

    /// Parse one or more consecutive symbol literals into a single Sym or Vector.
    fn parse_sym_or_vector(&mut self) -> TdResult<Expr> {
        let mut values = Vec::new();
        while let Token::Sym(ref s) = self.peek().clone() {
            let s = s.clone();
            self.advance();
            values.push(Expr::Sym(s));
            if !matches!(self.peek(), Token::Sym(_)) {
                break;
            }
        }
        if values.len() == 1 {
            Ok(values.into_iter().next().unwrap())
        } else {
            Ok(Expr::Vector(values))
        }
    }

    /// Parse a lambda: `{...}` or `{[params] ...}`.
    fn parse_lambda(&mut self) -> TdResult<Expr> {
        self.expect(&Token::LBrace)?;

        // Check for explicit params: `{[a;b] ...}`
        let explicit_params = if matches!(self.peek(), Token::LBrack) {
            self.advance(); // consume '['
            let mut params = Vec::new();
            if let Token::Ident(name) = self.peek().clone() {
                self.advance();
                params.push(name);
                while matches!(self.peek(), Token::Semi) {
                    self.advance(); // consume ';'
                    if let Token::Ident(name) = self.advance() {
                        params.push(name);
                    } else {
                        return Err(TdError::Parse("expected parameter name".into()));
                    }
                }
            }
            self.expect(&Token::RBrack)?;
            Some(params)
        } else {
            None
        };

        // Parse the body (everything until RBrace)
        let body = if matches!(self.peek(), Token::RBrace) {
            Expr::Nil
        } else {
            self.parse_expr()?
        };

        self.expect(&Token::RBrace)?;

        let params = match explicit_params {
            Some(p) => p,
            None => {
                // Scan body for implicit x, y, z variables
                let mut implicit = Vec::new();
                Self::collect_implicit_params(&body, &mut implicit);
                implicit.sort();
                implicit.dedup();
                implicit
            }
        };

        Ok(Expr::Lambda {
            params,
            body: Box::new(body),
        })
    }

    /// Recursively scan an expression for implicit lambda parameters (x, y, z).
    fn collect_implicit_params(expr: &Expr, params: &mut Vec<String>) {
        match expr {
            Expr::Ident(name) if name == "x" || name == "y" || name == "z" => {
                params.push(name.clone());
            }
            Expr::BinOp(_, lhs, rhs) => {
                Self::collect_implicit_params(lhs, params);
                Self::collect_implicit_params(rhs, params);
            }
            Expr::UnaryOp(_, operand) => {
                Self::collect_implicit_params(operand, params);
            }
            Expr::Verb(_, args) => {
                for arg in args {
                    Self::collect_implicit_params(arg, params);
                }
            }
            Expr::Adverb(_, op, arg) => {
                Self::collect_implicit_params(op, params);
                Self::collect_implicit_params(arg, params);
            }
            Expr::Vector(elems) => {
                for elem in elems {
                    Self::collect_implicit_params(elem, params);
                }
            }
            Expr::Assign(_, val) => {
                Self::collect_implicit_params(val, params);
            }
            Expr::Lambda { .. } => {
                // Don't descend into nested lambdas
            }
            Expr::Call(func, args) => {
                Self::collect_implicit_params(func, params);
                for arg in args {
                    Self::collect_implicit_params(arg, params);
                }
            }
            Expr::Index(expr, indices) => {
                Self::collect_implicit_params(expr, params);
                for idx in indices {
                    Self::collect_implicit_params(idx, params);
                }
            }
            Expr::Block(stmts) => {
                for stmt in stmts {
                    Self::collect_implicit_params(stmt, params);
                }
            }
            Expr::Dict(keys, vals) => {
                Self::collect_implicit_params(keys, params);
                Self::collect_implicit_params(vals, params);
            }
            Expr::Cond(parts) => {
                for part in parts {
                    Self::collect_implicit_params(part, params);
                }
            }
            _ => {}
        }
    }

    /// Parse a conditional: `$[cond; true_branch; false_branch]`.
    fn parse_cond(&mut self) -> TdResult<Expr> {
        self.expect(&Token::Dollar)?;
        self.expect(&Token::LBrack)?;
        let mut parts = Vec::new();
        parts.push(self.parse_expr()?);
        while matches!(self.peek(), Token::Semi) {
            self.advance(); // consume ';'
            parts.push(self.parse_expr()?);
        }
        self.expect(&Token::RBrack)?;
        Ok(Expr::Cond(parts))
    }

    /// Check if the current token is a binary operator and return the BinOp.
    fn peek_binop(&self) -> Option<BinOp> {
        match self.peek() {
            Token::Plus => Some(BinOp::Add),
            Token::Minus => Some(BinOp::Sub),
            Token::Star => Some(BinOp::Mul),
            Token::Slash => Some(BinOp::Div),
            Token::Percent => Some(BinOp::Mod),
            Token::Eq => Some(BinOp::Eq),
            Token::Ne => Some(BinOp::Ne),
            Token::Lt => Some(BinOp::Lt),
            Token::Le => Some(BinOp::Le),
            Token::Gt => Some(BinOp::Gt),
            Token::Ge => Some(BinOp::Ge),
            _ => None,
        }
    }

    fn is_operator_token(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Plus
                | Token::Minus
                | Token::Star
                | Token::Slash
                | Token::Percent
                | Token::Eq
                | Token::Ne
                | Token::Lt
                | Token::Le
                | Token::Gt
                | Token::Ge
        )
    }

    fn parse_verb_name(name: &str) -> TdResult<Verb> {
        match name {
            "sum" => Ok(Verb::Sum),
            "avg" => Ok(Verb::Avg),
            "min" => Ok(Verb::Min),
            "max" => Ok(Verb::Max),
            "count" => Ok(Verb::Count),
            "first" => Ok(Verb::First),
            "last" => Ok(Verb::Last),
            "where" => Ok(Verb::Where),
            "til" => Ok(Verb::Til),
            "neg" => Ok(Verb::Neg),
            "abs" => Ok(Verb::Abs),
            "asc" => Ok(Verb::Asc),
            "desc" => Ok(Verb::Desc),
            "distinct" => Ok(Verb::Distinct),
            "sqrt" => Ok(Verb::Sqrt),
            "log" => Ok(Verb::Log),
            "exp" => Ok(Verb::Exp),
            "ceil" => Ok(Verb::Ceil),
            "floor" => Ok(Verb::Floor),
            "null" => Ok(Verb::IsNull),
            "cols" => Ok(Verb::Cols),
            "meta" => Ok(Verb::Meta),
            "key" => Ok(Verb::Key),
            "value" => Ok(Verb::ValueOf),
            "read" => Ok(Verb::Read),
            "load" => Ok(Verb::Load),
            "save" => Ok(Verb::Save),
            "enlist" => Ok(Verb::Enlist),
            _ => Err(TdError::Parse(format!("unknown verb: {}", name))),
        }
    }

    fn parse_adverb_name(name: &str) -> TdResult<Adverb> {
        match name {
            "over" => Ok(Adverb::Over),
            "scan" => Ok(Adverb::Scan),
            "each" => Ok(Adverb::Each),
            _ => Err(TdError::Parse(format!("unknown adverb: {}", name))),
        }
    }
}

/// Parse a Td source string into an Expr AST.
pub fn parse(input: &str) -> TdResult<Expr> {
    let tokens: Vec<Token> = Lexer::new(input).collect();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_program()?;
    Ok(expr)
}
