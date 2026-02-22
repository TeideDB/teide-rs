#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Sym(String),
    Str(String),
    Nil,

    Vector(Vec<Expr>),
    Dict(Box<Expr>, Box<Expr>),

    Ident(String),
    Assign(String, Box<Expr>),

    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
    Verb(Verb, Vec<Expr>),
    Adverb(Adverb, Box<Expr>, Box<Expr>),

    Lambda { params: Vec<String>, body: Box<Expr> },
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Vec<Expr>),

    Select { cols: Vec<SelectCol>, by: Vec<SelectCol>, from: Box<Expr>, wheres: Vec<Expr> },
    Update { cols: Vec<SelectCol>, from: Box<Expr>, wheres: Vec<Expr> },
    Delete { from: Box<Expr>, wheres: Vec<Expr> },

    Cond(Vec<Expr>),
    Block(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectCol {
    pub alias: Option<String>,
    pub expr: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    Min2, Max2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verb {
    Sum, Avg, Min, Max, Count, First, Last,
    Where, Til, Neg, Abs, Asc, Desc, Distinct,
    Sqrt, Log, Exp, Ceil, Floor, IsNull,
    Cols, Meta, Key, ValueOf,
    Read, Load, Save,
    Enlist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adverb {
    Over, Scan, Each,
}
