# Td Language Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a K-inspired, readable array language (Td) with a bytecode VM over the Teide C columnar engine.

**Architecture:** Lexer → Parser → AST → Compiler → Bytecode → Stack-based VM. Arithmetic and aggregate ops delegate to the C engine via `td_graph_t` / `td_execute`. Dicts and lambdas live in the VM layer. All code in `src/td/`.

**Tech Stack:** Rust, Teide C17 FFI (existing `src/ffi.rs` + `src/engine.rs`), reedline (REPL)

**Design doc:** `docs/plans/2026-02-22-td-language-design.md`

---

## Conventions

- Tests serialize via `ENGINE_LOCK` (see `tests/sql.rs`) because the C engine uses global state
- `Value::drop` calls `td_release`, `Value::clone` calls `td_retain`
- All files get the MIT license header matching `src/engine.rs`
- The `src/td/mod.rs` re-exports the public API

---

### Task 1: Error Type

**Files:**
- Create: `src/td/error.rs`
- Create: `src/td/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Write the failing test**

Create `tests/td.rs`:

```rust
use std::sync::Mutex;
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn td_error_display() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let e = teide::td::TdError::Parse("unexpected token".into());
    assert_eq!(e.to_string(), "parse error: unexpected token");

    let e2 = teide::td::TdError::Runtime("type mismatch".into());
    assert_eq!(e2.to_string(), "runtime error: type mismatch");

    let e3 = teide::td::TdError::Engine(teide::Error::Type);
    assert!(e3.to_string().contains("type error"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib --test td td_error_display -- --exact`
Expected: FAIL — `teide::td` module does not exist

**Step 3: Write minimal implementation**

Create `src/td/error.rs`:

```rust
use crate::Error;

#[derive(Debug)]
pub enum TdError {
    Parse(String),
    Compile(String),
    Runtime(String),
    Engine(Error),
}

impl std::fmt::Display for TdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TdError::Parse(msg) => write!(f, "parse error: {msg}"),
            TdError::Compile(msg) => write!(f, "compile error: {msg}"),
            TdError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            TdError::Engine(err) => write!(f, "engine error: {err}"),
        }
    }
}

impl std::error::Error for TdError {}

impl From<Error> for TdError {
    fn from(err: Error) -> Self {
        TdError::Engine(err)
    }
}

pub type TdResult<T> = Result<T, TdError>;
```

Create `src/td/mod.rs`:

```rust
pub mod error;

pub use error::{TdError, TdResult};
```

Add to `src/lib.rs`:

```rust
pub mod td;
```

**Step 4: Run test to verify it passes**

Run: `cargo test --lib --test td td_error_display -- --exact`
Expected: PASS

**Step 5: Commit**

```
git add src/td/ tests/td.rs src/lib.rs
git commit -m "feat(td): add TdError type"
```

---

### Task 2: Value Type

**Files:**
- Create: `src/td/value.rs`
- Modify: `src/td/mod.rs`
- Modify: `tests/td.rs`

**Step 1: Write the failing test**

Add to `tests/td.rs`:

```rust
#[test]
fn td_value_i64_roundtrip() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let v = teide::td::Value::i64(42);
    assert_eq!(v.as_i64(), Some(42));
}

#[test]
fn td_value_f64_roundtrip() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let v = teide::td::Value::f64(3.14);
    assert!((v.as_f64().unwrap() - 3.14).abs() < 1e-10);
}

#[test]
fn td_value_bool_roundtrip() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let v = teide::td::Value::bool(true);
    assert_eq!(v.as_bool(), Some(true));
}

#[test]
fn td_value_i64_vec() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let v = teide::td::Value::i64_vec(&[1, 2, 3]);
    assert_eq!(v.len(), Some(3));
    assert!(v.is_vec());
}

#[test]
fn td_value_nil() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let v = teide::td::Value::nil();
    assert!(v.is_nil());
}

#[test]
fn td_value_clone_retains() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let v = teide::td::Value::i64(42);
    let v2 = v.clone();
    assert_eq!(v.as_i64(), Some(42));
    assert_eq!(v2.as_i64(), Some(42));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib --test td td_value -- 2>&1 | head -20`
Expected: FAIL — `Value` does not exist

**Step 3: Write minimal implementation**

Create `src/td/value.rs`:

```rust
use crate::ffi;
use std::rc::Rc;

/// A Td runtime value.
///
/// Wraps Teide `td_t*` pointers with automatic retain/release.
/// Dicts and Lambdas are VM-level constructs.
#[derive(Debug)]
pub enum Value {
    /// Atom or vector backed by a C engine td_t*.
    Td(*mut ffi::td_t),
    /// Dictionary: keys (sym vector) and values (vector or list).
    Dict {
        keys: *mut ffi::td_t,
        vals: *mut ffi::td_t,
    },
    /// Lambda: compiled bytecode chunk + captured upvalues.
    Lambda(Rc<crate::td::chunk::Chunk>, Vec<Value>),
    /// Null / uninitialized.
    Nil,
}

impl Value {
    // -- Constructors --

    pub fn nil() -> Self {
        Value::Nil
    }

    pub fn i64(v: i64) -> Self {
        let ptr = unsafe { ffi::td_i64(v) };
        Value::Td(ptr)
    }

    pub fn f64(v: f64) -> Self {
        let ptr = unsafe { ffi::td_f64(v) };
        Value::Td(ptr)
    }

    pub fn bool(v: bool) -> Self {
        let ptr = unsafe { ffi::td_bool(v) };
        Value::Td(ptr)
    }

    pub fn i64_vec(data: &[i64]) -> Self {
        unsafe {
            let vec = ffi::td_vec_new(ffi::TD_I64, data.len() as i64);
            let dst = ffi::td_data(vec) as *mut i64;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            (*vec).val.len = data.len() as i64;
            Value::Td(vec)
        }
    }

    pub fn f64_vec(data: &[f64]) -> Self {
        unsafe {
            let vec = ffi::td_vec_new(ffi::TD_F64, data.len() as i64);
            let dst = ffi::td_data(vec) as *mut f64;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            (*vec).val.len = data.len() as i64;
            Value::Td(vec)
        }
    }

    pub fn table(t: crate::Table) -> Self {
        let raw = t.as_raw();
        unsafe { ffi::td_retain(raw) };
        // Table::drop will release once, our Value::drop will release again
        // so we retain an extra time here.
        Value::Td(raw)
    }

    /// Wrap a raw td_t* that is already retained (caller transfers ownership).
    ///
    /// # Safety
    /// `ptr` must be a valid, retained td_t* from the current engine runtime.
    pub unsafe fn from_raw(ptr: *mut ffi::td_t) -> Self {
        if ptr.is_null() || ffi::td_is_err(ptr) {
            Value::Nil
        } else {
            Value::Td(ptr)
        }
    }

    // -- Accessors --

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn is_vec(&self) -> bool {
        match self {
            Value::Td(ptr) => !ptr.is_null() && unsafe { ffi::td_is_vec(*ptr) },
            _ => false,
        }
    }

    pub fn is_atom(&self) -> bool {
        match self {
            Value::Td(ptr) => !ptr.is_null() && unsafe { ffi::td_is_atom(*ptr) },
            _ => false,
        }
    }

    pub fn is_table(&self) -> bool {
        match self {
            Value::Td(ptr) => {
                !ptr.is_null() && unsafe { ffi::td_type(*ptr as *const ffi::td_t) } == ffi::TD_TABLE
            }
            _ => false,
        }
    }

    pub fn type_tag(&self) -> Option<i8> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => {
                Some(unsafe { ffi::td_type(*ptr as *const ffi::td_t) })
            }
            _ => None,
        }
    }

    pub fn len(&self) -> Option<i64> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => {
                let t = unsafe { ffi::td_type(*ptr as *const ffi::td_t) };
                if t > 0 {
                    Some(unsafe { ffi::td_len(*ptr as *const ffi::td_t) })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => {
                let t = unsafe { ffi::td_type(*ptr as *const ffi::td_t) };
                if t == ffi::TD_ATOM_I64 {
                    Some(unsafe { (**ptr).val.i64_ })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => {
                let t = unsafe { ffi::td_type(*ptr as *const ffi::td_t) };
                if t == ffi::TD_ATOM_F64 {
                    Some(unsafe { (**ptr).val.f64_ })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => {
                let t = unsafe { ffi::td_type(*ptr as *const ffi::td_t) };
                if t == ffi::TD_ATOM_BOOL {
                    Some(unsafe { (**ptr).val.b8 } != 0)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get the raw td_t pointer (for passing to C engine).
    pub fn as_raw(&self) -> Option<*mut ffi::td_t> {
        match self {
            Value::Td(ptr) if !ptr.is_null() => Some(*ptr),
            _ => None,
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Td(ptr) => {
                if !ptr.is_null() && !ffi::td_is_err(*ptr) {
                    unsafe { ffi::td_retain(*ptr) };
                }
                Value::Td(*ptr)
            }
            Value::Dict { keys, vals } => {
                if !keys.is_null() {
                    unsafe { ffi::td_retain(*keys) };
                }
                if !vals.is_null() {
                    unsafe { ffi::td_retain(*vals) };
                }
                Value::Dict {
                    keys: *keys,
                    vals: *vals,
                }
            }
            Value::Lambda(chunk, upvals) => Value::Lambda(chunk.clone(), upvals.clone()),
            Value::Nil => Value::Nil,
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        match self {
            Value::Td(ptr) => {
                if !ptr.is_null() && !ffi::td_is_err(*ptr) {
                    unsafe { ffi::td_release(*ptr) };
                }
            }
            Value::Dict { keys, vals } => {
                if !keys.is_null() && !ffi::td_is_err(*keys) {
                    unsafe { ffi::td_release(*keys) };
                }
                if !vals.is_null() && !ffi::td_is_err(*vals) {
                    unsafe { ffi::td_release(*vals) };
                }
            }
            _ => {}
        }
    }
}
```

Note: `Value::Lambda` references `crate::td::chunk::Chunk` which doesn't exist yet.
Create a stub `src/td/chunk.rs`:

```rust
/// Compiled bytecode chunk. Placeholder — fleshed out in Task 4.
#[derive(Debug)]
pub struct Chunk;
```

Update `src/td/mod.rs`:

```rust
pub mod chunk;
pub mod error;
pub mod value;

pub use error::{TdError, TdResult};
pub use value::Value;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib --test td td_value -- -v`
Expected: all 6 tests PASS

**Step 5: Commit**

```
git add src/td/value.rs src/td/chunk.rs src/td/mod.rs tests/td.rs
git commit -m "feat(td): add Value type with retain/release bridge"
```

---

### Task 3: Lexer

**Files:**
- Create: `src/td/token.rs`
- Create: `src/td/lexer.rs`
- Modify: `src/td/mod.rs`
- Modify: `tests/td.rs`

**Step 1: Write the failing test**

Add to `tests/td.rs`:

```rust
use teide::td::token::Token;
use teide::td::lexer::Lexer;

#[test]
fn td_lex_assignment() {
    let tokens: Vec<Token> = Lexer::new("a: 42").collect();
    assert_eq!(tokens, vec![
        Token::Ident("a".into()),
        Token::Colon,
        Token::Int(42),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_vector() {
    let tokens: Vec<Token> = Lexer::new("1 2 3").collect();
    assert_eq!(tokens, vec![
        Token::Int(1),
        Token::Int(2),
        Token::Int(3),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_symbols() {
    let tokens: Vec<Token> = Lexer::new("`a`b`c").collect();
    assert_eq!(tokens, vec![
        Token::Sym("a".into()),
        Token::Sym("b".into()),
        Token::Sym("c".into()),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_operators() {
    let tokens: Vec<Token> = Lexer::new("a + b * c").collect();
    assert_eq!(tokens, vec![
        Token::Ident("a".into()),
        Token::Plus,
        Token::Ident("b".into()),
        Token::Star,
        Token::Ident("c".into()),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_lambda() {
    let tokens: Vec<Token> = Lexer::new("{x + y}").collect();
    assert_eq!(tokens, vec![
        Token::LBrace,
        Token::Ident("x".into()),
        Token::Plus,
        Token::Ident("y".into()),
        Token::RBrace,
        Token::Eof,
    ]);
}

#[test]
fn td_lex_string() {
    let tokens: Vec<Token> = Lexer::new("\"hello\"").collect();
    assert_eq!(tokens, vec![
        Token::Str("hello".into()),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_comparison() {
    let tokens: Vec<Token> = Lexer::new("x >= 5").collect();
    assert_eq!(tokens, vec![
        Token::Ident("x".into()),
        Token::Ge,
        Token::Int(5),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_comment() {
    let tokens: Vec<Token> = Lexer::new("a: 1 / this is a comment").collect();
    assert_eq!(tokens, vec![
        Token::Ident("a".into()),
        Token::Colon,
        Token::Int(1),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_semicolons() {
    let tokens: Vec<Token> = Lexer::new("a: 1; b: 2").collect();
    assert_eq!(tokens, vec![
        Token::Ident("a".into()),
        Token::Colon,
        Token::Int(1),
        Token::Semi,
        Token::Ident("b".into()),
        Token::Colon,
        Token::Int(2),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_float() {
    let tokens: Vec<Token> = Lexer::new("3.14").collect();
    assert_eq!(tokens, vec![Token::Float(3.14), Token::Eof]);
}

#[test]
fn td_lex_bool() {
    let tokens: Vec<Token> = Lexer::new("1b 0b").collect();
    assert_eq!(tokens, vec![
        Token::Bool(true),
        Token::Bool(false),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_negative_number() {
    let tokens: Vec<Token> = Lexer::new("-5").collect();
    assert_eq!(tokens, vec![Token::Int(-5), Token::Eof]);
}

#[test]
fn td_lex_verbs() {
    let tokens: Vec<Token> = Lexer::new("sum avg count").collect();
    assert_eq!(tokens, vec![
        Token::Verb("sum".into()),
        Token::Verb("avg".into()),
        Token::Verb("count".into()),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_adverbs() {
    let tokens: Vec<Token> = Lexer::new("+ over 1 2 3").collect();
    assert_eq!(tokens, vec![
        Token::Plus,
        Token::Adverb("over".into()),
        Token::Int(1),
        Token::Int(2),
        Token::Int(3),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_select() {
    let tokens: Vec<Token> = Lexer::new("select from t").collect();
    assert_eq!(tokens, vec![
        Token::Select,
        Token::From,
        Token::Ident("t".into()),
        Token::Eof,
    ]);
}

#[test]
fn td_lex_bang() {
    let tokens: Vec<Token> = Lexer::new("`a`b ! 1 2").collect();
    assert_eq!(tokens, vec![
        Token::Sym("a".into()),
        Token::Sym("b".into()),
        Token::Bang,
        Token::Int(1),
        Token::Int(2),
        Token::Eof,
    ]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib --test td td_lex -- 2>&1 | head -5`
Expected: FAIL — modules don't exist

**Step 3: Write implementation**

Create `src/td/token.rs` with the Token enum (all variants used above).
Create `src/td/lexer.rs` with a `Lexer` struct implementing `Iterator<Item=Token>`.

The lexer must:
- Recognize keywords: `sum`, `avg`, `count`, `min`, `max`, `first`, `last`, `where`, `til`, `neg`, `abs`, `asc`, `desc`, `distinct`, `read`, `load`, `save`, `cols`, `meta`, `key`, `value` → `Token::Verb`
- Recognize adverbs: `over`, `scan`, `each` → `Token::Adverb`
- Recognize query words: `select` → `Token::Select`, `update` → `Token::Update`, `delete` → `Token::Delete`, `from` → `Token::From`, `where` (context: after `from`) → `Token::Where`, `by` → `Token::By`
- `/ ` (slash-space) starts a line comment
- `/` without space is `Token::Slash` (divide)
- `!` is `Token::Bang` (dict constructor)
- Handle negative numbers: `-` followed immediately by digit → negative literal
- `1b` / `0b` → `Token::Bool`
- Backtick symbols: `` `name `` → `Token::Sym("name")`

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib --test td td_lex -- -v`
Expected: all lexer tests PASS

**Step 5: Commit**

```
git add src/td/token.rs src/td/lexer.rs src/td/mod.rs tests/td.rs
git commit -m "feat(td): add lexer and token types"
```

---

### Task 4: AST

**Files:**
- Create: `src/td/ast.rs`
- Modify: `src/td/mod.rs`

**Step 1: Define the AST types**

No test for this step — it's pure type definitions. Create `src/td/ast.rs` with:

```rust
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
```

Update `src/td/mod.rs` to include `pub mod ast;`.

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: success

**Step 3: Commit**

```
git add src/td/ast.rs src/td/mod.rs
git commit -m "feat(td): add AST types"
```

---

### Task 5: Parser (Expressions)

**Files:**
- Create: `src/td/parser.rs`
- Modify: `src/td/mod.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

Add to `tests/td.rs`:

```rust
use teide::td::ast::*;
use teide::td::parser::parse;

#[test]
fn td_parse_int() {
    let expr = parse("42").unwrap();
    assert_eq!(expr, Expr::Int(42));
}

#[test]
fn td_parse_assignment() {
    let expr = parse("a: 42").unwrap();
    assert_eq!(expr, Expr::Assign("a".into(), Box::new(Expr::Int(42))));
}

#[test]
fn td_parse_binop() {
    let expr = parse("1 + 2").unwrap();
    assert_eq!(expr, Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Int(1)),
        Box::new(Expr::Int(2)),
    ));
}

#[test]
fn td_parse_right_to_left() {
    // 1 + 2 * 3 in right-to-left = 1 + (2 * 3) = 1 + 6 = 7
    // But right-to-left means: + takes everything to its right as operand
    // So: 1 + 2 * 3 => add(1, mul(2, 3))
    let expr = parse("1 + 2 * 3").unwrap();
    assert_eq!(expr, Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Int(1)),
        Box::new(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Int(2)),
            Box::new(Expr::Int(3)),
        )),
    ));
}

#[test]
fn td_parse_verb() {
    let expr = parse("sum 1 2 3").unwrap();
    assert_eq!(expr, Expr::Verb(
        Verb::Sum,
        vec![Expr::Vector(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)])],
    ));
}

#[test]
fn td_parse_lambda() {
    let expr = parse("{x + y}").unwrap();
    assert_eq!(expr, Expr::Lambda {
        params: vec!["x".into(), "y".into()],
        body: Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Ident("x".into())),
            Box::new(Expr::Ident("y".into())),
        )),
    });
}

#[test]
fn td_parse_explicit_lambda() {
    let expr = parse("{[a;b] a + b}").unwrap();
    assert_eq!(expr, Expr::Lambda {
        params: vec!["a".into(), "b".into()],
        body: Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Ident("a".into())),
            Box::new(Expr::Ident("b".into())),
        )),
    });
}

#[test]
fn td_parse_block() {
    let expr = parse("a: 1; b: 2").unwrap();
    assert!(matches!(expr, Expr::Block(_)));
}

#[test]
fn td_parse_dict() {
    let expr = parse("`a`b ! 1 2").unwrap();
    assert_eq!(expr, Expr::Dict(
        Box::new(Expr::Vector(vec![Expr::Sym("a".into()), Expr::Sym("b".into())])),
        Box::new(Expr::Vector(vec![Expr::Int(1), Expr::Int(2)])),
    ));
}

#[test]
fn td_parse_index() {
    let expr = parse("a[0]").unwrap();
    assert_eq!(expr, Expr::Index(
        Box::new(Expr::Ident("a".into())),
        vec![Expr::Int(0)],
    ));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib --test td td_parse -- 2>&1 | head -5`
Expected: FAIL

**Step 3: Write implementation**

Create `src/td/parser.rs`. The parser consumes tokens from the lexer and produces `Expr`.

Key parsing rules (right-to-left):
- Parse from left to right through tokens, but binary operators bind to everything to their right
- Verbs consume the expression to their right as argument
- Adverbs modify the verb to their left
- `{...}` is a lambda — scan body for implicit x/y/z
- `[...]` after an expression is indexing / function call
- Semicolons separate expressions in a block
- `!` is the dict constructor (binary, infix)

Public API: `pub fn parse(input: &str) -> TdResult<Expr>`

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib --test td td_parse -- -v`
Expected: PASS

**Step 5: Commit**

```
git add src/td/parser.rs src/td/mod.rs tests/td.rs
git commit -m "feat(td): add parser for expressions, lambdas, dicts"
```

---

### Task 6: Bytecode Chunk & Compiler

**Files:**
- Modify: `src/td/chunk.rs` (replace stub)
- Create: `src/td/compiler.rs`
- Modify: `src/td/mod.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

Add to `tests/td.rs`:

```rust
use teide::td::compiler::compile;
use teide::td::chunk::Op;

#[test]
fn td_compile_int_literal() {
    let chunk = compile("42").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
}

#[test]
fn td_compile_add() {
    let chunk = compile("1 + 2").unwrap();
    assert!(chunk.code.contains(&(Op::Add as u8)));
}
```

**Step 2: Write implementation**

Replace `src/td/chunk.rs` with opcode enum, constant pool, and Chunk struct.
Create `src/td/compiler.rs` — walks the AST, emits bytecodes into a Chunk.

**Step 3: Run tests, verify pass, commit**

```
git add src/td/chunk.rs src/td/compiler.rs src/td/mod.rs tests/td.rs
git commit -m "feat(td): add bytecode chunk and compiler"
```

---

### Task 7: VM Core (Arithmetic)

**Files:**
- Create: `src/td/vm.rs`
- Modify: `src/td/mod.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
use teide::td::vm::Vm;

#[test]
fn td_vm_int_literal() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("42").unwrap();
    assert_eq!(result.as_i64(), Some(42));
}

#[test]
fn td_vm_add_atoms() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("1 + 2").unwrap();
    assert_eq!(result.as_i64(), Some(3));
}

#[test]
fn td_vm_add_vectors() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("1 2 3 + 4 5 6").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(3));
}

#[test]
fn td_vm_assignment() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("a: 10").unwrap();
    let result = vm.eval("a + 5").unwrap();
    assert_eq!(result.as_i64(), Some(15));
}

#[test]
fn td_vm_vector_literal() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("1 2 3 4 5").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(5));
}
```

**Step 2: Write implementation**

Create `src/td/vm.rs` with:
- `Vm` struct: globals HashMap, value stack
- `eval(&mut self, input: &str) -> TdResult<Value>`: lex → parse → compile → execute
- Execute loop: match on opcodes, dispatch arithmetic to C engine
- For atom+atom: call `td_i64(a + b)` etc. directly
- For vec ops: build a `td_graph_t`, add `td_add`/`td_sub`/etc., call `td_execute`

**Step 3: Run tests, verify pass, commit**

```
git add src/td/vm.rs src/td/mod.rs tests/td.rs
git commit -m "feat(td): add VM with arithmetic ops"
```

---

### Task 8: Built-in Verbs

**Files:**
- Modify: `src/td/vm.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn td_vm_sum() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("sum 1 2 3 4 5").unwrap();
    assert_eq!(result.as_i64(), Some(15));
}

#[test]
fn td_vm_count() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("count 1 2 3").unwrap();
    assert_eq!(result.as_i64(), Some(3));
}

#[test]
fn td_vm_til() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("til 5").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(5));
}

#[test]
fn td_vm_where_filter() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("a: 1 2 3 4 5").unwrap();
    let result = vm.eval("a where a > 3").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(2)); // 4, 5
}
```

**Step 2: Implement verb dispatch in VM**

Each verb maps to a C engine reduction (`td_sum`, `td_avg`, etc.) or a Rust-level operation (`til`, `where`).

**Step 3: Run tests, verify pass, commit**

```
git add src/td/vm.rs tests/td.rs
git commit -m "feat(td): add built-in verbs (sum, avg, count, til, where, etc.)"
```

---

### Task 9: Adverbs (over, scan, each)

**Files:**
- Modify: `src/td/vm.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn td_vm_over() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("+ over 1 2 3 4").unwrap();
    assert_eq!(result.as_i64(), Some(10));
}

#[test]
fn td_vm_scan() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("+ scan 1 2 3 4").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(4));
}
```

**Step 2: Implement adverb execution**

- `over`: fold the vector with the binary op
- `scan`: running fold producing a vector
- `each`: apply a unary verb/lambda to each element

**Step 3: Run tests, verify pass, commit**

```
git add src/td/vm.rs tests/td.rs
git commit -m "feat(td): add adverbs (over, scan, each)"
```

---

### Task 10: Lambdas & Closures

**Files:**
- Modify: `src/td/vm.rs`
- Modify: `src/td/compiler.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn td_vm_lambda_call() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("{x * x} 5").unwrap();
    assert_eq!(result.as_i64(), Some(25));
}

#[test]
fn td_vm_lambda_two_args() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("{x + y}[3;4]").unwrap();
    assert_eq!(result.as_i64(), Some(7));
}

#[test]
fn td_vm_named_lambda() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("f: {x * x}").unwrap();
    let result = vm.eval("f 5").unwrap();
    assert_eq!(result.as_i64(), Some(25));
}

#[test]
fn td_vm_lambda_with_each() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("{x * x} each 1 2 3").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(3));
}
```

**Step 2: Implement lambda compilation and VM call frames**

The compiler emits a sub-Chunk for the lambda body. The VM pushes a call frame and binds args to locals.

**Step 3: Run tests, verify pass, commit**

```
git add src/td/vm.rs src/td/compiler.rs tests/td.rs
git commit -m "feat(td): add lambdas and closures"
```

---

### Task 11: Dictionaries

**Files:**
- Modify: `src/td/vm.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn td_vm_dict_create_and_lookup() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("d: `a`b`c ! 1 2 3").unwrap();
    let result = vm.eval("d`b").unwrap();
    assert_eq!(result.as_i64(), Some(2));
}

#[test]
fn td_vm_dict_bracket_lookup() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("d: `a`b`c ! 1 2 3").unwrap();
    let result = vm.eval("d[`a]").unwrap();
    assert_eq!(result.as_i64(), Some(1));
}
```

**Step 2: Implement dict construction and lookup**

**Step 3: Run tests, verify pass, commit**

```
git add src/td/vm.rs tests/td.rs
git commit -m "feat(td): add dictionary support"
```

---

### Task 12: Query Templates (select/from/where)

**Files:**
- Modify: `src/td/parser.rs`
- Modify: `src/td/compiler.rs`
- Modify: `src/td/vm.rs`
- Modify: `tests/td.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn td_vm_select_from_table() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("t: read \"tests/data/small.csv\"").unwrap();
    let result = vm.eval("select from t").unwrap();
    assert!(result.is_table());
}

#[test]
fn td_vm_select_where() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("t: read \"tests/data/small.csv\"").unwrap();
    let result = vm.eval("select from t where v1 > 3").unwrap();
    assert!(result.is_table());
}

#[test]
fn td_vm_select_group_by() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    vm.eval("t: read \"tests/data/small.csv\"").unwrap();
    let result = vm.eval("select sum v1 by id1 from t").unwrap();
    assert!(result.is_table());
}
```

Note: needs a small test CSV at `tests/data/small.csv`. Create it in the test setup or as a fixture.

**Step 2: Implement query parsing and compilation**

Parser: detect `select`/`update`/`delete` keywords and parse the Q-style template into `Expr::Select` etc.
Compiler: emit specialized bytecodes that build a `td_graph_t`.
VM: execute graph ops via `td_filter`, `td_group`, `td_project`, `td_execute`.

**Step 3: Run tests, verify pass, commit**

```
git add src/td/parser.rs src/td/compiler.rs src/td/vm.rs tests/td.rs tests/data/
git commit -m "feat(td): add query templates (select/update/delete)"
```

---

### Task 13: REPL

**Files:**
- Create: `src/td/repl.rs`
- Modify: `Cargo.toml`

**Step 1: Create the REPL entry point**

Create `src/td/repl.rs` as a binary that:
- Initializes `Context`
- Creates a `Vm`
- Uses reedline for line editing (reuse patterns from `src/cli/main.rs`)
- Prints the Td banner
- Reads input, calls `vm.eval()`, prints results
- Handles `\\` (quit) and `\h` (help)

Add to `Cargo.toml`:

```toml
[[bin]]
name = "td"
path = "src/td/repl.rs"
required-features = ["cli"]
```

**Step 2: Build and smoke test**

Run: `cargo build --features cli --bin td`
Expected: compiles

Run manually: `echo "1 + 2" | cargo run --features cli --bin td`
Expected: prints `3`

**Step 3: Commit**

```
git add src/td/repl.rs Cargo.toml
git commit -m "feat(td): add interactive REPL"
```

---

## Summary

| Task | What | Key files |
|------|------|-----------|
| 1 | Error type | `error.rs`, `mod.rs` |
| 2 | Value type | `value.rs`, `chunk.rs` stub |
| 3 | Lexer | `token.rs`, `lexer.rs` |
| 4 | AST | `ast.rs` |
| 5 | Parser | `parser.rs` |
| 6 | Bytecode + Compiler | `chunk.rs`, `compiler.rs` |
| 7 | VM (arithmetic) | `vm.rs` |
| 8 | Built-in verbs | `vm.rs` |
| 9 | Adverbs | `vm.rs` |
| 10 | Lambdas | `vm.rs`, `compiler.rs` |
| 11 | Dictionaries | `vm.rs` |
| 12 | Query templates | `parser.rs`, `compiler.rs`, `vm.rs` |
| 13 | REPL | `repl.rs`, `Cargo.toml` |
