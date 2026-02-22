use std::sync::Mutex;
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

use teide::td::token::Token;
use teide::td::lexer::Lexer;

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
