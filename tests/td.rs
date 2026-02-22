use std::sync::Mutex;
static ENGINE_LOCK: Mutex<()> = Mutex::new(());

use teide::td::token::Token;
use teide::td::lexer::Lexer;
use teide::td::ast::*;
use teide::td::parser::parse;
use teide::td::compiler::compile;
use teide::td::chunk::{Op, Const};
use teide::td::vm::Vm;

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
    // Right-to-left: 1 + 2 * 3 => add(1, mul(2, 3))
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

// ---- Compiler tests ----

#[test]
fn td_compile_int_literal() {
    let chunk = compile("42").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants.len(), 1);
    assert_eq!(chunk.constants[0], Const::Int(42));
}

#[test]
fn td_compile_add() {
    let chunk = compile("1 + 2").unwrap();
    assert!(chunk.code.contains(&(Op::Add as u8)));
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants.len(), 2);
    assert_eq!(chunk.constants[0], Const::Int(1));
    assert_eq!(chunk.constants[1], Const::Int(2));
}

#[test]
fn td_compile_float_literal() {
    let chunk = compile("3.14").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants[0], Const::Float(3.14));
}

#[test]
fn td_compile_bool_literal() {
    let chunk = compile("1b").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants[0], Const::Bool(true));
}

#[test]
fn td_compile_string_literal() {
    let chunk = compile("\"hello\"").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants[0], Const::Str("hello".into()));
}

#[test]
fn td_compile_symbol_literal() {
    let chunk = compile("`abc").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    assert_eq!(chunk.constants[0], Const::Sym("abc".into()));
}

#[test]
fn td_compile_nil() {
    let chunk = compile("").unwrap();
    assert!(chunk.code.contains(&(Op::Nil as u8)));
}

#[test]
fn td_compile_vector() {
    let chunk = compile("1 2 3").unwrap();
    assert!(chunk.code.contains(&(Op::MakeVec as u8)));
    assert_eq!(chunk.constants.len(), 3);
}

#[test]
fn td_compile_assignment() {
    let chunk = compile("a: 42").unwrap();
    assert!(chunk.code.contains(&(Op::StoreGlobal as u8)));
}

#[test]
fn td_compile_ident() {
    let chunk = compile("a").unwrap();
    assert!(chunk.code.contains(&(Op::LoadGlobal as u8)));
}

#[test]
fn td_compile_unary_neg() {
    let _chunk = compile("(- 5)").unwrap();
    // The parser parses -5 as Int(-5), but -(expr) uses UnaryOp::Neg
    // We test with a form that definitely produces a UnaryOp
    let chunk2 = compile("(- a)").unwrap();
    assert!(chunk2.code.contains(&(Op::Neg as u8)));
}

#[test]
fn td_compile_verb_sum() {
    let chunk = compile("sum 1 2 3").unwrap();
    assert!(chunk.code.contains(&(Op::Sum as u8)));
}

#[test]
fn td_compile_lambda() {
    let chunk = compile("{x + y}").unwrap();
    assert!(chunk.code.contains(&(Op::Const as u8)));
    // The lambda body should be stored as a Const::Chunk
    assert!(matches!(chunk.constants.last(), Some(Const::Chunk(_))));
}

#[test]
fn td_compile_call() {
    let chunk = compile("{x + y}").unwrap();
    // Lambda itself is a Const::Chunk; verify the sub-chunk
    if let Some(Const::Chunk(sub)) = chunk.constants.last() {
        assert!(sub.code.contains(&(Op::Add as u8)));
        assert!(sub.code.contains(&(Op::Return as u8)));
        assert_eq!(sub.num_locals, 2); // x and y
    } else {
        panic!("expected Const::Chunk for lambda body");
    }
}

#[test]
fn td_compile_index() {
    let chunk = compile("a[0]").unwrap();
    assert!(chunk.code.contains(&(Op::Index as u8)));
}

#[test]
fn td_compile_block() {
    let chunk = compile("a: 1; b: 2").unwrap();
    assert!(chunk.code.contains(&(Op::Pop as u8)));
    assert!(chunk.code.contains(&(Op::StoreGlobal as u8)));
}

#[test]
fn td_compile_dict() {
    let chunk = compile("`a`b ! 1 2").unwrap();
    assert!(chunk.code.contains(&(Op::MakeDict as u8)));
}

#[test]
fn td_compile_comparison() {
    let chunk = compile("1 = 2").unwrap();
    assert!(chunk.code.contains(&(Op::Eq as u8)));
}

#[test]
fn td_compile_returns() {
    // Every compiled chunk should end with Op::Return
    let chunk = compile("42").unwrap();
    assert_eq!(*chunk.code.last().unwrap(), Op::Return as u8);
}

#[test]
fn td_compile_cond() {
    let chunk = compile("$[1b; 42; 0]").unwrap();
    assert!(chunk.code.contains(&(Op::JumpIfFalse as u8)));
    assert!(chunk.code.contains(&(Op::Jump as u8)));
}

#[test]
fn td_compile_sub_expr() {
    let chunk = compile("1 + 2 * 3").unwrap();
    // Right-to-left: add(1, mul(2,3))
    // Should have: Const(1), Const(2), Const(3), Mul, Add, Return
    assert!(chunk.code.contains(&(Op::Add as u8)));
    assert!(chunk.code.contains(&(Op::Mul as u8)));
    assert_eq!(chunk.constants.len(), 3);
}

// ---- VM tests ----

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

#[test]
fn td_vm_avg() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("avg 2 4 6").unwrap();
    assert!((result.as_f64().unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn td_vm_min() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("min 3 1 4 1 5").unwrap();
    assert_eq!(result.as_i64(), Some(1));
}

#[test]
fn td_vm_max() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("max 3 1 4 1 5").unwrap();
    assert_eq!(result.as_i64(), Some(5));
}

#[test]
fn td_vm_first() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("first 10 20 30").unwrap();
    assert_eq!(result.as_i64(), Some(10));
}

#[test]
fn td_vm_last() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("last 10 20 30").unwrap();
    assert_eq!(result.as_i64(), Some(30));
}

#[test]
fn td_vm_asc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("asc 3 1 4 1 5").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(5));
}

#[test]
fn td_vm_desc() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("desc 3 1 4 1 5").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(5));
}

#[test]
fn td_vm_distinct() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("distinct 1 2 2 3 3 3").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(3)); // 1, 2, 3
}

// ---- Adverb tests ----

#[test]
fn td_vm_over() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("+ over 1 2 3 4").unwrap();
    assert_eq!(result.as_i64(), Some(10));
}

#[test]
fn td_vm_over_mul() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    let result = vm.eval("* over 1 2 3 4").unwrap();
    assert_eq!(result.as_i64(), Some(24));
}

#[test]
fn td_vm_over_single() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    // Over on a single-element vector returns that element
    let result = vm.eval("+ over 42").unwrap();
    assert_eq!(result.as_i64(), Some(42));
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

#[test]
fn td_vm_scan_values() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    // + scan 1 2 3 4 = 1 3 6 10
    let result = vm.eval("+ scan 1 2 3 4").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(4));
    // Verify the first element (1) and last element (10)
    let first = vm.eval("first + scan 1 2 3 4").unwrap();
    assert_eq!(first.as_i64(), Some(1));
    let last = vm.eval("last + scan 1 2 3 4").unwrap();
    assert_eq!(last.as_i64(), Some(10));
}

#[test]
fn td_vm_each_neg() {
    let _guard = ENGINE_LOCK.lock().unwrap();
    let _ctx = teide::Context::new().unwrap();
    let mut vm = Vm::new();
    // neg each 1 2 3 => -1 -2 -3
    let result = vm.eval("neg each 1 2 3").unwrap();
    assert!(result.is_vec());
    assert_eq!(result.len(), Some(3));
}

// ---- Lambda / closure tests ----

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
