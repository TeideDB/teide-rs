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
