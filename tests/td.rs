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
