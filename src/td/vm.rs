use std::collections::HashMap;
use crate::td::chunk::{Chunk, Const, Op};
use crate::td::compiler;
use crate::td::value::Value;
use crate::td::error::{TdError, TdResult};
use crate::ffi;

/// Read a big-endian u16 from bytecode and advance ip.
fn read_u16(code: &[u8], ip: &mut usize) -> u16 {
    let v = (code[*ip] as u16) << 8 | code[*ip + 1] as u16;
    *ip += 2;
    v
}

pub struct Vm {
    globals: HashMap<String, Value>,
}

impl Vm {
    pub fn new() -> Self {
        Vm {
            globals: HashMap::new(),
        }
    }

    pub fn eval(&mut self, input: &str) -> TdResult<Value> {
        let chunk = compiler::compile(input)?;
        self.execute(&chunk)
    }

    fn execute(&mut self, chunk: &Chunk) -> TdResult<Value> {
        let mut stack: Vec<Value> = Vec::new();
        let mut ip: usize = 0;

        while ip < chunk.code.len() {
            let op = Op::from_u8(chunk.code[ip])
                .ok_or_else(|| TdError::Runtime(format!("unknown opcode: {}", chunk.code[ip])))?;
            ip += 1;

            match op {
                Op::Const => {
                    let idx = read_u16(&chunk.code, &mut ip);
                    let val = self.const_to_value(&chunk.constants[idx as usize])?;
                    stack.push(val);
                }
                Op::Nil => stack.push(Value::nil()),
                Op::Pop => { stack.pop(); }
                Op::LoadLocal => {
                    let slot = chunk.code[ip]; ip += 1;
                    let val = stack.get(slot as usize)
                        .ok_or_else(|| TdError::Runtime(format!("invalid local slot: {slot}")))?
                        .clone();
                    stack.push(val);
                }
                Op::StoreLocal => {
                    let slot = chunk.code[ip]; ip += 1;
                    let val = stack.last()
                        .ok_or_else(|| TdError::Runtime("empty stack".into()))?
                        .clone();
                    if (slot as usize) < stack.len() {
                        stack[slot as usize] = val;
                    }
                }
                Op::LoadGlobal => {
                    let idx = read_u16(&chunk.code, &mut ip);
                    let name = self.const_to_string(&chunk.constants[idx as usize]);
                    let val = self.globals.get(&name)
                        .ok_or_else(|| TdError::Runtime(format!("undefined: {name}")))?
                        .clone();
                    stack.push(val);
                }
                Op::StoreGlobal => {
                    let idx = read_u16(&chunk.code, &mut ip);
                    let name = self.const_to_string(&chunk.constants[idx as usize]);
                    let val = stack.last()
                        .ok_or_else(|| TdError::Runtime("empty stack".into()))?
                        .clone();
                    self.globals.insert(name, val);
                }
                Op::Add => binary_arith(&mut stack, "add")?,
                Op::Sub => binary_arith(&mut stack, "sub")?,
                Op::Mul => binary_arith(&mut stack, "mul")?,
                Op::Div => binary_arith(&mut stack, "div")?,
                Op::Mod => binary_arith(&mut stack, "mod")?,
                Op::Eq => binary_arith(&mut stack, "eq")?,
                Op::Ne => binary_arith(&mut stack, "ne")?,
                Op::Lt => binary_arith(&mut stack, "lt")?,
                Op::Le => binary_arith(&mut stack, "le")?,
                Op::Gt => binary_arith(&mut stack, "gt")?,
                Op::Ge => binary_arith(&mut stack, "ge")?,
                Op::Neg => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    if val.is_atom() {
                        if let Some(v) = val.as_i64() {
                            stack.push(Value::i64(-v));
                        } else if let Some(v) = val.as_f64() {
                            stack.push(Value::f64(-v));
                        } else {
                            return Err(TdError::Runtime("neg: unsupported type".into()));
                        }
                    } else if val.is_vec() {
                        let result = unary_vec_op(&val, "neg")?;
                        stack.push(result);
                    } else {
                        return Err(TdError::Runtime("neg: unsupported type".into()));
                    }
                }
                Op::Not => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    if val.is_atom() {
                        if let Some(v) = val.as_bool() {
                            stack.push(Value::bool(!v));
                        } else if let Some(v) = val.as_i64() {
                            stack.push(Value::bool(v == 0));
                        } else {
                            return Err(TdError::Runtime("not: unsupported type".into()));
                        }
                    } else {
                        return Err(TdError::Runtime("not: unsupported type for atom".into()));
                    }
                }
                Op::MakeVec => {
                    let count = chunk.code[ip] as usize; ip += 1;
                    make_vec(&mut stack, count)?;
                }
                Op::Return => break,

                // ---- Built-in verbs ----
                Op::Sum => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_sum(&val)?);
                }
                Op::Avg => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_avg(&val)?);
                }
                Op::Min => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_min(&val)?);
                }
                Op::Max => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_max(&val)?);
                }
                Op::Count => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_count(&val)?);
                }
                Op::First => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_first(&val)?);
                }
                Op::Last => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_last(&val)?);
                }
                Op::Til => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_til(&val)?);
                }
                Op::Where => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_where(&val)?);
                }
                Op::Asc => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_asc(&val)?);
                }
                Op::Desc => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_desc(&val)?);
                }
                Op::Distinct => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_distinct(&val)?);
                }
                Op::Enlist => {
                    let val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    // Enlist wraps an atom into a 1-element vector
                    if val.is_atom() {
                        if let Some(v) = val.as_i64() {
                            stack.push(Value::i64_vec(&[v]));
                        } else if let Some(v) = val.as_f64() {
                            stack.push(Value::f64_vec(&[v]));
                        } else {
                            stack.push(val);
                        }
                    } else {
                        stack.push(val); // already a vector
                    }
                }

                // ---- Indexing ----
                Op::Index => {
                    let idx_val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    let src = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    stack.push(verb_index(&src, &idx_val)?);
                }

                // ---- Jump / Control ----
                Op::Jump => {
                    let offset = read_u16(&chunk.code, &mut ip) as i16;
                    let target = (ip as isize + offset as isize - 3) as usize;
                    ip = target;
                }
                Op::JumpIfFalse => {
                    let offset = read_u16(&chunk.code, &mut ip) as i16;
                    let cond = stack.pop()
                        .ok_or_else(|| TdError::Runtime("stack underflow".into()))?;
                    let is_false = if let Some(b) = cond.as_bool() {
                        !b
                    } else if let Some(v) = cond.as_i64() {
                        v == 0
                    } else {
                        cond.is_nil()
                    };
                    if is_false {
                        let target = (ip as isize + offset as isize - 3) as usize;
                        ip = target;
                    }
                }

                // ---- Adverbs ----
                Op::Over => {
                    let op_byte = chunk.code[ip]; ip += 1;
                    let vec_val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("over: stack underflow".into()))?;
                    stack.push(adverb_over(&vec_val, op_byte)?);
                }
                Op::Scan => {
                    let op_byte = chunk.code[ip]; ip += 1;
                    let vec_val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("scan: stack underflow".into()))?;
                    stack.push(adverb_scan(&vec_val, op_byte)?);
                }
                Op::Each => {
                    let op_byte = chunk.code[ip]; ip += 1;
                    let vec_val = stack.pop()
                        .ok_or_else(|| TdError::Runtime("each: stack underflow".into()))?;
                    stack.push(adverb_each(&vec_val, op_byte)?);
                }

                // Stub out remaining unimplemented ops
                Op::MakeDict | Op::MakeTable
                | Op::Call
                | Op::QuerySelect | Op::QueryUpdate | Op::QueryDelete => {
                    return Err(TdError::Runtime(format!("unimplemented op: {:?}", op)));
                }
            }
        }

        Ok(stack.pop().unwrap_or(Value::nil()))
    }

    /// Convert a compile-time constant to a runtime Value.
    fn const_to_value(&self, c: &Const) -> TdResult<Value> {
        match c {
            Const::Int(v) => Ok(Value::i64(*v)),
            Const::Float(v) => Ok(Value::f64(*v)),
            Const::Bool(v) => Ok(Value::bool(*v)),
            Const::Sym(_s) => {
                // Symbols: for now, treat as a string-like value.
                // Later tasks will intern them properly.
                // For the VM, symbols are often used as names (LoadGlobal/StoreGlobal),
                // not pushed as values directly.
                Ok(Value::nil())
            }
            Const::Str(_s) => {
                // Strings: not yet backed by td_t. Return nil for now.
                Ok(Value::nil())
            }
            Const::Nil => Ok(Value::nil()),
            Const::Chunk(_) => {
                // Lambda bodies: handled in later tasks (Task 10).
                Ok(Value::nil())
            }
        }
    }

    /// Extract a string from a constant (for global variable names).
    fn const_to_string(&self, c: &Const) -> String {
        match c {
            Const::Sym(s) | Const::Str(s) => s.clone(),
            Const::Int(v) => v.to_string(),
            Const::Float(v) => v.to_string(),
            Const::Bool(v) => v.to_string(),
            Const::Nil => "nil".to_string(),
            Const::Chunk(_) => "<chunk>".to_string(),
        }
    }
}

/// Binary arithmetic on two values from the stack.
///
/// For atom-atom: computes directly in Rust.
/// For vector ops: performs element-wise computation.
fn binary_arith(stack: &mut Vec<Value>, op_name: &str) -> TdResult<()> {
    let rhs = stack.pop().ok_or(TdError::Runtime("stack underflow".into()))?;
    let lhs = stack.pop().ok_or(TdError::Runtime("stack underflow".into()))?;

    // Both atoms: direct Rust arithmetic
    if lhs.is_atom() && rhs.is_atom() {
        let result = atom_atom_op(&lhs, &rhs, op_name)?;
        stack.push(result);
        return Ok(());
    }

    // At least one vector: element-wise operation
    let result = vec_binary_op(&lhs, &rhs, op_name)?;
    stack.push(result);
    Ok(())
}

/// Atom-atom binary operation in pure Rust.
fn atom_atom_op(lhs: &Value, rhs: &Value, op_name: &str) -> TdResult<Value> {
    // Try i64-i64 first
    if let (Some(a), Some(b)) = (lhs.as_i64(), rhs.as_i64()) {
        let result = match op_name {
            "add" => Value::i64(a.wrapping_add(b)),
            "sub" => Value::i64(a.wrapping_sub(b)),
            "mul" => Value::i64(a.wrapping_mul(b)),
            "div" => {
                if b == 0 {
                    return Err(TdError::Runtime("division by zero".into()));
                }
                Value::i64(a / b)
            }
            "mod" => {
                if b == 0 {
                    return Err(TdError::Runtime("division by zero".into()));
                }
                Value::i64(a % b)
            }
            "eq" => Value::bool(a == b),
            "ne" => Value::bool(a != b),
            "lt" => Value::bool(a < b),
            "le" => Value::bool(a <= b),
            "gt" => Value::bool(a > b),
            "ge" => Value::bool(a >= b),
            _ => return Err(TdError::Runtime(format!("unknown op: {op_name}"))),
        };
        return Ok(result);
    }

    // Try f64-f64 (or mixed i64/f64 promoted to f64)
    let a = lhs.as_f64().or_else(|| lhs.as_i64().map(|v| v as f64));
    let b = rhs.as_f64().or_else(|| rhs.as_i64().map(|v| v as f64));

    if let (Some(a), Some(b)) = (a, b) {
        let result = match op_name {
            "add" => Value::f64(a + b),
            "sub" => Value::f64(a - b),
            "mul" => Value::f64(a * b),
            "div" => {
                if b == 0.0 {
                    return Err(TdError::Runtime("division by zero".into()));
                }
                Value::f64(a / b)
            }
            "mod" => {
                if b == 0.0 {
                    return Err(TdError::Runtime("division by zero".into()));
                }
                Value::f64(a % b)
            }
            "eq" => Value::bool(a == b),
            "ne" => Value::bool(a != b),
            "lt" => Value::bool(a < b),
            "le" => Value::bool(a <= b),
            "gt" => Value::bool(a > b),
            "ge" => Value::bool(a >= b),
            _ => return Err(TdError::Runtime(format!("unknown op: {op_name}"))),
        };
        return Ok(result);
    }

    // Bool-bool comparisons
    if let (Some(a), Some(b)) = (lhs.as_bool(), rhs.as_bool()) {
        let result = match op_name {
            "eq" => Value::bool(a == b),
            "ne" => Value::bool(a != b),
            _ => return Err(TdError::Runtime(format!("{op_name}: not supported for booleans"))),
        };
        return Ok(result);
    }

    Err(TdError::Runtime(format!("{op_name}: type mismatch")))
}

/// Vector binary operation: element-wise computation in Rust.
///
/// Handles Vec op Vec, Atom op Vec, Vec op Atom.
fn vec_binary_op(lhs: &Value, rhs: &Value, op_name: &str) -> TdResult<Value> {
    // Extract raw data from both operands
    let lhs_raw = lhs.as_raw().ok_or(TdError::Runtime("nil operand".into()))?;
    let rhs_raw = rhs.as_raw().ok_or(TdError::Runtime("nil operand".into()))?;

    let lhs_is_vec = lhs.is_vec();
    let rhs_is_vec = rhs.is_vec();

    // Determine lengths
    let lhs_len = if lhs_is_vec { lhs.len().unwrap_or(0) } else { 1 };
    let rhs_len = if rhs_is_vec { rhs.len().unwrap_or(0) } else { 1 };

    let result_len = if lhs_is_vec && rhs_is_vec {
        if lhs_len != rhs_len {
            return Err(TdError::Runtime(format!(
                "length mismatch: {} vs {}", lhs_len, rhs_len
            )));
        }
        lhs_len
    } else if lhs_is_vec {
        lhs_len
    } else {
        rhs_len
    };

    // Determine types
    let lhs_type = unsafe { ffi::td_type(lhs_raw as *const ffi::td_t) };
    let rhs_type = unsafe { ffi::td_type(rhs_raw as *const ffi::td_t) };

    // Determine if we're working with i64 or f64 data
    let lhs_base = lhs_type.unsigned_abs();
    let rhs_base = rhs_type.unsigned_abs();

    let is_comparison = matches!(op_name, "eq" | "ne" | "lt" | "le" | "gt" | "ge");

    // If both are i64-based, do i64 element-wise
    if lhs_base == ffi::TD_I64 as u8 && rhs_base == ffi::TD_I64 as u8 {
        let lhs_data = get_i64_data(lhs_raw, lhs_is_vec, lhs_len as usize);
        let rhs_data = get_i64_data(rhs_raw, rhs_is_vec, rhs_len as usize);

        if is_comparison {
            let mut result = Vec::with_capacity(result_len as usize);
            for i in 0..result_len as usize {
                let a = if lhs_is_vec { lhs_data[i] } else { lhs_data[0] };
                let b = if rhs_is_vec { rhs_data[i] } else { rhs_data[0] };
                let v = match op_name {
                    "eq" => a == b,
                    "ne" => a != b,
                    "lt" => a < b,
                    "le" => a <= b,
                    "gt" => a > b,
                    "ge" => a >= b,
                    _ => unreachable!(),
                };
                result.push(v);
            }
            // Build a bool vector
            unsafe {
                let vec = ffi::td_vec_new(ffi::TD_BOOL, result_len);
                let dst = ffi::td_data(vec) as *mut u8;
                for (i, &v) in result.iter().enumerate() {
                    *dst.add(i) = v as u8;
                }
                (*vec).val.len = result_len;
                Ok(Value::Td(vec))
            }
        } else {
            let mut result = Vec::with_capacity(result_len as usize);
            for i in 0..result_len as usize {
                let a = if lhs_is_vec { lhs_data[i] } else { lhs_data[0] };
                let b = if rhs_is_vec { rhs_data[i] } else { rhs_data[0] };
                let v = i64_op(a, b, op_name)?;
                result.push(v);
            }
            Ok(Value::i64_vec(&result))
        }
    } else {
        // Fall back to f64 element-wise
        let lhs_data = get_f64_data(lhs_raw, lhs_is_vec, lhs_len as usize, lhs_base);
        let rhs_data = get_f64_data(rhs_raw, rhs_is_vec, rhs_len as usize, rhs_base);

        if is_comparison {
            let mut result = Vec::with_capacity(result_len as usize);
            for i in 0..result_len as usize {
                let a = if lhs_is_vec { lhs_data[i] } else { lhs_data[0] };
                let b = if rhs_is_vec { rhs_data[i] } else { rhs_data[0] };
                let v = match op_name {
                    "eq" => a == b,
                    "ne" => a != b,
                    "lt" => a < b,
                    "le" => a <= b,
                    "gt" => a > b,
                    "ge" => a >= b,
                    _ => unreachable!(),
                };
                result.push(v);
            }
            unsafe {
                let vec = ffi::td_vec_new(ffi::TD_BOOL, result_len);
                let dst = ffi::td_data(vec) as *mut u8;
                for (i, &v) in result.iter().enumerate() {
                    *dst.add(i) = v as u8;
                }
                (*vec).val.len = result_len;
                Ok(Value::Td(vec))
            }
        } else {
            let mut result = Vec::with_capacity(result_len as usize);
            for i in 0..result_len as usize {
                let a = if lhs_is_vec { lhs_data[i] } else { lhs_data[0] };
                let b = if rhs_is_vec { rhs_data[i] } else { rhs_data[0] };
                let v = f64_op(a, b, op_name)?;
                result.push(v);
            }
            Ok(Value::f64_vec(&result))
        }
    }
}

/// Get i64 data from a raw td_t pointer (atom or vector).
fn get_i64_data(raw: *mut ffi::td_t, is_vec: bool, len: usize) -> Vec<i64> {
    unsafe {
        if is_vec {
            let data = ffi::td_data(raw) as *const i64;
            (0..len).map(|i| *data.add(i)).collect()
        } else {
            vec![(*raw).val.i64_]
        }
    }
}

/// Get f64 data from a raw td_t pointer (atom or vector), with type promotion.
fn get_f64_data(raw: *mut ffi::td_t, is_vec: bool, len: usize, base_type: u8) -> Vec<f64> {
    unsafe {
        if is_vec {
            let data_ptr = ffi::td_data(raw);
            if base_type == ffi::TD_F64 as u8 {
                let data = data_ptr as *const f64;
                (0..len).map(|i| *data.add(i)).collect()
            } else if base_type == ffi::TD_I64 as u8 {
                let data = data_ptr as *const i64;
                (0..len).map(|i| *data.add(i) as f64).collect()
            } else {
                vec![0.0; len]
            }
        } else {
            if base_type == ffi::TD_F64 as u8 {
                vec![(*raw).val.f64_]
            } else if base_type == ffi::TD_I64 as u8 {
                vec![(*raw).val.i64_ as f64]
            } else {
                vec![0.0]
            }
        }
    }
}

fn i64_op(a: i64, b: i64, op_name: &str) -> TdResult<i64> {
    match op_name {
        "add" => Ok(a.wrapping_add(b)),
        "sub" => Ok(a.wrapping_sub(b)),
        "mul" => Ok(a.wrapping_mul(b)),
        "div" => {
            if b == 0 { return Err(TdError::Runtime("division by zero".into())); }
            Ok(a / b)
        }
        "mod" => {
            if b == 0 { return Err(TdError::Runtime("division by zero".into())); }
            Ok(a % b)
        }
        _ => Err(TdError::Runtime(format!("unknown op: {op_name}"))),
    }
}

fn f64_op(a: f64, b: f64, op_name: &str) -> TdResult<f64> {
    match op_name {
        "add" => Ok(a + b),
        "sub" => Ok(a - b),
        "mul" => Ok(a * b),
        "div" => {
            if b == 0.0 { return Err(TdError::Runtime("division by zero".into())); }
            Ok(a / b)
        }
        "mod" => {
            if b == 0.0 { return Err(TdError::Runtime("division by zero".into())); }
            Ok(a % b)
        }
        _ => Err(TdError::Runtime(format!("unknown op: {op_name}"))),
    }
}

/// Unary vector operation (e.g., neg).
fn unary_vec_op(val: &Value, op_name: &str) -> TdResult<Value> {
    let raw = val.as_raw().ok_or(TdError::Runtime("nil operand".into()))?;
    let len = val.len().unwrap_or(0);
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) } as u8;

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len as usize);
        let result: Vec<i64> = match op_name {
            "neg" => data.iter().map(|&v| -v).collect(),
            _ => return Err(TdError::Runtime(format!("unknown unary op: {op_name}"))),
        };
        Ok(Value::i64_vec(&result))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len as usize, base_type);
        let result: Vec<f64> = match op_name {
            "neg" => data.iter().map(|&v| -v).collect(),
            _ => return Err(TdError::Runtime(format!("unknown unary op: {op_name}"))),
        };
        Ok(Value::f64_vec(&result))
    } else {
        Err(TdError::Runtime(format!("{op_name}: unsupported vector type")))
    }
}

/// Build a vector from the top N stack items.
///
/// If all items are i64 atoms, create an i64_vec.
/// If all items are f64 atoms (or mixed numeric), create an f64_vec.
/// Otherwise, return an error for now.
fn make_vec(stack: &mut Vec<Value>, count: usize) -> TdResult<()> {
    if count == 0 {
        stack.push(Value::nil());
        return Ok(());
    }

    if stack.len() < count {
        return Err(TdError::Runtime(format!(
            "stack underflow: need {} items, have {}", count, stack.len()
        )));
    }

    // Pop items in order (they were pushed left-to-right, so the first element
    // is deepest in the stack).
    let start = stack.len() - count;
    let items: Vec<Value> = stack.drain(start..).collect();

    // Check if all items are i64 atoms
    let all_i64 = items.iter().all(|v| v.as_i64().is_some());
    if all_i64 {
        let data: Vec<i64> = items.iter().map(|v| v.as_i64().unwrap()).collect();
        stack.push(Value::i64_vec(&data));
        return Ok(());
    }

    // Check if all items are numeric (i64 or f64)
    let all_numeric = items.iter().all(|v| v.as_i64().is_some() || v.as_f64().is_some());
    if all_numeric {
        let data: Vec<f64> = items.iter().map(|v| {
            v.as_f64().unwrap_or_else(|| v.as_i64().unwrap() as f64)
        }).collect();
        stack.push(Value::f64_vec(&data));
        return Ok(());
    }

    // Check if all items are bool atoms
    let all_bool = items.iter().all(|v| v.as_bool().is_some());
    if all_bool {
        // Build a bool vector
        let len = items.len() as i64;
        unsafe {
            let vec = ffi::td_vec_new(ffi::TD_BOOL, len);
            let dst = ffi::td_data(vec) as *mut u8;
            for (i, v) in items.iter().enumerate() {
                *dst.add(i) = v.as_bool().unwrap() as u8;
            }
            (*vec).val.len = len;
            stack.push(Value::Td(vec));
        }
        return Ok(());
    }

    Err(TdError::Runtime("MakeVec: mixed or unsupported types".into()))
}

// ======== Built-in verb implementations ========

/// `sum` verb: sum elements of a vector, return atom.
fn verb_sum(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        // sum of a single atom is itself
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("sum: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let total: i64 = data.iter().sum();
        Ok(Value::i64(total))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let total: f64 = data.iter().sum();
        Ok(Value::f64(total))
    } else {
        Err(TdError::Runtime("sum: unsupported type".into()))
    }
}

/// `avg` verb: average of vector elements, returns f64.
fn verb_avg(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        if let Some(v) = val.as_i64() {
            return Ok(Value::f64(v as f64));
        }
        if let Some(v) = val.as_f64() {
            return Ok(Value::f64(v));
        }
        return Err(TdError::Runtime("avg: unsupported atom type".into()));
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("avg: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Ok(Value::f64(f64::NAN));
    }
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    let total: f64 = if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        data.iter().map(|&v| v as f64).sum()
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        data.iter().sum()
    } else {
        return Err(TdError::Runtime("avg: unsupported type".into()));
    };
    Ok(Value::f64(total / len as f64))
}

/// `min` verb: minimum element of a vector.
fn verb_min(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("min: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Err(TdError::Runtime("min: empty vector".into()));
    }
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let m = *data.iter().min().unwrap();
        Ok(Value::i64(m))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let m = data.iter().cloned().fold(f64::INFINITY, f64::min);
        Ok(Value::f64(m))
    } else {
        Err(TdError::Runtime("min: unsupported type".into()))
    }
}

/// `max` verb: maximum element of a vector.
fn verb_max(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("max: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Err(TdError::Runtime("max: empty vector".into()));
    }
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let m = *data.iter().max().unwrap();
        Ok(Value::i64(m))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let m = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(Value::f64(m))
    } else {
        Err(TdError::Runtime("max: unsupported type".into()))
    }
}

/// `count` verb: return length as i64 atom.
fn verb_count(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(Value::i64(1));
    }
    if val.is_nil() {
        return Ok(Value::i64(0));
    }
    let len = val.len().unwrap_or(0);
    Ok(Value::i64(len))
}

/// `first` verb: return first element of vector.
fn verb_first(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("first: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Ok(Value::nil());
    }
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        Ok(Value::i64(data[0]))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        Ok(Value::f64(data[0]))
    } else {
        Err(TdError::Runtime("first: unsupported type".into()))
    }
}

/// `last` verb: return last element of vector.
fn verb_last(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("last: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Ok(Value::nil());
    }
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        Ok(Value::i64(data[len - 1]))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        Ok(Value::f64(data[len - 1]))
    } else {
        Err(TdError::Runtime("last: unsupported type".into()))
    }
}

/// `til` verb: generate vector 0..N from i64 atom.
fn verb_til(val: &Value) -> TdResult<Value> {
    let n = val.as_i64()
        .ok_or_else(|| TdError::Runtime("til: expected i64 atom".into()))?;
    if n < 0 {
        return Err(TdError::Runtime("til: negative argument".into()));
    }
    let data: Vec<i64> = (0..n).collect();
    Ok(Value::i64_vec(&data))
}

/// `where` verb: given a bool vector, return indices where true.
fn verb_where(val: &Value) -> TdResult<Value> {
    if !val.is_vec() {
        return Err(TdError::Runtime("where: expected vector".into()));
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("where: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_BOOL as u8 {
        let data = unsafe {
            let ptr = ffi::td_data(raw) as *const u8;
            std::slice::from_raw_parts(ptr, len)
        };
        let indices: Vec<i64> = data.iter().enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, _)| i as i64)
            .collect();
        Ok(Value::i64_vec(&indices))
    } else if base_type == ffi::TD_I64 as u8 {
        // Treat nonzero as true
        let data = get_i64_data(raw, true, len);
        let indices: Vec<i64> = data.iter().enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, _)| i as i64)
            .collect();
        Ok(Value::i64_vec(&indices))
    } else {
        Err(TdError::Runtime("where: unsupported vector type".into()))
    }
}

/// `asc` verb: sort vector ascending.
fn verb_asc(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("asc: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let mut data = get_i64_data(raw, true, len);
        data.sort();
        Ok(Value::i64_vec(&data))
    } else if base_type == ffi::TD_F64 as u8 {
        let mut data = get_f64_data(raw, true, len, base_type);
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Value::f64_vec(&data))
    } else {
        Err(TdError::Runtime("asc: unsupported type".into()))
    }
}

/// `desc` verb: sort vector descending.
fn verb_desc(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("desc: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let mut data = get_i64_data(raw, true, len);
        data.sort_by(|a, b| b.cmp(a));
        Ok(Value::i64_vec(&data))
    } else if base_type == ffi::TD_F64 as u8 {
        let mut data = get_f64_data(raw, true, len, base_type);
        data.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Value::f64_vec(&data))
    } else {
        Err(TdError::Runtime("desc: unsupported type".into()))
    }
}

/// `distinct` verb: unique values (preserving order of first occurrence).
fn verb_distinct(val: &Value) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("distinct: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for v in data {
            if seen.insert(v) {
                result.push(v);
            }
        }
        Ok(Value::i64_vec(&result))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        // For f64 we use bit representation for hashing
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for v in data {
            if seen.insert(v.to_bits()) {
                result.push(v);
            }
        }
        Ok(Value::f64_vec(&result))
    } else {
        Err(TdError::Runtime("distinct: unsupported type".into()))
    }
}

/// Index operation: `src[idx]`.
///
/// Supports integer atom index, integer vector index (take multiple elements),
/// and bool vector index (filter by mask).
fn verb_index(src: &Value, idx: &Value) -> TdResult<Value> {
    // Atom index: return single element
    if let Some(i) = idx.as_i64() {
        if !src.is_vec() {
            return Err(TdError::Runtime("index: source is not a vector".into()));
        }
        let raw = src.as_raw().ok_or(TdError::Runtime("index: nil source".into()))?;
        let len = src.len().unwrap_or(0);
        if i < 0 || i >= len {
            return Err(TdError::Runtime(format!("index out of range: {i}")));
        }
        let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();
        if base_type == ffi::TD_I64 as u8 {
            let data = get_i64_data(raw, true, len as usize);
            return Ok(Value::i64(data[i as usize]));
        } else if base_type == ffi::TD_F64 as u8 {
            let data = get_f64_data(raw, true, len as usize, base_type);
            return Ok(Value::f64(data[i as usize]));
        } else {
            return Err(TdError::Runtime("index: unsupported element type".into()));
        }
    }

    // Vector index: take multiple elements
    if idx.is_vec() {
        if !src.is_vec() {
            return Err(TdError::Runtime("index: source is not a vector".into()));
        }
        let src_raw = src.as_raw().ok_or(TdError::Runtime("index: nil source".into()))?;
        let src_len = src.len().unwrap_or(0) as usize;
        let src_type = unsafe { ffi::td_type(src_raw as *const ffi::td_t) }.unsigned_abs();

        let idx_raw = idx.as_raw().ok_or(TdError::Runtime("index: nil index".into()))?;
        let idx_len = idx.len().unwrap_or(0) as usize;
        let idx_type = unsafe { ffi::td_type(idx_raw as *const ffi::td_t) }.unsigned_abs();

        // If the index is a bool vector, filter by mask
        if idx_type == ffi::TD_BOOL as u8 {
            if idx_len != src_len {
                return Err(TdError::Runtime(format!(
                    "index: bool mask length mismatch: {} vs {}", idx_len, src_len
                )));
            }
            let mask = unsafe {
                let ptr = ffi::td_data(idx_raw) as *const u8;
                std::slice::from_raw_parts(ptr, idx_len)
            };
            if src_type == ffi::TD_I64 as u8 {
                let data = get_i64_data(src_raw, true, src_len);
                let result: Vec<i64> = data.iter().zip(mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(&v, _)| v)
                    .collect();
                return Ok(Value::i64_vec(&result));
            } else if src_type == ffi::TD_F64 as u8 {
                let data = get_f64_data(src_raw, true, src_len, src_type);
                let result: Vec<f64> = data.iter().zip(mask.iter())
                    .filter(|(_, &m)| m != 0)
                    .map(|(&v, _)| v)
                    .collect();
                return Ok(Value::f64_vec(&result));
            } else {
                return Err(TdError::Runtime("index: unsupported source type for bool mask".into()));
            }
        }

        // Integer vector index: gather elements
        if idx_type == ffi::TD_I64 as u8 {
            let indices = get_i64_data(idx_raw, true, idx_len);
            if src_type == ffi::TD_I64 as u8 {
                let data = get_i64_data(src_raw, true, src_len);
                let mut result = Vec::with_capacity(idx_len);
                for &i in &indices {
                    if i < 0 || (i as usize) >= src_len {
                        return Err(TdError::Runtime(format!("index out of range: {i}")));
                    }
                    result.push(data[i as usize]);
                }
                return Ok(Value::i64_vec(&result));
            } else if src_type == ffi::TD_F64 as u8 {
                let data = get_f64_data(src_raw, true, src_len, src_type);
                let mut result = Vec::with_capacity(idx_len);
                for &i in &indices {
                    if i < 0 || (i as usize) >= src_len {
                        return Err(TdError::Runtime(format!("index out of range: {i}")));
                    }
                    result.push(data[i as usize]);
                }
                return Ok(Value::f64_vec(&result));
            } else {
                return Err(TdError::Runtime("index: unsupported source type for int index".into()));
            }
        }

        return Err(TdError::Runtime("index: unsupported index type".into()));
    }

    Err(TdError::Runtime("index: unsupported index type".into()))
}

// ======== Adverb implementations ========

/// Apply a binary operation to two atom Values, returning an atom Value.
fn apply_binop(lhs: &Value, rhs: &Value, op_byte: u8) -> TdResult<Value> {
    let op = Op::from_u8(op_byte)
        .ok_or_else(|| TdError::Runtime(format!("adverb: unknown op byte: {op_byte}")))?;
    let op_name = match op {
        Op::Add => "add",
        Op::Sub => "sub",
        Op::Mul => "mul",
        Op::Div => "div",
        Op::Mod => "mod",
        Op::Eq => "eq",
        Op::Ne => "ne",
        Op::Lt => "lt",
        Op::Le => "le",
        Op::Gt => "gt",
        Op::Ge => "ge",
        Op::Min => "min",
        Op::Max => "max",
        _ => return Err(TdError::Runtime(format!("adverb: unsupported op: {:?}", op))),
    };

    // For Min/Max used as binary ops, compute the binary min/max
    if op == Op::Min {
        return binary_min_max(lhs, rhs, true);
    }
    if op == Op::Max {
        return binary_min_max(lhs, rhs, false);
    }

    atom_atom_op(lhs, rhs, op_name)
}

/// Binary min/max between two atoms.
fn binary_min_max(lhs: &Value, rhs: &Value, is_min: bool) -> TdResult<Value> {
    if let (Some(a), Some(b)) = (lhs.as_i64(), rhs.as_i64()) {
        let result = if is_min { a.min(b) } else { a.max(b) };
        return Ok(Value::i64(result));
    }
    let a = lhs.as_f64().or_else(|| lhs.as_i64().map(|v| v as f64));
    let b = rhs.as_f64().or_else(|| rhs.as_i64().map(|v| v as f64));
    if let (Some(a), Some(b)) = (a, b) {
        let result = if is_min { a.min(b) } else { a.max(b) };
        return Ok(Value::f64(result));
    }
    Err(TdError::Runtime("min/max: type mismatch".into()))
}

/// `over` (fold): applies a binary op cumulatively across vector elements.
/// `+ over 1 2 3 4` = ((1+2)+3)+4 = 10
fn adverb_over(val: &Value, op_byte: u8) -> TdResult<Value> {
    if val.is_atom() {
        return Ok(val.clone());
    }
    if !val.is_vec() {
        return Err(TdError::Runtime("over: expected vector".into()));
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("over: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Err(TdError::Runtime("over: empty vector".into()));
    }

    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let mut acc = Value::i64(data[0]);
        for &elem in &data[1..] {
            let rhs = Value::i64(elem);
            acc = apply_binop(&acc, &rhs, op_byte)?;
        }
        Ok(acc)
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let mut acc = Value::f64(data[0]);
        for &elem in &data[1..] {
            let rhs = Value::f64(elem);
            acc = apply_binop(&acc, &rhs, op_byte)?;
        }
        Ok(acc)
    } else {
        Err(TdError::Runtime("over: unsupported vector type".into()))
    }
}

/// `scan` (running fold): like over but returns all intermediate results.
/// `+ scan 1 2 3 4` = 1 3 6 10
fn adverb_scan(val: &Value, op_byte: u8) -> TdResult<Value> {
    if val.is_atom() {
        // Single atom: wrap in a 1-element vector
        if let Some(v) = val.as_i64() {
            return Ok(Value::i64_vec(&[v]));
        }
        if let Some(v) = val.as_f64() {
            return Ok(Value::f64_vec(&[v]));
        }
        return Err(TdError::Runtime("scan: unsupported atom type".into()));
    }
    if !val.is_vec() {
        return Err(TdError::Runtime("scan: expected vector".into()));
    }
    let raw = val.as_raw().ok_or(TdError::Runtime("scan: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Err(TdError::Runtime("scan: empty vector".into()));
    }

    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let mut results = Vec::with_capacity(len);
        let mut acc = Value::i64(data[0]);
        results.push(acc.as_i64().unwrap());
        for &elem in &data[1..] {
            let rhs = Value::i64(elem);
            acc = apply_binop(&acc, &rhs, op_byte)?;
            results.push(acc.as_i64().unwrap_or_else(|| {
                // If the result promoted to f64, convert back (lossy but
                // preserves type for common i64 ops)
                acc.as_f64().unwrap_or(0.0) as i64
            }));
        }
        Ok(Value::i64_vec(&results))
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let mut results = Vec::with_capacity(len);
        let mut acc = Value::f64(data[0]);
        results.push(acc.as_f64().unwrap());
        for &elem in &data[1..] {
            let rhs = Value::f64(elem);
            acc = apply_binop(&acc, &rhs, op_byte)?;
            results.push(acc.as_f64().unwrap_or(0.0));
        }
        Ok(Value::f64_vec(&results))
    } else {
        Err(TdError::Runtime("scan: unsupported vector type".into()))
    }
}

/// `each` (map): applies a unary verb to each element of a vector.
/// `count each ("ab";"cde";"f")` = 2 3 1
fn adverb_each(val: &Value, op_byte: u8) -> TdResult<Value> {
    let op = Op::from_u8(op_byte)
        .ok_or_else(|| TdError::Runtime(format!("each: unknown op byte: {op_byte}")))?;

    if val.is_atom() {
        // Apply verb to single atom
        return apply_unary_verb_to_value(val, op);
    }
    if !val.is_vec() {
        return Err(TdError::Runtime("each: expected vector".into()));
    }

    let raw = val.as_raw().ok_or(TdError::Runtime("each: nil operand".into()))?;
    let len = val.len().unwrap_or(0) as usize;
    if len == 0 {
        return Ok(Value::i64_vec(&[]));
    }

    let base_type = unsafe { ffi::td_type(raw as *const ffi::td_t) }.unsigned_abs();

    // For each element, apply the verb and collect results
    if base_type == ffi::TD_I64 as u8 {
        let data = get_i64_data(raw, true, len);
        let mut results = Vec::with_capacity(len);
        for &elem in &data {
            let elem_val = Value::i64(elem);
            let result = apply_unary_verb_to_value(&elem_val, op)?;
            results.push(result);
        }
        collect_results_to_vec(results)
    } else if base_type == ffi::TD_F64 as u8 {
        let data = get_f64_data(raw, true, len, base_type);
        let mut results = Vec::with_capacity(len);
        for &elem in &data {
            let elem_val = Value::f64(elem);
            let result = apply_unary_verb_to_value(&elem_val, op)?;
            results.push(result);
        }
        collect_results_to_vec(results)
    } else {
        Err(TdError::Runtime("each: unsupported vector type".into()))
    }
}

/// Apply a unary verb (identified by opcode) to a single Value.
fn apply_unary_verb_to_value(val: &Value, op: Op) -> TdResult<Value> {
    match op {
        Op::Count => verb_count(val),
        Op::Sum => verb_sum(val),
        Op::Avg => verb_avg(val),
        Op::Min => verb_min(val),
        Op::Max => verb_max(val),
        Op::First => verb_first(val),
        Op::Last => verb_last(val),
        Op::Asc => verb_asc(val),
        Op::Desc => verb_desc(val),
        Op::Distinct => verb_distinct(val),
        Op::Til => verb_til(val),
        Op::Enlist => {
            if val.is_atom() {
                if let Some(v) = val.as_i64() {
                    Ok(Value::i64_vec(&[v]))
                } else if let Some(v) = val.as_f64() {
                    Ok(Value::f64_vec(&[v]))
                } else {
                    Ok(val.clone())
                }
            } else {
                Ok(val.clone())
            }
        }
        Op::Neg => {
            if let Some(v) = val.as_i64() {
                Ok(Value::i64(-v))
            } else if let Some(v) = val.as_f64() {
                Ok(Value::f64(-v))
            } else {
                Err(TdError::Runtime("neg: unsupported type".into()))
            }
        }
        Op::Not => {
            if let Some(v) = val.as_bool() {
                Ok(Value::bool(!v))
            } else if let Some(v) = val.as_i64() {
                Ok(Value::bool(v == 0))
            } else {
                Err(TdError::Runtime("not: unsupported type".into()))
            }
        }
        _ => Err(TdError::Runtime(format!("each: unsupported verb op: {:?}", op))),
    }
}

/// Collect a Vec<Value> of atoms into a single vector Value.
fn collect_results_to_vec(results: Vec<Value>) -> TdResult<Value> {
    if results.is_empty() {
        return Ok(Value::i64_vec(&[]));
    }

    // Check if all results are i64
    let all_i64 = results.iter().all(|v| v.as_i64().is_some());
    if all_i64 {
        let data: Vec<i64> = results.iter().map(|v| v.as_i64().unwrap()).collect();
        return Ok(Value::i64_vec(&data));
    }

    // Check if all results are numeric
    let all_numeric = results.iter().all(|v| v.as_i64().is_some() || v.as_f64().is_some());
    if all_numeric {
        let data: Vec<f64> = results.iter().map(|v| {
            v.as_f64().unwrap_or_else(|| v.as_i64().unwrap() as f64)
        }).collect();
        return Ok(Value::f64_vec(&data));
    }

    Err(TdError::Runtime("each: mixed or unsupported result types".into()))
}
