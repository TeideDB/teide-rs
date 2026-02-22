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
                !ptr.is_null()
                    && unsafe { ffi::td_type(*ptr as *const ffi::td_t) } == ffi::TD_TABLE
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
