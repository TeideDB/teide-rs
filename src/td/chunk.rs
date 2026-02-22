/// Bytecode opcodes for the Td VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    // Stack manipulation
    Const,          // push constant from pool (u16 index follows)
    Nil,            // push nil
    Pop,            // discard top

    // Variables
    LoadLocal,      // push local (u8 index follows)
    StoreLocal,     // pop into local (u8 index follows)
    LoadGlobal,     // push global by name (u16 constant pool index)
    StoreGlobal,    // pop into global (u16 constant pool index)

    // Arithmetic (operate on td_t* via C engine)
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    Neg, Not,

    // Built-in verbs
    Sum, Avg, Min, Max, Count,
    First, Last, Where, Til,
    Asc, Desc, Distinct,
    Enlist,

    // Adverbs
    Over, Scan, Each,

    // Structure
    MakeVec,        // pop n items, build vector (u8 count follows)
    MakeDict,       // pop values-vec, keys-vec, build dict
    MakeTable,      // pop n (name,col) pairs, build table (u8 count follows)
    Index,          // x[y] -- index into vector/dict/table

    // Functions
    Call,           // call lambda/verb (u8 argc follows)
    Return,

    // Query
    QuerySelect,
    QueryUpdate,
    QueryDelete,

    // Control
    Jump,           // unconditional jump (i16 offset follows)
    JumpIfFalse,    // conditional jump (i16 offset follows)
}

impl Op {
    pub fn from_u8(v: u8) -> Option<Op> {
        if v <= Op::JumpIfFalse as u8 {
            Some(unsafe { std::mem::transmute(v) })
        } else {
            None
        }
    }
}

/// A compile-time constant (no FFI needed).
///
/// The VM converts these into runtime `Value` types when the engine is
/// available.
#[derive(Debug, Clone)]
pub enum Const {
    Int(i64),
    Float(f64),
    Bool(bool),
    Sym(String),
    Str(String),
    Nil,
    Chunk(Chunk),   // for lambda bodies
}

impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Const::Int(a), Const::Int(b)) => a == b,
            (Const::Float(a), Const::Float(b)) => a == b,
            (Const::Bool(a), Const::Bool(b)) => a == b,
            (Const::Sym(a), Const::Sym(b)) => a == b,
            (Const::Str(a), Const::Str(b)) => a == b,
            (Const::Nil, Const::Nil) => true,
            _ => false,
        }
    }
}

/// A compiled bytecode chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Bytecode instructions.
    pub code: Vec<u8>,
    /// Constant pool (compile-time constants, no FFI needed).
    pub constants: Vec<Const>,
    /// Number of local variable slots needed.
    pub num_locals: u8,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            num_locals: 0,
        }
    }

    /// Add a constant and return its index.
    pub fn add_constant(&mut self, val: Const) -> u16 {
        let idx = self.constants.len() as u16;
        self.constants.push(val);
        idx
    }

    /// Emit a single opcode byte.
    pub fn emit(&mut self, op: Op) {
        self.code.push(op as u8);
    }

    /// Emit an opcode followed by a u8 operand.
    pub fn emit_u8(&mut self, op: Op, operand: u8) {
        self.code.push(op as u8);
        self.code.push(operand);
    }

    /// Emit an opcode followed by a u16 operand (big-endian).
    pub fn emit_u16(&mut self, op: Op, operand: u16) {
        self.code.push(op as u8);
        self.code.push((operand >> 8) as u8);
        self.code.push((operand & 0xff) as u8);
    }

    /// Emit an opcode followed by an i16 operand (big-endian).
    pub fn emit_i16(&mut self, op: Op, operand: i16) {
        self.emit_u16(op, operand as u16);
    }

    /// Current offset in the code (for jump patching).
    pub fn current_offset(&self) -> usize {
        self.code.len()
    }

    /// Patch a u16 value at a given offset (for jump targets).
    pub fn patch_u16(&mut self, offset: usize, val: u16) {
        self.code[offset] = (val >> 8) as u8;
        self.code[offset + 1] = (val & 0xff) as u8;
    }
}
