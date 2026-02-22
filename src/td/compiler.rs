use crate::td::ast::*;
use crate::td::chunk::{Chunk, Const, Op};
use crate::td::error::{TdError, TdResult};
use crate::td::parser;

/// Parse and compile a Td source string into a bytecode `Chunk`.
pub fn compile(input: &str) -> TdResult<Chunk> {
    let expr = parser::parse(input)?;
    let mut compiler = Compiler::new();
    compiler.compile_expr(&expr)?;
    compiler.chunk.emit(Op::Return);
    Ok(compiler.chunk)
}

struct Compiler {
    chunk: Chunk,
    locals: Vec<String>,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            chunk: Chunk::new(),
            locals: Vec::new(),
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> TdResult<()> {
        match expr {
            Expr::Int(v) => {
                let idx = self.chunk.add_constant(Const::Int(*v));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Float(v) => {
                let idx = self.chunk.add_constant(Const::Float(*v));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Bool(v) => {
                let idx = self.chunk.add_constant(Const::Bool(*v));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Sym(s) => {
                let idx = self.chunk.add_constant(Const::Sym(s.clone()));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Str(s) => {
                let idx = self.chunk.add_constant(Const::Str(s.clone()));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Nil => {
                self.chunk.emit(Op::Nil);
            }
            Expr::Vector(elems) => {
                for elem in elems {
                    self.compile_expr(elem)?;
                }
                let count = elems.len().min(255) as u8;
                self.chunk.emit_u8(Op::MakeVec, count);
            }
            Expr::Dict(keys, vals) => {
                self.compile_expr(keys)?;
                self.compile_expr(vals)?;
                self.chunk.emit(Op::MakeDict);
            }
            Expr::Ident(name) => {
                if let Some(slot) = self.resolve_local(name) {
                    self.chunk.emit_u8(Op::LoadLocal, slot);
                } else {
                    let idx = self.chunk.add_constant(Const::Sym(name.clone()));
                    self.chunk.emit_u16(Op::LoadGlobal, idx);
                }
            }
            Expr::Assign(name, val) => {
                self.compile_expr(val)?;
                if let Some(slot) = self.resolve_local(name) {
                    self.chunk.emit_u8(Op::StoreLocal, slot);
                } else {
                    let idx = self.chunk.add_constant(Const::Sym(name.clone()));
                    self.chunk.emit_u16(Op::StoreGlobal, idx);
                }
            }
            Expr::BinOp(op, lhs, rhs) => {
                self.compile_expr(lhs)?;
                self.compile_expr(rhs)?;
                let opcode = Self::binop_to_op(op);
                self.chunk.emit(opcode);
            }
            Expr::UnaryOp(op, operand) => {
                self.compile_expr(operand)?;
                match op {
                    UnaryOp::Neg => self.chunk.emit(Op::Neg),
                    UnaryOp::Not => self.chunk.emit(Op::Not),
                }
            }
            Expr::Verb(verb, args) => {
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let opcode = Self::verb_to_op(verb)?;
                self.chunk.emit(opcode);
            }
            Expr::Adverb(adverb, verb_expr, arg) => {
                // Compile the argument (vector) first
                self.compile_expr(arg)?;
                // Extract the operator opcode from the verb expression
                let op_byte = self.verb_expr_to_op_byte(verb_expr)?;
                match adverb {
                    Adverb::Over => self.chunk.emit_u8(Op::Over, op_byte),
                    Adverb::Scan => self.chunk.emit_u8(Op::Scan, op_byte),
                    Adverb::Each => self.chunk.emit_u8(Op::Each, op_byte),
                }
            }
            Expr::Lambda { params, body } => {
                let mut sub = Compiler::new();
                // Register params as locals in the sub-compiler
                for p in params {
                    sub.define_local(p.clone());
                }
                sub.compile_expr(body)?;
                sub.chunk.emit(Op::Return);
                let sub_chunk = sub.chunk;
                let idx = self.chunk.add_constant(Const::Chunk(sub_chunk));
                self.chunk.emit_u16(Op::Const, idx);
            }
            Expr::Call(func, args) => {
                self.compile_expr(func)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let argc = args.len().min(255) as u8;
                self.chunk.emit_u8(Op::Call, argc);
            }
            Expr::Index(expr, indices) => {
                self.compile_expr(expr)?;
                // Compile the first index expression (common case)
                if indices.len() == 1 {
                    self.compile_expr(&indices[0])?;
                } else {
                    // Multiple indices: build a vector of them
                    for idx_expr in indices {
                        self.compile_expr(idx_expr)?;
                    }
                    let count = indices.len().min(255) as u8;
                    self.chunk.emit_u8(Op::MakeVec, count);
                }
                self.chunk.emit(Op::Index);
            }
            Expr::Block(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    self.compile_expr(e)?;
                    // Pop intermediate results; keep the last value on the stack
                    if i < exprs.len() - 1 {
                        self.chunk.emit(Op::Pop);
                    }
                }
            }
            Expr::Cond(parts) => {
                // $[cond; true_branch; false_branch]
                // parts: [condition, true_expr, false_expr]
                // or    [cond1, val1, cond2, val2, ..., default]
                self.compile_cond(parts)?;
            }
            Expr::Select { cols, by: _, from, wheres } => {
                // Compile from-table expression
                self.compile_expr(from)?;
                // Compile where clauses
                for w in wheres {
                    self.compile_expr(w)?;
                }
                // Compile column expressions
                for col in cols {
                    self.compile_expr(&col.expr)?;
                }
                self.chunk.emit(Op::QuerySelect);
            }
            Expr::Update { cols, from, wheres } => {
                self.compile_expr(from)?;
                for w in wheres {
                    self.compile_expr(w)?;
                }
                for col in cols {
                    self.compile_expr(&col.expr)?;
                }
                self.chunk.emit(Op::QueryUpdate);
            }
            Expr::Delete { from, wheres } => {
                self.compile_expr(from)?;
                for w in wheres {
                    self.compile_expr(w)?;
                }
                self.chunk.emit(Op::QueryDelete);
            }
        }
        Ok(())
    }

    fn compile_cond(&mut self, parts: &[Expr]) -> TdResult<()> {
        if parts.is_empty() {
            self.chunk.emit(Op::Nil);
            return Ok(());
        }
        if parts.len() == 1 {
            // Default value (odd-length last part)
            self.compile_expr(&parts[0])?;
            return Ok(());
        }

        // Compile condition
        self.compile_expr(&parts[0])?;
        // Emit JumpIfFalse with placeholder offset
        let false_jump = self.chunk.current_offset();
        self.chunk.emit_i16(Op::JumpIfFalse, 0);

        // Compile true branch
        self.compile_expr(&parts[1])?;
        // Emit Jump to skip false branch
        let end_jump = self.chunk.current_offset();
        self.chunk.emit_i16(Op::Jump, 0);

        // Patch the false jump target
        let false_target = self.chunk.current_offset();
        let false_offset = (false_target as isize - false_jump as isize) as i16;
        self.chunk.patch_u16(false_jump + 1, false_offset as u16);

        // Compile remaining parts (else-if chain or default)
        if parts.len() > 2 {
            self.compile_cond(&parts[2..])?;
        } else {
            self.chunk.emit(Op::Nil);
        }

        // Patch the end jump target
        let end_target = self.chunk.current_offset();
        let end_offset = (end_target as isize - end_jump as isize) as i16;
        self.chunk.patch_u16(end_jump + 1, end_offset as u16);

        Ok(())
    }

    fn resolve_local(&self, name: &str) -> Option<u8> {
        self.locals
            .iter()
            .rposition(|n| n == name)
            .map(|i| i as u8)
    }

    fn define_local(&mut self, name: String) -> u8 {
        let slot = self.locals.len() as u8;
        self.locals.push(name);
        self.chunk.num_locals = self.chunk.num_locals.max(slot + 1);
        slot
    }

    /// Extract the opcode byte for the verb/operator used by an adverb.
    ///
    /// For `+ over 1 2 3`, the verb_expr is `BinOp(Add, Nil, Nil)` -> Op::Add.
    /// For `count each (...)`, the verb_expr is `Verb(Count, [])` -> Op::Count.
    fn verb_expr_to_op_byte(&self, verb_expr: &Expr) -> TdResult<u8> {
        match verb_expr {
            Expr::BinOp(op, _, _) => Ok(Self::binop_to_op(op) as u8),
            Expr::Verb(verb, _) => Ok(Self::verb_to_op(verb)? as u8),
            _ => Err(TdError::Compile(format!(
                "unsupported verb expression in adverb: {:?}",
                verb_expr
            ))),
        }
    }

    fn binop_to_op(op: &BinOp) -> Op {
        match op {
            BinOp::Add => Op::Add,
            BinOp::Sub => Op::Sub,
            BinOp::Mul => Op::Mul,
            BinOp::Div => Op::Div,
            BinOp::Mod => Op::Mod,
            BinOp::Eq => Op::Eq,
            BinOp::Ne => Op::Ne,
            BinOp::Lt => Op::Lt,
            BinOp::Le => Op::Le,
            BinOp::Gt => Op::Gt,
            BinOp::Ge => Op::Ge,
            // And/Or/Min2/Max2 map to the closest available ops.
            // These will be expanded by the VM to call the C engine.
            BinOp::And => Op::Min,  // placeholder -- VM handles logic
            BinOp::Or => Op::Max,   // placeholder -- VM handles logic
            BinOp::Min2 => Op::Min, // binary min
            BinOp::Max2 => Op::Max, // binary max
        }
    }

    fn verb_to_op(verb: &Verb) -> TdResult<Op> {
        match verb {
            Verb::Sum => Ok(Op::Sum),
            Verb::Avg => Ok(Op::Avg),
            Verb::Min => Ok(Op::Min),
            Verb::Max => Ok(Op::Max),
            Verb::Count => Ok(Op::Count),
            Verb::First => Ok(Op::First),
            Verb::Last => Ok(Op::Last),
            Verb::Where => Ok(Op::Where),
            Verb::Til => Ok(Op::Til),
            Verb::Asc => Ok(Op::Asc),
            Verb::Desc => Ok(Op::Desc),
            Verb::Distinct => Ok(Op::Distinct),
            Verb::Enlist => Ok(Op::Enlist),
            // Verbs without dedicated opcodes use Call with the verb name
            Verb::Neg => Ok(Op::Neg),
            _ => Err(TdError::Compile(format!(
                "verb {:?} not yet supported in bytecode",
                verb
            ))),
        }
    }
}
