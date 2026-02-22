# Td Language Design

A terse, array-oriented language over the Teide columnar engine.
K-inspired but readable: multi-character keywords replace single-glyph operators.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Name | **Td** |
| Philosophy | K-inspired, readable (multi-char names) |
| Execution model | Bytecode VM |
| Type system | Teide types + dicts in VM layer |
| Evaluation order | Right-to-left |
| Adverbs (v1) | `over`, `scan`, `each` |
| Lambda syntax | `{x+y}` with implicit x/y/z, explicit `{[a;b] ...}` |
| Table queries | Q-style templates (`select ... from ... where`) |
| Statement separator | Semicolons + newlines |

## Syntax

### Atoms & Vectors

```
42                   / i64 atom
3.14                 / f64 atom
1b                   / bool true (0b = false)
`sym                 / symbol atom
"hello"              / string
2025.01.15           / date literal
12:30:00             / time literal

1 2 3 4 5            / i64 vector (whitespace-separated)
1.0 2.5 3.7          / f64 vector
`a`b`c               / symbol vector
```

### Assignment & Expressions

```
a: 10                / assign
b: 1 2 3 4 5
c: a + b             / 11 12 13 14 15  (atom + vector = broadcast)
d: b * b             / 1 4 9 16 25     (vector * vector = element-wise)
```

### Verbs

```
sum b                / 15       — reduction
avg b                / 3.0
min b; max b         / 1; 5
count b              / 5
first b; last b      / 1; 5
where b > 3          / 3 4      — filter indices, then index
til 5                / 0 1 2 3 4
neg b                / -1 -2 -3 -4 -5
abs neg b            / 1 2 3 4 5
asc b; desc b        / sort ascending/descending
distinct b           / unique values
```

### Adverbs

```
+ over 1 2 3 4       / 10       — fold
+ scan 1 2 3 4       / 1 3 6 10 — running fold
count each ("ab";"cde";"f")  / 2 3 1  — apply to each element
```

### Lambdas

```
f: {x * x}           / implicit x
f 5                   / 25
g: {x + y}           / implicit x, y
g[3;4]                / 7
h: {[a;b;c] a+b+c}  / explicit params
```

### Tables

```
t: ([] name:`alice`bob`carol; age: 30 25 35; score: 88.5 92.0 76.3)
/ [] = empty key columns (unkeyed table)

kt: ([id: 1 2 3] name:`alice`bob`carol; score: 88.5 92.0 76.3)
/ [id] = keyed table
```

### Query Templates

```
select from t                          / all rows, all columns
select name, score from t              / project columns
select from t where age > 28           / filter
select avg score by name from t        / group-by with aggregation
select max score, count name from t    / scalar aggregation

/ multiple where clauses
select from t where age > 25, score > 80

/ computed columns
select name, pct: score % 100.0 from t

/ update (returns new table, no mutation)
update score: score * 1.1 from t where age < 30

/ delete rows
delete from t where score < 80
```

### Table Verbs

```
cols t                / `name`age`score  — column names
meta t                / type metadata table
count t               / 3 — row count
t , t                 / vertical join (append rows)
5 # t                 / take first 5 rows
-3 # t                / take last 3 rows
```

### Dictionaries

```
d: `a`b`c ! 1 2 3    / create dict (keys ! values)
d`a                   / 1  — lookup
d[`b]                 / 2  — bracket lookup
key d                 / `a`b`c
value d               / 1 2 3
```

### File I/O

```
t: read "data.csv"                     / load CSV
t: load `:db/trades                    / open splayed table (mmap)
save[`:db/trades; t]                   / persist to disk
```

### Control Flow

```
$[cond; true-expr; false-expr]         / ternary
$[c1; t1; c2; t2; else-expr]          / multi-branch cond
```

## Architecture

### Value Representation

Every value in the VM is a tagged enum. Atoms, vectors, and tables wrap
Teide's refcounted `td_t*` pointers. Dicts and lambdas are Rust-level.

```rust
enum Value {
    Atom(*mut td_t),                        // scalar (negative type tag)
    Vec(*mut td_t),                         // vector (positive type tag)
    Table(Table),                           // td_t* with type=TD_TABLE
    Dict { keys: *mut td_t, vals: *mut td_t }, // sym-vec ! value-vec
    Lambda(Rc<Chunk>, Vec<Value>),          // bytecode + captured upvalues
    Nil,                                    // null / unset
}
```

`Value::drop` calls `td_release`, `Value::clone` calls `td_retain`.
The C engine's COW and slab allocator handle memory management.

### Compiler Pipeline

```
Source text
    |
    v
  Lexer --> Token stream
    |
    v
  Parser --> AST (Expr enum)
    |
    v
  Compiler --> Chunk (bytecode + constants)
    |
    v
  VM --> execute bytecode, delegate to C engine
```

Right-to-left evaluation is handled in the parser: `sum x where x > 3`
parses as `sum(where(x, gt(x, 3)))`.

### AST

```rust
enum Expr {
    // Literals
    Int(i64),
    Float(f64),
    Bool(bool),
    Sym(String),
    Str(String),
    Date(i32),
    Time(i64),
    Nil,

    // Collections
    Vector(Vec<Expr>),
    Dict(Box<Expr>, Box<Expr>),           // keys ! values

    // Names & assignment
    Ident(String),
    Assign(String, Box<Expr>),            // name : expr

    // Operations
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
    Verb(Verb, Vec<Expr>),                // built-in verb application
    Adverb(Adverb, Box<Expr>, Box<Expr>), // verb adverb args

    // Functions
    Lambda(Vec<String>, Box<Expr>),       // params, body
    Call(Box<Expr>, Vec<Expr>),           // f[args]
    Index(Box<Expr>, Box<Expr>),          // x[y]

    // Queries
    Select { cols, by, from, where_ },
    Update { cols, from, where_ },
    Delete { from, where_ },

    // Control
    Cond(Vec<(Expr, Expr)>),              // $[c1;t1;c2;t2;...;else]
    Block(Vec<Expr>),                     // semicolon-separated exprs
}
```

### Bytecode Instructions

Stack-based. Each instruction is a `u8` opcode + inline operands.

```
// Stack manipulation
OP_CONST idx:u16        // push constant from chunk's constant pool
OP_NIL                  // push nil
OP_POP                  // discard top

// Variables
OP_LOAD_LOCAL idx:u8    // push local variable
OP_STORE_LOCAL idx:u8   // pop into local
OP_LOAD_GLOBAL idx:u16  // push global by name (constant pool index)
OP_STORE_GLOBAL idx:u16

// Arithmetic (operate on td_t* via C engine)
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD
OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE
OP_NEG, OP_NOT

// Built-in verbs
OP_SUM, OP_AVG, OP_MIN, OP_MAX, OP_COUNT
OP_FIRST, OP_LAST, OP_WHERE, OP_TIL
OP_ASC, OP_DESC, OP_DISTINCT

// Adverbs
OP_OVER, OP_SCAN, OP_EACH

// Structure
OP_MAKE_VEC n:u8        // pop n items, build vector
OP_MAKE_DICT            // pop values-vec, keys-vec, build dict
OP_MAKE_TABLE n:u8      // pop n (name,col) pairs, build table
OP_INDEX                // x[y] — index into vector/dict/table

// Functions
OP_CALL argc:u8         // call lambda/verb with argc args
OP_RETURN

// Query (specialized graph-building ops)
OP_QUERY_SELECT
OP_QUERY_UPDATE
OP_QUERY_DELETE

// Control
OP_JUMP off:i16
OP_JUMP_IF_FALSE off:i16
```

### Execution Model

The VM core is a `match` loop over opcodes:

- **Stack**: `Vec<Value>` — operand stack
- **IP**: instruction pointer into current Chunk
- **Frames**: call stack of `(Chunk, IP, stack_base)`
- **Globals**: `HashMap<String, Value>`

For arithmetic ops, the VM checks types on the stack:

- **Atom op Atom**: call C atom constructor, return atom
- **Vec op Vec** or **Atom op Vec**: build a `td_graph_t`, add ops,
  call `td_execute`, return result

The bytecode VM handles dispatch, variable binding, and control flow.
All bulk computation delegates to the C engine via the graph API.

## Project Layout

```
src/
  td/
    mod.rs          // pub mod declarations
    lexer.rs        // Token enum, Lexer struct
    token.rs        // Token type definitions
    parser.rs       // Parser -> Expr AST
    ast.rs          // Expr, BinOp, Verb, etc.
    compiler.rs     // AST -> Chunk (bytecode + constants)
    chunk.rs        // Chunk struct, opcode definitions
    vm.rs           // VM execution loop
    value.rs        // Value enum, refcount bridge
    repl.rs         // Td interactive REPL (binary entry point)
    error.rs        // TdError type
  lib.rs            // add `pub mod td;`
```

New binary in `Cargo.toml`:

```toml
[[bin]]
name = "td"
path = "src/td/repl.rs"
required-features = ["cli"]
```

## Implementation Order

| Phase | Module | Milestone |
|-------|--------|-----------|
| 1 | `value.rs` | Value enum with `td_retain`/`td_release` bridge. Values round-trip through C engine. |
| 2 | `lexer.rs`, `token.rs` | Tokenize Td source. Lex `a: 1 2 3; sum a` correctly. |
| 3 | `ast.rs`, `parser.rs` | Parse atoms, vectors, assignment, verbs, lambdas into Expr tree. |
| 4 | `chunk.rs`, `compiler.rs` | Emit bytecode from AST. Compile simple expressions. |
| 5 | `vm.rs` | Execution loop, arithmetic via C engine. `1 + 2` -> 3, `1 2 3 + 4 5 6` -> `5 7 9`. |
| 6 | verbs in `vm.rs` | `sum`, `avg`, `count`, `where`, `til`, etc. `sum 1 2 3` -> 6. |
| 7 | adverbs in `vm.rs` | `over`, `scan`, `each`. `+ over 1 2 3` -> 6. |
| 8 | lambdas in `compiler.rs` + `vm.rs` | `{x*x} 5` -> 25. Closures capture upvalues. |
| 9 | queries in `parser.rs` + `compiler.rs` | `select avg x from t where y > 5` compiles to graph ops. |
| 10 | `repl.rs` | Interactive REPL with reedline. `td` binary. |

## Not in v1

- No `eachright`, `eachleft`, `prior` adverbs
- No user-defined types
- No error trapping (`@[f;x;handler]`)
- No string interpolation
- No IPC / socket server (existing PgWire server covers that)
- No file I/O beyond `read` (CSV) and `load`/`save` (splayed)
