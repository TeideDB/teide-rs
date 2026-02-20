//   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
//   All rights reserved.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

// SQL planner: translates sqlparser AST into Teide execution graph.

use std::collections::{HashMap, HashSet};

use sqlparser::ast::{
    BinaryOperator, ColumnDef, DataType, Distinct, Expr, FunctionArg, FunctionArgExpr, GroupByExpr,
    Ident, Insert, JoinConstraint, JoinOperator, ObjectName, ObjectType, Query, SelectItem,
    SetExpr, Statement, TableFactor, TableWithJoins, UnaryOperator, Value, Values,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use crate::{Column, Context, Graph, Table};

use super::expr::{
    agg_op_from_name, collect_aggregates, collect_window_functions, expr_default_name,
    format_agg_name, has_window_functions, is_aggregate, is_count_distinct, is_pure_aggregate,
    parse_window_frame, plan_agg_input, plan_expr, plan_having_expr, plan_post_agg_expr,
    predict_c_agg_name,
};
use super::{ExecResult, Session, SqlError, SqlResult, StoredTable};

// ---------------------------------------------------------------------------
// Session-aware entry point
// ---------------------------------------------------------------------------

/// Parse and execute a SQL statement within a session context.
/// Supports SELECT, CREATE TABLE AS SELECT, and DROP TABLE.
pub fn session_execute(session: &mut Session, sql: &str) -> Result<ExecResult, SqlError> {
    let dialect = DuckDbDialect {};
    let statements =
        Parser::parse_sql(&dialect, sql).map_err(|e| SqlError::Parse(e.to_string()))?;

    let stmt = statements
        .into_iter()
        .next()
        .ok_or_else(|| SqlError::Plan("Empty query".into()))?;

    match stmt {
        Statement::Query(q) => {
            let result = plan_query(&session.ctx, &q, Some(&session.tables))?;
            Ok(ExecResult::Query(result))
        }

        Statement::CreateTable(create) => {
            let table_name = object_name_to_string(&create.name).to_lowercase();

            if session.tables.contains_key(&table_name) && !create.or_replace {
                if create.if_not_exists {
                    return Ok(ExecResult::Ddl(format!(
                        "Table '{table_name}' already exists (skipped)"
                    )));
                }
                return Err(SqlError::Plan(format!(
                    "Table '{table_name}' already exists (use CREATE OR REPLACE TABLE)"
                )));
            }

            if let Some(query) = &create.query {
                // CREATE TABLE ... AS SELECT
                let result = plan_query(&session.ctx, query, Some(&session.tables))?;
                let nrows = result.table.nrows();
                let ncols = result.columns.len();

                let table = result.table.with_column_names(&result.columns)?;
                session.tables.insert(
                    table_name.clone(),
                    StoredTable {
                        table,
                        columns: result.columns,
                    },
                );

                Ok(ExecResult::Ddl(format!(
                    "Created table '{table_name}' ({nrows} rows, {ncols} cols)"
                )))
            } else if !create.columns.is_empty() {
                // CREATE TABLE t (col1 TYPE, col2 TYPE, ...)
                let (table, columns) = create_empty_table(&create.columns)?;
                let ncols = columns.len();
                session
                    .tables
                    .insert(table_name.clone(), StoredTable { table, columns });

                Ok(ExecResult::Ddl(format!(
                    "Created table '{table_name}' (0 rows, {ncols} cols)"
                )))
            } else {
                Err(SqlError::Plan(
                    "CREATE TABLE requires column definitions or AS SELECT".into(),
                ))
            }
        }

        Statement::Drop {
            object_type: ObjectType::Table,
            names,
            if_exists,
            ..
        } => {
            let mut msgs = Vec::new();
            for name in &names {
                let table_name = object_name_to_string(name).to_lowercase();
                if session.tables.remove(&table_name).is_some() {
                    msgs.push(format!("Dropped table '{table_name}'"));
                } else if if_exists {
                    msgs.push(format!("Table '{table_name}' not found (skipped)"));
                } else {
                    return Err(SqlError::Plan(format!("Table '{table_name}' not found")));
                }
            }
            Ok(ExecResult::Ddl(msgs.join("\n")))
        }

        Statement::Insert(insert) => plan_insert(session, &insert),

        _ => Err(SqlError::Plan(
            "Only SELECT, CREATE TABLE AS, DROP TABLE, and INSERT INTO are supported".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// CREATE TABLE (col TYPE, ...) — bare table creation with schema
// ---------------------------------------------------------------------------

/// Map a SQL DataType to a Teide type tag.
fn sql_type_to_td(dt: &DataType) -> Result<i8, SqlError> {
    use crate::ffi;
    match dt {
        DataType::Int(_)
        | DataType::Integer(_)
        | DataType::BigInt(_)
        | DataType::SmallInt(_)
        | DataType::TinyInt(_) => Ok(ffi::TD_I64),
        DataType::Real
        | DataType::Float(_)
        | DataType::Double
        | DataType::DoublePrecision
        | DataType::Numeric(_)
        | DataType::Decimal(_)
        | DataType::Dec(_) => Ok(ffi::TD_F64),
        DataType::Boolean => Ok(ffi::TD_BOOL),
        DataType::Varchar(_)
        | DataType::Text
        | DataType::Char(_)
        | DataType::CharVarying(_)
        | DataType::String(_) => Ok(ffi::TD_SYM),
        _ => Err(SqlError::Plan(format!(
            "CREATE TABLE: unsupported column type {dt}"
        ))),
    }
}

/// Create an empty table from column definitions.
fn create_empty_table(columns: &[ColumnDef]) -> Result<(Table, Vec<String>), SqlError> {
    let ncols = columns.len();
    if ncols == 0 {
        return Err(SqlError::Plan("CREATE TABLE: no columns defined".into()));
    }

    let mut col_names = Vec::with_capacity(ncols);
    let mut builder = RawTableBuilder::new(ncols as i64)?;

    for col_def in columns {
        let name = col_def.name.value.to_lowercase();
        let typ = sql_type_to_td(&col_def.data_type)?;

        // Create an empty vector with capacity 0
        let vec = unsafe { crate::raw::td_vec_new(typ, 0) };
        if vec.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(vec) {
            return Err(engine_err_from_raw(vec));
        }

        let name_id = crate::sym_intern(&name)?;
        let res = builder.add_col(name_id, vec);
        unsafe { crate::ffi_release(vec) };
        res?;

        col_names.push(name);
    }

    let table = builder.finish()?;
    Ok((table, col_names))
}

// ---------------------------------------------------------------------------
// INSERT INTO
// ---------------------------------------------------------------------------

fn plan_insert(session: &mut Session, insert: &Insert) -> Result<ExecResult, SqlError> {
    let table_name = object_name_to_string(&insert.table_name).to_lowercase();

    let stored = session
        .tables
        .get(&table_name)
        .ok_or_else(|| SqlError::Plan(format!("Table '{table_name}' not found")))?;

    let target_types: Vec<i8> = (0..stored.table.ncols())
        .map(|c| stored.table.col_type(c as usize))
        .collect();
    let target_cols = stored.columns.clone();

    let source_query = insert
        .source
        .as_ref()
        .ok_or_else(|| SqlError::Plan("INSERT INTO requires VALUES or SELECT".into()))?;

    // Build source table from VALUES or SELECT
    let (source_table, source_cols) = match source_query.body.as_ref() {
        SetExpr::Values(values) => {
            let tbl = build_table_from_values(values, &target_types, &target_cols)?;
            let cols = target_cols.clone();
            (tbl, cols)
        }
        _ => {
            // Treat as a subquery (SELECT ...)
            let result = plan_query(&session.ctx, source_query, Some(&session.tables))?;
            (result.table, result.columns)
        }
    };

    // Handle optional column list reordering
    let source_table = if !insert.columns.is_empty() {
        reorder_insert_columns(
            &insert.columns,
            &target_cols,
            &target_types,
            &source_table,
            &source_cols,
        )?
    } else {
        if source_table.ncols() != stored.table.ncols() {
            return Err(SqlError::Plan(format!(
                "INSERT INTO: source has {} columns but target '{}' has {}",
                source_table.ncols(),
                table_name,
                stored.table.ncols()
            )));
        }
        source_table
    };

    let nrows = source_table.nrows();

    // Concatenate with existing table
    let existing = &stored.table;
    let merged = concat_tables(&session.ctx, existing, &source_table)?;

    // Rename columns to match target schema
    let merged = merged.with_column_names(&target_cols)?;

    session.tables.insert(
        table_name.clone(),
        StoredTable {
            table: merged,
            columns: target_cols,
        },
    );

    Ok(ExecResult::Ddl(format!(
        "Inserted {nrows} rows into '{table_name}'"
    )))
}

/// Build a Table from VALUES (...), (...) literal rows.
fn build_table_from_values(
    values: &Values,
    target_types: &[i8],
    target_cols: &[String],
) -> Result<Table, SqlError> {
    let nrows = values.rows.len();
    let ncols = target_types.len();

    if nrows == 0 {
        return Err(SqlError::Plan("INSERT INTO: empty VALUES".into()));
    }

    // Validate row widths
    for (i, row) in values.rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(SqlError::Plan(format!(
                "INSERT INTO: row {} has {} values but expected {}",
                i,
                row.len(),
                ncols
            )));
        }
    }

    // Create column vectors
    let mut col_vecs: Vec<*mut crate::td_t> = Vec::with_capacity(ncols);
    for &typ in target_types {
        let vec = unsafe { crate::raw::td_vec_new(typ, nrows as i64) };
        if vec.is_null() {
            // Release already-allocated vectors
            for v in &col_vecs {
                unsafe { crate::ffi_release(*v) };
            }
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(vec) {
            for v in &col_vecs {
                unsafe { crate::ffi_release(*v) };
            }
            return Err(engine_err_from_raw(vec));
        }
        col_vecs.push(vec);
    }

    // Fill column data
    for row in &values.rows {
        for (c, expr) in row.iter().enumerate() {
            let typ = target_types[c];
            let vec = col_vecs[c];
            match append_value_to_vec(vec, typ, expr, c, target_cols) {
                Ok(next) => col_vecs[c] = next,
                Err(e) => {
                    for v in &col_vecs {
                        unsafe { crate::ffi_release(*v) };
                    }
                    return Err(e);
                }
            }
        }
    }

    // Build table
    let mut builder = RawTableBuilder::new(ncols as i64)?;
    for (c, vec) in col_vecs.iter().enumerate() {
        let name_id = crate::sym_intern(&target_cols[c])?;
        let res = builder.add_col(name_id, *vec);
        unsafe { crate::ffi_release(*vec) };
        res?;
    }
    builder.finish()
}

/// Append a single literal value to a vector, returning the (possibly reallocated) vector.
fn append_value_to_vec(
    vec: *mut crate::td_t,
    typ: i8,
    expr: &Expr,
    col_idx: usize,
    col_names: &[String],
) -> Result<*mut crate::td_t, SqlError> {
    use crate::ffi;
    use std::ffi::c_void;

    match typ {
        // Integer types (I32, I64, SYM-as-integer, BOOL)
        ffi::TD_I32 | ffi::TD_I64 | ffi::TD_BOOL => {
            let val = eval_i64_literal(expr)
                .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?;
            match typ {
                ffi::TD_I64 => {
                    let next =
                        unsafe { ffi::td_vec_append(vec, &val as *const i64 as *const c_void) };
                    check_vec_append(next)
                }
                ffi::TD_I32 => {
                    let v32 = val as i32;
                    let next =
                        unsafe { ffi::td_vec_append(vec, &v32 as *const i32 as *const c_void) };
                    check_vec_append(next)
                }
                ffi::TD_BOOL => {
                    let b = if val != 0 { 1u8 } else { 0u8 };
                    let next = unsafe { ffi::td_vec_append(vec, &b as *const u8 as *const c_void) };
                    check_vec_append(next)
                }
                _ => unreachable!(),
            }
        }

        ffi::TD_F64 => {
            let val = eval_f64_literal(expr)
                .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?;
            let next = unsafe { ffi::td_vec_append(vec, &val as *const f64 as *const c_void) };
            check_vec_append(next)
        }

        ffi::TD_SYM => {
            let s = eval_str_literal(expr)
                .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?;
            let sym_id = crate::sym_intern(&s)?;
            let next = unsafe { ffi::td_vec_append(vec, &sym_id as *const i64 as *const c_void) };
            check_vec_append(next)
        }

        _ => Err(SqlError::Plan(format!(
            "INSERT INTO: unsupported column type {} for '{}'",
            typ, col_names[col_idx]
        ))),
    }
}

fn check_vec_append(next: *mut crate::td_t) -> Result<*mut crate::td_t, SqlError> {
    if next.is_null() {
        return Err(SqlError::Engine(crate::Error::Oom));
    }
    if crate::ffi_is_err(next) {
        return Err(engine_err_from_raw(next));
    }
    Ok(next)
}

/// Evaluate a literal expression to an i64 value.
fn eval_i64_literal(expr: &Expr) -> Result<i64, String> {
    match expr {
        Expr::Value(Value::Number(s, _)) => s
            .parse::<i64>()
            .map_err(|e| format!("invalid integer '{s}': {e}")),
        Expr::Value(Value::Boolean(b)) => Ok(if *b { 1 } else { 0 }),
        Expr::Value(Value::Null) => Ok(0), // null sentinel for integer
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let val = eval_i64_literal(expr)?;
            Ok(-val)
        }
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_i64_literal(expr),
        _ => Err(format!("expected integer literal, got {expr}")),
    }
}

/// Evaluate a literal expression to an f64 value.
fn eval_f64_literal(expr: &Expr) -> Result<f64, String> {
    match expr {
        Expr::Value(Value::Number(s, _)) => s
            .parse::<f64>()
            .map_err(|e| format!("invalid float '{s}': {e}")),
        Expr::Value(Value::Null) => Ok(f64::NAN), // null sentinel for float
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let val = eval_f64_literal(expr)?;
            Ok(-val)
        }
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_f64_literal(expr),
        _ => Err(format!("expected numeric literal, got {expr}")),
    }
}

/// Evaluate a literal expression to a string value.
fn eval_str_literal(expr: &Expr) -> Result<String, String> {
    match expr {
        Expr::Value(Value::SingleQuotedString(s)) => Ok(s.clone()),
        Expr::Value(Value::DoubleQuotedString(s)) => Ok(s.clone()),
        Expr::Value(Value::Null) => Ok(String::new()), // null sentinel for string
        _ => Err(format!("expected string literal, got {expr}")),
    }
}

/// Reorder source columns to match target schema when INSERT specifies an explicit column list.
fn reorder_insert_columns(
    insert_cols: &[Ident],
    target_cols: &[String],
    target_types: &[i8],
    source: &Table,
    source_cols: &[String],
) -> Result<Table, SqlError> {
    let _ = source_cols;
    let ncols = target_cols.len();
    let nrows = source.nrows() as usize;

    if insert_cols.len() != source.ncols() as usize {
        return Err(SqlError::Plan(format!(
            "INSERT INTO: column list has {} entries but source has {} columns",
            insert_cols.len(),
            source.ncols()
        )));
    }

    // Map insert column names to target column indices
    let mut col_map: Vec<Option<usize>> = vec![None; ncols]; // target_idx -> source_idx
    for (src_idx, ident) in insert_cols.iter().enumerate() {
        let name = ident.value.to_lowercase();
        let tgt_idx = target_cols
            .iter()
            .position(|c| c.to_lowercase() == name)
            .ok_or_else(|| {
                SqlError::Plan(format!(
                    "INSERT INTO: column '{}' not found in target table",
                    name
                ))
            })?;
        if col_map[tgt_idx].is_some() {
            return Err(SqlError::Plan(format!(
                "INSERT INTO: duplicate column '{name}' in column list"
            )));
        }
        col_map[tgt_idx] = Some(src_idx);
    }

    // Build new table: for each target column, either copy from source or fill with defaults
    let mut builder = RawTableBuilder::new(ncols as i64)?;
    for tgt_idx in 0..ncols {
        let name_id = crate::sym_intern(&target_cols[tgt_idx])?;
        let typ = target_types[tgt_idx];

        let col = if let Some(src_idx) = col_map[tgt_idx] {
            // Copy column from source
            source
                .get_col_idx(src_idx as i64)
                .ok_or_else(|| SqlError::Plan("INSERT INTO: source column missing".into()))?
        } else {
            // Create a default-filled column (zeros/empty)
            let new_col = unsafe { crate::raw::td_vec_new(typ, nrows as i64) };
            if new_col.is_null() {
                return Err(SqlError::Engine(crate::Error::Oom));
            }
            unsafe { crate::raw::td_set_len(new_col, nrows as i64) };
            // Zero-initialized by td_vec_new — acceptable default
            let res = builder.add_col(name_id, new_col);
            unsafe { crate::ffi_release(new_col) };
            res?;
            continue;
        };

        unsafe { crate::ffi_retain(col) };
        let res = builder.add_col(name_id, col);
        unsafe { crate::ffi_release(col) };
        res?;
    }
    builder.finish()
}

// ---------------------------------------------------------------------------
// Stateless entry point
// ---------------------------------------------------------------------------

/// Parse, plan, and execute a SQL query.
pub(crate) fn plan_and_execute(
    ctx: &Context,
    sql: &str,
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<SqlResult, SqlError> {
    let dialect = DuckDbDialect {};
    let statements =
        Parser::parse_sql(&dialect, sql).map_err(|e| SqlError::Parse(e.to_string()))?;

    let stmt = statements
        .into_iter()
        .next()
        .ok_or_else(|| SqlError::Plan("Empty query".into()))?;

    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Err(SqlError::Plan("Only SELECT queries are supported".into())),
    };

    plan_query(ctx, &query, tables)
}

// ---------------------------------------------------------------------------
// Query planning
// ---------------------------------------------------------------------------

fn plan_query(
    ctx: &Context,
    query: &Query,
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<SqlResult, SqlError> {
    // Handle CTEs (WITH clause)
    let cte_tables: HashMap<String, StoredTable>;
    let effective_tables: Option<&HashMap<String, StoredTable>>;

    if let Some(with) = &query.with {
        let mut cte_map: HashMap<String, StoredTable> = match tables {
            Some(t) => t.clone(),
            None => HashMap::new(),
        };
        for cte in &with.cte_tables {
            let cte_name = cte.alias.name.value.to_lowercase();
            let result = plan_query(ctx, &cte.query, Some(&cte_map))?;
            // Rename table columns to match SQL aliases so downstream scans work
            let table = result.table.with_column_names(&result.columns)?;
            cte_map.insert(
                cte_name,
                StoredTable {
                    table,
                    columns: result.columns,
                },
            );
        }
        cte_tables = cte_map;
        effective_tables = Some(&cte_tables);
    } else {
        effective_tables = tables;
    }

    // Handle UNION ALL / set operations
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s,
        SetExpr::SetOperation {
            op: sqlparser::ast::SetOperator::Union,
            set_quantifier,
            left,
            right,
        } => {
            let is_all = matches!(set_quantifier, sqlparser::ast::SetQuantifier::All);
            // Execute both sides
            let left_query = Query {
                with: None,
                body: left.clone(),
                order_by: None,
                limit: None,
                offset: None,
                fetch: None,
                locks: vec![],
                limit_by: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
            };
            let right_query = Query {
                with: None,
                body: right.clone(),
                order_by: None,
                limit: None,
                offset: None,
                fetch: None,
                locks: vec![],
                limit_by: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
            };
            let left_result = plan_query(ctx, &left_query, effective_tables)?;
            let right_result = plan_query(ctx, &right_query, effective_tables)?;

            if left_result.columns.len() != right_result.columns.len() {
                return Err(SqlError::Plan(format!(
                    "UNION: column count mismatch ({} vs {})",
                    left_result.columns.len(),
                    right_result.columns.len()
                )));
            }

            // Concatenate tables column by column
            let result = concat_tables(ctx, &left_result.table, &right_result.table)?;

            // Without ALL: apply DISTINCT
            let result = if !is_all {
                let aliases: Vec<String> = (0..result.ncols() as usize)
                    .map(|i| result.col_name_str(i).to_string())
                    .collect();
                let schema = build_schema(&result);
                let (distinct_result, _) = plan_distinct(ctx, &result, &aliases, &schema)?;
                distinct_result
            } else {
                result
            };

            // Apply ORDER BY and LIMIT from the outer query
            return apply_post_processing(
                ctx,
                query,
                result,
                left_result.columns,
                effective_tables,
            );
        }
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            let is_all = matches!(set_quantifier, sqlparser::ast::SetQuantifier::All);

            let left_query = Query {
                with: None,
                body: left.clone(),
                order_by: None,
                limit: None,
                offset: None,
                fetch: None,
                locks: vec![],
                limit_by: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
            };
            let right_query = Query {
                with: None,
                body: right.clone(),
                order_by: None,
                limit: None,
                offset: None,
                fetch: None,
                locks: vec![],
                limit_by: vec![],
                for_clause: None,
                settings: None,
                format_clause: None,
            };
            let left_result = plan_query(ctx, &left_query, effective_tables)?;
            let right_result = plan_query(ctx, &right_query, effective_tables)?;

            if left_result.columns.len() != right_result.columns.len() {
                return Err(SqlError::Plan(format!(
                    "{:?}: column count mismatch ({} vs {})",
                    op,
                    left_result.columns.len(),
                    right_result.columns.len()
                )));
            }

            let keep_matches = matches!(op, sqlparser::ast::SetOperator::Intersect);

            let result =
                exec_set_operation(ctx, &left_result.table, &right_result.table, keep_matches)?;

            // Without ALL: apply DISTINCT
            let result = if !is_all {
                let aliases: Vec<String> = (0..result.ncols() as usize)
                    .map(|i| result.col_name_str(i).to_string())
                    .collect();
                let schema = build_schema(&result);
                let (distinct_result, _) = plan_distinct(ctx, &result, &aliases, &schema)?;
                distinct_result
            } else {
                result
            };

            return apply_post_processing(
                ctx,
                query,
                result,
                left_result.columns,
                effective_tables,
            );
        }
        _ => {
            return Err(SqlError::Plan(
                "Only simple SELECT queries are supported".into(),
            ))
        }
    };

    // DISTINCT flag
    let is_distinct = matches!(&select.distinct, Some(Distinct::Distinct));

    // Resolve FROM clause, with predicate pushdown for subqueries.
    // When FROM is a single subquery with window functions or GROUP BY,
    // equality predicates on PARTITION BY / GROUP BY keys are injected into the
    // subquery's WHERE before materialization — avoids processing all rows.
    let (table, schema, effective_where): (Table, HashMap<String, usize>, Option<Expr>) = if select
        .from
        .len()
        == 1
        && select.from[0].joins.is_empty()
        && select.selection.is_some()
    {
        if let (TableFactor::Derived { subquery, .. }, Some(where_expr)) =
            (&select.from[0].relation, select.selection.as_ref())
        {
            let pushable_cols = get_pushable_columns_from_query(subquery);
            if !pushable_cols.is_empty() {
                let terms = split_conjunction(where_expr);
                let mut push = Vec::new();
                let mut keep = Vec::new();
                for term in &terms {
                    if extract_equality_column(term)
                        .map(|c| pushable_cols.contains(&c))
                        .unwrap_or(false)
                    {
                        push.push((*term).clone());
                    } else {
                        keep.push((*term).clone());
                    }
                }
                if !push.is_empty() {
                    let modified = inject_predicates_into_query(subquery, &push);
                    let result = plan_query(ctx, &modified, effective_tables)?;
                    let tbl = result.table.with_column_names(&result.columns)?;
                    let sch = build_result_schema(&tbl, &result.columns);
                    (tbl, sch, join_conjunction(keep))
                } else {
                    let (tbl, sch) = resolve_from(ctx, &select.from, effective_tables)?;
                    (tbl, sch, select.selection.clone())
                }
            } else {
                let (tbl, sch) = resolve_from(ctx, &select.from, effective_tables)?;
                (tbl, sch, select.selection.clone())
            }
        } else {
            let (tbl, sch) = resolve_from(ctx, &select.from, effective_tables)?;
            (tbl, sch, select.selection.clone())
        }
    } else {
        let (tbl, sch) = resolve_from(ctx, &select.from, effective_tables)?;
        (tbl, sch, select.selection.clone())
    };

    // Build SELECT alias → expression map (for GROUP BY on aliases)
    let select_items = &select.projection;
    let mut alias_exprs: HashMap<String, Expr> = HashMap::new();
    for item in select_items {
        if let SelectItem::ExprWithAlias { expr, alias } = item {
            alias_exprs.insert(alias.value.to_lowercase(), expr.clone());
        }
    }

    // Collect SELECT aliases for positional GROUP BY (GROUP BY 1, 2)
    let select_aliases_for_gb: Vec<String> = select_items
        .iter()
        .map(|item| match item {
            SelectItem::ExprWithAlias { alias, .. } => alias.value.to_lowercase(),
            SelectItem::UnnamedExpr(e) => super::expr::expr_default_name(e),
            _ => String::new(),
        })
        .collect();

    // GROUP BY column names (accepts table columns, SELECT aliases, expressions, and positions)
    let group_by_cols = extract_group_by_columns(
        &select.group_by,
        &schema,
        &mut alias_exprs,
        &select_aliases_for_gb,
    )?;
    let has_group_by = !group_by_cols.is_empty();

    // Detect aggregates in SELECT
    let has_aggregates = select_items.iter().any(|item| match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => is_aggregate(e),
        _ => false,
    });

    // ORDER/LIMIT/OFFSET metadata (extracted early for WHERE+LIMIT fusion).
    let order_by_exprs = extract_order_by(query)?;
    let offset_val = extract_offset(query)?;
    let limit_val = extract_limit(query)?;
    let has_windows = has_window_functions(select_items);

    // Stage 1: WHERE filter (resolve subqueries first)
    // Uses effective_where which may have had predicates removed by pushdown above.
    //
    // WHERE + LIMIT fusion: when the query has no GROUP BY, ORDER BY, HAVING,
    // or window functions, we can fuse HEAD(FILTER) in a single graph.
    // The C executor detects HEAD(FILTER) and gathers only the first N
    // matching rows, avoiding full-table materialization.
    let can_fuse_where_limit = effective_where.is_some()
        && limit_val.is_some()
        && !has_group_by
        && !has_aggregates
        && !has_windows
        && order_by_exprs.is_empty()
        && select.having.is_none();
    let mut where_limit_fused = false;

    let (working_table, selection): (Table, Option<*mut crate::td_t>) =
        if let Some(ref where_expr) = effective_where {
            let resolved = if has_subqueries(where_expr) {
                resolve_subqueries(ctx, where_expr, effective_tables)?
            } else {
                where_expr.clone()
            };
            {
                let mut g = ctx.graph(&table)?;
                let table_node = g.const_table(&table)?;
                let pred = plan_expr(&mut g, &resolved, &schema)?;
                let filtered = g.filter(table_node, pred)?;
                if can_fuse_where_limit {
                    // Fuse LIMIT into WHERE: HEAD(FILTER(table, pred), n)
                    let total = match (offset_val, limit_val) {
                        (Some(off), Some(lim)) => off.saturating_add(lim),
                        (_, Some(lim)) => lim,
                        _ => unreachable!(),
                    };
                    let head_node = g.head(filtered, total)?;
                    where_limit_fused = true;
                    (g.execute(head_node)?, None)
                } else {
                    (g.execute(filtered)?, None)
                }
            }
        } else {
            (table, None)
        };

    // Stage 1.5: Window functions (before GROUP BY)
    let (working_table, schema, select_items) = if has_windows {
        let (wt, ws, wi) = plan_window_stage(ctx, &working_table, select_items, &schema)?;
        (wt, ws, std::borrow::Cow::Owned(wi))
    } else {
        (
            working_table,
            schema,
            std::borrow::Cow::Borrowed(select_items),
        )
    };
    let select_items: &[SelectItem] = &select_items;

    // Stage 2: GROUP BY / aggregation / DISTINCT
    // Fuse LIMIT into GROUP BY graph when safe (no ORDER BY, no HAVING).
    // The C engine uses HEAD(GROUP) to short-circuit per-partition loops.
    let group_limit =
        if (has_group_by || has_aggregates) && order_by_exprs.is_empty() && select.having.is_none()
        {
            match (offset_val, limit_val) {
                (Some(off), Some(lim)) => Some(off.saturating_add(lim)),
                (None, Some(lim)) => Some(lim),
                _ => None,
            }
        } else {
            None
        };
    let (result_table, result_aliases) = if has_group_by || has_aggregates {
        plan_group_select(
            ctx,
            &working_table,
            select_items,
            &group_by_cols,
            &schema,
            &alias_exprs,
            selection,
            group_limit,
            select.having.as_ref(),
        )?
    } else if is_distinct {
        // DISTINCT without GROUP BY: use GROUP BY on all selected columns
        let aliases = extract_projection_aliases(select_items, &schema)?;
        plan_distinct(ctx, &working_table, &aliases, &schema)?
    } else {
        let aliases = extract_projection_aliases(select_items, &schema)?;
        // SQL allows ORDER BY columns not present in SELECT output.
        // Keep those as hidden columns during sorting, then trim before returning.
        let hidden_order_cols = collect_hidden_order_columns(&order_by_exprs, &aliases, &schema);
        // Skip projection only for true identity projections (`SELECT *` or
        // selecting all base columns in table order). This prevents silently
        // returning wrong columns for reordered/missing identifiers.
        let can_passthrough = !has_windows && is_identity_projection(select_items, &schema);
        if can_passthrough {
            (working_table, aliases)
        } else {
            plan_expr_select(
                ctx,
                &working_table,
                select_items,
                &schema,
                &hidden_order_cols,
            )?
        }
    };

    let (result_table, limit_fused) = if !order_by_exprs.is_empty() {
        let table_col_names: Vec<String> = (0..result_table.ncols() as usize)
            .map(|i| result_table.col_name_str(i).to_string())
            .collect();
        let mut g = ctx.graph(&result_table)?;
        let table_node = g.const_table(&result_table)?;
        let sort_node = plan_order_by(
            &mut g,
            table_node,
            &order_by_exprs,
            &result_aliases,
            &table_col_names,
        )?;

        // Fuse LIMIT into HEAD(SORT) so the engine only gathers N rows
        let total_limit = match (offset_val, limit_val) {
            (Some(off), Some(lim)) => Some(
                off.checked_add(lim)
                    .ok_or_else(|| SqlError::Plan("OFFSET + LIMIT overflow".into()))?,
            ),
            (None, Some(lim)) => Some(lim),
            _ => None,
        };
        let root = match total_limit {
            Some(n) => g.head(sort_node, n)?,
            None => sort_node,
        };
        (g.execute(root)?, total_limit.is_some())
    } else {
        (result_table, group_limit.is_some() || where_limit_fused)
    };

    // Stage 4: OFFSET + LIMIT (only parts not already fused)
    let result_table = if limit_fused {
        match offset_val {
            Some(off) => skip_rows(ctx, &result_table, off)?,
            None => result_table,
        }
    } else {
        match (offset_val, limit_val) {
            (Some(off), Some(lim)) => {
                let total = off.saturating_add(lim);
                let g = ctx.graph(&result_table)?;
                let table_node = g.const_table(&result_table)?;
                let head_node = g.head(table_node, total)?;
                let trimmed = g.execute(head_node)?;
                skip_rows(ctx, &trimmed, off)?
            }
            (Some(off), None) => skip_rows(ctx, &result_table, off)?,
            (None, Some(lim)) => {
                let g = ctx.graph(&result_table)?;
                let table_node = g.const_table(&result_table)?;
                let root = g.head(table_node, lim)?;
                g.execute(root)?
            }
            (None, None) => result_table,
        }
    };

    // Drop hidden ORDER BY helper columns (if any) before exposing SQL result.
    let result_table = trim_to_visible_columns(ctx, result_table, &result_aliases)?;

    validate_result_table(&result_table)?;
    Ok(SqlResult {
        table: result_table,
        columns: result_aliases,
    })
}

// ---------------------------------------------------------------------------
// Table resolution
// ---------------------------------------------------------------------------

/// Resolve a table name: check session registry first, then CSV file.
fn resolve_table(
    name: &str,
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<Table, SqlError> {
    // Only match session-registered tables (case-insensitive).
    // File-based loading uses explicit table functions:
    //   read_csv('/path'), read_splayed('/path'), read_parted('/db', 'table')
    if let Some(registry) = tables {
        let lower = name.to_lowercase();
        if let Some(stored) = registry.get(&lower) {
            return Ok(stored.table.clone_ref());
        }
    }
    Err(SqlError::Plan(format!(
        "Table '{}' not found. Use read_csv(), read_splayed(), or read_parted() for file-based tables",
        name
    )))
}

// ---------------------------------------------------------------------------
// Table functions: read_csv, read_splayed, read_parted
// ---------------------------------------------------------------------------

/// Extract a string literal from a FunctionArg.
fn extract_string_arg(arg: &FunctionArg) -> Result<String, SqlError> {
    match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(Value::SingleQuotedString(s)))) => {
            Ok(s.clone())
        }
        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(Value::DoubleQuotedString(s)))) => {
            Ok(s.clone())
        }
        _ => Err(SqlError::Plan(format!(
            "Expected a string literal argument, got: {arg}"
        ))),
    }
}

/// Resolve a table function call (read_csv, read_splayed, read_parted).
fn resolve_table_function(
    ctx: &Context,
    name: &str,
    args: &[FunctionArg],
) -> Result<Table, SqlError> {
    match name {
        "read_csv" => {
            if args.is_empty() || args.len() > 3 {
                return Err(SqlError::Plan(
                    "read_csv() requires 1-3 arguments: read_csv('/path/to/file.csv' [, delimiter, header])".into(),
                ));
            }
            let path = extract_string_arg(&args[0])?;
            if args.len() == 1 {
                ctx.read_csv(&path)
                    .map_err(|e| SqlError::Plan(format!("read_csv('{path}'): {e}")))
            } else {
                let delim_str = extract_string_arg(&args[1])?;
                let delimiter = delim_str.chars().next().unwrap_or(',');
                let header = if args.len() == 3 {
                    let h = extract_string_arg(&args[2])?;
                    !matches!(h.to_lowercase().as_str(), "false" | "0" | "no")
                } else {
                    true
                };
                ctx.read_csv_opts(&path, delimiter, header, None)
                    .map_err(|e| SqlError::Plan(format!("read_csv('{path}'): {e}")))
            }
        }
        "read_splayed" => {
            if args.is_empty() || args.len() > 2 {
                return Err(SqlError::Plan(
                    "read_splayed() requires 1-2 arguments: read_splayed('/path/to/dir' [, '/path/to/sym'])".into(),
                ));
            }
            let dir = extract_string_arg(&args[0])?;
            let sym_path = if args.len() == 2 {
                Some(extract_string_arg(&args[1])?)
            } else {
                None
            };
            ctx.read_splayed(&dir, sym_path.as_deref())
                .map_err(|e| SqlError::Plan(format!("read_splayed('{dir}'): {e}")))
        }
        "read_parted" => {
            if args.len() != 2 {
                return Err(SqlError::Plan(
                    "read_parted() requires exactly 2 arguments: read_parted('/db_root', 'table_name')".into(),
                ));
            }
            let db_root = extract_string_arg(&args[0])?;
            let table_name = extract_string_arg(&args[1])?;
            ctx.read_parted(&db_root, &table_name).map_err(|e| {
                SqlError::Plan(format!("read_parted('{db_root}', '{table_name}'): {e}"))
            })
        }
        _ => Err(SqlError::Plan(format!(
            "Unknown table function '{name}'. Supported: read_csv(), read_splayed(), read_parted()"
        ))),
    }
}

/// Extract equi-join keys from an ON condition.
/// Returns (left_col_name, right_col_name) pairs.
fn extract_join_keys(
    expr: &Expr,
    left_schema: &HashMap<String, usize>,
    right_schema: &HashMap<String, usize>,
) -> Result<Vec<(String, String)>, SqlError> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            let l_name = extract_col_name(left)?;
            let r_name = extract_col_name(right)?;

            // Determine which side belongs to which table
            if left_schema.contains_key(&l_name) && right_schema.contains_key(&r_name) {
                Ok(vec![(l_name, r_name)])
            } else if left_schema.contains_key(&r_name) && right_schema.contains_key(&l_name) {
                Ok(vec![(r_name, l_name)])
            } else {
                Err(SqlError::Plan(format!(
                    "JOIN ON columns '{l_name}' and '{r_name}' not found in respective tables"
                )))
            }
        }
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut keys = extract_join_keys(left, left_schema, right_schema)?;
            keys.extend(extract_join_keys(right, left_schema, right_schema)?);
            Ok(keys)
        }
        _ => Err(SqlError::Plan(
            "Only equi-join conditions (col1 = col2 [AND ...]) are supported".into(),
        )),
    }
}

/// Extract a column name from an expression (handles Identifier and CompoundIdentifier).
fn extract_col_name(expr: &Expr) -> Result<String, SqlError> {
    match expr {
        Expr::Identifier(ident) => Ok(ident.value.to_lowercase()),
        Expr::CompoundIdentifier(parts) => {
            if parts.len() == 2 {
                Ok(parts[1].value.to_lowercase())
            } else {
                Err(SqlError::Plan(format!(
                    "Unsupported compound identifier in JOIN: {expr}"
                )))
            }
        }
        _ => Err(SqlError::Plan(format!(
            "Unsupported expression in JOIN ON: {expr}"
        ))),
    }
}

/// Resolve the FROM clause into a working table + schema.
/// Handles simple tables, table aliases, FROM subqueries, and JOINs.
fn resolve_from(
    ctx: &Context,
    from: &[TableWithJoins],
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<(Table, HashMap<String, usize>), SqlError> {
    if from.is_empty() {
        return Err(SqlError::Plan("Missing FROM clause".into()));
    }

    // Multiple FROM tables = implicit CROSS JOIN: SELECT * FROM t1, t2
    if from.len() > 1 {
        let (mut result_table, mut result_schema) =
            resolve_table_factor(ctx, &from[0].relation, tables)?;
        // Process joins on first table
        for join in &from[0].joins {
            let (right_table, right_schema) = resolve_table_factor(ctx, &join.relation, tables)?;
            result_table = exec_cross_join(ctx, &result_table, &right_table)?;
            result_schema = build_schema(&result_table);
            let _ = right_schema;
        }
        // Cross join with subsequent FROM tables
        for twj in &from[1..] {
            let (right_table, _) = resolve_table_factor(ctx, &twj.relation, tables)?;
            result_table = exec_cross_join(ctx, &result_table, &right_table)?;
            result_schema = build_schema(&result_table);
            for join in &twj.joins {
                let (right_table2, _) = resolve_table_factor(ctx, &join.relation, tables)?;
                result_table = exec_cross_join(ctx, &result_table, &right_table2)?;
                result_schema = build_schema(&result_table);
            }
        }
        return Ok((result_table, result_schema));
    }

    let twj = &from[0];

    // Resolve the base (left) table
    let (mut left_table, mut left_schema) = resolve_table_factor(ctx, &twj.relation, tables)?;

    // Process JOINs
    for join in &twj.joins {
        let (right_table, right_schema) = resolve_table_factor(ctx, &join.relation, tables)?;

        // Determine join type
        let join_type: u8 = match &join.join_operator {
            JoinOperator::Inner(_) => 0,
            JoinOperator::LeftOuter(_) => 1,
            JoinOperator::RightOuter(_) => {
                // RIGHT JOIN = swap left/right then LEFT JOIN
                // We'll handle this by swapping and post-reordering
                1
            }
            JoinOperator::FullOuter(..) => 2,
            JoinOperator::CrossJoin => {
                let result = exec_cross_join(ctx, &left_table, &right_table)?;
                let merged_schema = build_schema(&result);
                left_table = result;
                left_schema = merged_schema;
                continue;
            }
            _ => {
                return Err(SqlError::Plan(format!(
                    "Unsupported join type: {:?}",
                    join.join_operator
                )));
            }
        };

        let is_right_join = matches!(&join.join_operator, JoinOperator::RightOuter(_));

        // Extract ON condition
        let on_expr = match &join.join_operator {
            JoinOperator::Inner(c)
            | JoinOperator::LeftOuter(c)
            | JoinOperator::RightOuter(c)
            | JoinOperator::FullOuter(c, ..) => match c {
                JoinConstraint::On(expr) => expr.clone(),
                _ => {
                    return Err(SqlError::Plan(
                        "Only ON conditions are supported for JOINs".into(),
                    ))
                }
            },
            _ => {
                return Err(SqlError::Plan("JOIN requires ON condition".into()));
            }
        };

        // For RIGHT JOIN, determine which is actual_left/right
        // We clone_ref to avoid borrow conflicts with the graph
        let (al_table, al_schema, ar_table, ar_schema) = if is_right_join {
            (
                right_table.clone_ref(),
                right_schema.clone(),
                left_table.clone_ref(),
                left_schema.clone(),
            )
        } else {
            (
                left_table.clone_ref(),
                left_schema.clone(),
                right_table.clone_ref(),
                right_schema.clone(),
            )
        };

        // Extract equi-join keys
        let join_keys = extract_join_keys(&on_expr, &al_schema, &ar_schema)?;
        if join_keys.is_empty() {
            return Err(SqlError::Plan(
                "JOIN ON must have at least one equi-join key".into(),
            ));
        }

        // Build join graph (scoped to avoid borrow conflict)
        let result = {
            let mut g = ctx.graph(&al_table)?;
            let left_table_node = g.const_table(&al_table)?;
            let right_table_node = g.const_table(&ar_table)?;

            let left_key_nodes: Vec<crate::Column> = join_keys
                .iter()
                .map(|(lk, _)| g.scan(lk))
                .collect::<crate::Result<Vec<_>>>()?;

            // Right keys: use const_vec to avoid cross-graph references
            let mut right_key_nodes: Vec<crate::Column> = Vec::new();
            for (_, rk) in &join_keys {
                let right_sym = crate::sym_intern(rk)?;
                let right_col_ptr =
                    unsafe { crate::ffi_table_get_col(ar_table.as_raw(), right_sym) };
                if right_col_ptr.is_null() || crate::ffi_is_err(right_col_ptr) {
                    return Err(SqlError::Plan(format!(
                        "Right key column '{}' not found",
                        rk
                    )));
                }
                // SAFETY: right_col_ptr is a valid column vector obtained from
                // ffi_table_get_col and checked for null/error above.
                right_key_nodes.push(unsafe { g.const_vec(right_col_ptr)? });
            }

            let joined = g.join(
                left_table_node,
                &left_key_nodes,
                right_table_node,
                &right_key_nodes,
                join_type,
            )?;

            g.execute(joined)?
        };

        // Build merged schema
        let merged_schema = build_schema(&result);

        left_table = result;
        left_schema = merged_schema;
    }

    Ok((left_table, left_schema))
}

/// Resolve a single TableFactor (table name or FROM subquery) into a table + schema.
fn resolve_table_factor(
    ctx: &Context,
    factor: &TableFactor,
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<(Table, HashMap<String, usize>), SqlError> {
    match factor {
        TableFactor::Table { name, args, .. } => {
            let table_name = object_name_to_string(name);
            // Table functions: read_csv(...), read_splayed(...), read_parted(...)
            if let Some(func_args) = args {
                let func_name = table_name.to_lowercase();
                let table = resolve_table_function(ctx, &func_name, &func_args.args)?;
                let schema = build_schema(&table);
                return Ok((table, schema));
            }
            let table = resolve_table(&table_name, tables)?;
            let schema = build_schema(&table);
            Ok((table, schema))
        }
        TableFactor::Derived { subquery, .. } => {
            let result = plan_query(ctx, subquery, tables)?;
            // Rename columns to match SQL aliases so outer scans work
            let table = result.table.with_column_names(&result.columns)?;
            let schema = build_result_schema(&table, &result.columns);
            Ok((table, schema))
        }
        _ => Err(SqlError::Plan(
            "Only table references, table functions, and subqueries are supported in FROM".into(),
        )),
    }
}

/// Build a schema from a result table using provided column aliases.
fn build_result_schema(table: &Table, aliases: &[String]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, alias) in aliases.iter().enumerate() {
        map.insert(alias.clone(), i);
    }
    // Also add native column names
    let ncols = table.ncols() as usize;
    for i in 0..ncols {
        let name = table.col_name_str(i);
        if !name.is_empty() {
            map.entry(name.to_lowercase()).or_insert(i);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// GROUP BY with post-aggregation expressions
// ---------------------------------------------------------------------------

/// Plan a GROUP BY query with support for:
/// - Expressions as aggregate inputs: SUM(v1 + v2)
/// - Post-aggregation arithmetic: SUM(v1) * 2, SUM(v1) / COUNT(v1)
/// - Mixed expressions in SELECT
#[allow(clippy::too_many_arguments)]
fn plan_group_select(
    ctx: &Context,
    working_table: &Table,
    select_items: &[SelectItem],
    group_by_cols: &[String],
    schema: &HashMap<String, usize>,
    alias_exprs: &HashMap<String, Expr>,
    selection: Option<*mut crate::td_t>,
    group_limit: Option<i64>,
    having: Option<&Expr>,
) -> Result<(Table, Vec<String>), SqlError> {
    // RAII guard: ensures the selection is released on all exit paths
    // (including early returns). set_selection does its own retain, so the
    // graph keeps the mask alive independently.
    struct MaskGuard(*mut crate::td_t);
    impl Drop for MaskGuard {
        fn drop(&mut self) {
            unsafe {
                crate::ffi_release(self.0);
            }
        }
    }
    let _mask_guard = selection.map(MaskGuard);

    let has_group_by = !group_by_cols.is_empty();

    // Phase 1: Analyze SELECT items, collect all unique aggregates
    let key_names: Vec<String> = group_by_cols.to_vec();
    let mut all_aggs: Vec<AggInfo> = Vec::new(); // (op, func_ref, alias)
    let mut select_plan: Vec<SelectPlan> = Vec::new();
    let mut final_aliases: Vec<String> = Vec::new();

    for item in select_items {
        let (expr, explicit_alias) = match item {
            SelectItem::UnnamedExpr(e) => (e, None),
            SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias.value.to_lowercase())),
            SelectItem::Wildcard(_) => {
                return Err(SqlError::Plan(
                    "SELECT * not supported with GROUP BY".into(),
                ))
            }
            _ => return Err(SqlError::Plan("Unsupported SELECT item".into())),
        };

        // Check if this SELECT item is an expression whose alias is a GROUP BY key
        if let Some(ref alias) = explicit_alias {
            if group_by_cols.contains(alias) && alias_exprs.contains_key(alias) {
                select_plan.push(SelectPlan::KeyRef(alias.clone()));
                final_aliases.push(alias.clone());
                continue;
            }
        }

        if let Expr::Identifier(ident) = expr {
            let name = ident.value.to_lowercase();
            if has_group_by && !group_by_cols.contains(&name) {
                return Err(SqlError::Plan(format!(
                    "Column '{}' must appear in GROUP BY or be in an aggregate function",
                    name
                )));
            }
            // Key reference — already included via GROUP BY keys
            let alias = explicit_alias.unwrap_or(name);
            select_plan.push(SelectPlan::KeyRef(alias.clone()));
            final_aliases.push(alias);
        } else if is_pure_aggregate(expr) {
            // Pure aggregate: SUM(v1), COUNT(*)
            let func = match expr {
                Expr::Function(f) => f,
                _ => {
                    return Err(SqlError::Plan(format!(
                        "Expected aggregate function, got expression '{expr}'"
                    )))
                }
            };
            let agg_alias = format_agg_name(func);
            let agg_idx = register_agg(&mut all_aggs, func, &agg_alias);
            let display = explicit_alias.unwrap_or(agg_alias);
            select_plan.push(SelectPlan::PureAgg(agg_idx, display.clone()));
            final_aliases.push(display);
        } else if is_aggregate(expr) {
            // Mixed expression containing aggregates: SUM(v1) * 2
            let agg_refs = collect_aggregates(expr);
            for (_agg_expr, agg_alias) in &agg_refs {
                if let Expr::Function(f) = _agg_expr {
                    register_agg(&mut all_aggs, f, agg_alias);
                }
            }
            let display = explicit_alias.unwrap_or_else(|| expr_default_name(expr));
            select_plan.push(SelectPlan::PostAggExpr(
                Box::new(expr.clone()),
                display.clone(),
            ));
            final_aliases.push(display);
        } else {
            // Check if this expression matches a GROUP BY expression key
            let expr_str = format!("{expr}").to_lowercase();
            let mut matched_key = None;
            for (alias, gb_expr) in alias_exprs.iter() {
                if group_by_cols.contains(alias) {
                    let gb_str = format!("{gb_expr}").to_lowercase();
                    if gb_str == expr_str {
                        matched_key = Some(alias.clone());
                        break;
                    }
                }
            }
            if let Some(key) = matched_key {
                let display = explicit_alias.unwrap_or_else(|| expr_default_name(expr));
                select_plan.push(SelectPlan::KeyRef(key));
                final_aliases.push(display);
            } else if !has_group_by {
                // No GROUP BY — this shouldn't happen (would have been caught earlier)
                return Err(SqlError::Plan(format!(
                    "Expression '{}' must be in GROUP BY or contain an aggregate",
                    expr
                )));
            } else {
                return Err(SqlError::Plan(format!(
                    "Expression '{}' must be in GROUP BY or contain an aggregate",
                    expr
                )));
            }
        }
    }

    // Check for COUNT(DISTINCT col) — handle via two-phase aggregation
    let has_count_distinct = all_aggs.iter().any(|a| is_count_distinct(&a.func));
    if has_count_distinct {
        return plan_count_distinct_group(
            ctx,
            working_table,
            &key_names,
            &all_aggs,
            &select_plan,
            &final_aliases,
            schema,
            alias_exprs,
        );
    }

    // Phase 2: Execute GROUP BY with keys + all unique aggregates
    let mut g = ctx.graph(working_table)?;

    let mut key_nodes: Vec<Column> = Vec::new();
    for k in &key_names {
        if let Some(expr) = alias_exprs.get(k) {
            // Expression-based key (e.g., CASE WHEN ... AS bucket, GROUP BY bucket)
            key_nodes.push(plan_expr(&mut g, expr, schema)?);
        } else {
            key_nodes.push(g.scan(k)?);
        }
    }

    let mut agg_ops = Vec::new();
    let mut agg_inputs = Vec::new();
    for agg in &all_aggs {
        let base_op = agg_op_from_name(&agg.func_name)?;
        let (op, input) = plan_agg_input(&mut g, &agg.func, base_op, schema)?;
        agg_ops.push(op);
        agg_inputs.push(input);
    }

    let group_node = g.group_by(&key_nodes, &agg_ops, &agg_inputs)?;

    // Push filter mask into the graph so exec_group skips filtered rows.
    // Ownership: set_selection retains (rc=2). MaskGuard (created at the
    // top of this function) releases our reference on any exit path (rc=1).
    // Graph::drop releases the graph's reference (rc=0).
    if let Some(mask) = selection {
        unsafe {
            g.set_selection(mask);
        }
    }
    // Fuse LIMIT into GROUP BY graph so the C engine can optimize
    // (e.g. short-circuit per-partition loop for MAPCOMMON-only keys).
    let mut exec_root = match group_limit {
        Some(n) => g.head(group_node, n)?,
        None => group_node,
    };

    // HAVING fusion: build FILTER(GROUP, having_pred) in the same graph.
    // The C executor detects FILTER(GROUP) and temporarily swaps g->table
    // to the GROUP result so SCAN nodes resolve against output columns.
    if let Some(having_expr) = having {
        // Predict GROUP output schema without executing.
        // Layout: [key_0, ..., key_n, agg_0, ..., agg_m]
        let mut predicted_schema: HashMap<String, usize> = HashMap::new();
        let mut predicted_names: Vec<String> = Vec::new();
        for (i, k) in key_names.iter().enumerate() {
            predicted_schema.insert(k.clone(), i);
            predicted_names.push(k.clone());
        }
        for (i, agg) in all_aggs.iter().enumerate() {
            let idx = key_names.len() + i;
            // Predict C engine native name (e.g. "v1_sum")
            let native = predict_c_agg_name(&agg.func, schema).unwrap_or_else(|| agg.alias.clone());
            predicted_schema.insert(native.clone(), idx);
            // Also register the format_agg_name alias (e.g. "sum(v1)")
            predicted_schema.entry(agg.alias.clone()).or_insert(idx);
            predicted_names.push(native);
        }
        let having_pred = plan_having_expr(
            &mut g,
            having_expr,
            &predicted_schema,
            schema,
            &predicted_names,
        )?;
        exec_root = g.filter(exec_root, having_pred)?;
    }

    let group_result = g.execute(exec_root)?;

    // Build result schema from NATIVE column names + our format_agg_name aliases.
    // The C engine names agg columns as "{col}_{suffix}" (e.g., "v1_sum").
    // We also add our aliases so plan_post_agg_expr can resolve either style.
    let mut group_schema = build_schema(&group_result);
    for (i, agg) in all_aggs.iter().enumerate() {
        group_schema
            .entry(agg.alias.clone())
            .or_insert(key_names.len() + i);
    }

    // Phase 3: Check if post-processing or projection is needed
    let needs_post_processing = select_plan
        .iter()
        .any(|p| matches!(p, SelectPlan::PostAggExpr(..)));

    // Check if selected columns match the group result layout exactly.
    // Group result is: [key_0, ..., key_n, agg_0, ..., agg_m].
    // If the user didn't select all keys or selected in a different order,
    // we need to project even without post-agg expressions.
    let group_ncols = key_names.len() + all_aggs.len();
    let needs_projection = final_aliases.len() != group_ncols;

    if !needs_post_processing && !needs_projection {
        // Simple case: result columns match GROUP BY output directly
        return Ok((group_result, final_aliases));
    }

    // Build mapping: display alias → native column name in the group result.
    let mut alias_to_native: HashMap<String, String> = HashMap::new();
    for (i, agg) in all_aggs.iter().enumerate() {
        let col_idx = key_names.len() + i;
        let native = group_result.col_name_str(col_idx);
        alias_to_native.insert(agg.alias.clone(), native.to_string());
    }

    // Simple projection (no post-agg expressions): pick columns directly
    // from the group result without creating a second graph.
    if !needs_post_processing {
        let mut pick_names: Vec<String> = Vec::new();
        for plan in &select_plan {
            match plan {
                SelectPlan::KeyRef(alias) => pick_names.push(alias.clone()),
                SelectPlan::PureAgg(idx, _) => {
                    let col_idx = key_names.len() + *idx;
                    pick_names.push(group_result.col_name_str(col_idx));
                }
                _ => unreachable!(),
            }
        }
        let pick_refs: Vec<&str> = pick_names.iter().map(|s| s.as_str()).collect();
        let result = group_result
            .pick_columns(&pick_refs)
            .map_err(|e| SqlError::Plan(format!("column projection failed: {e}")))?;
        return Ok((result, final_aliases));
    }

    // Phase 4: Post-aggregation expressions — requires a second graph
    let mut pg = ctx.graph(&group_result)?;
    let table_node = pg.const_table(&group_result)?;

    let mut proj_cols = Vec::new();
    let mut proj_aliases = Vec::new();

    for plan in &select_plan {
        match plan {
            SelectPlan::KeyRef(alias) => {
                proj_cols.push(pg.scan(alias)?);
                proj_aliases.push(alias.clone());
            }
            SelectPlan::PureAgg(idx, alias) => {
                let col_idx = key_names.len() + *idx;
                let native = group_result.col_name_str(col_idx);
                proj_cols.push(pg.scan(&native)?);
                proj_aliases.push(alias.clone());
            }
            SelectPlan::PostAggExpr(expr, alias) => {
                let col = plan_post_agg_expr(&mut pg, expr.as_ref(), &alias_to_native)?;
                proj_cols.push(col);
                proj_aliases.push(alias.clone());
            }
        }
    }

    let proj = pg.select(table_node, &proj_cols)?;
    let result = pg.execute(proj)?;

    Ok((result, final_aliases))
}

struct AggInfo {
    func_name: String,
    func: sqlparser::ast::Function,
    alias: String,
}

fn register_agg(
    all_aggs: &mut Vec<AggInfo>,
    func: &sqlparser::ast::Function,
    alias: &str,
) -> usize {
    // Check for existing aggregate with same alias
    if let Some(idx) = all_aggs.iter().position(|a| a.alias == alias) {
        return idx;
    }
    let idx = all_aggs.len();
    all_aggs.push(AggInfo {
        func_name: func.name.to_string().to_lowercase(),
        func: func.clone(),
        alias: alias.to_string(),
    });
    idx
}

enum SelectPlan {
    KeyRef(String),
    PureAgg(usize, String), // (agg index, display alias)
    PostAggExpr(Box<Expr>, String),
}

// ---------------------------------------------------------------------------
// COUNT(DISTINCT) via two-phase GROUP BY
// ---------------------------------------------------------------------------

/// Handle GROUP BY queries containing COUNT(DISTINCT col).
/// Phase 1: GROUP BY [original_keys + distinct_col] to get unique combos
/// Phase 2: GROUP BY [original_keys] with COUNT(*) to count unique values
/// Non-DISTINCT aggregates are computed in phase 1 and use FIRST in phase 2.
#[allow(clippy::too_many_arguments)]
fn plan_count_distinct_group(
    ctx: &Context,
    working_table: &Table,
    key_names: &[String],
    all_aggs: &[AggInfo],
    _select_plan: &[SelectPlan],
    final_aliases: &[String],
    schema: &HashMap<String, usize>,
    alias_exprs: &HashMap<String, Expr>,
) -> Result<(Table, Vec<String>), SqlError> {
    // Collect the DISTINCT column names and regular aggs
    let mut distinct_cols: Vec<String> = Vec::new();
    let mut regular_aggs: Vec<&AggInfo> = Vec::new();

    for agg in all_aggs {
        if is_count_distinct(&agg.func) {
            // Extract the column name from the aggregate argument
            if let sqlparser::ast::FunctionArguments::List(args) = &agg.func.args {
                if let Some(sqlparser::ast::FunctionArg::Unnamed(
                    sqlparser::ast::FunctionArgExpr::Expr(Expr::Identifier(ident)),
                )) = args.args.first()
                {
                    let col = ident.value.to_lowercase();
                    if !distinct_cols.contains(&col) {
                        distinct_cols.push(col);
                    }
                } else {
                    return Err(SqlError::Plan(
                        "COUNT(DISTINCT) requires a simple column reference".into(),
                    ));
                }
            }
        } else {
            regular_aggs.push(agg);
        }
    }

    // Phase 1: GROUP BY [original_keys + distinct_cols] with regular aggs
    let mut phase1_keys: Vec<String> = key_names.to_vec();
    for dc in &distinct_cols {
        if !phase1_keys.contains(dc) {
            phase1_keys.push(dc.clone());
        }
    }

    let mut g = ctx.graph(working_table)?;
    let mut key_nodes: Vec<Column> = Vec::new();
    for k in &phase1_keys {
        if let Some(expr) = alias_exprs.get(k) {
            key_nodes.push(plan_expr(&mut g, expr, schema)?);
        } else {
            key_nodes.push(g.scan(k)?);
        }
    }

    // Regular aggregates computed in phase 1
    let mut phase1_agg_ops = Vec::new();
    let mut phase1_agg_inputs = Vec::new();
    for agg in &regular_aggs {
        let base_op = agg_op_from_name(&agg.func_name)?;
        let (op, input) = plan_agg_input(&mut g, &agg.func, base_op, schema)?;
        phase1_agg_ops.push(op);
        phase1_agg_inputs.push(input);
    }

    // Need at least one aggregate; if only COUNT(DISTINCT), use dummy COUNT(*)
    if phase1_agg_ops.is_empty() {
        let first_col = schema
            .iter()
            .min_by_key(|(_, v)| **v)
            .map(|(k, _)| k.clone())
            .ok_or_else(|| SqlError::Plan("Empty schema".into()))?;
        phase1_agg_ops.push(crate::AggOp::Count);
        phase1_agg_inputs.push(g.scan(&first_col)?);
    }

    let group_node = g.group_by(&key_nodes, &phase1_agg_ops, &phase1_agg_inputs)?;
    let phase1_result = g.execute(group_node)?;

    // Phase 2: GROUP BY [original_keys] with COUNT(*) for each distinct col
    // and FIRST for each regular aggregate
    let mut g2 = ctx.graph(&phase1_result)?;
    let phase2_keys: Vec<Column> = key_names
        .iter()
        .map(|k| g2.scan(k))
        .collect::<crate::Result<Vec<_>>>()?;

    // For no-GROUP-BY case (e.g., SELECT COUNT(DISTINCT id1) FROM t),
    // we need a scalar reduction. Use the distinct col as key in phase 1,
    // then count rows.
    if key_names.is_empty() {
        // Phase 1 grouped by distinct_cols → nrows = unique count.
        // Use a scalar GROUP BY (no keys) with COUNT(*) to produce a 1-row table.
        let first_col_name = phase1_result.col_name_str(0).to_string();
        let mut g2 = ctx.graph(&phase1_result)?;
        let count_input = g2.scan(&first_col_name)?;
        let group_node = g2.group_by(&[], &[crate::AggOp::Count], &[count_input])?;
        let result = g2.execute(group_node)?;
        return Ok((result, final_aliases.to_vec()));
    }

    let mut phase2_agg_ops = Vec::new();
    let mut phase2_agg_inputs = Vec::new();

    // COUNT(DISTINCT col) → COUNT(*) on the distinct col (counts unique groups)
    for dc in &distinct_cols {
        phase2_agg_ops.push(crate::AggOp::Count);
        phase2_agg_inputs.push(g2.scan(dc)?);
    }

    // Regular aggs → re-aggregate in phase 2 with compatible ops:
    // SUM→SUM, MIN→MIN, MAX→MAX, COUNT→SUM (sum of partial counts), AVG→not directly supported
    let phase1_schema = build_schema(&phase1_result);
    for agg in &regular_aggs {
        let native =
            predict_phase1_col(&phase1_result, &agg.alias, phase1_keys.len(), all_aggs, agg);
        if phase1_schema.contains_key(&native) {
            let phase2_op = match agg.func_name.as_str() {
                "sum" => crate::AggOp::Sum,
                "min" => crate::AggOp::Min,
                "max" => crate::AggOp::Max,
                "count" => crate::AggOp::Sum, // sum of partial counts
                "avg" => {
                    return Err(SqlError::Plan(
                        "AVG cannot be mixed with COUNT(DISTINCT) yet".into(),
                    ));
                }
                _ => crate::AggOp::First,
            };
            phase2_agg_ops.push(phase2_op);
            phase2_agg_inputs.push(g2.scan(&native)?);
        } else {
            return Err(SqlError::Plan(format!(
                "Aggregate '{}' not found in phase 1 result (looked for '{}')",
                agg.alias, native
            )));
        }
    }

    let group_node2 = g2.group_by(&phase2_keys, &phase2_agg_ops, &phase2_agg_inputs)?;
    let phase2_result = g2.execute(group_node2)?;

    Ok((phase2_result, final_aliases.to_vec()))
}

/// Predict the native column name for an aggregate in the phase 1 result.
fn predict_phase1_col(
    result: &Table,
    _alias: &str,
    n_keys: usize,
    all_aggs: &[AggInfo],
    target: &AggInfo,
) -> String {
    // Find the index of this agg among non-count-distinct aggs
    let mut reg_idx = 0;
    for agg in all_aggs {
        if std::ptr::eq(agg, target) {
            break;
        }
        if !is_count_distinct(&agg.func) {
            reg_idx += 1;
        }
    }
    // The phase1 result has keys first, then regular agg columns
    let col_idx = n_keys + reg_idx;
    result.col_name_str(col_idx).to_string()
}

// ---------------------------------------------------------------------------
// DISTINCT via GROUP BY
// ---------------------------------------------------------------------------

fn plan_distinct(
    ctx: &Context,
    working_table: &Table,
    col_names: &[String],
    schema: &HashMap<String, usize>,
) -> Result<(Table, Vec<String>), SqlError> {
    if col_names.is_empty() {
        return Err(SqlError::Plan("DISTINCT on empty projection".into()));
    }
    for name in col_names {
        if !schema.contains_key(name) {
            return Err(SqlError::Plan(format!(
                "DISTINCT column '{}' not found",
                name
            )));
        }
    }

    // Native DISTINCT: GROUP BY with 0 aggregates — single graph, no dummy COUNT
    let mut g = ctx.graph(working_table)?;
    let key_nodes: Vec<Column> = col_names
        .iter()
        .map(|k| g.scan(k))
        .collect::<crate::Result<Vec<_>>>()?;

    let distinct_node = g.distinct(&key_nodes)?;
    let result = g.execute(distinct_node)?;

    Ok((result, col_names.to_vec()))
}

// ---------------------------------------------------------------------------
// Expression SELECT (non-GROUP-BY with computed columns)
// ---------------------------------------------------------------------------

fn plan_expr_select(
    ctx: &Context,
    working_table: &Table,
    select_items: &[SelectItem],
    schema: &HashMap<String, usize>,
    hidden_order_cols: &[String],
) -> Result<(Table, Vec<String>), SqlError> {
    let mut g = ctx.graph(working_table)?;
    let table_node = g.const_table(working_table)?;

    let mut proj_cols = Vec::new();
    let mut aliases = Vec::new();

    for item in select_items {
        match item {
            SelectItem::Wildcard(_) => {
                let mut cols: Vec<_> = schema.iter().collect();
                cols.sort_by_key(|(_name, idx)| **idx);
                for (name, _) in cols {
                    proj_cols.push(g.scan(name)?);
                    aliases.push(name.clone());
                }
            }
            SelectItem::UnnamedExpr(expr) => {
                let col = plan_expr(&mut g, expr, schema)?;
                proj_cols.push(col);
                aliases.push(expr_default_name(expr));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let col = plan_expr(&mut g, expr, schema)?;
                proj_cols.push(col);
                aliases.push(alias.value.to_lowercase());
            }
            _ => return Err(SqlError::Plan("Unsupported SELECT item".into())),
        }
    }

    // Keep non-projected ORDER BY source columns as hidden fields for sorting.
    for name in hidden_order_cols {
        if !schema.contains_key(name) {
            return Err(SqlError::Plan(format!(
                "ORDER BY column '{}' not found",
                name
            )));
        }
        proj_cols.push(g.scan(name)?);
    }

    let proj = g.select(table_node, &proj_cols)?;
    let result = g.execute(proj)?;
    Ok((result, aliases))
}

// ---------------------------------------------------------------------------
// Window function stage: execute window functions and append result columns
// ---------------------------------------------------------------------------

type WindowStageResult = (Table, HashMap<String, usize>, Vec<SelectItem>);

/// Execute window functions and return (updated_table, updated_schema, rewritten_select_items).
/// Window function calls in SELECT are replaced with identifier references to the new columns.
fn plan_window_stage(
    ctx: &Context,
    table: &Table,
    select_items: &[SelectItem],
    schema: &HashMap<String, usize>,
) -> Result<WindowStageResult, SqlError> {
    let win_funcs = collect_window_functions(select_items)?;
    if win_funcs.is_empty() {
        // No actual window functions found (shouldn't happen, caller checked)
        let new_schema = schema.clone();
        return Ok((table.clone_ref(), new_schema, select_items.to_vec()));
    }
    // Group window functions by WindowSpec so each spec gets a dedicated OP_WINDOW.
    // This avoids applying the first spec to all functions when multiple OVER specs
    // are present in the same SELECT list.
    let mut spec_groups: Vec<(sqlparser::ast::WindowSpec, Vec<usize>)> = Vec::new();
    for (func_idx, (_item_idx, info)) in win_funcs.iter().enumerate() {
        if let Some((_, idxs)) = spec_groups.iter_mut().find(|(spec, _)| *spec == info.spec) {
            idxs.push(func_idx);
        } else {
            spec_groups.push((info.spec.clone(), vec![func_idx]));
        }
    }

    let mut current_table = table.clone_ref();
    let mut current_schema = schema.clone();
    let mut next_win_col = 0usize;
    let mut win_col_names: Vec<String> = vec![String::new(); win_funcs.len()];
    let win_display_names: Vec<String> = win_funcs
        .iter()
        .map(|(_, info)| info.display_name.clone())
        .collect();

    for (spec, func_indices) in spec_groups {
        let stage_result = {
            let mut g = ctx.graph(&current_table)?;
            let table_node = g.const_table(&current_table)?;
            let (frame_type, frame_start, frame_end) = parse_window_frame(&spec)?;

            let mut part_key_cols: Vec<Column> = Vec::new();
            for part_expr in &spec.partition_by {
                part_key_cols.push(plan_expr(&mut g, part_expr, &current_schema)?);
            }

            let mut order_key_cols: Vec<Column> = Vec::new();
            let mut order_descs: Vec<bool> = Vec::new();
            for ob in &spec.order_by {
                order_key_cols.push(plan_expr(&mut g, &ob.expr, &current_schema)?);
                order_descs.push(ob.asc == Some(false));
            }

            let mut funcs: Vec<crate::WindowFunc> = Vec::new();
            let mut func_input_cols: Vec<Column> = Vec::new();
            for &func_idx in &func_indices {
                let info = &win_funcs[func_idx].1;
                funcs.push(info.func);
                let input_col = if let Some(ref input_expr) = info.input_expr {
                    plan_expr(&mut g, input_expr, &current_schema)?
                } else {
                    let first_col_name = current_schema
                        .iter()
                        .min_by_key(|(_, v)| **v)
                        .map(|(k, _)| k.clone())
                        .ok_or_else(|| {
                            SqlError::Plan(
                                "Window function requires at least one input column".into(),
                            )
                        })?;
                    g.scan(&first_col_name)?
                };
                func_input_cols.push(input_col);
            }

            let win_node = g.window_op(
                table_node,
                &part_key_cols,
                &order_key_cols,
                &order_descs,
                &funcs,
                &func_input_cols,
                frame_type,
                frame_start,
                frame_end,
            )?;
            g.execute(win_node)?
        };

        // Normalize generated window column names to stable unique names.
        let prev_ncols = current_table.ncols() as usize;
        let stage_ncols = stage_result.ncols() as usize;
        if stage_ncols != prev_ncols + func_indices.len() {
            return Err(SqlError::Plan(format!(
                "Window stage produced unexpected column count: expected {}, got {}",
                prev_ncols + func_indices.len(),
                stage_ncols
            )));
        }
        let mut renamed_cols: Vec<String> = (0..stage_ncols)
            .map(|i| stage_result.col_name_str(i).to_string())
            .collect();
        for (offset, &func_idx) in func_indices.iter().enumerate() {
            let col_name = format!("_w{next_win_col}");
            next_win_col += 1;
            renamed_cols[prev_ncols + offset] = col_name.clone();
            win_col_names[func_idx] = col_name;
        }

        current_table = stage_result.with_column_names(&renamed_cols)?;
        current_schema = build_schema(&current_table);
    }

    let mut new_schema = current_schema.clone();
    // Also add display names as aliases (e.g. "row_number()" -> _w0 column index)
    for (i, display) in win_display_names.iter().enumerate() {
        if let Some(col_idx) = new_schema.get(&win_col_names[i]).copied() {
            new_schema.entry(display.clone()).or_insert(col_idx);
        }
    }

    // Rewrite SELECT items: replace window function calls with identifier refs.
    // Handles both top-level window functions and nested ones (e.g. ROW_NUMBER() OVER(...) <= 3).
    // Wildcards are expanded to original columns only (excluding _w0, _w1, ... intermediates).
    let mut win_idx = 0;
    let mut new_items: Vec<SelectItem> = Vec::new();
    for item in select_items {
        match item {
            SelectItem::UnnamedExpr(expr) => {
                let rewritten = rewrite_window_refs(expr, &win_col_names, &mut win_idx)?;
                new_items.push(SelectItem::UnnamedExpr(rewritten));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let rewritten = rewrite_window_refs(expr, &win_col_names, &mut win_idx)?;
                new_items.push(SelectItem::ExprWithAlias {
                    expr: rewritten,
                    alias: alias.clone(),
                });
            }
            SelectItem::Wildcard(_) => {
                // Expand to original columns only — skip _w* intermediates
                let mut cols: Vec<_> = schema.iter().collect();
                cols.sort_by_key(|(_, idx)| **idx);
                for (name, _) in cols {
                    new_items.push(SelectItem::UnnamedExpr(Expr::Identifier(Ident::new(
                        name.clone(),
                    ))));
                }
            }
            other => new_items.push(other.clone()),
        }
    }

    Ok((current_table, new_schema, new_items))
}

/// Recursively replace window function calls in an expression with identifier
/// references to pre-computed window result columns (_w0, _w1, ...).
fn rewrite_window_refs(
    expr: &Expr,
    col_names: &[String],
    idx: &mut usize,
) -> Result<Expr, SqlError> {
    match expr {
        Expr::Function(f) if f.over.is_some() => {
            let col_name = col_names
                .get(*idx)
                .cloned()
                .ok_or_else(|| SqlError::Plan("Window function rewrite mismatch".into()))?;
            *idx += 1;
            Ok(Expr::Identifier(Ident::new(col_name)))
        }
        Expr::BinaryOp { left, op, right } => Ok(Expr::BinaryOp {
            left: Box::new(rewrite_window_refs(left, col_names, idx)?),
            op: op.clone(),
            right: Box::new(rewrite_window_refs(right, col_names, idx)?),
        }),
        Expr::UnaryOp { op, expr: inner } => Ok(Expr::UnaryOp {
            op: *op,
            expr: Box::new(rewrite_window_refs(inner, col_names, idx)?),
        }),
        Expr::Nested(inner) => Ok(Expr::Nested(Box::new(rewrite_window_refs(
            inner, col_names, idx,
        )?))),
        Expr::Cast {
            expr: inner,
            data_type,
            format,
            kind,
        } => Ok(Expr::Cast {
            expr: Box::new(rewrite_window_refs(inner, col_names, idx)?),
            data_type: data_type.clone(),
            format: format.clone(),
            kind: kind.clone(),
        }),
        other => Ok(other.clone()),
    }
}

// ---------------------------------------------------------------------------
// OFFSET: skip first N rows
// ---------------------------------------------------------------------------

fn skip_rows(ctx: &Context, table: &Table, offset: i64) -> Result<Table, SqlError> {
    let nrows = table.nrows();
    if offset >= nrows {
        let g = ctx.graph(table)?;
        let table_node = g.const_table(table)?;
        let root = g.head(table_node, 0)?;
        return Ok(g.execute(root)?);
    }
    // td_tail takes the last N rows — exactly what we need after skipping offset
    let keep = nrows - offset;
    let g = ctx.graph(table)?;
    let table_node = g.const_table(table)?;
    let root = g.tail(table_node, keep)?;
    Ok(g.execute(root)?)
}

fn engine_err_from_raw(ptr: *mut crate::td_t) -> SqlError {
    match crate::ffi_error_from_ptr(ptr) {
        Some(err) => SqlError::Engine(err),
        None => SqlError::Engine(crate::Error::Oom),
    }
}

struct RawTableBuilder {
    raw: *mut crate::td_t,
}

impl RawTableBuilder {
    fn new(ncols: i64) -> Result<Self, SqlError> {
        let raw = unsafe { crate::ffi_table_new(ncols) };
        if raw.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(raw) {
            return Err(engine_err_from_raw(raw));
        }
        Ok(Self { raw })
    }

    fn add_col(&mut self, name_id: i64, col: *mut crate::td_t) -> Result<(), SqlError> {
        let next = unsafe { crate::ffi_table_add_col(self.raw, name_id, col) };
        if next.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(next) {
            return Err(engine_err_from_raw(next));
        }
        self.raw = next;
        Ok(())
    }

    fn finish(mut self) -> Result<Table, SqlError> {
        let raw = self.raw;
        self.raw = std::ptr::null_mut(); // prevent Drop from releasing
                                         // No retain: transfer existing rc=1 ownership to Table
        match unsafe { Table::from_raw(raw) } {
            Ok(t) => Ok(t),
            Err(e) => {
                // Release the allocation's rc=1 to avoid a leak
                unsafe { crate::ffi_release(raw) };
                Err(SqlError::Engine(e))
            }
        }
    }
}

impl Drop for RawTableBuilder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { crate::ffi_release(self.raw) };
        }
    }
}

fn is_vector_column(col: *mut crate::td_t) -> bool {
    unsafe { crate::raw::td_type(col) > 0 }
}

fn ensure_vector_columns(table: &Table, op: &str) -> Result<Table, SqlError> {
    for c in 0..table.ncols() {
        let col = table
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("table column missing".into()))?;
        if !is_vector_column(col) {
            let col_type = unsafe { crate::raw::td_type(col) };
            return Err(SqlError::Plan(format!(
                "{op}: scalar column type {col_type} at index {c} is not supported"
            )));
        }
    }
    Ok(table.clone_ref())
}

/// Create a new vector matching the type and width of a source column.
/// For TD_SYM, uses `td_sym_vec_new` to preserve the adaptive width.
fn col_vec_new(src: *const crate::td_t, capacity: i64) -> *mut crate::td_t {
    let col_type = unsafe { (*src).type_ };
    if col_type == crate::ffi::TD_SYM {
        let attrs = unsafe { (*src).attrs };
        unsafe { crate::ffi::td_sym_vec_new(attrs & crate::ffi::TD_SYM_W_MASK, capacity) }
    } else {
        unsafe { crate::raw::td_vec_new(col_type, capacity) }
    }
}

/// Element size for a column pointer, handling TD_SYM adaptive width.
fn col_elem_size(col: *const crate::td_t) -> usize {
    let col_type = unsafe { (*col).type_ };
    if col_type == crate::ffi::TD_SYM {
        let attrs = unsafe { (*col).attrs };
        match attrs & crate::ffi::TD_SYM_W_MASK {
            crate::ffi::TD_SYM_W8 => 1,
            crate::ffi::TD_SYM_W16 => 2,
            crate::ffi::TD_SYM_W32 => 4,
            _ => 8,
        }
    } else {
        let sizes = unsafe { &crate::raw::td_type_sizes };
        sizes.get(col_type as usize).copied().unwrap_or(0) as usize
    }
}

// ---------------------------------------------------------------------------
// UNION ALL: concatenate two tables
// ---------------------------------------------------------------------------

fn concat_tables(ctx: &Context, left: &Table, right: &Table) -> Result<Table, SqlError> {
    let _ = ctx;
    let left = ensure_vector_columns(left, "UNION ALL")?;
    let right = ensure_vector_columns(right, "UNION ALL")?;

    let ncols = left.ncols();
    if ncols != right.ncols() {
        return Err(SqlError::Plan("UNION ALL: column count mismatch".into()));
    }

    let mut result = RawTableBuilder::new(ncols)?;
    for c in 0..ncols {
        let l_col = left
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("UNION ALL: left column missing".into()))?;
        let r_col = right
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("UNION ALL: right column missing".into()))?;
        let merged = unsafe { crate::ffi_vec_concat(l_col, r_col) };
        if merged.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(merged) {
            return Err(engine_err_from_raw(merged));
        }
        let name_id = left.col_name(c);
        let add_res = result.add_col(name_id, merged);
        unsafe { crate::ffi_release(merged) };
        add_res?;
    }
    result.finish()
}

/// Execute a CROSS JOIN (Cartesian product) of two tables.
///
/// Rejects columns with null bitmaps because the memcpy expansion does not
/// preserve them. This is fine in practice: cross join is only used for
/// small literal tables that never contain nulls.
fn exec_cross_join(ctx: &Context, left: &Table, right: &Table) -> Result<Table, SqlError> {
    let _ = ctx;
    let left = ensure_vector_columns(left, "CROSS JOIN")?;
    let right = ensure_vector_columns(right, "CROSS JOIN")?;

    // Reject columns that carry null bitmaps — memcpy cannot preserve them.
    for tbl in [&left, &right] {
        for c in 0..tbl.ncols() {
            if let Some(col) = tbl.get_col_idx(c) {
                let attrs = unsafe { (*col).attrs };
                if attrs & crate::ffi::TD_ATTR_HAS_NULLS != 0 {
                    return Err(SqlError::Plan(
                        "CROSS JOIN does not support columns with NULL values".into(),
                    ));
                }
            }
        }
    }

    let l_nrows = left.nrows() as usize;
    let r_nrows = right.nrows() as usize;
    let out_nrows = l_nrows
        .checked_mul(r_nrows)
        .ok_or_else(|| SqlError::Plan("CROSS JOIN result too large".into()))?;
    let l_ncols = left.ncols();
    let r_ncols = right.ncols();

    let mut result = RawTableBuilder::new(l_ncols + r_ncols)?;

    // Left columns: repeat each row r_nrows times
    for c in 0..l_ncols {
        let col = left
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("CROSS JOIN: left column missing".into()))?;
        let name_id = left.col_name(c);
        let esz = col_elem_size(col);
        let new_col = col_vec_new(col, out_nrows as i64);
        if new_col.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(new_col) {
            return Err(engine_err_from_raw(new_col));
        }
        unsafe { crate::raw::td_set_len(new_col, out_nrows as i64) };
        let src = unsafe { crate::raw::td_data(col) };
        let dst = unsafe { crate::raw::td_data(new_col) };
        for lr in 0..l_nrows {
            for rr in 0..r_nrows {
                let out_row = lr * r_nrows + rr;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.add(lr * esz), dst.add(out_row * esz), esz);
                }
            }
        }
        let add_res = result.add_col(name_id, new_col);
        unsafe { crate::ffi_release(new_col) };
        add_res?;
    }

    // Right columns: tile the entire column l_nrows times
    for c in 0..r_ncols {
        let col = right
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("CROSS JOIN: right column missing".into()))?;
        let name_id = right.col_name(c);
        let esz = col_elem_size(col);
        let new_col = col_vec_new(col, out_nrows as i64);
        if new_col.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(new_col) {
            return Err(engine_err_from_raw(new_col));
        }
        unsafe { crate::raw::td_set_len(new_col, out_nrows as i64) };
        let src = unsafe { crate::raw::td_data(col) };
        let dst = unsafe { crate::raw::td_data(new_col) };
        for lr in 0..l_nrows {
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.add(lr * r_nrows * esz), r_nrows * esz);
            }
        }
        let add_res = result.add_col(name_id, new_col);
        unsafe { crate::ffi_release(new_col) };
        add_res?;
    }
    result.finish()
}

/// Execute EXCEPT ALL or INTERSECT ALL between two tables.
/// `keep_matches = true` → INTERSECT (keep left rows that exist in right).
/// `keep_matches = false` → EXCEPT (keep left rows that do NOT exist in right).
fn exec_set_operation(
    ctx: &Context,
    left: &Table,
    right: &Table,
    keep_matches: bool,
) -> Result<Table, SqlError> {
    use std::collections::HashMap as StdMap;

    let _ = ctx;
    let left = ensure_vector_columns(left, "SET operation")?;
    let right = ensure_vector_columns(right, "SET operation")?;

    let l_nrows = left.nrows() as usize;
    let r_nrows = right.nrows() as usize;
    let ncols = left.ncols();

    let left_cols = collect_setop_columns(&left, ncols)?;
    let right_cols = collect_setop_columns(&right, ncols)?;

    // Hash all right-side rows into buckets; exact row equality is checked on probe.
    // NOTE: DefaultHasher is non-deterministic across Rust versions (SipHash with
    // random seed). This is acceptable here because the hash is only used as a
    // partition key for the probe phase — correctness relies on setop_rows_equal,
    // not on hash stability across runs.
    let mut right_buckets: StdMap<u64, Vec<usize>> = StdMap::new();
    for r in 0..r_nrows {
        let h = hash_setop_row(&right_cols, r);
        right_buckets.entry(h).or_default().push(r);
    }

    // Probe with left-side rows, collect indices to keep
    let mut keep_indices: Vec<usize> = Vec::new();
    let mut remaining = vec![1usize; r_nrows];
    for r in 0..l_nrows {
        let h = hash_setop_row(&left_cols, r);
        let matched_right_row = right_buckets.get(&h).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&rr| remaining[rr] > 0 && setop_rows_equal(&left_cols, r, &right_cols, rr))
        });

        if keep_matches {
            // INTERSECT: keep if in right
            if let Some(rr) = matched_right_row {
                keep_indices.push(r);
                remaining[rr] -= 1;
            }
        } else {
            // EXCEPT: keep if NOT in right
            if let Some(rr) = matched_right_row {
                remaining[rr] -= 1;
            } else {
                keep_indices.push(r);
            }
        }
    }

    let mut result = RawTableBuilder::new(ncols)?;
    let out_nrows = keep_indices.len();

    for c in 0..ncols {
        let col = left
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("SET operation: column missing".into()))?;
        let name_id = left.col_name(c);
        let esz = col_elem_size(col);
        let new_col = col_vec_new(col, out_nrows as i64);
        if new_col.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(new_col) {
            return Err(engine_err_from_raw(new_col));
        }
        unsafe { crate::raw::td_set_len(new_col, out_nrows as i64) };
        let src = unsafe { crate::raw::td_data(col) };
        let dst = unsafe { crate::raw::td_data(new_col) };
        for (out_row, &in_row) in keep_indices.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(in_row * esz), dst.add(out_row * esz), esz);
            }
        }
        let add_res = result.add_col(name_id, new_col);
        unsafe { crate::ffi_release(new_col) };
        add_res?;
    }
    result.finish()
}

/// Raw column data pointer -- valid only while the source Table is alive.
/// Do not store beyond the scope of exec_set_operation.
#[derive(Clone, Copy)]
struct SetOpCol {
    col_type: i8,
    elem_size: usize,
    len: usize,
    data: *const u8,
}

fn collect_setop_columns(table: &Table, ncols: i64) -> Result<Vec<SetOpCol>, SqlError> {
    let mut cols = Vec::with_capacity(ncols as usize);
    for c in 0..ncols {
        let col = table
            .get_col_idx(c)
            .ok_or_else(|| SqlError::Plan("SET operation: column missing".into()))?;
        let col_type = unsafe { crate::raw::td_type(col) };
        let elem_size = col_elem_size(col);
        let len = unsafe { crate::ffi::td_len(col as *const crate::td_t) } as usize;
        let data = unsafe { crate::raw::td_data(col) } as *const u8;
        cols.push(SetOpCol {
            col_type,
            elem_size,
            len,
            data,
        });
    }
    Ok(cols)
}

fn hash_setop_row(cols: &[SetOpCol], row: usize) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for col in cols {
        col.col_type.hash(&mut hasher);
        unsafe { setop_cell_bytes(col, row) }.hash(&mut hasher);
    }
    hasher.finish()
}

fn setop_rows_equal(
    left_cols: &[SetOpCol],
    left_row: usize,
    right_cols: &[SetOpCol],
    right_row: usize,
) -> bool {
    left_cols.iter().zip(right_cols.iter()).all(|(l, r)| {
        l.col_type == r.col_type
            && l.elem_size == r.elem_size
            && unsafe { setop_cell_bytes(l, left_row) == setop_cell_bytes(r, right_row) }
    })
}

/// Return one cell as raw bytes for fixed-width set-operation comparison.
///
/// # Safety
/// `col.data` must point to a contiguous allocation of at least
/// `col.len * col.elem_size` bytes.
///
/// # Panics
/// Panics if `row >= col.len`.
unsafe fn setop_cell_bytes(col: &SetOpCol, row: usize) -> &[u8] {
    assert!(
        row < col.len,
        "setop_cell_bytes: row {} out of bounds (len {})",
        row,
        col.len
    );
    let byte_offset = row * col.elem_size;
    unsafe { std::slice::from_raw_parts(col.data.add(byte_offset), col.elem_size) }
}

/// Apply ORDER BY and LIMIT from the outer query to a result.
fn apply_post_processing(
    ctx: &Context,
    query: &Query,
    result_table: Table,
    result_aliases: Vec<String>,
    _tables: Option<&HashMap<String, StoredTable>>,
) -> Result<SqlResult, SqlError> {
    // ORDER BY (optionally fused with LIMIT)
    let order_by_exprs = extract_order_by(query)?;
    let offset_val = extract_offset(query)?;
    let limit_val = extract_limit(query)?;

    let (result_table, limit_fused) = if !order_by_exprs.is_empty() {
        let table_col_names: Vec<String> = (0..result_table.ncols() as usize)
            .map(|i| result_table.col_name_str(i).to_string())
            .collect();
        let mut g = ctx.graph(&result_table)?;
        let table_node = g.const_table(&result_table)?;
        let sort_node = plan_order_by(
            &mut g,
            table_node,
            &order_by_exprs,
            &result_aliases,
            &table_col_names,
        )?;

        let total_limit = match (offset_val, limit_val) {
            (Some(off), Some(lim)) => Some(off.saturating_add(lim)),
            (None, Some(lim)) => Some(lim),
            _ => None,
        };
        let root = match total_limit {
            Some(n) => g.head(sort_node, n)?,
            None => sort_node,
        };
        (g.execute(root)?, total_limit.is_some())
    } else {
        (result_table, false)
    };

    // OFFSET + LIMIT (only parts not already fused)
    let result_table = if limit_fused {
        match offset_val {
            Some(off) => skip_rows(ctx, &result_table, off)?,
            None => result_table,
        }
    } else {
        match (offset_val, limit_val) {
            (Some(off), Some(lim)) => {
                let total = off.saturating_add(lim);
                let g = ctx.graph(&result_table)?;
                let table_node = g.const_table(&result_table)?;
                let head_node = g.head(table_node, total)?;
                let trimmed = g.execute(head_node)?;
                skip_rows(ctx, &trimmed, off)?
            }
            (Some(off), None) => skip_rows(ctx, &result_table, off)?,
            (None, Some(lim)) => {
                let g = ctx.graph(&result_table)?;
                let table_node = g.const_table(&result_table)?;
                let root = g.head(table_node, lim)?;
                g.execute(root)?
            }
            (None, None) => result_table,
        }
    };

    validate_result_table(&result_table)?;
    Ok(SqlResult {
        table: result_table,
        columns: result_aliases,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_result_table(table: &Table) -> Result<(), SqlError> {
    let nrows = table.nrows();
    for col_idx in 0..table.ncols() {
        let col = table.get_col_idx(col_idx).ok_or_else(|| {
            SqlError::Plan(format!("Result column at index {col_idx} is missing"))
        })?;
        let col_type = unsafe { crate::raw::td_type(col) };
        if col_type <= 0 {
            return Err(SqlError::Plan(format!(
                "Result column '{}' has scalar type {} and is not supported",
                table.col_name_str(col_idx as usize),
                col_type
            )));
        }
        // TD_PARTED and MAPCOMMON columns: len = partition count, not row count.
        // Skip the length check — td_table_nrows() already handles them correctly.
        if crate::ffi::td_is_parted(col_type) || col_type == crate::ffi::TD_MAPCOMMON {
            continue;
        }
        let len = unsafe { crate::raw::td_len(col) };
        if len != nrows {
            return Err(SqlError::Plan(format!(
                "Result column '{}' has length {} but table has {} rows",
                table.col_name_str(col_idx as usize),
                len,
                nrows
            )));
        }
    }
    Ok(())
}

/// Convert ObjectName to a string.
/// Multi-part names (schema.table) are joined with '.' as a flat key.
fn object_name_to_string(name: &ObjectName) -> String {
    name.0
        .iter()
        .map(|ident| ident.value.clone())
        .collect::<Vec<_>>()
        .join(".")
}

/// Check that a table path is safe: no parent traversal or null bytes.
/// Absolute paths are allowed (local library, user has full file access).
/// Build a column name -> index map from the table.
fn build_schema(table: &Table) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    let ncols = table.ncols() as usize;
    for i in 0..ncols {
        let name = table.col_name_str(i);
        if !name.is_empty() {
            map.insert(name.to_lowercase(), i);
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Subquery resolution: walk AST, execute subqueries, replace with literals
// ---------------------------------------------------------------------------

/// Recursively walk an expression and replace scalar subqueries and IN subqueries
/// with their evaluated values.
fn resolve_subqueries(
    ctx: &Context,
    expr: &Expr,
    tables: Option<&HashMap<String, StoredTable>>,
) -> Result<Expr, SqlError> {
    match expr {
        // Scalar subquery: (SELECT single_value FROM ...)
        Expr::Subquery(query) => {
            let result = plan_query(ctx, query, tables)?;
            if result.columns.len() != 1 {
                return Err(SqlError::Plan(format!(
                    "Scalar subquery must return exactly 1 column, got {}",
                    result.columns.len()
                )));
            }
            let nrows = result.table.nrows();
            if nrows != 1 {
                return Err(SqlError::Plan(format!(
                    "Scalar subquery must return exactly 1 row, got {}",
                    nrows
                )));
            }
            scalar_value_from_table(&result.table, 0, 0)
        }

        // IN (subquery): rewrite to IN (value_list)
        Expr::InSubquery {
            expr: inner,
            subquery,
            negated,
        } => {
            let resolved_inner = resolve_subqueries(ctx, inner, tables)?;
            let result = plan_query(ctx, subquery, tables)?;
            if result.columns.len() != 1 {
                return Err(SqlError::Plan(format!(
                    "IN subquery must return exactly 1 column, got {}",
                    result.columns.len()
                )));
            }
            let nrows = result.table.nrows() as usize;
            let mut values = Vec::with_capacity(nrows);
            for r in 0..nrows {
                values.push(scalar_value_from_table(&result.table, 0, r)?);
            }
            Ok(Expr::InList {
                expr: Box::new(resolved_inner),
                list: values,
                negated: *negated,
            })
        }

        // EXISTS (subquery): evaluate and replace with boolean literal
        Expr::Exists { subquery, negated } => {
            let result = plan_query(ctx, subquery, tables)?;
            let exists = result.table.nrows() > 0;
            Ok(Expr::Value(Value::Boolean(exists ^ negated)))
        }

        // Recurse into compound expressions
        Expr::BinaryOp { left, op, right } => Ok(Expr::BinaryOp {
            left: Box::new(resolve_subqueries(ctx, left, tables)?),
            op: op.clone(),
            right: Box::new(resolve_subqueries(ctx, right, tables)?),
        }),
        Expr::UnaryOp { op, expr: inner } => Ok(Expr::UnaryOp {
            op: *op,
            expr: Box::new(resolve_subqueries(ctx, inner, tables)?),
        }),
        Expr::Nested(inner) => Ok(Expr::Nested(Box::new(resolve_subqueries(
            ctx, inner, tables,
        )?))),
        Expr::Between {
            expr: inner,
            negated,
            low,
            high,
        } => Ok(Expr::Between {
            expr: Box::new(resolve_subqueries(ctx, inner, tables)?),
            negated: *negated,
            low: Box::new(resolve_subqueries(ctx, low, tables)?),
            high: Box::new(resolve_subqueries(ctx, high, tables)?),
        }),
        Expr::IsNull(inner) => Ok(Expr::IsNull(Box::new(resolve_subqueries(
            ctx, inner, tables,
        )?))),
        Expr::IsNotNull(inner) => Ok(Expr::IsNotNull(Box::new(resolve_subqueries(
            ctx, inner, tables,
        )?))),

        // Leaf nodes: no subqueries to resolve
        _ => Ok(expr.clone()),
    }
}

/// Extract a scalar value from a result table cell as an AST expression literal.
fn scalar_value_from_table(table: &Table, col: usize, row: usize) -> Result<Expr, SqlError> {
    let col_type = table.col_type(col);
    match col_type {
        crate::types::F64 => {
            let v = table.get_f64(col, row).unwrap_or(f64::NAN);
            if v.is_nan() {
                Ok(Expr::Value(Value::Null))
            } else {
                Ok(Expr::Value(Value::Number(format!("{v}"), false)))
            }
        }
        crate::types::I64 | crate::types::I32 => match table.get_i64(col, row) {
            Some(v) => Ok(Expr::Value(Value::Number(v.to_string(), false))),
            None => Ok(Expr::Value(Value::Null)),
        },
        crate::types::SYM => {
            let v = table.get_str(col, row).unwrap_or_default();
            Ok(Expr::Value(Value::SingleQuotedString(v)))
        }
        crate::types::BOOL => {
            let v = table.get_i64(col, row).unwrap_or(0);
            Ok(Expr::Value(Value::Boolean(v != 0)))
        }
        _ => Err(SqlError::Plan(format!(
            "Unsupported column type {} in subquery result",
            col_type
        ))),
    }
}

/// Check if an expression tree contains any subqueries that need resolution.
fn has_subqueries(expr: &Expr) -> bool {
    match expr {
        Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists { .. } => true,
        Expr::BinaryOp { left, right, .. } => has_subqueries(left) || has_subqueries(right),
        Expr::UnaryOp { expr, .. } => has_subqueries(expr),
        Expr::Nested(inner) => has_subqueries(inner),
        Expr::Between {
            expr, low, high, ..
        } => has_subqueries(expr) || has_subqueries(low) || has_subqueries(high),
        Expr::IsNull(inner) | Expr::IsNotNull(inner) => has_subqueries(inner),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Predicate pushdown into FROM subqueries
// ---------------------------------------------------------------------------

/// Split a conjunction (AND chain) into individual terms.
fn split_conjunction(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut terms = split_conjunction(left);
            terms.extend(split_conjunction(right));
            terms
        }
        other => vec![other],
    }
}

/// Join terms back into a conjunction (AND chain). Returns None if empty.
fn join_conjunction(terms: Vec<Expr>) -> Option<Expr> {
    let mut iter = terms.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, term| Expr::BinaryOp {
        left: Box::new(acc),
        op: BinaryOperator::And,
        right: Box::new(term),
    }))
}

/// If expr is `column = literal`, return the column name.
fn extract_equality_column(expr: &Expr) -> Option<String> {
    if let Expr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    {
        if let Expr::Identifier(ident) = left.as_ref() {
            if matches!(right.as_ref(), Expr::Value(_)) {
                return Some(ident.value.to_lowercase());
            }
        }
        if let Expr::Identifier(ident) = right.as_ref() {
            if matches!(left.as_ref(), Expr::Value(_)) {
                return Some(ident.value.to_lowercase());
            }
        }
    }
    None
}

/// Determine which columns can accept pushdown predicates from an outer query.
/// - Window functions: PARTITION BY key columns (intersection of all windows)
/// - GROUP BY: GROUP BY key columns
/// - Neither: empty set (no pushdown — can't verify column origin safely)
fn get_pushable_columns_from_query(query: &Query) -> HashSet<String> {
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s,
        _ => return HashSet::new(),
    };

    if has_window_functions(&select.projection) {
        let win_funcs = match collect_window_functions(&select.projection) {
            Ok(wf) => wf,
            Err(_) => return HashSet::new(),
        };
        let mut pkeys: Option<HashSet<String>> = None;
        for (_, info) in &win_funcs {
            let mut keys = HashSet::new();
            for e in &info.spec.partition_by {
                if let Expr::Identifier(id) = e {
                    keys.insert(id.value.to_lowercase());
                }
            }
            pkeys = Some(match pkeys {
                None => keys,
                Some(existing) => existing.intersection(&keys).cloned().collect(),
            });
        }
        pkeys.unwrap_or_default()
    } else {
        match &select.group_by {
            GroupByExpr::Expressions(exprs, _) if !exprs.is_empty() => {
                let mut cols = HashSet::new();
                for e in exprs {
                    if let Expr::Identifier(id) = e {
                        cols.insert(id.value.to_lowercase());
                    }
                }
                cols
            }
            _ => HashSet::new(),
        }
    }
}

/// Clone a query and inject additional WHERE predicates (ANDed with existing WHERE).
fn inject_predicates_into_query(query: &Query, preds: &[Expr]) -> Query {
    let mut q = query.clone();
    if preds.is_empty() {
        return q;
    }
    let Some(new_pred) = join_conjunction(preds.to_vec()) else {
        return q;
    };
    if let SetExpr::Select(ref mut select) = *q.body {
        select.selection = Some(match select.selection.take() {
            Some(existing) => Expr::BinaryOp {
                left: Box::new(existing),
                op: BinaryOperator::And,
                right: Box::new(new_pred),
            },
            None => new_pred,
        });
    }
    q
}

// ---------------------------------------------------------------------------

/// Extract GROUP BY column names.
/// Accepts table column names or SELECT alias names (for expression-based keys).
fn extract_group_by_columns(
    group_by: &GroupByExpr,
    schema: &HashMap<String, usize>,
    alias_exprs: &mut HashMap<String, Expr>,
    select_aliases: &[String],
) -> Result<Vec<String>, SqlError> {
    match group_by {
        GroupByExpr::All(_) => Err(SqlError::Plan("GROUP BY ALL not supported".into())),
        GroupByExpr::Expressions(exprs, _modifiers) => {
            let mut cols = Vec::new();
            let mut gb_counter = 0usize;
            for expr in exprs {
                match expr {
                    Expr::Identifier(ident) => {
                        let name = ident.value.to_lowercase();
                        if !schema.contains_key(&name) && !alias_exprs.contains_key(&name) {
                            return Err(SqlError::Plan(format!(
                                "GROUP BY column '{}' not found",
                                name
                            )));
                        }
                        cols.push(name);
                    }
                    // Positional GROUP BY: GROUP BY 1, 2
                    Expr::Value(Value::Number(n, _)) => {
                        let pos = n.parse::<usize>().map_err(|_| {
                            SqlError::Plan(format!("Invalid positional GROUP BY: {n}"))
                        })?;
                        if pos == 0 || pos > select_aliases.len() {
                            return Err(SqlError::Plan(format!(
                                "GROUP BY position {} out of range (1-{})",
                                pos,
                                select_aliases.len()
                            )));
                        }
                        cols.push(select_aliases[pos - 1].clone());
                    }
                    // Expression GROUP BY: GROUP BY FLOOR(v1/10)
                    other => {
                        let alias = format!("_gb_{}", gb_counter);
                        gb_counter += 1;
                        alias_exprs.insert(alias.clone(), other.clone());
                        cols.push(alias);
                    }
                }
            }
            Ok(cols)
        }
    }
}

/// Extract column aliases from a SELECT list (for simple projection).
fn extract_projection_aliases(
    select_items: &[SelectItem],
    schema: &HashMap<String, usize>,
) -> Result<Vec<String>, SqlError> {
    let mut aliases = Vec::new();

    for item in select_items {
        match item {
            SelectItem::Wildcard(_) => {
                let mut cols: Vec<_> = schema.iter().collect();
                cols.sort_by_key(|(_name, idx)| **idx);
                for (name, _idx) in cols {
                    aliases.push(name.clone());
                }
            }
            SelectItem::UnnamedExpr(expr) => {
                aliases.push(expr_default_name(expr));
            }
            SelectItem::ExprWithAlias { alias, .. } => {
                aliases.push(alias.value.to_lowercase());
            }
            _ => return Err(SqlError::Plan("Unsupported SELECT item".into())),
        }
    }

    if aliases.is_empty() {
        return Err(SqlError::Plan("SELECT list is empty".into()));
    }

    Ok(aliases)
}

/// Source column name for projection passthrough checks.
/// Returns `None` for computed expressions.
fn projection_source_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Identifier(ident) => Some(ident.value.to_lowercase()),
        Expr::CompoundIdentifier(parts) => parts.last().map(|p| p.value.to_lowercase()),
        Expr::Nested(inner) => projection_source_name(inner),
        _ => None,
    }
}

/// True when SELECT items are an identity projection over the current schema.
/// Allows `SELECT *` and selecting all base columns in table order.
fn is_identity_projection(select_items: &[SelectItem], schema: &HashMap<String, usize>) -> bool {
    if select_items.len() == 1 && matches!(select_items[0], SelectItem::Wildcard(_)) {
        return true;
    }

    let mut schema_cols: Vec<_> = schema.iter().collect();
    schema_cols.sort_by_key(|(_name, idx)| **idx);
    let schema_names: Vec<String> = schema_cols
        .into_iter()
        .map(|(name, _idx)| name.clone())
        .collect();

    let mut projected_names: Vec<String> = Vec::with_capacity(select_items.len());
    for item in select_items {
        let src = match item {
            SelectItem::UnnamedExpr(expr) => projection_source_name(expr),
            SelectItem::ExprWithAlias { expr, .. } => projection_source_name(expr),
            SelectItem::Wildcard(_) => None, // wildcard mixed with others is not identity
            _ => None,
        };
        let Some(name) = src else {
            return false;
        };
        projected_names.push(name);
    }

    projected_names == schema_names
}

/// Collect ORDER BY columns that are valid source columns but not part of the
/// visible SELECT projection. These will be carried as hidden columns.
fn collect_hidden_order_columns(
    order_by: &[(OrderByItem, bool, Option<bool>)],
    visible_aliases: &[String],
    schema: &HashMap<String, usize>,
) -> Vec<String> {
    let mut extra = Vec::new();
    for (item, _, _) in order_by {
        let OrderByItem::Name(name) = item else {
            continue;
        };
        if visible_aliases.iter().any(|a| a == name) {
            continue;
        }
        if schema.contains_key(name) && !extra.iter().any(|c| c == name) {
            extra.push(name.clone());
        }
    }
    extra
}

/// Ensure the physical table column count matches visible SQL columns by
/// projecting away any hidden helper columns.
fn trim_to_visible_columns(
    ctx: &Context,
    table: Table,
    visible_aliases: &[String],
) -> Result<Table, SqlError> {
    if table.ncols() as usize == visible_aliases.len() {
        return Ok(table);
    }

    let g = ctx.graph(&table)?;
    let table_node = g.const_table(&table)?;
    let proj_cols: Vec<Column> = visible_aliases
        .iter()
        .map(|name| g.scan(name))
        .collect::<crate::Result<Vec<_>>>()?;
    let proj = g.select(table_node, &proj_cols)?;
    Ok(g.execute(proj)?)
}

/// An ORDER BY item: either a column name, positional index, or arbitrary expression.
enum OrderByItem {
    Name(String),
    Position(usize), // 1-based index
    Expression(Box<Expr>),
}

/// Extract ORDER BY items from the query.
fn extract_order_by(query: &Query) -> Result<Vec<(OrderByItem, bool, Option<bool>)>, SqlError> {
    match &query.order_by {
        None => Ok(Vec::new()),
        Some(order_by) => {
            let mut result = Vec::new();
            for ob in &order_by.exprs {
                let item = match &ob.expr {
                    Expr::Identifier(ident) => OrderByItem::Name(ident.value.to_lowercase()),
                    Expr::Value(Value::Number(n, _)) => {
                        let pos = n.parse::<usize>().map_err(|_| {
                            SqlError::Plan(format!("Invalid positional ORDER BY: {n}"))
                        })?;
                        if pos == 0 {
                            return Err(SqlError::Plan(
                                "ORDER BY position is 1-based, got 0".into(),
                            ));
                        }
                        OrderByItem::Position(pos)
                    }
                    other => OrderByItem::Expression(Box::new(other.clone())),
                };
                let desc = ob.asc.map(|asc| !asc).unwrap_or(false);
                // ob.nulls_first: Some(true)=NULLS FIRST, Some(false)=NULLS LAST, None=default
                result.push((item, desc, ob.nulls_first));
            }
            Ok(result)
        }
    }
}

/// Plan ORDER BY: handles column names, positional refs, and expressions.
/// `table_col_names` are the actual C-level column names in the table (may differ
/// from `result_aliases` when expressions produce internal names like `_e1`).
fn plan_order_by(
    g: &mut Graph,
    input: Column,
    order_by: &[(OrderByItem, bool, Option<bool>)],
    result_aliases: &[String],
    table_col_names: &[String],
) -> Result<Column, SqlError> {
    // Build a schema for expression planning:
    // 1) visible result aliases (take precedence), then
    // 2) physical table column names (includes hidden ORDER BY helpers).
    let schema: HashMap<String, usize> = result_aliases
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .chain(
            table_col_names
                .iter()
                .enumerate()
                .filter(|(_, name)| !result_aliases.contains(name))
                .map(|(i, name)| (name.clone(), i)),
        )
        .collect();

    let mut sort_keys = Vec::new();
    let mut descs = Vec::new();
    let mut has_explicit_nulls = false;
    let mut nulls_first_flags: Vec<bool> = Vec::new();

    for (item, desc, nulls_first) in order_by {
        let key = match item {
            OrderByItem::Name(name) => {
                let idx = result_aliases
                    .iter()
                    .position(|a| a == name)
                    .or_else(|| table_col_names.iter().position(|c| c == name))
                    // Fallback: match aggregate input column name.
                    // e.g. ORDER BY v1 resolves to avg(v1) when the result
                    // alias is "avg(v1)" and the inner arg is "v1".
                    .or_else(|| {
                        let mut matches = result_aliases.iter().enumerate().filter(|(_, a)| {
                            if let Some(start) = a.find('(') {
                                if a.ends_with(')') {
                                    let inner = &a[start + 1..a.len() - 1];
                                    return inner == name.as_str();
                                }
                            }
                            false
                        });
                        let first = matches.next();
                        if first.is_some() && matches.next().is_none() {
                            // Exactly one aggregate matches — unambiguous
                            first.map(|(i, _)| i)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        SqlError::Plan(format!("ORDER BY column '{}' not found", name))
                    })?;
                // Use actual table column name, not the SQL alias
                g.scan(&table_col_names[idx])?
            }
            OrderByItem::Position(pos) => {
                if *pos > result_aliases.len() {
                    return Err(SqlError::Plan(format!(
                        "ORDER BY position {} exceeds column count {}",
                        pos,
                        result_aliases.len()
                    )));
                }
                g.scan(&table_col_names[*pos - 1])?
            }
            OrderByItem::Expression(expr) => plan_expr(g, expr.as_ref(), &schema)?,
        };
        sort_keys.push(key);
        descs.push(*desc);
        if nulls_first.is_some() {
            has_explicit_nulls = true;
        }
        // Default: NULLS LAST for ASC, NULLS FIRST for DESC (PostgreSQL convention)
        nulls_first_flags.push(nulls_first.unwrap_or(*desc));
    }

    let nf = if has_explicit_nulls {
        Some(nulls_first_flags.as_slice())
    } else {
        None // let C-side apply defaults
    };
    Ok(g.sort(input, &sort_keys, &descs, nf)?)
}

/// Extract LIMIT value.
fn extract_limit(query: &Query) -> Result<Option<i64>, SqlError> {
    match &query.limit {
        None => Ok(None),
        Some(expr) => match expr {
            Expr::Value(Value::Number(n, _)) => {
                let limit = n
                    .parse::<i64>()
                    .map_err(|_| SqlError::Plan(format!("Invalid LIMIT value: {n}")))?;
                if limit < 0 {
                    return Err(SqlError::Plan("LIMIT must be non-negative".into()));
                }
                Ok(Some(limit))
            }
            _ => Err(SqlError::Plan("LIMIT must be an integer literal".into())),
        },
    }
}

/// Extract OFFSET value.
fn extract_offset(query: &Query) -> Result<Option<i64>, SqlError> {
    match &query.offset {
        None => Ok(None),
        Some(offset) => match &offset.value {
            Expr::Value(Value::Number(n, _)) => {
                let off = n
                    .parse::<i64>()
                    .map_err(|_| SqlError::Plan(format!("Invalid OFFSET value: {n}")))?;
                if off < 0 {
                    return Err(SqlError::Plan("OFFSET must be non-negative".into()));
                }
                Ok(Some(off))
            }
            _ => Err(SqlError::Plan("OFFSET must be an integer literal".into())),
        },
    }
}
