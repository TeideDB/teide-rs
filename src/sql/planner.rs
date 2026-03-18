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
    AssignmentTarget, BinaryOperator, ColumnDef, DataType, Delete, Distinct, Expr, FromTable,
    FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, Insert, JoinConstraint,
    JoinOperator, ObjectName, ObjectType, Query, SelectItem, SetExpr, Statement, TableFactor,
    TableWithJoins, UnaryOperator, Value, Values,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use crate::{Column, Context, Graph, Table};

use super::expr::{
    agg_op_from_name, collect_aggregates, collect_window_functions, expr_default_name,
    extract_col_name as try_extract_col_name, extract_col_name_qualified, format_agg_name,
    has_window_functions, is_aggregate, is_count_distinct, is_pure_aggregate, parse_window_frame,
    plan_agg_input, plan_expr, plan_having_expr, plan_post_agg_expr, predict_c_agg_name,
};
use super::{pgq, ExecResult, Session, SqlError, SqlResult, StoredTable};

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
            // Try KNN fast path (HNSW-accelerated or brute-force).
            if let Some(result) = try_hnsw_knn(session, &q)? {
                return Ok(ExecResult::Query(result));
            }
            let result = plan_query(&session.ctx, &q, Some(&session.tables), Some(&session.graphs))?;
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
                // Try KNN fast path (HNSW-accelerated or brute-force),
                // then fall back to the general planner.
                let result = if let Some(r) = try_hnsw_knn(session, query)? {
                    r
                } else {
                    plan_query(&session.ctx, query, Some(&session.tables), Some(&session.graphs))?
                };
                let nrows = result.nrows as i64;
                let ncols = result.columns.len();

                let table = result.table.with_column_names(&result.columns)?;
                let old_table = session.tables.insert(
                    table_name.clone(),
                    StoredTable {
                        table,
                        columns: result.columns,
                        embedding_dims: result.embedding_dims,
                    },
                );
                if create.or_replace {
                    if let Err(e) = session.invalidate_graphs_for_table(&table_name) {
                        // Rollback: restore the old table
                        if let Some(old) = old_table {
                            session.tables.insert(table_name, old);
                        } else {
                            session.tables.remove(&table_name);
                        }
                        return Err(e);
                    }
                    // Vector indexes hold raw pointers into the old column data
                    // which is now freed — drop after rollback checks succeed.
                    session.remove_vector_indexes_for_table(&table_name);
                }

                Ok(ExecResult::Ddl(format!(
                    "Created table '{table_name}' ({nrows} rows, {ncols} cols)"
                )))
            } else if !create.columns.is_empty() {
                // CREATE TABLE t (col1 TYPE, col2 TYPE, ...)
                let (table, columns) = create_empty_table(&create.columns)?;
                let ncols = columns.len();
                let old_table = session
                    .tables
                    .insert(table_name.clone(), StoredTable { table, columns, embedding_dims: HashMap::new() });
                if create.or_replace {
                    if let Err(e) = session.invalidate_graphs_for_table(&table_name) {
                        if let Some(old) = old_table {
                            session.tables.insert(table_name, old);
                        } else {
                            session.tables.remove(&table_name);
                        }
                        return Err(e);
                    }
                    session.remove_vector_indexes_for_table(&table_name);
                }

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
                    session.remove_graphs_for_table(&table_name);
                    session.remove_vector_indexes_for_table(&table_name);
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

        Statement::Delete(delete) => plan_delete(session, &delete),

        Statement::Update { table, assignments, selection, .. } => {
            plan_update(session, &table, &assignments, &selection)
        }

        _ => Err(SqlError::Plan(
            "Only SELECT, CREATE TABLE, DROP TABLE, INSERT INTO, DELETE, and UPDATE are supported".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// HNSW-accelerated KNN optimization
// ---------------------------------------------------------------------------

/// Information extracted from a KNN query pattern:
/// `SELECT ... FROM table ORDER BY similarity_func(col, ARRAY[...]) LIMIT k`
struct KnnPattern {
    table_name: String,
    emb_column: String,
    query_vec: Vec<f32>,
    k: i64,
    func_name: String, // "cosine_similarity" or "euclidean_distance"
    #[allow(dead_code)]
    desc: bool, // ORDER BY direction (reserved for future reverse-order support)
}

/// Try to detect a KNN query pattern and execute it directly.
/// Uses an HNSW index when available, otherwise falls back to brute-force KNN.
/// Returns `Some(SqlResult)` if the pattern was matched, `None` otherwise.
fn try_hnsw_knn(session: &Session, query: &Query) -> Result<Option<SqlResult>, SqlError> {
    let pattern = match detect_knn_pattern(query) {
        Some(p) => p,
        None => return Ok(None),
    };

    let stored = session
        .tables
        .get(&pattern.table_name)
        .ok_or_else(|| SqlError::Plan(format!("Table '{}' not found", pattern.table_name)))?;

    // Validate query vector dimension against the embedding column.
    let expected_dim = stored
        .embedding_dims
        .get(&pattern.emb_column)
        .copied()
        .unwrap_or(0);
    if expected_dim > 0 && pattern.query_vec.len() as i32 != expected_dim {
        return Err(SqlError::Plan(format!(
            "Query vector has {} elements but column '{}' has dimension {}",
            pattern.query_vec.len(),
            pattern.emb_column,
            expected_dim
        )));
    }

    // Run search: HNSW-accelerated if an index exists, brute-force otherwise.
    // Both paths produce (row_id, cosine_distance) pairs sorted by distance
    // ascending (= similarity descending).
    let vi = session.find_vector_index(&pattern.table_name, &pattern.emb_column);
    let results: Vec<(i64, f64)> = if let Some(vi) = vi {
        let ef_search = std::cmp::max(50, i32::try_from(pattern.k).unwrap_or(i32::MAX));
        vi.index
            .search(&pattern.query_vec, pattern.k, ef_search)
            .map_err(SqlError::Engine)?
    } else {
        // Brute-force: use td_knn which returns (_rowid, _similarity).
        let source_table = &stored.table;
        let mut g = session.ctx.graph(source_table).map_err(SqlError::Engine)?;
        let emb_col = g.scan(&pattern.emb_column).map_err(SqlError::Engine)?;
        let knn_node = g
            .knn_owned(emb_col, pattern.query_vec.clone(), pattern.k)
            .map_err(SqlError::Engine)?;
        let knn_table = g.execute(knn_node).map_err(SqlError::Engine)?;
        let n = knn_table.nrows() as usize;
        if n == 0 {
            Vec::new()
        } else {
            let rowid_ptr = knn_table
                .get_col_idx(0)
                .ok_or_else(|| SqlError::Plan("KNN result missing _rowid".into()))?;
            let sim_ptr = knn_table
                .get_col_idx(1)
                .ok_or_else(|| SqlError::Plan("KNN result missing _similarity".into()))?;
            let rowids =
                unsafe { std::slice::from_raw_parts(crate::ffi::td_data(rowid_ptr) as *const i64, n) };
            let sims =
                unsafe { std::slice::from_raw_parts(crate::ffi::td_data(sim_ptr) as *const f64, n) };
            // Convert similarity to cosine distance for uniform handling
            // with the HNSW path (which returns distances).
            rowids
                .iter()
                .copied()
                .zip(sims.iter().map(|&s| 1.0 - s))
                .collect()
        }
    };
    let n_results = results.len();

    // Build the result table from the source table + HNSW results.
    // For each SELECT column, extract values at the returned row IDs.
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s,
        _ => return Ok(None),
    };

    let source_table = &stored.table;
    let source_ncols = source_table.ncols() as usize;

    // Determine which columns to include and their names.
    let mut output_cols: Vec<OutputCol> = Vec::new();
    for item in &select.projection {
        match item {
            SelectItem::Wildcard(_) => {
                for c in 0..source_ncols {
                    let name = source_table.col_name_str(c);
                    let dim = stored.embedding_dims.get(&name).copied().unwrap_or(0);
                    if dim > 1 {
                        // Embedding columns cannot be projected correctly by
                        // the HNSW fast path — fall back to brute-force,
                        // consistent with explicit column projection.
                        return Ok(None);
                    }
                    output_cols.push(OutputCol::SourceColumn(c, name.clone(), name));
                }
            }
            SelectItem::UnnamedExpr(expr) => {
                if let Some(func) = extract_similarity_func(expr) {
                    if func.name == pattern.func_name
                        && func.column == pattern.emb_column
                    {
                        // The projected similarity must use the same query
                        // vector as the ORDER BY; otherwise the HNSW search
                        // distances would be reported under the wrong vector.
                        let proj_vec = extract_query_vec_from_func(expr);
                        if proj_vec.as_ref() != Some(&pattern.query_vec) {
                            return Ok(None);
                        }
                        let alias = expr_default_name(expr);
                        output_cols.push(OutputCol::Similarity(alias));
                    } else {
                        return Ok(None); // Unsupported expression
                    }
                } else if let Some(col_name) = try_extract_col_name(expr) {
                    let col_name = col_name.to_lowercase();
                    // Embedding columns cannot be projected correctly by the
                    // HNSW fast path (they are multi-dimensional F32 arrays) —
                    // fall through to the brute-force planner.
                    if stored.embedding_dims.get(&col_name).is_some_and(|&d| d > 1) {
                        return Ok(None);
                    }
                    let col_idx = find_col_index(source_table, &col_name)
                        .ok_or_else(|| SqlError::Plan(format!("Column '{col_name}' not found")))?;
                    output_cols.push(OutputCol::SourceColumn(col_idx, col_name.clone(), col_name));
                } else {
                    return Ok(None); // Complex expression — fall through
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                if let Some(func) = extract_similarity_func(expr) {
                    if func.name == pattern.func_name
                        && func.column == pattern.emb_column
                    {
                        let proj_vec = extract_query_vec_from_func(expr);
                        if proj_vec.as_ref() != Some(&pattern.query_vec) {
                            return Ok(None);
                        }
                        output_cols.push(OutputCol::Similarity(alias.value.to_lowercase()));
                    } else {
                        return Ok(None);
                    }
                } else if let Some(col_name) = try_extract_col_name(expr) {
                    let col_name = col_name.to_lowercase();
                    if stored.embedding_dims.get(&col_name).is_some_and(|&d| d > 1) {
                        return Ok(None);
                    }
                    let col_idx = find_col_index(source_table, &col_name)
                        .ok_or_else(|| SqlError::Plan(format!("Column '{col_name}' not found")))?;
                    output_cols.push(OutputCol::SourceColumn(col_idx, alias.value.to_lowercase(), col_name));
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
    }

    if output_cols.is_empty() {
        return Ok(None);
    }

    // Build the result table using raw FFI.
    let ncols = output_cols.len();
    let mut builder = RawTableBuilder::new(ncols as i64)?;
    let mut col_names = Vec::with_capacity(ncols);

    for output_col in &output_cols {
        match output_col {
            OutputCol::SourceColumn(src_idx, name, _) => {
                let src_col = source_table
                    .get_col_idx(*src_idx as i64)
                    .ok_or_else(|| SqlError::Plan("Column missing".into()))?;
                // Null bitmaps are not preserved by the memcpy gather below;
                // fall back to the brute-force planner for nullable columns.
                let attrs = unsafe { (*src_col).attrs };
                if attrs & crate::ffi::TD_ATTR_HAS_NULLS != 0 {
                    return Ok(None);
                }
                let new_vec = col_vec_new(src_col, n_results as i64);
                if new_vec.is_null() || crate::ffi_is_err(new_vec) {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                // Copy values at HNSW row IDs.
                let elem_size = col_elem_size(src_col);
                let src_nrows =
                    unsafe { crate::ffi::td_len(src_col as *const crate::ffi::td_t) } as i64;
                let src_data = unsafe { crate::ffi::td_data(src_col) as *const u8 };
                let dst_data = unsafe { crate::ffi::td_data(new_vec) as *mut u8 };
                for (out_row, &(row_id, _)) in results.iter().enumerate() {
                    if row_id < 0 || row_id >= src_nrows {
                        unsafe { crate::ffi_release(new_vec) };
                        return Err(SqlError::Plan(format!(
                            "KNN returned row_id {} out of range (table has {} rows)",
                            row_id, src_nrows
                        )));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_data.add(row_id as usize * elem_size),
                            dst_data.add(out_row * elem_size),
                            elem_size,
                        );
                    }
                }
                unsafe { (*new_vec).val.len = n_results as i64 };
                let name_id = match crate::sym_intern(name) {
                    Ok(id) => id,
                    Err(e) => {
                        unsafe { crate::ffi_release(new_vec) };
                        return Err(SqlError::Engine(e));
                    }
                };
                let res = builder.add_col(name_id, new_vec);
                unsafe { crate::ffi_release(new_vec) };
                res?;
                col_names.push(name.clone());
            }
            OutputCol::Similarity(name) => {
                let new_vec =
                    unsafe { crate::raw::td_vec_new(crate::ffi::TD_F64, n_results as i64) };
                if new_vec.is_null() || crate::ffi_is_err(new_vec) {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                let dst = unsafe { crate::ffi::td_data(new_vec) as *mut f64 };
                for (i, &(_, dist)) in results.iter().enumerate() {
                    // HNSW returns cosine distance; convert to similarity
                    let sim = if pattern.func_name == "cosine_similarity" {
                        1.0 - dist
                    } else {
                        dist // euclidean_distance is already a distance
                    };
                    unsafe { *dst.add(i) = sim };
                }
                unsafe { (*new_vec).val.len = n_results as i64 };
                let name_id = match crate::sym_intern(name) {
                    Ok(id) => id,
                    Err(e) => {
                        unsafe { crate::ffi_release(new_vec) };
                        return Err(SqlError::Engine(e));
                    }
                };
                let res = builder.add_col(name_id, new_vec);
                unsafe { crate::ffi_release(new_vec) };
                res?;
                col_names.push(name.clone());
            }
        }
    }

    let table = builder.finish()?;

    // Propagate dim-1 embedding metadata for source columns that pass through.
    // Dim-1 embeddings have layout identical to plain F32 columns (one float
    // per row) so the memcpy gather above handles them correctly, but we must
    // preserve the metadata so that CTAS inherits it into the derived table.
    let mut result_embedding_dims = HashMap::new();
    for output_col in &output_cols {
        if let OutputCol::SourceColumn(_, output_name, src_name) = output_col {
            if let Some(&d) = stored.embedding_dims.get(src_name) {
                result_embedding_dims.insert(output_name.clone(), d);
            }
        }
    }

    // HNSW results are sorted by cosine distance ascending (= similarity
    // descending), which matches the required ORDER BY ... DESC semantics.
    Ok(Some(SqlResult {
        nrows: n_results,
        columns: col_names,
        embedding_dims: result_embedding_dims,
        table,
    }))
}

enum OutputCol {
    /// (column index in source table, output name, original source column name)
    SourceColumn(usize, String, String),
    Similarity(String), // output name for the similarity value
}

struct SimilarityFunc {
    name: String,
    column: String,
}

/// Extract similarity function info from an expression if it is
/// cosine_similarity(col, ARRAY[...]) or euclidean_distance(col, ARRAY[...]).
fn extract_similarity_func(expr: &Expr) -> Option<SimilarityFunc> {
    if let Expr::Function(func) = expr {
        let name = object_name_to_string(&func.name).to_lowercase();
        if name == "cosine_similarity" || name == "euclidean_distance" {
            if let FunctionArguments::List(arg_list) = &func.args {
                if arg_list.args.len() == 2 {
                    // Accept both bare identifier and table-qualified (e.g. d.embedding)
                    let col_name = match &arg_list.args[0] {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Identifier(ident))) => {
                            Some(ident.value.to_lowercase())
                        }
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::CompoundIdentifier(parts))) => {
                            parts.last().map(|p| p.value.to_lowercase())
                        }
                        _ => None,
                    };
                    if let Some(column) = col_name {
                        return Some(SimilarityFunc { name, column });
                    }
                }
            }
        }
    }
    None
}

/// Extract the query vector from a similarity function's second argument
/// (an ARRAY literal).
fn extract_query_vec_from_func(expr: &Expr) -> Option<Vec<f32>> {
    if let Expr::Function(func) = expr {
        if let FunctionArguments::List(arg_list) = &func.args {
            if arg_list.args.len() == 2 {
                if let FunctionArg::Unnamed(FunctionArgExpr::Expr(array_expr)) = &arg_list.args[1]
                {
                    return super::expr::try_parse_array_literal(array_expr);
                }
            }
        }
    }
    None
}

/// Detect a KNN query pattern:
/// `SELECT ... FROM table ORDER BY cosine_similarity(col, ARRAY[...]) DESC LIMIT k`
/// with no WHERE, GROUP BY, HAVING, DISTINCT, JOINs, OFFSET, or subqueries.
///
/// Only matches cosine_similarity (the HNSW index is cosine-only).
/// Only matches DESC ordering (most similar first), which is the natural
/// HNSW output order.  OFFSET is rejected because the HNSW search returns
/// exactly k results and cannot skip rows.
fn detect_knn_pattern(query: &Query) -> Option<KnnPattern> {
    // CTEs may shadow session table names; bail out and let the normal
    // planner handle CTE resolution.
    if query.with.is_some() {
        return None;
    }

    // Must have ORDER BY and LIMIT, no OFFSET.
    let order_by = query.order_by.as_ref()?;
    if order_by.exprs.len() != 1 {
        return None;
    }
    if query.offset.is_some() {
        return None;
    }
    let limit_expr = query.limit.as_ref()?;
    let k = match limit_expr {
        Expr::Value(Value::Number(n, _)) => n.parse::<i64>().ok()?,
        _ => return None,
    };
    if k <= 0 {
        return None;
    }

    // ORDER BY must be a similarity function.
    let ob = &order_by.exprs[0];
    let desc = ob.asc.map(|asc| !asc).unwrap_or(false);
    let sim_func = extract_similarity_func(&ob.expr)?;
    let query_vec = extract_query_vec_from_func(&ob.expr)?;

    // HNSW index is cosine-only — reject euclidean_distance.
    if sim_func.name != "cosine_similarity" {
        return None;
    }

    // Only DESC ordering matches HNSW output (most similar first).
    // ASC would require returning the *least* similar rows, which HNSW
    // cannot do efficiently.
    if !desc {
        return None;
    }

    // Body must be a simple SELECT.
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s,
        _ => return None,
    };

    // No WHERE, GROUP BY, HAVING, DISTINCT.
    if select.selection.is_some() || select.having.is_some() {
        return None;
    }
    if !matches!(&select.group_by, GroupByExpr::Expressions(v, _) if v.is_empty()) {
        return None;
    }
    if select.distinct.is_some() {
        return None;
    }

    // Single table, no JOINs.
    if select.from.len() != 1 {
        return None;
    }
    let from = &select.from[0];
    if !from.joins.is_empty() {
        return None;
    }
    let table_name = match &from.relation {
        TableFactor::Table { name, .. } => object_name_to_string(name).to_lowercase(),
        _ => return None,
    };

    Some(KnnPattern {
        table_name,
        emb_column: sim_func.column,
        query_vec,
        k,
        func_name: sim_func.name,
        desc,
    })
}

/// Find a column index by name in a table.
fn find_col_index(table: &Table, name: &str) -> Option<usize> {
    (0..table.ncols() as usize).find(|&c| table.col_name_str(c).eq_ignore_ascii_case(name))
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
        DataType::Date => Ok(ffi::TD_DATE),
        DataType::Time(_, _) => Ok(ffi::TD_TIME),
        DataType::Timestamp(_, _) => Ok(ffi::TD_TIMESTAMP),
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

    // Embedding columns are flat N*D F32 arrays created via td_embedding_new.
    // VALUES rows produce scalar F32 vectors via td_vec_new, which are not
    // dimension-aware.  Concatenating them would corrupt the embedding buffer.
    // Exception: dim=1 embeddings have layout identical to plain F32 columns,
    // so VALUES insertion is safe for them.
    if matches!(source_query.body.as_ref(), SetExpr::Values(_)) {
        let high_dim: HashMap<String, i32> = stored
            .embedding_dims
            .iter()
            .filter(|(_, &d)| d > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        reject_if_has_embeddings(&high_dim, "INSERT INTO ... VALUES")?;
    }

    // Build source table from VALUES or SELECT.
    // `source_logical_nrows` is the true row count: for embedding tables,
    // `table.nrows()` returns N*D but the logical count is N.
    let (source_table, source_cols, source_emb_dims, source_logical_nrows) =
        match source_query.body.as_ref() {
            SetExpr::Values(values) => {
                let tbl = build_table_from_values(values, &target_types, &target_cols)?;
                let nrows = tbl.nrows() as usize;
                let cols = target_cols.clone();
                (tbl, cols, HashMap::new(), nrows)
            }
            _ => {
                // Treat as a subquery (SELECT ...).
                // Try KNN fast path first, then fall back to the general planner.
                let result = if let Some(r) = try_hnsw_knn(session, source_query)? {
                    r
                } else {
                    plan_query(&session.ctx, source_query, Some(&session.tables), Some(&session.graphs))?
                };
                let nrows = result.nrows;
                (result.table, result.columns, result.embedding_dims, nrows)
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
            &stored.embedding_dims,
            source_logical_nrows,
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

    // Validate embedding dimension compatibility: if the target table has
    // registered embedding dims and the source also carries dim metadata for
    // the same column, the dimensions must match.  Mismatched dimensions
    // would silently corrupt the flat F32 buffer after concat.
    //
    // Remap source embedding dims to target column names using positional
    // mapping so that aliased columns (e.g. INSERT INTO t(emb) SELECT src_emb
    // FROM ...) are validated correctly.
    let remapped_source_emb: HashMap<String, i32> = if !insert.columns.is_empty() {
        // Explicit column list: insert_cols[i] maps source col i → target col name
        let mut map = HashMap::new();
        for (src_idx, ident) in insert.columns.iter().enumerate() {
            let target_name = ident.value.to_lowercase();
            if let Some(src_name) = source_cols.get(src_idx) {
                if let Some(&dim) = source_emb_dims.get(&src_name.to_lowercase()) {
                    map.insert(target_name, dim);
                }
            }
        }
        map
    } else {
        // Positional insert: source col i maps to target col i
        let mut map = HashMap::new();
        for (i, src_name) in source_cols.iter().enumerate() {
            if let Some(&dim) = source_emb_dims.get(&src_name.to_lowercase()) {
                if let Some(tgt_name) = target_cols.get(i) {
                    map.insert(tgt_name.to_lowercase(), dim);
                }
            }
        }
        map
    };
    for (col_name, &target_dim) in &stored.embedding_dims {
        match remapped_source_emb.get(col_name) {
            Some(&source_dim) => {
                if source_dim != target_dim {
                    return Err(SqlError::Plan(format!(
                        "INSERT INTO: embedding column '{col_name}' dimension mismatch \
                         (source has {source_dim}, target has {target_dim})"
                    )));
                }
            }
            None => {
                // Source has no embedding metadata for this target embedding
                // column.  For dim > 1 this means the source column is a plain
                // F32 vector whose flat layout would corrupt the N*D buffer.
                // For dim = 1 the storage layout is identical to a plain F32
                // column (one float per row), so we allow it.
                if target_dim != 1 {
                    return Err(SqlError::Plan(format!(
                        "INSERT INTO: target column '{col_name}' is an embedding (dim={target_dim}) \
                         but the source does not provide matching embedding metadata"
                    )));
                }
            }
        }
    }

    // Also check the reverse: source has a dim > 1 embedding but the target
    // column is plain FLOAT.  The raw N*D buffer would be concatenated into a
    // scalar column, changing row semantics instead of erroring.
    for (src_col, &src_dim) in &remapped_source_emb {
        if src_dim > 1 && !stored.embedding_dims.contains_key(src_col) {
            return Err(SqlError::Plan(format!(
                "INSERT INTO: source column '{src_col}' is an embedding (dim={src_dim}) \
                 but the target column is a plain FLOAT — this would corrupt row semantics"
            )));
        }
    }

    // Concatenate with existing table
    let existing = &stored.table;
    let merged = concat_tables(&session.ctx, existing, &source_table)?;

    // Rename columns to match target schema
    let merged = merged.with_column_names(&target_cols)?;

    let prev_embedding_dims = stored.embedding_dims.clone();
    let old_table = session.tables.insert(
        table_name.clone(),
        StoredTable {
            table: merged,
            columns: target_cols,
            embedding_dims: prev_embedding_dims,
        },
    );

    if let Err(e) = session.invalidate_graphs_for_table(&table_name) {
        // Rollback: restore the previous table state (and keep vector indexes
        // intact since old_table still holds the data they reference).
        if let Some(old) = old_table {
            session.tables.insert(table_name, old);
        }
        return Err(e);
    }

    // Vector indexes hold raw pointers into the old column data which is now
    // freed, so they must be dropped after rollback checks succeed.
    session.remove_vector_indexes_for_table(&table_name);

    let nrows = source_logical_nrows;
    Ok(ExecResult::Ddl(format!(
        "Inserted {nrows} rows into '{table_name}'"
    )))
}

// ---------------------------------------------------------------------------
// DELETE FROM
// ---------------------------------------------------------------------------

fn plan_delete(session: &mut Session, delete: &Delete) -> Result<ExecResult, SqlError> {
    // Extract table name from FROM clause
    let tables = match &delete.from {
        FromTable::WithFromKeyword(t) | FromTable::WithoutKeyword(t) => t,
    };

    if tables.len() != 1 {
        return Err(SqlError::Plan("DELETE supports exactly one table".into()));
    }

    let table_name = match &tables[0].relation {
        TableFactor::Table { name, .. } => object_name_to_string(name).to_lowercase(),
        _ => return Err(SqlError::Plan("DELETE FROM requires a table name".into())),
    };

    let stored = session
        .tables
        .get(&table_name)
        .ok_or_else(|| SqlError::Plan(format!("Table '{table_name}' not found")))?;

    let columns = stored.columns.clone();
    let old_nrows = stored.logical_nrows();

    // Embedding columns are flat N*D F32 arrays.  The C engine's filter
    // kernel operates element-wise and is not dimension-aware, so filtering
    // a table that contains embedding columns would corrupt their data.
    // Exception: dim=1 embeddings have layout identical to plain F32 columns.
    if delete.selection.is_some() {
        let high_dim: HashMap<String, i32> = stored.embedding_dims
            .iter()
            .filter(|(_, &d)| d > 1)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        reject_if_has_embeddings(&high_dim, "DELETE with WHERE")?;
    }

    let new_table = if let Some(selection) = &delete.selection {
        // DELETE WHERE pred → keep rows where NOT pred
        let schema: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut g = session.ctx.graph(&stored.table)?;
        g.column_embedding_dims = stored.embedding_dims.clone();
        let table_node = g.const_table(&stored.table)?;
        let pred = plan_expr(&mut g, selection, &schema)?;
        let not_pred = g.not(pred)?;
        let filtered = g.filter(table_node, not_pred)?;
        g.execute(filtered)?
    } else {
        // DELETE without WHERE → delete all rows → build empty table with same schema
        build_empty_table(&stored.table, &columns)?
    };

    let new_nrows = new_table.nrows();
    let deleted = old_nrows - new_nrows;

    let new_table = new_table.with_column_names(&columns)?;

    // Preserve embedding dims on both paths.  The truncated table has
    // zero-length F32 columns whose layout is compatible with later concat
    // from INSERT … SELECT.  Keeping dims ensures reject_if_has_embeddings
    // still blocks INSERT … VALUES on the empty table.
    let prev_embedding_dims = stored.embedding_dims.clone();
    let old_table = session.tables.insert(
        table_name.clone(),
        StoredTable {
            table: new_table,
            columns,
            embedding_dims: prev_embedding_dims,
        },
    );

    if let Err(e) = session.invalidate_graphs_for_table(&table_name) {
        if let Some(old) = old_table {
            session.tables.insert(table_name, old);
        }
        return Err(e);
    }

    // Vector indexes hold raw pointers into the old column data which is now
    // freed, so they must be dropped after rollback checks succeed.
    session.remove_vector_indexes_for_table(&table_name);

    Ok(ExecResult::Ddl(format!(
        "Deleted {deleted} rows from '{table_name}'"
    )))
}

/// Build an empty table with the same schema (column types) as the source table.
fn build_empty_table(source: &Table, columns: &[String]) -> Result<Table, SqlError> {
    let ncols = source.ncols();
    let mut builder = RawTableBuilder::new(ncols)?;

    for (i, name) in columns.iter().enumerate() {
        let typ = source.col_type(i);
        let name_id = crate::sym_intern(name)?;
        let vec = unsafe { crate::raw::td_vec_new(typ, 0) };
        if vec.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        if crate::ffi_is_err(vec) {
            return Err(engine_err_from_raw(vec));
        }

        let res = builder.add_col(name_id, vec);
        unsafe { crate::ffi_release(vec) };
        res?;
    }

    let table = builder.finish()?;
    Ok(table)
}

fn plan_update(
    session: &mut Session,
    table: &TableWithJoins,
    assignments: &[sqlparser::ast::Assignment],
    selection: &Option<Expr>,
) -> Result<ExecResult, SqlError> {
    let table_name = match &table.relation {
        TableFactor::Table { name, .. } => object_name_to_string(name).to_lowercase(),
        _ => return Err(SqlError::Plan("UPDATE: expected table name".into())),
    };

    let stored = session
        .tables
        .get(&table_name)
        .ok_or_else(|| SqlError::Plan(format!("Table '{table_name}' not found")))?;

    let original_nrows = stored.table.nrows();
    let columns = stored.columns.clone();
    let prev_embedding_dims = stored.embedding_dims.clone();

    // Embedding columns are flat N*D F32 arrays.  The C engine's
    // if_then_else / select kernels operate element-wise and are not
    // dimension-aware, so UPDATE on a table with embedding columns would
    // corrupt their data.
    // Exception: dim=1 embeddings have layout identical to plain F32 columns.
    let high_dim_update: HashMap<String, i32> = stored.embedding_dims
        .iter()
        .filter(|(_, &d)| d > 1)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    reject_if_has_embeddings(&high_dim_update, "UPDATE")?;

    let schema: HashMap<String, usize> = columns
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    // Parse and validate SET assignments up front (only needs string keys + references to AST)
    let mut set_cols: Vec<(String, &Expr)> = Vec::new();
    for assignment in assignments {
        let col_name = match &assignment.target {
            AssignmentTarget::ColumnName(name) => object_name_to_string(name).to_lowercase(),
            AssignmentTarget::Tuple(_) => {
                return Err(SqlError::Plan("UPDATE: tuple assignments not supported".into()));
            }
        };
        if !schema.contains_key(&col_name) {
            return Err(SqlError::Plan(format!(
                "UPDATE: column '{col_name}' not found in table '{table_name}'"
            )));
        }
        set_cols.push((col_name, &assignment.value));
    }
    let set_map: HashMap<&str, &Expr> = set_cols.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    // Count matching rows first (separate graph scope so borrow of stored is released)
    let updated_count = match selection {
        Some(where_expr) => {
            let count = {
                let stored = session.tables.get(&table_name).unwrap();
                let mut g2 = session.ctx.graph(&stored.table)?;
                g2.column_embedding_dims = stored.embedding_dims.clone();
                let t2 = g2.const_table(&stored.table)?;
                let pred2 = plan_expr(&mut g2, where_expr, &schema)?;
                let filtered = g2.filter(t2, pred2)?;
                let count_table = g2.execute(filtered)?;
                count_table.nrows()
            };
            count
        }
        None => original_nrows,
    };

    // Build the update graph (scoped so `g` is dropped before we mutate session.tables)
    let result = {
        let stored = session.tables.get(&table_name).unwrap();
        let mut g = session.ctx.graph(&stored.table)?;
        g.column_embedding_dims = stored.embedding_dims.clone();
        let mut out_cols: Vec<Column> = Vec::with_capacity(columns.len());

        // Evaluate the WHERE predicate once (Column is Copy, so we reuse it).
        // For no-WHERE, build a boolean vector of all-true values because the
        // C engine requires vector predicates for if_then_else.
        let pred = match selection {
            Some(where_expr) => plan_expr(&mut g, where_expr, &schema)?,
            None => {
                let nrows = stored.table.nrows();
                assert!(nrows >= 0, "table nrows must be non-negative");
                let vec = unsafe { crate::raw::td_vec_new(crate::ffi::TD_BOOL, nrows) };
                if vec.is_null() || crate::ffi_is_err(vec) {
                    return Err(SqlError::Plan("Failed to allocate boolean vector".into()));
                }
                unsafe {
                    let data = crate::ffi::td_data(vec) as *mut u8;
                    std::ptr::write_bytes(data, 1u8, nrows as usize);
                    (*vec).val.len = nrows;
                }
                let col = unsafe { g.const_vec(vec)? };
                unsafe { crate::ffi_release(vec) };
                col
            }
        };

        for col_name in &columns {
            let old_col = g.scan(col_name)?;
            if let Some(new_expr) = set_map.get(col_name.as_str()) {
                let new_col = plan_expr(&mut g, new_expr, &schema)?;
                let conditional = g.if_then_else(pred, new_col, old_col)?;
                out_cols.push(conditional);
            } else {
                out_cols.push(old_col);
            }
        }

        let table_node = g.const_table(&stored.table)?;
        let projected = g.select(table_node, &out_cols)?;
        g.execute(projected)?
    };

    let result = result.with_column_names(&columns)?;

    let old_table = session.tables.insert(
        table_name.clone(),
        StoredTable {
            table: result,
            columns,
            embedding_dims: prev_embedding_dims,
        },
    );

    if let Err(e) = session.invalidate_graphs_for_table(&table_name) {
        if let Some(old) = old_table {
            session.tables.insert(table_name, old);
        }
        return Err(e);
    }

    // Vector indexes hold raw pointers into the old column data which is now
    // freed, so they must be dropped after rollback checks succeed.
    session.remove_vector_indexes_for_table(&table_name);

    Ok(ExecResult::Ddl(format!(
        "Updated {updated_count} rows in '{table_name}'"
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

        ffi::TD_DATE | ffi::TD_TIME => {
            // Try integer literal first, then date/time string literal
            let v32 = if let Ok(val) = eval_i64_literal(expr) {
                val as i32
            } else if let Ok(s) = eval_str_literal(expr) {
                if typ == ffi::TD_DATE {
                    parse_date_str(&s)
                        .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
                } else {
                    parse_time_str(&s)
                        .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
                }
            } else {
                return Err(SqlError::Plan(format!(
                    "column '{}': expected integer or date/time string literal, got {expr}",
                    col_names[col_idx]
                )));
            };
            let next =
                unsafe { ffi::td_vec_append(vec, &v32 as *const i32 as *const c_void) };
            check_vec_append(next)
        }

        ffi::TD_TIMESTAMP => {
            let val = if let Ok(v) = eval_i64_literal(expr) {
                v
            } else if let Ok(s) = eval_str_literal(expr) {
                parse_timestamp_str(&s)
                    .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?
            } else {
                return Err(SqlError::Plan(format!(
                    "column '{}': expected integer or timestamp string literal, got {expr}",
                    col_names[col_idx]
                )));
            };
            let next =
                unsafe { ffi::td_vec_append(vec, &val as *const i64 as *const c_void) };
            check_vec_append(next)
        }

        ffi::TD_I16 => {
            let val = eval_i64_literal(expr)
                .map_err(|e| SqlError::Plan(format!("column '{}': {e}", col_names[col_idx])))?;
            let v16 = val as i16;
            let next =
                unsafe { ffi::td_vec_append(vec, &v16 as *const i16 as *const c_void) };
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

/// Parse a date string "YYYY-MM-DD" to days since 2000-01-01.
/// Uses the Hinnant civil_from_days algorithm (inverse of Table::format_date).
pub(crate) fn parse_date_str(s: &str) -> Result<i32, String> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("invalid date literal '{s}', expected YYYY-MM-DD"));
    }
    let y: i64 = parts[0]
        .parse()
        .map_err(|_| format!("invalid year in date '{s}'"))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid month in date '{s}'"))?;
    let d: u32 = parts[2]
        .parse()
        .map_err(|_| format!("invalid day in date '{s}'"))?;
    if m < 1 || m > 12 || d < 1 || d > 31 {
        return Err(format!("date out of range: '{s}'"));
    }
    // Per-month day validation (accounts for leap years)
    let max_day = match m {
        2 => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > max_day {
        return Err(format!("date out of range: '{s}'"));
    }
    // Hinnant days_from_civil: compute days since 1970-01-01, then adjust to 2000-01-01
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (m as u64) + 2) / 5 + (d as u64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146097 + doe as i64 - 719468; // days since 1970-01-01
    Ok((days_since_epoch - 10957) as i32) // adjust to 2000-01-01 epoch
}

/// Parse a time string "HH:MM:SS" or "HH:MM:SS.mmm" to milliseconds since midnight.
pub(crate) fn parse_time_str(s: &str) -> Result<i32, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("invalid time literal '{s}', expected HH:MM:SS"));
    }
    let h: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid hour in time '{s}'"))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid minute in time '{s}'"))?;
    // seconds may have fractional part
    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let s_val: u32 = sec_parts[0]
        .parse()
        .map_err(|e| format!("invalid second in time '{s}': {e}"))?;
    let ms = if sec_parts.len() > 1 {
        let frac = sec_parts[1];
        // Pad or truncate to 3 digits for milliseconds
        let padded = format!("{:0<3}", frac);
        padded[..3]
            .parse::<u32>()
            .map_err(|e| format!("invalid fractional seconds in time '{s}': {e}"))?
    } else {
        0
    };
    if h > 23 || m > 59 || s_val > 59 {
        return Err(format!("time out of range: '{s}'"));
    }
    Ok((h * 3_600_000 + m * 60_000 + s_val * 1_000 + ms) as i32)
}

/// Parse a timestamp string "YYYY-MM-DD HH:MM:SS[.ffffff]" to microseconds since 2000-01-01.
pub(crate) fn parse_timestamp_str(s: &str) -> Result<i64, String> {
    // Split on space or 'T' separator
    let sep_pos = s.find(' ').or_else(|| s.find('T'));
    let (date_part, time_part) = match sep_pos {
        Some(pos) => (&s[..pos], &s[pos + 1..]),
        None => return Err(format!("invalid timestamp '{s}', expected YYYY-MM-DD HH:MM:SS")),
    };
    let days = parse_date_str(date_part)?;
    // Parse time with microsecond precision
    let time_parts: Vec<&str> = time_part.split(':').collect();
    if time_parts.len() != 3 {
        return Err(format!("invalid time in timestamp '{s}'"));
    }
    let h: i64 = time_parts[0]
        .parse()
        .map_err(|_| format!("invalid hour in timestamp '{s}'"))?;
    let m: i64 = time_parts[1]
        .parse()
        .map_err(|_| format!("invalid minute in timestamp '{s}'"))?;
    let sec_parts: Vec<&str> = time_parts[2].split('.').collect();
    let secs: i64 = sec_parts[0]
        .parse()
        .map_err(|_| format!("invalid second in timestamp '{s}'"))?;
    let us = if sec_parts.len() > 1 {
        let frac = sec_parts[1];
        let padded = format!("{:0<6}", frac);
        padded[..6]
            .parse::<i64>()
            .map_err(|e| format!("invalid fractional seconds in timestamp '{s}': {e}"))?
    } else {
        0
    };
    if h > 23 || m > 59 || secs > 59 {
        return Err(format!("time out of range in timestamp: '{s}'"));
    }
    let day_us = days as i64 * 86_400_000_000;
    let time_us = h * 3_600_000_000 + m * 60_000_000 + secs * 1_000_000 + us;
    Ok(day_us + time_us)
}

/// Reorder source columns to match target schema when INSERT specifies an explicit column list.
fn reorder_insert_columns(
    insert_cols: &[Ident],
    target_cols: &[String],
    target_types: &[i8],
    source: &Table,
    source_cols: &[String],
    embedding_dims: &HashMap<String, i32>,
    logical_nrows: usize,
) -> Result<Table, SqlError> {
    let _ = source_cols;
    let ncols = target_cols.len();
    let nrows = logical_nrows;

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
            // Embedding columns cannot be default-filled: td_vec_new allocates
            // nrows elements, but an embedding column needs nrows * dim floats.
            if let Some(&dim) = embedding_dims.get(&target_cols[tgt_idx]) {
                if dim > 1 {
                    return Err(SqlError::Plan(format!(
                        "INSERT INTO: embedding column '{}' (dim={}) cannot be omitted; \
                         provide it explicitly in the column list",
                        target_cols[tgt_idx], dim
                    )));
                }
            }
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

    plan_query(ctx, &query, tables, None)
}

// ---------------------------------------------------------------------------
// Query planning
// ---------------------------------------------------------------------------

fn plan_query(
    ctx: &Context,
    query: &Query,
    tables: Option<&HashMap<String, StoredTable>>,
    graphs: Option<&HashMap<String, pgq::PropertyGraph>>,
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
            let result = plan_query(ctx, &cte.query, Some(&cte_map), graphs)?;
            // Rename table columns to match SQL aliases so downstream scans work
            let table = result.table.with_column_names(&result.columns)?;
            cte_map.insert(
                cte_name,
                StoredTable {
                    table,
                    columns: result.columns,
                    embedding_dims: result.embedding_dims,
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
            let left_result = plan_query(ctx, &left_query, effective_tables, graphs)?;
            let right_result = plan_query(ctx, &right_query, effective_tables, graphs)?;

            if left_result.columns.len() != right_result.columns.len() {
                return Err(SqlError::Plan(format!(
                    "UNION: column count mismatch ({} vs {})",
                    left_result.columns.len(),
                    right_result.columns.len()
                )));
            }

            // Concatenate tables column by column
            let result = concat_tables(ctx, &left_result.table, &right_result.table)?;

            // Intersect embedding dims positionally: set operations are
            // positional, so match left/right columns by index, not name.
            // Error on dimension mismatch to prevent silent data corruption
            // when concatenating flat F32 embedding buffers.
            let merged_dims = intersect_embedding_dims_positional(
                &left_result.columns,
                &left_result.embedding_dims,
                &right_result.columns,
                &right_result.embedding_dims,
                "UNION",
            )?;

            // Without ALL: apply DISTINCT — reject if high-dim embeddings present
            // (the C distinct kernel is not dimension-aware).
            // dim=1 embeddings have scalar layout and are safe for DISTINCT.
            if !is_all {
                let high_dim: HashMap<String, i32> = merged_dims
                    .iter()
                    .filter(|(_, &d)| d > 1)
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                reject_if_has_embeddings(&high_dim, "UNION (implicit DISTINCT)")?;
            }
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
                merged_dims,
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
            let left_result = plan_query(ctx, &left_query, effective_tables, graphs)?;
            let right_result = plan_query(ctx, &right_query, effective_tables, graphs)?;

            if left_result.columns.len() != right_result.columns.len() {
                return Err(SqlError::Plan(format!(
                    "{:?}: column count mismatch ({} vs {})",
                    op,
                    left_result.columns.len(),
                    right_result.columns.len()
                )));
            }

            let keep_matches = matches!(op, sqlparser::ast::SetOperator::Intersect);

            let merged_dims = intersect_embedding_dims_positional(
                &left_result.columns,
                &left_result.embedding_dims,
                &right_result.columns,
                &right_result.embedding_dims,
                if keep_matches { "INTERSECT" } else { "EXCEPT" },
            )?;
            let result =
                exec_set_operation(ctx, &left_result.table, &right_result.table, keep_matches, &merged_dims)?;

            // Without ALL: apply DISTINCT — reject if high-dim embeddings present
            // (the C distinct kernel is not dimension-aware).
            // dim=1 embeddings have scalar layout and are safe for DISTINCT.
            if !is_all {
                let high_dim: HashMap<String, i32> = merged_dims
                    .iter()
                    .filter(|(_, &d)| d > 1)
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                let op_name = if keep_matches { "INTERSECT" } else { "EXCEPT" };
                reject_if_has_embeddings(&high_dim, &format!("{op_name} (implicit DISTINCT)"))?;
            }
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
                merged_dims,
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
    let (table, schema, from_embedding_dims, effective_where): (Table, HashMap<String, usize>, HashMap<String, i32>, Option<Expr>) = if select
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
                    let result = plan_query(ctx, &modified, effective_tables, graphs)?;
                    let tbl = result.table.with_column_names(&result.columns)?;
                    let sch = build_result_schema(&tbl, &result.columns);
                    (tbl, sch, result.embedding_dims, join_conjunction(keep))
                } else {
                    let (tbl, sch, emb) = resolve_from(ctx, &select.from, effective_tables, graphs)?;
                    (tbl, sch, emb, select.selection.clone())
                }
            } else {
                let (tbl, sch, emb) = resolve_from(ctx, &select.from, effective_tables, graphs)?;
                (tbl, sch, emb, select.selection.clone())
            }
        } else {
            let (tbl, sch, emb) = resolve_from(ctx, &select.from, effective_tables, graphs)?;
            (tbl, sch, emb, select.selection.clone())
        }
    } else {
        let (tbl, sch, emb) = resolve_from(ctx, &select.from, effective_tables, graphs)?;
        (tbl, sch, emb, select.selection.clone())
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

    // Embedding columns are flat N*D F32 arrays.  The C engine's filter,
    // sort, head, group-by, and distinct kernels operate element-wise and
    // are not dimension-aware, so filtering, ordering, limiting, grouping,
    // or deduplicating a table that contains embedding columns would corrupt
    // their data.
    // Exception: dim=1 embeddings have layout identical to plain F32 columns
    // (one float per row), so they are safe for these operations.
    let high_dim_embeddings: HashMap<String, i32> = from_embedding_dims
        .iter()
        .filter(|(_, &d)| d > 1)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    if effective_where.is_some() {
        reject_if_has_embeddings(&high_dim_embeddings, "SELECT with WHERE")?;
    }
    // ORDER BY and LIMIT/OFFSET are safe when the result table being sorted
    // contains no high-dim embedding columns.  The sort/head kernels operate
    // element-wise, which is fine for scalar columns.  We check both the
    // SELECT output and hidden ORDER BY columns for embedding references.
    let sort_safe_for_embeddings = !high_dim_embeddings.is_empty()
        && !select_output_has_embeddings(select_items, &high_dim_embeddings)
        && !order_by_has_embedding_ref(&order_by_exprs, &high_dim_embeddings);
    if !order_by_exprs.is_empty() && !sort_safe_for_embeddings {
        reject_if_has_embeddings(&high_dim_embeddings, "SELECT with ORDER BY")?;
    }
    if (limit_val.is_some() || offset_val.is_some()) && !sort_safe_for_embeddings {
        reject_if_has_embeddings(&high_dim_embeddings, "SELECT with LIMIT/OFFSET")?;
    }
    if has_group_by || has_aggregates {
        reject_if_has_embeddings(&high_dim_embeddings, "SELECT with GROUP BY/aggregation")?;
    }
    if is_distinct {
        reject_if_has_embeddings(&high_dim_embeddings, "SELECT DISTINCT")?;
    }

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
                g.column_embedding_dims = from_embedding_dims.clone();
                let table_node = g.const_table(&table)?;
                let pred = plan_expr(&mut g, &resolved, &schema)?;
                let filtered = g.filter(table_node, pred)?;
                if can_fuse_where_limit {
                    // Fuse LIMIT into WHERE: HEAD(FILTER(table, pred), n)
                    let total = match (offset_val, limit_val) {
                        (Some(off), Some(lim)) => off.checked_add(lim)
                            .ok_or_else(|| SqlError::Plan("OFFSET + LIMIT overflow".into()))?,
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
        let (wt, ws, wi) = plan_window_stage(ctx, &working_table, select_items, &schema, &from_embedding_dims)?;
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
                (Some(off), Some(lim)) => Some(off.checked_add(lim)
                    .ok_or_else(|| SqlError::Plan("OFFSET + LIMIT overflow".into()))?),
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
            &from_embedding_dims.clone(),
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
        // Reject SELECT * without a real FROM clause (only __teide_const_dummy__ in schema).
        let is_constant_select = schema.len() == 1 && schema.contains_key("__teide_const_dummy__");
        if is_constant_select
            && select_items.len() == 1
            && matches!(select_items[0], SelectItem::Wildcard(_))
        {
            return Err(SqlError::Plan(
                "SELECT * requires a FROM clause".into(),
            ));
        }
        // Skip projection only for true identity projections (`SELECT *` or
        // selecting all base columns in table order). This prevents silently
        // returning wrong columns for reordered/missing identifiers.
        let can_passthrough = !has_windows && is_identity_projection(select_items, &schema);
        if can_passthrough {
            (working_table, aliases)
        } else {
            let emb_dims = from_embedding_dims.clone();
            plan_expr_select(
                ctx,
                &working_table,
                select_items,
                &schema,
                &hidden_order_cols,
                &emb_dims,
            )?
        }
    };

    let (result_table, limit_fused) = if !order_by_exprs.is_empty() {
        let table_col_names: Vec<String> = (0..result_table.ncols() as usize)
            .map(|i| result_table.col_name_str(i).to_string())
            .collect();
        let mut g = ctx.graph(&result_table)?;
        g.column_embedding_dims = from_embedding_dims.clone();
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
                let total = off.checked_add(lim)
                    .ok_or_else(|| SqlError::Plan("OFFSET + LIMIT overflow".into()))?;
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

    // Propagate embedding dimensions for columns that appear in the result.
    // This allows CTEs and derived tables to inherit dimension metadata.
    // Only propagate for columns that are direct passthroughs of source
    // embedding columns — computed expressions (e.g. `SELECT 0.0 AS embedding`)
    // must NOT inherit embedding metadata even if the alias matches.
    let source_dims = from_embedding_dims.clone();
    // Map result alias -> source column name for all direct column references.
    // Uses qualified names (e.g. "t.col") when available so that JOIN-poisoned
    // bare names can still be resolved via their qualified entries.
    let alias_to_source: HashMap<String, String> = select_items
        .iter()
        .filter_map(|item| match item {
            SelectItem::ExprWithAlias { expr, alias } => {
                extract_col_name_qualified(expr)
                    .map(|src| (alias.value.to_lowercase(), src))
            }
            SelectItem::UnnamedExpr(expr) => {
                // Result alias is always the bare name; source may be qualified.
                let bare = try_extract_col_name(expr);
                let qualified = extract_col_name_qualified(expr);
                match (bare, qualified) {
                    (Some(b), Some(q)) => Some((b, q)),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    let has_wildcard = select_items.iter().any(|item| {
        matches!(
            item,
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..)
        )
    });
    // Collect aliases that are explicitly bound to non-column-reference
    // expressions (e.g. `0.0 AS embedding`). These must NOT inherit source
    // embedding metadata even when a wildcard is present.
    let computed_aliases: HashSet<String> = select_items
        .iter()
        .filter_map(|item| match item {
            SelectItem::ExprWithAlias { expr, alias } if try_extract_col_name(expr).is_none() => {
                Some(alias.value.to_lowercase())
            }
            _ => None,
        })
        .collect();
    let result_embedding_dims: HashMap<String, i32> = result_aliases
        .iter()
        .filter_map(|col| {
            if let Some(src) = alias_to_source.get(col) {
                // Try exact (possibly qualified) match first, then strip the
                // qualifier and retry with the bare column name.  This handles
                // both JOINs (qualified entry survives poisoning) and non-JOINs
                // (only bare entries exist in source_dims).
                source_dims
                    .get(src)
                    .or_else(|| {
                        src.rsplit('.')
                            .next()
                            .and_then(|bare| source_dims.get(bare))
                    })
                    .map(|&dim| (col.clone(), dim))
            } else if has_wildcard && !computed_aliases.contains(col) {
                // Wildcard-expanded columns keep their source embedding dims,
                // but only if not shadowed by a computed expression alias.
                source_dims.get(col).map(|&dim| (col.clone(), dim))
            } else {
                None
            }
        })
        .collect();
    // Validate using the filtered result embedding dims so that computed aliases
    // (e.g. `SELECT 0.0 AS embedding`) are not mistakenly treated as embeddings.
    let nrows = validate_result_table(&result_table, &result_embedding_dims, &result_aliases)?;
    Ok(SqlResult {
        table: result_table,
        columns: result_aliases,
        embedding_dims: result_embedding_dims,
        nrows: nrows as usize,
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
    graphs: Option<&HashMap<String, pgq::PropertyGraph>>,
    tables: Option<&HashMap<String, StoredTable>>,
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
        "pagerank" | "connected_component" | "louvain" | "clustering_coefficient" => {
            if args.is_empty() || args.len() > 2 {
                return Err(SqlError::Plan(format!(
                    "{name}() requires 1-2 arguments: {name}('graph_name' [, 'edge_label'])"
                )));
            }
            let graph_name = extract_string_arg(&args[0])?.to_lowercase();
            let graphs = graphs.ok_or_else(|| {
                SqlError::Plan(format!("{name}(): no property graphs available"))
            })?;
            let graph = graphs.get(&graph_name).ok_or_else(|| {
                SqlError::Plan(format!("{name}(): property graph '{graph_name}' not found"))
            })?;
            // Determine edge label: explicit arg or sole label
            let edge_label_name = if args.len() == 2 {
                extract_string_arg(&args[1])?
            } else if graph.edge_labels.len() == 1 {
                graph.edge_labels.keys().next().unwrap().clone()
            } else {
                return Err(SqlError::Plan(format!(
                    "{name}(): graph '{graph_name}' has {} edge labels, specify one as second argument",
                    graph.edge_labels.len()
                )));
            };
            let tables = tables.ok_or_else(|| {
                SqlError::Plan(format!("{name}(): no tables available"))
            })?;
            let result = pgq::execute_standalone_algorithm(
                ctx, tables, graph, name, &edge_label_name,
            )?;
            Ok(result)
        }
        _ => Err(SqlError::Plan(format!(
            "Unknown table function '{name}'. Supported: read_csv(), read_splayed(), read_parted(), pagerank(), connected_component(), louvain(), clustering_coefficient()"
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
            let (l_bare, l_qual) = extract_col_name(left)?;
            let (r_bare, r_qual) = extract_col_name(right)?;

            // Determine which side belongs to which table, trying qualified
            // names first so that `b.id` resolves in schemas where the bare
            // `id` has been renamed to `b.id` by a prior chained join.
            let l_in_left = resolve_col_in_schema(&l_bare, l_qual.as_deref(), left_schema);
            let r_in_right = resolve_col_in_schema(&r_bare, r_qual.as_deref(), right_schema);
            let r_in_left = resolve_col_in_schema(&r_bare, r_qual.as_deref(), left_schema);
            let l_in_right = resolve_col_in_schema(&l_bare, l_qual.as_deref(), right_schema);

            if let (Some(lk), Some(rk)) = (l_in_left, r_in_right) {
                Ok(vec![(lk, rk)])
            } else if let (Some(rk), Some(lk)) = (r_in_left, l_in_right) {
                Ok(vec![(rk, lk)])
            } else {
                let l_display = l_qual.as_deref().unwrap_or(&l_bare);
                let r_display = r_qual.as_deref().unwrap_or(&r_bare);
                Err(SqlError::Plan(format!(
                    "JOIN ON columns '{l_display}' and '{r_display}' not found in respective tables"
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
/// Returns `(bare_name, Option<qualified_name>)`.  For `b.id` the result is
/// `("id", Some("b.id"))`.  For plain `id` the result is `("id", None)`.
fn extract_col_name(expr: &Expr) -> Result<(String, Option<String>), SqlError> {
    match expr {
        Expr::Identifier(ident) => Ok((ident.value.to_lowercase(), None)),
        Expr::CompoundIdentifier(parts) => {
            if parts.len() == 2 {
                let bare = parts[1].value.to_lowercase();
                let qualified = format!("{}.{}", parts[0].value.to_lowercase(), bare);
                Ok((bare, Some(qualified)))
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

/// Look up a column in a schema, trying the qualified form first (`table.col`),
/// then falling back to the bare name (`col`).  Returns the key that matched.
fn resolve_col_in_schema(
    bare: &str,
    qualified: Option<&str>,
    schema: &HashMap<String, usize>,
) -> Option<String> {
    if let Some(q) = qualified {
        if schema.contains_key(q) {
            return Some(q.to_string());
        }
    }
    if schema.contains_key(bare) {
        return Some(bare.to_string());
    }
    None
}

/// Resolve the FROM clause into a working table + schema + embedding dims.
/// Handles simple tables, table aliases, FROM subqueries, and JOINs.
#[allow(clippy::type_complexity)]
fn resolve_from(
    ctx: &Context,
    from: &[TableWithJoins],
    tables: Option<&HashMap<String, StoredTable>>,
    graphs: Option<&HashMap<String, pgq::PropertyGraph>>,
) -> Result<(Table, HashMap<String, usize>, HashMap<String, i32>), SqlError> {
    if from.is_empty() {
        // Constant SELECT (no FROM): synthesize a 1-row table with a dummy
        // column so that constant expressions broadcast to 1 row during graph
        // execution.  The `__teide_const_dummy__` column is included in the schema so the
        // graph can reference it; scalar_table_to_vector_table strips it from output.
        let name_id = crate::sym_intern("__teide_const_dummy__")?;
        let mut builder = RawTableBuilder::new(1)?;
        let vec = unsafe { crate::raw::td_vec_new(crate::ffi::TD_I64, 1) };
        if vec.is_null() {
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        let zero: i64 = 0;
        let vec2 = unsafe {
            crate::ffi::td_vec_append(vec, &zero as *const i64 as *const std::ffi::c_void)
        };
        if vec2.is_null() || crate::ffi_is_err(vec2) {
            unsafe { crate::ffi_release(vec) };
            return Err(SqlError::Engine(crate::Error::Oom));
        }
        let vec = vec2;
        let res = builder.add_col(name_id, vec);
        unsafe { crate::ffi_release(vec) };
        res?;
        let table = builder.finish()?;
        let mut schema = HashMap::new();
        schema.insert("__teide_const_dummy__".to_string(), 0);
        let emb_dims = HashMap::new();
        return Ok((table, schema, emb_dims));
    }

    // Multiple FROM tables = implicit CROSS JOIN: SELECT * FROM t1, t2
    if from.len() > 1 {
        let (mut result_table, mut result_schema, mut result_emb_dims) =
            resolve_table_factor(ctx, &from[0].relation, tables, graphs)?;
        if let Some(ref alias) = table_factor_alias(&from[0].relation) {
            let snapshot = result_emb_dims.clone();
            add_qualified_embedding_dims(&mut result_emb_dims, alias, &snapshot);
        }
        let mut poisoned = HashSet::new();
        // Process joins on first table
        for join in &from[0].joins {
            let (right_table, right_schema, right_emb) = resolve_table_factor(ctx, &join.relation, tables, graphs)?;
            // Reject cross join if right side has embedding columns too
            reject_if_has_embeddings(&right_emb, "CROSS JOIN")?;
            let left_ncols = result_table.ncols() as usize;
            result_table = exec_cross_join(ctx, &result_table, &right_table, &result_emb_dims)?;
            result_table = qualify_join_result(
                &result_table, left_ncols, &result_schema, &right_schema,
                false, table_factor_alias(&from[0].relation).as_deref(),
                table_factor_alias(&join.relation).as_deref(),
            )?;
            result_schema = build_schema(&result_table);
            let right_emb = if let Some(alias) = table_factor_alias(&join.relation) {
                let mut extended = right_emb;
                let bare: Vec<_> = extended.iter().filter(|(k, _)| !k.contains('.')).map(|(k, &v)| (format!("{alias}.{k}"), v)).collect();
                for (qk, qv) in bare { extended.insert(qk, qv); }
                extended
            } else { right_emb };
            merge_embedding_dims(&mut result_emb_dims, right_emb, &mut poisoned);
        }
        // Cross join with subsequent FROM tables
        for twj in &from[1..] {
            let (right_table, right_schema, right_emb) = resolve_table_factor(ctx, &twj.relation, tables, graphs)?;
            // Reject cross join if right side has embedding columns too
            reject_if_has_embeddings(&right_emb, "CROSS JOIN")?;
            let left_ncols = result_table.ncols() as usize;
            result_table = exec_cross_join(ctx, &result_table, &right_table, &result_emb_dims)?;
            result_table = qualify_join_result(
                &result_table, left_ncols, &result_schema, &right_schema,
                false, None,
                table_factor_alias(&twj.relation).as_deref(),
            )?;
            result_schema = build_schema(&result_table);
            let right_emb = if let Some(alias) = table_factor_alias(&twj.relation) {
                let mut extended = right_emb;
                let bare: Vec<_> = extended.iter().filter(|(k, _)| !k.contains('.')).map(|(k, &v)| (format!("{alias}.{k}"), v)).collect();
                for (qk, qv) in bare { extended.insert(qk, qv); }
                extended
            } else { right_emb };
            merge_embedding_dims(&mut result_emb_dims, right_emb, &mut poisoned);
            for join in &twj.joins {
                let (right_table2, right_schema2, right_emb2) = resolve_table_factor(ctx, &join.relation, tables, graphs)?;
                reject_if_has_embeddings(&right_emb2, "CROSS JOIN")?;
                let left_ncols2 = result_table.ncols() as usize;
                result_table = exec_cross_join(ctx, &result_table, &right_table2, &result_emb_dims)?;
                result_table = qualify_join_result(
                    &result_table, left_ncols2, &result_schema, &right_schema2,
                    false, None,
                    table_factor_alias(&join.relation).as_deref(),
                )?;
                result_schema = build_schema(&result_table);
                let right_emb2 = if let Some(alias) = table_factor_alias(&join.relation) {
                    let mut extended = right_emb2;
                    let bare: Vec<_> = extended.iter().filter(|(k, _)| !k.contains('.')).map(|(k, &v)| (format!("{alias}.{k}"), v)).collect();
                    for (qk, qv) in bare { extended.insert(qk, qv); }
                    extended
                } else { right_emb2 };
                merge_embedding_dims(&mut result_emb_dims, right_emb2, &mut poisoned);
            }
        }
        return Ok((result_table, result_schema, result_emb_dims));
    }

    let twj = &from[0];

    // Resolve the base (left) table
    let (mut left_table, mut left_schema, mut from_emb_dims) = resolve_table_factor(ctx, &twj.relation, tables, graphs)?;

    // Add qualified entries for the left table so that `table.col` references
    // survive bare-name poisoning when joining tables with same-named columns.
    if let Some(ref alias) = table_factor_alias(&twj.relation) {
        let snapshot = from_emb_dims.clone();
        add_qualified_embedding_dims(&mut from_emb_dims, alias, &snapshot);
    }

    // Process JOINs
    let mut join_poisoned = HashSet::new();
    for join in &twj.joins {
        let (right_table, right_schema, right_emb) = resolve_table_factor(ctx, &join.relation, tables, graphs)?;
        // Add qualified entries for the right table before merging.
        let right_emb = if let Some(alias) = table_factor_alias(&join.relation) {
            let mut extended = right_emb;
            let bare_entries: Vec<_> = extended
                .iter()
                .filter(|(k, _)| !k.contains('.'))
                .map(|(k, &v)| (format!("{alias}.{k}"), v))
                .collect();
            for (qk, qv) in bare_entries {
                extended.insert(qk, qv);
            }
            extended
        } else {
            right_emb
        };
        merge_embedding_dims(&mut from_emb_dims, right_emb, &mut join_poisoned);

        // Reject non-CROSS JOINs when either side has embedding columns.
        // The C join kernel copies elements one at a time and is not
        // dimension-aware, so N*D embedding buffers would be corrupted.
        if !matches!(&join.join_operator, JoinOperator::CrossJoin) {
            reject_if_has_embeddings(&from_emb_dims, "JOIN")?;
        }

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
                let left_ncols = left_table.ncols() as usize;
                let result = exec_cross_join(ctx, &left_table, &right_table, &from_emb_dims)?;
                let result = qualify_join_result(
                    &result,
                    left_ncols,
                    &left_schema,
                    &right_schema,
                    false,
                    table_factor_alias(&twj.relation).as_deref(),
                    table_factor_alias(&join.relation).as_deref(),
                )?;
                left_table = result;
                left_schema = build_schema(&left_table);
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

        // Rename duplicate columns in the join result so that `g.scan("alias.col")`
        // resolves to the correct physical column.  Without renaming, the C engine's
        // td_scan picks the first physical match for a bare name, which is wrong when
        // both sides have a column with the same name (e.g. embedding).
        // For RIGHT JOIN the C engine places right_table columns first (it was
        // the actual-left after the swap).  Pass the correct first-side column
        // count so that qualify_join_result computes the boundary correctly.
        let first_side_ncols = if is_right_join {
            right_table.ncols() as usize
        } else {
            left_table.ncols() as usize
        };
        let result = qualify_join_result(
            &result,
            first_side_ncols,
            &left_schema,
            &right_schema,
            is_right_join,
            table_factor_alias(&twj.relation).as_deref(),
            table_factor_alias(&join.relation).as_deref(),
        )?;
        let merged_schema = build_schema(&result);

        left_table = result;
        left_schema = merged_schema;
    }

    Ok((left_table, left_schema, from_emb_dims))
}

/// Extract the effective alias (or table name) from a `TableFactor`.
/// Used to build qualified embedding-dim entries (`alias.col`) that survive
/// bare-name poisoning during JOIN merges.
fn table_factor_alias(factor: &TableFactor) -> Option<String> {
    match factor {
        TableFactor::Table { name, alias, .. } => alias
            .as_ref()
            .map(|a| a.name.value.to_lowercase())
            .or_else(|| Some(object_name_to_string(name).to_lowercase())),
        TableFactor::Derived { alias, .. } => alias
            .as_ref()
            .map(|a| a.name.value.to_lowercase()),
        _ => None,
    }
}

/// Add qualified entries (`"alias.col" -> dim`) to an embedding-dims map.
/// Qualified keys never collide across different tables, so they survive
/// the bare-name poisoning in [`merge_embedding_dims`].
fn add_qualified_embedding_dims(
    dims: &mut HashMap<String, i32>,
    table_alias: &str,
    source_dims: &HashMap<String, i32>,
) {
    for (col, &dim) in source_dims {
        // Only add qualified entries for bare column names (skip already-qualified).
        if !col.contains('.') {
            dims.insert(format!("{table_alias}.{col}"), dim);
        }
    }
}

/// Rename duplicate columns in a join result so that `g.scan("alias.col")`
/// resolves to the correct physical column in the C engine.
///
/// Columns with unique bare names keep their original name.  Columns whose
/// bare name appears in both sides are renamed to `"alias.col"` using
/// the left/right schemas and the right-side alias.  Previously-qualified
/// names from the left side (from chained joins) are preserved.
fn qualify_join_result(
    result: &Table,
    left_ncols: usize,
    left_schema: &HashMap<String, usize>,
    _right_schema: &HashMap<String, usize>,
    is_right_join: bool,
    left_alias: Option<&str>,
    right_alias: Option<&str>,
) -> Result<Table, SqlError> {
    let ncols = result.ncols() as usize;

    // Detect duplicates from the ACTUAL result table columns (not the pre-join
    // schemas), because the C engine may deduplicate join key columns.
    let mut name_count: HashMap<String, u32> = HashMap::new();
    for i in 0..ncols {
        let bare = result.col_name_str(i).to_lowercase();
        *name_count.entry(bare).or_default() += 1;
    }
    let has_dups = name_count.values().any(|&c| c > 1);
    if !has_dups {
        return Ok(result.clone_ref());
    }

    // `left_ncols` is the first-side column count: for a normal join this is the
    // left table's ncols; for a RIGHT JOIN the caller passes the right table's
    // ncols (since the C engine places right columns first after the swap).
    // Cap at ncols to handle C engine key deduplication from the second side.
    let boundary = left_ncols.min(ncols);

    // Build a reverse map: index -> qualified name for left columns.
    // Left columns may already be qualified from a prior chained join.
    let mut left_idx_to_name: HashMap<usize, String> = HashMap::new();
    for (name, &idx) in left_schema {
        if name.contains('.') || !left_idx_to_name.contains_key(&idx) {
            left_idx_to_name.insert(idx, name.clone());
        }
    }

    let mut names = Vec::with_capacity(ncols);
    for i in 0..ncols {
        let bare = result.col_name_str(i).to_lowercase();
        if name_count.get(&bare).copied().unwrap_or(0) <= 1 {
            names.push(bare);
            continue;
        }
        let is_first_side = i < boundary;
        if is_right_join {
            if is_first_side {
                // First side = right table in a right join.
                if let Some(ra) = right_alias {
                    names.push(format!("{ra}.{bare}"));
                } else {
                    names.push(bare);
                }
            } else {
                // Second side = left table (may have chained qualifications).
                let left_idx = i - boundary;
                if let Some(qname) = left_idx_to_name.get(&left_idx) {
                    if qname.contains('.') {
                        names.push(qname.clone());
                    } else if let Some(la) = left_alias {
                        names.push(format!("{la}.{bare}"));
                    } else {
                        names.push(qname.clone());
                    }
                } else if let Some(la) = left_alias {
                    names.push(format!("{la}.{bare}"));
                } else {
                    names.push(bare);
                }
            }
        } else if is_first_side {
            // First side = left table.
            if let Some(qname) = left_idx_to_name.get(&i) {
                if qname.contains('.') {
                    names.push(qname.clone());
                } else if let Some(la) = left_alias {
                    names.push(format!("{la}.{bare}"));
                } else {
                    names.push(qname.clone());
                }
            } else if let Some(la) = left_alias {
                names.push(format!("{la}.{bare}"));
            } else {
                names.push(bare);
            }
        } else {
            // Second side = right table.
            if let Some(ra) = right_alias {
                names.push(format!("{ra}.{bare}"));
            } else {
                names.push(bare);
            }
        }
    }
    result.with_column_names(&names).map_err(|e| {
        SqlError::Plan(format!("Failed to qualify join columns: {e}"))
    })
}

/// Resolve a single TableFactor (table name or FROM subquery) into a table + schema + embedding dims.
#[allow(clippy::type_complexity)]
fn resolve_table_factor(
    ctx: &Context,
    factor: &TableFactor,
    tables: Option<&HashMap<String, StoredTable>>,
    graphs: Option<&HashMap<String, pgq::PropertyGraph>>,
) -> Result<(Table, HashMap<String, usize>, HashMap<String, i32>), SqlError> {
    match factor {
        TableFactor::Table { name, args, .. } => {
            let table_name = object_name_to_string(name);
            // Table functions: read_csv(...), read_splayed(...), read_parted(...)
            if let Some(func_args) = args {
                let func_name = table_name.to_lowercase();
                let table = resolve_table_function(ctx, &func_name, &func_args.args, graphs, tables)?;
                // Normalize column names to lowercase for case-insensitive SQL resolution.
                // Without this, g.scan("limitname") can't find "LimitName" from CSV headers.
                let col_names: Vec<String> = (0..table.ncols() as usize)
                    .map(|i| table.col_name_str(i).to_lowercase())
                    .collect();
                let table = table.with_column_names(&col_names)?;
                let schema = build_schema(&table);
                return Ok((table, schema, HashMap::new()));
            }
            let table_name_lower = table_name.to_lowercase();
            let emb_dims = tables
                .and_then(|t| t.get(&table_name_lower))
                .map(|s| s.embedding_dims.clone())
                .unwrap_or_default();
            let table = resolve_table(&table_name, tables)?;
            let schema = build_schema(&table);
            Ok((table, schema, emb_dims))
        }
        TableFactor::Derived { subquery, .. } => {
            let result = plan_query(ctx, subquery, tables, graphs)?;
            // Rename columns to match SQL aliases so outer scans work
            let table = result.table.with_column_names(&result.columns)?;
            let schema = build_result_schema(&table, &result.columns);
            Ok((table, schema, result.embedding_dims))
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
    embedding_dims: &HashMap<String, i32>,
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
            embedding_dims,
        );
    }

    // Phase 2: Execute GROUP BY with keys + all unique aggregates
    let mut g = ctx.graph(working_table)?;
    g.column_embedding_dims = embedding_dims.clone();

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
    embedding_dims: &HashMap<String, i32>,
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
    g.column_embedding_dims = embedding_dims.clone();
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
    embedding_dims: &HashMap<String, i32>,
) -> Result<(Table, Vec<String>), SqlError> {
    let mut g = ctx.graph(working_table)?;
    g.column_embedding_dims = embedding_dims.clone();
    let table_node = g.const_table(working_table)?;

    let mut proj_cols = Vec::new();
    let mut aliases = Vec::new();

    for item in select_items {
        match item {
            SelectItem::Wildcard(_) => {
                let mut cols: Vec<_> = schema.iter().collect();
                cols.sort_by_key(|(_name, idx)| **idx);
                for (name, _) in cols {
                    // Skip internal __teide_const_dummy__ column used for constant SELECT
                    if name == "__teide_const_dummy__" {
                        continue;
                    }
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

    // For constant SELECT (no real columns, only __teide_const_dummy__), the graph produces
    // scalar atoms for constant expressions. Convert each scalar column to a
    // 1-element vector so the result validator accepts them.
    let is_constant_select = schema.len() == 1 && schema.contains_key("__teide_const_dummy__");
    if is_constant_select {
        let result = scalar_table_to_vector_table(&result)?;
        return Ok((result, aliases));
    }

    Ok((result, aliases))
}

/// Convert a table whose columns may be scalar atoms into a table where every
/// column is a 1-element vector. Used for constant SELECT results.
fn scalar_table_to_vector_table(table: &Table) -> Result<Table, SqlError> {
    let ncols = table.ncols();
    let mut builder = RawTableBuilder::new(ncols as i64)?;
    for c in 0..ncols {
        let col = table
            .get_col_idx(c as i64)
            .ok_or_else(|| SqlError::Plan("missing column in constant select".into()))?;
        let col_type = unsafe { crate::raw::td_type(col) };
        let name = table.col_name_str(c as usize);
        let name_id = crate::sym_intern(&name)?;
        if col_type > 0 {
            // Already a vector — reuse it directly
            unsafe { crate::ffi_retain(col) };
            let res = builder.add_col(name_id, col);
            if res.is_err() {
                unsafe { crate::ffi_release(col) };
            }
            res?;
        } else {
            // Scalar atom: create a 1-element vector and copy the value.
            // For atoms, the value lives in the `val` union (bytes 24-31),
            // not at td_data() offset 32 which is for vector payloads.
            // Special case: TD_ATOM_STR (-8) maps to TD_SYM (20), not TD_F32.
            let vec_type = if col_type == crate::ffi::TD_ATOM_STR {
                crate::ffi::TD_SYM
            } else {
                -col_type
            };
            let vec = if col_type == crate::ffi::TD_ATOM_STR {
                // String atoms store inline string data, not symbol IDs.
                // Read the string, intern it, and create a SYM vector.
                let ptr = unsafe { crate::ffi::td_str_ptr(col) };
                let slen = unsafe { crate::ffi::td_str_len(col) };
                let s = if ptr.is_null() || slen == 0 {
                    ""
                } else {
                    unsafe {
                        std::str::from_utf8(std::slice::from_raw_parts(
                            ptr as *const u8,
                            slen,
                        ))
                        .unwrap_or("")
                    }
                };
                let sym_id = crate::sym_intern(s)? as u32;
                let vec = unsafe { crate::ffi::td_sym_vec_new(crate::ffi::TD_SYM_W32, 1) };
                if vec.is_null() {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                unsafe {
                    crate::ffi::td_vec_append(
                        vec,
                        &sym_id as *const u32 as *const std::ffi::c_void,
                    )
                }
            } else if vec_type == crate::ffi::TD_SYM {
                // SYM atoms store the symbol ID as i64 in val.
                let sym_id = unsafe { (*col).val.i64_ } as u32;
                let vec = unsafe { crate::ffi::td_sym_vec_new(crate::ffi::TD_SYM_W32, 1) };
                if vec.is_null() {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                unsafe {
                    crate::ffi::td_vec_append(
                        vec,
                        &sym_id as *const u32 as *const std::ffi::c_void,
                    )
                }
            } else {
                let vec = unsafe { crate::raw::td_vec_new(vec_type, 1) };
                if vec.is_null() {
                    return Err(SqlError::Engine(crate::Error::Oom));
                }
                let val_ptr = unsafe {
                    &(*col).val as *const crate::ffi::td_t_val as *const std::ffi::c_void
                };
                unsafe { crate::ffi::td_vec_append(vec, val_ptr) }
            };
            if vec.is_null() || crate::ffi_is_err(vec) {
                return Err(SqlError::Engine(crate::Error::Oom));
            }
            let res = builder.add_col(name_id, vec);
            unsafe { crate::ffi_release(vec) };
            res?;
        }
    }
    builder.finish()
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
    embedding_dims: &HashMap<String, i32>,
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
            g.column_embedding_dims = embedding_dims.clone();
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
fn exec_cross_join(
    ctx: &Context,
    left: &Table,
    right: &Table,
    embedding_dims: &HashMap<String, i32>,
) -> Result<Table, SqlError> {
    let _ = ctx;

    // Embedding columns are flat N*D F32 arrays.  The element-wise memcpy
    // expansion does not handle multi-element rows, so reject early.
    if !embedding_dims.is_empty() {
        return Err(SqlError::Plan(
            "CROSS JOIN is not supported on tables with embedding columns".into(),
        ));
    }

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
    embedding_dims: &HashMap<String, i32>,
) -> Result<Table, SqlError> {
    use std::collections::HashMap as StdMap;

    let _ = ctx;

    // Embedding columns are flat N*D F32 arrays.  The per-element hash/compare
    // and copy logic is not dimension-aware, so reject early.
    if !embedding_dims.is_empty() {
        let op = if keep_matches { "INTERSECT" } else { "EXCEPT" };
        return Err(SqlError::Plan(format!(
            "{op} is not supported on tables with embedding columns"
        )));
    }

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
    embedding_dims: HashMap<String, i32>,
) -> Result<SqlResult, SqlError> {
    // ORDER BY (optionally fused with LIMIT)
    let order_by_exprs = extract_order_by(query)?;
    let offset_val = extract_offset(query)?;
    let limit_val = extract_limit(query)?;

    // Embedding columns are flat N*D F32 arrays; the C sort/head kernels
    // are not dimension-aware and would scramble per-row vector grouping.
    // dim=1 embeddings have scalar layout and are safe for these operations.
    let high_dim: HashMap<String, i32> = embedding_dims
        .iter()
        .filter(|(_, &d)| d > 1)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    if !order_by_exprs.is_empty() {
        reject_if_has_embeddings(&high_dim, "ORDER BY on set operation result")?;
    }
    if limit_val.is_some() || offset_val.is_some() {
        reject_if_has_embeddings(&high_dim, "LIMIT/OFFSET on set operation result")?;
    }

    let (result_table, limit_fused) = if !order_by_exprs.is_empty() {
        let table_col_names: Vec<String> = (0..result_table.ncols() as usize)
            .map(|i| result_table.col_name_str(i).to_string())
            .collect();
        let mut g = ctx.graph(&result_table)?;
        g.column_embedding_dims = embedding_dims.clone();
        let table_node = g.const_table(&result_table)?;
        let sort_node = plan_order_by(
            &mut g,
            table_node,
            &order_by_exprs,
            &result_aliases,
            &table_col_names,
        )?;

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
                let total = off.checked_add(lim)
                    .ok_or_else(|| SqlError::Plan("OFFSET + LIMIT overflow".into()))?;
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

    let nrows = validate_result_table(&result_table, &embedding_dims, &result_aliases)?;
    Ok(SqlResult {
        table: result_table,
        columns: result_aliases,
        embedding_dims,
        nrows: nrows as usize,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return an error if `dims` is non-empty, indicating that the operation is
/// not supported on tables with embedding columns.
fn reject_if_has_embeddings(
    dims: &HashMap<String, i32>,
    operation: &str,
) -> Result<(), SqlError> {
    if !dims.is_empty() {
        let emb_list: Vec<_> = dims.keys().cloned().collect();
        return Err(SqlError::Plan(format!(
            "{operation} is not yet supported on tables with embedding columns \
             (source has embedding columns: {})",
            emb_list.join(", ")
        )));
    }
    Ok(())
}

fn validate_result_table(
    table: &Table,
    embedding_dims: &HashMap<String, i32>,
    result_aliases: &[String],
) -> Result<i64, SqlError> {
    // td_table_nrows() returns cols[0]->len, which is N*D when the first
    // column is an embedding.  Derive the true row count from the first
    // non-embedding column.  If every column is an embedding, compute N
    // from the first column's len / dim.
    let raw_nrows = table.nrows();
    let nrows = 'nrows: {
        for ci in 0..table.ncols() {
            let name = table.col_name_str(ci as usize).to_lowercase();
            let alias = result_aliases.get(ci as usize);
            let is_emb = embedding_dims.contains_key(&name)
                || alias.is_some_and(|a| embedding_dims.contains_key(a));
            if !is_emb {
                // First non-embedding column – its len equals the true row count.
                if let Some(col) = table.get_col_idx(ci) {
                    let col_type = unsafe { crate::raw::td_type(col) };
                    if col_type > 0
                        && !crate::ffi::td_is_parted(col_type)
                        && col_type != crate::ffi::TD_MAPCOMMON
                    {
                        break 'nrows unsafe { crate::raw::td_len(col) };
                    }
                }
                break 'nrows raw_nrows;
            }
        }
        // All columns are embeddings — derive N from the first column.
        if table.ncols() > 0 {
            let name = table.col_name_str(0).to_lowercase();
            let alias = result_aliases.first();
            let dim = embedding_dims
                .get(&name)
                .or_else(|| alias.and_then(|a| embedding_dims.get(a)));
            if let Some(&d) = dim {
                if d > 0 {
                    break 'nrows raw_nrows / d as i64;
                }
            }
        }
        raw_nrows
    };
    for col_idx in 0..table.ncols() {
        let col = table.get_col_idx(col_idx).ok_or_else(|| {
            SqlError::Plan(format!("Result column at index {col_idx} is missing"))
        })?;
        let col_type = unsafe { crate::raw::td_type(col) };
        if col_type < 0 {
            return Err(SqlError::Plan(format!(
                "Result column '{}' has scalar type {} and is not supported",
                table.col_name_str(col_idx as usize),
                col_type
            )));
        }
        // TD_LIST (type 0): list-of-lists column — skip length check since
        // the outer list length equals nrows but inner lists vary.
        if col_type == 0 {
            continue;
        }
        // TD_PARTED and MAPCOMMON columns: len = partition count, not row count.
        // Skip the length check — td_table_nrows() already handles them correctly.
        if crate::ffi::td_is_parted(col_type) || col_type == crate::ffi::TD_MAPCOMMON {
            continue;
        }
        let len = unsafe { crate::raw::td_len(col) };
        if len != nrows {
            // Embedding columns are flat TD_F32 arrays with len = nrows * dim.
            // Check if this column is a known embedding and its length is consistent.
            // Try both the engine column name and the result alias (set operations
            // store dims under the alias which may differ from the engine name).
            let col_name = table.col_name_str(col_idx as usize).to_lowercase();
            let alias_name = result_aliases.get(col_idx as usize);
            let dim = embedding_dims
                .get(&col_name)
                .or_else(|| alias_name.and_then(|a| embedding_dims.get(a)));
            if let Some(&dim) = dim {
                if let Some(expected) = nrows.checked_mul(dim as i64) {
                    if len == expected {
                        continue;
                    }
                }
            }
            return Err(SqlError::Plan(format!(
                "Result column '{}' has length {} but table has {} rows",
                table.col_name_str(col_idx as usize),
                len,
                nrows
            )));
        }
    }
    Ok(nrows)
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

/// Merge embedding dims from a right table into the left map.
/// If both sides define the same bare column name with different dimensions,
/// the entry is dropped to avoid silently trusting the wrong metadata.
/// `poisoned` tracks names that have already been seen with conflicting dims
/// so they are never reinserted by a later table.
fn merge_embedding_dims(
    left: &mut HashMap<String, i32>,
    right: HashMap<String, i32>,
    poisoned: &mut HashSet<String>,
) {
    for (name, dim) in right {
        if poisoned.contains(&name) {
            continue;
        }
        match left.entry(name) {
            std::collections::hash_map::Entry::Occupied(e) => {
                if *e.get() != dim {
                    // Conflicting dimensions — remove and poison to prevent
                    // a later table from reinserting.
                    let key = e.key().clone();
                    e.remove();
                    poisoned.insert(key);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(dim);
            }
        }
    }
}

/// Intersect embedding dims from two set-operation sides, matched by column
/// position.  Returns an error if both sides define dims for the same
/// positional column but with different values (would corrupt the flat F32
/// buffer after concat).
fn intersect_embedding_dims_positional(
    left_cols: &[String],
    left_dims: &HashMap<String, i32>,
    right_cols: &[String],
    right_dims: &HashMap<String, i32>,
    op_name: &str,
) -> Result<HashMap<String, i32>, SqlError> {
    let mut merged = HashMap::new();
    for (left_name, right_name) in left_cols.iter().zip(right_cols.iter()) {
        let l = left_dims.get(left_name).copied();
        let r = right_dims.get(right_name).copied();
        match (l, r) {
            (Some(ld), Some(rd)) => {
                if ld != rd {
                    return Err(SqlError::Plan(format!(
                        "{op_name}: embedding column '{left_name}' dimension mismatch \
                         (left has {ld}, right has {rd})"
                    )));
                }
                merged.insert(left_name.clone(), ld);
            }
            (Some(ld), None) if ld > 1 => {
                return Err(SqlError::Plan(format!(
                    "{op_name}: left column '{left_name}' is an embedding (dim={ld}) \
                     but right column '{right_name}' is not — the flat N*D buffer \
                     would be misinterpreted as scalar rows"
                )));
            }
            (None, Some(rd)) if rd > 1 => {
                return Err(SqlError::Plan(format!(
                    "{op_name}: right column '{right_name}' is an embedding (dim={rd}) \
                     but left column '{left_name}' is not — the flat N*D buffer \
                     would be misinterpreted as scalar rows"
                )));
            }
            _ => {}
        }
    }
    Ok(merged)
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
            let result = plan_query(ctx, query, tables, None)?;
            if result.columns.len() != 1 {
                return Err(SqlError::Plan(format!(
                    "Scalar subquery must return exactly 1 column, got {}",
                    result.columns.len()
                )));
            }
            let nrows = result.nrows;
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
            let result = plan_query(ctx, subquery, tables, None)?;
            if result.columns.len() != 1 {
                return Err(SqlError::Plan(format!(
                    "IN subquery must return exactly 1 column, got {}",
                    result.columns.len()
                )));
            }
            let nrows = result.nrows;
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
            let result = plan_query(ctx, subquery, tables, None)?;
            let exists = result.nrows > 0;
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

/// Recursively collect column identifier references from an expression.
fn collect_expr_column_refs(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Identifier(ident) => {
            out.push(ident.value.to_lowercase());
        }
        Expr::Function(f) => {
            if let sqlparser::ast::FunctionArguments::List(arg_list) = &f.args {
                for arg in &arg_list.args {
                    if let sqlparser::ast::FunctionArg::Unnamed(
                        sqlparser::ast::FunctionArgExpr::Expr(e),
                    ) = arg
                    {
                        collect_expr_column_refs(e, out);
                    }
                }
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_expr_column_refs(left, out);
            collect_expr_column_refs(right, out);
        }
        Expr::UnaryOp { expr: inner, .. } => {
            collect_expr_column_refs(inner, out);
        }
        Expr::Nested(inner) => {
            collect_expr_column_refs(inner, out);
        }
        Expr::Cast { expr: inner, .. } => {
            collect_expr_column_refs(inner, out);
        }
        _ => {} // literals, arrays, etc. — no column refs
    }
}

/// Collect ORDER BY columns that are valid source columns but not part of the
/// visible SELECT projection. These will be carried as hidden columns.
fn collect_hidden_order_columns(
    order_by: &[(OrderByItem, bool, Option<bool>)],
    visible_aliases: &[String],
    schema: &HashMap<String, usize>,
) -> Vec<String> {
    let mut extra = Vec::new();
    let add_if_hidden = |name: &String, extra: &mut Vec<String>| {
        if visible_aliases.iter().any(|a| a == name) {
            return;
        }
        if schema.contains_key(name) && !extra.iter().any(|c| c == name) {
            extra.push(name.clone());
        }
    };
    for (item, _, _) in order_by {
        match item {
            OrderByItem::Name(name) => {
                add_if_hidden(name, &mut extra);
            }
            OrderByItem::Expression(expr) => {
                let mut refs = Vec::new();
                collect_expr_column_refs(expr, &mut refs);
                for name in &refs {
                    add_if_hidden(name, &mut extra);
                }
            }
            OrderByItem::Position(_) => {}
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

    let mut g = ctx.graph(&table)?;
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

/// Check if the SELECT output would include any high-dimensional embedding
/// columns.  Returns `true` if a wildcard or direct column reference to a
/// high-dim embedding column is found.  Function calls and other expressions
/// produce scalar results and are safe.
fn select_output_has_embeddings(
    select_items: &[SelectItem],
    embedding_dims: &HashMap<String, i32>,
) -> bool {
    if embedding_dims.is_empty() {
        return false;
    }
    for item in select_items {
        match item {
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => {
                // Wildcard includes all source columns, including embeddings.
                return true;
            }
            SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
                if expr_is_embedding_ref(expr, embedding_dims) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if ORDER BY items would place any high-dimensional embedding column
/// into the result table as a hidden sort key.
fn order_by_has_embedding_ref(
    order_by: &[(OrderByItem, bool, Option<bool>)],
    embedding_dims: &HashMap<String, i32>,
) -> bool {
    for (item, _, _) in order_by {
        match item {
            OrderByItem::Name(name) => {
                if embedding_dims.get(name).is_some_and(|&d| d > 1) {
                    return true;
                }
            }
            OrderByItem::Expression(expr) => {
                if expr_is_embedding_ref(expr, embedding_dims) {
                    return true;
                }
            }
            OrderByItem::Position(_) => {
                // Positions reference SELECT output columns which are already
                // checked by select_output_has_embeddings.
            }
        }
    }
    false
}

/// Check if an expression references a high-dim embedding column, recursively
/// walking into sub-expressions.  Arithmetic over an embedding column (e.g.
/// `embedding + 0.0`) still operates on the flat N*D buffer, so it must be
/// rejected just like a direct reference.
fn expr_is_embedding_ref(expr: &Expr, embedding_dims: &HashMap<String, i32>) -> bool {
    match expr {
        Expr::Identifier(ident) => {
            embedding_dims
                .get(&ident.value.to_lowercase())
                .is_some_and(|&d| d > 1)
        }
        Expr::CompoundIdentifier(parts) => {
            if let Some(last) = parts.last() {
                embedding_dims
                    .get(&last.value.to_lowercase())
                    .is_some_and(|&d| d > 1)
            } else {
                false
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            expr_is_embedding_ref(left, embedding_dims)
                || expr_is_embedding_ref(right, embedding_dims)
        }
        Expr::UnaryOp { expr: inner, .. } => expr_is_embedding_ref(inner, embedding_dims),
        Expr::Nested(inner) => expr_is_embedding_ref(inner, embedding_dims),
        Expr::Function(func) => {
            // cosine_similarity / euclidean_distance consume an embedding
            // column but produce a scalar f64 result — they are safe.
            let fname = object_name_to_string(&func.name).to_lowercase();
            if fname == "cosine_similarity" || fname == "euclidean_distance" {
                return false;
            }
            if let FunctionArguments::List(arg_list) = &func.args {
                arg_list.args.iter().any(|arg| match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e))
                    | FunctionArg::Named { arg: FunctionArgExpr::Expr(e), .. } => {
                        expr_is_embedding_ref(e, embedding_dims)
                    }
                    _ => false,
                })
            } else {
                false
            }
        }
        Expr::Cast { expr: inner, .. } => expr_is_embedding_ref(inner, embedding_dims),
        Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            expr_is_embedding_ref(inner, embedding_dims)
        }
        Expr::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            expr_is_embedding_ref(inner, embedding_dims)
                || expr_is_embedding_ref(low, embedding_dims)
                || expr_is_embedding_ref(high, embedding_dims)
        }
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(op) = operand {
                if expr_is_embedding_ref(op, embedding_dims) {
                    return true;
                }
            }
            for c in conditions {
                if expr_is_embedding_ref(c, embedding_dims) {
                    return true;
                }
            }
            for r in results {
                if expr_is_embedding_ref(r, embedding_dims) {
                    return true;
                }
            }
            if let Some(e) = else_result {
                if expr_is_embedding_ref(e, embedding_dims) {
                    return true;
                }
            }
            false
        }
        Expr::Like {
            expr: inner,
            pattern,
            ..
        }
        | Expr::ILike {
            expr: inner,
            pattern,
            ..
        } => {
            expr_is_embedding_ref(inner, embedding_dims)
                || expr_is_embedding_ref(pattern, embedding_dims)
        }
        Expr::Trim { expr: inner, .. } => expr_is_embedding_ref(inner, embedding_dims),
        Expr::Substring { expr: inner, .. } => expr_is_embedding_ref(inner, embedding_dims),
        Expr::Extract { expr: inner, .. } => expr_is_embedding_ref(inner, embedding_dims),
        Expr::Ceil { expr: inner, .. } | Expr::Floor { expr: inner, .. } => {
            expr_is_embedding_ref(inner, embedding_dims)
        }
        Expr::InList {
            expr: inner, list, ..
        } => {
            expr_is_embedding_ref(inner, embedding_dims)
                || list.iter().any(|e| expr_is_embedding_ref(e, embedding_dims))
        }
        _ => false,
    }
}
