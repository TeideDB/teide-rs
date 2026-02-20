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

// Expression tree walker: translates sqlparser AST expressions into Teide DAG nodes.

use std::collections::HashMap;

use sqlparser::ast::{
    BinaryOperator, CastKind, DataType, DateTimeField, DuplicateTreatment, Expr, Function,
    FunctionArg, FunctionArgExpr, FunctionArguments, SelectItem, UnaryOperator, Value,
    WindowFrameBound, WindowFrameUnits, WindowSpec, WindowType,
};

use crate::{AggOp, Column, FrameBound, FrameType, Graph, WindowFunc};

use super::SqlError;

/// Recursively plan a scalar expression into a Teide DAG column node.
pub fn plan_expr(
    g: &mut Graph,
    expr: &Expr,
    schema: &HashMap<String, usize>,
) -> Result<Column, SqlError> {
    match expr {
        Expr::Identifier(ident) => {
            let name = ident.value.to_lowercase();
            if !schema.contains_key(&name) {
                return Err(SqlError::Plan(format!("Column '{}' not found", name)));
            }
            Ok(g.scan(&name)?)
        }

        Expr::Value(val) => match val {
            Value::Number(n, _) => {
                if let Ok(i) = n.parse::<i64>() {
                    Ok(g.const_i64(i)?)
                } else {
                    let f = n
                        .parse::<f64>()
                        .map_err(|_| SqlError::Plan(format!("Invalid number literal: {n}")))?;
                    Ok(g.const_f64(f)?)
                }
            }
            Value::SingleQuotedString(s) => Ok(g.const_str(s)?),
            Value::Boolean(b) => Ok(g.const_bool(*b)?),
            Value::Null => {
                // NULL represented as f64 NaN constant — C engine uses NaN-as-null
                Ok(g.const_f64(f64::NAN)?)
            }
            _ => Err(SqlError::Plan(format!("Unsupported value: {val}"))),
        },

        Expr::BinaryOp { left, op, right } => {
            let l = plan_expr(g, left, schema)?;
            let r = plan_expr(g, right, schema)?;
            match op {
                BinaryOperator::Plus => Ok(g.add(l, r)?),
                BinaryOperator::Minus => Ok(g.sub(l, r)?),
                BinaryOperator::Multiply => Ok(g.mul(l, r)?),
                BinaryOperator::Divide => Ok(g.div(l, r)?),
                BinaryOperator::Modulo => Ok(g.modulo(l, r)?),
                BinaryOperator::Eq => Ok(g.eq(l, r)?),
                BinaryOperator::NotEq => Ok(g.ne(l, r)?),
                BinaryOperator::Lt => Ok(g.lt(l, r)?),
                BinaryOperator::LtEq => Ok(g.le(l, r)?),
                BinaryOperator::Gt => Ok(g.gt(l, r)?),
                BinaryOperator::GtEq => Ok(g.ge(l, r)?),
                BinaryOperator::And => Ok(g.and(l, r)?),
                BinaryOperator::Or => Ok(g.or(l, r)?),
                BinaryOperator::StringConcat => Ok(g.concat(&[l, r])?),
                _ => Err(SqlError::Plan(format!("Unsupported operator: {op}"))),
            }
        }

        Expr::UnaryOp { op, expr: inner } => {
            let e = plan_expr(g, inner, schema)?;
            match op {
                UnaryOperator::Not => Ok(g.not(e)?),
                UnaryOperator::Minus => Ok(g.neg(e)?),
                _ => Err(SqlError::Plan(format!("Unsupported unary operator: {op}"))),
            }
        }

        Expr::Nested(inner) => plan_expr(g, inner, schema),

        // IS NULL / IS NOT NULL
        Expr::IsNull(inner) => {
            let e = plan_expr(g, inner, schema)?;
            Ok(g.isnull(e)?)
        }
        Expr::IsNotNull(inner) => {
            let e = plan_expr(g, inner, schema)?;
            Ok(g.not(g.isnull(e)?)?)
        }

        // BETWEEN: x BETWEEN a AND b  →  x >= a AND x <= b
        Expr::Between {
            expr: inner,
            negated,
            low,
            high,
        } => {
            let x = plan_expr(g, inner, schema)?;
            let lo = plan_expr(g, low, schema)?;
            let hi = plan_expr(g, high, schema)?;
            let ge = g.ge(x, lo)?;
            let le = g.le(x, hi)?;
            let result = g.and(ge, le)?;
            if *negated {
                Ok(g.not(result)?)
            } else {
                Ok(result)
            }
        }

        // IN list: x IN (a, b, c)  →  x = a OR x = b OR x = c
        // Hoist the inner expression once and reuse for all comparisons to
        // avoid creating redundant scan nodes in the graph.
        Expr::InList {
            expr: inner,
            list,
            negated,
        } => {
            if list.is_empty() {
                return Ok(g.const_bool(*negated)?);
            }
            let x = plan_expr(g, inner, schema)?;
            let first_val = plan_expr(g, &list[0], schema)?;
            let mut result = g.eq(x, first_val)?;
            for item in &list[1..] {
                let val = plan_expr(g, item, schema)?;
                let cmp = g.eq(x, val)?;
                result = g.or(result, cmp)?;
            }
            if *negated {
                Ok(g.not(result)?)
            } else {
                Ok(result)
            }
        }

        // CAST(expr AS type) / expr::type
        Expr::Cast {
            expr: inner,
            data_type,
            kind,
            ..
        } => {
            if *kind == CastKind::TryCast {
                return Err(SqlError::Plan("TRY_CAST not supported".into()));
            }
            let e = plan_expr(g, inner, schema)?;
            let target = map_sql_type(data_type)?;
            Ok(g.cast(e, target)?)
        }

        // CASE WHEN → nested IF
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            let else_val = match else_result {
                Some(e) => plan_expr(g, e, schema)?,
                None => g.const_f64(f64::NAN)?, // NULL default
            };
            let mut result = else_val;

            if let Some(op) = operand {
                // Simple CASE: CASE x WHEN v1 THEN r1 ...
                for (cond_val, then_val) in conditions.iter().zip(results.iter()).rev() {
                    let x = plan_expr(g, op, schema)?;
                    let v = plan_expr(g, cond_val, schema)?;
                    let c = g.eq(x, v)?;
                    let t = plan_expr(g, then_val, schema)?;
                    result = g.if_then_else(c, t, result)?;
                }
            } else {
                // Searched CASE: CASE WHEN c1 THEN r1 ...
                for (cond_expr, then_val) in conditions.iter().zip(results.iter()).rev() {
                    let c = plan_expr(g, cond_expr, schema)?;
                    let t = plan_expr(g, then_val, schema)?;
                    result = g.if_then_else(c, t, result)?;
                }
            }
            Ok(result)
        }

        // LIKE / NOT LIKE
        Expr::Like {
            negated,
            expr: inner,
            pattern,
            ..
        } => {
            let input = plan_expr(g, inner, schema)?;
            let pat = plan_expr(g, pattern, schema)?;
            let result = g.like(input, pat)?;
            if *negated {
                Ok(g.not(result)?)
            } else {
                Ok(result)
            }
        }

        // ILIKE (case-insensitive) — native OP_ILIKE avoids LOWER() temporaries
        Expr::ILike {
            negated,
            expr: inner,
            pattern,
            ..
        } => {
            let input = plan_expr(g, inner, schema)?;
            let pat = plan_expr(g, pattern, schema)?;
            let result = g.ilike(input, pat)?;
            if *negated {
                Ok(g.not(result)?)
            } else {
                Ok(result)
            }
        }

        // CompoundIdentifier: table_alias.column
        Expr::CompoundIdentifier(parts) => {
            if parts.len() == 2 {
                let col_name = parts[1].value.to_lowercase();
                if schema.contains_key(&col_name) {
                    return Ok(g.scan(&col_name)?);
                }
                // Try fully qualified name "alias.col"
                let full = format!("{}.{}", parts[0].value.to_lowercase(), col_name);
                if schema.contains_key(&full) {
                    return Ok(g.scan(&full)?);
                }
                return Err(SqlError::Plan(format!("Column '{}' not found", col_name)));
            }
            Err(SqlError::Plan(format!(
                "Unsupported compound identifier: {expr}"
            )))
        }

        // Subquery: these should be resolved by resolve_subqueries() before plan_expr
        Expr::Subquery(_query) => Err(SqlError::Plan(
            "Scalar subquery must be pre-resolved; this is a planner bug".into(),
        )),

        // IN (subquery): should be rewritten to IN (list) by resolve_subqueries()
        Expr::InSubquery { .. } => Err(SqlError::Plan(
            "IN (subquery) must be pre-resolved; this is a planner bug".into(),
        )),

        // Scalar functions, aggregate detection, and window function detection
        Expr::Function(f) => {
            let name = f.name.to_string().to_lowercase();
            if f.over.is_some() {
                Err(SqlError::Plan(format!(
                    "Window function '{name}' not allowed in this context (should be pre-resolved)"
                )))
            } else if is_aggregate_name(&name) {
                Err(SqlError::Plan(format!(
                    "Aggregate function '{name}' not allowed in this context"
                )))
            } else {
                plan_scalar_function(g, &name, f, schema)
            }
        }

        Expr::Trim { expr, .. } => {
            let a = plan_expr(g, expr, schema)?;
            Ok(g.trim(a)?)
        }

        Expr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let s = plan_expr(g, expr, schema)?;
            let start = if let Some(from) = substring_from {
                plan_expr(g, from, schema)?
            } else {
                g.const_i64(1)?
            };
            let len = if let Some(for_expr) = substring_for {
                plan_expr(g, for_expr, schema)?
            } else {
                g.const_i64(i64::MAX)?
            };
            Ok(g.substr(s, start, len)?)
        }

        Expr::Extract {
            field, expr: inner, ..
        } => {
            let col = plan_expr(g, inner, schema)?;
            let field_id = map_datetime_field(field)?;
            Ok(g.extract(col, field_id)?)
        }

        // sqlparser v0.53+ parses CEIL/FLOOR as dedicated Expr variants
        Expr::Ceil { expr, .. } => {
            let a = plan_expr(g, expr, schema)?;
            Ok(g.ceil(a)?)
        }
        Expr::Floor { expr, .. } => {
            let a = plan_expr(g, expr, schema)?;
            Ok(g.floor(a)?)
        }

        _ => Err(SqlError::Plan(format!("Unsupported expression: {expr}"))),
    }
}

/// Map sqlparser DateTimeField to Teide extract_field constants.
fn map_datetime_field(field: &DateTimeField) -> Result<i64, SqlError> {
    match field {
        DateTimeField::Year => Ok(crate::extract_field::YEAR),
        DateTimeField::Month => Ok(crate::extract_field::MONTH),
        DateTimeField::Day => Ok(crate::extract_field::DAY),
        DateTimeField::Hour => Ok(crate::extract_field::HOUR),
        DateTimeField::Minute => Ok(crate::extract_field::MINUTE),
        DateTimeField::Second => Ok(crate::extract_field::SECOND),
        DateTimeField::Dow | DateTimeField::DayOfWeek => Ok(crate::extract_field::DOW),
        DateTimeField::Doy | DateTimeField::DayOfYear => Ok(crate::extract_field::DOY),
        DateTimeField::Epoch => Ok(crate::extract_field::EPOCH),
        _ => Err(SqlError::Plan(format!(
            "Unsupported EXTRACT field: {field}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------

fn plan_scalar_function(
    g: &mut Graph,
    name: &str,
    func: &Function,
    schema: &HashMap<String, usize>,
) -> Result<Column, SqlError> {
    let args = extract_func_args(func)?;

    match name {
        // 1-arg math functions
        "abs" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.abs(a)?)
        }
        "ceil" | "ceiling" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.ceil(a)?)
        }
        "floor" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.floor(a)?)
        }
        "sqrt" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.sqrt(a)?)
        }
        "ln" | "log" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.log(a)?)
        }
        "exp" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.exp(a)?)
        }

        // 2-arg functions
        "round" => {
            if args.is_empty() || args.len() > 2 {
                return Err(SqlError::Plan("ROUND takes 1 or 2 arguments".into()));
            }
            let a = plan_expr(g, &args[0], schema)?;
            // Helper: round_val = IF(val >= 0, FLOOR(val + 0.5), CEIL(val - 0.5))
            // This handles negative numbers correctly (banker's-style half-away-from-zero)
            let build_round =
                |g: &mut crate::Graph, val: crate::Column| -> Result<crate::Column, SqlError> {
                    let zero = g.const_f64(0.0)?;
                    let half = g.const_f64(0.5)?;
                    let cond = g.ge(val, zero)?;
                    let pos = g.floor(g.add(val, half)?)?;
                    let neg = g.ceil(g.sub(val, half)?)?;
                    Ok(g.if_then_else(cond, pos, neg)?)
                };
            if args.len() == 1 {
                build_round(g, a)
            } else {
                // ROUND(x, n): extract n as integer constant
                let n = match &args[1] {
                    Expr::Value(Value::Number(s, _)) => s
                        .parse::<i32>()
                        .map_err(|_| SqlError::Plan("ROUND precision must be an integer".into()))?,
                    Expr::UnaryOp {
                        op: UnaryOperator::Minus,
                        expr,
                    } => match expr.as_ref() {
                        Expr::Value(Value::Number(s, _)) => -s.parse::<i32>().map_err(|_| {
                            SqlError::Plan("ROUND precision must be an integer".into())
                        })?,
                        _ => {
                            return Err(SqlError::Plan(
                                "ROUND precision must be an integer literal".into(),
                            ))
                        }
                    },
                    _ => {
                        return Err(SqlError::Plan(
                            "ROUND precision must be an integer literal".into(),
                        ))
                    }
                };
                // ROUND(x, n) → round(x * scale) / scale
                let scale = 10.0_f64.powi(n);
                let scale_node = g.const_f64(scale)?;
                let scaled = g.mul(a, scale_node)?;
                let rounded = build_round(g, scaled)?;
                let inv_scale = g.const_f64(1.0 / scale)?;
                Ok(g.mul(rounded, inv_scale)?)
            }
        }

        "least" => {
            if args.len() < 2 {
                return Err(SqlError::Plan("LEAST requires at least 2 arguments".into()));
            }
            let mut result = plan_expr(g, &args[0], schema)?;
            for arg in &args[1..] {
                let b = plan_expr(g, arg, schema)?;
                result = g.min2(result, b)?;
            }
            Ok(result)
        }
        "greatest" => {
            if args.len() < 2 {
                return Err(SqlError::Plan(
                    "GREATEST requires at least 2 arguments".into(),
                ));
            }
            let mut result = plan_expr(g, &args[0], schema)?;
            for arg in &args[1..] {
                let b = plan_expr(g, arg, schema)?;
                result = g.max2(result, b)?;
            }
            Ok(result)
        }

        // COALESCE(a, b, ...) → nested IF(NOT ISNULL(a), a, IF(NOT ISNULL(b), b, c))
        "coalesce" => {
            if args.is_empty() {
                return Err(SqlError::Plan(
                    "COALESCE requires at least 1 argument".into(),
                ));
            }
            if args.len() == 1 {
                return plan_expr(g, &args[0], schema);
            }
            // Build right-to-left: last arg is the fallback, then wrap in IF chains
            let fallback = &args[args.len() - 1];
            let mut result = plan_expr(g, fallback, schema)?;
            for arg in args[..args.len() - 1].iter().rev() {
                let val = plan_expr(g, arg, schema)?;
                let is_null = g.isnull(val)?;
                let not_null = g.not(is_null)?;
                result = g.if_then_else(not_null, val, result)?;
            }
            Ok(result)
        }

        // NULLIF(a, b) → IF(a = b, NULL, a)
        "nullif" => {
            check_arg_count(name, &args, 2)?;
            let a = plan_expr(g, &args[0], schema)?;
            let b = plan_expr(g, &args[1], schema)?;
            let eq = g.eq(a, b)?;
            let null_val = g.const_f64(f64::NAN)?;
            Ok(g.if_then_else(eq, null_val, a)?)
        }

        // String functions
        "upper" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.upper(a)?)
        }
        "lower" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.lower(a)?)
        }
        "length" | "len" | "char_length" | "character_length" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.strlen(a)?)
        }
        "trim" | "btrim" => {
            check_arg_count(name, &args, 1)?;
            let a = plan_expr(g, &args[0], schema)?;
            Ok(g.trim(a)?)
        }
        "substr" | "substring" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(SqlError::Plan(format!(
                    "SUBSTR requires 2 or 3 arguments, got {}",
                    args.len()
                )));
            }
            let s = plan_expr(g, &args[0], schema)?;
            let start = plan_expr(g, &args[1], schema)?;
            let len = if args.len() == 3 {
                plan_expr(g, &args[2], schema)?
            } else {
                g.const_i64(i64::MAX)? // take remainder
            };
            Ok(g.substr(s, start, len)?)
        }
        "replace" => {
            check_arg_count(name, &args, 3)?;
            let s = plan_expr(g, &args[0], schema)?;
            let from = plan_expr(g, &args[1], schema)?;
            let to = plan_expr(g, &args[2], schema)?;
            Ok(g.replace(s, from, to)?)
        }
        "concat" => {
            if args.len() < 2 {
                return Err(SqlError::Plan(
                    "CONCAT requires at least 2 arguments".into(),
                ));
            }
            let cols: Vec<_> = args
                .iter()
                .map(|a| plan_expr(g, a, schema))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(g.concat(&cols)?)
        }

        // Date/time functions
        "current_date" => {
            // Microseconds since 2000-01-01 for start of today (UTC)
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            // Convert Unix seconds to Teide epoch (2000-01-01 = Unix 946684800)
            let teide_secs = now - 946684800;
            // Truncate to start of day, convert to microseconds
            let day_us = (teide_secs / 86400) * 86400 * 1_000_000;
            Ok(g.const_i64(day_us)?)
        }
        "current_timestamp" | "now" => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let unix_us = now.as_micros() as i64;
            // Convert to Teide epoch (2000-01-01 = Unix 946684800 seconds)
            let teide_us = unix_us - 946_684_800_000_000i64;
            Ok(g.const_i64(teide_us)?)
        }
        "extract" => {
            // EXTRACT can also be called as a function: extract('year', col)
            if args.len() != 2 {
                return Err(SqlError::Plan("EXTRACT() requires 2 arguments".into()));
            }
            let field_name = parse_field_arg(&args[0])?;
            let col = plan_expr(g, &args[1], schema)?;
            let field_id = resolve_field_name(&field_name)?;
            Ok(g.extract(col, field_id)?)
        }
        "date_trunc" => {
            // DATE_TRUNC('unit', timestamp)
            if args.len() != 2 {
                return Err(SqlError::Plan("DATE_TRUNC() requires 2 arguments".into()));
            }
            let field_name = parse_field_arg(&args[0])?;
            let col = plan_expr(g, &args[1], schema)?;
            let field_id = resolve_field_name(&field_name)?;
            Ok(g.date_trunc(col, field_id)?)
        }
        "date_diff" | "datediff" => {
            // DATE_DIFF('unit', start, end) → integer count of units between timestamps
            if args.len() != 3 {
                return Err(SqlError::Plan("DATE_DIFF() requires 3 arguments".into()));
            }
            let field_name = parse_field_arg(&args[0])?;
            let start = plan_expr(g, &args[1], schema)?;
            let end = plan_expr(g, &args[2], schema)?;

            match field_name.as_str() {
                "second" => {
                    let diff = g.sub(end, start)?;
                    let divisor = g.const_i64(1_000_000)?;
                    let fv = g.div(diff, divisor)?;
                    Ok(g.cast(g.floor(fv)?, crate::types::I64)?)
                }
                "minute" => {
                    let diff = g.sub(end, start)?;
                    let divisor = g.const_i64(60_000_000)?;
                    let fv = g.div(diff, divisor)?;
                    Ok(g.cast(g.floor(fv)?, crate::types::I64)?)
                }
                "hour" => {
                    let diff = g.sub(end, start)?;
                    let divisor = g.const_i64(3_600_000_000)?;
                    let fv = g.div(diff, divisor)?;
                    Ok(g.cast(g.floor(fv)?, crate::types::I64)?)
                }
                "day" => {
                    let diff = g.sub(end, start)?;
                    let divisor = g.const_i64(86_400_000_000)?;
                    let fv = g.div(diff, divisor)?;
                    Ok(g.cast(g.floor(fv)?, crate::types::I64)?)
                }
                "month" => {
                    // (year2*12+month2) - (year1*12+month1)
                    let y1 = g.extract(start, crate::extract_field::YEAR)?;
                    let m1 = g.extract(start, crate::extract_field::MONTH)?;
                    let y2 = g.extract(end, crate::extract_field::YEAR)?;
                    let m2 = g.extract(end, crate::extract_field::MONTH)?;
                    let twelve = g.const_i64(12)?;
                    let twelve2 = g.const_i64(12)?;
                    let ym1 = g.add(g.mul(y1, twelve)?, m1)?;
                    let ym2 = g.add(g.mul(y2, twelve2)?, m2)?;
                    Ok(g.sub(ym2, ym1)?)
                }
                "year" => {
                    let y1 = g.extract(start, crate::extract_field::YEAR)?;
                    let y2 = g.extract(end, crate::extract_field::YEAR)?;
                    Ok(g.sub(y2, y1)?)
                }
                _ => Err(SqlError::Plan(format!(
                    "Unsupported DATE_DIFF unit: {field_name}"
                ))),
            }
        }

        _ => Err(SqlError::Plan(format!("Unsupported function: {name}"))),
    }
}

/// Parse a field/unit argument from a function call (string literal or identifier).
fn parse_field_arg(arg: &Expr) -> Result<String, SqlError> {
    match arg {
        Expr::Value(Value::SingleQuotedString(s)) => Ok(s.to_lowercase()),
        Expr::Identifier(id) => Ok(id.value.to_lowercase()),
        _ => Err(SqlError::Plan(
            "Field/unit argument must be a string or identifier".into(),
        )),
    }
}

/// Resolve a field name string to a Teide extract_field constant.
fn resolve_field_name(name: &str) -> Result<i64, SqlError> {
    match name {
        "year" => Ok(crate::extract_field::YEAR),
        "month" => Ok(crate::extract_field::MONTH),
        "day" => Ok(crate::extract_field::DAY),
        "hour" => Ok(crate::extract_field::HOUR),
        "minute" => Ok(crate::extract_field::MINUTE),
        "second" => Ok(crate::extract_field::SECOND),
        "dow" | "dayofweek" => Ok(crate::extract_field::DOW),
        "doy" | "dayofyear" => Ok(crate::extract_field::DOY),
        "epoch" => Ok(crate::extract_field::EPOCH),
        _ => Err(SqlError::Plan(format!(
            "Unsupported date/time field: {name}"
        ))),
    }
}

fn extract_func_args(func: &Function) -> Result<Vec<Expr>, SqlError> {
    match &func.args {
        FunctionArguments::List(arg_list) => {
            let mut exprs = Vec::new();
            for arg in &arg_list.args {
                match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => exprs.push(e.clone()),
                    _ => {
                        return Err(SqlError::Plan(format!(
                            "Unsupported argument syntax in {}()",
                            func.name
                        )))
                    }
                }
            }
            Ok(exprs)
        }
        FunctionArguments::None => Ok(Vec::new()),
        _ => Err(SqlError::Plan(format!(
            "Unsupported argument syntax for '{}'",
            func.name
        ))),
    }
}

fn check_arg_count(name: &str, args: &[Expr], expected: usize) -> Result<(), SqlError> {
    if args.len() != expected {
        Err(SqlError::Plan(format!(
            "{name}() expects {expected} argument(s), got {}",
            args.len()
        )))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Type mapping
// ---------------------------------------------------------------------------

fn map_sql_type(dt: &DataType) -> Result<i8, SqlError> {
    match dt {
        DataType::Boolean | DataType::Bool => Ok(crate::types::BOOL),
        DataType::Int(None) | DataType::Integer(None) | DataType::Int4(_) => Ok(crate::types::I32),
        DataType::BigInt(None) | DataType::Int8(_) | DataType::Int64 => Ok(crate::types::I64),
        DataType::Float(None)
        | DataType::Float64
        | DataType::Double
        | DataType::DoublePrecision
        | DataType::Real => Ok(crate::types::F64),
        DataType::Varchar(_) | DataType::Text | DataType::String(_) => Ok(crate::types::SYM),
        DataType::Date => Ok(crate::types::DATE),
        DataType::Time(_, _) => Ok(crate::types::TIME),
        DataType::Timestamp(_, _) => Ok(crate::types::TIMESTAMP),
        _ => Err(SqlError::Plan(format!(
            "Unsupported CAST target type: {dt}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Aggregate helpers
// ---------------------------------------------------------------------------

/// Check if a function name is a known aggregate.
pub fn is_aggregate_name(name: &str) -> bool {
    matches!(
        name,
        "sum"
            | "avg"
            | "min"
            | "max"
            | "count"
            | "stddev"
            | "stddev_samp"
            | "stddev_pop"
            | "variance"
            | "var_samp"
            | "var_pop"
    )
}

/// Check if a function is COUNT(DISTINCT ...).
pub fn is_count_distinct(func: &Function) -> bool {
    let name = func.name.to_string().to_lowercase();
    if name != "count" {
        return false;
    }
    if let FunctionArguments::List(ref list) = func.args {
        matches!(list.duplicate_treatment, Some(DuplicateTreatment::Distinct))
    } else {
        false
    }
}

/// Check if a sqlparser Expr contains any aggregate function call.
/// Window functions (with OVER clause) are NOT counted as aggregates.
pub fn is_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Function(f) => {
            if f.over.is_some() {
                return false; // Window function, not a plain aggregate
            }
            let name = f.name.to_string().to_lowercase();
            is_aggregate_name(&name)
        }
        Expr::BinaryOp { left, right, .. } => is_aggregate(left) || is_aggregate(right),
        Expr::UnaryOp { expr, .. } => is_aggregate(expr),
        Expr::Nested(inner) => is_aggregate(inner),
        Expr::Cast { expr, .. } => is_aggregate(expr),
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            operand.as_ref().is_some_and(|e| is_aggregate(e))
                || conditions.iter().any(is_aggregate)
                || results.iter().any(is_aggregate)
                || else_result.as_ref().is_some_and(|e| is_aggregate(e))
        }
        _ => false,
    }
}

/// Check if an expression is a pure aggregate call (not wrapped in arithmetic).
/// Window functions (with OVER) are excluded.
pub fn is_pure_aggregate(expr: &Expr) -> bool {
    matches!(expr, Expr::Function(f) if f.over.is_none() && is_aggregate_name(&f.name.to_string().to_lowercase()))
}

/// Collect all aggregate sub-expressions from an expression tree.
/// Returns (agg_expr, auto_alias) pairs.
pub fn collect_aggregates(expr: &Expr) -> Vec<(&Expr, String)> {
    let mut aggs = Vec::new();
    collect_aggregates_inner(expr, &mut aggs);
    aggs
}

fn collect_aggregates_inner<'a>(expr: &'a Expr, aggs: &mut Vec<(&'a Expr, String)>) {
    match expr {
        Expr::Function(f) => {
            // Skip window functions (they have OVER clause)
            if f.over.is_some() {
                return;
            }
            let name = f.name.to_string().to_lowercase();
            if is_aggregate_name(&name) {
                let alias = format_agg_name(f);
                // Avoid duplicates
                if !aggs.iter().any(|(_, a)| *a == alias) {
                    aggs.push((expr, alias));
                }
                return; // Don't recurse into aggregate args
            }
            // Non-aggregate function: recurse into args
            if let FunctionArguments::List(arg_list) = &f.args {
                for arg in &arg_list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                        collect_aggregates_inner(e, aggs);
                    }
                }
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_aggregates_inner(left, aggs);
            collect_aggregates_inner(right, aggs);
        }
        Expr::UnaryOp { expr, .. } => collect_aggregates_inner(expr, aggs),
        Expr::Nested(inner) => collect_aggregates_inner(inner, aggs),
        Expr::Cast { expr, .. } => collect_aggregates_inner(expr, aggs),
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                collect_aggregates_inner(op, aggs);
            }
            for c in conditions {
                collect_aggregates_inner(c, aggs);
            }
            for r in results {
                collect_aggregates_inner(r, aggs);
            }
            if let Some(e) = else_result {
                collect_aggregates_inner(e, aggs);
            }
        }
        Expr::Between {
            expr, low, high, ..
        } => {
            collect_aggregates_inner(expr, aggs);
            collect_aggregates_inner(low, aggs);
            collect_aggregates_inner(high, aggs);
        }
        Expr::IsFalse(e) | Expr::IsTrue(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => {
            collect_aggregates_inner(e, aggs)
        }
        _ => {}
    }
}

/// Plan an expression in a post-aggregation context, where aggregate references
/// resolve to column scans on the GROUP BY result.
pub fn plan_post_agg_expr(
    g: &mut Graph,
    expr: &Expr,
    alias_to_native: &HashMap<String, String>,
) -> Result<Column, SqlError> {
    match expr {
        Expr::Function(f) => {
            let name = f.name.to_string().to_lowercase();
            if is_aggregate_name(&name) {
                // This aggregate should already be in the result — scan by native name
                let alias = format_agg_name(f);
                if let Some(native) = alias_to_native.get(&alias) {
                    return Ok(g.scan(native)?);
                }
                return Err(SqlError::Plan(format!(
                    "Aggregate '{alias}' not found in GROUP BY result"
                )));
            }
            // Non-aggregate function: resolve args recursively
            Err(SqlError::Plan(format!(
                "Non-aggregate function '{name}' in post-agg expression not yet supported"
            )))
        }
        Expr::BinaryOp { left, op, right } => {
            let l = plan_post_agg_expr(g, left, alias_to_native)?;
            let r = plan_post_agg_expr(g, right, alias_to_native)?;
            match op {
                BinaryOperator::Plus => Ok(g.add(l, r)?),
                BinaryOperator::Minus => Ok(g.sub(l, r)?),
                BinaryOperator::Multiply => Ok(g.mul(l, r)?),
                BinaryOperator::Divide => Ok(g.div(l, r)?),
                BinaryOperator::Modulo => Ok(g.modulo(l, r)?),
                BinaryOperator::Eq => Ok(g.eq(l, r)?),
                BinaryOperator::NotEq => Ok(g.ne(l, r)?),
                BinaryOperator::Lt => Ok(g.lt(l, r)?),
                BinaryOperator::LtEq => Ok(g.le(l, r)?),
                BinaryOperator::Gt => Ok(g.gt(l, r)?),
                BinaryOperator::GtEq => Ok(g.ge(l, r)?),
                BinaryOperator::And => Ok(g.and(l, r)?),
                BinaryOperator::Or => Ok(g.or(l, r)?),
                _ => Err(SqlError::Plan(format!("Unsupported operator: {op}"))),
            }
        }
        Expr::UnaryOp { op, expr: inner } => {
            let e = plan_post_agg_expr(g, inner, alias_to_native)?;
            match op {
                UnaryOperator::Not => Ok(g.not(e)?),
                UnaryOperator::Minus => Ok(g.neg(e)?),
                _ => Err(SqlError::Plan(format!("Unsupported unary operator: {op}"))),
            }
        }
        Expr::Nested(inner) => plan_post_agg_expr(g, inner, alias_to_native),
        Expr::Cast {
            expr: inner,
            data_type,
            ..
        } => {
            let e = plan_post_agg_expr(g, inner, alias_to_native)?;
            let target = map_sql_type(data_type)?;
            Ok(g.cast(e, target)?)
        }
        Expr::Value(val) => match val {
            Value::Number(n, _) => {
                if let Ok(i) = n.parse::<i64>() {
                    Ok(g.const_i64(i)?)
                } else {
                    let f = n
                        .parse::<f64>()
                        .map_err(|_| SqlError::Plan(format!("Invalid number literal: {n}")))?;
                    Ok(g.const_f64(f)?)
                }
            }
            Value::SingleQuotedString(s) => Ok(g.const_str(s)?),
            Value::Boolean(b) => Ok(g.const_bool(*b)?),
            Value::Null => Ok(g.const_f64(f64::NAN)?),
            _ => Err(SqlError::Plan(format!("Unsupported value: {val}"))),
        },
        Expr::Identifier(ident) => {
            // Could be a key column name or an aggregate alias
            let name = ident.value.to_lowercase();
            if let Some(native) = alias_to_native.get(&name) {
                return Ok(g.scan(native)?);
            }
            // Try as direct column name (key columns use native names)
            Ok(g.scan(&name)?)
        }
        _ => Err(SqlError::Plan(format!(
            "Unsupported expression in post-agg context: {expr}"
        ))),
    }
}

/// Plan a HAVING expression. Resolves aggregates via the result schema, then
/// scans using the native column name (the actual name in the result table).
/// `native_names[i]` is the real column name at index `i` in the result table.
pub fn plan_having_expr(
    g: &mut Graph,
    expr: &Expr,
    result_schema: &HashMap<String, usize>,
    original_schema: &HashMap<String, usize>,
    native_names: &[String],
) -> Result<Column, SqlError> {
    match expr {
        Expr::Function(f) => {
            let name = f.name.to_string().to_lowercase();
            if is_aggregate_name(&name) {
                // Try format_agg_name alias first ("sum(v1)")
                let alias = format_agg_name(f);
                if let Some(&idx) = result_schema.get(&alias) {
                    let col_name = &native_names[idx];
                    return Ok(g.scan(col_name)?);
                }
                // Try C engine naming convention ("v1_sum")
                if let Some(native) = predict_c_agg_name(f, original_schema) {
                    if let Some(&idx) = result_schema.get(&native) {
                        let col_name = &native_names[idx];
                        return Ok(g.scan(col_name)?);
                    }
                }
                return Err(SqlError::Plan(format!(
                    "Aggregate '{alias}' not found in GROUP BY result"
                )));
            }
            plan_expr(g, expr, result_schema)
        }
        Expr::BinaryOp { left, op, right } => {
            let l = plan_having_expr(g, left, result_schema, original_schema, native_names)?;
            let r = plan_having_expr(g, right, result_schema, original_schema, native_names)?;
            match op {
                BinaryOperator::Plus => Ok(g.add(l, r)?),
                BinaryOperator::Minus => Ok(g.sub(l, r)?),
                BinaryOperator::Multiply => Ok(g.mul(l, r)?),
                BinaryOperator::Divide => Ok(g.div(l, r)?),
                BinaryOperator::Modulo => Ok(g.modulo(l, r)?),
                BinaryOperator::Eq => Ok(g.eq(l, r)?),
                BinaryOperator::NotEq => Ok(g.ne(l, r)?),
                BinaryOperator::Lt => Ok(g.lt(l, r)?),
                BinaryOperator::LtEq => Ok(g.le(l, r)?),
                BinaryOperator::Gt => Ok(g.gt(l, r)?),
                BinaryOperator::GtEq => Ok(g.ge(l, r)?),
                BinaryOperator::And => Ok(g.and(l, r)?),
                BinaryOperator::Or => Ok(g.or(l, r)?),
                _ => Err(SqlError::Plan(format!("Unsupported operator: {op}"))),
            }
        }
        Expr::UnaryOp { op, expr: inner } => {
            let e = plan_having_expr(g, inner, result_schema, original_schema, native_names)?;
            match op {
                UnaryOperator::Not => Ok(g.not(e)?),
                UnaryOperator::Minus => Ok(g.neg(e)?),
                _ => Err(SqlError::Plan(format!("Unsupported unary operator: {op}"))),
            }
        }
        Expr::Nested(inner) => {
            plan_having_expr(g, inner, result_schema, original_schema, native_names)
        }
        Expr::Value(_) => plan_expr(g, expr, result_schema),
        Expr::Identifier(ident) => {
            let name = ident.value.to_lowercase();
            if let Some(&idx) = result_schema.get(&name) {
                let col_name = &native_names[idx];
                return Ok(g.scan(col_name)?);
            }
            Err(SqlError::Plan(format!(
                "Column '{}' not found in HAVING result",
                name
            )))
        }
        _ => plan_expr(g, expr, result_schema),
    }
}

/// Predict the C engine's naming convention for an aggregate output column.
/// SUM(v1) → "v1_sum", COUNT(v1) → "v1_count", AVG(v1) → "v1_mean", etc.
pub fn predict_c_agg_name(
    func: &Function,
    original_schema: &HashMap<String, usize>,
) -> Option<String> {
    let op = func.name.to_string().to_lowercase();
    let suffix = match op.as_str() {
        "sum" => "_sum",
        "count" => "_count",
        "avg" => "_mean",
        "min" => "_min",
        "max" => "_max",
        "stddev" | "stddev_samp" => "_stddev",
        "stddev_pop" => "_stddev_pop",
        "variance" | "var_samp" => "_var",
        "var_pop" => "_var_pop",
        _ => return None,
    };
    if let FunctionArguments::List(args) = &func.args {
        if let Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Identifier(ident)))) =
            args.args.first()
        {
            return Some(format!("{}{}", ident.value.to_lowercase(), suffix));
        }
        if let Some(FunctionArg::Unnamed(FunctionArgExpr::Wildcard)) = args.args.first() {
            // COUNT(*) uses first column as proxy
            let first = original_schema
                .iter()
                .min_by_key(|(_, v)| **v)
                .map(|(k, _)| k.clone())?;
            return Some(format!("{}{}", first, suffix));
        }
    }
    None
}

/// Extract the column name or plan expression from an aggregate function's argument.
/// When the function has a FILTER clause, the input is wrapped in a conditional
/// and the AggOp may be adjusted (e.g. COUNT → SUM for filtered count).
/// Returns (possibly-adjusted AggOp, planned input column).
pub fn plan_agg_input(
    g: &mut Graph,
    func: &Function,
    op: AggOp,
    schema: &HashMap<String, usize>,
) -> Result<(AggOp, Column), SqlError> {
    let name = func.name.to_string().to_lowercase();

    let args = match &func.args {
        FunctionArguments::List(arg_list) => &arg_list.args,
        FunctionArguments::None => {
            return Err(SqlError::Plan(format!(
                "Function '{}' requires an argument",
                func.name
            )));
        }
        _ => {
            return Err(SqlError::Plan(format!(
                "Unsupported function argument syntax for '{}'",
                func.name
            )));
        }
    };

    if args.len() != 1 {
        return Err(SqlError::Plan(format!(
            "Function '{}' expects 1 argument, got {}",
            func.name,
            args.len()
        )));
    }

    let input = match &args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
            // Plan arbitrary expression as aggregate input
            plan_expr(g, expr, schema)?
        }
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
            // COUNT(*) — use first column as proxy
            if name != "count" {
                return Err(SqlError::Plan(format!(
                    "Wildcard (*) not supported for {name}()"
                )));
            }
            let first_col = schema
                .iter()
                .min_by_key(|(_k, v)| **v)
                .map(|(k, _)| k.clone())
                .ok_or_else(|| SqlError::Plan("COUNT(*) on empty schema".into()))?;
            g.scan(&first_col)?
        }
        _ => {
            return Err(SqlError::Plan(format!(
                "Only expressions and * supported as arguments to {}()",
                func.name
            )));
        }
    };

    // FILTER (WHERE cond): rewrite aggregate input so the C engine only
    // accumulates matching rows.  The rewrite depends on the aggregate type
    // because the C engine does NOT universally skip NaN:
    //   SUM      → IF(cond, CAST(x, F64), 0.0)     (0 is additive identity)
    //   MIN/MAX  → IF(cond, CAST(x, F64), NaN)      (NaN loses all comparisons)
    //   COUNT    → SUM(IF(cond, 1.0, 0.0))           (turn count into sum-of-ones)
    //   AVG      → not supported (would need two separate aggregates)
    if let Some(ref filter_expr) = func.filter {
        if op == AggOp::Avg {
            return Err(SqlError::Plan(
                "AVG(...) FILTER is not yet supported; use \
                 SUM(x) FILTER(...) / COUNT(x) FILTER(...) instead"
                    .into(),
            ));
        }

        let pred = plan_expr(g, filter_expr, schema)?;

        match op {
            AggOp::Count => {
                // COUNT FILTER → SUM of indicator: IF(cond, 1.0, 0.0)
                let one = g.const_f64(1.0)?;
                let zero = g.const_f64(0.0)?;
                Ok((AggOp::Sum, g.if_then_else(pred, one, zero)?))
            }
            AggOp::Sum => {
                // Zero-fill filtered rows (0 is neutral for addition)
                let input_f64 = g.cast(input, crate::types::F64)?;
                let zero = g.const_f64(0.0)?;
                Ok((op, g.if_then_else(pred, input_f64, zero)?))
            }
            AggOp::Min | AggOp::Max => {
                // NaN-fill filtered rows (NaN loses all < and > comparisons)
                let input_f64 = g.cast(input, crate::types::F64)?;
                let nan = g.const_f64(f64::NAN)?;
                Ok((op, g.if_then_else(pred, input_f64, nan)?))
            }
            AggOp::Avg => unreachable!(), // handled above
            _ => Err(SqlError::Plan(format!(
                "FILTER clause not supported for {}()",
                func.name
            ))),
        }
    } else {
        Ok((op, input))
    }
}

/// Map aggregate function name to AggOp.
pub fn agg_op_from_name(name: &str) -> Result<AggOp, SqlError> {
    match name {
        "sum" => Ok(AggOp::Sum),
        "avg" => Ok(AggOp::Avg),
        "min" => Ok(AggOp::Min),
        "max" => Ok(AggOp::Max),
        "count" => Ok(AggOp::Count),
        "stddev" | "stddev_samp" => Ok(AggOp::Stddev),
        "stddev_pop" => Ok(AggOp::StddevPop),
        "variance" | "var_samp" => Ok(AggOp::Var),
        "var_pop" => Ok(AggOp::VarPop),
        _ => Err(SqlError::Plan(format!(
            "Unknown aggregate function: {name}"
        ))),
    }
}

/// Generate a default name for an aggregate function expression.
pub fn format_agg_name(func: &Function) -> String {
    let fname = func.name.to_string().to_lowercase();
    let arg_str = match &func.args {
        FunctionArguments::List(args) => {
            if args.args.is_empty() {
                "*".to_string()
            } else {
                args.args
                    .iter()
                    .map(|a| format!("{a}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
        _ => "*".to_string(),
    };
    let base = format!("{fname}({arg_str})");
    if let Some(ref filter) = func.filter {
        format!("{base}_filter_{filter}")
    } else {
        base
    }
}

/// Get a default display name for a bare expression.
pub fn expr_default_name(expr: &Expr) -> String {
    match expr {
        Expr::Identifier(ident) => ident.value.to_lowercase(),
        Expr::CompoundIdentifier(parts) => {
            // Return just the column name, not the full qualified name
            parts
                .last()
                .map(|p| p.value.to_lowercase())
                .unwrap_or_default()
        }
        // Parenthesized identifiers should keep the underlying column name.
        Expr::Nested(inner) => expr_default_name(inner),
        _ => format!("{expr}").to_lowercase(),
    }
}

// ---------------------------------------------------------------------------
// Window function helpers
// ---------------------------------------------------------------------------

/// Check if a SQL expression is a window function call (has OVER clause).
pub fn is_window_function(expr: &Expr) -> bool {
    matches!(expr, Expr::Function(f) if f.over.is_some())
}

/// Check if any SELECT item contains a window function.
pub fn has_window_functions(items: &[SelectItem]) -> bool {
    items.iter().any(|item| match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
            contains_window_function(e)
        }
        _ => false,
    })
}

fn contains_window_function(expr: &Expr) -> bool {
    match expr {
        Expr::Function(f) if f.over.is_some() => true,
        Expr::BinaryOp { left, right, .. } => {
            contains_window_function(left) || contains_window_function(right)
        }
        Expr::UnaryOp { expr, .. } => contains_window_function(expr),
        Expr::Nested(inner) => contains_window_function(inner),
        Expr::Cast { expr, .. } => contains_window_function(expr),
        _ => false,
    }
}

/// Info about a single window function call extracted from a SELECT item.
pub struct WindowFuncInfo {
    pub func_name: String,
    pub func: WindowFunc,
    pub input_expr: Option<Expr>,
    pub spec: WindowSpec,
    pub display_name: String,
}

/// Map a SQL function name + args to a Teide WindowFunc.
pub fn window_func_from_name(name: &str, args: &[Expr]) -> Result<WindowFunc, SqlError> {
    match name {
        "row_number" => Ok(WindowFunc::RowNumber),
        "rank" => Ok(WindowFunc::Rank),
        "dense_rank" => Ok(WindowFunc::DenseRank),
        "ntile" => {
            if args.len() != 1 {
                return Err(SqlError::Plan("NTILE requires exactly 1 argument".into()));
            }
            let n = parse_i64_literal(&args[0])?;
            Ok(WindowFunc::Ntile(n))
        }
        "sum" => Ok(WindowFunc::Sum),
        "avg" => Ok(WindowFunc::Avg),
        "min" => Ok(WindowFunc::Min),
        "max" => Ok(WindowFunc::Max),
        "count" => Ok(WindowFunc::Count),
        "lag" => {
            let offset = if args.len() >= 2 {
                parse_i64_literal(&args[1])?
            } else {
                1
            };
            Ok(WindowFunc::Lag(offset))
        }
        "lead" => {
            let offset = if args.len() >= 2 {
                parse_i64_literal(&args[1])?
            } else {
                1
            };
            Ok(WindowFunc::Lead(offset))
        }
        "first_value" => Ok(WindowFunc::FirstValue),
        "last_value" => Ok(WindowFunc::LastValue),
        "nth_value" => {
            if args.len() != 2 {
                return Err(SqlError::Plan(
                    "NTH_VALUE requires exactly 2 arguments".into(),
                ));
            }
            let n = parse_i64_literal(&args[1])?;
            Ok(WindowFunc::NthValue(n))
        }
        _ => Err(SqlError::Plan(format!("Unknown window function: {name}"))),
    }
}

fn parse_i64_literal(expr: &Expr) -> Result<i64, SqlError> {
    match expr {
        Expr::Value(Value::Number(n, _)) => n
            .parse::<i64>()
            .map_err(|_| SqlError::Plan(format!("Expected integer literal, got: {n}"))),
        _ => Err(SqlError::Plan(format!(
            "Expected integer literal, got: {expr}"
        ))),
    }
}

/// Numeric ordering for frame bounds (lower = earlier in the window).
fn frame_bound_order(b: &FrameBound) -> i64 {
    match b {
        FrameBound::UnboundedPreceding => i64::MIN,
        FrameBound::Preceding(n) => -n,
        FrameBound::CurrentRow => 0,
        FrameBound::Following(n) => *n,
        FrameBound::UnboundedFollowing => i64::MAX,
    }
}

/// Convert sqlparser WindowSpec to Teide FrameType + FrameBound pair.
pub fn parse_window_frame(
    spec: &WindowSpec,
) -> Result<(FrameType, FrameBound, FrameBound), SqlError> {
    match &spec.window_frame {
        Some(frame) => {
            let ft = match frame.units {
                WindowFrameUnits::Rows => FrameType::Rows,
                _ => FrameType::Range,
            };
            let start = convert_frame_bound(&frame.start_bound)?;
            let end = match &frame.end_bound {
                Some(b) => convert_frame_bound(b)?,
                None => FrameBound::CurrentRow,
            };
            if frame_bound_order(&start) > frame_bound_order(&end) {
                return Err(SqlError::Plan(
                    "Window frame start must not be after frame end".into(),
                ));
            }
            Ok((ft, start, end))
        }
        None => {
            // SQL default: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            // But with no ORDER BY, it's the whole partition
            if spec.order_by.is_empty() {
                Ok((
                    FrameType::Range,
                    FrameBound::UnboundedPreceding,
                    FrameBound::UnboundedFollowing,
                ))
            } else {
                Ok((
                    FrameType::Range,
                    FrameBound::UnboundedPreceding,
                    FrameBound::CurrentRow,
                ))
            }
        }
    }
}

fn convert_frame_bound(b: &WindowFrameBound) -> Result<FrameBound, SqlError> {
    match b {
        WindowFrameBound::CurrentRow => Ok(FrameBound::CurrentRow),
        WindowFrameBound::Preceding(None) => Ok(FrameBound::UnboundedPreceding),
        WindowFrameBound::Following(None) => Ok(FrameBound::UnboundedFollowing),
        WindowFrameBound::Preceding(Some(expr)) => {
            let n = parse_i64_literal(expr)?;
            Ok(FrameBound::Preceding(n))
        }
        WindowFrameBound::Following(Some(expr)) => {
            let n = parse_i64_literal(expr)?;
            Ok(FrameBound::Following(n))
        }
    }
}

/// Generate display name for a window function expression.
pub fn format_window_name(func: &Function) -> String {
    let fname = func.name.to_string().to_lowercase();
    let arg_str = match &func.args {
        FunctionArguments::List(args) => {
            if args.args.is_empty() {
                String::new()
            } else {
                args.args
                    .iter()
                    .map(|a| format!("{a}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
        _ => String::new(),
    };
    format!("{fname}({arg_str})")
}

/// Collect all window function calls from SELECT items.
/// Returns (select_item_index, WindowFuncInfo) pairs.
pub fn collect_window_functions(
    items: &[SelectItem],
) -> Result<Vec<(usize, WindowFuncInfo)>, SqlError> {
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let expr = match item {
            SelectItem::UnnamedExpr(e) => e,
            SelectItem::ExprWithAlias { expr: e, .. } => e,
            _ => continue,
        };
        collect_win_funcs_inner(expr, i, &mut result)?;
    }
    Ok(result)
}

fn collect_win_funcs_inner(
    expr: &Expr,
    item_idx: usize,
    out: &mut Vec<(usize, WindowFuncInfo)>,
) -> Result<(), SqlError> {
    match expr {
        Expr::Function(f) => {
            let Some(over) = f.over.as_ref() else {
                return Ok(());
            };
            let name = f.name.to_string().to_lowercase();
            let args = extract_func_args(f)?;

            // Determine the WindowFunc variant
            let wf = window_func_from_name(&name, &args)?;

            // Extract window spec
            let spec = match over {
                WindowType::WindowSpec(s) => s.clone(),
                WindowType::NamedWindow(_) => {
                    return Err(SqlError::Plan("Named windows not supported".into()));
                }
            };

            // Input expression (first arg for agg/value funcs, None for rank funcs)
            let input_expr = match &wf {
                WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank => None,
                WindowFunc::Ntile(_) => None,
                WindowFunc::Lag(_) | WindowFunc::Lead(_) => args.first().cloned(),
                _ => args.first().cloned(),
            };

            let display = format_window_name(f);

            out.push((
                item_idx,
                WindowFuncInfo {
                    func_name: name,
                    func: wf,
                    input_expr,
                    spec,
                    display_name: display,
                },
            ));
            Ok(())
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_win_funcs_inner(left, item_idx, out)?;
            collect_win_funcs_inner(right, item_idx, out)
        }
        Expr::UnaryOp { expr, .. } => collect_win_funcs_inner(expr, item_idx, out),
        Expr::Nested(inner) => collect_win_funcs_inner(inner, item_idx, out),
        Expr::Cast { expr, .. } => collect_win_funcs_inner(expr, item_idx, out),
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                collect_win_funcs_inner(op, item_idx, out)?;
            }
            for c in conditions {
                collect_win_funcs_inner(c, item_idx, out)?;
            }
            for r in results {
                collect_win_funcs_inner(r, item_idx, out)?;
            }
            if let Some(e) = else_result {
                collect_win_funcs_inner(e, item_idx, out)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
