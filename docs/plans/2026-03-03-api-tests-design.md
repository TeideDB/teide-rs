# Full Functional API Tests Design

## Goal

Fill test coverage gaps across all three API layers (Engine, SQL, Server) with ~103 new tests in 3 new test files.

## Files

### `tests/engine_api.rs` (~52 tests)

Covers untested Graph, Table, Rel, and utility APIs.

- **Graph constants**: `const_f64`, `const_bool`, `const_str`, `const_table`
- **Binary/comparison ops**: `sub`, `mul`, `div`, `modulo`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `and`, `or`
- **Unary/math ops**: `not`, `negate`, `sqrt`, `log`, `exp`, `ceil`, `floor`, `is_null`, `is_not_null`
- **String ops**: `upper`, `lower`, `strlen`, `trim`, `substr`, `replace`, `like`, `ilike`
- **Aggregates**: `avg`, `min_op`, `max_op`, `count`, `count_distinct`, `distinct`, `min2`, `max2`, `if_then_else`
- **Temporal**: `extract`, `date_trunc`
- **Table methods**: `pick_columns`, `with_column_names`, `write_csv`, `get_bool`
- **Utility functions**: `sym_intern`, `mem_stats`, `format_date`, `format_time`, `format_timestamp`
- **Graph traversal/CSR**: `Rel::build`, `Rel::from_edges`, `expand`, `var_expand`, `shortest_path`

### `tests/sql_extended.rs` (~34 tests)

Covers untested SQL session ops, NULL handling, error paths, and edge cases.

- **Session ops**: `execute_script`, `execute_script_file`, `table_names`, `table_info`
- **NULL handling**: IS NULL/IS NOT NULL, NULL arithmetic, NULL in aggregates, NULL in GROUP BY, NULL comparisons, COALESCE, NULLIF
- **Error paths**: division by zero, type mismatch, unknown function, duplicate column, INSERT type mismatch, ambiguous column
- **Edge cases**: empty table ops, LIMIT 0, OFFSET > row count, single-row aggregation, SELECT without FROM, DISTINCT single column
- **Untested SQL features**: SUBSTRING, TRIM, REPLACE, ABS, ROUND, CAST variants, multiple CTEs, RIGHT JOIN, self-join

### `tests/server_extended.rs` (~17 tests)

Covers untested server session management, catalog, and edge cases.

- **Multi-query sessions**: multiple SELECTs, CREATE+SELECT, SET persistence, DDL+DML
- **Catalog queries**: information_schema.columns, schemata, pg_catalog
- **Result edge cases**: empty results, NULLs, large results, all column types, single-column
- **Error responses**: parse error, unknown table, unknown column
- **Concurrent connections**: simultaneous queries, client disconnect

## Test Infrastructure

All tests follow the existing ENGINE_LOCK pattern. Server tests use the existing `start_server`/`connect` helpers from `tests/server.rs`.

## Constraints

- CSR/Graph traversal tests need edge table construction — may need to create CSV with foreign-key relationships
- Server tests require `server` feature and spawn child processes
- Some tests may need to be `#[ignore]`-gated if they depend on features not always available
