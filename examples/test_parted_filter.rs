use std::time::Instant;
use teide::sql::{ExecResult, Session};

fn print_result(r: &ExecResult) {
    match r {
        ExecResult::Query(result) => {
            let t = &result.table;
            let nrows = t.nrows() as usize;
            let ncols = t.ncols() as usize;
            println!("  {} rows x {} cols", nrows, ncols);
            let show = if nrows > 10 { 10 } else { nrows };
            for row in 0..show {
                let mut cells = Vec::new();
                for col in 0..ncols {
                    let typ = t.col_type(col);
                    let cell = match typ {
                        7 => t
                            .get_f64(col, row)
                            .map(|v| format!("{v:.4}"))
                            .unwrap_or("NULL".into()),
                        4..=6 => t
                            .get_i64(col, row)
                            .map(|v| format!("{v}"))
                            .unwrap_or("NULL".into()),
                        15 | 20 => t
                            .get_str(col, row)
                            .map(|s| s.to_string())
                            .unwrap_or("NULL".into()),
                        _ => format!("?t{typ}"),
                    };
                    cells.push(cell);
                }
                println!("  [{}] {}", row, cells.join(" | "));
            }
        }
        ExecResult::Ddl(msg) => println!("  {msg}"),
    }
}

fn run_query(s: &mut Session, label: &str, sql: &str) {
    println!("\n--- {} ---", label);
    println!("  SQL: {}", sql);
    let t0 = Instant::now();
    match s.execute(sql) {
        Ok(r) => {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_result(&r);
            println!("  Time: {:.1} ms", ms);
        }
        Err(e) => {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  ERROR: {:?} ({:.1} ms)", e, ms);
        }
    }
}

fn main() {
    let mut s = Session::new().unwrap();

    // Mount the 500M-row partitioned table (lazy mmap)
    let t0 = Instant::now();
    let r = s
        .execute("CREATE TABLE t AS SELECT * FROM '/tmp/db/quotes'")
        .unwrap();
    let mount_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Mount: {:.1} ms", mount_ms);
    print_result(&r);

    // Aggregates with filter
    run_query(
        &mut s,
        "count_star_filtered",
        "SELECT count(*) FROM t WHERE v2 > 5",
    );
    run_query(
        &mut s,
        "count_v1_filtered",
        "SELECT count(v1) FROM t WHERE v2 > 5",
    );
    run_query(
        &mut s,
        "sum_v1_filtered",
        "SELECT sum(v1) FROM t WHERE v2 > 5",
    );
    run_query(
        &mut s,
        "avg_v1_filtered",
        "SELECT avg(v1) FROM t WHERE v2 > 5",
    );

    // Projection with LIMIT
    run_query(&mut s, "select_star_limit", "SELECT * FROM t LIMIT 5");
    run_query(&mut s, "select_v1_v2_limit", "SELECT v1, v2 FROM t LIMIT 5");
}
