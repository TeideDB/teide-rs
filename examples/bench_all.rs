/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

use std::time::Instant;

fn bench(session: &mut teide::sql::Session, label: &str, sql: &str) {
    for _ in 0..3 {
        session.execute(sql).unwrap();
    }
    let mut times = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        let _r = session.execute(sql).unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!("{:20}: {:7.1} ms", label, med * 1000.0);
}

fn main() {
    let base = std::env::var("BENCH_DIR").unwrap_or_else(|_| {
        std::fs::canonicalize("../rayforce-bench/datasets")
            .expect("bench datasets dir not found")
            .to_string_lossy()
            .to_string()
    });
    let mut s = teide::sql::Session::new().unwrap();

    // --- Groupby ---
    let csv_g = format!("{}/G1_1e7_1e2_0_0/G1_1e7_1e2_0_0.csv", base);
    s.execute(&format!(
        "CREATE TABLE g AS SELECT * FROM read_csv('{csv_g}')"
    ))
    .unwrap();
    println!("=== Groupby (10M rows) ===");
    bench(
        &mut s,
        "q1",
        "SELECT id1, SUM(v1) AS v1 FROM g GROUP BY id1",
    );
    bench(
        &mut s,
        "q2",
        "SELECT id1, id2, SUM(v1) AS v1 FROM g GROUP BY id1, id2",
    );
    bench(
        &mut s,
        "q3",
        "SELECT id3, SUM(v1) AS v1, AVG(v3) AS v3 FROM g GROUP BY id3",
    );
    bench(
        &mut s,
        "q4",
        "SELECT id4, AVG(v1) AS v1, AVG(v2) AS v2, AVG(v3) AS v3 FROM g GROUP BY id4",
    );
    bench(
        &mut s,
        "q5",
        "SELECT id6, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM g GROUP BY id6",
    );
    bench(
        &mut s,
        "q6",
        "SELECT id3, MAX(v1) - MIN(v2) AS range_v1_v2 FROM g GROUP BY id3",
    );
    bench(&mut s, "q7", "SELECT id1, id2, id3, id4, id5, id6, SUM(v3) AS v3, COUNT(*) AS cnt FROM g GROUP BY id1, id2, id3, id4, id5, id6");

    // --- Sort ---
    println!("\n=== Sort (10M rows) ===");
    bench(&mut s, "s1", "SELECT * FROM g ORDER BY id1");
    bench(&mut s, "s2", "SELECT * FROM g ORDER BY id3");
    bench(&mut s, "s3", "SELECT * FROM g ORDER BY id4");
    bench(&mut s, "s4", "SELECT * FROM g ORDER BY v3 DESC");
    bench(&mut s, "s5", "SELECT * FROM g ORDER BY id1, id2");
    bench(&mut s, "s6", "SELECT * FROM g ORDER BY id1, id2, id3");

    // --- Join ---
    let csv_big = format!("{}/h2oai_join_1e7/J1_1e7_NA_0_0.csv", base);
    let csv_small = format!("{}/h2oai_join_1e7/J1_1e7_1e7_0_0.csv", base);
    s.execute(&format!(
        "CREATE TABLE x AS SELECT * FROM read_csv('{csv_big}')"
    ))
    .unwrap();
    s.execute(&format!(
        "CREATE TABLE small AS SELECT * FROM read_csv('{csv_small}')"
    ))
    .unwrap();
    println!("\n=== Join (10M rows) ===");
    bench(&mut s, "j1-inner", "SELECT x.id1, x.id2, x.id3, x.v1, small.v2 FROM x INNER JOIN small ON x.id1 = small.id1 AND x.id2 = small.id2 AND x.id3 = small.id3");
    bench(&mut s, "j2-left", "SELECT x.id1, x.id2, x.id3, x.v1, small.v2 FROM x LEFT JOIN small ON x.id1 = small.id1 AND x.id2 = small.id2 AND x.id3 = small.id3");

    // --- Window ---
    println!("\n=== Window (10M rows) ===");
    bench(
        &mut s,
        "w1",
        "SELECT id1, v1, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1) AS rn FROM g",
    );
    bench(
        &mut s,
        "w2",
        "SELECT id1, id4, RANK() OVER (PARTITION BY id1 ORDER BY id4) AS rnk FROM g",
    );
    bench(
        &mut s,
        "w3",
        "SELECT id3, v1, SUM(v1) OVER (PARTITION BY id3 ORDER BY v1) AS csum FROM g",
    );
    bench(
        &mut s,
        "w4",
        "SELECT id1, v1, LAG(v1, 1) OVER (PARTITION BY id1 ORDER BY v1) AS lag_v1 FROM g",
    );
    bench(
        &mut s,
        "w5",
        "SELECT id1, v1, AVG(v1) OVER (PARTITION BY id1) AS avg_v1 FROM g",
    );
    bench(
        &mut s,
        "w6",
        "SELECT id1, id2, v1, ROW_NUMBER() OVER (PARTITION BY id1, id2 ORDER BY v1) AS rn FROM g",
    );
}
