use std::time::Instant;

fn bench(session: &mut teide::sql::Session, label: &str, sql: &str, n_warmup: u32, n_iter: u32) {
    for _ in 0..n_warmup {
        session.execute(sql).unwrap();
    }
    let mut times = Vec::new();
    for _ in 0..n_iter {
        let t0 = Instant::now();
        let _r = session.execute(sql).unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    println!(
        "{:20}: med {:6.1} ms  min {:6.1} ms  max {:6.1} ms",
        label,
        med * 1000.0,
        min * 1000.0,
        max * 1000.0
    );
}

fn main() {
    let base = std::env::var("BENCH_DIR").unwrap_or_else(|_| {
        std::fs::canonicalize("../rayforce-bench/datasets")
            .expect("bench datasets dir not found")
            .to_string_lossy()
            .to_string()
    });
    let mut s = teide::sql::Session::new().unwrap();
    let csv_g = format!("{}/G1_1e7_1e2_0_0/G1_1e7_1e2_0_0.csv", base);
    s.execute(&format!(
        "CREATE TABLE g AS SELECT * FROM read_csv('{csv_g}')"
    ))
    .unwrap();

    println!("=== q1 (50 warmup, 51 iterations) ===");
    bench(
        &mut s,
        "q1",
        "SELECT id1, SUM(v1) AS v1 FROM g GROUP BY id1",
        50,
        51,
    );
    bench(
        &mut s,
        "q2",
        "SELECT id1, id2, SUM(v1) AS v1 FROM g GROUP BY id1, id2",
        20,
        31,
    );
}
