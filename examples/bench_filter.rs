use std::time::Instant;

fn main() {
    let csv = std::env::var("CSV").expect("set CSV=path");
    let mut s = teide::sql::Session::new().unwrap();
    s.execute(&format!("CREATE TABLE t AS SELECT * FROM '{csv}'"))
        .unwrap();

    let queries: Vec<(&str, &str)> = vec![
        (
            "filter+group",
            "SELECT id1 FROM t WHERE v2 = 10 AND id2 = 'id080' GROUP BY id1",
        ),
        (
            "filter+sort",
            "SELECT id1, id2, v3, v2 FROM t WHERE v2 = 10 AND id2 = 'id080' ORDER BY id1",
        ),
        (
            "q4",
            "SELECT id4, AVG(v1) AS v1, AVG(v2) AS v2, AVG(v3) AS v3 FROM t GROUP BY id4",
        ),
        (
            "q5",
            "SELECT id6, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM t GROUP BY id6",
        ),
        // Isolate filter cost: just SELECT with WHERE, no group/sort
        (
            "filter_only",
            "SELECT id1, id2, v3, v2 FROM t WHERE v2 = 10 AND id2 = 'id080'",
        ),
        // q5 variant: force HT path by using string key
        (
            "q5_str_key",
            "SELECT id3, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM t GROUP BY id3",
        ),
    ];

    for (label, sql) in &queries {
        for _ in 0..3 {
            s.execute(sql).unwrap();
        }
        let mut times = Vec::new();
        for _ in 0..7 {
            let t0 = Instant::now();
            let _r = s.execute(sql).unwrap();
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        println!("{:20}: {:7.1} ms", label, med * 1000.0);
    }
}
