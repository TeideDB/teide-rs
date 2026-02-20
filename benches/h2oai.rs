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

//! Groupby benchmark suite for Teide SQL engine (criterion).
//!
//! Requires the 10M-row CSV datasets at `../rayforce-bench/datasets/`
//! relative to the workspace root. Benchmarks are silently skipped if CSVs
//! are not found.

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use std::path::PathBuf;
use std::time::Duration;

use teide::sql::Session;

fn bench_dataset_dir() -> Option<PathBuf> {
    std::env::var("TEIDE_BENCH_DATA").ok().map(PathBuf::from)
}

fn groupby_csv() -> Option<String> {
    let p = bench_dataset_dir()?.join("G1_1e7_1e2_0_0/G1_1e7_1e2_0_0.csv");
    if !p.exists() {
        eprintln!("CSV not found: {}", p.display());
        return None;
    }
    Some(p.to_str().unwrap().to_string())
}

fn join_csv_x() -> Option<String> {
    let p = bench_dataset_dir()?.join("h2oai_join_1e7/J1_1e7_NA_0_0.csv");
    p.exists().then(|| p.to_str().unwrap().to_string())
}

fn join_csv_y() -> Option<String> {
    let p = bench_dataset_dir()?.join("h2oai_join_1e7/J1_1e7_1e7_0_0.csv");
    p.exists().then(|| p.to_str().unwrap().to_string())
}

fn setup_groupby() -> Option<Session> {
    let csv = groupby_csv()?;
    let mut session = Session::new().ok()?;
    session
        .execute(&format!("CREATE TABLE t AS SELECT * FROM '{csv}'"))
        .ok()?;
    Some(session)
}

fn setup_join() -> Option<Session> {
    let x = join_csv_x()?;
    let y = join_csv_y()?;
    let mut session = Session::new().ok()?;
    session
        .execute(&format!("CREATE TABLE x AS SELECT * FROM '{x}'"))
        .ok()?;
    session
        .execute(&format!("CREATE TABLE y AS SELECT * FROM '{y}'"))
        .ok()?;
    Some(session)
}

// ---------------------------------------------------------------------------
// Groupby benchmarks (q1-q7)
// ---------------------------------------------------------------------------

fn bench_groupby(c: &mut Criterion) {
    let mut session = match setup_groupby() {
        Some(s) => s,
        None => {
            eprintln!("Skipping groupby benchmarks: set TEIDE_BENCH_DATA to dataset dir");
            return;
        }
    };

    let mut group = c.benchmark_group("groupby");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("q1", |b| {
        b.iter(|| {
            session
                .execute("SELECT id1, SUM(v1) as v1 FROM t GROUP BY id1")
                .unwrap()
        })
    });

    group.bench_function("q2", |b| {
        b.iter(|| {
            session
                .execute("SELECT id1, id2, SUM(v1) as v1 FROM t GROUP BY id1, id2")
                .unwrap()
        })
    });

    group.bench_function("q3", |b| {
        b.iter(|| {
            session
                .execute("SELECT id3, SUM(v1) as v1, AVG(v3) as v3 FROM t GROUP BY id3")
                .unwrap()
        })
    });

    group.bench_function("q4", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id4, AVG(v1) as v1, AVG(v2) as v2, AVG(v3) as v3 FROM t GROUP BY id4",
                )
                .unwrap()
        })
    });

    group.bench_function("q5", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id6, SUM(v1) as v1, SUM(v2) as v2, SUM(v3) as v3 FROM t GROUP BY id6",
                )
                .unwrap()
        })
    });

    group.bench_function("q6", |b| {
        b.iter(|| {
            session
                .execute("SELECT id3, MAX(v1) as v1, MIN(v2) as v2 FROM t GROUP BY id3")
                .unwrap()
        })
    });

    group.bench_function("q7", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id1, id2, id3, id4, id5, id6, SUM(v3) as v3, COUNT(v1) as cnt \
                     FROM t GROUP BY id1, id2, id3, id4, id5, id6",
                )
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Sort benchmarks (s1-s6)
// ---------------------------------------------------------------------------

fn bench_sort(c: &mut Criterion) {
    let mut session = match setup_groupby() {
        Some(s) => s,
        None => {
            eprintln!("Skipping sort benchmarks: set TEIDE_BENCH_DATA to dataset dir");
            return;
        }
    };

    let mut group = c.benchmark_group("sort");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("s1", |b| {
        b.iter(|| session.execute("SELECT * FROM t ORDER BY id1").unwrap())
    });

    group.bench_function("s2", |b| {
        b.iter(|| session.execute("SELECT * FROM t ORDER BY id3").unwrap())
    });

    group.bench_function("s3", |b| {
        b.iter(|| session.execute("SELECT * FROM t ORDER BY id4").unwrap())
    });

    group.bench_function("s4", |b| {
        b.iter(|| session.execute("SELECT * FROM t ORDER BY v3 DESC").unwrap())
    });

    group.bench_function("s5", |b| {
        b.iter(|| {
            session
                .execute("SELECT * FROM t ORDER BY id1, id2")
                .unwrap()
        })
    });

    group.bench_function("s6", |b| {
        b.iter(|| {
            session
                .execute("SELECT * FROM t ORDER BY id1, id2, id3")
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Join benchmarks (j1-j2)
// ---------------------------------------------------------------------------

fn bench_join(c: &mut Criterion) {
    let mut session = match setup_join() {
        Some(s) => s,
        None => {
            eprintln!("Skipping join benchmarks: set TEIDE_BENCH_DATA to dataset dir");
            return;
        }
    };

    let mut group = c.benchmark_group("join");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("j1", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT * FROM x INNER JOIN y ON x.id1 = y.id1 AND x.id2 = y.id2 AND x.id3 = y.id3",
                )
                .unwrap()
        })
    });

    group.bench_function("j2", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT * FROM x LEFT JOIN y ON x.id1 = y.id1 AND x.id2 = y.id2 AND x.id3 = y.id3",
                )
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Window benchmarks (w1-w6 + subquery)
// ---------------------------------------------------------------------------

fn bench_window(c: &mut Criterion) {
    let mut session = match setup_groupby() {
        Some(s) => s,
        None => {
            eprintln!("Skipping window benchmarks: set TEIDE_BENCH_DATA to dataset dir");
            return;
        }
    };

    let mut group = c.benchmark_group("window");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("w1", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id1, v1, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1) as rn FROM t",
                )
                .unwrap()
        })
    });

    group.bench_function("w2", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id1, id4, RANK() OVER (PARTITION BY id1 ORDER BY id4) as rnk FROM t",
                )
                .unwrap()
        })
    });

    group.bench_function("w3", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id3, v1, SUM(v1) OVER (PARTITION BY id3 ORDER BY v1 \
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_sum FROM t",
                )
                .unwrap()
        })
    });

    group.bench_function("w4", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id1, v1, LAG(v1, 1) OVER (PARTITION BY id1 ORDER BY v1) as lag_v1 FROM t",
                )
                .unwrap()
        })
    });

    group.bench_function("w5", |b| {
        b.iter(|| {
            session
                .execute("SELECT id1, v1, AVG(v1) OVER (PARTITION BY id1) as avg_v1 FROM t")
                .unwrap()
        })
    });

    group.bench_function("w6", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT id1, id2, v1, ROW_NUMBER() OVER (PARTITION BY id1, id2 ORDER BY v1) as rn FROM t",
                )
                .unwrap()
        })
    });

    group.bench_function("subquery_window_filter", |b| {
        b.iter(|| {
            session
                .execute(
                    "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY id1 ORDER BY v1 DESC) <= 3 FROM t) \
                     WHERE id1 = 'id001' AND id2 = 'id085' AND id3 = 'id000094499'",
                )
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_groupby, bench_sort, bench_join, bench_window);
criterion_main!(benches);
