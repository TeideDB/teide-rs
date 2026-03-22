#define _POSIX_C_SOURCE 199309L
#include <teide/td.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Generate a CSV with:
 *   - n_rows rows
 *   - 1 integer column, 1 float column, 1 symbol column
 *   - n_unique unique symbol values (cardinality)
 */
static void generate_csv(const char* path, int64_t n_rows, int64_t n_unique) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot create %s\n", path); return; }
    fprintf(f, "id,val,sym\n");
    for (int64_t i = 0; i < n_rows; i++) {
        fprintf(f, "%lld,%.2f,sym_%lld\n",
                (long long)i, (double)i * 0.1, (long long)(i % n_unique));
    }
    fclose(f);
}

static void bench_csv_load(const char* label, const char* path, int64_t n_rows) {
    /* Warm up */
    td_heap_init();
    td_sym_init();
    td_t* t = td_read_csv(path);
    if (t && !TD_IS_ERR(t)) td_release(t);
    td_sym_destroy();
    td_heap_destroy();

    /* Timed run */
    td_heap_init();
    td_sym_init();

    double start = now_ns();
    t = td_read_csv(path);
    double elapsed = now_ns() - start;

    if (!t || TD_IS_ERR(t)) {
        printf("%-32s  FAILED\n", label);
    } else {
        double ms = elapsed / 1e6;
        double rows_per_sec = (double)n_rows / (elapsed / 1e9);
        printf("%-32s  %8lld rows  %8.1f ms  %12.0f rows/sec\n",
               label, (long long)n_rows, ms, rows_per_sec);
        td_release(t);
    }

    td_sym_destroy();
    td_heap_destroy();
}

int main(void) {
    const char* csv_path = "/tmp/teide_bench_csv.csv";

    printf("%-32s  %8s       %8s  %12s\n",
           "Benchmark", "Rows", "Time", "Throughput");
    printf("%-32s  %8s       %8s  %12s\n",
           "--------------------------------", "--------", "--------", "------------");

    /* Low cardinality: 100 unique symbols */
    generate_csv(csv_path, 100000, 100);
    bench_csv_load("100K rows, 100 unique syms", csv_path, 100000);

    generate_csv(csv_path, 1000000, 100);
    bench_csv_load("1M rows, 100 unique syms", csv_path, 1000000);

    /* High cardinality: 100K unique symbols */
    generate_csv(csv_path, 100000, 100000);
    bench_csv_load("100K rows, 100K unique syms", csv_path, 100000);

    generate_csv(csv_path, 1000000, 100000);
    bench_csv_load("1M rows, 100K unique syms", csv_path, 1000000);

    /* Very high cardinality: 1M unique symbols */
    generate_csv(csv_path, 1000000, 1000000);
    bench_csv_load("1M rows, 1M unique syms", csv_path, 1000000);

    /* Numeric only (no symbols, baseline) */
    {
        FILE* f = fopen(csv_path, "w");
        fprintf(f, "id,val\n");
        for (int64_t i = 0; i < 1000000; i++)
            fprintf(f, "%lld,%.2f\n", (long long)i, (double)i * 0.1);
        fclose(f);
        bench_csv_load("1M rows, numeric only", csv_path, 1000000);
    }

    unlink(csv_path);
    return 0;
}
