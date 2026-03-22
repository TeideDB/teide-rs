#define _POSIX_C_SOURCE 199309L
#include <teide/td.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void generate_csv(const char* path, int64_t n_rows, int64_t n_unique) {
    FILE* f = fopen(path, "w");
    fprintf(f, "id,val,sym\n");
    for (int64_t i = 0; i < n_rows; i++) {
        fprintf(f, "%lld,%.2f,sym_%lld\n",
                (long long)i, (double)i * 0.1, (long long)(i % n_unique));
    }
    fclose(f);
}

int main(int argc, char** argv) {
    int64_t n_rows = 500000;
    int64_t n_unique = 100000;

    if (argc > 1) n_rows = atoll(argv[1]);
    if (argc > 2) n_unique = atoll(argv[2]);

    const char* csv_path = "/tmp/teide_profile.csv";
    generate_csv(csv_path, n_rows, n_unique);

    /* Pre-fault the file into page cache */
    {
        td_heap_init();
        td_sym_init();
        td_t* warmup = td_read_csv(csv_path);
        if (warmup && !TD_IS_ERR(warmup)) td_release(warmup);
        td_sym_destroy();
        td_heap_destroy();
    }

    td_heap_init();
    td_sym_init();

    td_t* t = td_read_csv(csv_path);
    if (t && !TD_IS_ERR(t)) td_release(t);

    td_sym_destroy();
    td_heap_destroy();
    unlink(csv_path);
    return 0;
}
