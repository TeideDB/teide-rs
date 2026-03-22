/* Stub implementations for functions declared in teide-rs FFI but not yet
   implemented in the upstream C engine. These return error values so the Rust
   wrappers can report "not implemented" rather than causing linker errors. */

#include "teide/td.h"
#include <stddef.h>
#include <stdint.h>

td_op_t* td_local_clustering_coeff(td_graph_t* g, td_rel_t* rel) {
    (void)g; (void)rel;
    return (td_op_t*)(uintptr_t)TD_ERR_TYPE;
}

const int64_t* td_rel_neighbors(td_rel_t* rel, int64_t node,
                                 uint8_t direction, int64_t* out_count) {
    (void)rel; (void)node; (void)direction;
    if (out_count) *out_count = 0;
    return NULL;
}

int64_t td_rel_n_nodes(td_rel_t* rel, uint8_t direction) {
    (void)rel; (void)direction;
    return 0;
}
