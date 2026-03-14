// SQL/PGQ: Property graph catalog and MATCH pattern planner.

use std::collections::HashMap;
use crate::{Rel, Table};
use super::SqlError;

// ---------------------------------------------------------------------------
// Property graph catalog types
// ---------------------------------------------------------------------------

/// A vertex label mapping: label name -> session table name.
pub(crate) struct VertexLabel {
    pub table_name: String,
    pub label: String,
}

/// An edge label mapping: label name -> edge table with source/dest references.
pub(crate) struct EdgeLabel {
    pub table_name: String,
    pub label: String,
    pub src_col: String,
    pub src_ref_table: String,
    pub src_ref_col: String,
    pub dst_col: String,
    pub dst_ref_table: String,
    pub dst_ref_col: String,
}

/// Stored relationship: the built CSR index + its edge label metadata.
pub(crate) struct StoredRel {
    pub rel: Rel,
    pub edge_label: EdgeLabel,
}

/// A property graph defined over session tables.
pub(crate) struct PropertyGraph {
    pub name: String,
    pub vertex_labels: HashMap<String, VertexLabel>,
    pub edge_labels: HashMap<String, StoredRel>,
}
