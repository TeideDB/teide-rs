# CSV Null Handling Design

**Date**: 2026-03-20
**Status**: Approved

## Problem

The CSV loader treats empty/unparseable fields as 0 (for numerics), false (for booleans), and valid empty strings (for symbols). No null bitmap is ever created during CSV loading, silently producing incorrect data.

## Design

### Principle

If a field cannot be parsed as the inferred column type, it is NULL. No special marker lists — parse failure = NULL.

Applies uniformly to all column types: i64, f64, bool, date, time, timestamp, and string/symbol.

### Parser Changes

Add `bool* is_null` output parameter to all fast parsers:

- **`fast_i64(s, len, &is_null)`** — empty or non-numeric → NULL
- **`fast_f64(s, len, &is_null)`** — empty or non-numeric → NULL. NaN/Inf are valid floats, not NULL.
- **`fast_bool(s, len, &is_null)`** — new extracted function. `true/TRUE/1` → 1, `false/FALSE/0` → 0, everything else → NULL.
- **Date/time/timestamp** — apply same validation as `detect_type()`; fail → NULL.
- **String/symbol** — empty field (`len == 0`) → NULL. Non-empty always valid.

### Nullmap Allocation

Bulk pre-allocate before parsing begins (after column vectors are created):

- **≤128 rows**: use inline `td_t.nullmap[16]` (already zeroed by `td_vec_new`). Set `TD_ATTR_HAS_NULLS`.
- **>128 rows**: allocate external bitmap vector (`td_vec_new(TD_U8, (n_rows+7)/8)`), zero it, assign to `vec->ext_nullmap`. Set `TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT`.

### Hot Loop Integration

Each parse worker gets a direct `uint8_t*` to the nullmap:

```c
uint8_t* nm = (n_rows <= 128)
    ? col_vecs[c]->nullmap
    : (uint8_t*)td_data(col_vecs[c]->ext_nullmap);

bool is_null = false;
int64_t val = fast_i64(fld, flen, &is_null);
((int64_t*)col_data[c])[row] = val;
if (is_null) nm[row >> 3] |= (uint8_t)(1u << (row & 7));
```

### Thread Safety

No concern. Workers are assigned morsel-aligned row ranges (multiples of 1024), so nullmap byte writes never overlap between workers.

### Parallel Path: Symbol Fixup

The PACK_SYM fixup pass must skip NULL elements:

```c
if (nm[row >> 3] & (1u << (row & 7)))
    continue;  /* NULL — don't unpack */
```

Column narrowing (W32 → W8/W16) does not affect the nullmap — it is per-row, independent of element width.

### Per-Worker Null Tracking

- Serial path: `bool col_had_null[ncols]` — set true when any NULL encountered.
- Parallel path: each worker maintains `bool worker_had_null[ncols]`. After join, OR into final array.

### Post-Parse Cleanup

Strip nullmaps from columns that had zero NULLs to avoid downstream overhead:

```c
if (!col_had_null[c]) {
    if (vec->attrs & TD_ATTR_NULLMAP_EXT)
        td_release(vec->ext_nullmap);
    vec->ext_nullmap = NULL;
    vec->attrs &= ~(TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT);
    memset(vec->nullmap, 0, 16);
}
```

## Change Surface

| Area | Change |
|------|--------|
| `fast_i64()`, `fast_f64()` | Add `bool* is_null` param, NULL on parse failure |
| New `fast_bool()` | Extract from inline, add null handling |
| Date/time/timestamp parse | NULL on validation failure |
| String/symbol parse | Empty field → NULL |
| Nullmap pre-allocation | Bulk allocate before parse loop |
| Serial parse loop | Set null bits directly, track `col_had_null` |
| Parallel parse loop | Same, per-worker `worker_had_null` |
| Parallel fixup | Skip NULL elements during PACK_SYM unpack |
| Post-parse cleanup | Strip nullmaps from all-valid columns |

## Non-Goals

- Symbol column performance (separate effort)
- Configurable null marker lists (not needed — parse failure = NULL)
- Changes to type inference (stays as-is, UNKNOWN is transparent)
