/*
 *   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

/* ============================================================================
 * csv.c — Fast parallel CSV reader
 *
 * Design:
 *   1. mmap + MAP_POPULATE for zero-copy file access
 *   2. memchr-based newline scan for row offset discovery
 *   3. Single-pass: sample-based type inference, then parallel value parsing
 *   4. Inline integer/float parsers (bypass strtoll/strtod overhead)
 *   5. Parallel row parsing via td_pool_dispatch
 *   6. Per-worker local sym tables, merged post-parse on main thread
 * ============================================================================ */

#if defined(__linux__)
  #define _GNU_SOURCE
#endif

#include "csv.h"
#include "mem/sys.h"
#include "ops/pool.h"
#include "ops/hash.h"
#include "table/sym.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>  /* strtoll fallback for fast_i64 overflow */
#include <sys/stat.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <sys/mman.h>

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

#define CSV_MAX_COLS      256
#define CSV_SAMPLE_ROWS   100

/* --------------------------------------------------------------------------
 * mmap flags
 * -------------------------------------------------------------------------- */

#ifdef __linux__
  #define MMAP_FLAGS (MAP_PRIVATE | MAP_POPULATE)
#else
  #define MMAP_FLAGS MAP_PRIVATE
#endif

/* --------------------------------------------------------------------------
 * Scratch memory helpers (same pattern as exec.c).
 * Uses td_alloc/td_free (buddy allocator) instead of malloc/free.
 * -------------------------------------------------------------------------- */

static inline void* scratch_alloc(td_t** hdr_out, size_t nbytes) {
    td_t* h = td_alloc(nbytes);
    if (!h) { *hdr_out = NULL; return NULL; }
    *hdr_out = h;
    return td_data(h);
}

static inline void* scratch_realloc(td_t** hdr_out, size_t old_bytes, size_t new_bytes) {
    td_t* old_h = *hdr_out;
    td_t* new_h = td_alloc(new_bytes);
    if (!new_h) return NULL;
    void* new_p = td_data(new_h);
    if (old_h) {
        memcpy(new_p, td_data(old_h), old_bytes < new_bytes ? old_bytes : new_bytes);
        td_free(old_h);
    }
    *hdr_out = new_h;
    return new_p;
}

static inline void scratch_free(td_t* hdr) {
    if (hdr) td_free(hdr);
}

/* Hash uses wyhash from ops/hash.h (td_hash_bytes) — much faster than FNV-1a
 * for short strings typical in CSV columns. */

/* String reference — raw pointer into mmap'd buffer + length.
 * Used during parse phase; interned into sym table after parse. */
typedef struct {
    const char* ptr;
    uint32_t    len;
} csv_strref_t;

/* --------------------------------------------------------------------------
 * Type inference
 * -------------------------------------------------------------------------- */

typedef enum {
    CSV_TYPE_UNKNOWN = 0,
    CSV_TYPE_BOOL,
    CSV_TYPE_I64,
    CSV_TYPE_F64,
    CSV_TYPE_STR,
    CSV_TYPE_DATE,
    CSV_TYPE_TIME,
    CSV_TYPE_TIMESTAMP
} csv_type_t;

static csv_type_t detect_type(const char* f, size_t len) {
    if (len == 0) return CSV_TYPE_UNKNOWN;

    /* Common null sentinel strings → UNKNOWN (will become NULL) */
    if ((len == 3 && (memcmp(f, "N/A", 3) == 0 || memcmp(f, "n/a", 3) == 0)) ||
        (len == 2 && (memcmp(f, "NA", 2) == 0 || memcmp(f, "na", 2) == 0)) ||
        (len == 4 && (memcmp(f, "null", 4) == 0 || memcmp(f, "NULL", 4) == 0 ||
                      memcmp(f, "None", 4) == 0 || memcmp(f, "none", 4) == 0)) ||
        (len == 1 && f[0] == '.'))  /* bare dot — not a valid value */
        return CSV_TYPE_UNKNOWN;

    /* NaN/Inf literals → float */
    if (len == 3) {
        if ((f[0]=='n'||f[0]=='N') && (f[1]=='a'||f[1]=='A') && (f[2]=='n'||f[2]=='N'))
            return CSV_TYPE_F64;
        if ((f[0]=='i'||f[0]=='I') && (f[1]=='n'||f[1]=='N') && (f[2]=='f'||f[2]=='F'))
            return CSV_TYPE_F64;
    }
    if ((len == 4 && (f[0]=='+' || f[0]=='-')) &&
        (f[1]=='i'||f[1]=='I') && (f[2]=='n'||f[2]=='N') && (f[3]=='f'||f[3]=='F'))
        return CSV_TYPE_F64;

    /* Boolean */
    if ((len == 4 && memcmp(f, "true", 4) == 0) ||
        (len == 5 && memcmp(f, "false", 5) == 0) ||
        (len == 4 && memcmp(f, "TRUE", 4) == 0) ||
        (len == 5 && memcmp(f, "FALSE", 5) == 0))
        return CSV_TYPE_BOOL;

    /* Numeric scan */
    const char* p = f;
    const char* end = f + len;
    if (*p == '-' || *p == '+') p++;
    bool has_dot = false, has_e = false, has_digit = false;
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        if (c >= '0' && c <= '9') { has_digit = true; p++; continue; }
        if (c == '.' && !has_dot) { has_dot = true; p++; continue; }
        if ((c == 'e' || c == 'E') && !has_e) {
            has_e = true; p++;
            if (p < end && (*p == '-' || *p == '+')) p++;
            continue;
        }
        break;
    }
    if (p == end && has_digit) {
        if (!has_dot && !has_e) return CSV_TYPE_I64;
        return CSV_TYPE_F64;
    }

    /* Date: YYYY-MM-DD (exactly 10 chars) or Timestamp: YYYY-MM-DD{T| }HH:MM:SS */
    if (len >= 10 && f[4] == '-' && f[7] == '-') {
        bool is_date = true;
        for (int i = 0; i < 10; i++) {
            if (i == 4 || i == 7) continue;
            if ((unsigned)(f[i] - '0') > 9) { is_date = false; break; }
        }
        if (is_date) {
            if (len == 10) return CSV_TYPE_DATE;
            if (len >= 19 && (f[10] == 'T' || f[10] == ' ') &&
                f[13] == ':' && f[16] == ':') {
                const int tp[] = {11,12,14,15,17,18};
                bool is_ts = true;
                for (int i = 0; i < 6; i++) {
                    if ((unsigned)(f[tp[i]] - '0') > 9) { is_ts = false; break; }
                }
                if (is_ts) return CSV_TYPE_TIMESTAMP;
            }
        }
    }

    /* Time: HH:MM:SS[.ffffff] (at least 8 chars) */
    if (len >= 8 && f[2] == ':' && f[5] == ':') {
        const int tp[] = {0,1,3,4,6,7};
        bool is_time = true;
        for (int i = 0; i < 6; i++) {
            if ((unsigned)(f[tp[i]] - '0') > 9) { is_time = false; break; }
        }
        if (is_time) return CSV_TYPE_TIME;
    }

    return CSV_TYPE_STR;
}

static csv_type_t promote_csv_type(csv_type_t cur, csv_type_t obs) {
    if (cur == CSV_TYPE_UNKNOWN) return obs;
    if (obs == CSV_TYPE_UNKNOWN) return cur;
    if (cur == obs) return cur;
    if (cur == CSV_TYPE_STR || obs == CSV_TYPE_STR) return CSV_TYPE_STR;
    /* DATE + TIMESTAMP → TIMESTAMP */
    if ((cur == CSV_TYPE_DATE && obs == CSV_TYPE_TIMESTAMP) ||
        (cur == CSV_TYPE_TIMESTAMP && obs == CSV_TYPE_DATE))
        return CSV_TYPE_TIMESTAMP;
    /* Numeric promotion: BOOL ⊂ I64 ⊂ F64 (enum values 1 < 2 < 3) */
    if (cur <= CSV_TYPE_F64 && obs <= CSV_TYPE_F64) {
        if (cur == CSV_TYPE_F64 || obs == CSV_TYPE_F64) return CSV_TYPE_F64;
        if (cur == CSV_TYPE_I64 || obs == CSV_TYPE_I64) return CSV_TYPE_I64;
        return cur;
    }
    /* All other mixed types (e.g. DATE+I64, TIME+BOOL) → STR */
    return CSV_TYPE_STR;
}

/* --------------------------------------------------------------------------
 * Zero-copy field scanner
 *
 * Returns pointer past the field's trailing delimiter (or at newline/end).
 * Sets *out and *out_len to the field content. For unquoted fields, *out
 * points directly into the mmap buffer. For quoted fields with escaped
 * quotes, content is unescaped into esc_buf.
 * -------------------------------------------------------------------------- */

static const char* scan_field_quoted(const char* p, const char* buf_end,
                                     char delim,
                                     const char** out, size_t* out_len,
                                     char* esc_buf, char** dyn_esc) {
    p++; /* skip opening quote */
    const char* fld_start = p;
    bool has_escape = false;

    while (p < buf_end) {
        if (*p == '"') {
            if (p + 1 < buf_end && *(p + 1) == '"') {
                has_escape = true;
                p += 2;
            } else {
                break; /* closing quote */
            }
        } else {
            p++;
        }
    }
    size_t raw_len = (size_t)(p - fld_start);
    if (p < buf_end && *p == '"') p++; /* skip closing quote */

    if (has_escape) {
        /* raw_len >= output length (quotes are collapsed); no overflow. */
        char* dest = esc_buf;
        if (TD_UNLIKELY(raw_len > 8192)) {
            /* Field too large for stack buffer — dynamically allocate */
            dest = (char*)td_sys_alloc(raw_len);
            if (!dest) {
                /* OOM: fall back to raw (quotes remain) */
                *out = fld_start;
                *out_len = raw_len;
                goto advance;
            }
            *dyn_esc = dest;
        }
        size_t olen = 0;
        for (const char* s = fld_start; s < fld_start + raw_len; s++) {
            if (*s == '"' && s + 1 < fld_start + raw_len && *(s + 1) == '"') {
                dest[olen++] = '"';
                s++;
            } else {
                dest[olen++] = *s;
            }
        }
        *out = dest;
        *out_len = olen;
    } else {
        *out = fld_start;
        *out_len = raw_len;
    }

advance:
    /* Advance past delimiter */
    if (p < buf_end && *p == delim) p++;
    /* Don't advance past newline — caller handles row boundaries */
    return p;
}

TD_INLINE const char* scan_field(const char* p, const char* buf_end,
                                  char delim,
                                  const char** out, size_t* out_len,
                                  char* esc_buf, char** dyn_esc) {
    if (TD_UNLIKELY(p >= buf_end)) {
        *out = p;
        *out_len = 0;
        return p;
    }

    if (TD_LIKELY(*p != '"')) {
        /* Unquoted field — fast path */
        const char* s = p;
        while (p < buf_end && *p != delim && *p != '\n' && *p != '\r') p++;
        *out = s;
        *out_len = (size_t)(p - s);
        if (p < buf_end && *p == delim) return p + 1;
        return p;
    }

    return scan_field_quoted(p, buf_end, delim, out, out_len, esc_buf, dyn_esc);
}

/* --------------------------------------------------------------------------
 * Fast inline integer parser (replaces strtoll)
 * -------------------------------------------------------------------------- */

TD_INLINE int64_t fast_i64(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len == 0)) { *is_null = true; return 0; }
    *is_null = false;

    const char* end = p + len;
    const char* start = p;
    bool neg = false;
    if (*p == '-') { neg = true; p++; }
    else if (*p == '+') { p++; }

    /* Count digit span; if >18, fall back to strtoll to avoid overflow */
    size_t digit_len = 0;
    for (const char* q = p; q < end && (unsigned char)(*q - '0') <= 9; q++)
        digit_len++;
    if (TD_UNLIKELY(digit_len > 18)) {
        /* max int64 = 20 chars; 31-byte limit safe for valid integers. */
        char tmp[32];
        size_t slen = (size_t)(end - start);
        if (slen > sizeof(tmp) - 1) slen = sizeof(tmp) - 1;
        memcpy(tmp, start, slen);
        tmp[slen] = '\0';
        char* endp;
        int64_t v = strtoll(tmp, &endp, 10);
        if (*endp != '\0') { *is_null = true; return 0; }
        return v;
    }

    uint64_t val = 0;
    while (p < end) {
        unsigned d = (unsigned char)*p - '0';
        if (d > 9) break; /* stop on non-digit */
        val = val * 10 + d;
        p++;
    }

    /* If digit scan didn't consume entire field → unparseable */
    if (p != end) { *is_null = true; return 0; }

    /* 18-digit values may exceed int64 range; fall back to strtoll for safety */
    if (TD_UNLIKELY(digit_len == 18)) {
        uint64_t limit = neg ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
        if (val > limit) {
            char tmp[32];
            size_t slen = (size_t)(end - start);
            if (slen > sizeof(tmp) - 1) slen = sizeof(tmp) - 1;
            memcpy(tmp, start, slen);
            tmp[slen] = '\0';
            char* endp;
            int64_t v = strtoll(tmp, &endp, 10);
            if (*endp != '\0') { *is_null = true; return 0; }
            return v;
        }
    }
    /* Negate in unsigned to avoid signed overflow UB */
    return neg ? (int64_t)(~val + 1u) : (int64_t)val;
}

/* --------------------------------------------------------------------------
 * Fast inline float parser (replaces strtod)
 *
 * Handles: [+-]digits[.digits][eE[+-]digits]
 * Uses pow10 lookup table for exponents up to +/-22.
 * -------------------------------------------------------------------------- */

static const double g_pow10[] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
    1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

TD_INLINE double fast_f64(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len == 0)) { *is_null = true; return 0.0; }
    *is_null = false;

    /* NaN/Inf string literals — check before numeric parse (valid, not null) */
    if (TD_UNLIKELY(len <= 4)) {
        if (len == 3 &&
            (p[0]=='n'||p[0]=='N') && (p[1]=='a'||p[1]=='A') && (p[2]=='n'||p[2]=='N'))
            return __builtin_nan("");
        if (len == 3 &&
            (p[0]=='i'||p[0]=='I') && (p[1]=='n'||p[1]=='N') && (p[2]=='f'||p[2]=='F'))
            return __builtin_inf();
        if (len == 4 && p[0] == '+' &&
            (p[1]=='i'||p[1]=='I') && (p[2]=='n'||p[2]=='N') && (p[3]=='f'||p[3]=='F'))
            return __builtin_inf();
        if (len == 4 && p[0] == '-' &&
            (p[1]=='i'||p[1]=='I') && (p[2]=='n'||p[2]=='N') && (p[3]=='f'||p[3]=='F'))
            return -__builtin_inf();
    }

    const char* start = p;
    const char* end = p + len;
    int negative = 0;
    if (*p == '-') { negative = 1; p++; }
    else if (*p == '+') { p++; }

    /* Integer part */
    uint64_t int_part = 0;
    int idigits = 0;
    while (p < end && (unsigned)(*p - '0') < 10) {
        int_part = int_part * 10 + (uint64_t)(*p - '0');
        idigits++;
        p++;
        if (idigits > 18) goto strtod_fallback;
    }
    /* 18-digit integer parts risk uint64 overflow in subsequent mul;
     * fall back to strtod for exact conversion. */
    if (TD_UNLIKELY(idigits == 18 && int_part > (uint64_t)999999999999999999ULL))
        goto strtod_fallback;
    double val = (double)int_part;

    /* Fractional part */
    if (p < end && *p == '.') {
        p++;
        uint64_t frac = 0;
        int frac_digits = 0;
        while (p < end && (unsigned)(*p - '0') < 10) {
            frac = frac * 10 + (uint64_t)(*p - '0');
            frac_digits++;
            p++;
            if (frac_digits > 18) {
                /* Cap fractional accumulation — skip remaining fractional digits */
                while (p < end && (unsigned)(*p - '0') < 10) p++;
                break;
            }
        }
        if (frac_digits > 0 && frac_digits <= 22) {
            val += (double)frac / g_pow10[frac_digits];
        } else if (frac_digits > 0) {
            double f = (double)frac;
            int d = frac_digits;
            while (d > 22) { f /= 1e22; d -= 22; }
            f /= g_pow10[d];
            val += f;
        }
    }

    /* Exponent */
    if (p < end && (*p == 'e' || *p == 'E')) {
        p++;
        int exp_neg = 0;
        if (p < end) {
            if (*p == '-') { exp_neg = 1; p++; }
            else if (*p == '+') { p++; }
        }
        int exp_val = 0;
        while (p < end && (unsigned)(*p - '0') < 10) {
            exp_val = exp_val * 10 + (*p - '0');
            /* Clamp exponent to avoid int overflow on crafted input (e.g. 1e9999999999).
             * 10^999 is far beyond double range (max ~10^308), so the result
             * will be inf/0.0 anyway, but the integer arithmetic stays defined. */
            if (exp_val > 999) exp_val = 999;
            p++;
        }
        if (exp_val <= 22) {
            if (exp_neg) val /= g_pow10[exp_val];
            else         val *= g_pow10[exp_val];
        } else {
            int e = exp_val;
            if (exp_neg) {
                while (e > 22) { val /= 1e22; e -= 22; }
                val /= g_pow10[e];
            } else {
                while (e > 22) { val *= 1e22; e -= 22; }
                val *= g_pow10[e];
            }
        }
    }

    /* If we didn't consume all input, field is unparseable */
    if (p != end) { *is_null = true; return 0.0; }

    return negative ? -val : val;

strtod_fallback:
    {
        char tmp[64];
        size_t slen = (size_t)(end - start);
        if (slen > sizeof(tmp) - 1) slen = sizeof(tmp) - 1;
        memcpy(tmp, start, slen);
        tmp[slen] = '\0';
        char* endp;
        double v = strtod(tmp, &endp);
        if (*endp != '\0') { *is_null = true; return 0.0; }
        return v;
    }
}

/* --------------------------------------------------------------------------
 * Fast inline date/time parsers
 *
 * DATE:      YYYY-MM-DD        → int32_t  (days since 2000-01-01)
 * TIME:      HH:MM:SS[.fff]    → int32_t  (milliseconds since midnight)
 * TIMESTAMP: YYYY-MM-DD{T| }HH:MM:SS[.ffffff] → int64_t (µs since 2000-01-01)
 *
 * Uses Howard Hinnant's civil-calendar algorithm (public domain) for the
 * date→days conversion — O(1), no tables, no branches.
 * -------------------------------------------------------------------------- */

TD_INLINE int32_t civil_to_days(int y, int m, int d) {
    /* Shift Jan/Feb to months 10/11 of the previous year */
    if (m <= 2) { y--; m += 9; } else { m -= 3; }
    int era = (y >= 0 ? y : y - 399) / 400;
    int yoe = y - era * 400;
    int doy = (153 * m + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (int32_t)(era * 146097 + doe - 719468 - 10957);
}

TD_INLINE int32_t fast_date(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len < 10)) { *is_null = true; return 0; }
    *is_null = false;
    int y = (p[0]-'0')*1000 + (p[1]-'0')*100 + (p[2]-'0')*10 + (p[3]-'0');
    int m = (p[5]-'0')*10 + (p[6]-'0');
    int d = (p[8]-'0')*10 + (p[9]-'0');
    if (TD_UNLIKELY(m < 1 || m > 12 || d < 1 || d > 31)) { *is_null = true; return 0; }
    return civil_to_days(y, m, d);
}

/* TIME → int32_t milliseconds since midnight (kdb+ convention) */
TD_INLINE int32_t fast_time(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len < 8)) { *is_null = true; return 0; }
    *is_null = false;
    int h  = (p[0]-'0')*10 + (p[1]-'0');
    int mi = (p[3]-'0')*10 + (p[4]-'0');
    int s  = (p[6]-'0')*10 + (p[7]-'0');
    if (TD_UNLIKELY(h > 23 || mi > 59 || s > 59)) { *is_null = true; return 0; }
    int32_t ms = h * 3600000 + mi * 60000 + s * 1000;
    /* Fractional seconds → milliseconds */
    if (len > 8 && p[8] == '.') {
        int frac = 0, digits = 0;
        for (size_t i = 9; i < len && digits < 3; i++, digits++) {
            unsigned di = (unsigned char)p[i] - '0';
            if (di > 9) break;
            frac = frac * 10 + (int)di;
        }
        while (digits < 3) { frac *= 10; digits++; }
        ms += (int32_t)frac;
    }
    return ms;
}

/* Timestamp time component → int64_t microseconds (higher precision) */
TD_INLINE int64_t fast_time_us(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len < 8)) { *is_null = true; return 0; }
    *is_null = false;
    int h  = (p[0]-'0')*10 + (p[1]-'0');
    int mi = (p[3]-'0')*10 + (p[4]-'0');
    int s  = (p[6]-'0')*10 + (p[7]-'0');
    if (TD_UNLIKELY(h > 23 || mi > 59 || s > 59)) { *is_null = true; return 0; }
    int64_t us = (int64_t)h * 3600000000LL + (int64_t)mi * 60000000LL +
                 (int64_t)s * 1000000LL;
    if (len > 8 && p[8] == '.') {
        int frac = 0, digits = 0;
        for (size_t i = 9; i < len && digits < 6; i++, digits++) {
            unsigned di = (unsigned char)p[i] - '0';
            if (di > 9) break;
            frac = frac * 10 + (int)di;
        }
        while (digits < 6) { frac *= 10; digits++; }
        us += frac;
    }
    return us;
}

TD_INLINE int64_t fast_timestamp(const char* p, size_t len, bool* is_null) {
    if (TD_UNLIKELY(len < 19)) { *is_null = true; return 0; }
    *is_null = false;
    int32_t days = fast_date(p, 10, is_null);
    if (*is_null) return 0;
    bool time_null = false;
    int64_t time_us = fast_time_us(p + 11, len - 11, &time_null);
    if (time_null) { *is_null = true; return 0; }
    return (int64_t)days * 86400000000LL + time_us;
}

/* --------------------------------------------------------------------------
 * Null-aware boolean parser
 * -------------------------------------------------------------------------- */

TD_INLINE uint8_t fast_bool(const char* s, size_t len, bool* is_null) {
    if (len == 0) { *is_null = true; return 0; }
    *is_null = false;
    if ((len == 4 && (memcmp(s, "true", 4) == 0 || memcmp(s, "TRUE", 4) == 0)) ||
        (len == 1 && s[0] == '1'))
        return 1;
    if ((len == 5 && (memcmp(s, "false", 5) == 0 || memcmp(s, "FALSE", 5) == 0)) ||
        (len == 1 && s[0] == '0'))
        return 0;
    *is_null = true;
    return 0;
}

/* --------------------------------------------------------------------------
 * Row offsets builder — memchr-accelerated
 *
 * Uses memchr (glibc: SIMD-accelerated ~15-20 GB/s) for newline scanning.
 * Fast path for quote-free files; falls back to byte-by-byte for quoted
 * fields with embedded newlines. Returns exact row count.
 *
 * Allocates offsets via scratch_alloc. Caller frees with scratch_free.
 * -------------------------------------------------------------------------- */

static int64_t build_row_offsets(const char* buf, size_t buf_size,
                                  size_t data_offset,
                                  int64_t** offsets_out, td_t** hdr_out) {
    const char* p = buf + data_offset;
    const char* end = buf + buf_size;

    /* Skip leading blank lines */
    while (p < end && (*p == '\r' || *p == '\n')) p++;
    if (p >= end) { *offsets_out = NULL; *hdr_out = NULL; return 0; }

    /* Estimate capacity: ~40 bytes per row + headroom.
     * 40 bytes/row is conservative for typical CSVs; realloc path handles
     * underestimates. */
    size_t remaining = (size_t)(end - p);
    int64_t est = (int64_t)(remaining / 40) + 16;
    td_t* hdr = NULL;
    int64_t* offs = (int64_t*)scratch_alloc(&hdr, (size_t)est * sizeof(int64_t));
    if (!offs) { *offsets_out = NULL; *hdr_out = NULL; return 0; }

    int64_t n = 0;
    offs[n++] = (int64_t)(p - buf);

    /* Check if file has any quotes — determines fast vs slow path */
    bool has_quotes = (memchr(p, '"', remaining) != NULL);

    if (TD_LIKELY(!has_quotes)) {
        /* Fast path: no quotes, use memchr for newlines.
         * Only scans for \n; pure \r line endings (old Mac) treated as single row.
         * Empty lines are preserved as rows (for NULL handling). */
        for (;;) {
            const char* nl = (const char*)memchr(p, '\n', (size_t)(end - p));
            if (!nl) break;
            p = nl + 1;
            /* Skip optional \r after \n (unusual \n\r endings) */
            if (p < end && *p == '\r') p++;
            if (p >= end) break;

            if (n >= est) {
                est *= 2;
                offs = (int64_t*)scratch_realloc(&hdr,
                    (size_t)n * sizeof(int64_t),
                    (size_t)est * sizeof(int64_t));
                if (!offs) { scratch_free(hdr); *offsets_out = NULL; *hdr_out = NULL; return 0; }
            }
            offs[n++] = (int64_t)(p - buf);
        }
    } else {
        /* Slow path: track quote parity, byte-by-byte.
         * Empty lines preserved as rows (for NULL handling). */
        bool in_quote = false;
        while (p < end) {
            char c = *p;
            if (c == '"') {
                in_quote = !in_quote;
                p++;
            } else if (!in_quote && (c == '\n' || c == '\r')) {
                if (c == '\r' && p + 1 < end && *(p + 1) == '\n') p++;
                p++;
                if (p < end) {
                    if (n >= est) {
                        est *= 2;
                        offs = (int64_t*)scratch_realloc(&hdr,
                            (size_t)n * sizeof(int64_t),
                            (size_t)est * sizeof(int64_t));
                        if (!offs) { scratch_free(hdr); *offsets_out = NULL; *hdr_out = NULL; return 0; }
                    }
                    offs[n++] = (int64_t)(p - buf);
                }
            } else {
                p++;
            }
        }
    }

    *offsets_out = offs;
    *hdr_out = hdr;
    return n;
}

/* --------------------------------------------------------------------------
 * Batch-intern string columns after parse.
 * Single-threaded — walks each string column, interns into global sym table,
 * writes sym IDs into the final uint32_t column.
 * -------------------------------------------------------------------------- */

static bool csv_intern_strings(csv_strref_t** str_refs, int n_cols,
                                const csv_type_t* col_types,
                                void** col_data, int64_t n_rows,
                                int64_t* col_max_ids,
                                uint8_t** col_nullmaps) {
    bool ok = true;
    for (int c = 0; c < n_cols; c++) {
        if (col_types[c] != CSV_TYPE_STR) continue;
        csv_strref_t* refs = str_refs[c];
        uint32_t* ids = (uint32_t*)col_data[c];
        uint8_t* nm = col_nullmaps ? col_nullmaps[c] : NULL;
        int64_t max_id = 0;

        /* Pre-grow: upper bound is n_rows unique strings */
        uint32_t current = td_sym_count();
        td_sym_ensure_cap(current + (uint32_t)(n_rows < UINT32_MAX ? n_rows : UINT32_MAX));

        for (int64_t r = 0; r < n_rows; r++) {
            if (nm && (nm[r >> 3] & (1u << (r & 7)))) {
                ids[r] = 0;
                continue;
            }
            uint32_t hash = (uint32_t)td_hash_bytes(refs[r].ptr, refs[r].len);
            int64_t id = td_sym_intern_prehashed(hash, refs[r].ptr, refs[r].len);
            if (id < 0) { ok = false; id = 0; }
            ids[r] = (uint32_t)id;
            if (id > max_id) max_id = id;
        }
        if (col_max_ids) col_max_ids[c] = max_id;
    }
    return ok;
}

/* --------------------------------------------------------------------------
 * Parallel parse context and callback
 * -------------------------------------------------------------------------- */

typedef struct {
    const char*       buf;
    size_t            buf_size;
    const int64_t*    row_offsets;
    int64_t           n_rows;
    int               n_cols;
    char              delim;
    const csv_type_t* col_types;
    void**            col_data;     /* non-const: workers write parsed values into columns */
    csv_strref_t**    str_refs;     /* [n_cols] — strref arrays for string columns, NULL for others */
    uint8_t**         col_nullmaps;
    bool*             worker_had_null; /* [n_workers * n_cols] */
} csv_par_ctx_t;

static void csv_parse_fn(void* arg, uint32_t worker_id,
                          int64_t start, int64_t end_row) {
    csv_par_ctx_t* ctx = (csv_par_ctx_t*)arg;
    char esc_buf[8192];
    const char* buf_end = ctx->buf + ctx->buf_size;
    bool* my_had_null = &ctx->worker_had_null[(size_t)worker_id * (size_t)ctx->n_cols];

    for (int64_t row = start; row < end_row; row++) {
        const char* p = ctx->buf + ctx->row_offsets[row];
        const char* row_end = (row + 1 < ctx->n_rows)
            ? ctx->buf + ctx->row_offsets[row + 1]
            : buf_end;

        for (int c = 0; c < ctx->n_cols; c++) {
            /* Guard: if past row boundary, fill remaining columns with defaults + null */
            if (p >= row_end) {
                for (; c < ctx->n_cols; c++) {
                    switch (ctx->col_types[c]) {
                        case CSV_TYPE_BOOL: ((uint8_t*)ctx->col_data[c])[row] = 0; break;
                        case CSV_TYPE_I64:  ((int64_t*)ctx->col_data[c])[row] = 0; break;
                        case CSV_TYPE_F64:  ((double*)ctx->col_data[c])[row] = 0.0; break;
                        case CSV_TYPE_DATE: ((int32_t*)ctx->col_data[c])[row] = 0; break;
                        case CSV_TYPE_TIME: ((int32_t*)ctx->col_data[c])[row] = 0; break;
                        case CSV_TYPE_TIMESTAMP:
                            ((int64_t*)ctx->col_data[c])[row] = 0; break;
                        case CSV_TYPE_STR:
                            ctx->str_refs[c][row].ptr = NULL;
                            ctx->str_refs[c][row].len = 0;
                            break;
                        default: break;
                    }
                    ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                    my_had_null[c] = true;
                }
                break;
            }

            const char* fld;
            size_t flen;
            char* dyn_esc = NULL;
            p = scan_field(p, buf_end, ctx->delim, &fld, &flen, esc_buf, &dyn_esc);

            /* Strip trailing \r from last field of row */
            if (c == ctx->n_cols - 1 && flen > 0 && fld[flen - 1] == '\r')
                flen--;

            switch (ctx->col_types[c]) {
                case CSV_TYPE_BOOL: {
                    bool is_null;
                    uint8_t v = fast_bool(fld, flen, &is_null);
                    ((uint8_t*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_I64: {
                    bool is_null;
                    int64_t v = fast_i64(fld, flen, &is_null);
                    ((int64_t*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_F64: {
                    bool is_null;
                    double v = fast_f64(fld, flen, &is_null);
                    ((double*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_DATE: {
                    bool is_null;
                    int32_t v = fast_date(fld, flen, &is_null);
                    ((int32_t*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_TIME: {
                    bool is_null;
                    int32_t v = fast_time(fld, flen, &is_null);
                    ((int32_t*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_TIMESTAMP: {
                    bool is_null;
                    int64_t v = fast_timestamp(fld, flen, &is_null);
                    ((int64_t*)ctx->col_data[c])[row] = v;
                    if (is_null) {
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_STR: {
                    if (flen == 0) {
                        ctx->str_refs[c][row].ptr = NULL;
                        ctx->str_refs[c][row].len = 0;
                        ctx->col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        my_had_null[c] = true;
                    } else {
                        ctx->str_refs[c][row].ptr = fld;
                        ctx->str_refs[c][row].len = (uint32_t)flen;
                    }
                    break;
                }
                default:
                    break;
            }
            if (TD_UNLIKELY(dyn_esc != NULL)) td_sys_free(dyn_esc);
        }
    }
}

/* --------------------------------------------------------------------------
 * Serial parse fallback (small files or no thread pool)
 * -------------------------------------------------------------------------- */

static void csv_parse_serial(const char* buf, size_t buf_size,
                              const int64_t* row_offsets, int64_t n_rows,
                              int n_cols, char delim,
                              const csv_type_t* col_types, void** col_data,
                              csv_strref_t** str_refs,
                              uint8_t** col_nullmaps, bool* col_had_null) {
    char esc_buf[8192];
    const char* buf_end = buf + buf_size;

    for (int64_t row = 0; row < n_rows; row++) {
        const char* p = buf + row_offsets[row];
        const char* row_end = (row + 1 < n_rows)
            ? buf + row_offsets[row + 1]
            : buf_end;

        for (int c = 0; c < n_cols; c++) {
            /* Guard: if past row boundary, fill remaining columns with defaults + null */
            if (p >= row_end) {
                for (; c < n_cols; c++) {
                    switch (col_types[c]) {
                        case CSV_TYPE_BOOL: ((uint8_t*)col_data[c])[row] = 0; break;
                        case CSV_TYPE_I64:  ((int64_t*)col_data[c])[row] = 0; break;
                        case CSV_TYPE_F64:  ((double*)col_data[c])[row] = 0.0; break;
                        case CSV_TYPE_DATE: ((int32_t*)col_data[c])[row] = 0; break;
                        case CSV_TYPE_TIME: ((int32_t*)col_data[c])[row] = 0; break;
                        case CSV_TYPE_TIMESTAMP:
                            ((int64_t*)col_data[c])[row] = 0; break;
                        case CSV_TYPE_STR:
                            str_refs[c][row].ptr = NULL;
                            str_refs[c][row].len = 0;
                            break;
                        default: break;
                    }
                    col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                    col_had_null[c] = true;
                }
                break;
            }

            const char* fld;
            size_t flen;
            char* dyn_esc = NULL;
            p = scan_field(p, buf_end, delim, &fld, &flen, esc_buf, &dyn_esc);

            /* Strip trailing \r from last field of row */
            if (c == n_cols - 1 && flen > 0 && fld[flen - 1] == '\r')
                flen--;

            switch (col_types[c]) {
                case CSV_TYPE_BOOL: {
                    bool is_null;
                    uint8_t v = fast_bool(fld, flen, &is_null);
                    ((uint8_t*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_I64: {
                    bool is_null;
                    int64_t v = fast_i64(fld, flen, &is_null);
                    ((int64_t*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_F64: {
                    bool is_null;
                    double v = fast_f64(fld, flen, &is_null);
                    ((double*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_DATE: {
                    bool is_null;
                    int32_t v = fast_date(fld, flen, &is_null);
                    ((int32_t*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_TIME: {
                    bool is_null;
                    int32_t v = fast_time(fld, flen, &is_null);
                    ((int32_t*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_TIMESTAMP: {
                    bool is_null;
                    int64_t v = fast_timestamp(fld, flen, &is_null);
                    ((int64_t*)col_data[c])[row] = v;
                    if (is_null) {
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    }
                    break;
                }
                case CSV_TYPE_STR: {
                    if (flen == 0) {
                        str_refs[c][row].ptr = NULL;
                        str_refs[c][row].len = 0;
                        col_nullmaps[c][row >> 3] |= (uint8_t)(1u << (row & 7));
                        col_had_null[c] = true;
                    } else {
                        str_refs[c][row].ptr = fld;
                        str_refs[c][row].len = (uint32_t)flen;
                    }
                    break;
                }
                default:
                    break;
            }
            if (TD_UNLIKELY(dyn_esc != NULL)) td_sys_free(dyn_esc);
        }
    }
}

/* --------------------------------------------------------------------------
 * td_read_csv_opts — main CSV parser
 * -------------------------------------------------------------------------- */

td_t* td_read_csv_opts(const char* path, char delimiter, bool header,
                        const int8_t* col_types_in, int32_t n_types) {
    /* ---- 1. Open file and get size ---- */
    int fd = open(path, O_RDONLY);
    if (fd < 0) return TD_ERR_PTR(TD_ERR_IO);

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return TD_ERR_PTR(TD_ERR_IO);
    }
    size_t file_size = (size_t)st.st_size;

    /* ---- 2. mmap the file ---- */
    char* buf = (char*)mmap(NULL, file_size, PROT_READ, MMAP_FLAGS, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) return TD_ERR_PTR(TD_ERR_IO);

#ifdef __APPLE__
    madvise(buf, file_size, MADV_SEQUENTIAL);
#endif

    const char* buf_end = buf + file_size;
    td_t* result = NULL;

    /* ---- 3. Detect delimiter ---- */
    /* Delimiter auto-detected from header row only. Files where the header
     * has a different delimiter distribution than data rows may be misdetected;
     * pass an explicit delimiter for such files.  Scanning additional data rows
     * was considered but adds complexity for a rare edge case. */
    if (delimiter == 0) {
        int commas = 0, tabs = 0;
        for (const char* p = buf; p < buf_end && *p != '\n'; p++) {
            if (*p == ',') commas++;
            if (*p == '\t') tabs++;
        }
        delimiter = (tabs > commas) ? '\t' : ',';
    }

    /* ---- 4. Count columns from first line ---- */
    int ncols = 1;
    {
        const char* p = buf;
        bool in_quote = false;
        while (p < buf_end && (in_quote || (*p != '\n' && *p != '\r'))) {
            if (*p == '"') in_quote = !in_quote;
            else if (!in_quote && *p == delimiter) ncols++;
            p++;
        }
    }
    if (ncols > CSV_MAX_COLS) {
        munmap(buf, file_size);
        close(fd);
        return TD_ERR_PTR(TD_ERR_RANGE);  /* too many columns */
    }

    /* ---- 5. Parse header row ---- */
    const char* p = buf;
    char esc_buf[8192];
    int64_t col_name_ids[CSV_MAX_COLS];

    if (header) {
        for (int c = 0; c < ncols; c++) {
            const char* fld;
            size_t flen;
            char* dyn_esc = NULL;
            p = scan_field(p, buf_end, delimiter, &fld, &flen, esc_buf, &dyn_esc);
            col_name_ids[c] = td_sym_intern(fld, flen);
            if (dyn_esc) td_sys_free(dyn_esc);
        }
        while (p < buf_end && (*p == '\r' || *p == '\n')) p++;
    } else {
        for (int c = 0; c < ncols; c++) {
            char name[32];
            snprintf(name, sizeof(name), "V%d", c + 1);
            col_name_ids[c] = td_sym_intern(name, strlen(name));
        }
    }

    size_t data_offset = (size_t)(p - buf);

    /* ---- 6. Build row offsets (memchr-accelerated) ---- */
    td_t* row_offsets_hdr = NULL;
    int64_t* row_offsets = NULL;
    int64_t n_rows = build_row_offsets(buf, file_size, data_offset,
                                        &row_offsets, &row_offsets_hdr);

    if (n_rows == 0) {
        /* Empty file → empty table */
        td_t* tbl = td_table_new(ncols);
        if (!tbl || TD_IS_ERR(tbl)) goto fail_unmap;
        for (int c = 0; c < ncols; c++) {
            td_t* empty_vec = td_vec_new(TD_F64, 0);
            if (empty_vec && !TD_IS_ERR(empty_vec)) {
                tbl = td_table_add_col(tbl, col_name_ids[c], empty_vec);
                td_release(empty_vec);
            }
        }
        munmap(buf, file_size);
        return tbl;
    }

    /* ---- 7. Resolve column types ---- */
    int8_t resolved_types[CSV_MAX_COLS];
    if (col_types_in && n_types >= ncols) {
        /* Explicit types provided by caller — validate against known types */
        for (int c = 0; c < ncols; c++) {
            int8_t t = col_types_in[c];
            if (t < TD_BOOL || t >= TD_TYPE_COUNT || t == TD_TABLE) {
                /* Invalid type constant — fall through to error */
                goto fail_offsets;
            }
            resolved_types[c] = t;
        }
    } else if (!col_types_in) {
        /* Auto-infer from sample rows */
        csv_type_t col_types[CSV_MAX_COLS];
        memset(col_types, 0, (size_t)ncols * sizeof(csv_type_t));
        /* Type inference from first 100 rows. Heterogeneous CSVs with type
         * changes after row 100 will be mistyped. Use explicit schema
         * (col_types_in) for such files. */
        int64_t sample_n = (n_rows < CSV_SAMPLE_ROWS) ? n_rows : CSV_SAMPLE_ROWS;
        for (int64_t r = 0; r < sample_n; r++) {
            const char* rp = buf + row_offsets[r];
            for (int c = 0; c < ncols; c++) {
                const char* fld;
                size_t flen;
                char* dyn_esc = NULL;
                rp = scan_field(rp, buf_end, delimiter, &fld, &flen, esc_buf, &dyn_esc);
                csv_type_t t = detect_type(fld, flen);
                if (dyn_esc) td_sys_free(dyn_esc);
                col_types[c] = promote_csv_type(col_types[c], t);
            }
        }
        for (int c = 0; c < ncols; c++) {
            switch (col_types[c]) {
                case CSV_TYPE_BOOL:      resolved_types[c] = TD_BOOL;      break;
                case CSV_TYPE_I64:       resolved_types[c] = TD_I64;       break;
                case CSV_TYPE_F64:       resolved_types[c] = TD_F64;       break;
                case CSV_TYPE_DATE:      resolved_types[c] = TD_DATE;      break;
                case CSV_TYPE_TIME:      resolved_types[c] = TD_TIME;      break;
                case CSV_TYPE_TIMESTAMP: resolved_types[c] = TD_TIMESTAMP; break;
                default:                 resolved_types[c] = TD_SYM;       break;
            }
        }
    } else {
        /* col_types_in provided but too short — error */
        goto fail_offsets;
    }

    /* ---- 8. Allocate column vectors ---- */
    td_t* col_vecs[CSV_MAX_COLS];
    void* col_data[CSV_MAX_COLS];

    for (int c = 0; c < ncols; c++) {
        int8_t type = resolved_types[c];
        /* String columns: allocate TD_SYM at W32 (4B/elem) for sym IDs.
         * After intern, narrow to W8/W16 if max sym ID permits. */
        col_vecs[c] = (type == TD_SYM) ? td_sym_vec_new(TD_SYM_W32, n_rows)
                                        : td_vec_new(type, n_rows);
        if (!col_vecs[c] || TD_IS_ERR(col_vecs[c])) {
            for (int j = 0; j < c; j++) td_release(col_vecs[j]);
            goto fail_offsets;
        }
        /* len set early so parallel workers can write to full extent;
         * parse errors return before table is used. */
        col_vecs[c]->len = n_rows;
        col_data[c] = td_data(col_vecs[c]);
    }

    /* ---- 8b. Pre-allocate nullmaps for all columns ---- */
    uint8_t* col_nullmaps[CSV_MAX_COLS];
    bool col_had_null[CSV_MAX_COLS];
    memset(col_had_null, 0, (size_t)ncols * sizeof(bool));

    for (int c = 0; c < ncols; c++) {
        td_t* vec = col_vecs[c];
        if (n_rows <= 128) {
            vec->attrs |= TD_ATTR_HAS_NULLS;
            memset(vec->nullmap, 0, 16);
            col_nullmaps[c] = vec->nullmap;
        } else {
            size_t bmp_bytes = ((size_t)n_rows + 7) / 8;
            td_t* ext = td_vec_new(TD_U8, (int64_t)bmp_bytes);
            if (!ext || TD_IS_ERR(ext)) {
                for (int j = 0; j <= c; j++) td_release(col_vecs[j]);
                goto fail_offsets;
            }
            ext->len = (int64_t)bmp_bytes;
            memset(td_data(ext), 0, bmp_bytes);
            vec->ext_nullmap = ext;
            vec->attrs |= TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT;
            col_nullmaps[c] = (uint8_t*)td_data(ext);
        }
    }

    /* Build csv_type_t array for parse functions (maps td types → csv types) */
    csv_type_t parse_types[CSV_MAX_COLS];
    for (int c = 0; c < ncols; c++) {
        switch (resolved_types[c]) {
            case TD_BOOL:      parse_types[c] = CSV_TYPE_BOOL;      break;
            case TD_I64:       parse_types[c] = CSV_TYPE_I64;       break;
            case TD_F64:       parse_types[c] = CSV_TYPE_F64;       break;
            case TD_DATE:      parse_types[c] = CSV_TYPE_DATE;      break;
            case TD_TIME:      parse_types[c] = CSV_TYPE_TIME;      break;
            case TD_TIMESTAMP: parse_types[c] = CSV_TYPE_TIMESTAMP; break;
            default:           parse_types[c] = CSV_TYPE_STR;       break;
        }
    }

    /* ---- 9. Parse data ---- */
    int64_t sym_max_ids[CSV_MAX_COLS];
    memset(sym_max_ids, 0, (size_t)ncols * sizeof(int64_t));

    /* Check if any string columns exist */
    int has_str_cols = 0;
    for (int c = 0; c < ncols; c++) {
        if (parse_types[c] == CSV_TYPE_STR) { has_str_cols = 1; break; }
    }

    /* Allocate strref arrays for string columns (temporary, freed after intern) */
    csv_strref_t* str_ref_bufs[CSV_MAX_COLS];
    td_t* str_ref_hdrs[CSV_MAX_COLS];
    memset(str_ref_bufs, 0, sizeof(str_ref_bufs));
    memset(str_ref_hdrs, 0, sizeof(str_ref_hdrs));
    for (int c = 0; c < ncols; c++) {
        if (parse_types[c] == CSV_TYPE_STR) {
            size_t sz = (size_t)n_rows * sizeof(csv_strref_t);
            str_ref_bufs[c] = (csv_strref_t*)scratch_alloc(&str_ref_hdrs[c], sz);
            if (!str_ref_bufs[c]) {
                for (int j = 0; j < ncols; j++) td_release(col_vecs[j]);
                for (int j = 0; j < c; j++) scratch_free(str_ref_hdrs[j]);
                goto fail_offsets;
            }
        }
    }

    {
        td_pool_t* pool = td_pool_get();
        bool use_parallel = pool && n_rows > 8192;

        if (use_parallel) {
            uint32_t n_workers = td_pool_total_workers(pool);
            size_t whn_sz = (size_t)n_workers * (size_t)ncols * sizeof(bool);
            bool* worker_had_null_buf = (bool*)td_sys_alloc(whn_sz);
            if (!worker_had_null_buf) {
                use_parallel = false;
            } else {
                memset(worker_had_null_buf, 0, whn_sz);

                csv_par_ctx_t ctx = {
                    .buf              = buf,
                    .buf_size         = file_size,
                    .row_offsets      = row_offsets,
                    .n_rows           = n_rows,
                    .n_cols           = ncols,
                    .delim            = delimiter,
                    .col_types        = parse_types,
                    .col_data         = col_data,
                    .str_refs         = str_ref_bufs,
                    .col_nullmaps     = col_nullmaps,
                    .worker_had_null  = worker_had_null_buf,
                };

                td_pool_dispatch(pool, csv_parse_fn, &ctx, n_rows);

                /* OR worker null flags into col_had_null */
                for (uint32_t w = 0; w < n_workers; w++) {
                    for (int c = 0; c < ncols; c++) {
                        if (worker_had_null_buf[(size_t)w * (size_t)ncols + (size_t)c])
                            col_had_null[c] = true;
                    }
                }
                td_sys_free(worker_had_null_buf);
            }
        }

        if (!use_parallel) {
            csv_parse_serial(buf, file_size, row_offsets, n_rows,
                             ncols, delimiter, parse_types, col_data,
                             str_ref_bufs, col_nullmaps, col_had_null);
        }
    }

    /* ---- 9b. Batch-intern string columns ---- */
    if (has_str_cols) {
        csv_intern_strings(str_ref_bufs, ncols, parse_types,
                           col_data, n_rows, sym_max_ids, col_nullmaps);
    }

    /* Free strref buffers */
    for (int c = 0; c < ncols; c++) scratch_free(str_ref_hdrs[c]);

    /* ---- 9c. Strip nullmaps from all-valid columns ---- */
    for (int c = 0; c < ncols; c++) {
        if (col_had_null[c]) continue;
        td_t* vec = col_vecs[c];
        if (vec->attrs & TD_ATTR_NULLMAP_EXT) {
            td_release(vec->ext_nullmap);
            vec->ext_nullmap = NULL;
        }
        vec->attrs &= (uint8_t)~(TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT);
        memset(vec->nullmap, 0, 16);
    }

    /* ---- 10. Narrow sym columns to optimal width ---- */
    for (int c = 0; c < ncols; c++) {
        if (resolved_types[c] != TD_SYM) continue;
        uint8_t new_w = td_sym_dict_width(sym_max_ids[c]);
        if (new_w >= TD_SYM_W32) continue; /* already at W32, no savings */
        td_t* narrow = td_sym_vec_new(new_w, n_rows);
        if (!narrow || TD_IS_ERR(narrow)) continue;
        narrow->len = n_rows;
        const uint32_t* src = (const uint32_t*)col_data[c];
        void* dst = td_data(narrow);
        if (new_w == TD_SYM_W8) {
            uint8_t* d = (uint8_t*)dst;
            for (int64_t r = 0; r < n_rows; r++) d[r] = (uint8_t)src[r];
        } else { /* TD_SYM_W16 */
            uint16_t* d = (uint16_t*)dst;
            for (int64_t r = 0; r < n_rows; r++) d[r] = (uint16_t)src[r];
        }
        /* Transfer nullmap to narrowed vector */
        if (col_vecs[c]->attrs & TD_ATTR_HAS_NULLS) {
            narrow->attrs |= (col_vecs[c]->attrs & (TD_ATTR_HAS_NULLS | TD_ATTR_NULLMAP_EXT));
            if (col_vecs[c]->attrs & TD_ATTR_NULLMAP_EXT) {
                narrow->ext_nullmap = col_vecs[c]->ext_nullmap;
                td_retain(narrow->ext_nullmap);
            } else {
                memcpy(narrow->nullmap, col_vecs[c]->nullmap, 16);
            }
        }
        td_release(col_vecs[c]);
        col_vecs[c] = narrow;
        col_data[c] = dst;
    }

    /* ---- 11. Build table ---- */
    {
        td_t* tbl = td_table_new(ncols);
        if (!tbl || TD_IS_ERR(tbl)) {
            for (int c = 0; c < ncols; c++) td_release(col_vecs[c]);
            goto fail_offsets;
        }

        for (int c = 0; c < ncols; c++) {
            tbl = td_table_add_col(tbl, col_name_ids[c], col_vecs[c]);
            td_release(col_vecs[c]);
        }

        result = tbl;
    }

    /* ---- 12. Cleanup ---- */
    scratch_free(row_offsets_hdr);
    munmap(buf, file_size);
    return result;

    /* Error paths */
fail_offsets:
    scratch_free(row_offsets_hdr);
fail_unmap:
    munmap(buf, file_size);
    return TD_ERR_PTR(TD_ERR_OOM);
}

/* --------------------------------------------------------------------------
 * td_read_csv — convenience wrapper with default options
 * -------------------------------------------------------------------------- */

td_t* td_read_csv(const char* path) {
    return td_read_csv_opts(path, 0, true, NULL, 0);
}

/* ============================================================================
 * td_write_csv — Write a table to a CSV file (RFC 4180)
 *
 * Writes header row with column names, then data rows.
 * Strings containing commas, quotes, or newlines are quoted.
 * Returns TD_OK on success, error code on failure.
 * ============================================================================ */

/* Write a string value, quoting if it contains special chars */
static void csv_write_str(FILE* fp, const char* s, size_t len) {
    int need_quote = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == ',' || s[i] == '"' || s[i] == '\n' || s[i] == '\r') {
            need_quote = 1;
            break;
        }
    }
    if (need_quote) {
        fputc('"', fp);
        for (size_t i = 0; i < len; i++) {
            if (s[i] == '"') fputc('"', fp);
            fputc(s[i], fp);
        }
        fputc('"', fp);
    } else {
        fwrite(s, 1, len, fp);
    }
}

td_err_t td_write_csv(td_t* table, const char* path) {
    if (!table || !path) return TD_ERR_TYPE;

    int64_t ncols = td_table_ncols(table);
    int64_t nrows = td_table_nrows(table);
    if (ncols <= 0) return TD_ERR_TYPE;

    FILE* fp = fopen(path, "w");
    if (!fp) return TD_ERR_IO;

    /* Header row: column names */
    for (int64_t c = 0; c < ncols; c++) {
        if (c > 0) fputc(',', fp);
        int64_t name_id = td_table_col_name(table, c);
        td_t* name_atom = td_sym_str(name_id);
        if (name_atom) {
            const char* s = td_str_ptr(name_atom);
            size_t slen = td_str_len(name_atom);
            csv_write_str(fp, s, slen);
        }
    }
    fputc('\n', fp);

    /* Data rows */
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t c = 0; c < ncols; c++) {
            if (c > 0) fputc(',', fp);
            td_t* col = td_table_get_col_idx(table, c);
            if (!col) continue;
            int8_t t = col->type;
            switch (t) {
            case TD_I64: {
                int64_t v = ((const int64_t*)td_data(col))[r];
                fprintf(fp, "%ld", (long)v);
                break;
            }
            case TD_I32: {
                int32_t v = ((const int32_t*)td_data(col))[r];
                fprintf(fp, "%d", v);
                break;
            }
            case TD_F64: {
                double v = ((const double*)td_data(col))[r];
                fprintf(fp, "%.17g", v);
                break;
            }
            case TD_BOOL: case TD_U8: {
                uint8_t v = ((const uint8_t*)td_data(col))[r];
                if (t == TD_BOOL) fputs(v ? "true" : "false", fp);
                else fprintf(fp, "%u", (unsigned)v);
                break;
            }
            case TD_DATE: {
                int32_t v = ((const int32_t*)td_data(col))[r];
                /* days since 2000-01-01 → YYYY-MM-DD */
                int32_t y, m, d;
                { /* civil_from_days: algorithm from Howard Hinnant */
                    int32_t z = v + 10957 + 719468;
                    int32_t era = (z >= 0 ? z : z - 146096) / 146097;
                    uint32_t doe = (uint32_t)(z - era * 146097);
                    uint32_t yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
                    y = (int32_t)yoe + era * 400;
                    uint32_t doy = doe - (365*yoe + yoe/4 - yoe/100);
                    uint32_t mp = (5*doy + 2) / 153;
                    d = (int32_t)(doy - (153*mp + 2)/5 + 1);
                    m = (int32_t)(mp < 10 ? mp + 3 : mp - 9);
                    if (m <= 2) y++;
                }
                fprintf(fp, "%04d-%02d-%02d", y, m, d);
                break;
            }
            case TD_TIME: {
                int32_t ms = ((const int32_t*)td_data(col))[r];
                uint32_t ums = (uint32_t)ms;
                uint32_t h = ums / 3600000;
                uint32_t mi = (ums % 3600000) / 60000;
                uint32_t s = (ums % 60000) / 1000;
                uint32_t frac = ums % 1000;
                if (frac) fprintf(fp, "%02u:%02u:%02u.%03u", h, mi, s, frac);
                else      fprintf(fp, "%02u:%02u:%02u", h, mi, s);
                break;
            }
            case TD_TIMESTAMP: {
                int64_t us = ((const int64_t*)td_data(col))[r];
                int32_t days = (int32_t)(us / 86400000000LL);
                int64_t time_us = us % 86400000000LL;
                if (time_us < 0) { days--; time_us += 86400000000LL; }
                /* days since 2000-01-01 → YYYY-MM-DD */
                int32_t y, mo, d;
                {
                    int32_t z = days + 10957 + 719468;
                    int32_t era = (z >= 0 ? z : z - 146096) / 146097;
                    uint32_t doe = (uint32_t)(z - era * 146097);
                    uint32_t yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
                    y = (int32_t)yoe + era * 400;
                    uint32_t doy = doe - (365*yoe + yoe/4 - yoe/100);
                    uint32_t mp = (5*doy + 2) / 153;
                    d = (int32_t)(doy - (153*mp + 2)/5 + 1);
                    mo = (int32_t)(mp < 10 ? mp + 3 : mp - 9);
                    if (mo <= 2) y++;
                }
                uint64_t tus = (uint64_t)time_us;
                uint32_t h = (uint32_t)(tus / 3600000000ULL);
                uint32_t mi = (uint32_t)((tus % 3600000000ULL) / 60000000ULL);
                uint32_t s = (uint32_t)((tus % 60000000ULL) / 1000000ULL);
                uint32_t frac = (uint32_t)(tus % 1000000ULL);
                if (frac) fprintf(fp, "%04d-%02d-%02dT%02u:%02u:%02u.%06u", y, mo, d, h, mi, s, frac);
                else      fprintf(fp, "%04d-%02d-%02dT%02u:%02u:%02u", y, mo, d, h, mi, s);
                break;
            }
            case TD_I16: {
                int16_t v = ((const int16_t*)td_data(col))[r];
                fprintf(fp, "%d", (int)v);
                break;
            }
            case TD_SYM: {
                int64_t sym = td_read_sym(td_data(col), r, col->type, col->attrs);
                td_t* s = td_sym_str(sym);
                if (s) csv_write_str(fp, td_str_ptr(s), td_str_len(s));
                break;
            }
            default:
                break;
            }
        }
        fputc('\n', fp);
    }

    fclose(fp);
    return TD_OK;
}
