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

// Terminal color theme using standard ANSI palette colors.
//
// Uses the terminal's own 16-color palette so colors automatically adapt
// when the user switches terminal profiles (dark / light) mid-session.
// No background detection needed.

// Formatting
pub const BOLD: &str = "\x1b[1m";
pub const ITALIC: &str = "\x1b[3m";
pub const R: &str = "\x1b[0m";

// Table structure
pub const BORDER: &str = "\x1b[90m"; // bright black (gray)
pub const HEADER: &str = "\x1b[1;36m"; // bold cyan
pub const TYPE_DIM: &str = "\x1b[90m"; // gray
pub const TEXT: &str = "\x1b[39m"; // default foreground
pub const NULL_CLR: &str = "\x1b[90m"; // gray
pub const FOOTER: &str = "\x1b[90m"; // gray

// Status
pub const ERROR: &str = "\x1b[1;31m"; // bold red
pub const SUCCESS: &str = "\x1b[32m"; // green
pub const TIMER: &str = "\x1b[90m"; // gray

// Banner
pub const BAN_BORDER: &str = "\x1b[34m"; // blue
pub const BAN_TITLE: &str = "\x1b[1;36m"; // bold cyan
pub const BAN_INFO: &str = "\x1b[39m"; // default foreground
pub const BAN_HELP: &str = "\x1b[90m"; // gray

// Aliases for help text
pub const NORD3: &str = "\x1b[90m"; // gray
pub const NORD7: &str = "\x1b[36m"; // cyan
