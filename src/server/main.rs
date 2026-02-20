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

//! Teide PostgreSQL wire protocol server binary.
//!
//! Usage:
//!     teide-server [OPTIONS]
//!
//! Options:
//!     --host <HOST>       Listen address (default: 127.0.0.1)
//!     --port <PORT>       Listen port (default: 5433)
//!     --load NAME=PATH    Load a CSV file as a named table (repeatable)
//!     --init <FILE>       Execute a SQL init script at startup
//!     --verbose           Enable verbose logging

use std::sync::Arc;

use clap::Parser;
use tokio::net::TcpListener;

use teide::server::handler::TeideHandlerFactory;
use teide::sql::Session;

#[derive(Parser)]
#[command(
    name = "teide-server",
    version,
    about = "PostgreSQL wire protocol server for the Teide columnar engine"
)]
struct Args {
    /// Listen address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port
    #[arg(long, default_value_t = 5433)]
    port: u16,

    /// Load a CSV file as a named table (NAME=PATH, repeatable)
    #[arg(long = "load", value_name = "NAME=PATH")]
    load: Vec<String>,

    /// Execute a SQL init script at startup
    #[arg(long)]
    init: Option<String>,

    /// Enable verbose logging
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize the Teide engine session on the main thread.
    let mut session = match Session::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: failed to initialize Teide engine: {e}");
            std::process::exit(1);
        }
    };

    // Process --load arguments: NAME=PATH
    for spec in &args.load {
        let (name, path) = match spec.split_once('=') {
            Some(pair) => pair,
            None => {
                eprintln!("Error: --load expects NAME=PATH, got: {spec}");
                std::process::exit(1);
            }
        };
        let abs_path = match std::fs::canonicalize(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error: cannot resolve path '{path}': {e}");
                std::process::exit(1);
            }
        };
        let abs_str = abs_path.display();
        let sql = format!("CREATE TABLE {name} AS SELECT * FROM '{abs_str}'");
        if args.verbose {
            eprintln!("[init] {sql}");
        }
        match session.execute(&sql) {
            Ok(teide::ExecResult::Ddl(msg)) => {
                if args.verbose {
                    eprintln!("[init] {msg}");
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error loading {name}={path}: {e}");
                std::process::exit(1);
            }
        }
    }

    // Process --init script
    if let Some(ref init_path) = args.init {
        let path = std::path::Path::new(init_path);
        if args.verbose {
            eprintln!("[init] Executing script: {}", path.display());
        }
        if let Err(e) = session.execute_script_file(path) {
            eprintln!("Error in init script {}: {e}", path.display());
            std::process::exit(1);
        }
    }

    // Report loaded tables
    let table_names = session.table_names();
    if !table_names.is_empty() {
        eprintln!(
            "Loaded {} table(s): {}",
            table_names.len(),
            table_names.join(", ")
        );
    }

    // Build the handler factory from the base session
    let factory = Arc::new(TeideHandlerFactory::from_session(&session));

    // Bind TCP listener
    let addr = format!("{}:{}", args.host, args.port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error: failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Teide server listening on {addr}");
    eprintln!("Connect with: psql -h {} -p {}", args.host, args.port);

    // Accept loop with graceful shutdown on Ctrl+C
    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((socket, peer)) => {
                        if args.verbose {
                            eprintln!("[conn] New connection from {peer}");
                        }
                        let factory = factory.clone();
                        tokio::spawn(async move {
                            if let Err(e) =
                                pgwire::tokio::process_socket(socket, None, factory).await
                            {
                                eprintln!("[conn] Error handling {peer}: {e}");
                            }
                            if false {
                                // verbose flag not easily available here; could
                                // be threaded through factory if needed.
                                eprintln!("[conn] Disconnected: {peer}");
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("Error accepting connection: {e}");
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nShutting down...");
                break;
            }
        }
    }
}
