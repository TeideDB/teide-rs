//   Copyright (c) 2024-2026 Anton Kundenko <singaraiona@gmail.com>
//   MIT License

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("vendor/teide");

    // Vendor the C core if not already present
    if !vendor.exists() {
        let status = Command::new("git")
            .args(["clone", "--depth=1", "https://github.com/TeideDB/teide.git"])
            .arg(&vendor)
            .status()
            .expect("failed to run git clone");
        assert!(status.success(), "git clone teide failed");
    }

    let src_dir = vendor.join("src");
    let include_dir = vendor.join("include");

    let c_files: Vec<PathBuf> = walkdir(&src_dir);

    let mut build = cc::Build::new();
    build.include(&include_dir).include(&src_dir).std("c17");

    let profile = std::env::var("PROFILE").unwrap_or_default();
    if profile == "debug" {
        build.flag("-O0").flag("-g");
    } else {
        build
            .flag("-O3")
            .flag("-mtune=generic")
            .define("NDEBUG", None);
    }

    for f in &c_files {
        build.file(f);
    }

    build.compile("teide");

    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-changed={}", include_dir.display());

    // Embed git commit hash for CLI banner
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GIT_HASH={}", hash.trim());

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");
}

/// Recursively collect all `.c` files under `dir`.
fn walkdir(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walkdir(&path));
            } else if path.extension().is_some_and(|e| e == "c") {
                out.push(path);
            }
        }
    }
    out
}
