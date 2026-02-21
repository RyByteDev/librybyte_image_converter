use std::process::Command;
use std::path::Path;

fn main() {
    let include_dir = Path::new("include");
    if !include_dir.exists() {
        std::fs::create_dir_all(include_dir).unwrap();
    }

    Command::new("cbindgen")
        .args(&[
            "--config", "cbindgen.toml",
            "--crate", "rybyte_image_converter",
            "--output", "include/rybyte_image_converter.h",
            "--lang", "C"
        ])
        .status()
        .expect("cbindgen failed for C");

    Command::new("cbindgen")
        .args(&[
            "--config", "cbindgen.toml",
            "--crate", "rybyte_image_converter",
            "--output", "include/rybyte_image_converter.hpp",
            "--lang", "C++"
        ])
        .status()
        .expect("cbindgen failed for C++");

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
