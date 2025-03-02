extern crate bindgen;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-search=./lib");
    println!("cargo:rustc-link-lib=dylib=gmsh");

    let bindings = bindgen::Builder::default()
    .header("include/wrapper.h")
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate().expect("Unable to generate bindings");

    let out_file = PathBuf::from("./src/gmsh_bindings.rs");
    bindings.write_to_file(out_file).expect("Failed to write bindings");
}