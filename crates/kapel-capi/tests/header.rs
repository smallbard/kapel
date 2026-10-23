//! `include/kapel.h` and the library agree: every function the header
//! declares is exported, and every function exported is declared.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn read(path: &str) -> String {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()))
}

/// The names `kapel_…` that follow `before` and come before `(`.
fn names(text: &str, before: &str) -> BTreeSet<String> {
    text.match_indices(before)
        .filter_map(|(at, _)| {
            let rest = &text[at + before.len()..];
            let name: String = rest
                .trim_start_matches(['*', ' '])
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let after = &rest[rest.find(&name)? + name.len()..];
            (name.starts_with("kapel_") && after.starts_with('(')).then_some(name)
        })
        .collect()
}

#[test]
fn the_header_declares_what_the_library_exports() {
    let header = read("include/kapel.h");
    // A declaration is a type, the name, then its arguments: the names
    // after a space or a star, not inside a comment.
    let code: String = header
        .lines()
        .filter(|l| {
            let l = l.trim_start();
            !(l.starts_with("/*") || l.starts_with('*') || l.starts_with("//"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut declared = names(&code, " ");
    declared.extend(names(&code, "*"));
    let source = read("src/lib.rs");
    let exported = names(&source, "extern \"C\" fn ");
    assert!(exported.len() >= 15, "{exported:?}");
    assert_eq!(declared, exported);
    // Each export keeps its symbol.
    let unmangled = source.matches("#[unsafe(no_mangle)]").count();
    assert_eq!(unmangled, exported.len());
}

#[test]
fn the_version_of_the_header_is_the_library_s() {
    let header = read("include/kapel.h");
    let line = header
        .lines()
        .find(|l| l.starts_with("#define KAPEL_API_VERSION"))
        .unwrap();
    let version: u32 = line.rsplit(' ').next().unwrap().parse().unwrap();
    assert_eq!(version, kapel::API_VERSION);
}
