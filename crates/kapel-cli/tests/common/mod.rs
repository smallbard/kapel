//! What the tests of the command line share.

#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

/// Runs `kapel` with `args` in `dir`: whether it succeeded, and what it
/// printed.
pub(crate) fn kapel(args: &[&str], dir: &Path) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_kapel"))
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

/// A file of llm.c's tokens with some structure.
pub(crate) fn tokens(path: &Path, count: usize, seed: u64) {
    let mut bytes = vec![0u8; 1024];
    bytes[0..4].copy_from_slice(&20_240_520i32.to_le_bytes());
    bytes[4..8].copy_from_slice(&1i32.to_le_bytes());
    bytes[8..12].copy_from_slice(&(count as i32).to_le_bytes());
    let mut state = seed;
    let mut t = 0u16;
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        t = if (state >> 33) % 10 < 7 {
            (t * 3 + 1) % 50
        } else {
            ((state >> 40) % 50) as u16
        };
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}
