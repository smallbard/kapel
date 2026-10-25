//! The log as TensorBoard reads it: a file of events, each scalar of a line
//! of `log.jsonl` under its name. Weights & Biases imports the same file
//! (`wandb sync --sync-tensorboard`).
//!
//! The file is a series of TFRecords (`u64` length, its masked CRC-32C, the
//! bytes, their masked CRC-32C), each an `Event` of TensorFlow's protocol
//! buffers: the first carries the version of the format, the others a
//! `Summary` of scalars at a step. Both are written here by hand, a few
//! fields of each.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

/// CRC-32C (Castagnoli), as TFRecords check their lengths and data.
fn crc32c(bytes: &[u8]) -> u32 {
    const POLY: u32 = 0x82F6_3B78;
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// The CRC as TFRecords store it, rotated and offset.
fn masked(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// A field of length-delimited bytes.
fn bytes_field(out: &mut Vec<u8>, number: u32, bytes: &[u8]) {
    varint(out, u64::from(number << 3 | 2));
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// An `Event` with its time, its step, and its other fields already
/// encoded.
fn event(wall_time: f64, step: u64, rest: &[u8]) -> Vec<u8> {
    let mut e = vec![1 << 3 | 1];
    e.extend_from_slice(&wall_time.to_le_bytes());
    varint(&mut e, 2 << 3);
    varint(&mut e, step);
    e.extend_from_slice(rest);
    e
}

/// A file of events, written as the run goes.
pub struct EventWriter {
    file: BufWriter<File>,
    path: PathBuf,
}

impl EventWriter {
    /// A new file of events in `dir`, `events.out.tfevents.<seconds>.kapel`.
    ///
    /// # Errors
    ///
    /// If the file cannot be created.
    pub fn create(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let now = now();
        let path = dir.join(format!("events.out.tfevents.{}.kapel", now as u64));
        let mut writer = Self {
            file: BufWriter::new(File::create(&path)?),
            path,
        };
        let mut version = Vec::new();
        bytes_field(&mut version, 3, b"brain.Event:2");
        writer.record(&event(now, 0, &version))?;
        writer.file.flush()?;
        Ok(writer)
    }

    /// The file written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn record(&mut self, data: &[u8]) -> std::io::Result<()> {
        let length = (data.len() as u64).to_le_bytes();
        self.file.write_all(&length)?;
        self.file
            .write_all(&masked(crc32c(&length)).to_le_bytes())?;
        self.file.write_all(data)?;
        self.file.write_all(&masked(crc32c(data)).to_le_bytes())
    }

    /// The scalars of `values` at `step`, each under its name.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn scalars(&mut self, step: u64, values: &[(&str, f64)]) -> std::io::Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let mut summary = Vec::new();
        for (tag, value) in values {
            let mut v = Vec::new();
            bytes_field(&mut v, 1, tag.as_bytes());
            v.push(2 << 3 | 5);
            v.extend_from_slice(&(*value as f32).to_le_bytes());
            bytes_field(&mut summary, 1, &v);
        }
        let mut rest = Vec::new();
        bytes_field(&mut rest, 5, &summary);
        self.record(&event(now(), step, &rest))?;
        self.file.flush()
    }

    /// The numbers of a line of `log.jsonl`, at its `step`.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn line(&mut self, line: &Value) -> std::io::Result<()> {
        let Some(fields) = line.as_object() else {
            return Ok(());
        };
        let step = fields.get("step").and_then(Value::as_u64).unwrap_or(0);
        let values: Vec<(&str, f64)> = fields
            .iter()
            .filter(|(k, _)| k.as_str() != "step")
            .filter_map(|(k, v)| Some((k.as_str(), v.as_f64()?)))
            .collect();
        self.scalars(step, &values)
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_of_the_standard_check() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn records_frame_their_events() {
        let dir = std::env::temp_dir().join(format!("kapel-tb-{}", std::process::id()));
        let mut w = EventWriter::create(&dir).unwrap();
        w.line(&serde_json::json!({"step": 3, "loss": 2.5, "note": "x"}))
            .unwrap();
        let bytes = std::fs::read(w.path()).unwrap();
        // Two records, each framed by its length and checks.
        let mut at = 0;
        let mut records = 0;
        while at < bytes.len() {
            let length = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
            let check = u32::from_le_bytes(bytes[at + 8..at + 12].try_into().unwrap());
            assert_eq!(check, masked(crc32c(&bytes[at..at + 8])));
            let data = &bytes[at + 12..at + 12 + length];
            let check = u32::from_le_bytes(
                bytes[at + 12 + length..at + 16 + length]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(check, masked(crc32c(data)));
            at += 16 + length;
            records += 1;
        }
        assert_eq!(records, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
