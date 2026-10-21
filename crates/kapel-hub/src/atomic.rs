//! Writing a file in one step: readers see the old contents or the new ones,
//! never a mix.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{HubError, Result};

/// Writes the file `path` with `write`, through a temporary file in the same
/// directory renamed into place once complete and flushed to disk.
///
/// The old file, if any, is replaced, not truncated: a process that has it
/// mapped in memory (a model saved over its own weights) keeps reading the old
/// contents. On Windows, a file that is mapped cannot be replaced, and the
/// rename fails with an error.
///
/// # Errors
///
/// [`HubError::Io`] if the temporary file cannot be written or renamed; it is
/// removed then.
pub fn write_atomic(
    path: impl AsRef<Path>,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
) -> Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let path = path.as_ref();
    let name = path.file_name().ok_or_else(|| {
        HubError::io(
            path,
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a file name"),
        )
    })?;
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let mut temporary_name = std::ffi::OsString::from(".");
    temporary_name.push(name);
    temporary_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = dir.join(temporary_name);
    let result = (|| -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut buffered = std::io::BufWriter::new(file);
        write(&mut buffered)?;
        let file = buffered.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(HubError::io(path, error));
    }
    // The rename is durable once the directory is flushed too; not every system
    // can open a directory for that, and the file is complete either way.
    #[cfg(unix)]
    if let Ok(dir) = std::fs::File::open(dir) {
        let _ = dir.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_replaced_whole_and_no_temporary_is_left() {
        let dir = std::env::temp_dir().join(format!("kapel-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.txt");
        write_atomic(&path, |w| w.write_all(b"first")).unwrap();
        // A reader of the old file keeps its contents.
        let mut old = std::fs::File::open(&path).unwrap();
        write_atomic(&path, |w| w.write_all(b"second, longer")).unwrap();
        let mut text = String::new();
        std::io::Read::read_to_string(&mut old, &mut text).unwrap();
        #[cfg(unix)]
        assert_eq!(text, "first");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second, longer");
        // A failed write leaves the old file and no temporary.
        let failed = write_atomic(&path, |w| {
            w.write_all(b"partial")?;
            Err(std::io::Error::other("stop"))
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second, longer");
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(names.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
