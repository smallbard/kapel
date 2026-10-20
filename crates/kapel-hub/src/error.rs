//! Errors of reading and writing model files.

use std::fmt;
use std::path::PathBuf;

/// What can go wrong with a model file.
#[derive(Debug)]
#[non_exhaustive]
pub enum HubError {
    /// The file could not be read or written.
    Io {
        /// The file.
        path: PathBuf,
        /// The system error.
        source: std::io::Error,
    },
    /// The file does not follow its format.
    Format {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },
    /// The file holds no tensor or entry of this name.
    Missing {
        /// The name that was asked for.
        name: String,
    },
    /// A tensor operation failed.
    Tensor(kapel_core::Error),
}

impl HubError {
    pub(crate) fn format(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Format {
            path: path.into(),
            message: message.into(),
        }
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Format { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Missing { name } => write!(f, "no entry named {name:?}"),
            Self::Tensor(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HubError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<kapel_core::Error> for HubError {
    fn from(error: kapel_core::Error) -> Self {
        Self::Tensor(error)
    }
}

/// The result type of `kapel-hub`.
pub type Result<T> = std::result::Result<T, HubError>;
