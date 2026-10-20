//! Shape rules shared by the operations.

use kapel_core::{Error, Result};

/// The shape two shapes broadcast to: dimensions are aligned from the last, and
/// each pair must be equal or contain a 1.
///
/// # Errors
///
/// [`Error::BroadcastIncompatible`] if two aligned dimensions differ and neither
/// is 1.
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        let lead = rank - s.len();
        if i < lead { 1 } else { s[i - lead] }
    };
    (0..rank)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(Error::BroadcastIncompatible {
                from: a.to_vec(),
                to: b.to_vec(),
            }),
        })
        .collect()
}

/// Checks that `dims` are distinct dimensions of a tensor of rank `rank`.
///
/// # Errors
///
/// [`Error::InvalidDim`] or [`Error::DuplicateDim`].
pub fn check_dims(op: &'static str, dims: &[usize], rank: usize) -> Result<()> {
    let mut seen = vec![false; rank];
    for &dim in dims {
        if dim >= rank {
            return Err(Error::InvalidDim { op, dim, rank });
        }
        if std::mem::replace(&mut seen[dim], true) {
            return Err(Error::DuplicateDim { op, dim });
        }
    }
    Ok(())
}

/// The shape left after reducing `dims`, kept as size 1 with `keepdim`.
pub fn reduced(shape: &[usize], dims: &[usize], keepdim: bool) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| match (dims.contains(&i), keepdim) {
            (false, _) => Some(n),
            (true, true) => Some(1),
            (true, false) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_aligns_from_the_last_dimension() {
        assert_eq!(broadcast(&[3, 1], &[2, 1, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast(&[], &[5]).unwrap(), vec![5]);
        assert!(broadcast(&[3], &[4]).is_err());
    }

    #[test]
    fn reduced_shapes() {
        assert_eq!(reduced(&[2, 3, 4], &[1], false), vec![2, 4]);
        assert_eq!(reduced(&[2, 3, 4], &[0, 2], true), vec![1, 3, 1]);
    }
}
