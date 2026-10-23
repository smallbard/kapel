//! Shapes and strided layouts.
//!
//! A [`Layout`] says where each element of a tensor sits in its storage: element
//! `(i₀, i₁, …)` is at `offset + i₀·s₀ + i₁·s₁ + …`, counted in elements. Views
//! (transpose, slice, broadcast) change the layout, never the data.

use crate::{Error, Result};

/// The shape, strides and offset of a tensor in its storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    /// The row-major layout of `shape`: the last dimension is contiguous.
    ///
    /// # Errors
    ///
    /// [`Error::Overflow`] if the number of elements does not fit in `usize`.
    pub fn contiguous(shape: &[usize]) -> Result<Self> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1usize;
        for (s, &size) in strides.iter_mut().zip(shape).rev() {
            *s = stride;
            stride = stride.checked_mul(size).ok_or(Error::Overflow {
                op: "Layout::contiguous",
            })?;
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        })
    }

    /// The size of each dimension.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The step, in elements, between two neighbours along each dimension. A
    /// broadcast dimension has stride 0.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// The position of the first element in the storage, in elements.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the elements are stored in row-major order, without gaps. The
    /// offset may be anything: a contiguous view can start inside a storage.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for (&size, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if size != 1 && stride != expected {
                return false;
            }
            expected *= size;
        }
        true
    }

    /// The number of storage elements the layout reaches: one past the last
    /// element it addresses. Zero for an empty tensor.
    pub fn storage_extent(&self) -> usize {
        if self.numel() == 0 {
            return 0;
        }
        self.offset
            + self
                .shape
                .iter()
                .zip(&self.strides)
                .map(|(&n, &s)| (n - 1) * s)
                .sum::<usize>()
            + 1
    }

    fn check_dim(&self, op: &'static str, dim: usize) -> Result<()> {
        if dim >= self.rank() {
            return Err(Error::InvalidDim {
                op,
                dim,
                rank: self.rank(),
            });
        }
        Ok(())
    }

    /// Swaps two dimensions.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDim`] if a dimension does not exist.
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self> {
        self.check_dim("transpose", a)?;
        self.check_dim("transpose", b)?;
        let mut out = self.clone();
        out.shape.swap(a, b);
        out.strides.swap(a, b);
        Ok(out)
    }

    /// Reorders the dimensions: dimension `i` of the result is dimension
    /// `order[i]` of this layout.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `order` does not name every dimension,
    /// [`Error::InvalidDim`] or [`Error::DuplicateDim`] if it is not a permutation.
    pub fn permute(&self, order: &[usize]) -> Result<Self> {
        if order.len() != self.rank() {
            return Err(Error::ShapeMismatch {
                op: "permute",
                expected: vec![self.rank()],
                got: vec![order.len()],
            });
        }
        let mut seen = vec![false; self.rank()];
        for &dim in order {
            self.check_dim("permute", dim)?;
            if std::mem::replace(&mut seen[dim], true) {
                return Err(Error::DuplicateDim { op: "permute", dim });
            }
        }
        Ok(Self {
            shape: order.iter().map(|&d| self.shape[d]).collect(),
            strides: order.iter().map(|&d| self.strides[d]).collect(),
            offset: self.offset,
        })
    }

    /// Keeps `len` elements of dimension `dim`, from `start`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDim`] or [`Error::OutOfRange`].
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        self.check_dim("narrow", dim)?;
        let size = self.shape[dim];
        if start.checked_add(len).is_none_or(|end| end > size) {
            return Err(Error::OutOfRange {
                op: "narrow",
                dim,
                start,
                len,
                size,
            });
        }
        let mut out = self.clone();
        out.shape[dim] = len;
        if len > 0 {
            out.offset += start * self.strides[dim];
        }
        Ok(out)
    }

    /// Repeats the layout to `shape` without copying: dimensions are aligned from
    /// the last, and each must be equal to the target or 1. New leading
    /// dimensions and dimensions of size 1 get stride 0.
    ///
    /// # Errors
    ///
    /// [`Error::BroadcastIncompatible`], or [`Error::Overflow`] if the target
    /// shape has more elements than `usize` can count.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let incompatible = || Error::BroadcastIncompatible {
            op: "broadcast_to",
            from: self.shape.clone(),
            to: shape.to_vec(),
        };
        if shape.len() < self.rank() {
            return Err(incompatible());
        }
        shape
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .ok_or(Error::Overflow { op: "broadcast_to" })?;
        let lead = shape.len() - self.rank();
        let mut strides = vec![0; shape.len()];
        for (i, (&size, &stride)) in self.shape.iter().zip(&self.strides).enumerate() {
            let target = shape[lead + i];
            strides[lead + i] = match size {
                s if s == target => stride,
                1 => 0,
                _ => return Err(incompatible()),
            };
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// The same elements with another shape, in the same row-major order,
    /// without moving them. A contiguous layout can take any shape with the
    /// same number of elements. A strided view can take the shapes that split
    /// or merge only dimensions that continue one another (`stride[i] =
    /// stride[i + 1] · size[i + 1]`): the columns of a matrix narrowed from a
    /// wider one can be cut into heads, but the rows of a transpose cannot be
    /// merged; those need a copy.
    ///
    /// # Errors
    ///
    /// [`Error::ElementCountMismatch`], or [`Error::NotContiguous`] for a
    /// strided view that cannot take the shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .ok_or(Error::Overflow { op: "reshape" })?;
        if count != self.numel() {
            return Err(Error::ElementCountMismatch {
                op: "reshape",
                expected: self.numel(),
                got: count,
            });
        }
        let mut out = Self::contiguous(shape)?;
        out.offset = self.offset;
        if self.is_contiguous() || count == 0 {
            return Ok(out);
        }
        out.strides = view_strides(&self.shape, &self.strides, shape)
            .ok_or(Error::NotContiguous { op: "reshape" })?;
        Ok(out)
    }

    /// The storage position of every element, in row-major order of the shape.
    pub fn offsets(&self) -> Offsets<'_> {
        Offsets {
            layout: self,
            index: vec![0; self.rank()],
            next: (self.numel() > 0).then_some(self.offset),
        }
    }
}

/// The storage positions of a layout's elements, in row-major order. See
/// [`Layout::offsets`].
#[derive(Debug)]
pub struct Offsets<'a> {
    layout: &'a Layout,
    index: Vec<usize>,
    next: Option<usize>,
}

impl Iterator for Offsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let current = self.next?;
        // Advance the index like an odometer, last dimension first.
        let mut position = current;
        self.next = None;
        for dim in (0..self.layout.rank()).rev() {
            let stride = self.layout.strides[dim];
            if self.index[dim] + 1 < self.layout.shape[dim] {
                self.index[dim] += 1;
                self.next = Some(position + stride);
                break;
            }
            position -= self.index[dim] * stride;
            self.index[dim] = 0;
        }
        Some(current)
    }
}

/// The strides of `new`, a reshape of the non-empty layout `old` (`shape`,
/// `strides`) that keeps its elements in place, if there are some: the old
/// dimensions are taken in groups that continue one another, from the last,
/// and the new dimensions must tile each group exactly.
fn view_strides(shape: &[usize], strides: &[usize], new: &[usize]) -> Option<Vec<usize>> {
    let mut out = vec![0; new.len()];
    let mut next = new.len();
    let (mut base, mut seen, mut placed) = (strides.last().copied().unwrap_or(1), 1, 1);
    for d in (0..shape.len()).rev() {
        seen *= shape[d];
        let group_ends = d == 0 || (shape[d - 1] != 1 && strides[d - 1] != seen * base);
        if group_ends {
            while next > 0 && (placed < seen || new[next - 1] == 1) {
                next -= 1;
                out[next] = placed * base;
                placed *= new[next];
            }
            if placed != seen {
                return None;
            }
            if d > 0 {
                base = strides[d - 1];
                seen = 1;
                placed = 1;
            }
        }
    }
    // Dimensions of size 1 left over, before every other.
    while next > 0 && new[next - 1] == 1 {
        next -= 1;
        out[next] = placed * base;
    }
    (next == 0).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offsets(layout: &Layout) -> Vec<usize> {
        layout.offsets().collect()
    }

    #[test]
    fn contiguous_layouts_are_row_major() {
        let layout = Layout::contiguous(&[2, 3, 4]).unwrap();
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
        assert_eq!(offsets(&layout), (0..24).collect::<Vec<_>>());
        assert_eq!(layout.storage_extent(), 24);
    }

    #[test]
    fn a_transpose_reads_columns() {
        let layout = Layout::contiguous(&[2, 3])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(layout.shape(), &[3, 2]);
        assert!(!layout.is_contiguous());
        assert_eq!(offsets(&layout), vec![0, 3, 1, 4, 2, 5]);
        assert_eq!(
            layout.reshape(&[6]),
            Err(Error::NotContiguous { op: "reshape" })
        );
    }

    #[test]
    fn narrow_moves_the_offset() {
        let layout = Layout::contiguous(&[4, 5])
            .unwrap()
            .narrow(1, 2, 2)
            .unwrap();
        assert_eq!(offsets(&layout), vec![2, 3, 7, 8, 12, 13, 17, 18]);
        let rows = Layout::contiguous(&[4, 5])
            .unwrap()
            .narrow(0, 1, 2)
            .unwrap();
        assert!(rows.is_contiguous());
        assert_eq!(rows.offset(), 5);
        assert_eq!(rows.reshape(&[10]).unwrap().offset(), 5);
        assert!(matches!(
            Layout::contiguous(&[4]).unwrap().narrow(0, 3, 2),
            Err(Error::OutOfRange { .. })
        ));
    }

    #[test]
    fn broadcast_repeats_without_copying() {
        let layout = Layout::contiguous(&[3, 1])
            .unwrap()
            .broadcast_to(&[2, 3, 4])
            .unwrap();
        assert_eq!(layout.strides(), &[0, 1, 0]);
        assert_eq!(offsets(&layout).len(), 24);
        assert_eq!(offsets(&layout)[..8], [0, 0, 0, 0, 1, 1, 1, 1]);
        assert!(matches!(
            Layout::contiguous(&[3]).unwrap().broadcast_to(&[4]),
            Err(Error::BroadcastIncompatible { .. })
        ));
    }

    #[test]
    fn permute_checks_its_order() {
        let layout = Layout::contiguous(&[2, 3, 4]).unwrap();
        assert_eq!(layout.permute(&[2, 0, 1]).unwrap().shape(), &[4, 2, 3]);
        assert_eq!(
            layout.permute(&[0, 0, 1]),
            Err(Error::DuplicateDim {
                op: "permute",
                dim: 0
            })
        );
        assert!(matches!(
            layout.permute(&[0, 1]),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn empty_and_scalar_layouts() {
        let empty = Layout::contiguous(&[3, 0]).unwrap();
        assert_eq!(offsets(&empty), Vec::<usize>::new());
        assert_eq!(empty.storage_extent(), 0);
        let scalar = Layout::contiguous(&[]).unwrap();
        assert_eq!(offsets(&scalar), vec![0]);
        assert_eq!(scalar.numel(), 1);
    }

    #[test]
    fn a_strided_view_takes_the_shapes_that_keep_its_elements_in_place() {
        // The columns 2..8 of a [4, 10] matrix, cut into 3 heads of 2.
        let narrow = Layout::contiguous(&[4, 10])
            .unwrap()
            .narrow(1, 2, 6)
            .unwrap();
        let heads = narrow.reshape(&[4, 3, 2]).unwrap();
        assert_eq!(heads.strides(), [10, 2, 1]);
        assert_eq!(offsets(&heads), offsets(&narrow));
        // A leading 1, and the heads merged back.
        let one = heads.reshape(&[1, 4, 6]).unwrap();
        assert_eq!(offsets(&one), offsets(&narrow));
        // Rows of a narrowed matrix cannot be merged; nor those of a transpose.
        assert!(narrow.reshape(&[24]).is_err());
        let t = Layout::contiguous(&[3, 4])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(t.reshape(&[12]).is_err());
        assert_eq!(t.reshape(&[4, 3, 1]).unwrap().strides()[..2], [1, 4]);
    }

    #[test]
    fn reshape_keeps_the_count() {
        let layout = Layout::contiguous(&[2, 6]).unwrap();
        assert_eq!(layout.reshape(&[3, 4]).unwrap().strides(), &[4, 1]);
        assert_eq!(
            layout.reshape(&[5]),
            Err(Error::ElementCountMismatch {
                op: "reshape",
                expected: 12,
                got: 5
            })
        );
    }
}
