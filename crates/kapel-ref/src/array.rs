//! A dense `f64` array: the working form of every reference computation.

use kapel_core::{Error, Result, Tensor};

/// Values in row-major order, with their shape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Array {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f64>,
}

impl Array {
    /// The exact values of a host tensor.
    pub(crate) fn of(tensor: &Tensor) -> Result<Self> {
        Ok(Self {
            shape: tensor.shape().to_vec(),
            data: tensor.to_f64_vec()?,
        })
    }

    pub(crate) fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }

    pub(crate) fn new(shape: &[usize], data: Vec<f64>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// An `f64` host tensor.
    pub(crate) fn tensor(&self) -> Result<Tensor> {
        Tensor::from_slice(&self.data, &self.shape)
    }

    /// The same values with the last two dimensions swapped.
    pub(crate) fn transpose_last(&self) -> Self {
        let rank = self.shape.len();
        let (rows, cols) = (self.shape[rank - 2], self.shape[rank - 1]);
        let mut shape = self.shape.clone();
        shape.swap(rank - 2, rank - 1);
        let size = rows * cols;
        let mut data = vec![0.0; self.data.len()];
        // An empty matrix has nothing to move (and `chunks(0)` would panic).
        if size > 0 {
            for (matrix, out) in self.data.chunks(size).zip(data.chunks_mut(size)) {
                for i in 0..rows {
                    for j in 0..cols {
                        out[j * rows + i] = matrix[i * cols + j];
                    }
                }
            }
        }
        Self { shape, data }
    }
}

/// The row-major strides of a shape.
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// For each element of `to`, in row-major order, the position in `from` of the
/// element it is broadcast from.
pub(crate) fn broadcast_map(from: &[usize], to: &[usize]) -> Vec<usize> {
    let lead = to.len() - from.len();
    let from_strides = strides(from);
    let to_strides = strides(to);
    let count: usize = to.iter().product();
    (0..count)
        .map(|flat| {
            let mut position = 0;
            for (i, (&size, &stride)) in from.iter().zip(&from_strides).enumerate() {
                let coordinate = flat / to_strides[lead + i] % to[lead + i];
                if size != 1 {
                    position += coordinate * stride;
                }
            }
            position
        })
        .collect()
}

/// Sums `grad`, of a broadcast shape, back to `shape`: the gradient of a
/// broadcast input.
pub(crate) fn sum_to(grad: &Array, shape: &[usize]) -> Array {
    let mut out = Array::zeros(shape);
    for (value, &at) in grad.data.iter().zip(&broadcast_map(shape, &grad.shape)) {
        out.data[at] += value;
    }
    out
}

/// The lines of `shape` along `dim`: for each, the position of its first
/// element and the step between its elements. There are `numel / shape[dim]`
/// lines of `shape[dim]` elements.
pub(crate) fn lines(shape: &[usize], dim: usize) -> impl Iterator<Item = (usize, usize)> {
    let len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    (0..outer).flat_map(move |o| (0..inner).map(move |i| (o * len * inner + i, inner)))
}

/// The integer values of an index tensor, each checked against `size`.
pub(crate) fn indices(op: &'static str, tensor: &Tensor, size: usize) -> Result<Vec<usize>> {
    tensor
        .to_f64_vec()?
        .into_iter()
        .map(|value| {
            let index = value as i64;
            usize::try_from(index)
                .ok()
                .filter(|&i| i < size)
                .ok_or(Error::IndexOutOfRange { op, index, size })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_maps_repeat_elements() {
        assert_eq!(
            broadcast_map(&[3, 1], &[2, 3, 2]),
            vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        );
        let grad = Array::new(&[2, 3], vec![1.0; 6]);
        assert_eq!(sum_to(&grad, &[3]).data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn lines_walk_a_dimension() {
        let found: Vec<_> = lines(&[2, 3, 4], 1).collect();
        assert_eq!(found.len(), 8);
        assert_eq!(found[0], (0, 4));
        assert_eq!(found[4], (12, 4));
    }
}
