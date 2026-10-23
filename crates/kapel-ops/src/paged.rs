//! The cache of generation, kept by pages: attention over it, and the rows
//! written into it in place.

use kapel_core::{DType, Error, Result, Tensor};

use crate::{Operation, Signature, arity, float, indices, rank, same_device};

/// Scaled dot-product attention over a cache kept by pages, for generation:
/// the queries of several sequences at once, each seeing the positions of its
/// own sequence.
///
/// Inputs: `q [R, Hq, D]`, the cache's `keys [L, Hkv, D]` and `values [L, Hkv,
/// Dv]`, a table of pages `pages [S, P]` and the queries' rows `rows [R, 2]`,
/// both indices. The cache is cut into pages of `page` consecutive lines:
/// position `j` of sequence `s` is at line `pages[s, j / page] · page + j %
/// page`. Query `r` belongs to sequence `rows[r, 0]` and sees its first `c =
/// rows[r, 1]` positions, the last of which is its own: position `j` is
/// visible if `j < c` and, with a `window`, `c − 1 − j < window`. `Hq` is a
/// multiple of `Hkv`: query head `h` reads key and value head `h / (Hq /
/// Hkv)`.
///
/// Then, as for [`Attention`](crate::Attention), `s[j] = scale · q·k[j]`
/// (capped by `softcap · tanh(· / softcap)` with a softcap), `a = softmax(s)`
/// over the visible positions, and `y = Σⱼ a[j] · v[j]`. A query that sees no
/// position (`c = 0`) gives zeros. `scale` defaults to `1 / √D`.
///
/// The keys and values may be of any float type, the cache's (`f16` or `f8`
/// with queries in `bf16`, say), the same for both; the output has the type
/// of `q`. The operation is for inference: no input receives a gradient.
///
/// The values of `pages` and `rows` are checked when the operation runs
/// ([`PagedAttention::check_indices`]): a sequence, a count or a line out of
/// range is an [`Error::IndexOutOfRange`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedAttention {
    /// The lines of a page.
    pub page: usize,
    /// Only the last `window` positions, the query's included, are visible.
    pub window: Option<usize>,
    /// The factor of the scores; `None` for `1 / √D`.
    pub scale: Option<f64>,
    /// The bound of the scores, approached through `tanh`; `None` for none.
    pub softcap: Option<f64>,
}

impl PagedAttention {
    /// The attention of a decoder over pages of `page` lines, with the
    /// default scale.
    pub fn new(page: usize) -> Self {
        Self {
            page,
            window: None,
            scale: None,
            softcap: None,
        }
    }

    /// The factor of the scores for a head width `d`.
    pub fn scale_for(&self, d: usize) -> f64 {
        self.scale.unwrap_or(1.0 / (d as f64).sqrt())
    }

    /// The first position query `count` sees, a query that sees the first
    /// `count` positions of its sequence: 0, or later with a window.
    pub fn first_visible(&self, count: usize) -> usize {
        match self.window {
            Some(window) => count.saturating_sub(window),
            None => 0,
        }
    }

    /// Checks the values of `pages` (`[S, P]`, row-major) and `rows` (`[R,
    /// 2]`) against a cache of `lines` lines: each query's sequence below
    /// `S`, its count at most `P · page`, and the pages it reads below
    /// `lines / page`. Backends call it with the values they read.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfRange`] for the first value out of range.
    pub fn check_indices(
        &self,
        pages: &[usize],
        sequences: usize,
        rows: &[usize],
        lines: usize,
    ) -> Result<()> {
        let op = Self::NAME;
        let per_sequence = pages.len() / sequences.max(1);
        let page_count = lines / self.page;
        let out = |index: usize, size: usize| Error::IndexOutOfRange {
            op,
            index: index as i64,
            size,
        };
        for row in rows.as_chunks::<2>().0 {
            let (s, c) = (row[0], row[1]);
            if s >= sequences {
                return Err(out(s, sequences));
            }
            if c > per_sequence * self.page {
                return Err(out(c, per_sequence * self.page + 1));
            }
            let first = self.first_visible(c) / self.page;
            for &p in &pages[s * per_sequence..][first..c.div_ceil(self.page)] {
                if p >= page_count {
                    return Err(out(p, page_count));
                }
            }
        }
        Ok(())
    }
}

impl Operation for PagedAttention {
    const NAME: &'static str = "paged_attention";

    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature> {
        let op = Self::NAME;
        arity(op, inputs, &[5], "5")?;
        same_device(op, inputs)?;
        let [q, keys, values, pages, rows] =
            [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]];
        for (tensor, what) in [(q, "q"), (keys, "keys"), (values, "values")] {
            float(op, tensor)?;
            rank(op, tensor, 3, what)?;
        }
        if values.dtype() != keys.dtype() {
            return Err(Error::DTypeMismatch {
                op,
                expected: keys.dtype(),
                got: values.dtype(),
            });
        }
        for (tensor, what) in [(pages, "pages"), (rows, "rows")] {
            indices(op, tensor)?;
            rank(op, tensor, 2, what)?;
        }
        let [r, hq, d] = [q.shape()[0], q.shape()[1], q.shape()[2]];
        let [l, hkv, dk] = [keys.shape()[0], keys.shape()[1], keys.shape()[2]];
        let [lv, hv, dv] = [values.shape()[0], values.shape()[1], values.shape()[2]];
        let invalid = |message: String| Error::InvalidArgument { op, message };
        if lv != l || hv != hkv || dk != d || rows.shape() != [r, 2] {
            return Err(invalid(format!(
                "expected q [R, Hq, D], keys [L, Hkv, D], values [L, Hkv, Dv], pages [S, P], \
                 rows [R, 2]; got {:?}, {:?}, {:?}, {:?}, {:?}",
                q.shape(),
                keys.shape(),
                values.shape(),
                pages.shape(),
                rows.shape()
            )));
        }
        if d == 0 {
            return Err(invalid(
                "the head width D must be positive: the default scale is 1 / √D".into(),
            ));
        }
        if hkv == 0 || hq % hkv != 0 {
            return Err(invalid(format!(
                "the {hq} query heads must be a multiple of the {hkv} key heads"
            )));
        }
        if self.page == 0 {
            return Err(invalid("a page holds at least one line".into()));
        }
        if self.window == Some(0) {
            return Err(invalid("a window of 0 positions hides every key".into()));
        }
        Ok(Signature {
            shape: vec![r, hq, dv],
            dtype: q.dtype(),
        })
    }

    fn differentiable(&self, _index: usize) -> bool {
        false
    }
}

/// Writing into tensors in place: the caches of generation, which receive a
/// few rows at each step and are read whole by the next.
///
/// A tensor is written only through its only handle ([`Tensor::storage_mut`]):
/// no clone nor view of it sees the change.
pub trait Write {
    /// A tensor of `shape` and `dtype` filled with zeros, on the backend's
    /// device, to be written in place.
    ///
    /// # Errors
    ///
    /// For a shape too large, or one the device cannot hold.
    fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor>;

    /// Writes `rows[i]` into row `index[i]` of `table`, converted to the
    /// table's type as [`Cast`](crate::Cast) converts (to the nearest, ties
    /// to even). The other rows keep their values. See [`check_write`] for
    /// what is accepted.
    ///
    /// # Errors
    ///
    /// Those of [`check_write`]; [`Error::IndexOutOfRange`] for a row out of
    /// range, and [`Error::InvalidArgument`] for a row written twice.
    fn write_rows(&self, table: &mut Tensor, index: &Tensor, rows: &Tensor) -> Result<()>;

    /// Copies the host tensor `values` into `target`, a tensor of the
    /// backend of the same shape and type, in place: the inputs of a
    /// [recorded](Record) pass, written before each replay.
    ///
    /// # Errors
    ///
    /// Those of [`check_write`] for `target`; [`Error::ShapeMismatch`] or
    /// [`Error::DTypeMismatch`] if `values` differs; [`Error::WrongDevice`]
    /// if it is not in host memory.
    fn copy_from_host(&self, target: &mut Tensor, values: &Tensor) -> Result<()>;
}

/// Checks a call of [`Write::copy_from_host`]: `target` row-major from the
/// start of its storage and the only handle on it, `values` in host memory,
/// of the same shape and type. The bytes of `values`, row-major.
///
/// # Errors
///
/// As [`Write::copy_from_host`].
pub fn check_copy(target: &mut Tensor, values: &Tensor) -> Result<Vec<u8>> {
    let op = COPY_FROM_HOST;
    if values.shape() != target.shape() {
        return Err(Error::ShapeMismatch {
            op,
            expected: target.shape().to_vec(),
            got: values.shape().to_vec(),
        });
    }
    if values.dtype() != target.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: target.dtype(),
            got: values.dtype(),
        });
    }
    if !target.is_contiguous() || target.layout().offset() != 0 {
        return Err(Error::InvalidArgument {
            op,
            message: "the target must be row-major from the start of its storage".into(),
        });
    }
    if target.storage_mut().is_none() {
        return Err(Error::InvalidArgument {
            op,
            message: "the target is shared: writing in place needs its only handle".into(),
        });
    }
    let values = values.contiguous()?;
    let bytes = values.storage().host_bytes().ok_or(Error::WrongDevice {
        op,
        expected: kapel_core::Device::Cpu,
        got: values.device(),
    })?;
    let size = values.dtype().storage_bytes(values.numel())?;
    let start = values.dtype().storage_bytes(values.layout().offset())?;
    Ok(bytes[start..start + size].to_vec())
}

/// The name of [`Write::copy_from_host`] in errors.
pub const COPY_FROM_HOST: &str = "copy_from_host";

/// A pass recorded once and replayed: on a card, a CUDA graph, which launches
/// all the kernels of a pass in one call of the driver instead of one call
/// each.
///
/// The recording runs nothing: [`Recording::replay`] runs the pass, reading
/// the tensors it read when it was recorded, whose contents may change in
/// between ([`Write::copy_from_host`], [`Write::write_rows`]), and writing
/// its output into [`Recording::output`]. A pass can be recorded only if
/// nothing in it depends on the values of its tensors (the backend reads
/// none back), and if it allocates nothing it cannot keep. While a pass
/// records, the backend must not be used from another thread: on a card, its
/// work would be recorded too.
pub trait Record {
    /// Whether the backend records passes. One that computes each operation
    /// when it is called, such as one on the host, does not: a pass is
    /// simply run there.
    fn records(&self) -> bool;

    /// Records the operations `pass` queues, and the tensor it returns,
    /// without running them; `None`, and nothing called, for a backend that
    /// does not record.
    ///
    /// # Errors
    ///
    /// Those of `pass`, and those of the backend; an operation that cannot
    /// be recorded makes the recording fail.
    fn record(&self, pass: &mut dyn FnMut() -> Result<Tensor>) -> Result<Option<Recording>>;
}

/// A recorded pass ([`Record`]).
pub struct Recording {
    output: Tensor,
    replay: Box<dyn Fn() -> Result<()> + Send + Sync>,
}

impl Recording {
    /// A recording of `output`, run by `replay`: what a backend makes.
    pub fn new(output: Tensor, replay: Box<dyn Fn() -> Result<()> + Send + Sync>) -> Self {
        Self { output, replay }
    }

    /// Runs the recorded pass again. Its output is [`Recording::output`],
    /// rewritten.
    ///
    /// # Errors
    ///
    /// Those of the backend.
    pub fn replay(&self) -> Result<()> {
        (self.replay)()
    }

    /// The tensor the pass writes.
    pub fn output(&self) -> &Tensor {
        &self.output
    }
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording")
            .field("output", &self.output.shape())
            .finish()
    }
}

/// The name of [`Write::write_rows`] in errors.
pub const WRITE_ROWS: &str = "write_rows";

/// Checks a call of [`Write::write_rows`]: `table [L, …]` of a float type,
/// row-major from the start of its storage, and the only handle on it;
/// `index [R]` of indices; `rows [R, …]` of a float type, with the rows of the
/// table; all on one device.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a table that is shared or not row-major,
/// [`Error::UnsupportedDType`], [`Error::ShapeMismatch`],
/// [`Error::DeviceMismatch`].
pub fn check_write(table: &mut Tensor, index: &Tensor, rows: &Tensor) -> Result<()> {
    let op = WRITE_ROWS;
    same_device(op, &[table, index, rows])?;
    float(op, table)?;
    float(op, rows)?;
    indices(op, index)?;
    rank(op, index, 1, "index")?;
    if table.rank() == 0 {
        return Err(Error::InvalidArgument {
            op,
            message: "a table has rows: rank at least 1".into(),
        });
    }
    let mut expected = vec![index.shape()[0]];
    expected.extend_from_slice(&table.shape()[1..]);
    if rows.shape() != expected.as_slice() {
        return Err(Error::ShapeMismatch {
            op,
            expected,
            got: rows.shape().to_vec(),
        });
    }
    if !table.is_contiguous() || table.layout().offset() != 0 {
        return Err(Error::InvalidArgument {
            op,
            message: "the table must be row-major from the start of its storage".into(),
        });
    }
    if table.storage_mut().is_none() {
        return Err(Error::InvalidArgument {
            op,
            message: "the table is shared: writing in place needs its only handle".into(),
        });
    }
    Ok(())
}

/// Checks the rows a [`Write::write_rows`] writes, as a backend reads them:
/// each below `lines`, none twice.
///
/// # Errors
///
/// [`Error::IndexOutOfRange`] or [`Error::InvalidArgument`].
pub fn check_written(rows: &[usize], lines: usize) -> Result<()> {
    let op = WRITE_ROWS;
    let mut seen = vec![false; lines];
    for &r in rows {
        if r >= lines {
            return Err(Error::IndexOutOfRange {
                op,
                index: r as i64,
                size: lines,
            });
        }
        if std::mem::replace(&mut seen[r], true) {
            return Err(Error::InvalidArgument {
                op,
                message: format!("row {r} written twice"),
            });
        }
    }
    Ok(())
}
