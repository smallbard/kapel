//! Attention over a cache kept by pages, and rows written into it in place.
//!
//! The cache may be large and mostly unused by a step: only the lines of the
//! sequences of the step are read, converted to `f32` once per task. A task is
//! one sequence, one key head and a block of that sequence's queries; each
//! output row is computed by one task, adding its terms in a fixed order.

use std::sync::OnceLock;

use kapel_core::{DType, Error, HostStorage, Result, Tensor};
use kapel_ops::{
    Operation, PagedAttention, Record, Recording, Run, WRITE_ROWS, Write, check_copy, check_write,
    check_written,
};

use crate::backend::Cpu;
use crate::io;
use crate::math;
use crate::pool::SharedMut;

/// Queries of one sequence per task.
const QUERIES: usize = 32;

/// The value of each byte of an 8-bit float type.
fn byte_values(dtype: DType) -> &'static [f32; 256] {
    static E4M3: OnceLock<[f32; 256]> = OnceLock::new();
    static E5M2: OnceLock<[f32; 256]> = OnceLock::new();
    let cell = if dtype == DType::F8E4M3 { &E4M3 } else { &E5M2 };
    cell.get_or_init(|| {
        std::array::from_fn(|b| dtype.decode(&[b as u8]).map_or(f32::NAN, |v| v as f32))
    })
}

/// The cache's types: the floats the backend reads, and the 8-bit floats.
fn cache_type(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F8E4M3 | DType::F8E5M2 => Ok(()),
        dtype => io::supported(op, dtype),
    }
}

/// `out.len()` values of a table of `dtype`, from element `at`, as `f32`.
fn read_into(dtype: DType, bytes: &[u8], at: usize, out: &mut [f32]) {
    let n = out.len();
    match dtype {
        DType::F32 => {
            for (o, b) in out
                .iter_mut()
                .zip(bytes[4 * at..4 * (at + n)].as_chunks::<4>().0)
            {
                *o = f32::from_le_bytes(*b);
            }
        }
        DType::BF16 => {
            for (o, b) in out
                .iter_mut()
                .zip(bytes[2 * at..2 * (at + n)].as_chunks::<2>().0)
            {
                *o = f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16);
            }
        }
        DType::F16 => {
            for (o, b) in out
                .iter_mut()
                .zip(bytes[2 * at..2 * (at + n)].as_chunks::<2>().0)
            {
                *o = kapel_core::F16(u16::from_le_bytes(*b)).to_f64() as f32;
            }
        }
        _ => {
            let table = byte_values(dtype);
            for (o, &b) in out.iter_mut().zip(&bytes[at..at + n]) {
                *o = table[b as usize];
            }
        }
    }
}

impl Run<PagedAttention> for Cpu {
    fn forward(&self, op: &PagedAttention, inputs: &[&Tensor]) -> Result<Tensor> {
        let name = PagedAttention::NAME;
        let signature = op.infer(inputs)?;
        let (keys, values) = (inputs[1], inputs[2]);
        cache_type(name, keys.dtype())?;
        let [lines, hkv, d] = [keys.shape()[0], keys.shape()[1], keys.shape()[2]];
        let (hq, dv) = (inputs[0].shape()[1], values.shape()[2]);
        let (sequences, per_sequence) = (inputs[3].shape()[0], inputs[3].shape()[1]);
        let pages = io::indices(name, inputs[3], usize::MAX)?;
        let rows = io::indices(name, inputs[4], usize::MAX)?;
        op.check_indices(&pages, sequences, &rows, lines)?;
        let q = io::contiguous(name, inputs[0])?;
        let (kb, vb) = (io::host_bytes(name, keys)?, io::host_bytes(name, values)?);
        let (k_start, v_start) = (keys.layout().offset(), values.layout().offset());
        if !keys.is_contiguous() || !values.is_contiguous() {
            return Err(Error::NotContiguous { op: name });
        }
        let dtype = keys.dtype();
        // The queries of each sequence, in order, cut into blocks.
        let mut by_sequence: Vec<Vec<usize>> = vec![Vec::new(); sequences];
        for (r, row) in rows.as_chunks::<2>().0.iter().enumerate() {
            by_sequence[row[0]].push(r);
        }
        let blocks: Vec<&[usize]> = by_sequence
            .iter()
            .flat_map(|queries| queries.chunks(QUERIES))
            .collect();
        let group = hq / hkv;
        let scale = op.scale_for(d) as f32;
        let mut out = vec![0.0f32; signature.shape.iter().product()];
        let shared = SharedMut::new(&mut out);
        let isa = self.isa;
        let line =
            |s: usize, j: usize| pages[s * per_sequence + j / op.page] * op.page + j % op.page;
        self.pool.run(blocks.len() * hkv, |task| {
            let (block, kv) = (blocks[task / hkv], task % hkv);
            let s = rows[2 * block[0]];
            let counts = block.iter().map(|&r| rows[2 * r + 1]);
            let end = counts.clone().max().unwrap_or(0);
            let start = counts.map(|c| op.first_visible(c)).min().unwrap_or(0);
            // The keys and values of this head, positions `start..end`.
            let n = end - start;
            let (mut k, mut v) = (vec![0.0f32; n * d], vec![0.0f32; n * dv]);
            for j in start..end {
                let at = line(s, j) * hkv + kv;
                let i = j - start;
                read_into(dtype, kb, k_start + at * d, &mut k[i * d..][..d]);
                read_into(dtype, vb, v_start + at * dv, &mut v[i * dv..][..dv]);
            }
            let mut scores = vec![0.0f32; n];
            for &r in block {
                let count = rows[2 * r + 1];
                let first = op.first_visible(count);
                if count == 0 {
                    continue;
                }
                let seen = first - start..count - start;
                for h in kv * group..(kv + 1) * group {
                    let query = &q[(r * hq + h) * d..][..d];
                    let row = &mut scores[seen.clone()];
                    for (score, i) in row.iter_mut().zip(seen.clone()) {
                        let raw = scale * math::dot(isa, query, &k[i * d..][..d]);
                        *score = match op.softcap {
                            Some(cap) => cap as f32 * (raw / cap as f32).tanh(),
                            None => raw,
                        };
                    }
                    let max = math::max(isa, row);
                    let total = math::exp_shifted_sum(isa, row, max);
                    // SAFETY: row `r`, head `h` belongs to this task alone (one
                    // task per sequence block and key head), and `out`
                    // outlives the call to `run`.
                    let y = unsafe { shared.slice((r * hq + h) * dv, dv) };
                    for (w, i) in row.iter().zip(seen.clone()) {
                        math::axpy(isa, w / total, &v[i * dv..][..dv], y);
                    }
                }
            }
        });
        io::write(&self.buffers, &out, &signature.shape, signature.dtype)
    }

    fn backward(
        &self,
        op: &PagedAttention,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        op.infer_backward(inputs, grad)?;
        Ok(vec![None; inputs.len()])
    }
}

impl Record for Cpu {
    /// Computes as it goes: nothing to record.
    fn records(&self) -> bool {
        false
    }

    fn record(&self, _pass: &mut dyn FnMut() -> Result<Tensor>) -> Result<Option<Recording>> {
        Ok(None)
    }
}

impl Write for Cpu {
    fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        Tensor::zeros(dtype, shape)
    }

    fn copy_from_host(&self, target: &mut Tensor, values: &Tensor) -> Result<()> {
        let bytes = check_copy(target, values)?;
        let storage = target
            .storage_mut()
            .and_then(|s| s.as_any_mut())
            .and_then(|any| any.downcast_mut::<HostStorage>())
            .ok_or_else(|| Error::InvalidArgument {
                op: kapel_ops::COPY_FROM_HOST,
                message: "the target is not in host memory that can be written".into(),
            })?;
        storage.bytes_mut()[..bytes.len()].copy_from_slice(&bytes);
        Ok(())
    }

    fn write_rows(&self, table: &mut Tensor, index: &Tensor, rows: &Tensor) -> Result<()> {
        check_write(table, index, rows)?;
        cache_type(WRITE_ROWS, table.dtype())?;
        let lines = table.shape()[0];
        let written = io::indices(WRITE_ROWS, index, lines)?;
        check_written(&written, lines)?;
        let dtype = table.dtype();
        let width: usize = table.shape()[1..].iter().product();
        if rows.dtype() == dtype {
            // The same type: the bytes, row by row (a table grown, say).
            let rows = rows.contiguous()?;
            let size = dtype.block_bytes();
            let from = io::host_bytes(WRITE_ROWS, &rows)?;
            let from = &from[rows.layout().offset() * size..][..written.len() * width * size];
            let storage = table
                .storage_mut()
                .and_then(|s| s.as_any_mut())
                .and_then(|any| any.downcast_mut::<HostStorage>())
                .ok_or_else(|| Error::InvalidArgument {
                    op: WRITE_ROWS,
                    message: "the table is not in host memory that can be written".into(),
                })?;
            let bytes = storage.bytes_mut();
            for (&r, row) in written.iter().zip(from.chunks_exact((width * size).max(1))) {
                bytes[r * width * size..][..width * size].copy_from_slice(row);
            }
            return Ok(());
        }
        let values = io::contiguous(WRITE_ROWS, rows)?;
        let storage = table
            .storage_mut()
            .and_then(|s| s.as_any_mut())
            .and_then(|any| any.downcast_mut::<HostStorage>())
            .ok_or_else(|| Error::InvalidArgument {
                op: WRITE_ROWS,
                message: "the table is not in host memory that can be written".into(),
            })?;
        let bytes = storage.bytes_mut();
        for (&r, row) in written.iter().zip(values.chunks_exact(width.max(1))) {
            match dtype {
                DType::F32 => {
                    let out = &mut io::as_f32_mut(bytes)[r * width..][..width];
                    out.copy_from_slice(row);
                }
                DType::BF16 => {
                    let out = &mut bytes[2 * r * width..][..2 * width];
                    for (o, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(row) {
                        *o = io::to_bf16(x).to_le_bytes();
                    }
                }
                _ => {
                    let size = dtype.block_bytes();
                    let out = &mut bytes[size * r * width..][..size * width];
                    for (o, &x) in out.chunks_exact_mut(size).zip(row) {
                        dtype.encode(f64::from(x), o)?;
                    }
                }
            }
        }
        Ok(())
    }
}
