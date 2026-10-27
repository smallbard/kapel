//! `training.offload`: the trained parameters in `f32`, the optimizer's
//! state and, over several passes, the gradients' sum in the processor's
//! memory; the card holds the parameters in the computation's type only
//! (ZeRO-Offload).
//!
//! With one pass a step, the gradients stay on the card, where the norm of
//! the clipping is computed; then each parameter in turn has its gradient
//! brought down, its step made by the processor, and its new value written
//! back into the card's copy. Only one gradient is in the processor's memory
//! at a time: 12 bytes a parameter there with AdamW (6 in 8 bits). With
//! several passes, each pass's gradients are added in place to sums in `f32`
//! there, 4 bytes a parameter more.

use kapel_core::{DType, Device, HostStorage, Tensor};
use kapel_ops::{Cast, Run as _};

use super::{Result, TrainError};
use crate::model::Parameters;
use crate::optim::Optimizer;
use crate::tape::{Compute, Grad};

/// The trained parameters on the processor.
pub(super) struct Offload {
    pub(super) cpu: kapel_cpu::Cpu,
    pub(super) params: Parameters,
    /// The type of the card's copies.
    dtype: DType,
}

impl Offload {
    /// The trained parameters of `params` in `f32` on the processor, the
    /// card's copies being in `dtype` ([`Parameters::to_mixed`]).
    pub(super) fn new<B: Compute>(backend: &B, params: &Parameters, dtype: DType) -> Result<Self> {
        let mut master = Parameters::default();
        for (name, p) in params.iter() {
            if !p.trainable {
                continue;
            }
            let mut m = p.clone();
            m.value = widened(&backend.download(&p.value)?)?;
            master.insert(name, m);
        }
        Ok(Self {
            cpu: kapel_cpu::Cpu::new(),
            params: master,
            dtype,
        })
    }

    /// Adds a pass's gradients, brought down, to the step's sums on the
    /// processor.
    pub(super) fn accumulate<B: Compute>(
        &self,
        backend: &B,
        sum: &mut std::collections::BTreeMap<String, Grad>,
        grads: Vec<(String, Grad)>,
    ) -> Result<()> {
        for (name, g) in grads {
            let shape = self.shape(&name)?;
            let host = backend.download(&row_major(backend, g.dense(backend, &shape)?)?)?;
            match sum.get_mut(&name) {
                Some(Grad::Dense(acc)) => add_into(acc, &host)?,
                _ => {
                    sum.insert(name, Grad::Dense(widened(&host)?));
                }
            }
        }
        Ok(())
    }

    /// One step of `optimizer` (AdamW), parameter after parameter: each
    /// gradient brought down if it is on the card, the step made on the
    /// processor, the new value written into the card's copy.
    pub(super) fn step<B: Compute>(
        &mut self,
        optimizer: &mut Optimizer,
        backend: &B,
        params: &mut Parameters,
        grads: Vec<(String, Grad)>,
        scale: f64,
    ) -> Result<()> {
        let Optimizer::AdamW(adamw) = optimizer else {
            return Err(TrainError(
                "training.offload: with adamw or adamw8bit only".into(),
            ));
        };
        adamw.advance();
        for (name, grad) in grads {
            let grad = match grad {
                Grad::Dense(t) if t.device() == Device::Cpu => Grad::Dense(t),
                g => {
                    let shape = self.shape(&name)?;
                    // Read in its type by the processor's step.
                    Grad::Dense(backend.download(&row_major(backend, g.dense(backend, &shape)?)?)?)
                }
            };
            adamw.update(
                &self.cpu,
                &mut self.params,
                vec![(name.clone(), grad)],
                scale,
            )?;
            let master = &self.params.get(&name).expect("a trained parameter").value;
            let copy = narrowed(master, self.dtype, &self.cpu)?;
            if let Some(p) = params.get_mut(&name) {
                backend.copy_from_host(&mut p.value, &copy)?;
            }
        }
        Ok(())
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.params
            .get(name)
            .map(|p| p.value.shape().to_vec())
            .ok_or_else(|| TrainError(format!("a gradient of {name}, which is not trained")))
    }
}

/// The `f32` values of a host tensor of its own, to be written.
fn f32s_mut(t: &mut Tensor) -> Result<&mut [f32]> {
    let n = t.numel();
    let storage = t
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| TrainError("a sum of gradients that is shared".into()))?;
    // SAFETY: host storage is 64-byte aligned, and any bits are a valid
    // `f32`.
    let (head, values, _) = unsafe { storage.bytes_mut().align_to_mut::<f32>() };
    if !head.is_empty() {
        return Err(TrainError("a host tensor not aligned".into()));
    }
    Ok(&mut values[..n])
}

/// A host tensor of its own in `f32` with the values of `t` (in host memory,
/// in `f32`, bf16 or f16).
fn widened(t: &Tensor) -> Result<Tensor> {
    let mut out = Tensor::zeros(DType::F32, t.shape())?;
    add_into(&mut out, t)?;
    Ok(out)
}

/// `t` row-major, copied on its device if it is not (the gradient of a
/// weight is a transposed view): the processor then reads it in order.
fn row_major<B: Compute>(backend: &B, t: Tensor) -> Result<Tensor> {
    Ok(if t.is_contiguous() {
        t
    } else {
        backend.forward(&kapel_ops::Contiguous, &[&t])?
    })
}

/// The values of `t`, a host tensor in `f32`, in `dtype`, converted on
/// every core: rounded to the nearest, ties to even, as the backends cast
/// (`f32` copied, f16 by the processor's backend).
fn narrowed(t: &Tensor, dtype: DType, cpu: &kapel_cpu::Cpu) -> Result<Tensor> {
    if dtype != DType::BF16 {
        return Ok(cpu.forward(&Cast { dtype }, &[t])?);
    }
    let t = t.contiguous()?;
    let bytes = t
        .storage()
        .host_bytes()
        .ok_or_else(|| TrainError("a parameter not in host memory".into()))?;
    let start = t.layout().offset() * 4;
    let bytes = &bytes[start..start + t.numel() * 4];
    let mut out = Tensor::zeros(DType::BF16, t.shape())?;
    let n = t.numel();
    let storage = out
        .storage_mut()
        .and_then(|s| s.as_any_mut())
        .and_then(|any| any.downcast_mut::<HostStorage>())
        .ok_or_else(|| TrainError("a new tensor not in host memory".into()))?;
    let target = &mut storage.bytes_mut()[..n * 2];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let piece = n.div_ceil(threads).max(1 << 16);
    std::thread::scope(|scope| {
        for (k, out) in target.chunks_mut(piece * 2).enumerate() {
            let from = &bytes[k * piece * 4..];
            scope.spawn(move || {
                for (i, o) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let x = f32::from_le_bytes(from[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
                    o.copy_from_slice(&bf16_bits(x).to_le_bytes());
                }
            });
        }
    });
    Ok(out)
}

/// `x` rounded to bf16, to the nearest, ties to even; a NaN stays a NaN.
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// `acc += g`, in place, on every core: `acc` a host tensor of its own in
/// `f32`, `g` of its shape in host memory, in `f32`, bf16 or f16.
fn add_into(acc: &mut Tensor, g: &Tensor) -> Result<()> {
    if acc.shape() != g.shape() {
        return Err(TrainError(format!(
            "gradients of shapes {:?} and {:?}",
            acc.shape(),
            g.shape()
        )));
    }
    let g = g.contiguous()?;
    let bytes = g
        .storage()
        .host_bytes()
        .ok_or_else(|| TrainError("a gradient not in host memory".into()))?;
    let size = g.dtype().block_bytes();
    let start = g.layout().offset() * size;
    let bytes = &bytes[start..start + g.numel() * size];
    let acc = f32s_mut(acc)?;
    match g.dtype() {
        DType::F32 => add_with(acc, bytes, 4, |b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }),
        DType::BF16 => add_with(acc, bytes, 2, |b| {
            f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
        }),
        DType::F16 => add_with(acc, bytes, 2, |b| {
            DType::F16.decode(b).unwrap_or(f64::NAN) as f32
        }),
        dtype => return Err(TrainError(format!("a gradient in {dtype}"))),
    }
    Ok(())
}

/// `acc[i] += read(element i of bytes)`, on every core.
fn add_with(acc: &mut [f32], bytes: &[u8], size: usize, read: impl Fn(&[u8]) -> f32 + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let piece = acc.len().div_ceil(threads).max(1 << 16);
    let read = &read;
    std::thread::scope(|scope| {
        for (out, from) in acc.chunks_mut(piece).zip(bytes.chunks(piece * size)) {
            scope.spawn(move || {
                for (a, b) in out.iter_mut().zip(from.chunks_exact(size)) {
                    *a += read(b);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conversions on every core give the bits of the processor's
    /// backend.
    #[test]
    fn conversions_give_the_bits_of_the_backend() {
        let cpu = kapel_cpu::Cpu::new();
        let mut rng = crate::model::Rng::new(3);
        let mut values: Vec<f32> = (0..300_000)
            .map(|_| ((rng.uniform() - 0.5) * 2f64.powi((rng.next_u64() % 60) as i32 - 30)) as f32)
            .collect();
        values.extend([
            0.0,
            -0.0,
            1.0 + f32::EPSILON,
            f32::MAX,
            f32::INFINITY,
            -f32::INFINITY,
        ]);
        // Ties: halfway between two bf16 values, below and above even.
        values.extend([f32::from_bits(0x3F80_8000), f32::from_bits(0x3F81_8000)]);
        let t = Tensor::from_slice(&values, &[values.len()]).unwrap();
        let ours = narrowed(&t, DType::BF16, &cpu).unwrap();
        let theirs = cpu.forward(&Cast { dtype: DType::BF16 }, &[&t]).unwrap();
        assert_eq!(ours.storage().host_bytes(), theirs.storage().host_bytes());
        let back = widened(&ours).unwrap();
        let theirs = cpu.forward(&Cast { dtype: DType::F32 }, &[&ours]).unwrap();
        assert_eq!(
            back.to_vec::<f32>().unwrap(),
            theirs.to_vec::<f32>().unwrap()
        );
    }
}
