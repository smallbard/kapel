//! The forward pass of a decoder, on any backend.

use std::collections::BTreeMap;

use kapel_core::{DType, Tensor};
use kapel_hub::Checkpoint;
use kapel_ops::{
    Affine, Attention, Backend, Binary, BinaryKind, Cast, Concat, Contiguous, GatherRows,
    LayerNorm, Matmul, RmsNorm, Rope, Transfer, Unary, UnaryKind,
};

use crate::ModelError;
use crate::config::{Block, Config, Norm};

/// A model: its configuration and its weights, by their names in the checkpoint.
pub struct Model {
    config: Config,
    /// `config.json` as read, written back unchanged by [`Model::save`].
    raw_config: serde_json::Value,
    weights: BTreeMap<String, Tensor>,
}

type Result<T> = std::result::Result<T, ModelError>;

/// The keys and values of the positions a model has already seen, layer by
/// layer, on the backend of the steps that filled them: what lets
/// [`Model::step`] compute only the new positions.
#[derive(Clone, Debug, Default)]
pub struct Cache {
    /// Per layer, the keys and values `[1, kv_heads, len, head_dim]`.
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl Cache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of positions it holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no position.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Which positions get logits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Logits {
    All,
    Last,
}

/// The operations of the forward pass, on one backend.
struct Ops<'a, B>(&'a B);

impl<B: Backend> Ops<'_, B> {
    fn linear(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        // `transformers` stores `[out, in]`: `y = x · wᵀ`, a view.
        let y = self.0.forward(&Matmul, &[x, &w.transpose(0, 1)?])?;
        self.bias(y, bias)
    }

    fn bias(&self, y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        Ok(match bias {
            Some(b) => self.0.forward(
                &Binary {
                    kind: BinaryKind::Add,
                },
                &[&y, b],
            )?,
            None => y,
        })
    }

    fn binary(&self, kind: BinaryKind, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Binary { kind }, &[a, b])?)
    }

    fn unary(&self, kind: UnaryKind, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Unary { kind }, &[x])?)
    }

    fn affine(&self, x: &Tensor, scale: f64) -> Result<Tensor> {
        Ok(self.0.forward(&Affine { scale, bias: 0.0 }, &[x])?)
    }

    /// `x` with its elements in row-major order, on the backend.
    fn contiguous(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(&Contiguous, &[x])?)
    }

    /// `cap · tanh(x / cap)`.
    fn softcap(&self, x: &Tensor, cap: f64) -> Result<Tensor> {
        let t = self.unary(UnaryKind::Tanh, &self.affine(x, 1.0 / cap)?)?;
        self.affine(&t, cap)
    }
}

impl Model {
    /// Reads the configuration and maps the weights of a model directory.
    ///
    /// # Errors
    ///
    /// For an unsupported architecture, or missing or unreadable weights.
    pub fn load(checkpoint: &Checkpoint) -> Result<Self> {
        let raw_config = checkpoint.config()?;
        let config = Config::from_json(&raw_config)?;
        let weights = checkpoint
            .names()
            .map(|name| Ok((name.to_string(), checkpoint.tensor(name)?)))
            .collect::<std::result::Result<_, kapel_hub::HubError>>()?;
        Ok(Self {
            config,
            raw_config,
            weights,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The model with its weights on `backend`'s device, ready for
    /// [`Model::forward`] on it.
    ///
    /// # Errors
    ///
    /// If a weight cannot be copied to the device.
    pub fn upload<B: Transfer>(&self, backend: &B) -> Result<Self> {
        self.map(|w| Ok(backend.upload(w)?))
    }

    /// The model with its float weights converted to `dtype` on `backend`, to
    /// compute in that type (`bf16` on a card, for instance). Quantized and
    /// integer weights are kept.
    ///
    /// # Errors
    ///
    /// If the backend cannot convert a weight.
    pub fn cast<B: Backend>(&self, backend: &B, dtype: DType) -> Result<Self> {
        self.map(|w| {
            Ok(if w.dtype().is_float() && w.dtype() != dtype {
                backend.forward(&Cast { dtype }, &[w])?
            } else {
                w.clone()
            })
        })
    }

    /// The model with `f` applied to each weight.
    fn map(&self, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Self> {
        Ok(Self {
            config: self.config.clone(),
            raw_config: self.raw_config.clone(),
            weights: self
                .weights
                .iter()
                .map(|(name, w)| Ok((name.clone(), f(w)?)))
                .collect::<Result<_>>()?,
        })
    }

    /// Writes the model to `dir` in the layout of the Hub: `config.json` and
    /// `model.safetensors`, which `transformers` loads back. The weights must
    /// be in host memory: a model as loaded, not [uploaded](Model::upload).
    ///
    /// `dir` may be the directory the model was loaded from: each file is
    /// written under a temporary name and renamed into place
    /// ([`kapel_hub::write_atomic`]), so the weights mapped from the old file
    /// stay valid. A `model.safetensors.index.json` in `dir` would take
    /// precedence over the new `model.safetensors` when loading: it is removed
    /// once the weights are written (the shards it listed are left).
    ///
    /// # Errors
    ///
    /// If the directory or the files cannot be written.
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<()> {
        let dir = dir.as_ref();
        let io = |path: std::path::PathBuf, source: std::io::Error| {
            ModelError::Hub(kapel_hub::HubError::Io { path, source })
        };
        std::fs::create_dir_all(dir).map_err(|e| io(dir.to_path_buf(), e))?;
        let json = serde_json::to_vec_pretty(&self.raw_config).expect("a JSON value serializes");
        kapel_hub::write_atomic(dir.join("config.json"), |w| w.write_all(&json))?;
        let tensors: Vec<(&str, &Tensor)> =
            self.weights.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        kapel_hub::safetensors::save(dir.join("model.safetensors"), &tensors, &metadata)?;
        let index = dir.join("model.safetensors.index.json");
        match std::fs::remove_file(&index) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io(index, e)),
            _ => Ok(()),
        }
    }

    fn weight(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }

    fn optional(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    fn norm<B: Backend>(&self, ops: &Ops<'_, B>, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.weight(&format!("{prefix}.weight"))?;
        Ok(match self.config.norm {
            Norm::Rms { offset } => ops.0.forward(
                &RmsNorm {
                    eps: self.config.eps,
                    weight_offset: offset,
                },
                &[x, w],
            )?,
            Norm::Layer => {
                let b = self.weight(&format!("{prefix}.bias"))?;
                ops.0.forward(
                    &LayerNorm {
                        eps: self.config.eps,
                    },
                    &[x, w, b],
                )?
            }
        })
    }

    /// The logits `[T, vocab]` of the token ids `ids`, one sequence, without a
    /// cache: every position sees the ones before it.
    ///
    /// # Errors
    ///
    /// If a weight is missing, or an operation fails on the backend.
    pub fn forward<B: Backend + Transfer>(&self, backend: &B, ids: &[u32]) -> Result<Tensor> {
        self.run(backend, ids, &mut Cache::new(), Logits::All)
    }

    /// The logits `[1, vocab]` of the last of `ids`, which follow the positions
    /// of `cache`; their keys and values are added to it. Each new position
    /// sees the cached ones and the new ones before it: a prompt is given at
    /// once (the prefill), then the generated tokens one at a time.
    ///
    /// # Errors
    ///
    /// If `ids` is empty, a weight is missing, or an operation fails on the
    /// backend. On an error, the cache may hold some layers of the failed step:
    /// start again from a new one.
    pub fn step<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(ModelError::Tensor(kapel_core::Error::InvalidArgument {
                op: "step",
                message: "no token to compute".into(),
            }));
        }
        self.run(backend, ids, cache, Logits::Last)
    }

    fn run<B: Backend + Transfer>(
        &self,
        backend: &B,
        ids: &[u32],
        cache: &mut Cache,
        logits: Logits,
    ) -> Result<Tensor> {
        let c = &self.config;
        let ops = Ops(backend);
        let t = ids.len();
        let past = cache.len;
        let ids_tensor = backend.upload(&Tensor::from_slice(ids, &[t])?)?;
        let first = u32::try_from(past).map_err(|_| kapel_core::Error::Overflow { op: "step" })?;
        let positions: Vec<u32> = (0..t as u32).map(|i| first + i).collect();
        let positions = backend.upload(&Tensor::from_slice(&positions, &[t])?)?;
        cache.layers.resize(c.layers, None);

        let (embed_name, prefix) = match c.block {
            Block::Gpt2 => ("transformer.wte.weight", "transformer.h"),
            _ => ("model.embed_tokens.weight", "model.layers"),
        };
        let embed = self.weight(embed_name)?;
        let mut x = backend.forward(&GatherRows, &[embed, &ids_tensor])?;
        if let Some(scale) = c.embed_scale {
            // Gemma rounds the factor to the type of the hidden states.
            x = ops.affine(&x, rounded(x.dtype(), scale)?)?;
        }
        if c.block == Block::Gpt2 {
            let wpe = backend.forward(
                &GatherRows,
                &[self.weight("transformer.wpe.weight")?, &positions],
            )?;
            x = ops.binary(BinaryKind::Add, &x, &wpe)?;
        }

        for layer in 0..c.layers {
            let p = format!("{prefix}.{layer}");
            x = match c.block {
                Block::Sequential | Block::Sandwich => {
                    let h = self.norm(&ops, &x, &format!("{p}.input_layernorm"))?;
                    let mut a =
                        self.attention(&ops, &h, &positions, &mut cache.layers[layer], layer, &p)?;
                    if c.block == Block::Sandwich {
                        a = self.norm(&ops, &a, &format!("{p}.post_attention_layernorm"))?;
                    }
                    let x = ops.binary(BinaryKind::Add, &x, &a)?;
                    let before = if c.block == Block::Sandwich {
                        "pre_feedforward_layernorm"
                    } else {
                        "post_attention_layernorm"
                    };
                    let h = self.norm(&ops, &x, &format!("{p}.{before}"))?;
                    let gate =
                        ops.linear(&h, self.weight(&format!("{p}.mlp.gate_proj.weight"))?, None)?;
                    let up =
                        ops.linear(&h, self.weight(&format!("{p}.mlp.up_proj.weight"))?, None)?;
                    let inner =
                        ops.binary(BinaryKind::Mul, &ops.unary(c.activation, &gate)?, &up)?;
                    let mut m = ops.linear(
                        &inner,
                        self.weight(&format!("{p}.mlp.down_proj.weight"))?,
                        None,
                    )?;
                    if c.block == Block::Sandwich {
                        m = self.norm(&ops, &m, &format!("{p}.post_feedforward_layernorm"))?;
                    }
                    ops.binary(BinaryKind::Add, &x, &m)?
                }
                Block::Parallel => {
                    let h = self.norm(&ops, &x, &format!("{p}.input_layernorm"))?;
                    let a =
                        self.attention(&ops, &h, &positions, &mut cache.layers[layer], layer, &p)?;
                    let fc1 = ops.linear(
                        &h,
                        self.weight(&format!("{p}.mlp.fc1.weight"))?,
                        self.optional(&format!("{p}.mlp.fc1.bias")),
                    )?;
                    let fc2 = ops.linear(
                        &ops.unary(c.activation, &fc1)?,
                        self.weight(&format!("{p}.mlp.fc2.weight"))?,
                        self.optional(&format!("{p}.mlp.fc2.bias")),
                    )?;
                    ops.binary(BinaryKind::Add, &ops.binary(BinaryKind::Add, &x, &a)?, &fc2)?
                }
                Block::Gpt2 => {
                    let h = self.norm(&ops, &x, &format!("{p}.ln_1"))?;
                    let a =
                        self.attention(&ops, &h, &positions, &mut cache.layers[layer], layer, &p)?;
                    let x = ops.binary(BinaryKind::Add, &x, &a)?;
                    let h = self.norm(&ops, &x, &format!("{p}.ln_2"))?;
                    // GPT-2's `Conv1D` stores `[in, out]`: `y = x · w`.
                    let fc = ops.bias(
                        backend.forward(
                            &Matmul,
                            &[&h, self.weight(&format!("{p}.mlp.c_fc.weight"))?],
                        )?,
                        self.optional(&format!("{p}.mlp.c_fc.bias")),
                    )?;
                    let out = ops.bias(
                        backend.forward(
                            &Matmul,
                            &[
                                &ops.unary(c.activation, &fc)?,
                                self.weight(&format!("{p}.mlp.c_proj.weight"))?,
                            ],
                        )?,
                        self.optional(&format!("{p}.mlp.c_proj.bias")),
                    )?;
                    ops.binary(BinaryKind::Add, &x, &out)?
                }
            };
        }

        let final_norm = match c.block {
            Block::Gpt2 => "transformer.ln_f",
            Block::Parallel => "model.final_layernorm",
            _ => "model.norm",
        };
        cache.len = past + t;
        // The norm and the head act row by row: only the rows wanted.
        if logits == Logits::Last {
            x = x.narrow(0, t - 1, 1)?;
        }
        let h = self.norm(&ops, &x, final_norm)?;
        let head = match self.optional("lm_head.weight") {
            Some(w) if !c.tie_embeddings => w,
            _ => embed,
        };
        let mut logits = ops.linear(
            &h,
            head,
            self.optional("lm_head.bias").filter(|_| c.lm_head_bias),
        )?;
        if let Some(cap) = c.final_softcap {
            logits = ops.softcap(&logits, cap)?;
        }
        Ok(logits)
    }

    /// The attention of one block: `[T, hidden]` to `[T, hidden]`, after the
    /// keys and values of `cached`, to which the new ones are added.
    fn attention<B: Backend>(
        &self,
        ops: &Ops<'_, B>,
        h: &Tensor,
        positions: &Tensor,
        cached: &mut Option<(Tensor, Tensor)>,
        layer: usize,
        p: &str,
    ) -> Result<Tensor> {
        let c = &self.config;
        let t = h.shape()[0];
        let (hq, hkv, d) = (c.heads, c.kv_heads, c.head_dim);
        let bias = |name: &str| {
            if c.qkv_bias {
                self.optional(name)
            } else {
                None
            }
        };
        let (q, k, v) = if c.block == Block::Gpt2 {
            let w = self.weight(&format!("{p}.attn.c_attn.weight"))?;
            let qkv = ops.bias(
                ops.0.forward(&Matmul, &[h, w])?,
                self.optional(&format!("{p}.attn.c_attn.bias")),
            )?;
            let width = hq * d;
            (
                qkv.narrow(1, 0, width)?,
                qkv.narrow(1, width, width)?,
                qkv.narrow(1, 2 * width, width)?,
            )
        } else {
            let proj = |name: &str| {
                ops.linear(
                    h,
                    self.weight(&format!("{p}.self_attn.{name}.weight"))?,
                    bias(&format!("{p}.self_attn.{name}.bias")),
                )
            };
            (proj("q_proj")?, proj("k_proj")?, proj("v_proj")?)
        };
        // [T, H·D] → [T, H, D] → [1, H, T, D], as views.
        let heads = |x: Tensor, n: usize, norm: Option<&str>| -> Result<Tensor> {
            let mut x = ops.contiguous(&x)?.reshape(&[t, n, d])?;
            if let Some(name) = norm {
                let w = self.weight(&format!("{p}.self_attn.{name}.weight"))?;
                x = ops.0.forward(
                    &RmsNorm {
                        eps: c.eps,
                        weight_offset: 0.0,
                    },
                    &[&x, w],
                )?;
            }
            Ok(x.reshape(&[1, t, n, d])?.transpose(1, 2)?)
        };
        let mut q = heads(q, hq, c.qk_norm.then_some("q_norm"))?;
        let mut k = heads(k, hkv, c.qk_norm.then_some("k_norm"))?;
        let v = heads(v, hkv, None)?;
        if let Some((theta, scaling, rotary_dim)) = c.rope {
            let rope = Rope {
                theta,
                layout: c.rope_layout,
                rotary_dim: Some(rotary_dim),
                scaling,
            };
            q = ops.0.forward(&rope, &[&q, positions])?;
            k = ops.0.forward(&rope, &[&k, positions])?;
        }
        let (k, v) = match cached.take() {
            Some((k0, v0)) => (
                ops.0.forward(&Concat { dim: 2 }, &[&k0, &k])?,
                ops.0.forward(&Concat { dim: 2 }, &[&v0, &v])?,
            ),
            None => (k, v),
        };
        *cached = Some((k.clone(), v.clone()));
        let attention = Attention {
            causal: true,
            window: c.windows[layer],
            scale: c.query_scale,
            softcap: c.attn_softcap,
        };
        let y = ops.0.forward(&attention, &[&q, &k, &v])?;
        // [1, H, T, D] → [T, H·D].
        let y = ops.contiguous(&y.transpose(1, 2)?)?.reshape(&[t, hq * d])?;
        Ok(if c.block == Block::Gpt2 {
            ops.bias(
                ops.0.forward(
                    &Matmul,
                    &[&y, self.weight(&format!("{p}.attn.c_proj.weight"))?],
                )?,
                self.optional(&format!("{p}.attn.c_proj.bias")),
            )?
        } else {
            let out = if c.block == Block::Parallel {
                "dense"
            } else {
                "o_proj"
            };
            let bias = if c.out_bias {
                self.optional(&format!("{p}.self_attn.{out}.bias"))
            } else {
                None
            };
            ops.linear(
                &y,
                self.weight(&format!("{p}.self_attn.{out}.weight"))?,
                bias,
            )?
        })
    }
}

/// `value` rounded to `dtype`, as a float of that type holds it.
fn rounded(dtype: DType, value: f64) -> Result<f64> {
    let mut bytes = vec![0; dtype.block_bytes()];
    dtype.encode(value, &mut bytes)?;
    Ok(dtype.decode(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resave d d`: a model saved over the directory it was loaded from.
    #[test]
    fn a_model_saves_over_its_own_directory() {
        let dir = std::env::temp_dir().join(format!("kapel-model-resave-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "model_type": "llama", "hidden_size": 8, "num_attention_heads": 2,
            "num_hidden_layers": 1, "vocab_size": 10, "intermediate_size": 16,
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let embed: Vec<f32> = (0..80).map(|i| i as f32 / 8.0).collect();
        let norm = vec![1.0f32; 8];
        let tensors = [
            Tensor::from_slice(&embed, &[10, 8]).unwrap(),
            Tensor::from_slice(&norm, &[8]).unwrap(),
        ];
        kapel_hub::safetensors::save(
            dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &tensors[0]),
                ("model.norm.weight", &tensors[1]),
            ],
            &BTreeMap::new(),
        )
        .unwrap();
        let model = Model::load(&Checkpoint::open(&dir).unwrap()).unwrap();
        // A stale index of shards would hide the saved weights: it goes.
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();
        for _ in 0..2 {
            model.save(&dir).unwrap();
            // The weights mapped from the replaced file still read.
            let embed_again = model.weight("model.embed_tokens.weight").unwrap();
            assert_eq!(embed_again.to_vec::<f32>().unwrap(), embed);
        }
        let again = Model::load(&Checkpoint::open(&dir).unwrap()).unwrap();
        assert_eq!(again.config(), model.config());
        assert_eq!(again.weights.len(), 2);
        for (name, tensor) in &model.weights {
            assert_eq!(
                again.weights[name].to_vec::<f32>().unwrap(),
                tensor.to_vec::<f32>().unwrap(),
                "{name}"
            );
        }
        // Only the two files are left: no temporary.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
