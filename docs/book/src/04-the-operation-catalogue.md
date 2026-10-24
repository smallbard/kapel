# 4. The operation catalogue: kapel-ops

`kapel-ops` is the contract of the whole library. It lists the twenty-three operations a transformer needs, to run and to train, gives each an exact mathematical definition, and checks every call in one place, so that every backend accepts and rejects the same calls with the same errors. It computes nothing itself.

## 4.1 The traits

An operation is a small struct holding its parameters: `Softmax { dim: 1 }`, `Attention::causal()`, `Matmul`. It implements `Operation`:

```rust
pub trait Operation: fmt::Debug + Sync {
    const NAME: &'static str;                                   // "softmax", used in errors
    fn infer(&self, inputs: &[&Tensor]) -> Result<Signature>;   // check, then shape and type of the output
    fn differentiable(&self, index: usize) -> bool;             // does input `index` get a gradient?
    fn infer_backward(&self, inputs: &[&Tensor], grad: &Tensor) -> Result<Signature>;
}

pub struct Signature { pub shape: Vec<usize>, pub dtype: DType }
```

`infer` validates everything that does not depend on the data: the number of inputs, their ranks and shapes, their types, their devices, and the operation's own parameters. `infer_backward` is a provided method: it calls `infer`, then checks that the gradient has the output's shape, a float type, and the inputs' device.

A backend implements `Run<O>` for each operation:

```rust
pub trait Run<O: Operation> {
    fn forward(&self, op: &O, inputs: &[&Tensor]) -> Result<Tensor>;
    fn backward(&self, op: &O, inputs: &[&Tensor], grad: &Tensor)
        -> Result<Vec<Option<Tensor>>>;   // one entry per input, None if not differentiable
}
```

The rule is that every `forward` starts with `op.infer(inputs)?` and every `backward` with `op.infer_backward(inputs, grad)?`. After that line, a backend can rely on the shapes and types it was promised, which keeps kernels free of defensive checks.

`Transfer` moves tensors between the host and the backend's device (`upload`, `download`; the identity for a host backend). Two more traits serve generation, which writes into its cache at every step and repeats the same pass thousands of times:

- `Write` writes into a tensor in place: `zeros` makes a tensor to be written, `write_rows(table, index, rows)` writes `rows[i]` into row `index[i]` of `table`, converted to the table's type, and `copy_from_host` fills a tensor of the backend from a host tensor of the same shape and type. A tensor is written only through its only handle (`Tensor::storage_mut`), so no clone or view sees the change.
- `Record` records a pass once and replays it. `record` takes a closure that queues the pass and returns a `Recording`, whose `replay` runs it again on the current contents of the tensors it read and rewrites its `output`. A backend that computes each operation when it is called answers `records() == false` and `record` gives `None`: the pass is simply run there. On the card, a recording is a CUDA graph (chapter 7).

`Backend` is the supertrait of `Write`, `Record` and all twenty-three `Run` implementations, with a blanket implementation, so any type that implements them all is a backend:

```rust
let y = backend.forward(&Matmul, &[&x, &w.transpose(0, 1)?])?;   // y = x · wᵀ, w read in place
let z = backend.forward(&Softmax { dim: 1 }, &[&y])?;
```

The shared checks are small private functions: `arity`, `float`, `indices` (the type must be `U32` or `I32`), `same_device` and `rank`. The checks of `Write` are public, since they are not operations with an `infer`: `check_write` (a float table, row-major from the start of its storage, its only handle; rows of the right shape), `check_written` (each written row in range, none twice, on the values the backend reads) and `check_copy`. The module `shape` exposes the rules other crates reuse: `broadcast` (NumPy-style, aligned from the last dimension, with an overflow check), `check_dims` and `reduced`. The module `special` provides `erf`, accurate to about 1e-15, and the normal distribution function used by the exact GELU.

## 4.2 The catalogue

| Operation | Inputs → output | Definition and rules | Gradient to |
| --- | --- | --- | --- |
| `Unary` | x → same shape | Neg, Exp, Log, Sqrt, Rsqrt, Square, Abs, Sigmoid, Tanh, Relu, Silu, GeluTanh, GeluErf | x |
| `Binary` | a, b → broadcast shape | Add, Sub, Mul, Div, Max, Min; a broadcast input's gradient is summed over the repeated dimensions | a, b |
| `Affine` | x → same | scale · x + bias | x |
| `Glu` | x \[…, 2n\] → \[…, n\] | f(x\[…, :n\]) · x\[…, n:\], the gate first: SwiGLU with SiLU, GeGLU with GELU | x |
| `Cast` | x → `dtype` | as `DType::encode`; the gradient passes between float types only | x if floats |
| `Contiguous` | x → same | a row-major copy, bit for bit | x |
| `Concat` | inputs → joined along `dim` | same type and shape except `dim` | each input |
| `Matmul` | \[…, M, K\] · \[…, K, N\] | batch dimensions broadcast; strided inputs allowed | a, b |
| `QuantizedMatmul` | x \[…, K\] · w₁ \[N₁, K\], w₂ … quantized | the weights, of any formats, at their exact dequantized value, their rows stacked into one `[ΣNᵢ, K]`; K a whole number of blocks | x only |
| `Sum`, `Mean` | x → reduced | over a list of dims, `keepdim` | x |
| `Max` | x → reduced | along one dim; the gradient goes to the first maximum | x |
| `ArgMax` | x → reduced, `U32` | the position of the maximum along one dim, the first on a tie; NaN is never the maximum | none |
| `Softmax`, `LogSoftmax` | x → same | along `dim`; log-softmax computed without forming the softmax | x |
| `CrossEntropy` | x \[N, V\], targets \[N\] → \[N\], f32 | ln Σⱼ e^(x\[n, j\]) − x\[n, t\[n\]\], 0 for the target `IGNORE` (all bits set); the softmax never formed (chapter 15) | x |
| `LayerNorm` | x \[…, C\], weight \[C\], bias \[C\] | (x − μ) / √(σ² + eps) · w + b, biased variance | all |
| `RmsNorm` | x \[…, C\], weight \[C\] | x / √(mean x² + eps) · (offset + w); offset 1 for Gemma | all |
| `Attention` | q, k, v, optional mask | scaled dot-product, grouped heads, causal, window, softcap | q, k, v |
| `PagedAttention` | q \[R, Hq, D\], keys, values \[L, Hkv, D\], pages \[S, P\], rows \[R, 2\] | attention of new rows over a cache kept by pages (section 4.3) | none |
| `Rope` | x \[…, D\], positions | rotary embeddings with five scalings | x |
| `GatherRows` | src \[R, …\], idx | y\[i\] = src\[idx\[i\]\]; out-of-range index is an error; a quantized table gives its rows in f32 | src, unless quantized |
| `ScatterAddRows` | src, idx → \[rows, …\] | adjoint of gather; repeated rows are summed | src |

Normalization weights and the attention mask may be in any float type, not necessarily that of the main input: a model with bf16 weights can normalize f32 activations, and the weight gradients come back in the weights' own type.

## 4.3 A closer look at attention

Attention carries most of the subtle rules, and all of them are fixed in its documentation so that three backends can agree.

- Inputs are `q [B, Hq, Tq, D]`, `k [B, Hkv, Tk, D]`, `v [B, Hkv, Tk, Dv]`. `Hq` is a multiple of `Hkv`, and query head h reads key head `h / (Hq / Hkv)`: grouped-query attention.
- Query i sits at position `p(i) = Tk − Tq + i`. The queries are the last `Tq` of the `Tk` positions, which is exactly the situation of generation with a cache.
- Key j is visible if causality is off or `j ≤ p(i)`, and the window is off or `p(i) − j < window`.
- The score is `s = scale · q·k + mask`, with `scale` defaulting to 1/√D. With a softcap (Gemma 2), `scale · q·k` becomes `softcap · tanh(scale · q·k / softcap)` before the mask is added.
- A query that sees no key, or whose visible keys are all masked with −∞, gives zeros and passes no gradient. Finite scores, however low, take part in the softmax: a padding mask of `f32::MIN`, as `transformers` uses, gives uniform weights rather than zeros.

`infer` rejects D = 0 (the default scale would be 1/√0), more queries than keys, a window of 0, and a mask that does not broadcast to `[B, Hq, Tq, Tk]`.

### Attention over pages

Generation keeps the keys and values of every sequence it serves in one cache, cut into pages of `page` consecutive lines (16 in `kapel-models`). A sequence owns a list of pages, not a contiguous block, so sequences of different lengths share the cache without moving anything when one grows or ends. `PagedAttention` computes the attention of the new rows of several sequences at once over that cache:

- `q [R, Hq, D]` holds one row per query, whatever its sequence. The cache is `keys [L, Hkv, D]` and `values [L, Hkv, Dv]`, `L` lines.
- `pages [S, P]` is the page table: position j of sequence s is at line `pages[s, j / page] · page + j % page`.
- `rows [R, 2]` says, for query r, its sequence `rows[r, 0]` and the number c = `rows[r, 1]` of positions it sees, the last of which is its own. Position j is visible if j < c and, with a window, c − 1 − j < window.

The scores, the softcap, the grouped heads and the default scale are those of `Attention`; a query that sees nothing gives zeros. The keys and values may have their own float type, the cache's (`f16` or `f8` with `bf16` queries), and the output has the type of `q`. The operation is for inference: no input receives a gradient. The values of `pages` and `rows` can only be checked when they are read, so `check_indices` is public and every backend calls it on the values it reads.

## 4.4 RoPE and its scalings

`Rope` rotates pairs of components by an angle proportional to the position. The pairs are either `(j, j + R/2)` (layout `Half`: GPT-NeoX, Llama, most models) or `(2j, 2j + 1)` (layout `Interleaved`: GPT-J and the original paper), and only the first `rotary_dim` components are rotated. Pair j at position p turns by `p · ωⱼ`:

```math
a' = s\,(a\cos\theta - b\sin\theta), \qquad b' = s\,(b\cos\theta + a\sin\theta), \qquad \theta = p\,\omega_j
```

The frequencies come from `Rope::frequencies`, which folds the scaling in. `None` gives ωⱼ = θ^(−2j/R). `Linear` divides them by a factor. `Ntk` replaces θ by θ · factor^(R/(R−2)). `Llama3` keeps high frequencies, divides low ones and blends in between, by wavelength. `Yarn` ramps between interpolated and original frequencies, and multiplies the rotation by an attention factor s, by default 0.1 · ln(factor) + 1. `infer` checks every parameter: finite and positive values, more than two rotated components for NTK, and the order of Llama 3's frequency factors.

## 4.5 Adding an operation

The catalogue grows by a fixed recipe, which the rest of the book will make concrete:

1. Define the struct and its `Operation` impl in `kapel-ops`, with the mathematical definition in its documentation and every check in `infer`.
2. Add it to `Backend`.
3. Write the reference `forward` and `backward` in `kapel-ref`, as a direct transcription of the definition.
4. Add cases to the conformity suite, including empty, strided and edge cases. The finite-difference check verifies the reference gradient.
5. Implement it in the CPU and CUDA backends until they pass the suite.
6. If the fast version is an optimization, add a benchmark program next to it.
