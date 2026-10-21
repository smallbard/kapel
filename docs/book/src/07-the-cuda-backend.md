# 7. The CUDA backend: kapel-cuda

`kapel-cuda` runs the catalogue on NVIDIA GPUs. It is written entirely in Rust, host and device alike, with two families of kernels: simple *thread kernels*, one thread per element or per line, for everything but products and attention; and *tile kernels*, compiled from Rust by cutile-rs, for matrix products and FlashAttention. On an RTX 3090, bf16 products reach 69 Top/s at 4096³, and attention 44 to 51 Top/s forward and 37 to 41 backward.

## 7.1 Two toolchains, one stable build

|  | Thread kernels | Tile kernels |
| --- | --- | --- |
| Tool | cuda-oxide (`cargo oxide`) | cutile-rs |
| Source | `kernels/src/lib.rs`, a separate crate on a pinned nightly | `tile.rs` and `attention.rs` inside the backend |
| Compiled | ahead of time to PTX, checked in as `kernels.ptx` | at first launch by `tileiras`, then cached in `~/.cache/cutile/kernels` |
| Covers | copies, conversions, element-wise, reductions, softmax, norms, RoPE, gathers | matrix products, attention forward and backward |
| Arithmetic | f32; 16-bit inputs are widened, then rounded back | f32, bf16, f16, with f32 accumulators |

The split exists because the published crates must build on stable Rust, and cuda-oxide needs a nightly compiler. The kernels crate is therefore kept outside the workspace and compiled by `kernels/build.sh`; the backend embeds the result with `include_str!("../kernels.ptx")` and loads it at startup. Tile kernels need CUDA 13.2 or later on the machine.

The whole backend sits behind the `cuda` feature, off by default, so the workspace builds anywhere. One Cargo detail matters for performance: cutile compiles kernels without optimization when its compiler crate carries debug information, so the root `Cargo.toml` turns debug information off for `cutile-compiler` in every profile.

## 7.2 One backend per device

`Cuda::new(ordinal)` returns the backend of that GPU, creating it only if none is alive: backends are kept in a process-wide table of weak references. All work on a device's tensors is thus queued on one stream, in order, and a kernel never reads a buffer that another stream may still be writing. A tensor that belongs to another backend is refused with a typed error.

Opening a backend creates a context and one stream, loads the PTX module, and prepares the tile-kernel runtime on the same stream.

**The buffer pool.** Freeing device memory synchronizes the device, so a dropped tensor's buffer goes back to a pool keyed by exact size instead. Because there is one stream, a buffer can be handed out again even while kernels that used it are still queued. If an allocation fails, the backend empties the pool and its cached tables and tries once more.

**Cached tables.** Reductions, softmax, max and RoPE need small index tables derived from shapes. Uploading from ordinary host memory waits for the stream, so these tables are cached on the device, up to 64 MiB, with oldest-first eviction.

**What waits.** Most operations return as soon as their kernels are queued. The exceptions are `download` and `upload`, the first use of a shape by an operation that needs a table, and the operations that still go through the host: gathers and scatters (their row lists), casts to or from non-float types, quantized products (dequantized on the host for now), and attention masks that are not `f32`.

## 7.3 Launching a thread kernel

Kernels are loaded by name at run time, so nothing checks at compile time that a launch passes what the kernel expects. Kapel closes that gap with a table:

```rust
// signatures.rs: every kernel and its parameters, as the host sees them
const KERNELS: &[(&str, &[Param])] = &[ ("affine", &[F32, F32, Slice, Slice]), /* … 29 kernels */ ];
```

Arguments are built with a small `Args` builder (`.u32()`, `.f32()`, `.slice()`, `.view()`), and `launch` checks them against the table before anything is queued. It also refuses more than 2³² threads or a slice longer than 2³² elements with `Overflow`, because kernels index in 32 bits. A host-only test parses the `.entry` declarations of the checked-in PTX and compares them with the table, so a kernel changed without updating the host fails the ordinary test suite on every platform. In CI, the GPU job also rebuilds the PTX and fails if it differs from the checked-in file.

Strided tensors reach kernels as a `View`, defined once in `kapel-cuda-abi`, a `no_std` crate compiled by both the nightly kernel crate and the stable backend:

```rust
#[repr(C)]
pub struct View { pub shape: [u32; 8], pub strides: [u32; 8], pub offset: u32 }  // 68 bytes
// View::at(flat) turns a row-major index into a storage position; a stride of 0 broadcasts
```

Operation codes such as `unary::EXP` are plain `u32` constants rather than enums, because an unknown value must not be undefined behaviour: kernels return NaN for a code they do not know.

## 7.4 The thread kernels

The kernels follow two rules from their module documentation: no two threads write the same place, and no sum depends on scheduling. There are no atomics.

| Kernels | Role |
| --- | --- |
| `copy_view_*`, `copy_into_*` | strided copies by element size; `Contiguous` and `Concat` |
| `widen_16`, `narrow_16` | bf16 and f16 to f32 and back, bit-exact with the host, NaN made quiet |
| `unary_forward`, `unary_backward`, `affine`, `binary_forward`, `binary_backward` | element-wise operations through broadcast views |
| `reduce_sum`, `spread` | sums and means, terms added in order, and their backward |
| `line_max`, `place` | maximum with its first position, and its backward |
| `softmax_lines`, `softmax_lines_backward` | one thread per line, softmax or log-softmax |
| `norm_rows`, `norm_rows_backward`, `column_sum` | LayerNorm and RMSNorm, one warp per row, weight gradients summed in order |
| `rope_table`, `rope` | angles in f64, then the rotation |
| `gather_rows`, `scatter_rows` | row gather, and a scatter-add that sums each target row's sources in order of appearance |

Only products and attention compute in the inputs' 16-bit type. Every other operation goes through a macro, `in_f32!`, that widens the inputs, computes in f32 and rounds the output, and each gradient, back to the right type.

## 7.5 Tiled products

The product kernel computes one output tile of `BM × BN` per tile block, accumulating along K in f32 with the hardware's matrix instructions:

```rust
let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
for i in 0i32..num_tiles(&part_x, 2) { /* load tx, ty */ acc = mma(tx, ty, acc); }
let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
```

Out-of-bounds loads read zeros and out-of-bounds stores are dropped, so tile sizes need not divide the problem. Operands are read in place through their strides. Tile sizes come from a sweep on the RTX 3090: two blocks of 128 × 64 × 32 per multiprocessor reach 70 Top/s at 4096³, against 53 for one block of 128 × 64 × 64, and neighbouring choices can be three times slower.

## 7.6 FlashAttention 2

Attention never forms the full score matrix. Each tile block owns one head and one tile of queries, and walks the key tiles that its queries can see, keeping a running maximum `m`, a running sum `l` and an accumulator for the output:

```math
m' = \max(m, \max_j s_j), \quad \alpha = 2^{(m - m')\log_2 e}, \quad l' = \alpha\,l + \sum_j 2^{(s_j - m')\log_2 e}, \quad \text{acc}' = \alpha\,\text{acc} + W V
```

The exponentials are computed in base 2, a single multiply-add per score when there is no mask. With a mask, the code computes `(s − m) · log₂e`, because `f32::MIN · log₂e` would overflow. The running maximum starts at −∞, with a guard so that a query that has seen only −∞ computes nothing undefined; such a query ends with zeros, as the catalogue requires.

Visibility is cheap to apply. A helper computes, for a query tile, the range of key tiles to visit and, inside it, the range that is fully visible to every query; only the edge tiles pay for the causal, window and bounds tests. Head widths are padded to a power of two, and a tile needed transposed is loaded then transposed, because reading through swapped strides is about three times slower.

The backward pass reruns the forward to obtain the output and the per-row statistics, then runs three kernels:

1. `attention_delta`: δ = rowsum(g ⊙ y).
2. `attention_grad_q`: one block per query tile; recompute P from the statistics, then ∂s = P ⊙ (g·vᵀ − δ), times the softcap slope, and ∂q = Σ ∂s · k.
3. `attention_grad_kv`: one block per key head and key tile, looping over every query head of its group; it accumulates ∂v += Pᵀg and ∂k += ∂sᵀq.

Because each output is written by exactly one tile block, adding its terms in a fixed order, there is no atomic accumulation and the same inputs give the same bits on every run.

| Type, head width | Forward tile | ∂q tile | ∂k, ∂v tile |
| --- | --- | --- | --- |
| f32 | 32 × 32 | 32 × 32 | 32 × 32 |
| 16-bit, ≤ 64 | 128 × 64 | 64 × 32 | 64 × 32 |
| 16-bit, ≤ 128 | 128 × 32, two blocks | 64 × 32, 8 warps | 32 × 32, 8 warps |
| 16-bit, > 128 | 64 × 32, 8 warps | 32 × 32, two blocks | 64 × 64, 8 warps |

These are the best of a sweep on the RTX 3090, causal, from 1 to 4 thousand positions; neighbouring choices can be up to six times slower.

## 7.7 Testing on the card

The GPU tests run the conformity suite in `f32`, `bf16` and `f16`, plus targeted tests: every one of the 65,536 bf16 and f16 bit patterns converted exactly as the host converts them; attention over many tiles with windows, masks, softcaps and grouped heads, run twice to check identical bits; extreme masks; norms with bf16 weights; one backend per device; and refused 32-bit overflows. A parity test runs whole models on the card and compares their logits with `transformers`: within 2e-5 in `f32`, and in `bf16` within twice the CPU's own `bf16` error.
