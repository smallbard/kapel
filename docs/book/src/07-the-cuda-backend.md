# 7. The CUDA backend: kapel-cuda

`kapel-cuda` runs the catalogue on NVIDIA GPUs. It is written entirely in Rust, host and device alike, with two families of kernels: *thread kernels*, one thread, warp or block per element, row or chunk, for everything but dense products and prompt attention; and *tile kernels*, compiled from Rust by cutile-rs, for matrix products and FlashAttention. Generation adds three pieces on top: attention over a cache kept by pages, products that read quantized weights in place, and passes recorded once as CUDA graphs and replayed. On an RTX 3090, bf16 products reach 69 Top/s at 4096³, attention 44 to 51 Top/s forward and 37 to 41 backward, and Llama 3.2 3B in bf16 decodes at 105 tokens per second for one sequence and 3330 for 64 together, 90 % of vLLM.

## 7.1 Two toolchains, one stable build

|  | Thread kernels | Tile kernels |
| --- | --- | --- |
| Tool | cuda-oxide (`cargo oxide`) | cutile-rs |
| Source | `kernels/src/lib.rs`, a separate crate on a pinned nightly | `tile.rs` and `attention.rs` inside the backend |
| Compiled | ahead of time to PTX, checked in as `kernels.ptx` | at first launch by `tileiras`, then cached in `~/.cache/cutile/kernels` |
| Covers | copies, conversions, element-wise, reductions, softmax, norms, RoPE, gathers, paged attention, quantized products | matrix products, attention forward and backward |
| Arithmetic | f32; 16-bit and 8-bit values are widened in registers, then rounded back | f32, bf16, f16, with f32 accumulators |

The split exists because the published crates must build on stable Rust, and cuda-oxide needs a nightly compiler. The kernels crate is therefore kept outside the workspace and compiled by `kernels/build.sh`; the backend embeds the result with `include_str!("../kernels.ptx")` and loads it at startup. Tile kernels need CUDA 13.2 or later on the machine.

The whole backend sits behind the `cuda` feature, off by default, so the workspace builds anywhere. One Cargo detail matters for performance: cutile compiles kernels without optimization when its compiler crate carries debug information, so the root `Cargo.toml` turns debug information off for `cutile-compiler` in every profile.

## 7.2 One backend per device

`Cuda::new(ordinal)` returns the backend of that GPU, creating it only if none is alive: backends are kept in a process-wide table of weak references. All work on a device's tensors is thus queued on one stream, in order, and a kernel never reads a buffer that another stream may still be writing. A tensor that belongs to another backend is refused with a typed error.

Opening a backend creates a context and one stream, loads the PTX module, and prepares the tile-kernel runtime on the same stream.

**The buffer pool.** Freeing device memory synchronizes the device, so a dropped tensor's buffer goes back to a pool keyed by exact size instead. Because there is one stream, a buffer can be handed out again even while kernels that used it are still queued. If an allocation fails, the backend empties the pool and tries again; only if that is not enough does it drop its cached tables too, and try a last time. A pass about to be recorded counts on finding the tables it just made.

**Cached tables.** Reductions, softmax, max and RoPE need small index tables derived from shapes. Uploading from ordinary host memory waits for the stream, so these tables are cached on the device, up to 64 MiB; the least recently used goes first.

**Host mirrors.** An upload of at most 1 MiB keeps a copy of its bytes on the host, next to the device buffer. Token ids, positions and page tables are such uploads, and an operation that needs their values on the host (the rows of a scatter, the prompts of a paged attention) reads the copy instead of waiting for the stream. A gather reads its row indices in place on the device when they are contiguous `u32` or `i32`. A kernel that writes into a storage drops its mirror.

**Free memory.** `Cuda::memory()` gives the free and total bytes of the device, as the driver counts them (other processes included); `kapel serve` sizes its cache with it, after `clear_pool`. `Cuda::memory_in_use()` counts the bytes held by the backend's live tensors, and `Cuda::peak_memory()` the most it has counted since `reset_peak_memory()`: what `torch.cuda.memory_allocated` and `max_memory_allocated` count, which training compares (chapter 15).

**Attentions kept for training.** The backward pass of an attention needs its output and the statistics of its softmax, which a call of `backward` does not receive. Without more, it computes the forward pass again. With `Cuda::keep_attention_for_backward(true)`, which `kapel train` sets, each forward pass keeps them, with weak handles on its inputs' storages, and the backward pass of the same inputs, views and operation takes them: 6 ms of a step of GPT-2 124M, at the cost of the outputs' memory until then. An entry whose inputs are gone is dropped, so a pass without gradient keeps nothing for long.

**What waits.** Most operations return as soon as their kernels are queued. The exceptions are `download` and `upload`, the first use of a shape by an operation that needs a table, index tensors made on the device and then needed on the host, and the operations that still go through the host: casts to or from non-float types, products and gathers by weights in Q2_K or Q3_K (dequantized on the host), and attention masks that are not `f32`.

## 7.3 Launching a thread kernel

Kernels are loaded by name at run time, so nothing checks at compile time that a launch passes what the kernel expects. Kapel closes that gap with a table:

```rust
// signatures.rs: every kernel and its parameters, as the host sees them
const KERNELS: &[(&str, &[Param])] = &[ ("affine", &[F32, F32, Slice, Slice]), /* … 58 kernels */ ];
```

Arguments are built with a small `Args` builder (`.u32()`, `.f32()`, `.slice()`, `.view()`), and `launch` checks them against the table before anything is queued. It also refuses more than 2³² threads or a slice longer than 2³² elements with `Overflow`, because kernels index in 32 bits. A host-only test parses the `.entry` declarations of the checked-in PTX and compares them with the table, so a kernel changed without updating the host fails the ordinary test suite on every platform. In CI, the GPU job also rebuilds the PTX and fails if it differs from the checked-in file.

Strided tensors reach kernels as a `View`, defined once in `kapel-cuda-abi`, a `no_std` crate compiled by both the nightly kernel crate and the stable backend:

```rust
#[repr(C)]
pub struct View { pub shape: [u32; 8], pub strides: [u32; 8], pub offset: u32 }  // 68 bytes
// View::at(flat) turns a row-major index into a storage position; a stride of 0 broadcasts
```

Operation codes such as `unary::EXP` are plain `u32` constants rather than enums, because an unknown value must not be undefined behaviour: kernels return NaN for a code they do not know. The same goes for types: `float::F32`, `BF16`, `F16`, `F8E4M3` and `F8E5M2`, and the quantized formats in `quant`, are codes that a kernel decodes per element.

## 7.4 The thread kernels

The kernels follow two rules from their module documentation: no two threads write the same place, and no sum depends on scheduling. There are no atomics.

| Kernels | Role |
| --- | --- |
| `copy_view_*`, `copy_into_*` | strided copies by element size; `Contiguous` and `Concat` |
| `widen_16`, `narrow_16`, `widen_8`, `narrow_8` | bf16, f16, f8 E4M3 and E5M2 to f32 and back, bit-exact with the host, NaN made quiet |
| `unary_forward`, `unary_backward`, `affine`, `binary_forward`, `binary_backward` | element-wise operations through broadcast views; `unary_backward` reads its inputs in their own types |
| `widen_16v`, `narrow_16v`, `binary_16`, `unary_16`, `unary_backward_16` | the same on 16-bit values in whole aligned pieces, eight values a thread in 128-bit loads and stores (800–840 GB/s where one value a thread gave 290–435); `binary_16` takes a second operand of the same shape or a row repeated (a bias) |
| `unary_any`, `affine_any`, `binary_any`, `glu_any`, `glu_backward` | the same, and `Glu`, reading and writing one type, 16-bit included |
| `reduce_sum`, `reduce_sum_segments`, `reduce_sum_rows`, `reduce_sum_columns`, `spread` | sums and means of any float type, and their backward; beyond 128 terms, in stages of 128 adjacent terms for all results at once, until a stage has 128 or fewer, in a fixed order; over the last dimensions (`_rows`) or the first (`_columns`, the gradient of a bias), without a table of offsets (chapter 15) |
| `line_max`, `place` | maximum with its first position, and its backward |
| `softmax_lines`, `softmax_lines_backward` | one thread per line, softmax or log-softmax |
| `norm_rows`, `norm_rows_backward` | LayerNorm and RMSNorm, one warp per row, the backward pass reading and writing the rows in their type, weight gradients summed by the stages of `reduce_sum` |
| `cross_entropy_rows`, `cross_entropy_rows_backward` | `CrossEntropy`, one block of 256 threads per row, the logits read in their type by their rows' stride, the maximum and the sum of exponentials in one reading; the gradient in the padded layout of the logits when they are the first columns of wider rows |
| `adamw` | `Update::adamw`: a parameter and its moments in place, one thread per element, the gradient read in its type through its view |
| `norm_any`, `norm_block_any`, `norm_16` | norms in the rows' own type; `norm_16`, one block per row in 128-bit loads, up to 8192 columns |
| `rope_table`, `rope`, `rope_any` | angles in f64, then the rotation, in f32 or in the rows' type |
| `gather_rows`, `scatter_rows`, `write_rows_any` | row gather, a scatter-add that sums each target row's sources in order of appearance, and rows written in place |
| `argmax_rows`, `argmax_parts`, `argmax_merge` | the largest value of each row, NaN ignored, for long rows in parts then merged |
| `paged_attention`, `paged_attention_16`, `paged_combine` | attention over a cache by pages (section 7.7) |
| `qgemv`, `qgemv_*`, `dequantize`, `dequantize_rows` | products and gathers by quantized weights (section 7.9) |

Backward passes, and forward passes in types without a kernel of their own, go through a macro, `in_f32!`, that widens the inputs, computes in f32 and rounds the output, and each gradient, back to the right type. The forward operations of a decoding step have a second branch, `native:`, taken when every input has the output's type: element-wise operations, norms, RoPE, `Glu` and row writes then read and write bf16 or f16 directly, computing each element in f32 in registers, with the same rounding. A 16-bit model thus never writes an f32 copy of its activations: at one token, those copies cost more than the products.

## 7.5 Tiled products

The product kernel computes one output tile of `BM × BN` per tile block, accumulating along K in f32 with the hardware's matrix instructions:

```rust
let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
for i in 0i32..num_tiles(&part_x, 2) { /* load tx, ty */ acc = mma(tx, ty, acc); }
let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
```

Out-of-bounds loads read zeros and out-of-bounds stores are dropped, so tile sizes need not divide the problem. Operands are read in place through their strides. The weights of a linear layer are stored `[N, K]` and multiplied transposed; reading them through swapped strides is five times slower, so a second kernel, `batched_matmul_nt`, loads each tile of `[N, K]` along `K`, its contiguous dimension, and transposes it in registers. A third, `batched_matmul_tn`, does the same for a first factor read transposed, `xᵀ · g`, the gradient of a weight in training: through swapped strides it ran at 13 to 25 Top/s on GPT-2's shapes, and at 54 to 60 with its own kernel. Each starts its `f32` accumulator from a bias row `[1, N]`, loaded and repeated over the tile's rows, zeros when the product has none: a linear layer's bias costs no pass of its own.

Tile sizes depend on the number of rows `m`, from sweeps on the RTX 3090 (`tiles_for` in `tile.rs`):

| Rows `m` (16-bit) | Tile `BM × BN × BK` | Blocks per multiprocessor |
| --- | --- | --- |
| ≤ 16, `n` ≤ 1024 | 16 × 32 × 256 | 4 |
| ≤ 16 | 16 × 64 × 128 | default |
| ≤ 32 | 32 × 64 × 128 | 2 |
| ≤ 256 | 64 × 64 × 64 | 2 |
| more | 64 × 128 × 32 | 2 |

Few rows (decoding) read the weights once each in long, thin tiles, and are limited by bandwidth: at one row, the products of Llama 3.2 3B take 7.9 ms per token, 92 % of the card's measured bandwidth. Many rows are limited by arithmetic: 60 to 70 Top/s, against 53 for one block of 128 × 64 × 64; neighbouring choices can be three times slower. f32 uses 64 × 64 × 32.

## 7.6 FlashAttention 2

Attention never forms the full score matrix. Each tile block owns one head and one tile of queries, and walks the key tiles that its queries can see, keeping a running maximum `m`, a running sum `l` and an accumulator for the output:

```math
m' = \max(m, \max_j s_j), \quad \alpha = 2^{(m - m')\log_2 e}, \quad l' = \alpha\,l + \sum_j 2^{(s_j - m')\log_2 e}, \quad \text{acc}' = \alpha\,\text{acc} + W V
```

The exponentials are computed in base 2, a single multiply-add per score when there is no mask. With a mask, the code computes `(s − m) · log₂e`, because `f32::MIN · log₂e` would overflow. The running maximum starts at −∞, with a guard so that a query that has seen only −∞ computes nothing undefined; such a query ends with zeros, as the catalogue requires.

Visibility is cheap to apply. A helper computes, for a query tile, the range of key tiles to visit and, inside it, the range that is fully visible to every query; only the edge tiles pay for the causal, window and bounds tests. Head widths are padded to a power of two, and a tile needed transposed is loaded then transposed, because reading through swapped strides is about three times slower.

With documents (section 4's `Attention`), each kernel reads, instead of a mask of `Tq × Tk` values, where the document of each token starts: a query and a key are compared by these starts, a tile of each, and the scores of other documents become −∞. Documents being contiguous, no query of a tile sees the keys before its first query's document: the forward kernel and the one of ∂q begin at that key tile, read as a scalar (`tile_to_scalar`), and the one of ∂k and ∂v leaves out the query tiles whose first query's document starts after its keys. Only −∞ hides a score, so the statistics are those of attention without a mask. On GPT-2 124M with documents of FineWeb-Edu (about a thousand tokens each), a step costs 317.6 ms with documents and 315.9 without; an additive mask of the same documents, `[16, 1, 1024, 1024]` in `f32`, cost 441 ms. Two rules of the tile compiler showed on the way: a helper that takes a tensor fails its type inference (the documents are read in the kernels' bodies), and a tile loaded at an index beyond a tensor faults, even for a placeholder never used (the placeholder's index is multiplied by zero).

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

## 7.7 Paged attention

Generation keeps the keys and values of every sequence in two tables `[L, Hkv, D]`, cut into pages of 16 lines; each sequence has a list of pages (see chapter 10). `PagedAttention` takes the new queries `[R, Hq, D]`, the tables, the page lists `[S, P]` and, for each query, its sequence and the number of positions it sees. New keys and values are first written into their lines by `Write::write_rows`, in place.

The kernel `paged_attention_16` reads bf16, f16 or f8 caches; `paged_attention` is the generic version, for f32 and the other cases. Each tile block takes up to four query heads of one key head, for one query and one *chunk* of positions; the lanes of a warp each read a key or a value in one load of 128 or 64 bits, keeping the queries in registers. f8 values go through a lookup table shared by the block. Every chunk writes its own maximum, sum and partial output, and `paged_combine` adds the chunks in order, as FlashAttention adds its tiles.

The chunk length depends on the shape alone, never on the data: positions are cut so that a launch has about `BLOCKS` = 164 tile blocks per 256 positions, two per multiprocessor, down to chunks of 32. The count that sets it is the *sequences* of the pass times its key heads and sub-blocks of query heads (`spread` in `paged.rs`), not its queries. The queries of one sequence therefore fall into the same chunks however many of them a pass holds, and give the same bits: a token decoded alone, and the same position checked with the proposals of speculative decoding (chapter 11), agree to the last bit. When the queries set the chunk, checking six proposals or more cut the positions differently from a token decoded alone, and greedy speculative answers drifted; batch throughput is the same either way. A query with fewer positions than its chunks leaves the last ones empty, and they return at once. With many sequences, each query already fills the card and a chunk covers a whole sequence; with one, its positions are spread over the card.

Paged kernels read each query's keys apart, which is cheap for a token but grows as the square of a long prompt. A sequence with at least `PREFILL_QUERIES` = 384 consecutive queries is therefore handled by FlashAttention 2: its keys are gathered from the pages into a contiguous tensor, attention runs causally from the first query's position, and the result is placed into the output rows. A prompt of 4096 tokens of Llama 3.2 3B takes 0.76 s this way, against 1.5 s through the paged kernels.

**Linear attention.** `causal_conv` takes one thread per channel, which goes through the rows of the pass in order, its column of the state its own. `delta_rule` takes one block of 8 warps per value head and 32 columns of its states: warp `w` keeps rows `32w .. 32w + 32` of the columns in its registers, lane `l` column `l`, while the rows are of the same sequence, so that every load and store of a row of the state is one contiguous piece; the sums over a column add each warp's rows in order, then the warps' through shared memory in order. The delta rule never mixes columns: a prompt reads and writes its states once, whatever its length, but its rows go one after the other, which a chunked form, with products by tiles, would compute faster.

**Experts and small products.** `top_k` takes one warp per row: each lane reads its values once and, at each of the `k` passes, offers its largest not yet taken to a butterfly. `gather_columns` takes one thread per output. `expert_gemv` takes one block of 8 warps per route (a token and one of its experts) and 8 rows of the expert's weights, the block's input first read once into shared memory in `f32`, one empty place after every 8 so that lanes reading 8 consecutive elements read different banks. IQ4_XS and IQ3_S have paths of their own, a lane a 32nd of each block of 256; the other formats are decoded as `qgemv` decodes them, and float weights read through their view. For Q8_0 weights and 16-bit activations, `bytes_of_rows` first rounds each row of activations to bytes by blocks of 32, as llama.cpp does, and `expert_gemv_bytes` multiplies four bytes at a time (`dp4a`): one and a half times faster (example `experts`); for IQ3_S and IQ4_XS it gains nothing, and they keep their activations. Products of at most 8 rows by float weights of at most 512 rows, without a bias (a router, the gates of linear attention), go to `gemv_float`, one block of 256 threads per output, which the tiles computed with a block or two: each output is one block's whatever the number of rows, so a row's bits do not depend on the others. The experts the card chose are never read back: an index out of range there gives NaN rather than an error, while indices uploaded from the host are checked.

## 7.8 Recorded passes

At one token, a model is a few hundred small kernels, and launching them costs as much as running them. `Record::record` runs a pass while the backend's stream is captured, in *relaxed* mode, and instantiates the CUDA graph it gives; `Recording::replay` then launches every kernel of the pass in one call to the driver (`graph.rs`).

A replay writes into the very buffers of the recorded pass, so these must not go back to the pool. While recording, the backend allocates from an `Arena` instead: buffers freed during the pass are reused by its next operations, as the pool does, and those in use at the end stay the recording's until it is dropped. Inputs are replaced between replays by `Write::copy_from_host`, which copies new bytes into the same buffer.

A recorded pass must not wait for the device or copy from the host, since nothing runs during capture. The first pass of a shape is therefore run once before it is recorded: its tile kernels are compiled and its tables made. Rotary angle tables are cached by the bytes of their positions, and kept with the recording, so that the layers of a pass share one table, computed once per replay. Host mirrors (section 7.2) give index values without waiting.

A graph also reads the backend's cached tables at every replay, long after the operation that looked them up has dropped its handle. While recording, every table looked up (`Cuda::table`, found or made) is therefore kept in the arena (`Arena::tables`, through `retain_in_arena`): the cache may let the table go, but its memory is not reused while the recording lives.

A capture that fails (the pass returned an error, or the driver refused to end the capture, for lack of memory say) ends the capture all the same, destroys the partial graph, empties the pool, so that a second attempt has the memory, and returns the error. The step itself was computed by the eager pass before the capture, so a caller loses only the recording (chapter 10, `Recorded::Unrecorded`).

The stream is the device's only one, so while a pass records, no other thread may use the backend: its work would be recorded too. The tests of `tests/recorded.rs` take a lock for this, and check that replays give the bits of eager steps, that a recording changes nothing around it, that a replay takes new inputs, that the positions of one sequence give the same logits checked sixteen at a time as one at a time (`proposals_checked_together_give_the_logits_of_steps`, which fails if the queries set the chunks of paged attention), that a recording of eight rows serves three sequences then two of them (its padding rows writing lines of their own, `a_recording_serves_smaller_batches`), and that a recording keeps the tables it reads: after `clear_pool` and fresh uploads written over the freed memory, replays still give the logits of eager steps (`a_recording_keeps_the_tables_it_reads`, which crashes the card without the arena's tables).

## 7.9 Quantized products

Products by quantized weights read the codes on the card (`quantized.rs`). For at most 8 rows of activations, the decoding case, one warp computes one weight row: each lane decodes eight weights at a time, takes the scale of each block out of the sum (`a · Σ q·x + b · Σ x`), and the lanes' sums are added by a fixed butterfly. The generic kernel `qgemv` decodes any format through its code; int8, int4, Q8_0, Q4_0, Q4_1, Q4_K and Q6_K have kernels of their own (`qgemv_int8`, `qgemv_q4_0`, `qgemv_q4k`, …), with aligned loads and activations read in 128-bit loads, and a `_1` variant for one row. Several weights that read the same input, stacked by the model, are computed in one call, each into its columns.

For more rows, the prompt, pieces of at most 32 MiB of weights are dequantized into the activations' type by `dequantize` and multiplied by the tiled product. Rows of quantized embedding tables are dequantized on the card by `dequantize_rows`. Q2_K and Q3_K have no kernels and go through the host. IQ4_NL, IQ4_XS and IQ3_S, the non-linear formats of llama.cpp, go through the generic kernels: their eight values share a scale, and `q` comes from a table, the 512 groups of IQ3_S's grid a global array of the PTX.

The backward pass, `g · W` for the gradient of the activations (QLoRA, section 16.2), dequantizes the weights stacked, all at once, into the gradient's type, and computes one tiled product; it went through the host until then. A `Cast` from a quantized type decodes it by the same `dequantize`.

## 7.10 Performance

Measured on an RTX 3090 with Llama 3.2 3B in bf16, 128 tokens of prompt and 256 generated, against vLLM 0.32 on the same random prompts (`tools/parity/README.md`):

| Measure | Kapel | Reference |
| --- | --- | --- |
| 64 sequences | 3328 tokens/s | vLLM 3710 (90 %) |
| one sequence | 105 tokens/s (110 in decoding) | vLLM 108.5 |
| bandwidth, one sequence | 720 GB/s | 81 % of the 885 GB/s a copy reaches |
| prompt of 4096 tokens | 0.76 s | |
| f8 E4M3 cache, 64 sequences | 3331 tokens/s | same as bf16 |

Served by `kapel serve` (chapter 12) with requests of 16 to 128 prompt tokens and 32 to 384 generated, the same 3B model sustains about 20 requests and 4300 generated tokens per second, 256 sequences at a time.

Speculative decoding (chapter 11) takes Qwen3 8B in bf16 from 49.9 to 85.6 tokens per second with Qwen3 0.6B proposing four tokens per pass, the greedy answers unchanged, and MiMo-7B-RL from 54.7 to 83.8 with its own multi-token layer.

Llama 3.2 1B decodes at 425 tokens per second in Q4_K_M, 412 in Q8_0 and 272 in bf16; Llama 3.2 3B at 166 in int8, 179 in int4 and 204 in Q4_0. Q4_0, the format `kapel quantize` writes by default (chapter 13), went through the generic kernel at 28 tokens per second until the tracked measures (chapter 19) showed it.

Qwen3.6-35B-A3B in UD-IQ4\_XS (Unsloth's GGUF: experts in IQ3\_S and IQ4\_XS, the rest in Q8\_0), against llama.cpp on the same file and card (`llama-bench -p 512 -n 128`, `llama-perplexity -c 512 --chunks 8`):

| Measure | Kapel | llama.cpp |
| --- | --- | --- |
| decoding | 93 tokens/s | 161 |
| prompt of 512 tokens | 398 tokens/s | 4493 |
| wikitext-2 perplexity, 8 chunks | 6.30 | 6.30 ± 0.34 |

Decoding started at 21.6 tokens per second; Nsight Systems showed where it went, and the kernels above were written from it (`top_k`, `delta_rule`, `gemv_float`, the paths of `expert_gemv`, the router's softmax over the chosen logits only). What is left: the products by experts reach 200 GB/s, reading their weights byte by byte; the dense products in Q8\_0 64 % of the bandwidth; and some 3 ms a token go to the small kernels around Gated DeltaNet. A prompt chooses its experts on the host and runs the delta rule row by row.

The limits are known. At one sequence, products alone reach 92 % of the bandwidth, but the small kernels between them (norms, RoPE, cache writes, attention) take the rest of the time. Quantized weights with 9 to a few hundred rows of activations are dequantized at every pass. 4-bit formats are limited by the instructions that decode them, not by bandwidth, and `qgemv` with 2 to 8 rows is slower than it should be.

## 7.11 Testing on the card

The GPU tests run the conformity suite in `f32`, `bf16` and `f16`, plus targeted tests: every one of the 65,536 bf16 and f16 bit patterns converted exactly as the host converts them; attention over many tiles with windows, masks, softcaps and grouped heads, run twice to check identical bits; extreme masks; norms with bf16 weights; one backend per device; refused 32-bit overflows. The conformity suite also covers paged attention, row writes and `ArgMax`, with caches in every float type. These tests call `Cuda::poison_buffers(true)`, which fills every new buffer with NaN, so that an output left partly unwritten fails instead of passing by luck. A parity test runs whole models on the card and compares their logits with `transformers`: within 2e-5 in `f32`, and in `bf16` within twice the CPU's own `bf16` error. Others check that a batch of sequences gives the logits of each one alone, that fused projections change nothing, and that quantized models predict on the card as on the processor.

Three examples measure the products of decoding: `examples/decode.rs` for bf16, against a plain copy, `examples/quantized.rs` for each quantized format, and `examples/experts.rs` for the products by experts at the shapes of Qwen3.6-35B-A3B and Qwen3.8-Flash-Next, all reporting the bandwidth reached on the weights.
