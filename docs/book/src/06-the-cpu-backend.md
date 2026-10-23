# 6. The CPU backend: kapel-cpu

`kapel-cpu` runs the whole catalogue on x86 processors with AVX-512 or AVX2, and on any other target with portable code that the compiler vectorizes. It computes in `f32`; inputs may be `f32`, `bf16` or `f16` (and a cache of keys and values `f8` too), and outputs take the input type. `f64` is left to the reference. Most of the speed comes from four pieces: a deterministic thread pool, a blocked matrix product in the style of BLIS, vectorized row functions, and products by quantized weights that read the codes in place. Most of the safety comes from one small type, `Isa`.

## 6.1 The backend value

```rust
pub struct Cpu {
    pub(crate) pool: Arc<Pool>,          // permanent worker threads
    pub(crate) isa: Isa,                 // the instructions it may use
    pub(crate) buffers: Arc<BufferPool>, // up to 1 GiB of reusable host buffers
    pub(crate) byte_activations: bool,   // quantized products with activations in bytes
    pub(crate) vnni: bool,               // proof that AVX-512 VNNI can run
}
```

`Cpu::new()` uses one thread per physical core, counted from `/proc/cpuinfo` when possible, because simultaneous multithreading brings nothing to matrix products. The environment variable `KAPEL_THREADS` overrides it, clamped to four times the number of logical cores. `with_isa(kind)` restricts the instruction set, which the tests use to check every path on one machine. `with_byte_activations(true)` changes how quantized products compute (section 6.7). Clones share threads and buffers.

## 6.2 Instruction sets as a proof

Calling an AVX-512 function on a processor without AVX-512 is undefined behaviour. Kapel makes it impossible from safe code with a proof token:

```rust
pub struct Isa(IsaKind);   // private field: safe code cannot forge one
// Isa::detect(), Isa::new(kind) -> Option<Isa> and Isa::SCALAR check the processor first
```

Holding an `Isa` proves its instructions can run, so the `unsafe` calls behind it need no other argument. Row functions are written once and compiled three times by the `multiversion!` macro: a body generic over `const FMA: bool`, wrapped in an AVX-512 function, an AVX2 function and a plain one, dispatched by `isa.kind()`.

One consequence is worth knowing. The vector versions use fused multiply-add and the scalar one does not, so results may differ in the last bits between instruction sets. Kapel promises the same bits for any number of threads, not for any processor.

## 6.3 A thread pool that cannot change the result

Spawning a thread costs tens of microseconds, as much as a small operation of a model, so the pool's threads are permanent. `pool.run(tasks, work)` runs `work(0)` to `work(tasks − 1)` and returns when all are done. Workers take task indices from a shared counter, and the calling thread works alongside them.

Determinism comes from one discipline: **the number of tasks is chosen from the shape of the tensor, never from the number of threads**, and each task computes the same thing whichever thread runs it. A sum split in 32 tasks is split in 32 tasks on a laptop and on a server, and the partial results are combined in task order.

The other details serve speed and robustness:

- Idle workers spin for about 50 µs before sleeping on a condition variable, so back-to-back operations do not pay for a wake-up.
- The work is published without a lock: `run` writes a pointer to the call, kept on its stack, then increments an epoch; the workers read the pointer once they see the epoch change, and `run` returns only after all of them have finished. The epoch, the task counter and the count of busy workers live on separate cache lines. A call costs 0.6 to 0.8 µs, against 2.5 µs when the call went through a mutex: a model's token makes a few hundred calls.
- A task that panics does not stop the others; the first panic is resumed in the caller once every task has finished, and the pool remains usable.
- A `run` called from inside a task executes serially in place, which avoids deadlocks. Attention relies on this: its tasks call the matrix product, which then runs on the task's thread.
- `SharedMut<T>` lets tasks write disjoint parts of one output slice. It is the pool's only shared-mutation primitive, and its methods are `unsafe` with the contract that slices never overlap.

The workhorse is `Cpu::fill`, which splits an output into chunks of 16,384 elements, one task each. Most element-wise operations are a few lines on top of it.

## 6.4 The matrix product

The product follows Goto's algorithm, as in BLIS: the operands are cut into blocks sized for each cache level, packed into contiguous panels, and multiplied by a small register-resident micro-kernel.

```
for each slice of nc columns of B              B[pc, jc] packed, in L3
  for each slice of kc shared indices
    for each block of mc rows of A               A[ic, pc] packed, in L2
      for each panel of NR columns of B          one panel, in L1
        for each panel of MR rows of A
          micro-kernel: C[MR × NR] += A[MR × kc] · B[kc × NR]
```

| Block | Zen 4 value | Sized for |
| --- | --- | --- |
| `kc` | 512 | a `kc × NR` panel of B in half of the 32 KiB L1 |
| `mc` | 392 | an `mc × kc` block of A in half of the 1 MiB L2 |
| `nc` | 4096 | a `kc × nc` slice of B in a share of L3 |

Packing absorbs transposition and strides: the micro-kernel only ever sees contiguous panels, padded with zeros at the edges. `Matrix` describes an operand by its data, rows, columns and two strides; a stride of 0 is a broadcast, and `t()` transposes by swapping strides. Its fields are private and its constructor checks that the last element lies inside the data, which is what makes the raw-pointer kernels sound.

| Micro-kernel | Tile MR × NR | Registers |
| --- | --- | --- |
| AVX-512 | 14 × 32 | 28 accumulators, 2 for B, 1 broadcast of A (of 32) |
| AVX2 | 6 × 16 | 12 accumulators, 2 for B, 1 broadcast (of 16) |
| Portable | 4 × 8 | left to the compiler |

The C block stays in registers for a whole `kc` slice and is written once at the end. On Zen 4, which executes 512-bit instructions in two halves, the AVX-512 kernel is not faster per cycle than the AVX2 one, but it loads half as much of A and B per multiply-add and decodes half as many instructions.

**Why the product is deterministic.** Here the split does depend on the thread count, but it does not matter: every element of C is summed `kc` slice by `kc` slice, in the same order, whatever the partition into blocks. `kc` is "the only size that changes the result".

**Few rows.** Generation multiplies one row, or a few, by a large weight matrix. There is nothing to reuse in the caches, and packing B would cost more than the product. Below 256 rows (16 when B is transposed), a separate path reads B directly with masked loads, splits the shared dimension into slices when there are too few columns to occupy the threads, and adds the slices in a fixed order. Its 32 tasks depend only on the shape.

**bf16.** On processors with AVX-512 BF16, `gemm_mixed` packs pairs of bf16 values and uses `vdpbf16ps`, which does twice the work of an f32 multiply-add per instruction. It is used for large products only; no operation of the catalogue calls it yet.

## 6.5 Rows, reductions and the exponential

`math.rs` holds the row functions: `dot`, `sum`, `max`, `scale`, `axpy`, `exp` and the activation functions. Each works in packets of 16 lanes and combines its partial sums pairwise in a fixed order, so the same row gives the same bits. The exponential is Cephes's algorithm: reduce x to n · ln 2 + r with a two-part ln 2, evaluate a degree-7 polynomial, and build 2ⁿ directly in the exponent bits. Tests hold it within two ulps over \[−87, 87\]. The bits of 2ⁿ come from adding 1.5 · 2²³ to n, which puts n in the low bits of the mantissa: a conversion `n as i32` gives the same bits, but Rust's conversion saturates, and that kept the compiler from vectorizing the loop. The trick made the exponential, and SiLU and softmax with it, 3.7 times faster.

`Glu` computes `f(gate) · up` in one pass over each row, the gate the first half of the row, and `ArgMax` scans each line for its first maximum, skipping NaN.

Binary operations on two inputs of the output's shape, the residual sums and gated products of a model, take a direct loop; the others compute the broadcast offsets of each input.

Sums and means are cut into blocks of 1,024 elements, summed with `math::sum`, then combined pairwise, so the error grows with the logarithm of the length. Blocks of all outputs are spread over the threads, so one huge sum uses every core: 16 million elements went from 64 ms to 0.8 ms when this was introduced, and became far more accurate.

Softmax subtracts the row maximum, computes `exp(x − max)` and its sum in one pass, then scales. Norms process rows in tasks; their backward accumulates the weight and bias gradients in per-task buffers, then adds them in task order.

## 6.6 Attention

The CPU attention is not FlashAttention. For a block of 128 query rows, it computes the full score matrix with the product, applies the scale, softcap, mask and visibility rules, then a row softmax, and multiplies by V. A task owns a batch, a key head and a query block, and loops over the query heads that share that key head, so keys and values stay in cache.

The backward computes `∂v += Pᵀg`, `∂P = g·vᵀ`, `∂S = P ⊙ (∂P − rowsum(∂P ⊙ P))`, then `∂q` and `∂k` from ∂S. Query blocks write partial `∂k` and `∂v` buffers that are added in block order. A row with no visible key gives zeros, but a row with a NaN score gives NaN: a NaN must never be hidden.

**Over pages.** `PagedAttention` reads a cache that may be large and mostly unused by the step. A task is one sequence, one key head and a block of up to 32 of that sequence's queries: it gathers the lines the block sees through the page table, converts them to `f32` once, then computes each query's scores with `math::dot`, their softmax and the weighted sum of the values with `math::axpy`. Each output row belongs to one task and adds its terms in position order, so the result does not depend on the threads. The cache may be in `f32`, `bf16`, `f16`, `f8e4m3` or `f8e5m2`; an 8-bit value is decoded through a table of its 256 byte values, built once.

**Writing the cache.** `write_rows` writes into the host storage of the table, its only handle. Rows of the table's own type are copied as bytes, row by row; others are converted, `f32` and `bf16` directly and the other types, `f8` included, by `DType::encode`. `Record` records nothing on the processor: each operation is computed when it is called, so a pass is simply run.

## 6.7 Quantized products

In generation, each weight of a model is read once per token, and the product does two operations per weight: the time of a token is the time to read the weights. A quantized format is fewer bytes to read, provided the product reads the codes as they are stored, and decodes them faster than memory brings them. `quantized.rs` does this for every format Kapel reads: its own int8 and int4 by groups, and GGUF's Q4\_0 to Q8\_0 and Q2\_K to Q6\_K, the bytes of the mapped file read in place.

**In registers.** Each thread takes a run of rows of `w [N, K]`, about 128 KiB of weights, taken in order so that the threads read neighbouring memory, and computes their dot products with up to eight rows of x. A block of weights is decoded sixteen codes at a time: the nibbles or bit planes are taken from the bytes in 128-bit registers, then widened to sixteen integers and converted to floats. Zen 4 runs a 512-bit instruction as two halves, so every step done on 128 bits instead is a step at a quarter of the cost. The scales and minimums are taken out of the sums: within a group that shares a scale d and a minimum m,

```math
\sum_i (d\, q_i - m)\, x_i = d \sum_i q_i x_i - m \sum_i x_i
```

so the codes are multiplied as they are, d once per group, and the sums of x over each group are computed once per row of x, for all the rows of w. The compiler does not find these kernels by itself; they are written with AVX-512 intrinsics, and the portable code, vectorized by the compiler, covers AVX2 and the rest. On a Ryzen 7 7700, every format reads its weights at 52 to 54 GB/s, the speed of the memory, for one row of x.

**Activations in bytes.** With `with_byte_activations(true)`, each row of x is quantized too, by blocks of 32: a scale `max |x| / 127` and integers between −127 and 127. A group of weights then gives d · s · Σ c · q − m · s · Σ q, where Σ c · q is a sum of products of bytes, exact in 32-bit integers, that AVX-512 VNNI's `vpdpbusd` computes 32 at a time; signed codes move their signs to the activations, since the instruction multiplies unsigned bytes by signed ones. This is what llama.cpp does, with blocks of 256. It is several times fewer instructions per weight, which leaves room when another program takes a core, but it adds the rounding of the activations to that of the weights: the mode is chosen explicitly. On Llama 3.2 1B in Q4\_K\_M, it moves the perplexity by 0.03 %, and generation reaches 57.4 tokens per second against 62.6 for llama.cpp; 56.3 with activations in f32.

**Many rows.** For the prefill, decoding each block once per row would cost more than dequantizing. Tiles of 2 MiB of the weights are dequantized in parallel, then multiplied by the blocked product, so no matrix is ever expanded whole in memory: a 570-token prompt of Llama 3.2 1B runs at 313 tokens per second, against 45 when whole matrices were expanded, and 774 for llama.cpp, whose products multiply bytes.

The weights of several matrices may be given to one product, their rows stacked (the `QuantizedMatmul` of several weights): the tasks then cover them all, in proportion to their bytes. And a gather from a quantized table, a `bf16` table or an `f16` one decodes only the rows it gathers.

## 6.8 Reading data

`io.rs` is where types and alignment are handled. An aligned `f32` tensor is read in place, strides included; anything else is converted to a contiguous `f32` copy. A tensor of a memory-mapped file may start at an odd address, so the code checks alignment with `align_to` before reinterpreting bytes. Outputs are written into pooled buffers, directly by the computation for `f32`, and rounded to `bf16` (nearest, ties to even, NaN kept quiet) or `f16` when needed.

RoPE computes its angles in `f64`, once per position and frequency, because positions in the hundreds of thousands times a frequency near 1 would lose their phase in `f32`.

## 6.9 Measuring

Each optimization has a benchmark program in `crates/kapel-cpu/examples/`, run with `cargo run --release -p kapel-cpu --example <name>`:

| Example | What it measures |
| --- | --- |
| `gemm` | f32 products from 1024³ to 4096³, in Gop/s |
| `gemm_few_rows` | 1, 16, 256 and 257 rows times a 4096 × 4096 weight, in GB/s of weights read |
| `gemm_bf16` | f32 against the mixed bf16 path |
| `attention` | causal attention forward and backward, and a decoding step |
| `reduce` | sums over contiguous, strided and flat data |
| `softmax` | softmax and exp on typical shapes |
| `quantized` | products by quantized weights of 0.3 to 2 GB, every format, 1, 4 and 16 rows, in GB/s of weights read (`BYTES=1`: activations in bytes) |

The examples `profile` and `perplexity` of kapel-cli measure a whole model (chapter 11).

The tests check conformity in `f32` and `bf16` and for every available instruction set, and bit-identical results with 1, 2, 3 and 8 threads for products, attention and sums.
