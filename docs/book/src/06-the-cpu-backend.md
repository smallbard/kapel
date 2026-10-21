# 6. The CPU backend: kapel-cpu

`kapel-cpu` runs the whole catalogue on x86 processors with AVX-512 or AVX2, and on any other target with portable code that the compiler vectorizes. It computes in `f32`; inputs may be `f32`, `bf16` or `f16`, and outputs take the input type. `f64` is left to the reference. Most of the speed comes from three pieces: a deterministic thread pool, a blocked matrix product in the style of BLIS, and vectorized row functions. Most of the safety comes from one small type, `Isa`.

## 6.1 The backend value

```rust
pub struct Cpu {
    pub(crate) pool: Arc<Pool>,          // permanent worker threads
    pub(crate) isa: Isa,                 // the instructions it may use
    pub(crate) buffers: Arc<BufferPool>, // up to 1 GiB of reusable host buffers
}
```

`Cpu::new()` uses one thread per physical core, counted from `/proc/cpuinfo` when possible, because simultaneous multithreading brings nothing to matrix products. The environment variable `KAPEL_THREADS` overrides it, clamped to four times the number of logical cores. `with_isa(kind)` restricts the instruction set, which the tests use to check every path on one machine. Clones share threads and buffers.

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

`math.rs` holds the row functions: `dot`, `sum`, `max`, `scale`, `axpy`, `exp` and the activation functions. Each works in packets of 16 lanes and combines its partial sums pairwise in a fixed order, so the same row gives the same bits. The exponential is Cephes's algorithm: reduce x to n · ln 2 + r with a two-part ln 2, evaluate a degree-7 polynomial, and build 2ⁿ directly in the exponent bits. Tests hold it within two ulps over \[−87, 87\].

Sums and means are cut into blocks of 1,024 elements, summed with `math::sum`, then combined pairwise, so the error grows with the logarithm of the length. Blocks of all outputs are spread over the threads, so one huge sum uses every core: 16 million elements went from 64 ms to 0.8 ms when this was introduced, and became far more accurate.

Softmax subtracts the row maximum, computes `exp(x − max)` and its sum in one pass, then scales. Norms process rows in tasks; their backward accumulates the weight and bias gradients in per-task buffers, then adds them in task order.

## 6.6 Attention

The CPU attention is not FlashAttention. For a block of 128 query rows, it computes the full score matrix with the product, applies the scale, softcap, mask and visibility rules, then a row softmax, and multiplies by V. A task owns a batch, a key head and a query block, and loops over the query heads that share that key head, so keys and values stay in cache.

The backward computes `∂v += Pᵀg`, `∂P = g·vᵀ`, `∂S = P ⊙ (∂P − rowsum(∂P ⊙ P))`, then `∂q` and `∂k` from ∂S. Query blocks write partial `∂k` and `∂v` buffers that are added in block order. A row with no visible key gives zeros, but a row with a NaN score gives NaN: a NaN must never be hidden.

## 6.7 Reading data

`io.rs` is where types and alignment are handled. An aligned `f32` tensor is read in place, strides included; anything else is converted to a contiguous `f32` copy. A tensor of a memory-mapped file may start at an odd address, so the code checks alignment with `align_to` before reinterpreting bytes. Outputs are written into pooled buffers, and rounded to `bf16` (nearest, ties to even, NaN kept quiet) or `f16` when needed.

RoPE computes its angles in `f64`, once per position and frequency, because positions in the hundreds of thousands times a frequency near 1 would lose their phase in `f32`. Quantized products dequantize the weights to `f32` first, row by row in parallel; kernels that read the codes directly are planned with the inference work.

## 6.8 Measuring

Each optimization has a benchmark program in `crates/kapel-cpu/examples/`, run with `cargo run --release -p kapel-cpu --example <name>`:

| Example | What it measures |
| --- | --- |
| `gemm` | f32 products from 1024³ to 4096³, in Gop/s |
| `gemm_few_rows` | 1, 16, 256 and 257 rows times a 4096 × 4096 weight, in GB/s of weights read |
| `gemm_bf16` | f32 against the mixed bf16 path |
| `attention` | causal attention forward and backward, and a decoding step |
| `reduce` | sums over contiguous, strided and flat data |
| `softmax` | softmax and exp on typical shapes |

The tests check conformity in `f32` and `bf16` and for every available instruction set, and bit-identical results with 1, 2, 3 and 8 threads for products, attention and sums.
