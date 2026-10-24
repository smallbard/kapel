# 5. The reference and the conformity suite: kapel-ref

The reference backend defines what correct means in Kapel. It computes every operation in `f64`, as a direct transcription of its definition, "slow on purpose". Its conformity suite then runs any other backend on the same inputs, forward and backward, and reports every value that strays further than a written tolerance.

## 5.1 A backend written to be read

`Reference` is a unit struct. Inputs are read at their exact value with `to_f64_vec`, which materializes strided, broadcast and quantized tensors into a dense, row-major `Array`:

```rust
pub(crate) struct Array { shape: Vec<usize>, data: Vec<f64> }
```

Outputs are `f64` tensors, except for `Cast` (its target type), the copies `Contiguous` and `Concat` (their inputs' type) and `ArgMax` (`u32` positions). `Transfer` is the identity on the host and `WrongDevice` elsewhere. `Write` writes into host storage, each value encoded in the table's type by `DType::encode`, and `Record` records nothing: `records()` is false, and a pass is simply run.

Each operation is a few lines. A matrix product is a triple loop; its backward is `∂a = g · bᵀ` and `∂b = aᵀ · g`, followed by `sum_to`, which sums the gradient of a broadcast input over its repeated dimensions. A unary operation is a table of pairs `(f, f')`, and its backward multiplies the gradient by `f'(x)`. Softmax subtracts the row maximum first and its backward is `y ⊙ (g − Σ g·y)`. RoPE's backward is the same rotation run in the other direction. Gather and scatter are each other's backward.

Attention shows how the reference handles edge cases explicitly. `Inputs::weights` keeps only the visible keys, then drops any score equal to −∞ before the softmax, which "keeps a row whose keys are all masked from computing −∞ − (−∞)": such a row gives zeros, as the catalogue promises.

Paged attention reads the page table and the rows at their values, calls `check_indices`, then follows the definition line by line: for each query and head, the scores of its visible positions, located through the page table, their softmax, and the weighted sum of the values.

Every `forward` starts with `op.infer`, and every `backward` with `op.infer_backward`, so the reference rejects bad calls exactly as the fast backends do.

## 5.2 How the suite works

```rust
pub fn check<B: Backend + Transfer>(backend: &B, dtype: DType) -> Vec<Failure>;
pub fn assert_conforms<B: Backend + Transfer>(backend: &B, dtype: DType);   // panics with the list
```

A backend's tests call `assert_conforms` once per type it supports: the CPU backend in `f32` and `bf16`, and in `f32` for each instruction set; the CUDA backend in `f32`, `bf16` and `f16`.

**The cases.** `cases()` draws inputs from a SplitMix64 generator seeded with 42, so the cases are the same on every run and every machine. Values are drawn in `f64` and then encoded to the type under test, so the reference and the backend see the same rounded inputs: the suite measures the backend's arithmetic, not the rounding of its inputs. Ranges are chosen per operation: \[−1, 1) by default, \[0.5, 2) for logarithms, roots and division, wider ranges for log-softmax and saturating casts. Quantized products use valid scales followed by random codes.

Beyond the main list, three groups target the places where backends break:

- **Empty tensors**: a dimension of size 0 in products, reductions, norms, attention, RoPE and gathers. A product with K = 0 must give zeros.
- **Strided inputs**: transposed, narrowed and stride-0 broadcast views, including attention on `[B, T, H, D]` tensors seen as `[B, H, T, D]`.
- **Masked attention**: one query whose keys are all masked with −∞ (zeros and no gradient), and the same row masked with `f32::MIN` (uniform weights).
- **Paged attention**: one sequence decoding; two sequences sharing a cache with their pages interleaved, one of them computing several queries; a window with capped scores and a set scale; queries narrowed out of a wider projection; a cache of another float type than the queries, including `f8e4m3` and `f8e5m2`; a query that sees nothing; heads of 64 and 128 over 300 positions, long enough for a backend to split them; and a prompt of 400 queries after 10 cached positions, next to a query of another sequence, with and without a window.
- **`ArgMax` and `Glu`**: rows of rounded values, so that ties occur, and rows mixing NaN and −∞; SiLU and GELU gates, on a transposed input too.
- **Products**: the first factor transposed (the gradient of a weight), both transposed, and a bias added, with sizes no tile divides.
- **Whole pieces of eight values**, which a backend may compute eight at a time: a bias added to rows, sums and products of one shape, a function, conversions.
- **`CrossEntropy`**: targets of both integer types with an ignored row each, rows of 100 logits (wider than a warp and not a multiple of it), transposed logits, logits in \[−30, 30), and no rows at all.

Each input is an argument plus a view (whole, transposed, narrowed or broadcast), and the view is applied after upload, so a GPU backend really receives a strided device tensor.

**Rows written in place.** `Write` is not an operation, so it has its own check: three rows written into tables of the type under test, of `f16` and of `f8e4m3`, compared bit for bit with the reference, and four calls that must be refused: a row out of range, a row written twice, a shared table and rows of another width. The values of an input that `infer` does not read are checked the same way: a cross-entropy target beyond the vocabulary, and one of −2, must be refused. AdamW in place (`Update::adamw`), last, is compared with the reference on a parameter, its moments and a transposed gradient, and must refuse a shared parameter and a gradient of another shape.

**One check.** For every case, `Conformity::run` does five things:

1. Run the reference on host views and the backend on uploaded views, then download the result.
2. Check that the output's type is the one `infer` announced.
3. Compare the values.
4. If the output is a float, send the same random gradient through both backward passes; the pattern of `Some` and `None` must match, and so must the values.
5. Send a gradient of the wrong shape: the backend must return an error. A panic is caught and reported as a failure.

## 5.3 Tolerances

A value passes when its error is small relative to the largest reference value of the case:

```math
|\text{got} - \text{expected}| \le \text{tol}(\text{dtype}) \times \max\left(1, \max |\text{reference}|\right)
```

| Type | Tolerance | Why |
| --- | --- | --- |
| `f64` | 1e-10 | rounding of long sums only |
| `f32` | 1e-4 | 24-bit significand, sums of a few thousand terms |
| `f16` | 1e-2 | 11-bit significand |
| `bf16` | 5e-2 | 8-bit significand |
| other | 0.25, or exact for integer outputs |  |

NaN matches NaN. Scaling by the largest reference value means a backend is judged on the magnitude of what it computes, not on cancellations it cannot avoid. An output narrower than the type under test, such as the f32 rows that `GatherRows` gives from a quantized table in the f64 suite, is compared at the tolerance of its own type.

## 5.4 Who checks the reference?

The reference's gradients are checked by finite differences. With a fixed seed, the loss `Σ weights · output` is perturbed element by element with a central difference of step `h = 1e-6 · max(1, |x|)`, and each analytic gradient must agree within `1e-5 · (1 + |numeric|)`. More than 1,500 elements are checked. `Cast` is skipped, because its gradient is a convention rather than a derivative, and so is the `f32::MIN` attention case, whose effect rounds away.

The suite also tests itself. A wrapper that deliberately multiplies `exp` by 1 + 1e-3 must produce exactly one failure, on the unary `Exp` case: proof that the tolerances are tight enough to catch a 0.1 % error. Another wrapper casts the reference's outputs to the announced type and must conform in `f64`, `f32` and `bf16`.

Finally, `golden.rs` checks the GGUF dequantization formats (Q4\_1, Q5\_0, Q5\_1, Q2\_K to Q6\_K) against an independent transcription of llama.cpp's `dequantize_row_*` loops, plus a few values worked out by hand. The two derivations walk the data differently, so an error would have to be made twice in the same way to go unnoticed.
