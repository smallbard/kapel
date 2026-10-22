# 3. Tensors: kapel-core

A Kapel tensor is three things: a shared buffer of bytes, a layout that says where each element lives in that buffer, and a data type that says how to read the bytes. Everything else in the library manipulates these three pieces. `kapel-core` has no Kapel dependency and defines them, together with devices and errors.

```rust
pub struct Tensor {
    storage: Arc<dyn Storage>, // the bytes, on some device, shared by views
    layout: Layout,            // shape, strides, offset, in elements
    dtype: DType,              // how to read the bytes
}
```

## 3.1 Data types

`DType` lists every type a tensor can hold. It is `#[non_exhaustive]`, so new formats can be added without breaking callers.

| Family | Types | Bytes per element | Notes |
| --- | --- | --- | --- |
| Wide floats | `F64`, `F32` | 8, 4 | `F64` is the type of the reference backend |
| Half floats | `F16`, `BF16` | 2 | IEEE binary16; bf16 keeps the f32 exponent range with 8 bits of precision |
| 8-bit floats | `F8E4M3`, `F8E5M2` | 1 | OCP formats; E4M3 has no infinity |
| Integers | `I8`, `U8`, `I32`, `U32` | 1, 1, 4, 4 | `U32` holds token ids and indices |
| Quantized | `Quantized(QuantFormat)` | per block | Kapel int8/int4 groups and the GGUF formats |

A quantized type stores whole blocks: a scale (and sometimes a minimum) followed by packed codes. Sizes come from two methods, `block_len` and `block_bytes`.

| Format | Elements per block | Bytes per block | Value of an element |
| --- | --- | --- | --- |
| `Int8{group}` | group | 4 + group | f32 scale × i8 code |
| `Int4{group}` (group even) | group | 4 + group/2 | f32 scale × (code − 8), low nibble first |
| `Q8_0` | 32 | 34 | f16 scale × i8 code |
| `Q4_0` | 32 | 18 | f16 scale × (code − 8) |
| `Q4_1` | 32 | 20 | f16 scale × code + f16 minimum |
| `Q5_0`, `Q5_1` | 32 | 22, 24 | as Q4, with a fifth bit per element from a u32 |
| `Q2K` to `Q6K` | 256 | 84 to 210 | super-blocks with 6-bit or 8-bit sub-block scales, as in llama.cpp |

Three functions do all the conversions. `decode(bytes) -> f64` reads one element exactly. `encode(value, out)` writes one: floats round to nearest, ties to even; integers round, then saturate, and NaN becomes 0. The free function `dequantize(dtype, bytes)` expands quantized blocks to `f64`, following llama.cpp's definitions, and `quantize(dtype, values)` goes the other way: Kapel's int8 with the scale `max |x| / 127`, int4 with the scale of least squared error among those that put the largest magnitude at either end of the 16 levels and those between, and GGUF's Q8\_0 and Q4\_0 exactly as llama.cpp's reference quantizes them. `storage_bytes(elements)` returns `BlockMisaligned` unless the count is a whole number of blocks.

## 3.2 Small floats without tables

The four small float formats share one generic description in `float.rs`: a number of exponent bits, a number of mantissa bits, and whether the format follows IEEE rules for infinities.

```rust
struct Format { exponent_bits: u32, mantissa_bits: u32, ieee: bool }
// F16 = (5, 10), BF16 = (8, 7), F8_E4M3 = (4, 3, ieee: false), F8_E5M2 = (5, 2)
```

Decoding is exact. Encoding rounds once, from `f64`, ties to even, as hardware conversions do: subnormals are counted in units of the smallest subnormal, a rounding that overflows the significand bumps the exponent, and values beyond the largest finite become infinity (or NaN for E4M3, which has no infinity). The tests check every bit pattern of the four formats in a round trip, so the code is exhaustively verified rather than sampled.

In Rust code, the formats appear as the wrappers `F16`, `BF16`, `F8E4M3` and `F8E5M2`. Together with the primitive types they implement `Element`, which ties a Rust type to its `DType` and reads and writes it in little-endian order.

## 3.3 Layouts: shape, strides, offset

A layout maps a multi-dimensional index to a position in the buffer. For an index `(i₀, …, iₙ)`, the position is `offset + Σ iₖ · strideₖ`, counted in elements. A row-major `[2, 3]` tensor has strides `[3, 1]`; its transpose is the same buffer with shape `[3, 2]` and strides `[1, 3]`.

This one formula makes most reshaping operations free: they change the layout, never the bytes.

| Method | What it changes | Fails with |
| --- | --- | --- |
| `transpose(a, b)`, `permute(order)` | swaps shape and strides | `InvalidDim`, `DuplicateDim` |
| `narrow(dim, start, len)` | shortens one dimension, moves the offset | `OutOfRange` |
| `broadcast_to(shape)` | adds dimensions with stride 0, so one element is read many times | `BroadcastIncompatible`, `Overflow` |
| `reshape(shape)` | new shape on a contiguous layout | `ElementCountMismatch`, `NotContiguous` |

A stride of 0 deserves a moment. Broadcasting `[3, 1]` to `[2, 3, 4]` gives strides `[0, s, 0]`: the twelve elements of each row all read the same three values, and no copy is made. Backends must therefore handle strided and broadcast inputs, and the conformity suite feeds them such views on purpose (chapter 5).

`is_contiguous` ignores dimensions of size 1 and accepts any offset. `offsets()` walks the positions of a layout in row-major order, like an odometer. `Layout::contiguous` and `broadcast_to` use checked multiplication, so a shape whose element count does not fit in `usize` is an `Overflow` error rather than a silent wrap.

## 3.4 Storage and the buffer pool

`Storage` is a trait object, which is how a tensor can live on any device without `kapel-core` knowing about CUDA.

```rust
pub trait Storage: Send + Sync + Debug + 'static {
    fn device(&self) -> Device;
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn Any;            // a backend downcasts to its own type
    fn host_bytes(&self) -> Option<&[u8]> { None }
}
```

Host memory is allocated in `Line`s, 64-byte aligned blocks of 64 bytes: one cache line, and one AVX-512 register. `HostStorage` owns a boxed slice of lines. A memory-mapped file is another storage, so weights can be read in place (chapter 8).

Allocating fresh memory for every intermediate tensor would make the operating system hand out zeroed pages one at a time. `BufferPool` avoids that: it keeps freed buffers by size, up to a byte capacity, and gives them back to the next request of exactly the same size. A `HostStorage` returns its buffer to the pool when dropped. Buffers taken with `take` have unspecified contents; `zeroed` clears them.

## 3.5 The tensor and its views

Cloning a tensor or taking a view clones the `Arc`, never the bytes. Constructors check what the type system cannot: `from_storage` verifies that the layout fits in the storage, and, for quantized types, that the layout is contiguous and starts on a block boundary.

- `from_slice`, `from_bytes` and `zeros` create host tensors.
- `transpose`, `permute`, `narrow` and `broadcast_to` create views. They refuse quantized tensors, whose blocks cannot be split; `reshape` is allowed because a contiguous reshape keeps blocks whole.
- `to_vec::<T>()` reads the elements in row-major order. It fails with `DTypeMismatch` if `T` is not the tensor's type, and `WrongDevice` if the bytes are not on the host.
- `to_f64_vec()` is the reference view of any tensor: every type, quantized or not, read at its exact value.
- `contiguous()` copies a strided host view into a fresh row-major buffer.

Moving a tensor between host and device is not a method of `Tensor`: it belongs to the backend, through the `Transfer` trait (chapter 4).

## 3.6 Devices and errors

`Device` is `Cpu` or `Cuda(ordinal)`, shown as `cpu` or `cuda:0`.

`Error` is the single error type of the computing crates. Its guiding sentence, at the top of `error.rs`, is: "Every error names the operation that failed and says what was expected, so that the message tells the caller what to fix." Almost every variant carries the operation's name, and messages end with the fix when there is one:

- `ShapeMismatch`, `ElementCountMismatch`, `BroadcastIncompatible`: shapes that do not agree.
- `InvalidDim`, `DuplicateDim`, `OutOfRange`, `IndexOutOfRange`: indices outside their range.
- `DTypeMismatch`, `UnsupportedDType`, `BlockMisaligned`: types.
- `DeviceMismatch`, `WrongDevice`, `NotContiguous`: where and how the data lives, such as "the tensor is a strided view; call `contiguous()` first".
- `BufferSize`, `Overflow`: sizes.
- `Arity`, `InvalidArgument`, `Device`: the rest, with a free-form message.

The enum is `#[non_exhaustive]` and implements `PartialEq`, so tests can assert on the exact error a bad call returns.
