"""A GGUF test file with one tensor per quantization format, and the values
`gguf-py` (llama.cpp's reference) dequantizes them to.

Writes data/parity/gguf/quants.gguf and data/parity/gguf/expected.safetensors.
Usage: python tools/parity/gguf_reference.py
"""

from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T, GGUFWriter, GGML_QUANT_SIZES
from gguf.quants import dequantize
from safetensors.numpy import save_file

# The byte ranges of the f16 scales in each block, to keep them finite.
SCALES = {
    T.Q8_0: [0], T.Q4_0: [0], T.Q5_0: [0], T.Q4_1: [0, 2], T.Q5_1: [0, 2],
    T.Q2_K: [80, 82], T.Q3_K: [108], T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208],
}


def main():
    out = Path(__file__).resolve().parents[2] / "data" / "parity" / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    writer = GGUFWriter(str(out / "quants.gguf"), "llama")
    writer.add_name("kapel quantization test")
    writer.add_uint32("test.answer", 42)
    writer.add_array("test.list", [1, 2, 3])
    expected = {}
    rows = 3
    for qtype, offsets in SCALES.items():
        block, size = GGML_QUANT_SIZES[qtype]
        cols = 2 * 256
        blocks = rng.integers(0, 256, size=(rows * cols // block, size), dtype=np.uint8)
        for at in offsets:
            scale = rng.uniform(0.001, 0.05, size=blocks.shape[0]).astype(np.float16)
            blocks[:, at:at + 2] = scale.view(np.uint8).reshape(-1, 2)
        raw = blocks.reshape(rows, cols // block * size)
        writer.add_tensor(qtype.name.lower(), raw, raw_dtype=qtype)
        expected[qtype.name.lower()] = dequantize(raw, qtype).astype(np.float32).reshape(rows, cols)
    plain = rng.standard_normal((2, 3, 4)).astype(np.float32)
    writer.add_tensor("f32", plain)
    expected["f32"] = plain
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    save_file(expected, str(out / "expected.safetensors"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
