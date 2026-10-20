#!/bin/sh
# Compiles the kernels to PTX and installs the result where the backend embeds it.
set -e
cd "$(dirname "$0")"
cargo oxide build
cp kapel_cuda_kernels.ptx ../crates/kapel-cuda/kernels.ptx
echo "kernels.ptx: $(grep -c '^.visible .entry' kapel_cuda_kernels.ptx) kernels"
