# Installing Kapel

Kapel is built from its source with Rust's toolchain. It takes a minute or two on a recent machine, and works on Linux, macOS and Windows.

## 1. Install Rust

Kapel needs Rust 1.89 or later. If `cargo --version` says nothing or an older version, install Rust with [rustup](https://rustup.rs):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh     # Linux, macOS
```

On Windows, run `rustup-init.exe` from the same site; it installs the C++ build tools it needs.

## 2. Build the `kapel` command

From a copy of Kapel's repository:

```sh
cargo install --locked --path crates/kapel-cli
```

This builds `kapel` in release mode and puts it in `~/.cargo/bin`, which rustup adds to your `PATH`. Check it:

```sh
kapel --version
```

The command runs models on the processor. On x86 processors it uses AVX-512 or AVX2 when the processor has them, found at run time; other processors (ARM, older x86) run portable code, slower.

## 3. Optional: NVIDIA cards

To compute on an NVIDIA card, build with the `cuda` feature. You need:

- an NVIDIA card of the Ampere generation or later (RTX 30xx, A100 and newer) and its driver;
- the CUDA toolkit, version 13.2 or later, with `CUDA_HOME` pointing to it (often `/usr/local/cuda`);
- `clang` and its library (`clang libclang-dev` on Debian and Ubuntu), which generate the bindings to CUDA.

```sh
cargo install --locked --path crates/kapel-cli --features cuda
```

The same binary still runs on the processor; `--device cuda` chooses the card.

## 4. Optional: the Python module

```sh
pip install ./crates/kapel-python
```

pip builds the module with maturin (it downloads it if needed) and installs `kapel` for the Python of the environment, any version from 3.9. See [Python](python.md).

## 5. Optional: the container image

For a server on an NVIDIA card without installing Rust or CUDA, build the image with Docker (and the NVIDIA Container Toolkit to run it):

```sh
docker build -t kapel .
```

See [Serving a model](serving.md#in-a-container).

## Where Kapel keeps models

Models downloaded from the Hugging Face Hub go to `~/.cache/kapel/hub`, one directory per repository and commit, downloaded once and checked against the Hub's checksums. Set `KAPEL_CACHE` to put them elsewhere. A model downloaded once also runs without network: when the Hub cannot be reached, Kapel uses the copy it has, and `KAPEL_OFFLINE=1` makes it skip the Hub altogether. Gated models, such as Meta's Llama, need a token: accept the model's license on its Hub page, create a token in your Hub settings, and export it as `HF_TOKEN`.
