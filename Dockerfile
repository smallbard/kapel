# Kapel's server, for NVIDIA cards.
#
#   docker build -t kapel .
#   docker run --gpus all -p 8000:8000 -v ~/.cache/kapel:/cache kapel \
#       meta-llama/Llama-3.2-3B-Instruct --device cuda
#
# The arguments are those of `kapel serve`; the server listens on 0.0.0.0:8000.
# Models from the Hub are downloaded into /cache (HF_TOKEN for gated ones).
# The host needs the NVIDIA driver (CUDA 13.2 or later) and the NVIDIA
# Container Toolkit.
#
# `--build-arg FEATURES=` builds for the processor only, on any machine.

ARG CUDA=13.3.1

FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG FEATURES=cuda
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
        clang libclang-dev pkg-config git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p kapel-cli --features "$FEATURES" \
    && cp target/release/kapel /usr/local/bin/kapel

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 kapel \
    && mkdir /cache && chown kapel /cache
# The tile kernels are compiled at their first launch by `tileiras`.
COPY --from=build /usr/local/cuda/bin/tileiras /usr/local/cuda/bin/tileiras
COPY --from=build /usr/local/bin/kapel /usr/local/bin/kapel
ENV CUDA_HOME=/usr/local/cuda KAPEL_CACHE=/cache
USER kapel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["kapel", "serve", "--host", "0.0.0.0", "--port", "8000"]
