# Troubleshooting

**`access denied (a gated model needs HF_TOKEN)`.** The model's owner asks users to accept a license: open the model's page on the Hub, accept it, create a token (Settings, Access Tokens, read access) and `export HF_TOKEN=hf_...`.

**`not found`.** Check the repository's name (`owner/name`, case matters) and `--revision`.

**No network.** A model downloaded once runs offline: Kapel uses its copy when the Hub cannot be reached. `KAPEL_OFFLINE=1` skips the Hub without waiting for its timeout.

**`unsupported architecture "..."`.** Kapel reads the architectures listed in the [Reference](reference.md#models). Mixtures of experts and models that read images are not among them.

**`this kapel was built without the feature cuda`.** Rebuild with `--features cuda` ([Installing Kapel](installation.md#3-optional-nvidia-cards)).

**Building with CUDA fails on bindings or `libclang`.** Install `clang` and `libclang-dev` (or your system's equivalent), and set `CUDA_HOME` to the toolkit (13.2 or later).

**Slow on the processor.** Generating reads every weight for each token: quantize (`--quantize int8`, or a Q4 GGUF file), and use a release build (`cargo install`, or `cargo run --release`). `--stats` shows the speeds. A debug build is tens of times slower.

**Out of memory on the card.** Lower `--gpu-memory-utilization` if other programs use the card, cap `--max-model-len`, use `--kv-cache f8e4m3` to halve the cache, or quantize the weights.

**The server answers 429.** More requests are in flight than `--max-requests`; raise it, or slow the clients.

**The server answers 400 about a field.** Kapel refuses the fields it cannot honour (images, tools, log probabilities) instead of ignoring them; the message names the field.

**Qwen 3 answers start with `<think>`.** The model reasons before answering; add `/no_think` to the message, or use an instruct model that does not think.

**The answer differs between runs.** Sampling draws at random: `--seed` fixes the draws, `--greedy` (or `temperature` 0) takes the most likely tokens. The same command with the same seed gives the same answer, whatever the number of threads.
