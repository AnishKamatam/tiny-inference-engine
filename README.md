# tie

**tie** (tiny inference engine) is an LLM inference engine for Apple Silicon,
written in C++ and Metal. It targets agent and tool-use workloads, where many
concurrent requests share long common prefixes, and is built toward running
models too large for one machine across several Macs on a local network.

## Features

- **Metal and CPU backends.** Metal is the default. The CPU backend matches
  HuggingFace Transformers token for token and serves as the reference the
  Metal kernels are tested against.
- **Paged KV cache.** KV memory is a single pool of fixed-size blocks, and each
  sequence holds a block table. Memory is allocated per block as a sequence
  grows, not reserved up front for the maximum length.
- **Paged attention.** The Metal attention kernel reads K and V directly
  through block tables, so sequences of different lengths share one batch
  without copying or padding.
- **Model formats.** GGUF (F32, F16, BF16, Q8_0) and HuggingFace safetensors.
- **Tokenizer.** Byte-level BPE with the Qwen2 pre-tokenizer and chat template,
  implemented natively with no Python dependency.

Supported models: Qwen3 (tested with Qwen3-0.6B).

## Roadmap

| Stage | Status |
|---|---|
| Base runtime: model loading, Metal/CPU backends, sampling, CLI | Done |
| Paged KV cache and paged attention | Done |
| Continuous batching: chunked prefill mixed with decode, SRPT ordering with aging, KV-aware admission and preemption | In progress |
| Radix prefix cache: share KV blocks between requests with a common prefix | Planned |
| OpenAI-compatible HTTP server | Planned |
| Distributed inference: pipeline-parallel execution across multiple Macs | Planned |

The long-term goal is to let a group of MacBooks serve one large model
together: each machine holds a contiguous range of layers and passes
activations to the next over the network. The scheduler and engine are kept
free of single-device assumptions so that pipeline stages can be added without
restructuring them.

## Requirements

- Apple Silicon Mac
- CMake 3.24 or newer
- Xcode Command Line Tools

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

## Usage

```sh
./build/tie generate -m models/Qwen3-0.6B-Q8_0.gguf --chat --no-think \
  -p "Name three primary colors."
```

`-m` accepts either a GGUF file or a HuggingFace model directory. Common
options:

| Option | Description |
|---|---|
| `--device metal\|cpu` | Compute device (default `metal`) |
| `--max-tokens N` | Tokens to generate (default 256) |
| `--temp T` | Sampling temperature; 0 is greedy (default 0.7) |
| `--top-k K`, `--top-p P` | Top-k and nucleus sampling (defaults 20, 0.8) |
| `--kv-mem SIZE` | KV cache budget, e.g. `4G` or `512M` (default 4G) |

Run `./build/tie --help` for the full list.

## Testing

```sh
tools/setup_test_data.sh   # one-time: downloads models and dumps HuggingFace reference outputs
./build/tie_tests
```

`setup_test_data.sh` requires `hf` and `uv`. Tests that depend on downloaded
models are skipped with a message if `models/` is missing.

## Project layout

| Directory | Contents |
|---|---|
| `src/core` | Tensors, dtypes, errors, thread pool |
| `src/model` | GGUF and safetensors loaders, Qwen3 model |
| `src/tokenizer` | Byte-level BPE, Qwen2 pre-tokenizer, chat template |
| `src/backend` | `Backend` interface, Metal backend, CPU reference backend |
| `src/kernels` | Metal shaders and CPU kernels |
| `src/kv` | Block pool and paged KV cache |
| `src/engine` | Sequences, schedulers, serving config, step loop |
| `src/sampling` | Temperature, top-k and top-p sampling |
| `tests` | Unit and model-level tests, mirroring `src` |
| `tools` | Test data setup and code generation scripts |

## Acknowledgements

A small number of Metal compute kernels are adapted from
[llama.cpp](https://github.com/ggml-org/llama.cpp) and
[MLX](https://github.com/ml-explore/mlx). Everything else is original. See
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) for details.
