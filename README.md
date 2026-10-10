# tie — tiny inference engine

A local LLM inference engine for small models on Apple Silicon, built for
agent and tool workloads: many concurrent requests sharing long prefixes. The
KV cache is paged from day one, so continuous batching and radix (prefix)
caching are additions rather than rewrites.

Status: (base runtime). Qwen3-0.6B runs on the GPU (Metal, the
default) and on the CPU, from GGUF (F32/F16/BF16/Q8_0) or HuggingFace
safetensors. The CPU path matches HuggingFace token for token and is the
reference the Metal kernels are tested against. Next sub-project: a
continuous-batching scheduler (aging, prefill-aware ordering, chunked prefill
mixed with decode), then a radix (prefix) cache, an OpenAI-compatible server,
and distributed inference across Macs.

## Build

Requires an Apple Silicon Mac, CMake ≥ 3.24 and the Xcode Command Line Tools.

```sh
cmake -S . -B build
cmake --build build -j
```

## Run

```sh
./build/tie generate -m models/Qwen3-0.6B-Q8_0.gguf --chat --no-think \
  -p "Name three primary colors."
```

`./build/tie --help` lists every option. `-m` also accepts a HuggingFace model
directory.

## Test

```sh
tools/setup_test_data.sh   # once: downloads models into models/, dumps HF reference outputs (needs hf and uv)
./build/tie_tests
```

Tests that need `models/` skip with a message when it is missing.

## Layout

| Directory | Contents |
|---|---|
| `src/core` | tensors, dtypes, errors, thread pool |
| `src/model` | GGUF/safetensors loaders, Qwen3 model |
| `src/tokenizer` | byte-level BPE, Qwen2 pre-tokenizer, chat template |
| `src/backend`, `src/kernels` | `Backend` interface; Metal backend and shaders; CPU reference backend |
| `src/kv` | block pool and paged KV cache |
| `src/engine` | sequences, scheduler, step loop |
| `src/sampling` | temperature / top-k / top-p sampling |

A small set of Metal compute kernels is ported from
[llama.cpp](https://github.com/ggml-org/llama.cpp) and [MLX](https://github.com/ml-explore/mlx);
everything else is original. See `THIRD_PARTY_NOTICES`.
