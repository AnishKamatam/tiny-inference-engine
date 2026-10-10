#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "cli/args.h"
#include "core/error.h"
#include "engine/engine.h"
#include "kv/paged_kv_cache.h"
#include "model/loader.h"
#include "model/qwen3.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void write(std::string_view s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fflush(stdout);
}

int run_generate(const tie::GenerateOptions& opt) {
  using namespace tie;
  const auto load_start = Clock::now();
  const LoadedModel weights = load_model(opt.model);
  const ModelConfig& config = weights.config;
  std::unique_ptr<Backend> backend_owner;
  if (opt.device == "metal") {
    backend_owner = std::make_unique<MetalBackend>();
  } else {
    backend_owner = std::make_unique<CpuBackend>(opt.threads);
  }
  Backend& backend = *backend_owner;
  const Tokenizer tokenizer(weights.tokenizer);

  EngineConfig engine_config;
  for (const char* stop : {"<|im_end|>", "<|endoftext|>"}) {
    if (const auto id = tokenizer.find_token(stop)) engine_config.stop_tokens.push_back(*id);
  }

  KVCacheConfig kv;
  kv.num_layers = static_cast<int32_t>(config.num_layers);
  kv.num_kv_heads = static_cast<int32_t>(config.num_kv_heads);
  kv.head_dim = static_cast<int32_t>(config.head_dim);
  kv.budget_bytes = opt.kv_bytes;
  PagedKVCache cache(backend, kv);

  ModelLimits limits;
  limits.max_tokens_per_step = engine_config.max_tokens_per_step;
  limits.max_blocks_per_seq = PagedKVCache::blocks_for(config.max_position_embeddings, kv.block_size);
  Qwen3Model model(backend, weights, limits);
  Engine engine(model, cache, std::make_unique<FcfsScheduler>(engine_config.max_tokens_per_step), engine_config);
  const double load_seconds = seconds_since(load_start);

  const std::string text =
      opt.chat ? apply_chat_template(std::vector<ChatMessage>{{"user", opt.prompt}}, opt.thinking) : opt.prompt;
  const std::vector<int32_t> prompt = tokenizer.encode(text);

  StreamDecoder decoder(tokenizer);
  const auto gen_start = Clock::now();
  double ttft = 0.0;
  size_t generated = 0;
  engine.set_token_callback([&](int32_t, int32_t token) {
    if (generated++ == 0) ttft = seconds_since(gen_start);
    const auto& stops = engine_config.stop_tokens;
    if (std::find(stops.begin(), stops.end(), token) == stops.end()) write(decoder.push(token));
  });
  // add_request + step loop rather than generate(), which releases the sequence before
  // the effective (capacity-clamped) max_tokens can be read.
  const int32_t seq_id = engine.add_request(prompt, opt.sampling);
  while (engine.sequence(seq_id).state != SeqState::Finished) engine.step();
  write(decoder.flush());
  write("\n");
  const int32_t effective_max_tokens = engine.sequence(seq_id).params.max_tokens;
  const FinishReason finish_reason = engine.sequence(seq_id).finish_reason;
  engine.release(seq_id);

  const EngineStats& s = engine.stats();
  std::fprintf(stderr,
               "\n[tie] %s | load %.2f s | prompt %zu tok | ttft %.0f ms | prefill %.1f tok/s | decode %.1f tok/s | "
               "%zu tok generated | kv %d blocks for %zu tok, pool %d blocks (%.2f GiB)\n",
               opt.device.c_str(), load_seconds, prompt.size(), ttft * 1e3,
               s.prefill_seconds > 0 ? double(s.prefill_tokens) / s.prefill_seconds : 0.0,
               s.decode_seconds > 0 ? double(s.decode_tokens) / s.decode_seconds : 0.0, generated,
               PagedKVCache::blocks_for(static_cast<int64_t>(prompt.size() + generated), kv.block_size),
               prompt.size() + generated, cache.num_blocks(),
               double(cache.bytes_per_block()) * cache.num_blocks() / double(size_t{1} << 30));
  if (effective_max_tokens < opt.sampling.max_tokens) {
    std::fprintf(stderr, "[tie] note: max_tokens clamped to %d by the KV cache / context\n", effective_max_tokens);
  }
  if (finish_reason == FinishReason::Error) std::fprintf(stderr, "[tie] warning: generation ended with an error\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  if (args.empty() || args[0] == "-h" || args[0] == "--help") {
    std::fputs(tie::usage().data(), args.empty() ? stderr : stdout);
    return args.empty() ? 1 : 0;
  }
  try {
    if (args[0] != "generate") tie::fail<tie::InvalidArgument>("unknown command '{}'", args[0]);
    return run_generate(tie::parse_generate_args(std::span(args).subspan(1)));
  } catch (const tie::Error& e) {
    std::fprintf(stderr, "tie: error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tie: internal error: %s\n", e.what());
    return 2;
  }
}
