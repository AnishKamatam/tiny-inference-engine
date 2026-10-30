#include "cli/args.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "core/error.h"

namespace tie {

namespace {

template <typename T>
T parse_number(std::string_view flag, std::string_view text) {
  if constexpr (std::is_floating_point_v<T>) {
    const std::string s(text);
    size_t used = 0;
    try {
      if (!s.empty() && !std::isspace(static_cast<unsigned char>(s[0]))) {
        const T value = static_cast<T>(std::stod(s, &used));  // accepts "inf" and "nan", which no option can use
        if (used == s.size() && std::isfinite(value)) return value;
      }
    } catch (const std::exception&) {
    }
    fail<InvalidArgument>("{} expects a finite number, got '{}'", flag, text);
  } else {
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec == std::errc{} && end == text.data() + text.size()) return value;
    fail<InvalidArgument>("{} expects an integer, got '{}'", flag, text);
  }
}

// Parses a flag `generate` and `bench` share. Returns false when `flag` is not one of them.
template <typename Options, typename Value>
bool parse_shared_flag(std::string_view flag, Value&& value, Options& o) {
  if (flag == "-m" || flag == "--model") {
    o.model = std::string(value());
  } else if (flag == "--device") {
    o.device = std::string(value());
    if (o.device != "cpu" && o.device != "metal") fail<InvalidArgument>("--device must be cpu or metal, got '{}'", o.device);
  } else if (flag == "--temp") {
    o.sampling.temperature = parse_number<float>(flag, value());
  } else if (flag == "--top-k") {
    o.sampling.top_k = parse_number<int32_t>(flag, value());
  } else if (flag == "--top-p") {
    o.sampling.top_p = parse_number<float>(flag, value());
  } else if (flag == "--seed") {
    o.sampling.seed = parse_number<uint64_t>(flag, value());
  } else if (flag == "--kv-mem") {
    try {
      o.kv_bytes = parse_size(value());
    } catch (const InvalidArgument& e) {
      fail<InvalidArgument>("{}: {}", flag, e.what());
    }
  } else if (flag == "--threads") {
    o.threads = parse_number<int>(flag, value());
  } else {
    return false;
  }
  return true;
}

}  // namespace

std::string_view usage() {
  return R"(usage: tie generate -m <model.gguf | model_dir> -p <prompt> [options]
       tie bench -m <model.gguf | model_dir> (--prompts FILE.jsonl | --synthetic N) [options]

common options:
  -m, --model PATH      GGUF file or HuggingFace model directory (safetensors)
  --device metal|cpu    compute device (default metal)
  --temp T              sampling temperature, 0 = greedy (generate default 0.7, bench default 0)
  --top-k K             keep the K most likely tokens, 0 = off (generate default 20)
  --top-p P             nucleus sampling threshold (generate default 0.8)
  --seed S              sampling seed (default 0); bench also seeds its workload with it
  --kv-mem SIZE         KV cache budget, e.g. 4G or 512M (default 4G)
  --threads N           CPU threads, 0 = performance cores (default 0)

generate options:
  -p, --prompt TEXT     prompt text
  --chat                wrap the prompt in the model's chat template
  --no-think            with --chat: skip Qwen3's thinking block
  --max-tokens N        tokens to generate (default 256)

bench options:
  --prompts FILE        JSONL, one {"prompt": "...", "max_tokens": N, "chat": bool} per line
  --synthetic N         N requests of random tokens
  --prompt-len A:B      synthetic prompt lengths (default 512:1024)
  --output-len C:D      synthetic output lengths (default 128)
  --max-tokens N        output tokens for JSONL lines without max_tokens (default 128)
  --rate R              requests per second, Poisson arrivals; inf = all at once (default inf)
  --max-seqs S          sequences per step (default 64)
  --max-tokens-per-step B  tokens per step (default 512)
  --policy fcfs|srpt-aging  admission order (default srpt-aging)
  --aging N             srpt-aging priority gained per step waited, in tokens (default 16)
  --json                print the report as one JSON object
)";
}

size_t parse_size(std::string_view text) {
  size_t shift = 0;
  std::string_view digits = text;
  if (!text.empty()) {
    switch (std::toupper(static_cast<unsigned char>(text.back()))) {
      case 'K': shift = 10; break;
      case 'M': shift = 20; break;
      case 'G': shift = 30; break;
      default: break;
    }
    if (shift != 0) digits.remove_suffix(1);
  }
  if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
    fail<InvalidArgument>("size '{}' must be a whole number with an optional K, M or G suffix", text);
  }
  size_t base = 0;
  const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), base);
  if (ec != std::errc{} || base > (SIZE_MAX >> shift)) fail<InvalidArgument>("size '{}' is too large", text);
  const size_t value = base << shift;
  if (value == 0) fail<InvalidArgument>("size '{}' must be positive", text);
  return value;
}

GenerateOptions parse_generate_args(std::span<const std::string_view> args) {
  GenerateOptions o;
  o.sampling.temperature = 0.7f;
  o.sampling.top_k = 20;
  o.sampling.top_p = 0.8f;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    const auto value = [&]() -> std::string_view {
      if (i + 1 >= args.size()) fail<InvalidArgument>("{} needs a value", flag);
      return args[++i];
    };
    if (parse_shared_flag(flag, value, o)) {
      continue;
    } else if (flag == "-p" || flag == "--prompt") {
      o.prompt = std::string(value());
    } else if (flag == "--chat") {
      o.chat = true;
    } else if (flag == "--no-think") {
      o.thinking = false;
    } else if (flag == "--max-tokens") {
      o.sampling.max_tokens = parse_number<int32_t>(flag, value());
    } else {
      fail<InvalidArgument>("unknown option '{}'", flag);
    }
  }
  if (o.model.empty()) fail<InvalidArgument>("--model is required");
  if (o.prompt.empty()) fail<InvalidArgument>("--prompt is required");
  validate(o.sampling);
  return o;
}

LengthRange parse_range(std::string_view flag, std::string_view text) {
  const size_t colon = text.find(':');
  LengthRange r;
  r.lo = parse_number<int32_t>(flag, text.substr(0, colon));
  r.hi = colon == std::string_view::npos ? r.lo : parse_number<int32_t>(flag, text.substr(colon + 1));
  if (r.lo < 1 || r.hi < r.lo) fail<InvalidArgument>("{} expects A:B with 1 <= A <= B, got '{}'", flag, text);
  return r;
}

BenchOptions parse_bench_args(std::span<const std::string_view> args) {
  BenchOptions o;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    const auto value = [&]() -> std::string_view {
      if (i + 1 >= args.size()) fail<InvalidArgument>("{} needs a value", flag);
      return args[++i];
    };
    if (parse_shared_flag(flag, value, o)) {
      continue;
    } else if (flag == "--prompts") {
      o.prompts = std::string(value());
    } else if (flag == "--synthetic") {
      o.synthetic = parse_number<int32_t>(flag, value());
      if (o.synthetic < 1) fail<InvalidArgument>("--synthetic needs at least 1 request, got {}", o.synthetic);
    } else if (flag == "--prompt-len") {
      o.prompt_len = parse_range(flag, value());
    } else if (flag == "--output-len") {
      o.output_len = parse_range(flag, value());
    } else if (flag == "--max-tokens") {
      o.max_tokens = parse_number<int32_t>(flag, value());
      if (o.max_tokens < 1) fail<InvalidArgument>("--max-tokens must be positive, got {}", o.max_tokens);
    } else if (flag == "--rate") {
      const std::string_view v = value();
      o.rate = v == "inf" ? std::numeric_limits<double>::infinity() : parse_number<double>(flag, v);
      if (!(o.rate > 0)) fail<InvalidArgument>("--rate must be positive or inf, got '{}'", v);
    } else if (flag == "--max-seqs") {
      o.serving.max_seqs = parse_number<int32_t>(flag, value());
    } else if (flag == "--max-tokens-per-step") {
      o.serving.max_tokens_per_step = parse_number<int32_t>(flag, value());
    } else if (flag == "--policy") {
      o.policy = parse_policy(value());
    } else if (flag == "--aging") {
      o.aging = parse_number<int32_t>(flag, value());
      if (o.aging < 0) fail<InvalidArgument>("--aging must be non-negative, got {}", o.aging);
    } else if (flag == "--json") {
      o.json = true;
    } else {
      fail<InvalidArgument>("unknown option '{}'", flag);
    }
  }
  if (o.model.empty()) fail<InvalidArgument>("--model is required");
  if (o.prompts.empty() == (o.synthetic == 0)) fail<InvalidArgument>("give exactly one of --prompts or --synthetic");
  validate(o.serving);
  validate(o.sampling);
  return o;
}

}  // namespace tie
