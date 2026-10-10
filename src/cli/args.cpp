#include "cli/args.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>
#include <string>
#include <type_traits>

#include "core/error.h"

namespace tie {

namespace {

template <typename T>
T parse_number(std::string_view flag, std::string_view text) {
  const std::string s(text);
  size_t used = 0;
  try {
    if constexpr (std::is_floating_point_v<T>) {
      const double d = std::stod(s, &used);  // accepts "inf" and "nan", which no option can use
      if (used == s.size() && std::isfinite(d)) return static_cast<T>(d);
    } else {
      const long long n = std::stoll(s, &used);
      if (used == s.size() && std::in_range<T>(n)) {
        return static_cast<T>(n);
      }
    }
  } catch (const std::exception&) {
  }
  fail<InvalidArgument>("{} expects a finite number, got '{}'", flag, text);
}

}  // namespace

std::string_view usage() {
  return R"(usage: tie generate -m <model.gguf | model_dir> -p <prompt> [options]

options:
  -m, --model PATH      GGUF file or HuggingFace model directory (safetensors)
  -p, --prompt TEXT     prompt text
  --chat                wrap the prompt in the model's chat template
  --no-think            with --chat: skip Qwen3's thinking block
  --device metal|cpu    compute device (default metal)
  --max-tokens N        tokens to generate (default 256)
  --temp T              sampling temperature, 0 = greedy (default 0.7)
  --top-k K             keep the K most likely tokens, 0 = off (default 20)
  --top-p P             nucleus sampling threshold (default 0.8)
  --seed S              sampling seed (default 0)
  --kv-mem SIZE         KV cache budget, e.g. 4G or 512M (default 4G)
  --threads N           CPU threads, 0 = performance cores (default 0)
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
  const size_t value = std::stoull(std::string(digits)) << shift;
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
    if (flag == "-m" || flag == "--model") {
      o.model = std::string(value());
    } else if (flag == "-p" || flag == "--prompt") {
      o.prompt = std::string(value());
    } else if (flag == "--chat") {
      o.chat = true;
    } else if (flag == "--no-think") {
      o.thinking = false;
    } else if (flag == "--device") {
      o.device = std::string(value());
      if (o.device != "cpu" && o.device != "metal") fail<InvalidArgument>("--device must be cpu or metal, got '{}'", o.device);
    } else if (flag == "--max-tokens") {
      o.sampling.max_tokens = parse_number<int32_t>(flag, value());
    } else if (flag == "--temp") {
      o.sampling.temperature = parse_number<float>(flag, value());
    } else if (flag == "--top-k") {
      o.sampling.top_k = parse_number<int32_t>(flag, value());
    } else if (flag == "--top-p") {
      o.sampling.top_p = parse_number<float>(flag, value());
    } else if (flag == "--seed") {
      o.sampling.seed = parse_number<uint64_t>(flag, value());
    } else if (flag == "--kv-mem") {
      o.kv_bytes = parse_size(value());
    } else if (flag == "--threads") {
      o.threads = parse_number<int>(flag, value());
    } else {
      fail<InvalidArgument>("unknown option '{}'", flag);
    }
  }
  if (o.model.empty()) fail<InvalidArgument>("--model is required");
  if (o.prompt.empty()) fail<InvalidArgument>("--prompt is required");
  validate(o.sampling);
  return o;
}

}  // namespace tie
