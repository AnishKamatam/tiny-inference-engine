#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include "bench/workload.h"
#include "engine/batch_scheduler.h"
#include "engine/serving_config.h"
#include "sampling/sampler.h"

namespace tie {

struct GenerateOptions {
  std::filesystem::path model;
  std::string prompt;
  bool chat = false;     // wrap the prompt in the chat template
  bool thinking = true;  // with --chat: let Qwen3 think before answering
  std::string device = "metal";
  SamplingParams sampling;
  size_t kv_bytes = size_t{4} << 30;
  int threads = 0;  // 0 = all performance cores
};

struct BenchOptions {
  std::filesystem::path model;
  std::filesystem::path prompts;  // JSONL workload; empty with --synthetic
  int32_t synthetic = 0;          // number of synthetic requests; 0 with --prompts
  LengthRange prompt_len{512, 1024};
  LengthRange output_len{128, 128};
  int32_t max_tokens = 128;  // for JSONL lines without max_tokens
  double rate = std::numeric_limits<double>::infinity();
  ServingConfig serving;
  OrderingPolicy policy = OrderingPolicy::SrptAging;
  int32_t aging = 16;
  SamplingParams sampling;  // greedy unless --temp; --seed also seeds the synthetic workload and arrivals
  std::string device = "metal";
  size_t kv_bytes = size_t{4} << 30;
  int threads = 0;
  bool json = false;
};

std::string_view usage();

// Parses the arguments that follow `tie generate`. Throws InvalidArgument.
GenerateOptions parse_generate_args(std::span<const std::string_view> args);

// Parses the arguments that follow `tie bench`. Throws InvalidArgument.
BenchOptions parse_bench_args(std::span<const std::string_view> args);

// "A:B" or "A", with 1 <= A <= B; errors name `flag`.
LengthRange parse_range(std::string_view flag, std::string_view text);

// "4G", "512M", "64K" or a plain byte count; must be positive.
size_t parse_size(std::string_view text);

}  // namespace tie
