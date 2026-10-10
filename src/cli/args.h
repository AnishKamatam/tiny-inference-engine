#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

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

std::string_view usage();

// Parses the arguments that follow `tie generate`. Throws InvalidArgument.
GenerateOptions parse_generate_args(std::span<const std::string_view> args);

// "4G", "512M", "64K" or a plain byte count; must be positive.
size_t parse_size(std::string_view text);

}  // namespace tie
