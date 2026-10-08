#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/tensor.h"
#include "model/config.h"

namespace tie {

// One tensor of a model file. `shape` is in HF order ([out, in] for matrices).
// `data == regions[region].data() + offset` in the owning LoadedModel.
struct TensorInfo {
  DType dtype;
  Shape shape;
  size_t region;
  size_t offset;
  size_t nbytes;
  const uint8_t* data;
};

// Tensors keyed by canonical (HuggingFace) name.
class TensorTable {
 public:
  void add(std::string name, TensorInfo info);
  const TensorInfo* find(std::string_view name) const;
  const TensorInfo& at(std::string_view name) const;  // LoadError naming the tensor if absent
  size_t size() const { return tensors_.size(); }
  std::vector<std::string> names() const;  // sorted

 private:
  std::map<std::string, TensorInfo, std::less<>> tensors_;
};

struct TokenizerData {
  std::vector<std::string> tokens;  // id -> token in the GPT-2 byte-level alphabet; "" for unused ids
  std::vector<std::pair<std::string, std::string>> merges;  // highest priority first
  std::vector<int32_t> special_ids;  // added and control tokens, matched verbatim before BPE
};

// Everything read from a model path. Owns the file mappings that `tensors` and
// `regions` point into, so it must outlive any model built from it.
struct LoadedModel {
  ModelConfig config;
  TensorTable tensors;
  TokenizerData tokenizer;
  std::vector<std::span<const uint8_t>> regions;  // whole mapped files, page aligned
  std::vector<std::shared_ptr<const void>> owners;
};

LoadedModel load_gguf(const std::filesystem::path& path);
LoadedModel load_safetensors(const std::filesystem::path& dir);
// A *.gguf file or a directory holding config.json, tokenizer.json and safetensors.
LoadedModel load_model(const std::filesystem::path& path);

}  // namespace tie
