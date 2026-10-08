#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/dtype.h"
#include "model/mapped_file.h"

namespace tie {

// GGML tensor type ids tie can compute with; everything else is rejected by the loader.
std::optional<DType> dtype_from_ggml(uint32_t ggml_type);

using GgufValue = std::variant<bool, int64_t, double, std::string, std::vector<int64_t>, std::vector<double>,
                               std::vector<std::string>>;

struct GgufTensorInfo {
  std::string name;
  uint32_t type;              // raw GGML type id
  std::vector<int64_t> dims;  // as stored: innermost (contiguous) dimension first
  uint64_t offset;            // relative to the start of the data section
};

// A parsed GGUF v2/v3 file. Metadata is decoded eagerly; tensor bytes stay in the mmap.
class GgufFile {
 public:
  explicit GgufFile(const std::filesystem::path& path);

  const std::filesystem::path& path() const { return file_.path(); }
  const MappedFile& file() const { return file_; }
  uint32_t version() const { return version_; }
  uint64_t alignment() const { return alignment_; }
  size_t data_offset() const { return data_offset_; }  // absolute file offset of the data section
  const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
  const uint8_t* tensor_data(const GgufTensorInfo& t) const { return file_.data() + data_offset_ + t.offset; }

  bool has(std::string_view key) const { return metadata_.find(key) != metadata_.end(); }
  int64_t get_int(std::string_view key) const;
  double get_float(std::string_view key) const;
  bool get_bool(std::string_view key) const;
  const std::string& get_string(std::string_view key) const;
  const std::vector<std::string>& get_strings(std::string_view key) const;
  const std::vector<int64_t>& get_ints(std::string_view key) const;

 private:
  template <typename T>
  const T& get(std::string_view key, std::string_view type_name) const;

  MappedFile file_;
  uint32_t version_ = 0;
  uint64_t alignment_ = 32;
  size_t data_offset_ = 0;
  std::map<std::string, GgufValue, std::less<>> metadata_;
  std::vector<GgufTensorInfo> tensors_;
};

}  // namespace tie
