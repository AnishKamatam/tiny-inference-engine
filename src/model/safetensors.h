#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/tensor.h"
#include "model/mapped_file.h"

namespace tie {

struct SafetensorsTensor {
  std::string name;
  DType dtype;
  Shape shape;           // as stored (HF order, outermost first)
  uint64_t file_offset;  // absolute offset of the first byte in the file
  size_t nbytes;
  const uint8_t* data;
};

// A parsed .safetensors file: an 8-byte header length, a JSON header, then raw tensor bytes.
class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::filesystem::path& path);

  const std::filesystem::path& path() const { return file_.path(); }
  const MappedFile& file() const { return file_; }
  const std::vector<SafetensorsTensor>& tensors() const { return tensors_; }

 private:
  MappedFile file_;
  std::vector<SafetensorsTensor> tensors_;
};

}  // namespace tie
