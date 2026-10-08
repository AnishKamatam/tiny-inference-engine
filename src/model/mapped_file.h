#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace tie {

// Read-only memory map of a whole file. The mapping starts on a page boundary,
// which lets GPU backends wrap model files without copying.
class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  std::span<const uint8_t> bytes() const { return {data_, size_}; }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace tie
