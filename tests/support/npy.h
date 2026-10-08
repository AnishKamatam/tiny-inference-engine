#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

#include "core/error.h"

namespace tie::test {

// Minimal reader for the little-endian, C-order .npy files tools/dump_reference.py writes.
struct NpyArray {
  std::string descr;  // "<f4" or "<i4"
  std::vector<int64_t> shape;
  std::vector<char> bytes;

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }

  template <typename T>
  const T* as() const {
    const char* want = std::is_same_v<T, float> ? "<f4" : std::is_same_v<T, int32_t> ? "<i4" : "?";
    if (descr != want) fail<InvalidArgument>("npy array has dtype {}, requested {}", descr, want);
    return reinterpret_cast<const T*>(bytes.data());
  }
};

NpyArray load_npy(const std::filesystem::path& path);

}  // namespace tie::test
