#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace tie::test {

template <typename T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  const auto* p = reinterpret_cast<const uint8_t*>(v.data());
  return {p, p + v.size() * sizeof(T)};
}

// Writes GGUF v3 files for parser tests. Mirrors the layout llama.cpp's gguf writer produces.
class GgufWriter {
 public:
  void add_u32(const std::string& key, uint32_t v) { kv(key, 4); put(kvs_, v); }
  void add_i32(const std::string& key, int32_t v) { kv(key, 5); put(kvs_, v); }
  void add_f32(const std::string& key, float v) { kv(key, 6); put(kvs_, v); }
  void add_bool(const std::string& key, bool v) { kv(key, 7); put(kvs_, static_cast<uint8_t>(v)); }
  void add_string(const std::string& key, const std::string& v) { kv(key, 8); put_string(kvs_, v); }
  void add_strings(const std::string& key, const std::vector<std::string>& v) {
    kv(key, 9);
    put(kvs_, uint32_t{8});
    put(kvs_, static_cast<uint64_t>(v.size()));
    for (const auto& s : v) put_string(kvs_, s);
  }
  void add_i32s(const std::string& key, const std::vector<int32_t>& v) {
    kv(key, 9);
    put(kvs_, uint32_t{5});
    put(kvs_, static_cast<uint64_t>(v.size()));
    for (int32_t x : v) put(kvs_, x);
  }
  void set_alignment(uint32_t alignment) {
    alignment_ = alignment;
    add_u32("general.alignment", alignment);
  }
  // `dims` innermost-first, exactly as GGUF stores them.
  void add_tensor(const std::string& name, uint32_t ggml_type, std::vector<uint64_t> dims, std::vector<uint8_t> data) {
    tensors_.push_back({name, ggml_type, std::move(dims), std::move(data)});
  }

  void write(const std::filesystem::path& path) const {
    std::vector<uint8_t> out = {'G', 'G', 'U', 'F'};
    put(out, uint32_t{3});
    put(out, static_cast<uint64_t>(tensors_.size()));
    put(out, n_kv_);
    out.insert(out.end(), kvs_.begin(), kvs_.end());
    uint64_t offset = 0;
    for (const auto& t : tensors_) {
      put_string(out, t.name);
      put(out, static_cast<uint32_t>(t.dims.size()));
      for (uint64_t d : t.dims) put(out, d);
      put(out, t.type);
      put(out, offset);
      offset += align(t.data.size());
    }
    out.resize(align(out.size()), 0);
    for (const auto& t : tensors_) {
      out.insert(out.end(), t.data.begin(), t.data.end());
      out.resize(align(out.size()), 0);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
  }

 private:
  struct Tensor {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> data;
  };

  template <typename T>
  static void put(std::vector<uint8_t>& out, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
  }
  static void put_string(std::vector<uint8_t>& out, const std::string& s) {
    put(out, static_cast<uint64_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
  }
  void kv(const std::string& key, uint32_t type) {
    put_string(kvs_, key);
    put(kvs_, type);
    ++n_kv_;
  }
  size_t align(size_t n) const { return (n + alignment_ - 1) / alignment_ * alignment_; }

  std::vector<uint8_t> kvs_;
  uint64_t n_kv_ = 0;
  std::vector<Tensor> tensors_;
  uint32_t alignment_ = 32;
};

}  // namespace tie::test
