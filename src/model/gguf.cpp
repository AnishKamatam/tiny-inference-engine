#include "model/gguf.h"

#include <algorithm>
#include <cstring>

#include "core/error.h"

namespace tie {

namespace {

enum GgufType : uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7,
  kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" read as a little-endian uint32
constexpr uint32_t kMaxDims = 4;

bool is_integer(uint32_t t) { return t <= kI32 || t == kU64 || t == kI64; }

// Bounds-checked sequential reader over the mapped file.
class Reader {
 public:
  explicit Reader(const MappedFile& file) : file_(file) {}

  template <typename T>
  T read() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, file_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }

  std::string read_string() {
    const uint64_t n = read<uint64_t>();
    need(n);
    std::string s(reinterpret_cast<const char*>(file_.data() + pos_), n);
    pos_ += n;
    return s;
  }

  int64_t read_integer(uint32_t type) {
    switch (type) {
      case kU8: return read<uint8_t>();
      case kI8: return read<int8_t>();
      case kU16: return read<uint16_t>();
      case kI16: return read<int16_t>();
      case kU32: return read<uint32_t>();
      case kI32: return read<int32_t>();
      case kU64: return static_cast<int64_t>(read<uint64_t>());
      case kI64: return read<int64_t>();
      default: fail<LoadError>("{}: type {} is not an integer", file_.path().string(), type);
    }
  }

  size_t pos() const { return pos_; }
  size_t remaining() const { return file_.size() - pos_; }

 private:
  void need(uint64_t n) const {
    if (n > remaining()) {
      fail<LoadError>("{}: truncated: need {} bytes at offset {} but the file is {} bytes", file_.path().string(), n,
                      pos_, file_.size());
    }
  }

  const MappedFile& file_;
  size_t pos_ = 0;
};

GgufValue read_value(Reader& r, uint32_t type, const std::string& key, const std::filesystem::path& path) {
  if (is_integer(type)) return r.read_integer(type);
  switch (type) {
    case kF32: return static_cast<double>(r.read<float>());
    case kF64: return r.read<double>();
    case kBool: return r.read<uint8_t>() != 0;
    case kString: return r.read_string();
    case kArray: {
      const uint32_t elem = r.read<uint32_t>();
      const uint64_t n = r.read<uint64_t>();
      // Every element takes at least one byte, so this caps reservations from corrupt counts.
      const size_t reserve = static_cast<size_t>(std::min<uint64_t>(n, r.remaining()));
      if (elem == kString) {
        std::vector<std::string> v;
        v.reserve(reserve);
        for (uint64_t i = 0; i < n; ++i) v.push_back(r.read_string());
        return v;
      }
      if (elem == kF32 || elem == kF64) {
        std::vector<double> v;
        v.reserve(reserve);
        for (uint64_t i = 0; i < n; ++i) v.push_back(elem == kF32 ? r.read<float>() : r.read<double>());
        return v;
      }
      if (is_integer(elem) || elem == kBool) {
        std::vector<int64_t> v;
        v.reserve(reserve);
        for (uint64_t i = 0; i < n; ++i) v.push_back(elem == kBool ? r.read<uint8_t>() : r.read_integer(elem));
        return v;
      }
      fail<UnsupportedError>("{}: metadata '{}' is an array of type {}", path.string(), key, elem);
    }
    default: fail<LoadError>("{}: metadata '{}' has unknown type {}", path.string(), key, type);
  }
}

}  // namespace

std::optional<DType> dtype_from_ggml(uint32_t ggml_type) {
  switch (ggml_type) {
    case 0: return DType::F32;
    case 1: return DType::F16;
    case 8: return DType::Q8_0;
    case 30: return DType::BF16;
    default: return std::nullopt;
  }
}

GgufFile::GgufFile(const std::filesystem::path& path) : file_(path) {
  const std::string where = path.string();
  Reader r(file_);
  if (r.read<uint32_t>() != kMagic) fail<LoadError>("{}: not a GGUF file (bad magic)", where);
  version_ = r.read<uint32_t>();
  if (version_ < 2 || version_ > 3) fail<UnsupportedError>("{}: GGUF version {} (supported: 2, 3)", where, version_);
  const uint64_t n_tensors = r.read<uint64_t>();
  const uint64_t n_kv = r.read<uint64_t>();
  if (n_tensors > file_.size() || n_kv > file_.size()) fail<LoadError>("{}: corrupt header counts", where);

  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key = r.read_string();
    const uint32_t type = r.read<uint32_t>();
    GgufValue value = read_value(r, type, key, path);
    if (!metadata_.emplace(key, std::move(value)).second) fail<LoadError>("{}: duplicate metadata key '{}'", where, key);
  }
  if (has("general.alignment")) {
    const int64_t a = get_int("general.alignment");
    if (a <= 0 || (a & (a - 1)) != 0) fail<LoadError>("{}: general.alignment {} is not a power of two", where, a);
    alignment_ = static_cast<uint64_t>(a);
  }

  tensors_.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    t.name = r.read_string();
    const uint32_t n_dims = r.read<uint32_t>();
    if (n_dims == 0 || n_dims > kMaxDims) fail<LoadError>("{}: tensor '{}' has {} dims", where, t.name, n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) t.dims.push_back(static_cast<int64_t>(r.read<uint64_t>()));
    t.type = r.read<uint32_t>();
    t.offset = r.read<uint64_t>();
    tensors_.push_back(std::move(t));
  }
  data_offset_ = (r.pos() + alignment_ - 1) / alignment_ * alignment_;

  for (const GgufTensorInfo& t : tensors_) {
    if (t.offset % alignment_ != 0) fail<LoadError>("{}: tensor '{}' is misaligned", where, t.name);
    const std::optional<DType> dtype = dtype_from_ggml(t.type);
    if (!dtype) continue;  // the loader rejects unsupported types with a precise message
    int64_t numel = 1;
    for (int64_t d : t.dims) {
      if (d <= 0) fail<LoadError>("{}: tensor '{}' has a non-positive dimension", where, t.name);
      // Every element occupies at least one byte, so a larger count can never fit in the file;
      // rejecting it here also keeps the byte-size arithmetic below from overflowing.
      if (__builtin_mul_overflow(numel, d, &numel) || static_cast<uint64_t>(numel) > file_.size()) {
        fail<LoadError>("{}: tensor '{}' has dims too large for the {}-byte file", where, t.name, file_.size());
      }
    }
    if (t.offset > file_.size()) {
      fail<LoadError>("{}: tensor '{}' offset {} is past the end of the {}-byte file", where, t.name, t.offset,
                      file_.size());
    }
    if (t.dims[0] % traits(*dtype).block_elems != 0) {
      fail<LoadError>("{}: tensor '{}' rows of {} do not fill whole {} blocks", where, t.name, t.dims[0], name(*dtype));
    }
    const uint64_t end = data_offset_ + t.offset + bytes_for(*dtype, numel);
    if (end > file_.size()) {
      fail<LoadError>("{}: tensor '{}' extends to byte {}, past the end of the {}-byte file", where, t.name, end,
                      file_.size());
    }
  }
}

template <typename T>
const T& GgufFile::get(std::string_view key, std::string_view type_name) const {
  const auto it = metadata_.find(key);
  if (it == metadata_.end()) fail<LoadError>("{}: missing metadata key '{}'", path().string(), key);
  const T* v = std::get_if<T>(&it->second);
  if (v == nullptr) fail<LoadError>("{}: metadata '{}' is not {}", path().string(), key, type_name);
  return *v;
}

int64_t GgufFile::get_int(std::string_view key) const { return get<int64_t>(key, "an integer"); }
double GgufFile::get_float(std::string_view key) const { return get<double>(key, "a float"); }
bool GgufFile::get_bool(std::string_view key) const { return get<bool>(key, "a bool"); }
const std::string& GgufFile::get_string(std::string_view key) const { return get<std::string>(key, "a string"); }
const std::vector<std::string>& GgufFile::get_strings(std::string_view key) const {
  return get<std::vector<std::string>>(key, "a string array");
}
const std::vector<int64_t>& GgufFile::get_ints(std::string_view key) const {
  return get<std::vector<int64_t>>(key, "an integer array");
}

}  // namespace tie
