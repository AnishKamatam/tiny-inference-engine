#include "model/safetensors.h"

#include <cstring>
#include <nlohmann/json.hpp>
#include <optional>

#include "core/error.h"

namespace tie {

namespace {

std::optional<DType> parse_dtype(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  return std::nullopt;
}

}  // namespace

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : file_(path) {
  const std::string where = path.string();
  if (file_.size() < 8) fail<LoadError>("{}: truncated: no header length", where);
  uint64_t header_len = 0;
  std::memcpy(&header_len, file_.data(), 8);
  if (header_len > file_.size() - 8) {
    fail<LoadError>("{}: header length {} runs past the end of the {}-byte file", where, header_len, file_.size());
  }
  const uint64_t data_start = 8 + header_len;
  const uint64_t data_size = file_.size() - data_start;
  const char* header_begin = reinterpret_cast<const char*>(file_.data() + 8);

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(header_begin, header_begin + header_len);
  } catch (const nlohmann::json::exception& e) {
    fail<LoadError>("{}: header is not valid JSON: {}", where, e.what());
  }
  if (!header.is_object()) fail<LoadError>("{}: header is not a JSON object", where);

  try {
    for (const auto& [name, entry] : header.items()) {
      if (name == "__metadata__") continue;
      const std::string dtype_str = entry.at("dtype").get<std::string>();
      const std::optional<DType> dtype = parse_dtype(dtype_str);
      if (!dtype) fail<UnsupportedError>("{}: tensor '{}' has dtype {} (supported: F32, F16, BF16)", where, name, dtype_str);

      std::vector<int64_t> dims = entry.at("shape").get<std::vector<int64_t>>();
      if (dims.empty()) dims.push_back(1);  // scalars are stored with shape []
      // Reject shapes whose element count overflows int64 (Shape::numel does not check).
      int64_t numel = 1;
      for (int64_t d : dims) {
        if (d < 0 || __builtin_mul_overflow(numel, d, &numel)) {
          fail<LoadError>("{}: tensor '{}' has an invalid or overflowing shape", where, name);
        }
      }
      std::optional<Shape> parsed;
      try {
        parsed = Shape::from(dims);
      } catch (const InvalidArgument& e) {
        fail<LoadError>("{}: tensor '{}' has an unsupported shape: {}", where, name, e.what());
      }
      const Shape shape = *parsed;

      const auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0]) fail<LoadError>("{}: tensor '{}' has bad data_offsets", where, name);
      // Compare against the data section size before any addition so nothing can wrap.
      if (offsets[1] > data_size) {
        fail<LoadError>("{}: tensor '{}' extends past the end of the file", where, name);
      }
      const uint64_t nbytes = offsets[1] - offsets[0];
      // Every supported dtype is at least 2 bytes per element, so this also keeps bytes_for from overflowing.
      if (static_cast<uint64_t>(numel) > nbytes || nbytes != bytes_for(*dtype, numel)) {
        fail<LoadError>("{}: tensor '{}' {} {} needs {} bytes but its data_offsets span {}", where, name, dtype_str,
                        shape.str(), static_cast<uint64_t>(numel) > nbytes ? "more" : std::to_string(bytes_for(*dtype, numel)),
                        nbytes);
      }
      tensors_.push_back(SafetensorsTensor{name, *dtype, shape, data_start + offsets[0], static_cast<size_t>(nbytes),
                                           file_.data() + data_start + offsets[0]});
    }
  } catch (const nlohmann::json::exception& e) {
    fail<LoadError>("{}: malformed header entry: {}", where, e.what());
  }
}

}  // namespace tie
