#include "support/npy.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

namespace tie::test {

namespace {

std::string field(const std::string& header, const std::string& key, const std::filesystem::path& path) {
  const size_t k = header.find("'" + key + "':");
  if (k == std::string::npos) fail<LoadError>("{}: npy header has no '{}'", path.string(), key);
  return header.substr(k + key.size() + 3);
}

}  // namespace

NpyArray load_npy(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail<LoadError>("{}: cannot open", path.string());
  std::vector<char> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (file.size() < 10 || std::memcmp(file.data(), "\x93NUMPY", 6) != 0) {
    fail<LoadError>("{}: not an npy file", path.string());
  }
  const int major = static_cast<unsigned char>(file[6]);
  size_t header_len = 0;
  size_t header_start = 0;
  if (major == 1) {
    header_len = static_cast<unsigned char>(file[8]) | (static_cast<size_t>(static_cast<unsigned char>(file[9])) << 8);
    header_start = 10;
  } else {
    uint32_t len32 = 0;
    std::memcpy(&len32, file.data() + 8, 4);
    header_len = len32;
    header_start = 12;
  }
  if (header_start + header_len > file.size()) fail<LoadError>("{}: truncated npy header", path.string());
  const std::string header(file.data() + header_start, header_len);

  NpyArray out;
  const std::string descr = field(header, "descr", path);
  out.descr = descr.substr(descr.find('\'') + 1, 3);
  if (field(header, "fortran_order", path).find("False") != 1) {
    fail<UnsupportedError>("{}: Fortran-order arrays are not supported", path.string());
  }
  const std::string shape = field(header, "shape", path);
  for (size_t i = shape.find('(') + 1; i < shape.size() && shape[i] != ')';) {
    if (std::isdigit(static_cast<unsigned char>(shape[i]))) {
      size_t used = 0;
      out.shape.push_back(std::stoll(shape.substr(i), &used));
      i += used;
    } else {
      ++i;
    }
  }
  const size_t data_start = header_start + header_len;
  const size_t want = static_cast<size_t>(out.numel()) * 4;
  if (file.size() - data_start != want) {
    fail<LoadError>("{}: expected {} data bytes, found {}", path.string(), want, file.size() - data_start);
  }
  out.bytes.assign(file.begin() + static_cast<std::ptrdiff_t>(data_start), file.end());
  return out;
}

}  // namespace tie::test
