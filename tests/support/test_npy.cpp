#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/error.h"
#include "support/npy.h"
#include "support/test_data.h"

using namespace tie;

namespace {

// Writes a version-1.0 .npy file the way numpy.save does.
std::filesystem::path write_npy(const std::string& descr, const std::string& shape, const void* data, size_t bytes) {
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  while ((10 + header.size() + 1) % 64 != 0) header += ' ';
  header += '\n';
  const auto path = test::temp_path("array_" + descr.substr(1) + ".npy");
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  const uint16_t len = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&len), 2);
  f << header;
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  return path;
}

}  // namespace

TEST_CASE("load_npy reads float32 and int32 arrays") {
  const float floats[6] = {0, 1, 2, 3, 4, 5};
  const auto fpath = write_npy("<f4", "(2, 3)", floats, sizeof(floats));
  test::NpyArray a = test::load_npy(fpath);
  CHECK(a.shape == std::vector<int64_t>{2, 3});
  CHECK(a.numel() == 6);
  CHECK(a.as<float>()[5] == 5.0f);

  const int32_t ints[3] = {7, 8, 9};
  const auto ipath = write_npy("<i4", "(3,)", ints, sizeof(ints));
  test::NpyArray b = test::load_npy(ipath);
  CHECK(b.shape == std::vector<int64_t>{3});
  CHECK(b.as<int32_t>()[2] == 9);
  CHECK_THROWS_AS(b.as<float>(), InvalidArgument);
}

TEST_CASE("load_npy rejects files that are not npy") {
  const auto path = test::temp_path("not_npy.npy");
  std::ofstream(path) << "hello";
  CHECK_THROWS_AS(test::load_npy(path), LoadError);
}
