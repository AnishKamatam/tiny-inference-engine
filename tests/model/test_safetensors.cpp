#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/error.h"
#include "model/safetensors.h"
#include "support/test_data.h"

using namespace tie;

namespace {

void write_safetensors(const std::filesystem::path& path, const nlohmann::json& header, const std::vector<uint8_t>& data) {
  const std::string h = header.dump();
  const uint64_t len = h.size();
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&len), 8);
  f << h;
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

}  // namespace

TEST_CASE("SafetensorsFile parses tensors and skips __metadata__") {
  std::vector<uint8_t> data(6 * 4 + 4 * 2);
  reinterpret_cast<float*>(data.data())[5] = 42.0f;
  nlohmann::json header = {
      {"__metadata__", {{"format", "pt"}}},
      {"w", {{"dtype", "F32"}, {"shape", {2, 3}}, {"data_offsets", {0, 24}}}},
      {"n", {{"dtype", "BF16"}, {"shape", {4}}, {"data_offsets", {24, 32}}}},
  };
  const auto path = test::temp_path("ok.safetensors");
  write_safetensors(path, header, data);

  SafetensorsFile f(path);
  REQUIRE(f.tensors().size() == 2);
  const SafetensorsTensor* w = nullptr;
  for (const auto& t : f.tensors()) {
    if (t.name == "w") w = &t;
  }
  REQUIRE(w != nullptr);
  CHECK(w->dtype == DType::F32);
  CHECK((w->shape == Shape{2, 3}));
  CHECK(w->nbytes == 24);
  CHECK(reinterpret_cast<const float*>(w->data)[5] == 42.0f);
}

TEST_CASE("SafetensorsFile rejects inconsistent or unsupported entries") {
  const auto path = test::temp_path("bad.safetensors");
  std::vector<uint8_t> data(16);

  SUBCASE("byte range does not match shape") {
    write_safetensors(path, {{"w", {{"dtype", "F32"}, {"shape", {2}}, {"data_offsets", {0, 12}}}}}, data);
    CHECK_THROWS_WITH_AS(SafetensorsFile{path}, doctest::Contains("'w'"), LoadError);
  }
  SUBCASE("byte range past end of file") {
    write_safetensors(path, {{"w", {{"dtype", "F32"}, {"shape", {8}}, {"data_offsets", {0, 32}}}}}, data);
    CHECK_THROWS_AS(SafetensorsFile{path}, LoadError);
  }
  SUBCASE("data_offsets near UINT64_MAX do not wrap") {
    write_safetensors(path,
                      {{"w", {{"dtype", "F32"}, {"shape", {2}}, {"data_offsets", {UINT64_MAX - 7, UINT64_MAX}}}}}, data);
    CHECK_THROWS_WITH_AS(SafetensorsFile{path}, doctest::Contains("'w'"), LoadError);
  }
  SUBCASE("unsupported dtype") {
    write_safetensors(path, {{"w", {{"dtype", "I64"}, {"shape", {2}}, {"data_offsets", {0, 16}}}}}, data);
    CHECK_THROWS_WITH_AS(SafetensorsFile{path}, doctest::Contains("I64"), UnsupportedError);
  }
  SUBCASE("header length past end of file") {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const uint64_t len = 1 << 20;
    f.write(reinterpret_cast<const char*>(&len), 8);
    f << "{}";
    f.close();
    CHECK_THROWS_WITH_AS(SafetensorsFile{path}, doctest::Contains(path.string().c_str()), LoadError);
  }
  SUBCASE("header is not JSON") {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const uint64_t len = 5;
    f.write(reinterpret_cast<const char*>(&len), 8);
    f << "nope!";
    f.close();
    CHECK_THROWS_AS(SafetensorsFile{path}, LoadError);
  }
}

TEST_CASE("SafetensorsFile rejects a shape whose element count overflows int64") {
  const auto path = test::temp_path("huge.safetensors");
  const int64_t big = int64_t{1} << 40;
  write_safetensors(path, {{"huge", {{"dtype", "F32"}, {"shape", {big, big}}, {"data_offsets", {0, 16}}}}},
                    std::vector<uint8_t>(16));
  CHECK_THROWS_WITH_AS(SafetensorsFile{path}, doctest::Contains("'huge'"), LoadError);
}
