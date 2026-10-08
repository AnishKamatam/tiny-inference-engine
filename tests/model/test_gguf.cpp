#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/error.h"
#include "model/gguf.h"
#include "support/gguf_writer.h"
#include "support/test_data.h"

using namespace tie;

TEST_CASE("GgufFile parses metadata of every scalar and array type") {
  test::GgufWriter w;
  w.add_string("general.architecture", "qwen3");
  w.add_u32("qwen3.block_count", 2);
  w.add_i32("signed", -5);
  w.add_f32("qwen3.rope.freq_base", 1e6f);
  w.add_bool("tokenizer.ggml.add_bos_token", false);
  w.add_strings("tokenizer.ggml.tokens", {"a", "b", "Ġc"});
  w.add_i32s("tokenizer.ggml.token_type", {1, 1, 3});
  const auto path = test::temp_path("meta.gguf");
  w.write(path);

  GgufFile f(path);
  CHECK(f.version() == 3);
  CHECK(f.alignment() == 32);
  CHECK(f.get_string("general.architecture") == "qwen3");
  CHECK(f.get_int("qwen3.block_count") == 2);
  CHECK(f.get_int("signed") == -5);
  CHECK(f.get_float("qwen3.rope.freq_base") == 1e6);
  CHECK(f.get_bool("tokenizer.ggml.add_bos_token") == false);
  CHECK((f.get_strings("tokenizer.ggml.tokens") == std::vector<std::string>{"a", "b", "Ġc"}));
  CHECK(f.get_ints("tokenizer.ggml.token_type")[2] == 3);
  CHECK_FALSE(f.has("missing"));
  CHECK_THROWS_WITH_AS(f.get_int("missing"), doctest::Contains("missing"), LoadError);
  CHECK_THROWS_AS(f.get_int("general.architecture"), LoadError);
}

TEST_CASE("GgufFile locates tensor data with default and custom alignment") {
  for (uint32_t alignment : {32u, 64u}) {
    test::GgufWriter w;
    if (alignment != 32) w.set_alignment(alignment);
    w.add_tensor("a", 0, {3, 2}, test::bytes_of(std::vector<float>{1, 2, 3, 4, 5, 6}));
    w.add_tensor("b", 0, {1}, test::bytes_of(std::vector<float>{7}));
    const auto path = test::temp_path("aligned.gguf");
    w.write(path);

    GgufFile f(path);
    CHECK(f.alignment() == alignment);
    CHECK(f.data_offset() % alignment == 0);
    REQUIRE(f.tensors().size() == 2);
    const GgufTensorInfo& a = f.tensors()[0];
    CHECK(a.name == "a");
    CHECK((a.dims == std::vector<int64_t>{3, 2}));
    CHECK(reinterpret_cast<const float*>(f.tensor_data(a))[5] == 6.0f);
    CHECK(reinterpret_cast<const float*>(f.tensor_data(f.tensors()[1]))[0] == 7.0f);
  }
}

TEST_CASE("GgufFile rejects corrupt files and names the path") {
  SUBCASE("bad magic") {
    const auto path = test::temp_path("bad_magic.gguf");
    std::ofstream(path, std::ios::binary) << "NOPE and some more bytes to read";
    CHECK_THROWS_WITH_AS(GgufFile{path}, doctest::Contains(path.string().c_str()), LoadError);
  }
  SUBCASE("truncated header") {
    test::GgufWriter w;
    w.add_string("general.architecture", "qwen3");
    const auto path = test::temp_path("truncated.gguf");
    w.write(path);
    std::filesystem::resize_file(path, 30);
    CHECK_THROWS_WITH_AS(GgufFile{path}, doctest::Contains("truncated"), LoadError);
  }
  SUBCASE("tensor data past end of file") {
    test::GgufWriter w;
    w.add_tensor("big", 0, {1024}, test::bytes_of(std::vector<float>(1024, 1.0f)));
    const auto path = test::temp_path("short_data.gguf");
    w.write(path);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 64);
    CHECK_THROWS_WITH_AS(GgufFile{path}, doctest::Contains("'big'"), LoadError);
  }
  SUBCASE("missing file") {
    CHECK_THROWS_AS(GgufFile{test::temp_path("does_not_exist.gguf")}, LoadError);
  }
}

TEST_CASE("GgufFile rejects tensor dims whose element count overflows") {
  test::GgufWriter w;
  w.add_tensor("huge", 0, {1ull << 40, 1ull << 40}, test::bytes_of(std::vector<float>{1.0f}));
  const auto path = test::temp_path("overflow.gguf");
  w.write(path);
  CHECK_THROWS_WITH_AS(GgufFile{path}, doctest::Contains("'huge'"), LoadError);
}
