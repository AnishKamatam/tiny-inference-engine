#include <doctest/doctest.h>

#include <string_view>
#include <vector>

#include "cli/args.h"
#include "core/error.h"

using namespace tie;
using Args = std::vector<std::string_view>;

TEST_CASE("parse_generate_args reads every option") {
  const Args args = {"-m", "model.gguf", "-p", "hello", "--chat", "--no-think", "--device", "cpu",
                     "--max-tokens", "64", "--temp", "0", "--top-k", "5", "--top-p", "0.9",
                     "--seed", "11", "--kv-mem", "512M", "--threads", "4"};
  const GenerateOptions o = parse_generate_args(args);
  CHECK(o.model == "model.gguf");
  CHECK(o.prompt == "hello");
  CHECK(o.chat);
  CHECK_FALSE(o.thinking);
  CHECK(o.sampling.max_tokens == 64);
  CHECK(o.sampling.temperature == 0.0f);
  CHECK(o.sampling.top_k == 5);
  CHECK(o.sampling.top_p == doctest::Approx(0.9));
  CHECK(o.sampling.seed == 11);
  CHECK(o.kv_bytes == size_t{512} << 20);
  CHECK(o.threads == 4);
}

TEST_CASE("parse_generate_args applies defaults and rejects bad input") {
  const GenerateOptions o = parse_generate_args(Args{"--model", "m", "--prompt", "p"});
  CHECK(o.device == "metal");
  CHECK(o.sampling.temperature == doctest::Approx(0.7));
  CHECK(o.kv_bytes == size_t{4} << 30);

  CHECK_THROWS_WITH_AS(parse_generate_args(Args{"-p", "x"}), doctest::Contains("--model"), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--bogus"}), doctest::Contains("--bogus"),
                       InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p"}), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--temp", "warm"}), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--top-p", "2"}), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--temp", "inf"}), doctest::Contains("--temp"),
                       InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--temp", "nan"}), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--top-p", "-inf"}), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--device", "tpu"}), InvalidArgument);
}

TEST_CASE("parse_size understands binary suffixes") {
  CHECK(parse_size("4G") == size_t{4} << 30);
  CHECK(parse_size("512M") == size_t{512} << 20);
  CHECK(parse_size("64K") == size_t{64} << 10);
  CHECK(parse_size("1000") == 1000);
  CHECK_THROWS_AS(parse_size("0"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("1.5G"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("G"), InvalidArgument);
}
