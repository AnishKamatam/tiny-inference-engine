#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
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

TEST_CASE("parse_generate_args rejects out-of-range and malformed numbers") {
  const auto with = [](std::string_view flag, std::string_view v) { return Args{"-m", "m", "-p", "x", flag, v}; };
  CHECK_THROWS_AS(parse_generate_args(with("--max-tokens", "99999999999")), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(with("--seed", "-1")), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(with("--top-k", "+5")), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(with("--top-k", "5x")), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_generate_args(with("--top-k", "2.5")), doctest::Contains("integer"), InvalidArgument);
  CHECK(parse_generate_args(with("--seed", "18446744073709551615")).sampling.seed == UINT64_MAX);
  CHECK_THROWS_WITH_AS(parse_generate_args(with("--temp", "1e39")), doctest::Contains("--temp"), InvalidArgument);
  CHECK_THROWS_AS(parse_generate_args(Args{"-m", "m", "-p", "x", "--threads"}), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_generate_args(with("--kv-mem", "1Q")), doctest::Contains("--kv-mem"), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_generate_args(with("--kv-mem", "99999999999999999999")), doctest::Contains("--kv-mem"),
                       InvalidArgument);
}

TEST_CASE("parse_size understands binary suffixes") {
  CHECK(parse_size("4G") == size_t{4} << 30);
  CHECK(parse_size("512M") == size_t{512} << 20);
  CHECK(parse_size("64K") == size_t{64} << 10);
  CHECK(parse_size("1000") == 1000);
  CHECK_THROWS_AS(parse_size("0"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("1.5G"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("G"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("99999999999999999999"), InvalidArgument);
  CHECK_THROWS_AS(parse_size("17179869184G"), InvalidArgument);
}

TEST_CASE("parse_bench_args reads a synthetic workload and serving options") {
  const Args args = {"-m", "m.gguf", "--synthetic", "32", "--prompt-len", "512:1024", "--output-len", "128",
                     "--rate", "4.5", "--max-seqs", "16", "--max-tokens-per-step", "256", "--policy", "fcfs",
                     "--aging", "4", "--seed", "3", "--device", "cpu", "--kv-mem", "1G", "--json"};
  const BenchOptions o = parse_bench_args(args);
  CHECK(o.model == "m.gguf");
  CHECK(o.synthetic == 32);
  CHECK(o.prompt_len.lo == 512);
  CHECK(o.prompt_len.hi == 1024);
  CHECK(o.output_len.lo == 128);
  CHECK(o.output_len.hi == 128);
  CHECK(o.rate == doctest::Approx(4.5));
  CHECK(o.serving.max_seqs == 16);
  CHECK(o.serving.max_tokens_per_step == 256);
  CHECK(o.policy == OrderingPolicy::Fcfs);
  CHECK(o.aging == 4);
  CHECK(o.sampling.seed == 3);
  CHECK(o.sampling.temperature == 0.0f);  // greedy by default
  CHECK(o.device == "cpu");
  CHECK(o.kv_bytes == size_t{1} << 30);
  CHECK(o.json);
}

TEST_CASE("parse_bench_args defaults and rejections") {
  const BenchOptions o = parse_bench_args(Args{"-m", "m", "--prompts", "w.jsonl"});
  CHECK(o.prompts == "w.jsonl");
  CHECK(std::isinf(o.rate));
  CHECK(o.policy == OrderingPolicy::SrptAging);
  CHECK(o.serving.max_seqs == 64);
  CHECK(o.max_tokens == 128);
  CHECK(parse_bench_args(Args{"-m", "m", "--synthetic", "1", "--rate", "inf"}).rate == INFINITY);

  CHECK_THROWS_WITH_AS(parse_bench_args(Args{"-m", "m"}), doctest::Contains("--prompts or --synthetic"),
                       InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--prompts", "w", "--synthetic", "2"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"--synthetic", "2"}), InvalidArgument);  // no model
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "0"}), InvalidArgument);
  CHECK_THROWS_WITH_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--prompt-len", "9:3"}),
                       doctest::Contains("--prompt-len"), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--prompt-len", "0:3"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--rate", "0"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--policy", "lifo"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--max-seqs", "600"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "--aging", "-1"}), InvalidArgument);
  CHECK_THROWS_AS(parse_bench_args(Args{"-m", "m", "--synthetic", "2", "-p", "x"}), InvalidArgument);
}
