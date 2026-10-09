#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/error.h"
#include "sampling/sampler.h"

using namespace tie;

namespace {

const std::vector<float> kLogits = {std::log(0.7f), std::log(0.2f), std::log(0.1f)};

std::vector<double> frequencies(const SamplingParams& p, int draws) {
  Sampler sampler;
  std::mt19937_64 rng(p.seed);
  std::vector<double> freq(kLogits.size(), 0.0);
  for (int i = 0; i < draws; ++i) freq[size_t(sampler.sample(kLogits, p, rng))] += 1.0 / draws;
  return freq;
}

}  // namespace

TEST_CASE("temperature 0 is greedy and breaks ties toward the lowest id") {
  Sampler sampler;
  std::mt19937_64 rng(0);
  const SamplingParams greedy;
  CHECK(sampler.sample(std::vector<float>{0.1f, 2.0f, -1.0f}, greedy, rng) == 1);
  CHECK(sampler.sample(std::vector<float>{3.0f, 1.0f, 3.0f}, greedy, rng) == 0);
}

TEST_CASE("sampling at temperature 1 follows the softmax") {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 42;
  const auto f = frequencies(p, 20000);
  CHECK(f[0] == doctest::Approx(0.7).epsilon(0.03));
  CHECK(f[1] == doctest::Approx(0.2).epsilon(0.1));
  CHECK(f[2] == doctest::Approx(0.1).epsilon(0.15));
}

TEST_CASE("top-k and top-p remove the tail") {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 7;

  p.top_k = 2;
  auto f = frequencies(p, 5000);
  CHECK(f[2] == 0.0);
  CHECK(f[0] == doctest::Approx(0.7 / 0.9).epsilon(0.05));

  p.top_k = 0;
  p.top_p = 0.75f;  // 0.7 alone falls short of 0.75, so {0.7, 0.2} survive
  f = frequencies(p, 5000);
  CHECK(f[2] == 0.0);
  CHECK(f[1] > 0.0);

  p.top_p = 0.01f;  // only the most likely token survives
  f = frequencies(p, 500);
  CHECK(f[0] == doctest::Approx(1.0));  // 500 sums of 1/500 are not exactly 1.0
  CHECK(f[1] == 0.0);
  CHECK(f[2] == 0.0);
}

TEST_CASE("the same seed reproduces the same draws") {
  SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 3;
  CHECK(frequencies(p, 200) == frequencies(p, 200));
}

TEST_CASE("validate rejects impossible parameters") {
  const auto with = [](auto mutate) {
    SamplingParams p;
    mutate(p);
    return p;
  };
  CHECK_NOTHROW(validate(SamplingParams{}));
  CHECK_THROWS_AS(validate(with([](SamplingParams& p) { p.temperature = -0.1f; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([](SamplingParams& p) { p.top_k = -1; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([](SamplingParams& p) { p.top_p = 0.0f; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([](SamplingParams& p) { p.top_p = 1.5f; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([](SamplingParams& p) { p.max_tokens = 0; })), InvalidArgument);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(validate(with([=](SamplingParams& p) { p.temperature = nan; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([=](SamplingParams& p) { p.temperature = inf; })), InvalidArgument);
  CHECK_THROWS_AS(validate(with([=](SamplingParams& p) { p.top_p = nan; })), InvalidArgument);
}

TEST_CASE("top-k and top-p combine") {
  const std::vector<float> logits = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 3;
  p.top_p = 0.7f;  // top 3 total 0.95; 0.5 < 0.665 <= 0.8, so only ids 0 and 1 survive
  Sampler sampler;
  std::mt19937_64 rng(11);
  bool seen0 = false, seen1 = false;
  for (int i = 0; i < 2000; ++i) {
    const int32_t t = sampler.sample(logits, p, rng);
    CHECK((t == 0 || t == 1));
    seen0 |= t == 0;
    seen1 |= t == 1;
  }
  CHECK(seen0);
  CHECK(seen1);
}

TEST_CASE("a subnormal temperature behaves like greedy") {
  Sampler sampler;
  std::mt19937_64 rng(0);
  SamplingParams p;
  p.temperature = 1e-40f;
  CHECK_NOTHROW(validate(p));
  CHECK(sampler.sample(std::vector<float>{0.0f, 1.0f, -2.0f}, p, rng) == 1);
  CHECK(sampler.sample(std::vector<float>{0.0f, -std::numeric_limits<float>::infinity(), 0.0f}, p, rng) == 0);
}

TEST_CASE("NaN logits are rejected on both paths and infinite logits are handled") {
  Sampler sampler;
  std::mt19937_64 rng(0);
  const std::vector<float> nan_logits = {1.0f, std::nanf(""), 2.0f};
  SamplingParams sampled;
  sampled.temperature = 1.0f;
  CHECK_THROWS_AS(sampler.sample(nan_logits, SamplingParams{}, rng), InvalidArgument);
  CHECK_THROWS_AS(sampler.sample(nan_logits, sampled, rng), InvalidArgument);

  // +inf is the certain token (lowest id among several), on both paths.
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> with_inf = {1.0f, inf, 2.0f, inf};
  CHECK(sampler.sample(with_inf, SamplingParams{}, rng) == 1);
  CHECK(sampler.sample(with_inf, sampled, rng) == 1);
  // -inf entries get zero probability; all -inf is an error.
  CHECK(sampler.sample(std::vector<float>{-inf, 0.0f, -inf}, sampled, rng) == 1);
  CHECK_THROWS_AS(sampler.sample(std::vector<float>{-inf, -inf}, sampled, rng), InvalidArgument);
}
