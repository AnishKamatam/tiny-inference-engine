#include <doctest/doctest.h>

#include <cmath>
#include <sstream>

#include <nlohmann/json.hpp>

#include "bench/report.h"
#include "bench/workload.h"
#include "core/error.h"

using namespace tie;

TEST_CASE("parse_prompts_jsonl reads prompts, optional max_tokens and chat, and names bad lines") {
  std::istringstream in(R"({"prompt": "hello", "max_tokens": 7, "chat": true}

{"prompt": "world"}
)");
  const auto lines = parse_prompts_jsonl(in, "w.jsonl");
  REQUIRE(lines.size() == 2);
  CHECK(lines[0].text == "hello");
  CHECK(lines[0].max_tokens == 7);
  CHECK(lines[0].chat);
  CHECK(lines[1].text == "world");
  CHECK_FALSE(lines[1].max_tokens.has_value());
  CHECK_FALSE(lines[1].chat);

  std::istringstream bad_json("{\"prompt\": \"a\"}\n{oops\n");
  CHECK_THROWS_WITH_AS(parse_prompts_jsonl(bad_json, "w.jsonl"), doctest::Contains("w.jsonl:2"), InvalidArgument);
  std::istringstream no_prompt("{\"max_tokens\": 3}\n");
  CHECK_THROWS_WITH_AS(parse_prompts_jsonl(no_prompt, "w.jsonl"), doctest::Contains("prompt"), InvalidArgument);
  std::istringstream zero("{\"prompt\": \"a\", \"max_tokens\": 0}\n");
  CHECK_THROWS_WITH_AS(parse_prompts_jsonl(zero, "w.jsonl"), doctest::Contains("max_tokens"), InvalidArgument);
  std::istringstream empty("\n\n");
  CHECK_THROWS_WITH_AS(parse_prompts_jsonl(empty, "w.jsonl"), doctest::Contains("no prompts"), InvalidArgument);
}

TEST_CASE("synthetic_requests is deterministic per seed and respects the length ranges") {
  const auto a = synthetic_requests(50, {5, 9}, {2, 4}, 100, 7);
  const auto b = synthetic_requests(50, {5, 9}, {2, 4}, 100, 7);
  const auto c = synthetic_requests(50, {5, 9}, {2, 4}, 100, 8);
  REQUIRE(a.size() == 50);
  bool differs = false;
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].prompt == b[i].prompt);
    CHECK(a[i].max_tokens == b[i].max_tokens);
    CHECK(a[i].prompt.size() >= 5);
    CHECK(a[i].prompt.size() <= 9);
    CHECK(a[i].max_tokens >= 2);
    CHECK(a[i].max_tokens <= 4);
    for (int32_t t : a[i].prompt) CHECK((t >= 0 && t < 100));
    differs |= a[i].prompt != c[i].prompt;
  }
  CHECK(differs);
}

TEST_CASE("assign_arrivals: all at zero for an infinite rate, increasing Poisson times otherwise") {
  auto reqs = synthetic_requests(200, {1, 1}, {1, 1}, 10, 1);
  assign_arrivals(reqs, INFINITY, 1);
  for (const auto& r : reqs) CHECK(r.arrival == 0.0);
  assign_arrivals(reqs, 50.0, 1);
  for (size_t i = 1; i < reqs.size(); ++i) CHECK(reqs[i].arrival >= reqs[i - 1].arrival);
  CHECK(reqs.back().arrival == doctest::Approx(200.0 / 50.0).epsilon(0.3));  // mean gap 1/rate
  CHECK_THROWS_AS(assign_arrivals(reqs, 0.0, 1), InvalidArgument);
}

TEST_CASE("percentile interpolates linearly") {
  CHECK(percentile({}, 50) == 0.0);
  CHECK(percentile({4.0}, 99) == 4.0);
  CHECK(percentile({1, 2, 3, 4}, 50) == doctest::Approx(2.5));
  CHECK(percentile({4, 3, 2, 1}, 0) == 1.0);
  CHECK(percentile({1, 2, 3, 4}, 100) == 4.0);
  CHECK(percentile({1, 2, 3, 4, 5}, 90) == doctest::Approx(4.6));
}

TEST_CASE("summarize computes throughput and latency from per-token timings") {
  BenchRun run;
  run.wall_seconds = 2.0;
  run.requests = {{0.0, 10, {0.1, 0.2, 0.3}}, {0.5, 20, {1.0, 1.5}}};
  run.stats.steps = 4;
  run.stats.seqs_scheduled = 6;
  run.stats.preemptions = 1;
  const BenchReport r = summarize(run);
  CHECK(r.requests == 2);
  CHECK(r.output_tokens == 5);
  CHECK(r.prompt_tokens == 30);
  CHECK(r.output_tokens_per_second == doctest::Approx(2.5));
  CHECK(r.total_tokens_per_second == doctest::Approx(17.5));
  CHECK(r.requests_per_second == doctest::Approx(1.0));
  CHECK(r.ttft_ms.p50 == doctest::Approx(300.0));  // ttfts 100 ms and 500 ms
  CHECK(r.e2e_ms.p50 == doctest::Approx(650.0));   // 300 ms and 1000 ms
  CHECK(r.itl_ms.p50 == doctest::Approx(100.0));   // gaps 100, 100, 500 ms
  CHECK(r.mean_seqs_per_step == doctest::Approx(1.5));
  CHECK(r.preemptions == 1);
  const auto j = report_json(r);
  for (const char* key : {"requests", "wall_s", "requests_per_s", "output_tok_per_s", "total_tok_per_s", "ttft_ms",
                          "itl_ms", "e2e_ms", "mean_seqs_per_step", "steps", "preemptions"}) {
    CHECK(j.contains(key));
  }
  CHECK(j["ttft_ms"].contains("p99"));
  CHECK(format_report(r).find("ttft") != std::string::npos);
}

TEST_CASE("parse_prompts_jsonl rejects an empty prompt, naming the line") {
  std::istringstream in("{\"prompt\": \"a\"}\n{\"prompt\": \"\"}\n");
  CHECK_THROWS_WITH_AS(parse_prompts_jsonl(in, "w.jsonl"), doctest::Contains("w.jsonl:2"), InvalidArgument);
}

TEST_CASE("percentile rejects p outside [0, 100] and NaN") {
  CHECK_THROWS_AS(percentile({1, 2, 3}, 100.5), InvalidArgument);
  CHECK_THROWS_AS(percentile({1, 2, 3}, -1), InvalidArgument);
  CHECK_THROWS_AS(percentile({1, 2, 3}, NAN), InvalidArgument);
  CHECK_THROWS_AS(percentile({}, 101), InvalidArgument);
}

TEST_CASE("synthetic_requests validates its arguments like the parser does") {
  CHECK_THROWS_AS(synthetic_requests(0, {1, 2}, {1, 2}, 10, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(-1, {1, 2}, {1, 2}, 10, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(1, {1, 2}, {1, 2}, 0, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(1, {0, 2}, {1, 2}, 10, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(1, {3, 2}, {1, 2}, 10, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(1, {1, 2}, {0, 2}, 10, 1), InvalidArgument);
  CHECK_THROWS_AS(synthetic_requests(1, {1, 2}, {3, 2}, 10, 1), InvalidArgument);
}
