#include "bench/report.h"

#include <algorithm>
#include <format>
#include <nlohmann/json.hpp>

#include "core/error.h"

namespace tie {

namespace {

Percentiles percentiles_ms(const std::vector<double>& seconds) {
  std::vector<double> ms;
  ms.reserve(seconds.size());
  for (double s : seconds) ms.push_back(s * 1e3);
  return {percentile(ms, 50), percentile(ms, 90), percentile(ms, 99)};
}

nlohmann::json to_json(const Percentiles& p) { return {{"p50", p.p50}, {"p90", p.p90}, {"p99", p.p99}}; }

}  // namespace

double percentile(std::vector<double> values, double p) {
  if (!(p >= 0.0 && p <= 100.0)) fail<InvalidArgument>("percentile p must be in [0, 100], got {}", p);
  if (values.empty()) return 0.0;
  std::ranges::sort(values);
  const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<size_t>(rank);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (rank - static_cast<double>(lo));
}

BenchReport summarize(const BenchRun& run) {
  BenchReport r;
  std::vector<double> ttft, itl, e2e;
  for (const RequestTiming& t : run.requests) {
    ++r.requests;
    r.prompt_tokens += t.prompt_len;
    r.output_tokens += static_cast<int64_t>(t.token_times.size());
    if (t.token_times.empty()) continue;
    ttft.push_back(t.token_times.front() - t.arrival);
    e2e.push_back(t.token_times.back() - t.arrival);
    for (size_t i = 1; i < t.token_times.size(); ++i) itl.push_back(t.token_times[i] - t.token_times[i - 1]);
  }
  r.wall_seconds = run.wall_seconds;
  if (run.wall_seconds > 0) {
    r.requests_per_second = double(r.requests) / run.wall_seconds;
    r.output_tokens_per_second = double(r.output_tokens) / run.wall_seconds;
    r.total_tokens_per_second = double(r.output_tokens + r.prompt_tokens) / run.wall_seconds;
  }
  r.steps = run.stats.steps;
  r.preemptions = run.stats.preemptions;
  r.mean_seqs_per_step = run.stats.steps > 0 ? double(run.stats.seqs_scheduled) / double(run.stats.steps) : 0.0;
  r.ttft_ms = percentiles_ms(ttft);
  r.itl_ms = percentiles_ms(itl);
  r.e2e_ms = percentiles_ms(e2e);
  return r;
}

std::string format_report(const BenchReport& r) {
  const auto row = [](const char* name, const Percentiles& p) {
    return std::format("  {:<5} ms   p50 {:>9.1f}   p90 {:>9.1f}   p99 {:>9.1f}\n", name, p.p50, p.p90, p.p99);
  };
  return std::format("requests {}  |  wall {:.2f} s  |  {:.2f} req/s\n", r.requests, r.wall_seconds,
                     r.requests_per_second) +
         std::format("tokens   {} prompt + {} output  |  output {:.1f} tok/s  |  total {:.1f} tok/s\n", r.prompt_tokens,
                     r.output_tokens, r.output_tokens_per_second, r.total_tokens_per_second) +
         row("ttft", r.ttft_ms) + row("itl", r.itl_ms) + row("e2e", r.e2e_ms) +
         std::format("steps    {}  |  {:.2f} sequences/step  |  {} preemptions\n", r.steps, r.mean_seqs_per_step,
                     r.preemptions);
}

nlohmann::json report_json(const BenchReport& r) {
  return {{"requests", r.requests},
          {"prompt_tokens", r.prompt_tokens},
          {"output_tokens", r.output_tokens},
          {"wall_s", r.wall_seconds},
          {"requests_per_s", r.requests_per_second},
          {"output_tok_per_s", r.output_tokens_per_second},
          {"total_tok_per_s", r.total_tokens_per_second},
          {"ttft_ms", to_json(r.ttft_ms)},
          {"itl_ms", to_json(r.itl_ms)},
          {"e2e_ms", to_json(r.e2e_ms)},
          {"mean_seqs_per_step", r.mean_seqs_per_step},
          {"steps", r.steps},
          {"preemptions", r.preemptions}};
}

}  // namespace tie
