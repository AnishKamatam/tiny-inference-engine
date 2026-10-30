#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "engine/engine.h"

namespace tie {

// What the bench runner records for one request; times are seconds after the run starts.
struct RequestTiming {
  double arrival = 0.0;
  int32_t prompt_len = 0;
  std::vector<double> token_times;
};

struct BenchRun {
  std::vector<RequestTiming> requests;
  double wall_seconds = 0.0;
  EngineStats stats;
};

struct Percentiles {
  double p50 = 0.0, p90 = 0.0, p99 = 0.0;
};

struct BenchReport {
  int64_t requests = 0, prompt_tokens = 0, output_tokens = 0, steps = 0, preemptions = 0;
  double wall_seconds = 0.0, requests_per_second = 0.0, output_tokens_per_second = 0.0, total_tokens_per_second = 0.0;
  double mean_seqs_per_step = 0.0;
  Percentiles ttft_ms;  // arrival to first token
  Percentiles itl_ms;   // gap between consecutive tokens of a request
  Percentiles e2e_ms;   // arrival to last token
};

// Linear interpolation between closest ranks, p in [0, 100]; 0 for no values.
double percentile(std::vector<double> values, double p);
BenchReport summarize(const BenchRun& run);
std::string format_report(const BenchReport& report);
nlohmann::json report_json(const BenchReport& report);

}  // namespace tie
