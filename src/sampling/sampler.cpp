#include "sampling/sampler.h"

#include <algorithm>
#include <cmath>

#include "core/error.h"

namespace tie {

void validate(const SamplingParams& p) {
  if (!(p.temperature >= 0.0f && std::isfinite(p.temperature)))
    fail<InvalidArgument>("temperature must be finite and >= 0, got {}", p.temperature);
  if (p.top_k < 0) fail<InvalidArgument>("top_k must be >= 0, got {}", p.top_k);
  if (!(p.top_p > 0.0f && p.top_p <= 1.0f)) fail<InvalidArgument>("top_p must be in (0, 1], got {}", p.top_p);
  if (p.max_tokens < 1) fail<InvalidArgument>("max_tokens must be >= 1, got {}", p.max_tokens);
}

int32_t Sampler::sample(std::span<const float> logits, const SamplingParams& p, std::mt19937_64& rng) {
  if (logits.empty()) fail<InvalidArgument>("cannot sample from empty logits");
  for (size_t i = 0; i < logits.size(); ++i) {
    if (std::isnan(logits[i])) fail<InvalidArgument>("logit {} is NaN", i);
  }
  // max_element returns the first maximum, so ties (including several +inf) go to the lowest id.
  const auto argmax = std::max_element(logits.begin(), logits.end());
  const float max_raw = *argmax;
  if (max_raw == -INFINITY) fail<InvalidArgument>("all logits are -inf");

  // A subnormal temperature has a non-finite reciprocal; scaling would produce NaN, so it is greedy.
  // A +inf logit is certain; exp(inf - inf) would be NaN.
  const float inv_temperature = p.temperature == 0.0f ? INFINITY : 1.0f / p.temperature;
  if (!std::isfinite(inv_temperature) || max_raw == INFINITY) return static_cast<int32_t>(argmax - logits.begin());

  // Shift before scaling so the maximum maps to 0 and every other logit to [-inf, 0]: a tiny
  // temperature can then neither overflow to +inf nor make every scaled logit -inf.
  candidates_.clear();
  candidates_.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    candidates_.emplace_back((logits[i] - max_raw) * inv_temperature, static_cast<int32_t>(i));
  }

  const auto more_likely = [](const auto& a, const auto& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  };
  size_t keep = candidates_.size();
  if (p.top_k > 0 && static_cast<size_t>(p.top_k) < keep) {
    keep = static_cast<size_t>(p.top_k);
    std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(),
                      more_likely);
  } else if (p.top_p < 1.0f) {
    std::sort(candidates_.begin(), candidates_.end(), more_likely);
  }

  // Softmax over the kept candidates (weights are left unnormalized).
  float max_logit = candidates_[0].first;
  for (size_t i = 1; i < keep; ++i) max_logit = std::max(max_logit, candidates_[i].first);
  double total = 0.0;
  for (size_t i = 0; i < keep; ++i) {
    candidates_[i].first = std::exp(candidates_[i].first - max_logit);
    total += candidates_[i].first;
  }

  // Candidates are sorted here: top_p < 1 always sorted them above.
  if (p.top_p < 1.0f) {
    double cumulative = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      cumulative += candidates_[i].first;
      if (cumulative >= p.top_p * total) {
        keep = i + 1;
        break;
      }
    }
    total = cumulative;
  }

  double r = std::uniform_real_distribution<double>(0.0, total)(rng);
  for (size_t i = 0; i < keep; ++i) {
    r -= candidates_[i].first;
    if (r < 0.0) return candidates_[i].second;
  }
  return candidates_[keep - 1].second;
}

}  // namespace tie
