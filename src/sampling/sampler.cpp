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
  const float max_raw = *std::max_element(logits.begin(), logits.end());
  if (max_raw == -INFINITY) fail<InvalidArgument>("all logits are -inf");

  // A subnormal temperature has a non-finite reciprocal; scaling would produce NaN, so it is greedy.
  const float inv_temperature = p.temperature == 0.0f ? INFINITY : 1.0f / p.temperature;
  if (!std::isfinite(inv_temperature)) {
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }

  candidates_.clear();
  candidates_.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const float scaled = logits[i] * inv_temperature;
    // A +inf logit (or one that overflows when scaled) is certain; exp(inf - inf) would be NaN.
    if (scaled == INFINITY) return static_cast<int32_t>(i);
    candidates_.emplace_back(scaled, static_cast<int32_t>(i));
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
