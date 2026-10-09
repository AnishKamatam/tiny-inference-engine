#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "kv/paged_kv_cache.h"
#include "model/qwen3.h"

namespace tie::test {

// Feeds one sequence through the model step by step, managing its block table.
class SequenceRunner {
 public:
  SequenceRunner(Qwen3Model& model, PagedKVCache& cache) : model_(model), cache_(cache) {}
  ~SequenceRunner() { cache_.release(table_); }

  // Runs `tokens` as the next step. Returns logits for every row when
  // all_logits is true, otherwise only for the last token.
  std::vector<float> step(std::span<const int32_t> tokens, bool all_logits = false) {
    const auto n = static_cast<int32_t>(tokens.size());
    std::vector<int32_t> positions, slots, rows;
    for (int32_t i = 0; i < n; ++i) positions.push_back(len_ + i);
    cache_.append_slots(table_, len_, len_ + n, slots);
    if (all_logits) {
      for (int32_t i = 0; i < n; ++i) rows.push_back(i);
    } else {
      rows.push_back(n - 1);
    }
    std::vector<int32_t> block_row(static_cast<size_t>(model_.limits().max_blocks_per_seq), 0);
    std::copy(table_.begin(), table_.end(), block_row.begin());
    const std::vector<int32_t> query_start = {0, n};
    const std::vector<int32_t> context_lens = {len_ + n};

    std::vector<float> logits;
    model_.forward(BatchInput{tokens, positions, slots, query_start, context_lens, block_row, rows}, cache_, logits);
    len_ += n;
    return logits;
  }

 private:
  Qwen3Model& model_;
  PagedKVCache& cache_;
  BlockTable table_;
  int32_t len_ = 0;
};

}  // namespace tie::test
