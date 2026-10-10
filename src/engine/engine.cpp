#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <span>

#include "core/error.h"

namespace tie {

Engine::Engine(Qwen3Model& model, PagedKVCache& cache, std::unique_ptr<Scheduler> scheduler, EngineConfig config)
    : model_(model), cache_(cache), scheduler_(std::move(scheduler)), config_(std::move(config)) {
  if (config_.max_tokens_per_step > model_.limits().max_tokens_per_step) {
    fail<InvalidArgument>("engine step size {} exceeds the model's limit of {}", config_.max_tokens_per_step,
                          model_.limits().max_tokens_per_step);
  }
  if (model_.limits().max_logit_rows < model_.limits().max_seqs) {
    fail<InvalidArgument>("the model must allow one logit row per sequence");
  }
}

int32_t Engine::add_request(std::vector<int32_t> prompt, const SamplingParams& params) {
  validate(params);
  if (prompt.empty()) fail<InvalidArgument>("prompt is empty");
  const int64_t vocab = model_.config().vocab_size;
  for (int32_t id : prompt) {
    if (id < 0 || id >= vocab) fail<InvalidArgument>("prompt token {} outside vocabulary of {}", id, vocab);
  }
  // Most tokens (prompt included) one sequence can hold: the context, the whole KV pool, one block-table row.
  const int64_t block_size = cache_.block_size();
  const int64_t context_cap = model_.config().max_position_embeddings;
  const int64_t pool_cap = static_cast<int64_t>(cache_.num_blocks()) * block_size;
  const int64_t row_cap = static_cast<int64_t>(model_.limits().max_blocks_per_seq) * block_size;
  const auto prompt_len = static_cast<int64_t>(prompt.size());
  if (prompt_len >= context_cap) {
    fail<CapacityError>("prompt of {} tokens leaves no room to generate in the model context of {}", prompt_len, context_cap);
  }
  if (prompt_len >= pool_cap) {
    fail<CapacityError>("prompt of {} tokens plus one generated token needs {} KV blocks but the cache holds {}; raise --kv-mem",
                        prompt_len, PagedKVCache::blocks_for(prompt_len + 1, cache_.block_size()), cache_.num_blocks());
  }
  if (prompt_len >= row_cap) {
    fail<CapacityError>("prompt of {} tokens needs {} KV blocks but the model allows {} blocks per sequence", prompt_len,
                        PagedKVCache::blocks_for(prompt_len + 1, cache_.block_size()), model_.limits().max_blocks_per_seq);
  }
  const int64_t capacity = std::min({context_cap, pool_cap, row_cap});

  auto seq = std::make_unique<Sequence>();
  seq->id = next_id_++;
  seq->prompt_len = static_cast<int32_t>(prompt.size());
  seq->tokens = std::move(prompt);
  seq->params = params;
  seq->params.max_tokens = static_cast<int32_t>(std::min<int64_t>(params.max_tokens, capacity - prompt_len));
  seq->rng.seed(params.seed);
  scheduler_->add(seq.get());
  const int32_t id = seq->id;
  sequences_.emplace(id, std::move(seq));
  return id;
}

void Engine::step() {
  const ScheduledBatch batch = scheduler_->schedule();
  if (batch.empty()) return;

  for (auto* v : {&tokens_, &positions_, &slots_, &query_start_, &context_lens_, &block_tables_, &logit_rows_}) v->clear();
  sampled_.clear();
  const int32_t stride = model_.limits().max_blocks_per_seq;
  bool has_prefill = false;
  int32_t rows = 0;
  query_start_.push_back(0);

  try {
    for (const ScheduledSeq& s : batch.seqs) {
      Sequence& seq = *s.seq;
      const int32_t begin = seq.num_computed;
      const int32_t end = begin + s.num_tokens;
      cache_.append_slots(seq.block_table, begin, end, slots_);
      tokens_.insert(tokens_.end(), seq.tokens.begin() + begin, seq.tokens.begin() + end);
      for (int32_t p = begin; p < end; ++p) positions_.push_back(p);
      rows += s.num_tokens;
      query_start_.push_back(rows);
      context_lens_.push_back(end);
      block_tables_.resize(block_tables_.size() + static_cast<size_t>(stride), 0);
      std::copy(seq.block_table.begin(), seq.block_table.end(), block_tables_.end() - stride);
      // A sequence samples once its whole pending input is in the cache.
      if (end == static_cast<int32_t>(seq.tokens.size())) {
        logit_rows_.push_back(rows - 1);
        sampled_.push_back(&seq);
      }
      has_prefill |= begin < seq.prompt_len;
    }

    const auto start = std::chrono::steady_clock::now();
    model_.forward(BatchInput{tokens_, positions_, slots_, query_start_, context_lens_, block_tables_, logit_rows_}, cache_,
                   logits_);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    (has_prefill ? stats_.prefill_tokens : stats_.decode_tokens) += static_cast<int64_t>(tokens_.size());
    (has_prefill ? stats_.prefill_seconds : stats_.decode_seconds) += seconds;
  } catch (...) {
    // Give back blocks reserved for a step that never ran, so the engine stays consistent.
    for (const ScheduledSeq& s : batch.seqs) {
      BlockTable& table = s.seq->block_table;
      const auto keep = static_cast<size_t>(PagedKVCache::blocks_for(s.seq->num_computed, cache_.block_size()));
      while (table.size() > keep) {
        cache_.pool().release(table.back());
        table.pop_back();
      }
    }
    throw;
  }

  for (const ScheduledSeq& s : batch.seqs) s.seq->num_computed += s.num_tokens;

  const auto vocab = static_cast<size_t>(model_.config().vocab_size);
  for (size_t i = 0; i < sampled_.size(); ++i) {
    Sequence& seq = *sampled_[i];
    int32_t token = 0;
    try {
      token = sampler_.sample(std::span(logits_).subspan(i * vocab, vocab), seq.params, seq.rng);
      seq.tokens.push_back(token);
      seq.state = SeqState::Decoding;
      if (on_token_) on_token_(seq.id, token);
    } catch (...) {
      // A sequence that cannot continue must not wedge the scheduler or leak its blocks.
      finish(seq, FinishReason::Error);
      throw;
    }
    if (std::ranges::find(config_.stop_tokens, token) != config_.stop_tokens.end()) {
      finish(seq, FinishReason::Stop);
    } else if (seq.num_generated() >= seq.params.max_tokens) {
      finish(seq, FinishReason::Length);
    }
  }
}

void Engine::finish(Sequence& seq, FinishReason reason) {
  seq.state = SeqState::Finished;
  seq.finish_reason = reason;
  cache_.release(seq.block_table);
  scheduler_->finish(&seq);
}

const Sequence& Engine::sequence(int32_t id) const {
  const auto it = sequences_.find(id);
  if (it == sequences_.end()) fail<InvalidArgument>("no sequence with id {}", id);
  return *it->second;
}

void Engine::release(int32_t id) {
  if (sequence(id).state != SeqState::Finished) fail<InvalidArgument>("sequence {} is still running", id);
  sequences_.erase(id);
}

std::vector<int32_t> Engine::generate(std::vector<int32_t> prompt, const SamplingParams& params) {
  const int32_t id = add_request(std::move(prompt), params);
  while (sequence(id).state != SeqState::Finished) step();
  const Sequence& seq = sequence(id);
  std::vector<int32_t> out(seq.tokens.begin() + seq.prompt_len, seq.tokens.end());
  release(id);
  return out;
}

}  // namespace tie
