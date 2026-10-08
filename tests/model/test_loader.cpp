#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/error.h"
#include "model/loader.h"
#include "support/gguf_writer.h"
#include "support/test_data.h"

using namespace tie;

TEST_CASE("load_gguf rejects other architectures and tensor types") {
  SUBCASE("architecture") {
    test::GgufWriter w;
    w.add_string("general.architecture", "llama");
    const auto path = test::temp_path("llama.gguf");
    w.write(path);
    CHECK_THROWS_WITH_AS(load_gguf(path), doctest::Contains("llama"), UnsupportedError);
  }
  SUBCASE("tensor type") {
    test::GgufWriter w;
    w.add_string("general.architecture", "qwen3");
    w.add_tensor("blk.0.attn_q.weight", 12 /* Q4_K */, {256}, std::vector<uint8_t>(144));
    const auto path = test::temp_path("q4k.gguf");
    w.write(path);
    CHECK_THROWS_WITH_AS(load_gguf(path), doctest::Contains("model.layers.0.self_attn.q_proj.weight"),
                         UnsupportedError);
  }
}

TEST_CASE("load_model rejects paths that are neither GGUF nor a model directory") {
  CHECK_THROWS_AS(load_model(test::temp_path("nothing_here")), LoadError);
}

TEST_CASE("GGUF and safetensors describe the same Qwen3-0.6B") {
  const auto gguf_path = test::models_dir() / "Qwen3-0.6B-BF16.gguf";
  const auto st_path = test::models_dir() / "Qwen3-0.6B";
  TIE_REQUIRE_DATA(gguf_path);
  TIE_REQUIRE_DATA(st_path);

  const LoadedModel g = load_gguf(gguf_path);
  const LoadedModel s = load_safetensors(st_path);

  CHECK(g.config == s.config);
  CHECK(g.config.tie_word_embeddings);
  CHECK(g.config.vocab_size == 151936);

  // safetensors additionally carries lm_head.weight even though embeddings are tied.
  CHECK(g.tensors.size() == 310);
  CHECK(s.tensors.size() == 311);
  for (const std::string& name : g.tensors.names()) {
    CAPTURE(name);
    const TensorInfo& gt = g.tensors.at(name);
    const TensorInfo& st = s.tensors.at(name);
    CHECK(gt.shape == st.shape);
    if (gt.shape.ndim() == 2) {
      CHECK(gt.dtype == DType::BF16);
      CHECK(st.dtype == DType::BF16);
    }
  }
  // Same bytes prove the name map and the dimension reversal are right.
  const TensorInfo& gq = g.tensors.at("model.layers.0.self_attn.q_proj.weight");
  const TensorInfo& sq = s.tensors.at("model.layers.0.self_attn.q_proj.weight");
  CHECK((gq.shape == Shape{2048, 1024}));
  REQUIRE(gq.nbytes == sq.nbytes);
  CHECK(std::memcmp(gq.data, sq.data, gq.nbytes) == 0);
  CHECK(g.regions[gq.region].data() + gq.offset == gq.data);
}

TEST_CASE("GGUF and tokenizer.json yield the same tokenizer") {
  const auto gguf_path = test::models_dir() / "Qwen3-0.6B-BF16.gguf";
  const auto st_path = test::models_dir() / "Qwen3-0.6B";
  TIE_REQUIRE_DATA(gguf_path);
  TIE_REQUIRE_DATA(st_path);

  const TokenizerData g = load_gguf(gguf_path).tokenizer;
  const TokenizerData s = load_safetensors(st_path).tokenizer;
  CHECK(g.tokens.size() == 151936);  // padded to the embedding size
  CHECK(s.tokens.size() == 151669);
  for (size_t i = 0; i < s.tokens.size(); ++i) {
    if (g.tokens[i] != s.tokens[i]) FAIL_CHECK("token " << i << " differs");
  }
  CHECK(g.tokens[151700].empty());  // unused padding ids decode to nothing
  CHECK(g.merges == s.merges);
  auto gs = g.special_ids;
  auto ss = s.special_ids;
  std::sort(gs.begin(), gs.end());
  std::sort(ss.begin(), ss.end());
  CHECK(gs == ss);
  CHECK(gs.size() == 26);
}

TEST_CASE("Q8_0 GGUF loads with quantized matrices and F32 norms") {
  const auto path = test::models_dir() / "Qwen3-0.6B-Q8_0.gguf";
  TIE_REQUIRE_DATA(path);
  const LoadedModel m = load_gguf(path);
  CHECK(m.tensors.at("model.layers.0.self_attn.q_proj.weight").dtype == DType::Q8_0);
  CHECK(m.tensors.at("model.embed_tokens.weight").dtype == DType::Q8_0);
  CHECK(m.tensors.at("model.norm.weight").dtype == DType::F32);
}

namespace {

constexpr const char* kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::filesystem::path write_tiny_model_dir(const std::string& name, const std::string& added_tokens) {
  // temp_path() only removes plain files, so clear a directory left by an earlier run first.
  std::filesystem::remove_all(std::filesystem::temp_directory_path() / "tie_tests" / name);
  const auto dir = test::temp_path(name);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "config.json") << R"({
    "model_type": "qwen3", "vocab_size": 8, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
    "max_position_embeddings": 128, "rms_norm_eps": 1e-06, "rope_theta": 1000000,
    "tie_word_embeddings": true})";
  nlohmann::json tok = {{"model", {{"type", "BPE"}, {"vocab", {{"a", 0}, {"b", 1}}}, {"merges", nlohmann::json::array()}}},
                        {"normalizer", {{"type", "NFC"}}},
                        {"pre_tokenizer", {{"pretokenizers", {{{"pattern", {{"Regex", kQwen2Pattern}}}}}}}},
                        {"added_tokens", nlohmann::json::parse(added_tokens)}};
  std::ofstream(dir / "tokenizer.json") << tok.dump();
  return dir;
}

}  // namespace

TEST_CASE("load_safetensors raises LoadError for malformed json") {
  SUBCASE("index without weight_map") {
    const auto dir = write_tiny_model_dir("bad_index_dir", "[]");
    std::ofstream(dir / "model.safetensors.index.json") << "{}";
    CHECK_THROWS_WITH_AS(load_safetensors(dir), doctest::Contains("model.safetensors.index.json"), LoadError);
  }
  SUBCASE("negative added token id") {
    const auto dir = write_tiny_model_dir("neg_id_dir", R"([{"id": -1, "content": "<x>"}])");
    CHECK_THROWS_WITH_AS(load_safetensors(dir), doctest::Contains("<x>"), LoadError);
  }
}
