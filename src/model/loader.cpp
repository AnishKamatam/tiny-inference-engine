#include "model/loader.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>

#include "core/error.h"
#include "model/gguf.h"
#include "model/safetensors.h"

namespace tie {

namespace {

// Qwen2/Qwen3 pre-tokenizer pattern; tokenizer/pretokenize.cpp implements exactly this.
constexpr std::string_view kQwen2SplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

constexpr std::pair<std::string_view, std::string_view> kGgufGlobalNames[] = {
    {"token_embd.weight", "model.embed_tokens.weight"},
    {"output_norm.weight", "model.norm.weight"},
    {"output.weight", "lm_head.weight"},
};

constexpr std::pair<std::string_view, std::string_view> kGgufLayerNames[] = {
    {"attn_norm.weight", "input_layernorm.weight"},
    {"attn_q.weight", "self_attn.q_proj.weight"},
    {"attn_k.weight", "self_attn.k_proj.weight"},
    {"attn_v.weight", "self_attn.v_proj.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_q_norm.weight", "self_attn.q_norm.weight"},
    {"attn_k_norm.weight", "self_attn.k_norm.weight"},
    {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"},
    {"ffn_up.weight", "mlp.up_proj.weight"},
    {"ffn_down.weight", "mlp.down_proj.weight"},
};

// Unknown names pass through unchanged so strict weight binding reports them by their file name.
std::string hf_name(std::string_view gguf) {
  for (const auto& [g, h] : kGgufGlobalNames) {
    if (gguf == g) return std::string(h);
  }
  if (gguf.starts_with("blk.")) {
    const size_t dot = gguf.find('.', 4);
    if (dot != std::string_view::npos) {
      const std::string_view layer = gguf.substr(4, dot - 4);
      const std::string_view suffix = gguf.substr(dot + 1);
      for (const auto& [g, h] : kGgufLayerNames) {
        if (suffix == g) return std::format("model.layers.{}.{}", layer, h);
      }
    }
  }
  return std::string(gguf);
}

ModelConfig config_from_gguf(const GgufFile& f, bool has_output_weight) {
  const std::string where = f.path().string();
  ModelConfig c;
  c.architecture = f.get_string("general.architecture");
  if (c.architecture != "qwen3") {
    fail<UnsupportedError>("{}: architecture '{}' is not supported (supported: qwen3)", where, c.architecture);
  }
  const auto key = [&](std::string_view k) { return std::format("{}.{}", c.architecture, k); };
  c.num_layers = f.get_int(key("block_count"));
  c.hidden_size = f.get_int(key("embedding_length"));
  c.intermediate_size = f.get_int(key("feed_forward_length"));
  c.num_heads = f.get_int(key("attention.head_count"));
  c.num_kv_heads = f.get_int(key("attention.head_count_kv"));
  c.max_position_embeddings = f.get_int(key("context_length"));
  c.head_dim = f.has(key("attention.key_length")) ? f.get_int(key("attention.key_length")) : c.hidden_size / c.num_heads;
  if (f.has(key("attention.value_length")) && f.get_int(key("attention.value_length")) != c.head_dim) {
    fail<UnsupportedError>("{}: key and value head sizes differ", where);
  }
  c.rms_norm_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")));
  c.rope_theta = static_cast<float>(f.get_float(key("rope.freq_base")));
  if (f.has(key("rope.scaling.type")) && f.get_string(key("rope.scaling.type")) != "none") {
    fail<UnsupportedError>("{}: rope scaling '{}' is not supported", where, f.get_string(key("rope.scaling.type")));
  }
  c.vocab_size = static_cast<int64_t>(f.get_strings("tokenizer.ggml.tokens").size());
  c.tie_word_embeddings = !has_output_weight;
  c.validate();
  return c;
}

TokenizerData tokenizer_from_gguf(const GgufFile& f) {
  const std::string where = f.path().string();
  if (f.get_string("tokenizer.ggml.model") != "gpt2") {
    fail<UnsupportedError>("{}: tokenizer model '{}' (supported: gpt2 byte-level BPE)", where,
                           f.get_string("tokenizer.ggml.model"));
  }
  const std::string pre = f.has("tokenizer.ggml.pre") ? f.get_string("tokenizer.ggml.pre") : "";
  if (pre != "qwen2") fail<UnsupportedError>("{}: pre-tokenizer '{}' (supported: qwen2)", where, pre);

  TokenizerData t;
  t.tokens = f.get_strings("tokenizer.ggml.tokens");
  const std::vector<int64_t>& types = f.get_ints("tokenizer.ggml.token_type");
  if (types.size() != t.tokens.size()) fail<LoadError>("{}: token_type and tokens differ in length", where);
  constexpr int64_t kControl = 3, kUserDefined = 4, kUnused = 5;
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == kControl || types[i] == kUserDefined) t.special_ids.push_back(static_cast<int32_t>(i));
    if (types[i] == kUnused) t.tokens[i].clear();
  }
  for (const std::string& m : f.get_strings("tokenizer.ggml.merges")) {
    const size_t sp = m.find(' ');
    if (sp == std::string::npos) fail<LoadError>("{}: merge '{}' has no separator", where, m);
    t.merges.emplace_back(m.substr(0, sp), m.substr(sp + 1));
  }
  return t;
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) fail<LoadError>("{}: cannot open", path.string());
  try {
    return nlohmann::json::parse(f);
  } catch (const nlohmann::json::exception& e) {
    fail<LoadError>("{}: invalid JSON: {}", path.string(), e.what());
  }
}

TokenizerData tokenizer_from_hf_json(const nlohmann::json& j, const std::filesystem::path& path) {
  const std::string where = path.string();
  try {
    const auto& model = j.at("model");
    if (model.at("type") != "BPE") fail<UnsupportedError>("{}: tokenizer type {} (supported: BPE)", where, model.at("type").dump());
    if (!j.at("normalizer").is_null() && j.at("normalizer").value("type", "") != "NFC") {
      fail<UnsupportedError>("{}: normalizer {} (supported: NFC)", where, j.at("normalizer").dump());
    }
    const auto& split = j.at("pre_tokenizer").at("pretokenizers").at(0);
    if (split.at("pattern").at("Regex").get<std::string>() != kQwen2SplitPattern) {
      fail<UnsupportedError>("{}: pre-tokenizer pattern is not Qwen2's", where);
    }

    TokenizerData t;
    int64_t max_id = -1;
    const auto checked_id = [&](const nlohmann::json& v, const std::string& token) {
      const int64_t id = v.get<int64_t>();
      if (id < 0 || id > INT32_MAX) fail<LoadError>("{}: token '{}' has invalid id {}", where, token, id);
      return id;
    };
    for (const auto& [token, id] : model.at("vocab").items()) max_id = std::max(max_id, checked_id(id, token));
    for (const auto& added : j.at("added_tokens")) {
      max_id = std::max(max_id, checked_id(added.at("id"), added.at("content").get<std::string>()));
    }
    t.tokens.resize(static_cast<size_t>(max_id + 1));
    for (const auto& [token, id] : model.at("vocab").items()) t.tokens[id.get<size_t>()] = token;
    for (const auto& added : j.at("added_tokens")) {
      const auto id = added.at("id").get<int32_t>();
      t.tokens[static_cast<size_t>(id)] = added.at("content").get<std::string>();
      t.special_ids.push_back(id);
    }
    for (const auto& m : model.at("merges")) {
      if (m.is_array()) {
        t.merges.emplace_back(m.at(0).get<std::string>(), m.at(1).get<std::string>());
      } else {
        const auto s = m.get<std::string>();
        const size_t sp = s.find(' ');
        if (sp == std::string::npos) fail<LoadError>("{}: merge '{}' has no separator", where, s);
        t.merges.emplace_back(s.substr(0, sp), s.substr(sp + 1));
      }
    }
    return t;
  } catch (const nlohmann::json::exception& e) {
    fail<LoadError>("{}: {}", where, e.what());
  }
}

}  // namespace

void TensorTable::add(std::string name, TensorInfo info) {
  const auto [it, inserted] = tensors_.emplace(std::move(name), info);
  if (!inserted) fail<LoadError>("duplicate tensor '{}'", it->first);
}

const TensorInfo* TensorTable::find(std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorInfo& TensorTable::at(std::string_view name) const {
  const TensorInfo* t = find(name);
  if (t == nullptr) fail<LoadError>("missing tensor '{}'", name);
  return *t;
}

std::vector<std::string> TensorTable::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_) out.push_back(name);
  return out;
}

LoadedModel load_gguf(const std::filesystem::path& path) {
  auto file = std::make_shared<GgufFile>(path);
  LoadedModel m;
  m.regions.push_back(file->file().bytes());
  m.owners.push_back(file);

  bool has_output_weight = false;
  for (const GgufTensorInfo& t : file->tensors()) {
    const std::string name = hf_name(t.name);
    const std::optional<DType> dtype = dtype_from_ggml(t.type);
    if (!dtype) {
      fail<UnsupportedError>("{}: tensor '{}' uses GGML type {} (supported: F32, F16, BF16, Q8_0)", path.string(),
                             name, t.type);
    }
    // GGUF lists the innermost dimension first; HF order is outermost first.
    const std::vector<int64_t> dims(t.dims.rbegin(), t.dims.rend());
    const Shape shape = Shape::from(dims);
    const size_t offset = file->data_offset() + t.offset;
    m.tensors.add(name, TensorInfo{*dtype, shape, 0, offset, bytes_for(*dtype, shape.numel()), file->file().data() + offset});
    has_output_weight |= name == "lm_head.weight";
  }
  m.config = config_from_gguf(*file, has_output_weight);
  m.tokenizer = tokenizer_from_gguf(*file);
  return m;
}

LoadedModel load_safetensors(const std::filesystem::path& dir) {
  LoadedModel m;
  m.config = config_from_hf_json(read_json(dir / "config.json"));
  m.tokenizer = tokenizer_from_hf_json(read_json(dir / "tokenizer.json"), dir / "tokenizer.json");

  std::set<std::string> files;
  if (const auto index = dir / "model.safetensors.index.json"; std::filesystem::exists(index)) {
    try {
      for (const auto& [tensor, file] : read_json(index).at("weight_map").items()) files.insert(file.get<std::string>());
    } catch (const nlohmann::json::exception& e) {
      fail<LoadError>("{}: {}", index.string(), e.what());
    }
  } else {
    files.insert("model.safetensors");
  }
  for (const std::string& name : files) {
    auto file = std::make_shared<SafetensorsFile>(dir / name);
    const size_t region = m.regions.size();
    m.regions.push_back(file->file().bytes());
    for (const SafetensorsTensor& t : file->tensors()) {
      m.tensors.add(t.name, TensorInfo{t.dtype, t.shape, region, t.file_offset, t.nbytes, t.data});
    }
    m.owners.push_back(std::move(file));
  }
  return m;
}

LoadedModel load_model(const std::filesystem::path& path) {
  if (std::filesystem::is_directory(path)) return load_safetensors(path);
  if (!std::filesystem::exists(path)) fail<LoadError>("{}: no such file or directory", path.string());
  if (path.extension() == ".gguf") return load_gguf(path);
  fail<UnsupportedError>("{}: expected a .gguf file or a directory containing config.json", path.string());
}

}  // namespace tie
