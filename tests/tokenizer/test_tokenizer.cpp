#include <doctest/doctest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/error.h"
#include "model/loader.h"
#include "support/test_data.h"
#include "tokenizer/tokenizer.h"

using namespace tie;
using Ids = std::vector<int32_t>;

namespace {

// A toy vocabulary: the 256 byte symbols, a few merges and one special token.
TokenizerData toy_data(bool ll_before_el) {
  TokenizerData d;
  for (const std::string& s : gpt2_byte_alphabet()) d.tokens.push_back(s);
  const auto add = [&](const std::string& t) {
    d.tokens.push_back(t);
    return static_cast<int32_t>(d.tokens.size() - 1);
  };
  add("he");
  add("ll");
  add("el");
  add("hell");
  add("hello");
  d.special_ids.push_back(add("<|x|>"));
  if (ll_before_el) {
    d.merges = {{"l", "l"}, {"e", "l"}, {"h", "e"}, {"he", "ll"}, {"hell", "o"}};
  } else {
    d.merges = {{"e", "l"}, {"l", "l"}, {"h", "e"}, {"he", "ll"}, {"hell", "o"}};
  }
  return d;
}

std::vector<std::string> pieces(const Tokenizer& t, const Ids& ids) {
  std::vector<std::string> out;
  for (int32_t id : ids) out.push_back(t.token_bytes(id));
  return out;
}

}  // namespace

TEST_CASE("Tokenizer merges by rank, not left to right") {
  const Tokenizer a(toy_data(/*ll_before_el=*/true));
  CHECK((pieces(a, a.encode("ell")) == std::vector<std::string>{"e", "ll"}));
  const Tokenizer b(toy_data(/*ll_before_el=*/false));
  CHECK((pieces(b, b.encode("ell")) == std::vector<std::string>{"el", "l"}));
}

TEST_CASE("Tokenizer matches special tokens verbatim and round-trips") {
  const Tokenizer t(toy_data(true));
  const Ids ids = t.encode("hello<|x|>hell");
  CHECK((pieces(t, ids) == std::vector<std::string>{"hello", "<|x|>", "hell"}));
  CHECK(t.decode(ids) == "hello<|x|>hell");
  CHECK(t.find_token("<|x|>").has_value());
  CHECK_FALSE(t.find_token("nope").has_value());
  CHECK_THROWS_AS(t.token_bytes(100000), InvalidArgument);
}

TEST_CASE("Tokenizer rejects a special token with empty text") {
  TokenizerData d = toy_data(true);
  d.tokens.push_back("");
  d.special_ids.push_back(static_cast<int32_t>(d.tokens.size() - 1));
  CHECK_THROWS_AS(Tokenizer{d}, LoadError);
}

TEST_CASE("Tokenizer round-trips invalid UTF-8 byte for byte") {
  const Tokenizer t(toy_data(true));
  const std::string input = "\xff\xfe ok \xc3";
  CHECK(t.decode(t.encode(input)) == input);
}

TEST_CASE("StreamDecoder never emits half a character") {
  const Tokenizer t(toy_data(true));
  StreamDecoder d(t);
  const std::string emoji = "😀";  // F0 9F 98 80, one byte token each in the toy vocabulary
  const Ids ids = t.encode(emoji);
  REQUIRE(ids.size() == 4);
  CHECK(d.push(ids[0]).empty());
  CHECK(d.push(ids[1]).empty());
  CHECK(d.push(ids[2]).empty());
  CHECK(d.push(ids[3]) == emoji);
  CHECK(d.push(t.encode("a")[0]) == "a");
  CHECK(d.push(ids[0]).empty());
  CHECK(d.flush() == "\xf0");
}

TEST_CASE("nfc_normalize composes and leaves invalid UTF-8 alone") {
  CHECK(nfc_normalize("café") == "café");
  CHECK(nfc_normalize("plain") == "plain");
  CHECK(nfc_normalize("\xff\xfe") == "\xff\xfe");
}

TEST_CASE("Tokenizer matches HuggingFace on the reference corpus") {
  const auto cases_path = test::fixtures_dir() / "tokenizer_cases.jsonl";
  const auto gguf_path = test::models_dir() / "Qwen3-0.6B-BF16.gguf";
  const auto st_path = test::models_dir() / "Qwen3-0.6B";
  TIE_REQUIRE_DATA(cases_path);
  TIE_REQUIRE_DATA(gguf_path);
  TIE_REQUIRE_DATA(st_path);

  const Tokenizer from_gguf(load_gguf(gguf_path).tokenizer);
  const Tokenizer from_json(load_safetensors(st_path).tokenizer);
  CHECK(from_gguf.find_token("<|im_end|>") == 151645);

  std::ifstream f(cases_path);
  int cases = 0;
  for (std::string line; std::getline(f, line); ++cases) {
    const auto j = nlohmann::json::parse(line);
    const auto text = j.at("text").get<std::string>();
    const auto want = j.at("ids").get<Ids>();
    CAPTURE(text);
    CHECK(from_gguf.encode(text) == want);
    CHECK(from_json.encode(text) == want);
    CHECK(from_gguf.decode(want) == nfc_normalize(text));
  }
  CHECK(cases > 50);
}
