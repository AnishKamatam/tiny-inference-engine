#include <doctest/doctest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/error.h"
#include "support/test_data.h"
#include "tokenizer/chat_template.h"

using namespace tie;

TEST_CASE("apply_chat_template renders a single user turn") {
  const std::vector<ChatMessage> msgs = {{"user", "Hi"}};
  CHECK(apply_chat_template(msgs, true) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
  CHECK(apply_chat_template(msgs, false) ==
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

TEST_CASE("apply_chat_template rejects empty conversations and unknown roles") {
  CHECK_THROWS_AS(apply_chat_template(std::vector<ChatMessage>{}, true), InvalidArgument);
  CHECK_THROWS_AS(apply_chat_template(std::vector<ChatMessage>{{"tool", "x"}}, true), InvalidArgument);
}

TEST_CASE("apply_chat_template matches HuggingFace") {
  const auto path = test::fixtures_dir() / "chat_cases.jsonl";
  TIE_REQUIRE_DATA(path);
  std::ifstream f(path);
  int cases = 0;
  for (std::string line; std::getline(f, line); ++cases) {
    const auto j = nlohmann::json::parse(line);
    std::vector<ChatMessage> msgs;
    for (const auto& m : j.at("messages")) {
      msgs.push_back({m.at("role").get<std::string>(), m.at("content").get<std::string>()});
    }
    CAPTURE(line);
    CHECK(apply_chat_template(msgs, j.at("enable_thinking").get<bool>()) == j.at("text").get<std::string>());
  }
  CHECK(cases == 5);
}
