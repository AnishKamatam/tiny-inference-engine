#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

using namespace tie;
using Pieces = std::vector<std::string>;

namespace {

Pieces split(std::string_view s) {
  const auto views = pretokenize_qwen2(s);
  return {views.begin(), views.end()};
}

}  // namespace

TEST_CASE("codepoint classes and UTF-8 helpers") {
  CHECK(is_letter('a'));
  CHECK(is_letter(0x4E2D));  // 中
  CHECK(is_letter(0x00E9));  // é
  CHECK(is_number('7'));
  CHECK(is_number(0x00BD));  // ½ (No)
  CHECK(is_whitespace(' '));
  CHECK(is_whitespace(0x3000));
  CHECK_FALSE(is_letter('!'));
  CHECK_FALSE(is_whitespace(0x200B));  // zero-width space is not White_Space
  CHECK_FALSE(is_letter(0x110000));

  size_t offset = 0;
  CHECK(utf8_decode("é", offset) == 0xE9);
  CHECK(offset == 2);
  offset = 0;
  CHECK(utf8_decode("\xff", offset) == kReplacementChar);
  CHECK(offset == 1);
  offset = 0;
  CHECK(utf8_decode("\xe2\x82", offset) == kReplacementChar);  // truncated
  CHECK(offset == 1);
  CHECK(utf8_encode(0x1F600) == "😀");
  CHECK(utf8_encode(0xE9) == "é");
}

// Every expectation below was produced with Python's `regex` module and Qwen's pattern.
TEST_CASE("pretokenize_qwen2 matches the reference regex") {
  CHECK((split("Hello world") == Pieces{"Hello", " world"}));
  CHECK((split("I'm here, aren't you?") == Pieces{"I", "'m", " here", ",", " aren", "'t", " you", "?"}));
  CHECK((split("we're you've they'll") == Pieces{"we", "'re", " you", "'ve", " they", "'ll"}));
  CHECK((split("DON'T") == Pieces{"DON", "'T"}));
  CHECK((split("  a\n\n b") == Pieces{" ", " a", "\n\n", " b"}));
  CHECK((split("a\r\nb") == Pieces{"a", "\r\n", "b"}));
  CHECK((split(" \n x") == Pieces{" \n", " x"}));
  CHECK((split("end   ") == Pieces{"end", "   "}));
  CHECK((split("x\t\ty") == Pieces{"x", "\t", "\ty"}));
  CHECK((split("tab\there") == Pieces{"tab", "\there"}));
  CHECK((split("x  = 1;\n") == Pieces{"x", " ", " =", " ", "1", ";\n"}));
  CHECK((split("数字123") == Pieces{"数字", "1", "2", "3"}));
  CHECK((split("emoji 😀!") == Pieces{"emoji", " 😀!"}));
  CHECK((split("(café) ¿qué?") == Pieces{"(café", ")", " ¿", "qué", "?"}));
  CHECK(split("").empty());
}

TEST_CASE("pretokenize_qwen2 pieces concatenate back to the input, even for invalid UTF-8") {
  const std::string input = "\xff\xfe" "abc \xc3 tail\xe2\x82";
  std::string joined;
  for (std::string_view piece : pretokenize_qwen2(input)) joined += piece;
  CHECK(joined == input);
}
