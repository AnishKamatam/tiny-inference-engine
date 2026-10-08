#include "tokenizer/chat_template.h"

#include <string_view>

#include "core/error.h"

namespace tie {

namespace {

// Python's str.lstrip("\n") / rstrip("\n") / strip("\n"), which the Jinja template uses.
std::string_view lstrip_newlines(std::string_view s) {
  while (!s.empty() && s.front() == '\n') s.remove_prefix(1);
  return s;
}

std::string_view rstrip_newlines(std::string_view s) {
  while (!s.empty() && s.back() == '\n') s.remove_suffix(1);
  return s;
}

bool is_blank(std::string_view s) { return s.find_first_not_of(" \t\n\r\f\v") == std::string_view::npos; }

void append_turn(std::string& out, std::string_view role, std::string_view content) {
  out.append("<|im_start|>").append(role).append("\n").append(content).append("<|im_end|>\n");
}

}  // namespace

std::string apply_chat_template(std::span<const ChatMessage> messages, bool enable_thinking) {
  if (messages.empty()) fail<InvalidArgument>("a chat needs at least one message");

  // Reasoning is only replayed for assistant turns after the last user query.
  size_t last_query = messages.size() - 1;
  for (size_t i = messages.size(); i-- > 0;) {
    if (messages[i].role == "user") {
      last_query = i;
      break;
    }
  }

  std::string out;
  for (size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (m.role == "system" || m.role == "user") {
      append_turn(out, m.role, m.content);
      continue;
    }
    if (m.role != "assistant") fail<InvalidArgument>("unsupported chat role '{}'", m.role);

    std::string_view content = m.content;
    std::string_view reasoning;
    if (const size_t close = content.rfind("</think>"); close != std::string_view::npos) {
      const std::string_view full = m.content;
      content = lstrip_newlines(full.substr(close + std::string_view("</think>").size()));
      reasoning = rstrip_newlines(full.substr(0, full.find("</think>")));
      if (const size_t open = reasoning.rfind("<think>"); open != std::string_view::npos) {
        reasoning = reasoning.substr(open + std::string_view("<think>").size());
      }
      reasoning = lstrip_newlines(reasoning);
    }

    out.append("<|im_start|>assistant\n");
    const bool replay_reasoning = i > last_query && (i + 1 == messages.size() || !is_blank(reasoning));
    if (replay_reasoning) {
      out.append("<think>\n").append(rstrip_newlines(lstrip_newlines(reasoning))).append("\n</think>\n\n");
      out.append(lstrip_newlines(content));
    } else {
      out.append(content);
    }
    out.append("<|im_end|>\n");
  }

  out.append("<|im_start|>assistant\n");
  if (!enable_thinking) out.append("<think>\n\n</think>\n\n");
  return out;
}

}  // namespace tie
