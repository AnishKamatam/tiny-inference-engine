#pragma once

#include <span>
#include <string>

namespace tie {

struct ChatMessage {
  std::string role;  // "system", "user" or "assistant"
  std::string content;
};

// Renders `messages` with Qwen3's ChatML template and appends the assistant
// generation prompt. With enable_thinking=false an empty think block is added,
// exactly as Qwen3's template does. Tool calls are not supported yet.
std::string apply_chat_template(std::span<const ChatMessage> messages, bool enable_thinking);

}  // namespace tie
