# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy"]
# ///
"""Dump HuggingFace fp32 reference outputs for tie's tests.

Usage: uv run tools/dump_reference.py <hf_model_dir> <out_dir>

Writes:
  tokenizer_cases.jsonl   {"text", "ids"} per corpus string
  chat_cases.jsonl        {"messages", "enable_thinking", "text"}
  parity_ids.npy          int32 [T]            prompt token ids
  parity_hidden_{i}.npy   float32 [T, hidden]  output of decoder layer i
  parity_logits.npy       float32 [T, vocab]   final logits
  greedy.json             {"prompt_ids", "output_ids"} greedy continuation
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZER_CORPUS = [
    "", "Hello world", "Hello, world!", "  leading spaces", "trailing spaces   ",
    "tabs\tand\nnewlines\r\n", "\n\n\n", "a\n\n b", "multiple     spaces between",
    "I'm you're we've they'll she'd it's I'M YOU'RE", "don't won't can't",
    "café", "café", "naïve façade", "Straße", "ÅÄÖ åäö",
    "数字123", "你好，世界！", "日本語のテキスト", "한국어 텍스트", "Привет, мир",
    "مرحبا بالعالم", "שלום עולם", "हिन्दी पाठ", "ไทย", "Ελληνικά",
    "emoji 😀🎉👍🏽 family 👨‍👩‍👧‍👦", "flags 🇺🇸🇯🇵",
    "numbers 1234567890 3.14159 -42 1e10", "1,000,000", "phone: +1 (555) 123-4567",
    "def foo(x):\n    return x * 2\n", "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "<html><body>hi</body></html>", '{"key": "value", "n": [1, 2, 3]}',
    "https://example.com/path?q=1&r=2#frag", "user@example.com",
    "<|im_start|>user\nhi<|im_end|>", "<think>reasoning</think>answer",
    "<|endoftext|>", "text<|im_end|>more", "<tool_call>{}</tool_call>",
    "!!!???...", "---===+++", "a" * 300, "ab " * 50,
    "Mixed CASE and lower and UPPER", "snake_case camelCase PascalCase kebab-case",
    " non-breaking space", "zero​width", "　ideographic space",
    "combining àéî", "ﬁ ligature", "Ⅻ roman", "½ ¾ ²",
]

CHAT_CASES = [
    ([{"role": "user", "content": "Hi"}], True),
    ([{"role": "user", "content": "Hi"}], False),
    ([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}], True),
    ([{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "<think>\nx\n</think>\n\nYo"},
      {"role": "user", "content": "Q"}], True),
    ([{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
      {"role": "user", "content": "C"}], False),
]

PARITY_PROMPT = "The quick brown fox jumps over the lazy dog. In 1969, humans first"
GREEDY_PROMPT = [{"role": "user", "content": "Name three primary colors."}]
GREEDY_NEW_TOKENS = 32


def main() -> None:
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)

    with open(out_dir / "tokenizer_cases.jsonl", "w") as f:
        for text in TOKENIZER_CORPUS:
            ids = tok.encode(text, add_special_tokens=False)
            f.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")

    with open(out_dir / "chat_cases.jsonl", "w") as f:
        for messages, thinking in CHAT_CASES:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking)
            f.write(json.dumps({"messages": messages, "enable_thinking": thinking, "text": text},
                               ensure_ascii=False) + "\n")

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    hidden: dict[int, np.ndarray] = {}

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            hidden[i] = out[0].detach().float().numpy()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.model.layers)]
    ids = tok.encode(PARITY_PROMPT, add_special_tokens=False)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()
    for h in handles:
        h.remove()

    np.save(out_dir / "parity_ids.npy", np.asarray(ids, dtype=np.int32))
    np.save(out_dir / "parity_logits.npy", logits.astype(np.float32))
    for i, h in hidden.items():
        np.save(out_dir / f"parity_hidden_{i}.npy", h.astype(np.float32))

    prompt_ids = tok.apply_chat_template(GREEDY_PROMPT, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
    if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids]), max_new_tokens=GREEDY_NEW_TOKENS,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    output_ids = out[0, len(prompt_ids):].tolist()
    with open(out_dir / "greedy.json", "w") as f:
        json.dump({"prompt_ids": list(prompt_ids), "output_ids": output_ids}, f)

    print(f"wrote reference fixtures to {out_dir}: {len(ids)} parity tokens, "
          f"{len(hidden)} layers, {len(output_ids)} greedy tokens")


if __name__ == "__main__":
    main()
