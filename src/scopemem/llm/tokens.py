"""Exact token counting with the backbone tokenizer, after applying the chat template.

- ``pref_tokens``  = tokens(prompt with memory block) - tokens(same prompt without it), so the
  chat template is accounted for exactly (D-012).
- ``total_input_tokens`` = tokens(full templated prompt incl. generation prompt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .engine import Messages

# Generic ChatML-style template used for non-HF counters (tests / smoke / laptop).
_SIMPLE_TEMPLATE_OPEN = "<|im_start|>{role}\n"
_SIMPLE_TEMPLATE_CLOSE = "<|im_end|>\n"
_SIMPLE_GEN_PROMPT = "<|im_start|>assistant\n"


class TokenCounter(Protocol):
    name: str

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Messages) -> int: ...


def _render_simple(messages: Messages) -> str:
    s = "".join(_SIMPLE_TEMPLATE_OPEN.format(role=m["role"]) + m["content"] + _SIMPLE_TEMPLATE_CLOSE
                for m in messages)
    return s + _SIMPLE_GEN_PROMPT


class WhitespaceCounter:
    """Whitespace tokens; chat-template markers count as tokens too."""

    name = "whitespace"

    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_messages(self, messages: Messages) -> int:
        return len(_render_simple(messages).replace("<|im_start|>", " <|im_start|> ")
                   .replace("<|im_end|>", " <|im_end|> ").split())


class TokenizersJsonCounter:
    """A ``tokenizers`` tokenizer.json (used for the tiny test fixture) with the simple template."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path))
        self.name = f"tokjson:{Path(path).name}"

    def count_text(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_messages(self, messages: Messages) -> int:
        return len(self._tok.encode(_render_simple(messages), add_special_tokens=False).ids)


class HFCounter:
    """Backbone tokenizer from the local HF cache, with its own chat template."""

    def __init__(self, hf_id: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(hf_id)
        self.name = f"hf:{hf_id}"

    def count_text(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_messages(self, messages: Messages) -> int:
        ids = self._tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids)


def build_counter(cfg: Any) -> TokenCounter:
    kind = str(cfg.tokens.counter)
    if kind == "whitespace":
        return WhitespaceCounter()
    if kind == "hf":
        return HFCounter(str(cfg.model_cfg.hf_id))
    return TokenizersJsonCounter(kind)


def pref_token_delta(counter: TokenCounter, with_block: Messages, without_block: Messages) -> int:
    return counter.count_messages(with_block) - counter.count_messages(without_block)
