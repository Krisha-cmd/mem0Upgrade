"""B5: the backbone reads a numbered list of all preferences and returns applicable indices.
All selector tokens (prompt + completion) count as overhead."""

from __future__ import annotations

import json
import re
from typing import Sequence

from ..llm.engine import GenParams
from ..prompts import load_prompt
from .base import Gate, Selection

_LIST = re.compile(r"\[[\d,\s]*\]")


def parse_indices(text: str, n: int) -> list[int] | None:
    m = _LIST.search(text)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return sorted({int(i) for i in v if isinstance(i, int) and 1 <= i <= n})


class LLMSelectGate(Gate):
    def __init__(self, cond, gate_cfg, llm, max_tokens: int = 64):
        self.id = str(cond.id)
        self.llm = llm
        self.template = load_prompt("llm_select")
        self.params = GenParams(temperature=0.0, max_tokens=max_tokens)

    def _messages(self, query, prefs):
        numbered = "\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(prefs))
        return [{"role": "user", "content": self.template.format(numbered=numbered, query=query)}]

    def select_many(self, queries: Sequence[str], prefs_lists, instances) -> list[Selection]:
        msgs = [self._messages(q, p) for q, p in zip(queries, prefs_lists)]
        outs = self.llm.generate(msgs, self.params)
        res = []
        for prefs, c in zip(prefs_lists, outs):
            idx = parse_indices(c.text, len(prefs))
            sel = [prefs[i - 1].id for i in (idx or [])]
            res.append(Selection(sel, c.prompt_tokens + c.completion_tokens, None, False,
                                 {"parse_ok": idx is not None}))
        return res
