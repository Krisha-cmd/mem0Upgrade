"""Deterministic FakeLLM for tests and smoke runs (no GPU, no network).

Output depends only on (name, messages, params) through a hash, so reruns are identical.
It recognises the project's own prompt templates and answers in the right format:
- scope writer  -> valid scope JSON derived from the preference words
- LLM selector  -> JSON list of indices whose preference shares words with the query
- PrefEval judge prompts -> well-formed XML
Tests can override behaviour with ``script`` (a queue of raw outputs) or ``handler``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from typing import Callable, Sequence

from .engine import Completion, GenParams, Messages

_WORD = re.compile(r"[a-z]+")
_STOP = set("""a an the i me my we you your to of and or in on for with at by from is are be been
am was were it this that these those as not no do does don't dont any all prefer like love hate
really very much more most would could should can will just only so but if when about what which
who how please want need into than then also have has had its im i'm get""".split())

_GLOBAL_HINTS = ("every answer", "every reply", "all answers", "always answer", "always format",
                 "in every", "all your", "your answers", "your responses", "your replies", "every response")


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class FakeLLM:
    def __init__(self, name: str = "fake", script: Sequence[str] | None = None,
                 handler: Callable[[Messages], str | None] | None = None):
        self.name = name
        self.script = deque(script or [])
        self.handler = handler
        self.calls = 0

    # --- public API ---------------------------------------------------------------
    def generate(self, prompts: Sequence[Messages], params: GenParams) -> list[Completion]:
        out = []
        for msgs in prompts:
            self.calls += 1
            text = self._respond(msgs, params)
            ptoks = sum(len(m["content"].split()) + 1 for m in msgs)
            out.append(Completion(text=text, prompt_tokens=ptoks,
                                  completion_tokens=len(text.split())))
        return out

    # --- behaviour ----------------------------------------------------------------
    def _respond(self, msgs: Messages, params: GenParams) -> str:
        if self.script:
            return self.script.popleft()
        if self.handler:
            r = self.handler(msgs)
            if r is not None:
                return r
        full = "\n".join(m["content"] for m in msgs)
        if "SCOPE_WRITER_V1" in full:
            return self._scope(full)
        if "LLM_SELECT_V1" in full:
            return self._select(full)
        if "<answer>" in full and "Examine" in full:
            return self._judge(full)
        if "FALSE_APPLICATION_V1" in full:
            return '{"applied_irrelevant": false, "which": []}'
        hx = _h(self.name, full, json.dumps(params.key()))
        n = 20 + int(hx[:2], 16) % 40
        return " ".join(f"tok{hx[i % 60:i % 60 + 4]}" for i in range(n))

    @staticmethod
    def _field(full: str, tag: str) -> str:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", full, re.S)
        return m.group(1).strip() if m else ""

    def _scope(self, full: str) -> str:
        pref = self._field(full, "preference_text")
        words = content_words(pref)[:6] or ["general"]
        always = any(h in pref.lower() for h in _GLOBAL_HINTS)
        obj = {
            "preference": pref,
            "scope": "Requests involving " + ", ".join(words[:4]) + ".",
            "example_requests": [f"Help me with {w}" for w in (words * 3)[:3]],
            "always_on": always,
        }
        return json.dumps(obj)

    def _select(self, full: str) -> str:
        query = self._field(full, "request")
        items = re.findall(r"^\[(\d+)\]\s*(.+)$", self._field(full, "preferences"), re.M)
        q = set(content_words(query))
        picked = [int(i) for i, t in items if q & set(content_words(t))]
        return json.dumps(picked[:5])

    def _judge(self, full: str) -> str:
        hx = _h(self.name, full)
        yes = "Yes" if int(hx[:2], 16) % 2 else "No"
        if "<preference>" in full and "acknowledges" in full:
            return f"<preference></preference>\n<answer>{yes}</answer>"
        return f"<explanation>fake judgement</explanation>\n<answer>{yes}</answer>"
