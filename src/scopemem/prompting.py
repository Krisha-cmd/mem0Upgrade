"""Final prompt = [system: fixed instruction (+ memory block)] + [history, B1 only] + [user: query].

The memory block template is identical across conditions; only its contents change. It is
appended to the system message (D-013). With no selected preferences the block is omitted.
"""

from __future__ import annotations

from typing import Any, Sequence

from .llm.engine import Messages
from .llm.tokens import TokenCounter


def memory_block(texts: Sequence[str], header: str = "User preferences:") -> str:
    if not texts:
        return ""
    return header + "\n" + "\n".join(f"- {t}" for t in texts)


def build_messages(cfg: Any, query: str, pref_texts: Sequence[str],
                   history: Messages | None = None) -> Messages:
    system = str(cfg.prompt.system)
    block = memory_block(pref_texts, str(cfg.prompt.memory_header))
    if block:
        system = system + "\n\n" + block
    msgs: Messages = [{"role": "system", "content": system}]
    if history:
        msgs += list(history)
    msgs.append({"role": "user", "content": query})
    return msgs


def truncate_history_left(cfg: Any, counter: TokenCounter, query: str, history: Messages,
                          budget: int) -> tuple[Messages, bool]:
    """Drop the oldest (user, assistant) exchanges until the prompt fits ``budget`` tokens."""
    h = list(history)
    if counter.count_messages(build_messages(cfg, query, [], h)) <= budget:
        return h, False
    # binary search on number of leading exchanges to drop (keeps pairs intact)
    n_ex = len(h) // 2
    lo, hi = 1, n_ex
    while lo < hi:
        mid = (lo + hi) // 2
        if counter.count_messages(build_messages(cfg, query, [], h[2 * mid:])) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return h[2 * lo:], True
