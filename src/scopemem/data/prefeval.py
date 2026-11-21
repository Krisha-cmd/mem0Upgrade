"""Loader for the raw PrefEval release (github.com/amazon-science/PrefEval, CC BY-NC 4.0).

Verified layout (commit 50795054, see docs/DATASETS.md):
  benchmark_dataset/explicit_preference/<topic>.json           list[{preference, question, explanation}]
  benchmark_dataset/implicit_preference/choice-based/<topic>.json   + implicit_query, options, conversation, aligned_op
  benchmark_dataset/implicit_preference/persona-driven/<topic>.json + persona, conversation{"0": {user, assistant}, ...}
  benchmark_dataset/filtered_inter_turns.json                  list of 24 LMSYS-style filler conversations
20 topics, 1,000 explicit pairs (31-62 per topic); the implicit files reuse the same 1,000 pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import PrefItem

_IMPLICIT_DIRS = {"choice": "choice-based", "persona": "persona-driven"}


def _bench(root: str | Path) -> Path:
    root = Path(root)
    return root / "benchmark_dataset" if (root / "benchmark_dataset").exists() else root


def list_topics(root: str | Path) -> list[str]:
    return sorted(p.stem for p in (_bench(root) / "explicit_preference").glob("*.json"))


def load_items(root: str | Path, form: str = "explicit") -> list[PrefItem]:
    b = _bench(root)
    out: list[PrefItem] = []
    for topic in list_topics(root):
        path = (b / "explicit_preference" / f"{topic}.json" if form == "explicit"
                else b / "implicit_preference" / _IMPLICIT_DIRS[form] / f"{topic}.json")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for i, r in enumerate(rows):
            extra = {k: v for k, v in r.items() if k not in {"preference", "question", "explanation"}}
            out.append(PrefItem(id=f"{topic}/{i}", topic=topic, preference=r["preference"].strip(),
                                question=r["question"].strip(), explanation=r.get("explanation", ""),
                                form=form, extra=extra))
    return out


def load_filler(root: str | Path) -> list[list[tuple[str, str]]]:
    """Filler conversations as lists of (user, assistant) exchanges."""
    rows = json.loads((_bench(root) / "filtered_inter_turns.json").read_text(encoding="utf-8"))
    convs = []
    for r in rows:
        turns = r["conversation"]
        ex = []
        for j in range(0, len(turns) - 1, 2):
            u, a = turns[j], turns[j + 1]
            if u["role"] == "user" and a["role"] == "assistant":
                ex.append((u["content"], a["content"]))
        if ex:
            convs.append(ex)
    return convs


def implicit_history_messages(item: PrefItem) -> list[dict[str, str]]:
    """The implicit-form turns that *reveal* the preference (E8), as chat messages."""
    conv = item.extra.get("conversation")
    if item.form == "choice":
        return [{"role": "user", "content": conv["query"]},
                {"role": "assistant", "content": conv["assistant_options"]},
                {"role": "user", "content": conv["user_selection"]},
                {"role": "assistant", "content": conv["assistant_acknowledgment"]}]
    if item.form == "persona":
        msgs = []
        for k in sorted(conv, key=int):
            msgs += [{"role": "user", "content": conv[k]["user"]},
                     {"role": "assistant", "content": conv[k]["assistant"]}]
        return msgs
    raise ValueError(item.form)
