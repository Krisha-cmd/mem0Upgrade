"""PersonaMem loader (Jiang et al., 2025, arXiv:2504.14225). Release: HF dataset
``bowen-upenn/PersonaMem-v1`` (MIT). Verified files (see docs/DATASETS.md):
  questions_{32k,128k,1M}.csv     one row per MCQ question
  shared_contexts_{32k,128k,1M}.jsonl  one line per context: {context_id: [ {role, content}, ... ]}
A question's context is ``shared_contexts[shared_context_id][: end_index_in_shared_context + 1]``.
The first message of a context is a system message carrying the persona.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

_LETTER = re.compile(r"^\(?([a-z])\)?", re.I)


class PersonaMemQ(BaseModel):
    question_id: str
    persona_id: str
    question_type: str
    topic: str
    question: str
    options: list[str]
    correct: str            # letter, e.g. "c"
    context_id: str
    end_index: int
    context_tokens: int


def _parse_options(s: str) -> list[str]:
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        v = ast.literal_eval(s)
    return [str(x) for x in v]


def load_questions(root: str | Path, split: str = "32k") -> list[PersonaMemQ]:
    path = Path(root) / f"questions_{split}.csv"
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            m = _LETTER.match(r["correct_answer"].strip())
            out.append(PersonaMemQ(
                question_id=r["question_id"], persona_id=str(r["persona_id"]),
                question_type=r["question_type"], topic=r["topic"],
                question=r["user_question_or_message"], options=_parse_options(r["all_options"]),
                correct=(m.group(1).lower() if m else r["correct_answer"].strip()),
                context_id=r["shared_context_id"],
                end_index=int(r["end_index_in_shared_context"]),
                context_tokens=int(float(r["context_length_in_tokens"] or 0))))
    return out


def iter_contexts(root: str | Path, split: str = "32k") -> Iterator[tuple[str, list[dict]]]:
    with open(Path(root) / f"shared_contexts_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for k, v in d.items():
                yield k, v


def load_contexts(root: str | Path, split: str = "32k") -> dict[str, list[dict]]:
    return dict(iter_contexts(root, split))


def context_for(q: PersonaMemQ, contexts: dict[str, list[dict]]) -> list[dict]:
    return contexts[q.context_id][: q.end_index + 1]


def option_letter(i: int) -> str:
    return "abcdefghijklmnopqrstuvwxyz"[i]
