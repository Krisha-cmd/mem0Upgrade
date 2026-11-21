"""LoCoMo loader (Maharana et al., 2024). Release: github.com/snap-research/locomo, data/locomo10.json
(10 conversations; licence file not machine-detectable — see docs/DATASETS.md).
Verified structure: list of {sample_id, conversation{speaker_a, speaker_b, session_<i>,
session_<i>_date_time}, qa[{question, answer, evidence, category}], ...}.
QA categories 1-5; following the Mem0 paper protocol, category 5 (adversarial) is excluded (verify
against the Mem0 evaluation code at Phase 7).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

_SESSION = re.compile(r"^session_(\d+)$")


class LocomoTurn(BaseModel):
    speaker: str
    dia_id: str
    text: str


class LocomoSession(BaseModel):
    index: int
    date_time: str
    turns: list[LocomoTurn]


class LocomoQA(BaseModel):
    question: str
    answer: str
    category: int
    evidence: list[str]


class LocomoConv(BaseModel):
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[LocomoSession]
    qa: list[LocomoQA]


def load_locomo(path: str | Path, exclude_categories: tuple[int, ...] = (5,)) -> list[LocomoConv]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        c = r["conversation"]
        sessions = []
        for k, v in c.items():
            m = _SESSION.match(k)
            if m and isinstance(v, list):
                i = int(m.group(1))
                sessions.append(LocomoSession(
                    index=i, date_time=str(c.get(f"session_{i}_date_time", "")),
                    turns=[LocomoTurn(speaker=t["speaker"], dia_id=t["dia_id"], text=t.get("text", ""))
                           for t in v]))
        sessions.sort(key=lambda s: s.index)
        qa = [LocomoQA(question=q["question"], answer=str(q.get("answer", q.get("adversarial_answer", ""))),
                       category=int(q["category"]), evidence=[str(e) for e in q.get("evidence", [])])
              for q in r["qa"] if int(q["category"]) not in exclude_categories]
        out.append(LocomoConv(sample_id=str(r["sample_id"]), speaker_a=c["speaker_a"],
                              speaker_b=c["speaker_b"], sessions=sessions, qa=qa))
    return out
