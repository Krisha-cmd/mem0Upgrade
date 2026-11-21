"""Mem0-style baselines: cosine(query, preference text) top-k (B3) and + cross-encoder rerank (B4)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..memory.store import PrefRecord
from .base import ScoringGate


class SimTopKGate(ScoringGate):
    mode = "topk"

    def __init__(self, cond, gate_cfg, embedder):
        super().__init__(cond, gate_cfg)
        self.embedder = embedder

    def score_many(self, queries: Sequence[str], prefs_lists: Sequence[Sequence[PrefRecord]]) -> list[np.ndarray]:
        qv = self.embedder.encode(list(queries), is_query=True)
        self.embedder.encode(list({p.text for prefs in prefs_lists for p in prefs}))   # batch warm-up
        out = []
        for i, prefs in enumerate(prefs_lists):
            if not prefs:
                out.append(np.zeros(0, dtype=np.float32))
                continue
            pv = self.embedder.encode([p.text for p in prefs])
            out.append(pv @ qv[i])
        return out


class SimRerankGate(ScoringGate):
    """Top-`rerank_top` by cosine, reranked by a cross-encoder on (query, preference text).

    Scores returned: CE score for the shortlisted prefs, -inf for the rest, so top-k on these
    scores is exactly "shortlist then rerank then keep k".
    """

    mode = "topk"

    def __init__(self, cond, gate_cfg, embedder, cross):
        super().__init__(cond, gate_cfg)
        self.embedder = embedder
        self.cross = cross
        self.rerank_top = int(cond.get("rerank_top", gate_cfg.rerank_top))

    def score_many(self, queries, prefs_lists):
        sims = SimTopKGate.score_many(self, queries, prefs_lists)  # type: ignore[arg-type]
        pairs, where = [], []
        for i, (q, prefs, s) in enumerate(zip(queries, prefs_lists, sims)):
            short = np.argsort(-s, kind="stable")[: self.rerank_top]
            for j in short:
                pairs.append((q, prefs[j].text))
                where.append((i, int(j)))
        ce = self.cross.score(pairs)
        out = [np.full(len(p), -np.inf, dtype=np.float32) for p in prefs_lists]
        for (i, j), v in zip(where, ce):
            out[i][j] = v
        return out
