"""S2: zero-shot cross-encoder score(query, scope + examples) >= tau, + always_on, capped at k_max."""

from __future__ import annotations

import numpy as np

from .base import ScoringGate


class ScopeXencGate(ScoringGate):
    mode = "threshold"

    def __init__(self, cond, gate_cfg, cross):
        super().__init__(cond, gate_cfg)
        self.cross = cross
        self.fields = list(cond.get("fields", ["scope", "examples"]))

    def score_many(self, queries, prefs_lists):
        pairs = [(q, p.joined(self.fields)) for q, prefs in zip(queries, prefs_lists) for p in prefs]
        flat = self.cross.score(pairs)
        out, k = [], 0
        for prefs in prefs_lists:
            out.append(np.asarray(flat[k:k + len(prefs)], dtype=np.float32))
            k += len(prefs)
        return out
