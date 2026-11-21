"""S1: max cosine(query, {scope} ∪ example_requests) >= tau, + always_on, capped at k_max."""

from __future__ import annotations

import numpy as np

from .base import ScoringGate


class ScopeEmbedGate(ScoringGate):
    mode = "threshold"

    def __init__(self, cond, gate_cfg, embedder):
        super().__init__(cond, gate_cfg)
        self.embedder = embedder
        self.fields = list(cond.get("fields", ["scope", "examples"]))

    def score_many(self, queries, prefs_lists):
        qv = self.embedder.encode(list(queries), is_query=True)
        # warm the cache in one big batch
        self.embedder.encode(list({t for prefs in prefs_lists for p in prefs for t in p.field_texts(self.fields)}))
        out = []
        for i, prefs in enumerate(prefs_lists):
            sc = np.zeros(len(prefs), dtype=np.float32)
            for j, p in enumerate(prefs):
                tv = self.embedder.encode(p.field_texts(self.fields))
                sc[j] = float(np.max(tv @ qv[i]))
            out.append(sc)
        return out
