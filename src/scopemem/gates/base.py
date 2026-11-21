"""Gate interface: ``Gate.select(query, prefs) -> Selection(selected_ids, overhead_tokens, ...)``.

``Selection`` unpacks as ``(selected_ids, overhead_tokens)`` to match the spec's tuple form.
Scoring gates separate ``score`` (expensive, cached by the runner) from ``apply`` (cheap
threshold / top-k / cap / always-on logic), so tuning can sweep parameters over cached scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from ..data.schema import Instance
from ..memory.store import PrefRecord


@dataclass
class Selection:
    selected_ids: list[str]
    overhead_tokens: int = 0
    scores: dict[str, float] | None = None
    cap_bound: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        yield self.selected_ids
        yield self.overhead_tokens


class Gate:
    id: str = "gate"
    tunable: str | None = None        # "tau" | "k" | None

    def select(self, query: str, prefs: Sequence[PrefRecord], instance: Instance | None = None) -> Selection:
        return self.select_many([query], [prefs], [instance])[0]

    def select_many(self, queries: Sequence[str], prefs_lists: Sequence[Sequence[PrefRecord]],
                    instances: Sequence[Instance | None]) -> list[Selection]:
        return [self.select(q, p, i) for q, p, i in zip(queries, prefs_lists, instances)]


class ScoringGate(Gate):
    """Gates whose decision is a function of per-preference scores."""

    mode: str = "threshold"           # "threshold" | "topk"

    def __init__(self, cond: Any, gate_cfg: Any):
        self.id = str(cond.id)
        self.k_max = cond.get("k_max", gate_cfg.k_max)
        self.k_max = None if self.k_max in (None, "inf", "none") else int(self.k_max)
        self.use_always_on = bool(cond.get("use_always_on", False))
        tau = cond.get("tau", None)
        k = cond.get("k", None)
        self.tau: float | None = None if tau in (None, "tune") else float(tau)
        self.k: int | None = None if k in (None, "tune") else int(k)
        if self.mode == "threshold":
            self.tunable = "tau" if tau == "tune" else None
        else:
            self.tunable = "k" if k == "tune" else None

    # subclasses implement
    def score_many(self, queries: Sequence[str], prefs_lists: Sequence[Sequence[PrefRecord]]) -> list[np.ndarray]:
        raise NotImplementedError

    def apply(self, scores: np.ndarray, prefs: Sequence[PrefRecord], *, tau: float | None = None,
              k: int | None = None, k_max: int | None | str = "default") -> Selection:
        k_max = self.k_max if k_max == "default" else k_max
        sdict = {p.id: float(s) for p, s in zip(prefs, scores)}
        if self.mode == "topk":
            k = self.k if k is None else k
            if k is None:
                raise ValueError(f"{self.id}: k not set (tune first)")
            order = np.argsort(-scores, kind="stable")[: min(k, len(prefs))]
            return Selection([prefs[i].id for i in order], 0, sdict)
        tau = self.tau if tau is None else tau
        if tau is None:
            raise ValueError(f"{self.id}: tau not set (tune first)")
        on = [p.id for p in prefs if self.use_always_on and p.always_on]
        cand = [(float(s), i) for i, (p, s) in enumerate(zip(prefs, scores))
                if s >= tau and not (self.use_always_on and p.always_on)]
        cand.sort(key=lambda x: (-x[0], x[1]))
        cap_bound = k_max is not None and len(cand) > k_max
        if k_max is not None:
            cand = cand[:k_max]
        sel = [prefs[i].id for _, i in cand] + on
        return Selection(sel, 0, sdict, cap_bound, {"n_always_on": len(on)})

    def select_many(self, queries, prefs_lists, instances):
        return [self.apply(s, p) for s, p in zip(self.score_many(queries, prefs_lists), prefs_lists)]
