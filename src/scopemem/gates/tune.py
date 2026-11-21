"""Tuning without leakage (section 6.4): tau / k chosen on the dev folds of each test fold
(retrieval only, no generation).

Objectives (``gate.tune_objective``):
- ``recall_floor`` (default, D-032): the strictest setting (highest tau / smallest k) whose pooled
  dev recall of G is >= ``gate.recall_floor`` (0.95). If no setting reaches the floor, the
  setting with the highest recall is used and the miss is recorded.
- ``f1``: micro-F1 (pooled TP/FP/FN; null instances add their false positives). Ties go to fewer
  selections. This was the build prompt's original objective (D-016).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..memory.store import PrefRecord
from .base import ScoringGate


@dataclass
class Scored:
    scores: np.ndarray
    prefs: Sequence[PrefRecord]
    gold: frozenset[str]


def prf(selected: Sequence[Sequence[str]], gold: Sequence[frozenset[str]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for s, g in zip(selected, gold):
        ss = set(s)
        tp += len(ss & g)
        fp += len(ss - g)
        fn += len(g - ss)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def tau_candidates(scored: Sequence[Scored], n_quantiles: int) -> np.ndarray:
    allv = np.concatenate([s.scores[np.isfinite(s.scores)] for s in scored if len(s.scores)]) \
        if scored else np.zeros(0)
    if allv.size == 0:
        return np.array([0.5])
    qs = np.quantile(allv, np.linspace(0.0, 1.0, n_quantiles))
    return np.unique(np.concatenate([qs, [allv.max() + 1e-6]]))


def _pick(results: list[tuple[Any, float, float, float]], objective: str, floor: float) -> dict:
    """results: (param, precision, recall, f1) ordered from strictest to loosest setting."""
    if objective == "f1":            # max F1; ties -> the strictest setting (earliest)
        best_f1 = max(r[3] for r in results)
        best = next(i for i, r in enumerate(results) if abs(r[3] - best_f1) <= 1e-12)
        p, pr, rc, f = results[best]
        return {"param": p, "precision": pr, "recall": rc, "f1": f, "floor_met": None}
    if objective == "recall_floor":
        for p, pr, rc, f in results:          # strictest first
            if rc >= floor - 1e-12:
                return {"param": p, "precision": pr, "recall": rc, "f1": f, "floor_met": True}
        i = max(range(len(results)), key=lambda j: (results[j][2], -j))
        p, pr, rc, f = results[i]
        return {"param": p, "precision": pr, "recall": rc, "f1": f, "floor_met": False}
    raise ValueError(f"unknown tune objective {objective}")


def tune_tau(gate: ScoringGate, scored: Sequence[Scored], n_quantiles: int = 101,
             objective: str = "f1", floor: float = 0.95) -> tuple[float, dict]:
    res = []
    for tau in sorted(tau_candidates(scored, n_quantiles), reverse=True):   # strictest first
        sel = [gate.apply(s.scores, s.prefs, tau=float(tau)).selected_ids for s in scored]
        res.append((float(tau), *prf(sel, [s.gold for s in scored])))
    d = _pick(res, objective, floor)
    return d.pop("param"), d


def tune_k(gate: ScoringGate, scored: Sequence[Scored], grid: Sequence[int],
           objective: str = "f1", floor: float = 0.95) -> tuple[int, dict]:
    res = []
    for k in sorted(grid):                                                  # strictest first
        sel = [gate.apply(s.scores, s.prefs, k=int(k)).selected_ids for s in scored]
        res.append((int(k), *prf(sel, [s.gold for s in scored])))
    d = _pick(res, objective, floor)
    return d.pop("param"), d
