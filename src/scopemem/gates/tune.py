"""Tuning without leakage (section 6.4): tau / k chosen on the dev folds of each test fold,
maximising micro-F1 of the gate's selection against G (retrieval only).

Micro-F1 pools TP/FP/FN over instances, so null instances (G = {}) contribute their false
positives (D-016). Ties are broken towards fewer selected preferences (higher tau / lower k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def tune_tau(gate: ScoringGate, scored: Sequence[Scored], n_quantiles: int = 101) -> tuple[float, float]:
    best = (-1.0, -np.inf)
    for tau in tau_candidates(scored, n_quantiles):
        sel = [gate.apply(s.scores, s.prefs, tau=float(tau)).selected_ids for s in scored]
        f = prf(sel, [s.gold for s in scored])[2]
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and tau > best[1]):
            best = (f, float(tau))
    return best[1], best[0]


def tune_k(gate: ScoringGate, scored: Sequence[Scored], grid: Sequence[int]) -> tuple[int, float]:
    best = (-1.0, 10 ** 9)
    for k in sorted(grid):
        sel = [gate.apply(s.scores, s.prefs, k=int(k)).selected_ids for s in scored]
        f = prf(sel, [s.gold for s in scored])[2]
        if f > best[0] + 1e-12:
            best = (f, int(k))
    return best[1], best[0]
