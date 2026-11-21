"""Statistics (section 8): paired bootstrap CIs, McNemar, Wilcoxon, TOST-style non-inferiority,
Holm correction, Cohen's kappa and effect sizes. All resampling uses seed 0 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sst


def bootstrap_ci(x: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
                 stat: Callable[[np.ndarray], float] = np.mean) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    bs = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float, np.ndarray]:
    """Mean of (a - b) over paired instances with a percentile CI. Returns (diff, lo, hi, boot)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    bs = d[idx].mean(axis=1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi), bs


@dataclass
class McNemar:
    statistic: float
    pvalue: float
    b01: int     # a=0, b=1
    b10: int     # a=1, b=0
    exact: bool


def mcnemar(a: Sequence[bool], b: Sequence[bool], exact: bool | None = None,
            correction: bool = True, alternative: str = "two-sided") -> McNemar:
    """Paired McNemar on binary outcomes. exact=None -> exact binomial if discordant < 25.

    Two-sided matches statsmodels.stats.contingency_tables.mcnemar. alternative='greater'
    tests P(a=1,b=0) > P(a=0,b=1) (a better than b) with the exact one-sided binomial.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    b10 = int(np.sum(a & ~b))
    b01 = int(np.sum(~a & b))
    n = b10 + b01
    if exact is None:
        exact = n < 25
    if alternative == "greater":
        p = float(sst.binomtest(b10, n, 0.5, alternative="greater").pvalue) if n else 1.0
        return McNemar(float(min(b01, b10)), p, b01, b10, True)
    if exact:
        stat = float(min(b01, b10))
        p = float(sst.binomtest(int(stat), n, 0.5).pvalue) if n else 1.0
        return McNemar(stat, min(p, 1.0), b01, b10, True)
    stat = (abs(b10 - b01) - (1.0 if correction else 0.0)) ** 2 / n if n else 0.0
    return McNemar(float(stat), float(sst.chi2.sf(stat, 1)), b01, b10, False)


def wilcoxon(a: Sequence[float], b: Sequence[float], alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank on paired values + matched-pairs rank-biserial effect size."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nz = d[d != 0]
    if nz.size == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "rank_biserial": 0.0, "n": int(d.size)}
    res = sst.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    r = sst.rankdata(np.abs(nz))
    rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "rank_biserial": float(rbc), "n": int(d.size)}


def noninferiority(a: Sequence[float], b: Sequence[float], margin: float, alpha: float = 0.05,
                   n_boot: int = 10_000, seed: int = 0, scale: float = 100.0) -> dict:
    """One-sided TOST-style non-inferiority of a vs b: H0 mean(a-b) <= margin.

    ``a``, ``b`` are paired per-instance outcomes (0/1); ``scale=100`` puts the difference in
    percentage points so ``margin=-2.0`` means -2 points. Decision: the lower bound of the
    one-sided (1-alpha) bootstrap interval exceeds the margin. p = share of bootstrap diffs
    <= margin.
    """
    da = np.asarray(a, dtype=float) * scale
    db = np.asarray(b, dtype=float) * scale
    diff, _, _, bs = paired_bootstrap_diff(da, db, n_boot, seed, alpha)
    lower = float(np.quantile(bs, alpha))
    p = float(np.mean(bs <= margin))
    return {"diff": diff, "lower_one_sided": lower, "margin": margin, "pvalue": max(p, 1.0 / n_boot),
            "non_inferior": bool(lower > margin)}


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a = list(a)
    b = list(b)
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
