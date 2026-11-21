import numpy as np
import pytest
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from scopemem.eval.stats import (bootstrap_ci, cohen_kappa, holm, mcnemar, noninferiority,
                                 paired_bootstrap_diff, wilcoxon)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(1)
    covered = 0
    for s in range(40):
        x = rng.normal(5.0, 2.0, size=200)
        _, lo, hi = bootstrap_ci(x, n_boot=2000, seed=s)
        covered += lo <= 5.0 <= hi
    assert covered >= 34          # ~95% nominal


def test_paired_diff_sign():
    a = np.ones(100)
    b = np.r_[np.ones(80), np.zeros(20)]
    d, lo, hi, _ = paired_bootstrap_diff(a, b, 2000)
    assert d == pytest.approx(0.2) and lo > 0


@pytest.mark.parametrize("b10,b01,exact", [(30, 12, False), (5, 2, True), (40, 40, False), (0, 9, True)])
def test_mcnemar_matches_statsmodels(b10, b01, exact):
    a = np.r_[np.ones(b10), np.zeros(b01), np.ones(50), np.zeros(20)].astype(bool)
    b = np.r_[np.zeros(b10), np.ones(b01), np.ones(50), np.zeros(20)].astype(bool)
    ours = mcnemar(a, b, exact=exact)
    table = [[50, b10], [b01, 20]]
    ref = sm_mcnemar(table, exact=exact, correction=True)
    assert ours.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-9)


def test_holm_ordering_and_values():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = holm(p)
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert np.argsort(adj, kind="stable").tolist()[0] == 0
    assert all(x <= 1 for x in holm([0.9, 0.8]))


def test_noninferiority_decisions():
    n = 1000
    b = np.r_[np.ones(800), np.zeros(200)]
    same = b.copy()
    assert noninferiority(same, b, margin=-2.0)["non_inferior"]
    worse = b.copy()
    worse[:100] = 0                  # -10 points
    r = noninferiority(worse, b, margin=-2.0)
    assert not r["non_inferior"] and r["diff"] == pytest.approx(-10.0)
    slightly = b.copy()
    slightly[:5] = 0                 # -0.5 points
    assert noninferiority(slightly, b, margin=-2.0)["non_inferior"]
    assert len(b) == n


def test_wilcoxon_and_kappa():
    a = np.arange(50, dtype=float)
    r = wilcoxon(a * 0.2, a, alternative="less")
    assert r["pvalue"] < 1e-6 and r["rank_biserial"] < 0
    assert cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0
