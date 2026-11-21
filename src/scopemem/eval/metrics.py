"""Retrieval metrics for gate selections (E1) — computed from raw records only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .stats import bootstrap_ci


def retrieval_metrics(rows: Sequence[dict], n_boot: int = 10_000) -> dict:
    pos = [r for r in rows if r["kind"] == "positive"]
    nul = [r for r in rows if r["kind"] == "null"]
    hit_t = [float(r["target_id"] in r["selected_ids"]) for r in pos]
    tp = fp = fn = 0
    for r in rows:
        s, g = set(r["selected_ids"]), set(r["gold_ids"])
        tp += len(s & g)
        fp += len(s - g)
        fn += len(g - s)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else 0.0
    rt, rt_lo, rt_hi = bootstrap_ci(hit_t, n_boot) if hit_t else (np.nan,) * 3
    toks = [r["pref_tokens"] for r in rows]
    pt, pt_lo, pt_hi = bootstrap_ci(toks, n_boot)
    g_rows = [r for r in pos if r.get("global_id")]
    return {
        "n": len(rows), "n_pos": len(pos), "n_null": len(nul),
        "recall_target": rt, "recall_target_lo": rt_lo, "recall_target_hi": rt_hi,
        "recall_G": rec, "precision": prec, "f1": f1,
        "recall_global": (np.mean([r["global_id"] in r["selected_ids"] for r in g_rows])
                          if g_rows else np.nan),
        "mean_selected": float(np.mean([len(r["selected_ids"]) for r in rows])),
        "pref_tokens": pt, "pref_tokens_lo": pt_lo, "pref_tokens_hi": pt_hi,
        "overhead_tokens": float(np.mean([r["overhead_tokens"] for r in rows])),
        "null_any_selected": (float(np.mean([any(not s.startswith("global/") for s in r["selected_ids"])
                                             for r in nul])) if nul else np.nan),
        "cap_bind_rate": float(np.mean([bool(r.get("cap_bound")) for r in rows])),
        "write_tokens_user": float(np.mean([r.get("write_tokens_user", 0) for r in rows])),
    }
