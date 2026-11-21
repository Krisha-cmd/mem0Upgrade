"""Figures (PDF) from aggregated tables. Static paper figures, light theme.

Palette: validated 8-slot categorical order (dataviz reference palette), assigned to conditions
in a FIXED order so a condition keeps its colour in every figure. Line style is the secondary
encoding (dashed = baseline, solid = ScopeMem). Every multi-series figure has a legend plus
direct end labels; the matching CSV in results/tables/ is the table view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3dd", "#fcfcfb"
REF = "#8a897f"     # reference conditions (inject-all, oracle) in neutral grey

# Fixed condition -> colour slot. Conditions not listed are drawn as grey references.
SLOTS = {
    "B3_sim_k1": 0, "B3_sim_k3": 0, "B3_sim_k5": 0, "B3_sim_k10": 0, "B3_sim_tuned": 0,
    "B4_sim_rerank": 1, "B5_llm_select": 3, "S1_scope_embed": 2, "S2_scope_xenc": 6,
    "S3_scope_ft": 5, "B1_full": 4, "B0_none": 7,
}
LABELS = {
    "B3_sim_k1": "Similarity k=1", "B3_sim_k3": "Similarity k=3", "B3_sim_k5": "Similarity k=5",
    "B3_sim_k10": "Similarity k=10", "B4_sim_rerank": "Sim + rerank", "B5_llm_select": "LLM selector",
    "S1_scope_embed": "ScopeMem S1 (embed)", "S2_scope_xenc": "ScopeMem S2 (CE)",
    "S3_scope_ft": "ScopeMem S3 (FT CE)", "B2_all": "Inject all", "UB_oracle": "Oracle",
    "B1_full": "Full context", "B0_none": "No memory",
}
MARKERS = {"B3_sim_k1": "v", "B3_sim_k3": "<", "B3_sim_k5": "o", "B3_sim_k10": ">"}
SIM_ALPHA = {"B3_sim_k1": 0.45, "B3_sim_k3": 0.65, "B3_sim_k5": 1.0, "B3_sim_k10": 0.85}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _color(c: str) -> str:
    return PALETTE[SLOTS[c]] if c in SLOTS else REF


def _ls(c: str) -> str:
    return "-" if c.startswith("S") else "--"


def recall_vs_n(df: pd.DataFrame, out: Path, conditions: Sequence[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    _style(ax)
    ends = []
    for c in conditions:
        d = df[df.condition == c].sort_values("N")
        if d.empty:
            continue
        ax.fill_between(d.N, d.recall_target_lo, d.recall_target_hi, color=_color(c), alpha=0.12, lw=0)
        ax.plot(d.N, d.recall_target, _ls(c), color=_color(c), lw=2, marker=MARKERS.get(c, "o"), ms=5,
                alpha=SIM_ALPHA.get(c, 1.0), label=LABELS.get(c, c))
        last = d.iloc[-1]
        ends.append([float(last.recall_target), float(last.N), LABELS.get(c, c)])
    # direct end labels, nudged apart vertically so converging lines stay legible
    ends.sort()
    gap = 0.05
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, lab in ends:
        ax.annotate(lab, (x, y), xytext=(8, 0), textcoords="offset points", fontsize=6.5, color=INK2,
                    va="center", annotation_clip=False)
    fig.subplots_adjust(right=0.78)
    ax.set_xscale("log")
    ns = sorted(df.N.unique())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Stored preferences per user (N, log scale)", color=INK, fontsize=9)
    ax.set_ylabel("Recall of the applicable preference", color=INK, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out)
    plt.close(fig)


def recall_vs_tokens(df: pd.DataFrame, out: Path, conditions: Sequence[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor=SURFACE)
    _style(ax)
    for c in conditions:
        d = df[df.condition == c]
        if d.empty:
            continue
        r = d.iloc[0]
        ref = c not in SLOTS
        ax.scatter([max(r.pref_tokens, 1)], [r.recall_target], s=60 if not ref else 70,
                   marker="D" if ref else MARKERS.get(c, "o"), color=_color(c),
                   alpha=SIM_ALPHA.get(c, 1.0), edgecolors=SURFACE, linewidths=2, zorder=3,
                   label=LABELS.get(c, c))
        ax.errorbar([max(r.pref_tokens, 1)], [r.recall_target],
                    yerr=[[r.recall_target - r.recall_target_lo], [r.recall_target_hi - r.recall_target]],
                    color=_color(c), lw=1, capsize=0, zorder=2)
        ax.annotate(LABELS.get(c, c), (max(r.pref_tokens, 1), r.recall_target), xytext=(6, 3),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Preference tokens per query (log scale)", color=INK, fontsize=9)
    ax.set_ylabel("Recall of the applicable preference", color=INK, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
