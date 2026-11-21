"""Raw JSONL -> tables (CSV + LaTeX) and figures (PDF). Uses raw records only.

    python -m scopemem.aggregate experiment=E1 [profile=local] [--allow-mixed]
    python -m scopemem.aggregate experiment=all        # make paper

Refuses to mix records with different config_hash within one (model, condition, N) cell unless
--allow-mixed is given. Missing cells are reported as NA with a reason, never estimated.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import exp_name, load_config, parse_cli, resolve_path
from .eval.metrics import retrieval_metrics
from .utils import read_jsonl, setup_logging


class MixedConfigError(RuntimeError):
    pass


def load_cells(raw_dir: Path, allow_mixed: bool = False) -> dict[tuple[str, str, str, int], list[dict]]:
    """(model, data_set, condition, N) -> records."""
    cells: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for f in sorted(raw_dir.glob("*/*/*.jsonl")):
        for r in read_jsonl(f):
            cells[(r["model"], r.get("data_set", "full"), r["condition"], int(r["N"]))].append(r)
    for key, rows in cells.items():
        hashes = {r["config_hash"] for r in rows}
        if len(hashes) > 1 and not allow_mixed:
            raise MixedConfigError(f"cell {key} mixes config hashes {sorted(hashes)}; use --allow-mixed")
    return cells


def _fmt(v: float, pct: bool = False) -> str:
    if v != v:  # NaN
        return "NA"
    return f"{100 * v:.1f}" if pct else f"{v:.1f}"


def aggregate_e1(cfg, allow_mixed: bool = False) -> pd.DataFrame:
    from .plots import recall_vs_n, recall_vs_tokens

    name = exp_name(cfg)
    raw = resolve_path(cfg.paths.raw_results) / name
    tables = resolve_path(cfg.paths.tables)
    figs = resolve_path(cfg.paths.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    cells = load_cells(raw, allow_mixed)
    exp = cfg.experiment
    levels = list(cfg.run.n_levels or exp.n_levels)
    rows = []
    models = sorted({k[0] for k in cells}) or [str(cfg.model_cfg.name)]
    sets = list(exp.get("data_sets") or [exp.get("data_set", "full")])
    for m in models:
        for ds in sets:
            for c in exp.conditions:
                for N in levels:
                    recs = cells.get((m, ds, c, int(N)))
                    base = {"model": m, "data_set": ds, "condition": c, "N": int(N)}
                    if not recs:
                        rows.append({**base, "status": "NA: no raw records"})
                        continue
                    rows.append({**base, "status": "ok", **retrieval_metrics(recs)})
    df = pd.DataFrame(rows)
    df.to_csv(tables / f"{name}_retrieval.csv", index=False)
    ok = df[df.status == "ok"]
    main_n = int(cfg.data.main_n) if int(cfg.data.main_n) in levels else levels[len(levels) // 2]
    for ds in sets:
        t = ok[(ok.N == main_n) & (ok.data_set == ds)]
        lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
                 r"Gate & Recall@target (95\% CI) & Precision & F1 & \#sel. & Pref.\ tok. & Null FP \\",
                 r"\midrule"]
        for _, r in t.iterrows():
            lines.append(f"{r.condition.replace('_', chr(92) + '_')} & {_fmt(r.recall_target, True)} "
                         f"[{_fmt(r.recall_target_lo, True)}, {_fmt(r.recall_target_hi, True)}] & "
                         f"{_fmt(r.precision, True)} & {_fmt(r.f1, True)} & {r.mean_selected:.2f} & "
                         f"{r.pref_tokens:.0f} & {_fmt(r.null_any_selected, True)} \\\\")
        for _, r in df[(df.status != "ok") & (df.N == main_n) & (df.data_set == ds)].iterrows():
            lines.append(f"{r.condition.replace('_', chr(92) + '_')} & \\multicolumn{{6}}{{l}}{{{r.status}}} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (tables / f"{name}_{ds}_retrieval_N{main_n}.tex").write_text("\n".join(lines) + "\n")
        okd = ok[ok.data_set == ds]
        if okd.empty:
            continue
        plot_conds = [c for c in ["B3_sim_k1", "B3_sim_k5", "B3_sim_k10", "B4_sim_rerank", "B5_llm_select",
                                  "S1_scope_embed", "S2_scope_xenc", "S3_scope_ft"] if c in set(okd.condition)]
        for m in models:
            d = okd[okd.model == m]
            recall_vs_n(d, figs / f"{name}_{ds}_{m}_recall_vs_N.pdf", plot_conds,
                        f"{name} [{ds}]: recall of the applicable preference vs N ({m})")
            tok_conds = plot_conds + [c for c in ["B2_all", "UB_oracle", "B3_sim_k3"] if c in set(okd.condition)]
            recall_vs_tokens(d[d.N == main_n], figs / f"{name}_{ds}_{m}_recall_vs_tokens_N{main_n}.pdf",
                             tok_conds, f"{name} [{ds}]: recall vs preference tokens at N={main_n} ({m})")
    return df


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    args = list(argv if argv is not None else sys.argv[1:])
    allow_mixed = "--allow-mixed" in args
    args = [a for a in args if a != "--allow-mixed"]
    sel, dot = parse_cli(args)
    exps = [sel.get("experiment", "E1")]
    if exps == ["all"]:
        exps = [f"E{i}" for i in range(0, 9)]
    for e in exps:
        cfg = load_config(e, sel.get("profile"), dot)
        kind = str(cfg.experiment.kind)
        if kind == "retrieval":
            df = aggregate_e1(cfg, allow_mixed)
            show = df[df.N == int(cfg.data.main_n)] if "N" in df else df
            cols = [c for c in ["model", "data_set", "condition", "N", "status", "recall_target", "precision", "f1",
                                "mean_selected", "pref_tokens", "null_any_selected", "cap_bind_rate"] if c in df]
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(show[cols].round(3).to_string(index=False))
        else:
            print(f"{e}: aggregation for kind={kind} is implemented in Phase 4+ (NA)")


if __name__ == "__main__":
    main()
