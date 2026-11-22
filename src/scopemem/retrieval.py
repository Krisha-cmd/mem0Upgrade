"""E1 runner (retrieval only): every gate at every N, tuned per fold on dev folds.

    python -m scopemem.retrieval experiment=E1 [profile=local] [run.shard=i run.num_shards=6]

Shards split the N levels. Output: results/raw/<exp>/<model>/<condition>/shard_N<N>.jsonl
(one record per instance; rewritten atomically per (condition, N) cell, skipped if already
complete with the same config_hash) and results/tuning/.../<condition>/N<N>.json.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from .config import condition_cfg, config_hash, exp_name, load_config, parse_cli, resolve_path
from .gates import GateContext
from .llm.engine import build_llm
from .llm.tokens import build_counter, pref_token_delta
from .memory.scope_writer import writer_name
from .prompting import build_messages
from .runtime import base_record, data_sets, get_instances, limit_instances, load_store
from .selection import run_condition, uses_scopes
from .utils import read_jsonl, setup_logging, write_jsonl


def _cell_complete(path: Path, n: int, chash: str) -> bool:
    if not path.exists():
        return False
    rows = list(read_jsonl(path))
    return len(rows) == n and all(r.get("config_hash") == chash for r in rows)


def _ensure_s3(cfg, conds: Sequence[str]) -> None:
    """Local/smoke convenience: train missing S3 fold models in-process."""
    from .gates.scope_ft import ft_model_dir, train_fold

    if not any(str(condition_cfg(c).gate) == "scope_ft" for c in conds) or cfg.xenc_cfg.type == "hash":
        return
    for f in range(int(cfg.data.n_folds)):
        if not (ft_model_dir(cfg, writer_name(cfg), f) / "config.json").exists():
            if cfg.get("profile") not in {"local", "smoke"}:
                raise FileNotFoundError(f"S3 fold {f} model missing; run the s3_train stage")
            train_fold(cfg, f)


def run(cfg) -> list[Path]:
    exp = cfg.experiment
    levels = list(cfg.run.n_levels or exp.n_levels)
    shard, nsh = int(cfg.run.shard), int(cfg.run.num_shards)
    my_levels = levels[shard::nsh] if nsh > 1 else levels
    llm = build_llm(cfg)
    ctx = GateContext(cfg, llm=llm)
    counter = build_counter(cfg)
    chash = config_hash(cfg)
    model = str(cfg.model_cfg.name)
    store = load_store(cfg, llm=llm)          # writes any missing scopes first (S3 needs them)
    for set_tag in data_sets(cfg):            # build every split first: S3 trains on xtopic items too
        for N in my_levels:
            get_instances(cfg, int(N), set_tag)
    _ensure_s3(cfg, list(exp.conditions))
    written: list[Path] = []
    for set_tag, N in [(s, n) for s in data_sets(cfg) for n in my_levels]:
        all_insts = get_instances(cfg, int(N), set_tag)
        eval_insts = limit_instances(all_insts, cfg.run.limit, int(cfg.seed), f"E1{set_tag}N{N}")
        tune_insts = eval_insts if cfg.get("profile") in ("local", "demo") else all_insts
        for cname in exp.conditions:
            out = (resolve_path(cfg.paths.raw_results) / exp_name(cfg) / model / cname
                   / f"shard_{set_tag}_N{N}.jsonl")
            if _cell_complete(out, len(eval_insts), chash):
                print(f"skip complete cell {cname} {set_tag} N={N}")
                continue
            cond = condition_cfg(cname)
            tun = resolve_path(cfg.paths.tuning) / model / cname / f"{set_tag}_N{N}.json"
            res = run_condition(cname, ctx, store, eval_insts, tune_insts, tun)
            rows = []
            for inst in eval_insts:
                r = res[inst.instance_id]
                recs = {p.id: p for p in store.records(inst.pref_ids)}
                texts = [recs[i].text for i in r.selection.selected_ids]
                with_b = build_messages(cfg, inst.query, texts)
                wtok = sum(recs[p].write_tokens for p in inst.pref_ids) if uses_scopes(cond) else 0
                rec = base_record(cfg, inst, cname, model, data_set=set_tag)
                rec.update({
                    "selected_ids": list(r.selection.selected_ids),
                    "n_selected": len(r.selection.selected_ids),
                    "pref_tokens": pref_token_delta(counter, with_b, build_messages(cfg, inst.query, [])),
                    "total_input_tokens": counter.count_messages(with_b),
                    "overhead_tokens": int(r.selection.overhead_tokens),
                    "output_tokens": 0, "truncated": False,
                    "write_tokens_user": int(wtok),
                    "write_tokens_amortised": wtok / float(cfg.write_amortise_queries),
                    "cap_bound": bool(r.selection.cap_bound), "gate_param": r.param,
                    "gate_meta": r.selection.meta,
                })
                rows.append(rec)
            tmp = out.with_suffix(".tmp")
            write_jsonl(tmp, rows)
            os.replace(tmp, out)
            written.append(out)
            print(f"{cname:16s} {set_tag:7s} N={N:<3d} n={len(rows)} -> {out}")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    sel, dot = parse_cli(argv if argv is not None else sys.argv[1:])
    cfg = load_config(sel.get("experiment", "E1"), sel.get("profile"), dot)
    run(cfg)


if __name__ == "__main__":
    main()
