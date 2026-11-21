"""Generation runner: one (experiment, model, condition, shard) per task.

    python -m scopemem.run experiment=E2 model=qwen7b task=7          # one Slurm array task
    python -m scopemem.run experiment=E0 profile=smoke                 # all tasks, sequentially

Tasks enumerate (condition, shard) pairs: task = cond_index * num_shards + shard.
Resumable: appends one flushed JSON line per finished instance, skips IDs already in the shard
file, and exits cleanly between batches if $SCRATCH_RUN/STOP (or ./STOP_<jobid>) exists.
Judge labels are added by a separate stage (scopemem.eval.judge, Phase 4).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import condition_cfg, exp_name, load_config, parse_cli, resolve_path
from .data.build_multi import materialize_history
from .data.prefeval import load_filler
from .gates import GateContext
from .llm.engine import GenParams, build_llm
from .llm.tokens import build_counter, pref_token_delta
from .prompting import build_messages, truncate_history_left
from .runtime import base_record, data_sets, get_instances, limit_instances, load_store
from .selection import run_condition, uses_scopes
from .utils import JsonlAppender, done_ids, setup_logging, shard_slice

REQUIRED_FIELDS = ["instance_id", "exp", "model", "condition", "N", "fold", "seed", "selected_ids",
                   "gold_ids", "pref_tokens", "total_input_tokens", "overhead_tokens", "output_tokens",
                   "write_tokens_amortised", "truncated", "latency_ms", "response", "judge_labels",
                   "judge_model", "timestamp", "git_sha", "config_hash"]


def stop_requested() -> bool:
    paths = []
    if os.environ.get("SCRATCH_RUN"):
        paths.append(Path(os.environ["SCRATCH_RUN"]) / "STOP")
    paths.append(Path(f"STOP_{os.environ.get('SLURM_JOB_ID', 'local')}"))
    return any(p.exists() for p in paths)


def enumerate_tasks(cfg: Any) -> list[tuple[str, str, int, int]]:
    """(data_set, condition, shard, N) tuples; task index = position in this list."""
    exp = cfg.experiment
    nsh = int(cfg.run.get("num_shards_override") or exp.get("num_shards", 1))
    levels = list(cfg.run.n_levels or exp.n_levels)
    return [(ds, c, s, int(N)) for ds in data_sets(cfg) for N in levels for c in exp.conditions
            for s in range(nsh)]


def run_task(cfg: Any, data_set: str, cond_name: str, shard: int, N: int, nsh: int, ctx: GateContext,
             counter, store, llm) -> Path:
    exp = cfg.experiment
    model = str(cfg.model_cfg.name)
    cond = condition_cfg(cond_name)
    all_insts = get_instances(cfg, N, data_set)
    limit = cfg.run.limit if cfg.run.limit is not None else exp.get("limit")
    eval_all = limit_instances(all_insts, limit, int(cfg.seed), f"{exp.name}{data_set}N{N}")
    mine = shard_slice(eval_all, shard, nsh)
    out = resolve_path(cfg.paths.raw_results) / exp_name(cfg) / model / cond_name / f"{data_set}_N{N}_shard_{shard}.jsonl"
    done = done_ids(out)
    todo = [i for i in mine if i.instance_id not in done]
    if not todo:
        print(f"{cond_name} {data_set} N={N} shard {shard}: complete ({len(done)})")
        return out
    sels = run_condition(cond_name, ctx, store, todo, all_insts)
    gp = GenParams(temperature=float(cfg.generation.temperature), max_tokens=int(cfg.generation.max_tokens),
                   seed=int(cfg.generation.seed))
    filler = load_filler(resolve_path(cfg.paths.prefeval_root)) if cond.get("history") else None
    budget = int(cfg.model_cfg.max_model_len) - int(cfg.generation.max_tokens) - 16
    bs = int(cfg.run.batch_size)
    with JsonlAppender(out) as app:
        for b in range(0, len(todo), bs):
            if stop_requested():
                print("STOP file found; exiting cleanly")
                sys.exit(0)
            batch = todo[b:b + bs]
            prompts, metas = [], []
            for inst in batch:
                r = sels[inst.instance_id]
                recs = {p.id: p for p in store.records(inst.pref_ids)}
                texts = [recs[i].text for i in r.selection.selected_ids]
                truncated = False
                if cond.get("history"):
                    hist = materialize_history(inst, store.prefs, filler, str(cfg.data.pref_ack))
                    hist, truncated = truncate_history_left(cfg, counter, inst.query, hist, budget)
                    msgs = build_messages(cfg, inst.query, [], hist)
                    ptok = pref_token_delta(counter, msgs, build_messages(cfg, inst.query, []))
                else:
                    msgs = build_messages(cfg, inst.query, texts)
                    ptok = pref_token_delta(counter, msgs, build_messages(cfg, inst.query, []))
                wtok = sum(recs[p].write_tokens for p in inst.pref_ids) if uses_scopes(cond) else 0
                prompts.append(msgs)
                metas.append((inst, r, ptok, truncated, wtok, counter.count_messages(msgs)))
            t0 = time.time()
            comps = llm.generate(prompts, gp)
            dt = (time.time() - t0) * 1000.0 / max(len(batch), 1)
            for (inst, r, ptok, truncated, wtok, tin), c in zip(metas, comps):
                rec = base_record(cfg, inst, cond_name, model, seed=int(cfg.generation.seed), data_set=data_set)
                rec.update({
                    "selected_ids": list(r.selection.selected_ids), "n_selected": len(r.selection.selected_ids),
                    "pref_tokens": int(ptok), "total_input_tokens": int(tin),
                    "overhead_tokens": int(r.selection.overhead_tokens),
                    "output_tokens": int(c.completion_tokens), "truncated": bool(truncated),
                    "write_tokens_user": int(wtok), "write_tokens_amortised": wtok / float(cfg.write_amortise_queries),
                    "cap_bound": bool(r.selection.cap_bound), "gate_param": r.param,
                    "response": c.text, "cached": c.cached, "batch_ms_per_item": dt,
                    "temperature": gp.temperature, "query": inst.query,
                })
                app.write(rec)
    print(f"{cond_name} {data_set} N={N} shard {shard}: wrote {len(todo)} -> {out}")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    args = list(argv if argv is not None else sys.argv[1:])
    sel, dot = parse_cli(args)
    task = None
    rest = []
    for d in dot:
        if d.startswith("task="):
            task = int(d.split("=", 1)[1])
        else:
            rest.append(d)
    cfg = load_config(sel.get("experiment"), sel.get("profile"), rest)
    tasks = enumerate_tasks(cfg)
    nsh = int(cfg.experiment.get("num_shards", 1))
    chosen = [tasks[task]] if task is not None else tasks
    llm = build_llm(cfg)
    ctx = GateContext(cfg, llm=llm)
    counter = build_counter(cfg)
    store = load_store(cfg, llm=llm)
    t0 = time.time()
    for data_set, cond_name, shard, N in chosen:
        run_task(cfg, data_set, cond_name, shard, N, nsh, ctx, counter, store, llm)
    print(f"done {len(chosen)} task(s) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
