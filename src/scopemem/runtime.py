"""Shared setup for runners: data (auto-built for smoke/local), scopes, store, subsets, records."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Sequence

from .config import config_hash, derive_seed, exp_name, resolve_path
from .data.build_multi import build_all, dataset_filename, load_instances, load_pref_table
from .data.schema import Instance
from .memory.scope_writer import ensure_scopes, scopes_path, writer_name
from .memory.store import PreferenceStore, ScopeBank
from .utils import git_sha

AUTO_PROFILES = {"smoke", "local"}


def processed_dir(cfg: Any) -> Path:
    return resolve_path(cfg.paths.processed)


def data_sets(cfg: Any) -> list[str]:
    exp = cfg.get("experiment") or {}
    return list(exp.get("data_sets") or [exp.get("data_set", "full")])


def ensure_data(cfg: Any, N: int, set_tag: str, form: str = "explicit") -> None:
    if (processed_dir(cfg) / dataset_filename(N, set_tag, form)).exists():
        return
    if cfg.get("profile") not in AUTO_PROFILES:
        raise FileNotFoundError(f"{dataset_filename(N, set_tag, form)} missing; run the build stage")
    if set_tag.startswith("xtopic"):
        from .data import cross_topic as xt

        if not xt.load_xtopic_items(processed_dir(cfg)):
            if not (processed_dir(cfg) / "prefs.jsonl").exists():
                build_all(cfg, sets=["full"], n_levels=[N], form=form)
            xt.generate(cfg)
            xt.verify(cfg)
            xt.select(cfg)
    build_all(cfg, sets=[set_tag], n_levels=[N], form=form)


def load_store(cfg: Any, llm: Any = None, write_missing: bool = True) -> PreferenceStore:
    if not (processed_dir(cfg) / "prefs.jsonl").exists():
        exp = cfg.get("experiment") or {}
        levels = list(cfg.run.n_levels or exp.get("n_levels") or [cfg.data.main_n])
        ensure_data(cfg, int(levels[0]), "full", str(cfg.data.form))
    prefs = load_pref_table(processed_dir(cfg))
    path = scopes_path(cfg, writer_name(cfg))
    if write_missing:
        path = ensure_scopes(cfg, list(prefs.values()), llm=llm)
    bank = ScopeBank.load(path) if path.exists() else ScopeBank()
    return PreferenceStore(prefs, bank)


def limit_instances(insts: Sequence[Instance], limit: int | None, seed: int, tag: str = "") -> list[Instance]:
    """Deterministic subset keeping the null share; order preserved."""
    if not limit or limit >= len(insts):
        return list(insts)
    pos = [i for i in insts if i.kind == "positive"]
    nul = [i for i in insts if i.kind == "null"]
    n_null = round(limit * len(nul) / len(insts))
    rng = random.Random(derive_seed(seed, "limit", tag, limit))
    keep = {i.instance_id for i in rng.sample(pos, limit - n_null) + rng.sample(nul, n_null)}
    return [i for i in insts if i.instance_id in keep]


def base_record(cfg: Any, inst: Instance, condition: str, model: str, seed: int = 0,
                data_set: str = "full") -> dict:
    return {
        "instance_id": inst.instance_id, "exp": exp_name(cfg), "model": model, "condition": condition,
        "data_set": data_set, "query_topic": inst.query_topic, "home_topic": inst.home_topic,
        "N": inst.N, "fold": inst.fold, "seed": seed, "kind": inst.kind, "form": inst.form,
        "target_id": inst.target_id, "global_id": inst.global_id, "gold_ids": list(inst.gold_ids),
        "response": None, "judge_labels": None, "judge_model": None, "latency_ms": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
    }


def get_instances(cfg: Any, N: int, set_tag: str | None = None) -> list[Instance]:
    set_tag = set_tag or data_sets(cfg)[0]
    form = str(cfg.data.form)
    ensure_data(cfg, N, set_tag, form)
    return load_instances(processed_dir(cfg), N, set_tag, form)
