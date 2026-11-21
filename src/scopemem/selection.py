"""Fold-aware selection: for each held-out fold f, tune the gate on instances of the other folds,
then apply it to fold f. Results are pooled over the 5 held-out folds (section 6.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import condition_cfg
from .data.schema import Instance
from .gates import GateContext, ScoringGate, Selection, build_gate, is_fold_specific
from .gates.tune import Scored, tune_k, tune_tau
from .memory.store import PreferenceStore

SCOPE_GATES = {"scope_embed", "scope_xenc", "scope_ft"}


@dataclass
class SelResult:
    selection: Selection
    fold: int
    param: dict[str, Any]


def uses_scopes(cond: Any) -> bool:
    return str(cond.gate) in SCOPE_GATES


def run_condition(cond_name: str, ctx: GateContext, store: PreferenceStore,
                  eval_insts: Sequence[Instance], tune_insts: Sequence[Instance],
                  tuning_out: Path | None = None) -> dict[str, SelResult]:
    cond = condition_cfg(cond_name)
    cfg = ctx.cfg
    folds_present = sorted({i.fold for i in eval_insts})
    out: dict[str, SelResult] = {}
    tuning_log: dict[str, Any] = {}

    def recs(insts):
        return [store.records(i.pref_ids) for i in insts]

    shared_gate = None if is_fold_specific(cond) else build_gate(cond, ctx)
    shared_scores: dict[str, Scored] | None = None

    for f in folds_present:
        gate = shared_gate or build_gate(cond, ctx, fold=f)
        test = [i for i in eval_insts if i.fold == f]
        if not isinstance(gate, ScoringGate):
            sels = gate.select_many([i.query for i in test], recs(test), test)
            for i, s in zip(test, sels):
                out[i.instance_id] = SelResult(s, f, {})
            continue
        # score tune ∪ eval once per gate object
        if shared_gate is not None and shared_scores is not None:
            scored = shared_scores
        else:
            pool = {i.instance_id: i for i in list(tune_insts) + list(eval_insts)}
            plist = list(pool.values())
            rl = recs(plist)
            sc = gate.score_many([i.query for i in plist], rl)
            scored = {i.instance_id: Scored(s, r, frozenset(i.gold_ids)) for i, s, r in zip(plist, sc, rl)}
            if shared_gate is not None:
                shared_scores = scored
        param: dict[str, Any] = {}
        if gate.tunable:
            dev = [scored[i.instance_id] for i in tune_insts if i.fold != f]
            obj, floor = str(cfg.gate.tune_objective), float(cfg.gate.recall_floor)
            if gate.tunable == "tau":
                tau, d = tune_tau(gate, dev, int(cfg.gate.tau_quantiles), obj, floor)
                param = {"tau": tau}
            else:
                k, d = tune_k(gate, dev, list(cfg.gate.k_grid), obj, floor)
                param = {"k": k}
            param.update({"objective": obj, "dev_recall": d["recall"], "dev_precision": d["precision"],
                          "dev_f1": d["f1"], "floor_met": d["floor_met"], "n_dev": len(dev)})
        tuning_log[str(f)] = param
        for i in test:
            s = scored[i.instance_id]
            sel = gate.apply(s.scores, s.prefs, tau=param.get("tau"), k=param.get("k"))
            out[i.instance_id] = SelResult(sel, f, param)
    if tuning_out is not None and tuning_log and any(tuning_log.values()):
        tuning_out.parent.mkdir(parents=True, exist_ok=True)
        tuning_out.write_text(json.dumps(tuning_log, indent=1, sort_keys=True))
    return out
