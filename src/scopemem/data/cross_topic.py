"""Cross-topic query split (D-033, added 2026-11-21 at the user's request).

PrefEval pairs each preference with a SAME-topic question, which makes similarity retrieval easy
(bge-base: 96% recall@5 at N=25). The paper's claim is about preferences that matter for requests
on a different topic. This module builds such requests:

1. generate  - the generator LLM (Llama-3.3-70B by default; a different family from the Qwen scope
               writer) writes `per_pref` requests per unique preference, each from a different
               PrefEval topic than the preference's own, without mentioning the preference.
2. verify    - the verifier LLM (the primary judge) keeps a request only if applies=yes,
               mentions=no, natural=yes. Requests are NOT filtered on embedding or lexical
               similarity, which would bias the split against the similarity baselines.
3. select    - per preference, the first kept request is the "primary" target for the xtopic
               split; other kept requests are "reserve" (used only as extra S3 training pairs for
               training folds).
4. audit     - export a blind, stratified sample for human checking; score it afterwards.

    python -m scopemem.data.cross_topic stage=generate|verify|select|export_audit|score_audit [profile=...]
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from ..config import derive_seed, load_named, resolve_path
from ..llm.engine import GenParams, LLM
from ..prompts import load_prompt
from ..utils import JsonlAppender, done_ids, read_jsonl, write_jsonl

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
RETRY = "Your previous reply was not valid JSON in the required schema. Reply again with ONLY the JSON."


def _json(text: str) -> Any:
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def parse_generation(text: str, home_topic: str, topics: Sequence[str]) -> list[dict] | None:
    obj = _json(text)
    if not isinstance(obj, list):
        return None
    allowed = set(topics) - {home_topic}
    out, used = [], set()
    for r in obj:
        if not isinstance(r, dict):
            continue
        t, q = str(r.get("topic", "")).strip(), str(r.get("request", "")).strip()
        if t in allowed and t not in used and q:
            used.add(t)
            out.append({"query_topic": t, "request": q, "why": str(r.get("why", "")).strip()})
    return out or None


def parse_verification(text: str) -> dict | None:
    obj = _json(text)
    if not isinstance(obj, dict):
        return None
    norm = {}
    for k in ("applies", "mentions", "natural"):
        v = str(obj.get(k, "")).strip().lower()
        if v not in ("yes", "no"):
            return None
        norm[k] = v
    norm["reason"] = str(obj.get("reason", ""))
    norm["keep"] = norm["applies"] == "yes" and norm["mentions"] == "no" and norm["natural"] == "yes"
    return norm


def _paths(cfg) -> dict[str, Path]:
    d = resolve_path(cfg.paths.processed) / "cross_topic"
    g, v = _model_name(cfg, "generator"), _model_name(cfg, "verifier")
    return {"dir": d, "cand": d / f"candidates__{g}.jsonl", "ver": d / f"verified__{g}__{v}.jsonl",
            "items": d / "xtopic_items.jsonl",
            "audit": resolve_path(cfg.paths.get("annotation", "results/annotation")) / "xtopic_audit.csv",
            "key": resolve_path(cfg.paths.get("annotation", "results/annotation")) / "xtopic_audit_key.json"}


def _model_name(cfg, role: str) -> str:
    if cfg.llm_backend == "fake":
        return f"fake-{role}"
    return str(load_named("models", str(cfg.cross_topic[role])).name)


def _build_llm(cfg, role: str) -> LLM:
    from ..llm.cache import CachedLLM, SQLiteCache
    from ..llm.engine import VLLMEngine
    from ..llm.fake import FakeLLM

    name = _model_name(cfg, role)
    inner: LLM = FakeLLM(name=name) if cfg.llm_backend == "fake" else VLLMEngine(
        load_named("models", str(cfg.cross_topic[role])))
    return CachedLLM(inner, SQLiteCache(resolve_path(cfg.paths.cache) / f"{name}.sqlite"))


def _with_retry(llm: LLM, msgs: list, params: GenParams, parse) -> list[tuple[Any, int]]:
    """Generate, re-ask once on parse failure. Returns (parsed or None, attempts)."""
    outs = llm.generate(msgs, params)
    parsed = [parse(c.text) for c in outs]
    bad = [i for i, p in enumerate(parsed) if p is None]
    if bad:
        re_msgs = [msgs[i] + [{"role": "assistant", "content": outs[i].text}, {"role": "user", "content": RETRY}]
                   for i in bad]
        for i, c in zip(bad, llm.generate(re_msgs, params)):
            parsed[i] = parse(c.text)
    return [(p, 2 if i in set(bad) else 1) for i, p in enumerate(parsed)]


def unique_prefs(cfg) -> list[Any]:
    from .build_multi import _dedup_pool, items_to_prefs
    from .prefeval import load_items

    prefs = _dedup_pool(items_to_prefs(load_items(resolve_path(cfg.paths.prefeval_root), "explicit")))
    cap = cfg.cross_topic.get("limit_per_topic")
    if cap:                                   # demo/preflight: first `cap` prefs of every topic
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for x in prefs:
            if seen[x.topic] < int(cap):
                seen[x.topic] += 1
                kept.append(x)
        prefs = kept
    return prefs


def generate(cfg, llm: LLM | None = None, batch: int = 256) -> Path:
    from .prefeval import list_topics

    p = _paths(cfg)
    llm = llm or _build_llm(cfg, "generator")
    topics = list_topics(resolve_path(cfg.paths.prefeval_root))
    tmpl = load_prompt("cross_topic_gen")
    n = int(cfg.cross_topic.per_pref)
    params = GenParams(temperature=0.0, max_tokens=int(cfg.cross_topic.max_tokens))
    done = done_ids(p["cand"], key="pref_id")
    todo = [x for x in unique_prefs(cfg) if x.id not in done]
    with JsonlAppender(p["cand"]) as app:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            msgs = [[{"role": "user", "content": tmpl.format(
                preference=x.text, home_topic=x.topic, n=n,
                other_topics="\n".join(f"- {t}" for t in topics if t != x.topic))}] for x in chunk]
            parsers = [lambda t, h=x.topic: parse_generation(t, h, topics) for x in chunk]
            outs = llm.generate(msgs, params)
            parsed_all = [f(c.text) for f, c in zip(parsers, outs)]
            bad = [i for i, v in enumerate(parsed_all) if v is None]
            if bad:                                   # one batched re-ask for all failures
                re_msgs = [msgs[i] + [{"role": "assistant", "content": outs[i].text},
                                      {"role": "user", "content": RETRY}] for i in bad]
                for i, c in zip(bad, llm.generate(re_msgs, params)):
                    parsed_all[i] = parsers[i](c.text)
            for i, x in enumerate(chunk):
                parsed, attempts = parsed_all[i], 2 if i in set(bad) else 1
                app.write({"pref_id": x.id, "home_topic": x.topic, "preference": x.text,
                           "attempts": attempts, "ok": parsed is not None,
                           "requests": (parsed or [])[:n], "generator": llm.name})
    return p["cand"]


def verify(cfg, llm: LLM | None = None) -> Path:
    p = _paths(cfg)
    llm = llm or _build_llm(cfg, "verifier")
    tmpl = load_prompt("cross_topic_verify")
    params = GenParams(temperature=0.0, max_tokens=128)
    rows = []
    for r in read_jsonl(p["cand"]):
        for j, q in enumerate(r["requests"]):
            rows.append({"id": f"xt/{r['pref_id']}/{j}", "pref_id": r["pref_id"], "home_topic": r["home_topic"],
                         "preference": r["preference"], **q})
    done = done_ids(p["ver"])
    todo = [r for r in rows if r["id"] not in done]
    msgs = [[{"role": "user", "content": tmpl.format(preference=r["preference"], request=r["request"])}]
            for r in todo]
    res = _with_retry(llm, msgs, params, parse_verification) if todo else []
    with JsonlAppender(p["ver"]) as app:
        for r, (v, att) in zip(todo, res):
            app.write({**r, "verify": v, "verify_attempts": att, "keep": bool(v and v["keep"]),
                       "verifier": llm.name})
    return p["ver"]


def select(cfg) -> dict:
    p = _paths(cfg)
    by_pref: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(p["ver"]):
        by_pref[r["pref_id"]].append(r)
    items = []
    stats = {"prefs": len(by_pref), "candidates": 0, "kept": 0, "prefs_with_primary": 0,
             "reject_reasons": defaultdict(int)}
    for pid in sorted(by_pref):
        rs = sorted(by_pref[pid], key=lambda r: r["id"])
        stats["candidates"] += len(rs)
        kept = [r for r in rs if r["keep"]]
        stats["kept"] += len(kept)
        for r in rs:
            if not r["keep"]:
                v = r.get("verify") or {}
                why = ("parse_fail" if not v else "not_applies" if v.get("applies") != "yes"
                       else "mentions" if v.get("mentions") != "no" else "unnatural")
                stats["reject_reasons"][why] += 1
        for k, r in enumerate(kept):
            items.append({"id": r["id"], "pref_id": pid, "home_topic": r["home_topic"],
                          "query_topic": r["query_topic"], "request": r["request"],
                          "role": "primary" if k == 0 else "reserve"})
        stats["prefs_with_primary"] += bool(kept)
    write_jsonl(p["items"], items)
    stats["reject_reasons"] = dict(stats["reject_reasons"])
    (p["dir"] / "select_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_xtopic_items(processed: str | Path) -> list[dict]:
    path = Path(processed) / "cross_topic" / "xtopic_items.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def export_audit(cfg) -> Path:
    """Blind sample for the human check: ~80% kept primaries, ~20% rejected candidates."""
    p = _paths(cfg)
    ver = list(read_jsonl(p["ver"]))
    kept_primary = {r["id"] for r in load_xtopic_items(resolve_path(cfg.paths.processed)) if r["role"] == "primary"}
    pos = [r for r in ver if r["id"] in kept_primary]
    neg = [r for r in ver if not r["keep"]]
    n = int(cfg.cross_topic.audit_n)
    rng = random.Random(derive_seed(cfg.seed, "xtopic_audit"))
    n_neg = min(len(neg), round(n * 0.2))
    sample = rng.sample(pos, min(len(pos), n - n_neg)) + rng.sample(neg, n_neg)
    rng.shuffle(sample)
    p["audit"].parent.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(p["audit"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "preference", "request", "applies (Y/N)", "mentions_preference (Y/N)", "notes"])
        for i, r in enumerate(sample):
            aid = f"a{i:03d}"
            key[aid] = {"id": r["id"], "keep": r["keep"], "verify": r.get("verify")}
            w.writerow([aid, r["preference"], r["request"], "", "", ""])
    p["key"].write_text(json.dumps(key, indent=1))
    return p["audit"]


def score_audit(cfg) -> dict:
    from ..eval.stats import cohen_kappa

    p = _paths(cfg)
    key = json.loads(p["key"].read_text())
    human, ver_app, keep, ok_kept = [], [], [], []
    with open(p["audit"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = row["applies (Y/N)"].strip().upper()[:1]
            m = row["mentions_preference (Y/N)"].strip().upper()[:1]
            if a not in ("Y", "N"):
                continue
            k = key[row["audit_id"]]
            v = k["verify"] or {}
            human.append(a == "Y")
            ver_app.append(v.get("applies") == "yes")
            keep.append(k["keep"])
            if k["keep"]:
                ok_kept.append(a == "Y" and m == "N")
    res = {"n_labelled": len(human),
           "kept_precision": sum(ok_kept) / len(ok_kept) if ok_kept else float("nan"),
           "n_kept_labelled": len(ok_kept),
           "kappa_applies_verifier_vs_human": cohen_kappa(ver_app, human) if human else float("nan")}
    out = resolve_path(cfg.paths.tables) / "xtopic_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def main(argv: Sequence[str] | None = None) -> None:
    from ..config import load_config, parse_cli
    from ..utils import setup_logging

    setup_logging()
    sel, dot = parse_cli(argv if argv is not None else sys.argv[1:])
    stage = None
    rest = []
    for d in dot:
        if d.startswith("stage="):
            stage = d.split("=", 1)[1]
        else:
            rest.append(d)
    cfg = load_config(sel.get("experiment"), sel.get("profile"), rest)
    if stage == "generate":
        print(generate(cfg))
    elif stage == "verify":
        print(verify(cfg))
    elif stage == "select":
        print(json.dumps(select(cfg), indent=1))
    elif stage == "export_audit":
        print(export_audit(cfg))
    elif stage == "score_audit":
        print(json.dumps(score_audit(cfg), indent=1))
    elif stage == "all":          # local/smoke convenience
        generate(cfg)
        verify(cfg)
        print(json.dumps(select(cfg), indent=1))
    else:
        raise SystemExit("stage=generate|verify|select|export_audit|score_audit|all")


if __name__ == "__main__":
    main()
