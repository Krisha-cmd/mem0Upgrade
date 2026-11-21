"""PrefEval-Multi builder (section 5.2): users with N preferences, one query, gold applicable set G.

Construction (deviations from the prompt are logged in docs/DECISIONS.md):
- positive: target item t; P = {t.pref} + (N-1) prefs from OTHER topics; G = {t.pref}.
- null (20% of instances): query from topic T; P = N prefs from topics != T; G = {} (D-010).
- global: with p=0.3 a positive user gets one global style pref (replacing one distractor so |P|=N,
  only when N >= 2); it is added to G (D-009).
- Distractor and filler draws are seeded per instance, not per N, so P at N=5 is a subset of P at
  N=10 for the same instance (nested; gives paired comparisons across N in E3).
- folds: 5 folds by topic (4 topics each); an instance's fold is its query topic's fold.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from ..config import derive_seed, load_config, resolve_path
from ..utils import sha256_file, write_jsonl
from .prefeval import implicit_history_messages, list_topics, load_filler, load_items
from .schema import HistorySeg, Instance, Preference, PrefItem

GLOBAL_PREFS_PATH = Path(__file__).parent / "global_prefs.yaml"


def load_globals(path: str | Path = GLOBAL_PREFS_PATH) -> list[Preference]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Preference(id=r["id"], text=r["text"], topic="global", is_global=True,
                       checker=r.get("checker")) for r in rows]


def make_topic_folds(topics: Sequence[str], n_folds: int, seed: int) -> dict[str, int]:
    ts = sorted(topics)
    random.Random(derive_seed(seed, "folds")).shuffle(ts)
    per = len(ts) // n_folds
    if per * n_folds != len(ts):
        raise ValueError(f"{len(ts)} topics do not split evenly into {n_folds} folds")
    return {t: i // per for i, t in enumerate(ts)}


def items_to_prefs(items: Iterable[PrefItem]) -> list[Preference]:
    return [Preference(id=it.id, text=it.preference, topic=it.topic) for it in items]


def stratified_counts(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n across topics proportional to size, capped at size."""
    total = sum(sizes.values())
    n = min(n, total)
    raw = {t: n * s / total for t, s in sizes.items()}
    alloc = {t: min(int(v), sizes[t]) for t, v in raw.items()}
    rest = sorted(sizes, key=lambda t: (-(raw[t] - int(raw[t])), t))
    i = 0
    while sum(alloc.values()) < n:
        t = rest[i % len(rest)]
        if alloc[t] < sizes[t]:
            alloc[t] += 1
        i += 1
    return alloc


def _dedup_pool(prefs: Sequence[Preference]) -> list[Preference]:
    seen: set[str] = set()
    out = []
    for p in prefs:
        key = p.text.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def select_targets(items: Sequence[PrefItem], n_pos: int | None, seed: int) -> list[PrefItem]:
    """All items (id order) if n_pos is None/too large; else topic-stratified, no replacement."""
    if n_pos is None or n_pos >= len(items):
        return list(items)
    by_topic: dict[str, list[PrefItem]] = defaultdict(list)
    for it in items:
        by_topic[it.topic].append(it)
    alloc = stratified_counts({t: len(v) for t, v in by_topic.items()}, n_pos)
    rng = random.Random(derive_seed(seed, "targets", n_pos))
    chosen = []
    for t in sorted(by_topic):
        chosen += rng.sample(by_topic[t], alloc[t])
    return sorted(chosen, key=lambda it: (it.topic, int(it.id.split("/")[1])))


def build_instances(items: Sequence[PrefItem], filler: Sequence[Sequence[tuple[str, str]]],
                    globals_: Sequence[Preference], N: int, *, n_pos: int | None = None,
                    null_frac: float = 0.2, global_prob: float = 0.3, filler_per_pref: int = 3,
                    folds: dict[str, int], seed: int = 13, form: str = "explicit",
                    set_tag: str = "full") -> list[Instance]:
    pool = _dedup_pool(items_to_prefs(items))
    by_topic_items: dict[str, list[PrefItem]] = defaultdict(list)
    for it in items:
        by_topic_items[it.topic].append(it)
    topics = sorted(by_topic_items)
    flat_filler = [(c, e) for c, conv in enumerate(filler) for e in range(len(conv))]
    targets = select_targets(items, n_pos, seed)
    n_null = round(len(targets) * null_frac / (1.0 - null_frac))
    out: list[Instance] = []

    def history_for(pref_ids: list[str], key: str) -> list[HistorySeg]:
        rng = random.Random(derive_seed(seed, "hist", form, set_tag, key, N))
        order = list(pref_ids)
        rng.shuffle(order)
        need = len(order) * filler_per_pref
        fill = (rng.sample(flat_filler, need) if need <= len(flat_filler)
                else [rng.choice(flat_filler) for _ in range(need)])
        segs: list[HistorySeg] = []
        for i, pid in enumerate(order):
            segs.append(HistorySeg(t="p", id=pid))
            for c, e in fill[i * filler_per_pref:(i + 1) * filler_per_pref]:
                segs.append(HistorySeg(t="f", c=c, e=e))
        return segs

    def distractors(exclude_topic: str, key: str) -> list[Preference]:
        rng = random.Random(derive_seed(seed, "distr", form, set_tag, key))
        cand = [p for p in pool if p.topic != exclude_topic]
        rng.shuffle(cand)
        return cand

    for j, t in enumerate(targets):
        key = f"pos{j}"
        rng = random.Random(derive_seed(seed, "global", form, set_tag, key))
        g = rng.choice(globals_) if (globals_ and rng.random() < global_prob) else None
        use_g = g is not None and N >= 2
        n_d = N - 1 - (1 if use_g else 0)
        d = distractors(t.topic, key)[:n_d]
        pref_ids = [t.id] + [p.id for p in d] + ([g.id] if use_g and g else [])
        gold = [t.id] + ([g.id] if use_g and g else [])
        out.append(Instance(
            instance_id=f"{form}-{set_tag}-N{N}-p{j:04d}", N=N, kind="positive", form=form,
            query=t.question, query_topic=t.topic, query_item_id=t.id, target_id=t.id,
            global_id=g.id if use_g and g else None, pref_ids=pref_ids, gold_ids=gold,
            fold=folds[t.topic], history=history_for(pref_ids, key)))

    topic_order = list(topics)
    random.Random(derive_seed(seed, "nulltopics", set_tag)).shuffle(topic_order)
    for j in range(n_null):
        key = f"null{j}"
        rng = random.Random(derive_seed(seed, "nullq", form, set_tag, key))
        qt = topic_order[j % len(topic_order)]
        q_item = rng.choice(by_topic_items[qt])          # with replacement (D-010)
        d = distractors(qt, key)[:N]
        pref_ids = [p.id for p in d]
        out.append(Instance(
            instance_id=f"{form}-{set_tag}-N{N}-n{j:04d}", N=N, kind="null", form=form,
            query=q_item.question, query_topic=qt, query_item_id=q_item.id, target_id=None,
            pref_ids=pref_ids, gold_ids=[], fold=folds[qt], history=history_for(pref_ids, key)))
    return out


def materialize_history(inst: Instance, prefs_by_id: dict[str, Preference],
                        filler: Sequence[Sequence[tuple[str, str]]], ack: str,
                        items_by_id: dict[str, PrefItem] | None = None) -> list[dict[str, str]]:
    """Expand compact history into chat messages. Implicit forms use the revealing conversation."""
    msgs: list[dict[str, str]] = []
    for s in inst.history:
        if s.t == "p":
            assert s.id is not None
            it = items_by_id.get(s.id) if items_by_id else None
            if it is not None and it.form != "explicit":
                msgs += implicit_history_messages(it)
            else:
                msgs += [{"role": "user", "content": prefs_by_id[s.id].text},
                         {"role": "assistant", "content": ack}]
        else:
            assert s.c is not None and s.e is not None
            u, a = filler[s.c][s.e]
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return msgs


# ---------------------------------------------------------------------------------------------
def dataset_filename(N: int, set_tag: str, form: str = "explicit") -> str:
    f = "" if form == "explicit" else f"{form}_"
    s = "" if set_tag == "full" else f"_{set_tag}"
    return f"prefeval_multi_{f}N{N}{s}.jsonl"


def load_instances(processed: str | Path, N: int, set_tag: str = "full",
                   form: str = "explicit") -> list[Instance]:
    from ..utils import read_jsonl

    path = Path(processed) / dataset_filename(N, set_tag, form)
    return [Instance(**r) for r in read_jsonl(path)]


def load_pref_table(processed: str | Path) -> dict[str, Preference]:
    from ..utils import read_jsonl

    return {r["id"]: Preference(**r) for r in read_jsonl(Path(processed) / "prefs.jsonl")}


def summarize(insts: Sequence[Instance]) -> dict:
    n = len(insts)
    nul = sum(i.kind == "null" for i in insts)
    pos = [i for i in insts if i.kind == "positive"]
    return {
        "n": n, "positive": n - nul, "null": nul, "null_share": round(nul / max(n, 1), 4),
        "with_global": sum(i.global_id is not None for i in insts),
        "mean_gold": round(sum(len(i.gold_ids) for i in pos) / max(len(pos), 1), 3),
        "mean_history_exchanges": round(sum(len(i.history) for i in insts) / max(n, 1), 1),
        "per_fold": dict(sorted(Counter(i.fold for i in insts).items())),
        "topics": len({i.query_topic for i in insts}),
    }


def build_all(cfg, sets: Sequence[str] = ("full", "sub500"), n_levels: Sequence[int] | None = None,
              form: str = "explicit", write: bool = True) -> dict:
    root = resolve_path(cfg.paths.prefeval_root)
    out_dir = resolve_path(cfg.paths.processed)
    items = load_items(root, form)
    filler = load_filler(root)
    globals_ = load_globals()
    folds = make_topic_folds(list_topics(root), int(cfg.data.n_folds), int(cfg.seed))
    levels = list(n_levels or cfg.data.n_levels)
    report: dict = {"form": form, "items": len(items), "topics": len(folds),
                    "filler_conversations": len(filler),
                    "filler_exchanges": sum(len(c) for c in filler), "sets": {}}
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        allp = items_to_prefs(load_items(root, "explicit")) + globals_
        write_jsonl(out_dir / "prefs.jsonl", (p.model_dump() for p in allp))
        (out_dir / "folds.json").write_text(json.dumps(folds, indent=1, sort_keys=True))
        for name in ("prefs.jsonl", "folds.json"):
            manifest["files"][name] = sha256_file(out_dir / name)
    for set_tag in sets:
        n_pos = None if set_tag == "full" else int(cfg.data.sub_n_pos)
        for N in levels:
            insts = build_instances(items, filler, globals_, N, n_pos=n_pos,
                                    null_frac=float(cfg.data.null_frac),
                                    global_prob=float(cfg.data.global_prob),
                                    filler_per_pref=int(cfg.data.filler_per_pref), folds=folds,
                                    seed=int(cfg.seed), form=form, set_tag=set_tag)
            report["sets"].setdefault(set_tag, {})[N] = summarize(insts)
            if write:
                fn = dataset_filename(N, set_tag, form)
                write_jsonl(out_dir / fn, (i.model_dump(exclude_none=False) for i in insts))
                manifest["files"][fn] = sha256_file(out_dir / fn)
    if write:
        manifest["params"] = {"seed": int(cfg.seed), "null_frac": float(cfg.data.null_frac),
                              "global_prob": float(cfg.data.global_prob),
                              "filler_per_pref": int(cfg.data.filler_per_pref),
                              "n_folds": int(cfg.data.n_folds), "sub_n_pos": int(cfg.data.sub_n_pos)}
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build PrefEval-Multi")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--sets", nargs="+", default=["full", "sub500"])
    ap.add_argument("--n-levels", nargs="+", type=int, default=None)
    ap.add_argument("--form", default="explicit", choices=["explicit", "choice", "persona"])
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(None, a.profile, a.overrides)
    rep = build_all(cfg, a.sets, a.n_levels, a.form, write=not a.stats_only)
    print(f"form={rep['form']} items={rep['items']} topics={rep['topics']} "
          f"filler={rep['filler_conversations']} conversations / {rep['filler_exchanges']} exchanges")
    for set_tag, levels in rep["sets"].items():
        for N, s in levels.items():
            print(f"[{set_tag:6s}] N={N:<3d} n={s['n']:<5d} pos={s['positive']:<5d} null={s['null']:<4d} "
                  f"null_share={s['null_share']:.3f} global={s['with_global']:<4d} "
                  f"mean|G|={s['mean_gold']:.2f} hist_segs={s['mean_history_exchanges']} folds={s['per_fold']}")


if __name__ == "__main__":
    main()
