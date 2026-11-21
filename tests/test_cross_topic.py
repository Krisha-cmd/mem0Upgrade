import json

from scopemem.data import cross_topic as xt
from scopemem.data.build_multi import build_all, build_instances, load_globals, load_instances
from scopemem.data.prefeval import list_topics, load_filler, load_items
from tests.conftest import make_cfg

TOPICS = ["a", "b", "c", "d"]


def test_parse_generation():
    ok = json.dumps([{"topic": "b", "request": "Plan a honeymoon", "why": "x"},
                     {"topic": "a", "request": "home topic: dropped"},
                     {"topic": "b", "request": "duplicate topic: dropped"},
                     {"topic": "zzz", "request": "unknown topic: dropped"},
                     {"topic": "c", "request": "Pick a gift"}])
    r = xt.parse_generation(ok, "a", TOPICS)
    assert [x["query_topic"] for x in r] == ["b", "c"]
    assert xt.parse_generation("```json\n" + ok + "\n```", "a", TOPICS) is not None
    assert xt.parse_generation("not json", "a", TOPICS) is None
    assert xt.parse_generation(json.dumps([{"topic": "a", "request": "x"}]), "a", TOPICS) is None


def test_parse_verification_keep_rule():
    v = xt.parse_verification('{"applies": "yes", "mentions": "no", "natural": "yes", "reason": "r"}')
    assert v["keep"]
    assert not xt.parse_verification('{"applies": "yes", "mentions": "yes", "natural": "yes"}')["keep"]
    assert not xt.parse_verification('{"applies": "no", "mentions": "no", "natural": "yes"}')["keep"]
    assert xt.parse_verification('{"applies": "maybe", "mentions": "no", "natural": "yes"}') is None


def _pipeline(tmp_path):
    cfg = make_cfg(tmp_path)
    build_all(cfg, sets=["full"], n_levels=[5])
    xt.generate(cfg)
    xt.verify(cfg)
    stats = xt.select(cfg)
    return cfg, stats


def test_fake_pipeline_and_xtopic_instances(tmp_path):
    cfg, stats = _pipeline(tmp_path)
    assert stats["candidates"] > 0 and 0 < stats["kept"] <= stats["candidates"]
    items = xt.load_xtopic_items(tmp_path / "processed")
    assert all(i["query_topic"] != i["home_topic"] for i in items)
    prim = [i for i in items if i["role"] == "primary"]
    assert len({i["pref_id"] for i in prim}) == len(prim)          # one primary per preference
    rep = build_all(cfg, sets=["xtopic"], n_levels=[5])
    assert rep["sets"]["xtopic"][5]["positive"] == len(prim)
    folds = json.loads((tmp_path / "processed" / "folds.json").read_text())
    topic_of = {it.id: it.topic for it in load_items(cfg.paths.prefeval_root)}
    insts = load_instances(tmp_path / "processed", 5, "xtopic")
    assert abs(sum(i.kind == "null" for i in insts) / len(insts) - 0.2) <= 0.02
    for i in insts:
        assert i.home_topic and i.query_topic != i.home_topic
        assert i.fold == folds[i.home_topic]
        others = [p for p in i.pref_ids if not p.startswith("global/") and p != i.target_id]
        assert all(topic_of[p] not in (i.query_topic, i.home_topic) for p in others)
        if i.kind == "positive":
            assert i.target_id in i.gold_ids and topic_of[i.target_id] == i.home_topic
        else:
            assert i.gold_ids == []


def test_prefeval_instances_unchanged_by_refactor(tiny_data):
    """For PrefEval targets home_topic == query_topic and the fold follows the topic."""
    insts = build_instances(tiny_data["items"], tiny_data["filler"], tiny_data["globals"], 5,
                            folds=tiny_data["folds"], seed=13)
    assert all(i.home_topic == i.query_topic and i.fold == tiny_data["folds"][i.query_topic] for i in insts)


def test_s3_pairs_use_only_training_fold_xtopic_items(tmp_path):
    from scopemem.embed import CachedEmbedder, HashEmbedder
    from scopemem.gates.scope_ft import build_training_pairs
    from scopemem.runtime import load_store

    cfg, _ = _pipeline(tmp_path)
    store = load_store(cfg)
    items = load_items(cfg.paths.prefeval_root)
    folds = json.loads((tmp_path / "processed" / "folds.json").read_text())
    xitems = xt.load_xtopic_items(tmp_path / "processed")
    emb = CachedEmbedder(HashEmbedder())
    base = build_training_pairs(items, store, folds, 0, emb, ["scope", "examples"], 13, global_prob=0.0)
    withx = build_training_pairs(items, store, folds, 0, emb, ["scope", "examples"], 13, global_prob=0.0,
                                 xtopic_items=xitems)
    allowed = [x for x in xitems if folds[x["home_topic"]] != 0 and folds[x["query_topic"]] != 0]
    assert len(withx) - len(base) == 7 * len(allowed)              # 1 positive + 3 hard + 3 random each
    test_requests = {x["request"] for x in xitems if folds[x["home_topic"]] == 0 or folds[x["query_topic"]] == 0}
    assert not (test_requests - {x["request"] for x in allowed}) & {q for q, _, _ in withx}


def test_audit_export_and_score(tmp_path, monkeypatch):
    cfg, _ = _pipeline(tmp_path)
    import csv

    import scopemem.data.cross_topic as mod

    paths = mod._paths(cfg)
    ann = tmp_path / "ann"
    mod_paths = {**paths, "audit": ann / "a.csv", "key": ann / "k.json"}
    monkeypatch.setattr(mod, "_paths", lambda c: mod_paths)            # keep the test inside tmp_path
    if True:
        out = xt.export_audit(cfg)
        rows = list(csv.reader(open(out, encoding="utf-8")))
        assert rows[0][:3] == ["audit_id", "preference", "request"] and len(rows) > 1
        assert "keep" not in ",".join(rows[0])                         # blind: no verifier labels
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0])
            for r in rows[1:]:
                w.writerow(r[:3] + ["Y", "N", ""])
        res = xt.score_audit(cfg)
        assert res["n_labelled"] == len(rows) - 1 and res["kept_precision"] == 1.0
