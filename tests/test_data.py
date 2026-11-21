import json
from collections import Counter

import pytest

from scopemem.data.build_multi import (build_all, build_instances, materialize_history,
                                       stratified_counts)


def _build(d, N, **kw):
    return build_instances(d["items"], d["filler"], d["globals"], N, folds=d["folds"], seed=13, **kw)


def test_counts_and_null_share(tiny_data):
    insts = _build(tiny_data, 5)
    pos = [i for i in insts if i.kind == "positive"]
    nul = [i for i in insts if i.kind == "null"]
    assert len(pos) == len(tiny_data["items"]) == 40
    assert len(nul) == 10
    assert abs(len(nul) / len(insts) - 0.20) <= 0.01
    assert all(len(i.pref_ids) == 5 for i in insts)


def test_subset_counts_and_stratification(tiny_data):
    insts = _build(tiny_data, 5, n_pos=20, set_tag="sub")
    pos = [i for i in insts if i.kind == "positive"]
    assert len(pos) == 20 and len(insts) - len(pos) == 5
    assert set(Counter(i.query_topic for i in pos).values()) == {2}   # 10 topics x 2
    assert len({i.target_id for i in pos}) == 20                         # no replacement for targets


def test_gold_is_correct(tiny_data):
    topic_of = {it.id: it.topic for it in tiny_data["items"]}
    for i in _build(tiny_data, 5):
        assert set(i.gold_ids) <= set(i.pref_ids)
        non_global = [p for p in i.pref_ids if not p.startswith("global/")]
        if i.kind == "positive":
            assert i.target_id in i.gold_ids and i.pref_ids[0] == i.target_id
            assert all(topic_of[p] != i.query_topic for p in non_global if p != i.target_id)
            assert set(i.gold_ids) == {i.target_id} | ({i.global_id} if i.global_id else set())
        else:
            assert i.gold_ids == [] and i.global_id is None
            assert all(topic_of[p] != i.query_topic for p in non_global)
        assert len(set(i.pref_ids)) == len(i.pref_ids)


def test_global_rate_and_n1(tiny_data):
    insts = _build(tiny_data, 5, global_prob=0.3)
    rate = sum(i.global_id is not None for i in insts if i.kind == "positive") / 40
    assert 0.1 <= rate <= 0.5
    assert all(i.global_id is None for i in _build(tiny_data, 1))   # no room for a global at N=1


def test_no_topic_leakage_across_folds(tiny_data):
    folds = tiny_data["folds"]
    assert sorted(Counter(folds.values()).values()) == [2] * 5
    for i in _build(tiny_data, 5):
        assert i.fold == folds[i.query_topic]


def test_determinism_and_nesting(tiny_data):
    a = [i.model_dump() for i in _build(tiny_data, 5)]
    b = [i.model_dump() for i in _build(tiny_data, 5)]
    assert a == b
    small = {i.instance_id.replace("N3", ""): i for i in _build(tiny_data, 3)}
    for i in _build(tiny_data, 5):
        j = small[i.instance_id.replace("N5", "")]
        if i.global_id is None and j.global_id is None:
            assert set(j.pref_ids) <= set(i.pref_ids)


def test_history_contains_every_pref_once(tiny_data):
    from scopemem.data.build_multi import items_to_prefs

    prefs = {p.id: p for p in items_to_prefs(tiny_data["items"]) + tiny_data["globals"]}
    for i in _build(tiny_data, 5)[:10]:
        ps = [s.id for s in i.history if s.t == "p"]
        assert sorted(ps) == sorted(i.pref_ids)
        assert sum(s.t == "f" for s in i.history) == 3 * len(i.pref_ids)
        msgs = materialize_history(i, prefs, tiny_data["filler"], "ok")
        assert len(msgs) == 2 * len(i.history) and msgs[0]["role"] == "user"


def test_manifest_hashes_stable(tmp_path):
    from tests.conftest import make_cfg

    c1 = make_cfg(tmp_path / "a")
    c2 = make_cfg(tmp_path / "b")
    build_all(c1, sets=["full"], n_levels=[5])
    build_all(c2, sets=["full"], n_levels=[5])
    m1 = json.loads((tmp_path / "a" / "processed" / "manifest.json").read_text())
    m2 = json.loads((tmp_path / "b" / "processed" / "manifest.json").read_text())
    assert m1["files"] == m2["files"] and "prefeval_multi_N5.jsonl" in m1["files"]


@pytest.mark.parametrize("sizes,n", [({"a": 10, "b": 30}, 20), ({"a": 3, "b": 3, "c": 3}, 7), ({"a": 2}, 5)])
def test_stratified_counts(sizes, n):
    out = stratified_counts(sizes, n)
    assert sum(out.values()) == min(n, sum(sizes.values()))
    assert all(out[t] <= sizes[t] for t in sizes)


def test_personamem_and_locomo_loaders(tmp_path):
    from scopemem.data.locomo import load_locomo
    from scopemem.data.personamem import context_for, load_contexts, load_questions

    (tmp_path / "questions_32k.csv").write_text(
        "persona_id,question_id,question_type,topic,context_length_in_tokens,user_question_or_message,"
        "correct_answer,all_options,shared_context_id,end_index_in_shared_context\n"
        '0,q1,suggest_new_ideas,music,100,Any ideas?,(b),"[""(a) x"", ""(b) y""]",ctx,1\n', encoding="utf-8")
    (tmp_path / "shared_contexts_32k.jsonl").write_text(json.dumps(
        {"ctx": [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "later"}]}) + "\n", encoding="utf-8")
    q = load_questions(tmp_path)
    assert q[0].correct == "b" and q[0].options == ["(a) x", "(b) y"]
    assert len(context_for(q[0], load_contexts(tmp_path))) == 2
    (tmp_path / "locomo.json").write_text(json.dumps([{
        "sample_id": "s1", "conversation": {"speaker_a": "A", "speaker_b": "B", "session_2": [],
                                            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
                                            "session_1_date_time": "d1", "session_2_date_time": "d2"},
        "qa": [{"question": "q", "answer": 2022, "evidence": ["D1:1"], "category": 1},
               {"question": "adv", "adversarial_answer": "x", "evidence": [], "category": 5}]}]))
    conv = load_locomo(tmp_path / "locomo.json")[0]
    assert [s.index for s in conv.sessions] == [1, 2] and len(conv.qa) == 1 and conv.qa[0].answer == "2022"
