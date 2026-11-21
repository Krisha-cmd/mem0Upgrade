import numpy as np
import pytest
from omegaconf import OmegaConf

from scopemem.data.schema import HistorySeg, Instance
from scopemem.embed import CachedCrossScorer, CachedEmbedder, HashCrossEncoder, HashEmbedder
from scopemem.gates import InjectAllGate, NoneGate, OracleGate
from scopemem.gates.llm_select import LLMSelectGate, parse_indices
from scopemem.gates.scope_embed import ScopeEmbedGate
from scopemem.gates.scope_xenc import ScopeXencGate
from scopemem.gates.sim_topk import SimRerankGate, SimTopKGate
from scopemem.gates.tune import Scored, prf, tune_k, tune_tau
from scopemem.llm.fake import FakeLLM
from scopemem.memory.store import PrefRecord

GC = OmegaConf.create({"k_max": 10, "rerank_top": 20})


def C(**kw):
    return OmegaConf.create(kw)


WORDS = ["apple", "bicycle", "cinema", "desert", "engine", "forest", "guitar", "harbour", "island",
         "jungle", "kitchen", "lantern"]


def W(i):
    return WORDS[i]


def _prefs(n=12):
    out = []
    for i in range(n):
        w = W(i)
        out.append(PrefRecord(f"t{i}/0", f"preference about {w} {w}", f"t{i}", False,
                              f"requests about {w}", (f"help with {w}", f"plan {w}"), False))
    out.append(PrefRecord("global/1", "keep answers short", "global", True, "every request", ("anything",), True))
    return out


def _inst(prefs, gold):
    return Instance(instance_id="x", N=len(prefs), kind="positive", query="q", query_topic="t0",
                    query_item_id="t0/0", target_id=gold[0], pref_ids=[p.id for p in prefs],
                    gold_ids=gold, fold=0, history=[HistorySeg(t="p", id=prefs[0].id)])


def test_inject_none_oracle():
    p = _prefs()
    inst = _inst(p, ["t3/0", "global/1"])
    assert InjectAllGate(C(id="B2")).select("q", p).selected_ids == [x.id for x in p]
    assert NoneGate(C(id="B0")).select("q", p).selected_ids == []
    assert sorted(OracleGate(C(id="UB")).select("q", p, inst).selected_ids) == ["global/1", "t3/0"]
    sel, oh = InjectAllGate(C(id="B2")).select("q", p)       # tuple-unpacking form of the spec
    assert oh == 0 and len(sel) == len(p)


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_sim_topk_exactly_k(k):
    g = SimTopKGate(C(id="B3", gate="sim_topk", k=k), GC, CachedEmbedder(HashEmbedder()))
    s = g.select("help me plan " + W(4), _prefs())
    assert len(s.selected_ids) == k and s.overhead_tokens == 0
    assert s.selected_ids[0] == "t4/0"


def test_sim_rerank_shortlist_then_k():
    g = SimRerankGate(C(id="B4", gate="sim_rerank", k=2, rerank_top=3), GC,
                      CachedEmbedder(HashEmbedder()), CachedCrossScorer(HashCrossEncoder()))
    s = g.select(W(5) + " stuff", _prefs())
    assert len(s.selected_ids) == 2 and "t5/0" in s.selected_ids


def _scope_gate(kind, **kw):
    cond = C(id="S", gate=kind, fields=["scope", "examples"], tau=kw.get("tau", 0.5),
             use_always_on=kw.get("always", True), k_max=kw.get("k_max", 10))
    if kind == "scope_embed":
        return ScopeEmbedGate(cond, GC, CachedEmbedder(HashEmbedder()))
    return ScopeXencGate(cond, GC, CachedCrossScorer(HashCrossEncoder()))


@pytest.mark.parametrize("kind", ["scope_embed", "scope_xenc"])
def test_scope_gates_tau_always_on_kmax(kind):
    p = _prefs()
    g = _scope_gate(kind, tau=0.3)
    s = g.select("plan " + W(7), p)
    assert "global/1" in s.selected_ids and "t7/0" in s.selected_ids and s.overhead_tokens == 0
    # tau above every score -> only always-on
    assert _scope_gate(kind, tau=10.0).select("plan " + W(7), p).selected_ids == ["global/1"]
    # always_on disabled -> global treated like any other pref
    assert "global/1" not in _scope_gate(kind, tau=10.0, always=False).select("plan " + W(7), p).selected_ids
    # k_max caps non-always-on selections and reports binding
    s = _scope_gate(kind, tau=-10.0, k_max=3).select("plan " + W(7), p)
    assert len([x for x in s.selected_ids if x != "global/1"]) == 3 and s.cap_bound
    s = _scope_gate(kind, tau=-10.0, k_max=None).select("plan " + W(7), p)
    assert len(s.selected_ids) == len(p) and not s.cap_bound


def test_scores_sorted_under_cap():
    g = _scope_gate("scope_embed", tau=-1.0, k_max=2)
    p = _prefs()
    scores = np.arange(len(p), dtype=np.float32)
    sel = g.apply(scores, p)
    assert sel.selected_ids[:2] == [p[-2].id, p[-3].id]   # highest non-always-on first


def test_llm_select_overhead_and_parse():
    llm = FakeLLM()
    g = LLMSelectGate(C(id="B5", gate="llm_select"), GC, llm)
    s = g.select("plan " + W(2) + " please", _prefs())
    assert "t2/0" in s.selected_ids and s.overhead_tokens > 0
    assert parse_indices("Answer: [1, 3, 99]", 5) == [1, 3]
    assert parse_indices("none", 5) is None
    bad = LLMSelectGate(C(id="B5", gate="llm_select"), GC, FakeLLM(script=["I think none"]))
    s = bad.select("q", _prefs())
    assert s.selected_ids == [] and s.meta["parse_ok"] is False


def test_tuning_picks_separating_threshold_and_k():
    g = _scope_gate("scope_embed", tau="tune", always=False)
    g.tau = None
    p = _prefs(4)
    scored = [Scored(np.array([0.9, 0.2, 0.1, 0.3, 0.0], dtype=np.float32), p, frozenset({"t0/0"})),
              Scored(np.array([0.1, 0.8, 0.2, 0.25, 0.0], dtype=np.float32), p, frozenset({"t1/0"}))]
    tau, f1 = tune_tau(g, scored, 101)
    assert f1 == 1.0 and 0.3 < tau <= 0.8
    gk = SimTopKGate(C(id="B3", gate="sim_topk", k="tune"), GC, None)
    k, f1k = tune_k(gk, scored, [1, 2, 3])
    assert k == 1 and f1k == 1.0
    assert prf([["a"], []], [frozenset({"a"}), frozenset()]) == (1.0, 1.0, 1.0)
