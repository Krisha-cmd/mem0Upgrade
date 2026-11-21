import json

from scopemem.data.schema import Preference
from scopemem.llm.cache import CachedLLM, SQLiteCache
from scopemem.llm.engine import GenParams
from scopemem.llm.fake import FakeLLM
from scopemem.llm.tokens import TokenizersJsonCounter, WhitespaceCounter, pref_token_delta
from scopemem.memory.scope_writer import ScopeWriter, parse_scope
from scopemem.prompting import build_messages, memory_block, truncate_history_left

P = GenParams()


def test_fake_llm_deterministic():
    m = [[{"role": "user", "content": "hello"}]]
    assert FakeLLM().generate(m, P)[0].text == FakeLLM().generate(m, P)[0].text
    assert FakeLLM().generate(m, P)[0].text != FakeLLM(name="other").generate(m, P)[0].text


def test_cache_hits_and_dedup(tmp_path):
    inner = FakeLLM()
    llm = CachedLLM(inner, SQLiteCache(tmp_path / "c.sqlite"))
    m = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    r1 = llm.generate(m, P)
    assert inner.calls == 2 and not r1[0].cached
    llm2 = CachedLLM(FakeLLM(), SQLiteCache(tmp_path / "c.sqlite"))
    r2 = llm2.generate(m, P)
    assert llm2.inner.calls == 0 and all(c.cached for c in r2) and [c.text for c in r1] == [c.text for c in r2]
    r3 = llm2.generate(m, GenParams(max_tokens=7))      # different params -> miss
    assert llm2.inner.calls == 2 and not r3[0].cached


VALID = json.dumps({"preference": "No flying", "scope": "Travel planning", "example_requests": ["Trip to Rome"],
                    "always_on": False})


def test_scope_parse_variants():
    assert parse_scope(VALID).scope == "Travel planning"
    assert parse_scope("```json\n" + VALID + "\n```") is not None
    assert parse_scope("Sure! " + VALID) is None                     # strict: no surrounding prose
    assert parse_scope('{"preference": "x", "scope": "y", "example_requests": [], "always_on": false}') is None
    assert parse_scope('{"preference": "x"}') is None


def test_scope_writer_valid_retry_fallback():
    prefs = [Preference(id="a/0", text="I hate flying", topic="a")]
    ok = ScopeWriter(FakeLLM(script=[VALID])).write(prefs)[0]
    assert not ok.fallback and ok.attempts == 1 and ok.write_tokens > 0
    retried = ScopeWriter(FakeLLM(script=["not json", VALID])).write(prefs)[0]
    assert not retried.fallback and retried.attempts == 2 and retried.write_tokens > ok.write_tokens
    fb = ScopeWriter(FakeLLM(script=["nope", "still nope"])).write(prefs)[0]
    assert fb.fallback and fb.scope == "I hate flying" and fb.example_requests == [] and fb.write_tokens > 0


def test_default_fake_scope_marks_global_always_on():
    w = ScopeWriter(FakeLLM())
    recs = w.write([Preference(id="g", text="Keep every answer under 150 words.", topic="global", is_global=True),
                    Preference(id="a", text="I am vegan.", topic="diet")])
    assert recs[0].always_on and not recs[1].always_on


def _tiny_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "<|im_start|>": 3, "<|im_end|>": 4}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    path = tmp_path / "tok.json"
    tok.save(str(path))
    return path


def test_token_counting_known_values(tmp_path):
    c = TokenizersJsonCounter(_tiny_tokenizer(tmp_path))
    assert c.count_text("hello world hello") == 3
    # "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n" -> 6 tokens (2 markers are special)
    assert c.count_messages([{"role": "user", "content": "hello"}]) == 6
    assert WhitespaceCounter().count_text("a b  c") == 3


def test_memory_block_counted_separately(smoke_cfg):
    c = WhitespaceCounter()
    with_b = build_messages(smoke_cfg, "plan a trip", ["I hate flying", "I am vegan"])
    without = build_messages(smoke_cfg, "plan a trip", [])
    block = memory_block(["I hate flying", "I am vegan"])
    assert pref_token_delta(c, with_b, without) == len(block.split())
    assert "User preferences:" not in without[0]["content"]
    assert with_b[-1] == {"role": "user", "content": "plan a trip"}


def test_left_truncation_keeps_recent(smoke_cfg):
    c = WhitespaceCounter()
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": f"u{i} " * 20}, {"role": "assistant", "content": f"a{i} " * 20}]
    full = c.count_messages(build_messages(smoke_cfg, "q", [], hist))
    h, trunc = truncate_history_left(smoke_cfg, c, "q", hist, full // 2)
    assert trunc and h[-1] == hist[-1] and len(h) % 2 == 0 and h[0]["role"] == "user"
    assert c.count_messages(build_messages(smoke_cfg, "q", [], h)) <= full // 2
    h2, t2 = truncate_history_left(smoke_cfg, c, "q", hist, full)
    assert not t2 and h2 == hist
