"""Scope writer: one LLM call per preference at store time -> strict scope JSON (section 6.2).

Validation with pydantic; one re-ask on invalid JSON; fallback scope = preference if the retry
also fails. Write tokens (input + output, all attempts) are recorded per preference.
JSON may be wrapped in a ```json fence; any other surrounding text is invalid (D-015).

CLI:  python -m scopemem.memory.scope_writer experiment=E1 model=qwen7b [profile=...]
      -> data/processed/scopes/<writer>.jsonl  (resumable)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from ..data.schema import Preference, ScopeJSON, ScopeRecord
from ..llm.engine import GenParams, LLM
from ..prompts import load_prompt

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
RETRY_MSG = ("Your previous reply was not valid JSON in the required schema. Reply again with ONLY "
             "the JSON object, with keys preference, scope, example_requests, always_on.")


def parse_scope(text: str) -> ScopeJSON | None:
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return ScopeJSON(**obj)
    except (ValidationError, TypeError):
        return None


class ScopeWriter:
    def __init__(self, llm: LLM, params: GenParams | None = None):
        self.llm = llm
        self.params = params or GenParams(temperature=0.0, max_tokens=256)
        self.template = load_prompt("scope_writer")

    def _messages(self, pref: Preference) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.template.format(preference=pref.text)}]

    def write(self, prefs: Sequence[Preference]) -> list[ScopeRecord]:
        msgs = [self._messages(p) for p in prefs]
        first = self.llm.generate(msgs, self.params)
        parsed = [parse_scope(c.text) for c in first]
        tin = [c.prompt_tokens for c in first]
        tout = [c.completion_tokens for c in first]
        attempts = [1] * len(prefs)
        retry_idx = [i for i, p in enumerate(parsed) if p is None]
        if retry_idx:
            rmsgs = [msgs[i] + [{"role": "assistant", "content": first[i].text},
                                {"role": "user", "content": RETRY_MSG}] for i in retry_idx]
            second = self.llm.generate(rmsgs, self.params)
            for i, c in zip(retry_idx, second):
                parsed[i] = parse_scope(c.text)
                tin[i] += c.prompt_tokens
                tout[i] += c.completion_tokens
                attempts[i] = 2
        out = []
        for i, p in enumerate(prefs):
            sj = parsed[i]
            if sj is None:
                out.append(ScopeRecord(pref_id=p.id, preference=p.text, original=p.text, scope=p.text,
                                       example_requests=[], always_on=False, fallback=True,
                                       attempts=attempts[i], write_input_tokens=tin[i],
                                       write_output_tokens=tout[i], writer_model=self.llm.name))
            else:
                out.append(ScopeRecord(pref_id=p.id, preference=sj.preference, original=p.text,
                                       scope=sj.scope, example_requests=sj.example_requests,
                                       always_on=sj.always_on, fallback=False, attempts=attempts[i],
                                       write_input_tokens=tin[i], write_output_tokens=tout[i],
                                       writer_model=self.llm.name))
        return out


def scopes_path(cfg, writer_name: str) -> Path:
    from ..config import resolve_path

    return resolve_path(cfg.paths.processed) / "scopes" / f"{writer_name}.jsonl"


def writer_model_cfg(cfg):
    from ..config import load_named

    if cfg.llm_backend == "fake":
        return cfg.model_cfg
    return load_named("models", cfg.scope_writer) if cfg.get("scope_writer") else cfg.model_cfg


def writer_name(cfg) -> str:
    """Name used for the scope file, fine-tuned S3 dirs and the writer's LLM cache."""
    return str(writer_model_cfg(cfg).name)


def ensure_scopes(cfg, prefs: Sequence[Preference], llm: LLM | None = None,
                  batch: int = 512) -> Path:
    """Write scopes for every pref not yet in the scope file. Returns the file path."""
    from ..llm.engine import build_llm
    from ..utils import JsonlAppender, done_ids

    mc = writer_model_cfg(cfg)
    llm = llm or build_llm(cfg, mc)
    path = scopes_path(cfg, llm.name)
    done = done_ids(path, key="pref_id")
    todo = [p for p in prefs if p.id not in done]
    writer = ScopeWriter(llm)
    with JsonlAppender(path) as app:
        for i in range(0, len(todo), batch):
            for rec in writer.write(todo[i:i + batch]):
                app.write(rec.model_dump())
    return path


def main(argv: Sequence[str] | None = None) -> None:
    from ..config import load_config, parse_cli, resolve_path
    from ..data.build_multi import load_pref_table
    from ..memory.store import ScopeBank
    from ..utils import setup_logging

    setup_logging()
    sel, dot = parse_cli(argv if argv is not None else sys.argv[1:])
    cfg = load_config(sel.get("experiment"), sel.get("profile"), dot)
    prefs = list(load_pref_table(resolve_path(cfg.paths.processed)).values())
    path = ensure_scopes(cfg, prefs)
    bank = ScopeBank.load(path)
    n = len(bank.records)
    wt = sum(r.write_tokens for r in bank.records.values())
    print(f"scopes: {n} -> {path}  fallback_rate={bank.fallback_rate():.4f}  "
          f"always_on={sum(r.always_on for r in bank.records.values())}  mean_write_tokens={wt / max(n, 1):.1f}")


if __name__ == "__main__":
    main()
