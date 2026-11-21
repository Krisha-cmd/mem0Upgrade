"""Track A preference store: clean preference text + its scope record, per user (instance)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..data.schema import Preference, ScopeRecord
from ..utils import read_jsonl


@dataclass(frozen=True)
class PrefRecord:
    id: str
    text: str
    topic: str
    is_global: bool
    scope: str
    examples: tuple[str, ...]
    always_on: bool
    write_tokens: int = 0

    def field_texts(self, fields: Sequence[str]) -> list[str]:
        """Texts a scope gate matches against. fields subset of {text, scope, examples}."""
        out: list[str] = []
        for f in fields:
            if f == "text":
                out.append(self.text)
            elif f == "scope":
                out.append(self.scope)
            elif f == "examples":
                out += list(self.examples)
            else:
                raise ValueError(f"unknown scope field {f}")
        return out or [self.text]

    def joined(self, fields: Sequence[str]) -> str:
        """Single-string view for cross-encoders: 'scope Examples: a; b; c'."""
        parts = []
        if "text" in fields:
            parts.append(self.text)
        if "scope" in fields:
            parts.append(self.scope)
        if "examples" in fields and self.examples:
            parts.append("Examples: " + "; ".join(self.examples))
        return " ".join(parts) or self.text


@dataclass
class ScopeBank:
    """pref_id -> ScopeRecord, loaded from data/processed/scopes/<writer>.jsonl."""

    records: dict[str, ScopeRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ScopeBank":
        return cls({r["pref_id"]: ScopeRecord(**r) for r in read_jsonl(path)})

    def get(self, pid: str) -> ScopeRecord | None:
        return self.records.get(pid)

    def fallback_rate(self) -> float:
        return sum(r.fallback for r in self.records.values()) / max(len(self.records), 1)


class PreferenceStore:
    def __init__(self, prefs: dict[str, Preference], scopes: ScopeBank | None = None):
        self.prefs = prefs
        self.scopes = scopes or ScopeBank()

    def record(self, pid: str) -> PrefRecord:
        p = self.prefs[pid]
        s = self.scopes.get(pid)
        if s is None:   # no scope written: behave like the fallback (scope = preference)
            return PrefRecord(p.id, p.text, p.topic, p.is_global, p.text, (), False, 0)
        return PrefRecord(p.id, p.text, p.topic, p.is_global, s.scope, tuple(s.example_requests),
                          s.always_on, s.write_tokens)

    def records(self, pids: Iterable[str]) -> list[PrefRecord]:
        return [self.record(pid) for pid in pids]
