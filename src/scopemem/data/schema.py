from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PrefItem(BaseModel):
    """One raw PrefEval (preference, question) pair."""

    id: str                      # "<topic>/<idx>"
    topic: str
    preference: str
    question: str
    explanation: str = ""
    form: Literal["explicit", "choice", "persona"] = "explicit"
    extra: dict[str, Any] = Field(default_factory=dict)   # implicit-form payload


class Preference(BaseModel):
    id: str                      # PrefItem.id, or "global/<k>"
    text: str
    topic: str                   # "global" for global style prefs
    is_global: bool = False
    checker: Optional[dict[str, Any]] = None


class HistorySeg(BaseModel):
    """Compact history segment. 'p' = preference turn, 'f' = one filler exchange."""

    t: Literal["p", "f"]
    id: Optional[str] = None     # pref id for 'p'
    c: Optional[int] = None      # filler conversation index for 'f'
    e: Optional[int] = None      # exchange index within that conversation


class Instance(BaseModel):
    instance_id: str
    N: int
    kind: Literal["positive", "null"]
    form: str = "explicit"
    query: str
    query_topic: str
    query_item_id: str
    target_id: Optional[str]
    global_id: Optional[str] = None
    pref_ids: list[str]
    gold_ids: list[str]
    fold: int
    history: list[HistorySeg]

    @field_validator("gold_ids")
    @classmethod
    def _gold_subset(cls, v: list[str], info: Any) -> list[str]:
        prefs = info.data.get("pref_ids") or []
        missing = [g for g in v if g not in prefs]
        if missing:
            raise ValueError(f"gold ids not in pref set: {missing}")
        return v


class ScopeJSON(BaseModel):
    """Strict schema the scope writer must return."""

    preference: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    example_requests: list[str] = Field(min_length=1, max_length=6)
    always_on: bool

    @field_validator("example_requests")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        v = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not v:
            raise ValueError("no example requests")
        return v


class ScopeRecord(BaseModel):
    pref_id: str
    preference: str              # canonical restatement (or original on fallback)
    original: str
    scope: str
    example_requests: list[str]
    always_on: bool
    fallback: bool = False
    attempts: int = 1
    write_input_tokens: int = 0
    write_output_tokens: int = 0
    writer_model: str = ""

    @property
    def write_tokens(self) -> int:
        return self.write_input_tokens + self.write_output_tokens
