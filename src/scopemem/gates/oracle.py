from __future__ import annotations

from .base import Gate, Selection


class OracleGate(Gate):
    """UB: exactly the gold applicable set G (needs the instance)."""

    def __init__(self, cond=None, *_):
        self.id = str(cond.id) if cond is not None else "UB_oracle"

    def select(self, query, prefs, instance=None) -> Selection:
        if instance is None:
            raise ValueError("OracleGate needs the instance (gold ids)")
        gold = set(instance.gold_ids)
        return Selection([p.id for p in prefs if p.id in gold], 0)
