from __future__ import annotations

from .base import Gate, Selection


class NoneGate(Gate):
    """B0 (and B1, whose preferences arrive through the history instead of a memory block)."""

    def __init__(self, cond=None, *_):
        self.id = str(cond.id) if cond is not None else "B0_none"

    def select(self, query, prefs, instance=None) -> Selection:
        return Selection([], 0)
