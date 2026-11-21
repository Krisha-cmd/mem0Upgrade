from __future__ import annotations

from .base import Gate, Selection


class InjectAllGate(Gate):
    """B2: every stored preference goes into the prompt."""

    def __init__(self, cond=None, *_):
        self.id = str(cond.id) if cond is not None else "B2_all"

    def select(self, query, prefs, instance=None) -> Selection:
        return Selection([p.id for p in prefs], 0)
