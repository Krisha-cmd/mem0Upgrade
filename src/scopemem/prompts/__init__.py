"""Versioned prompt templates. Each file has a header terminated by '### END HEADER ###'."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent
_SEP = "### END HEADER ###\n"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """``name`` like 'scope_writer' or 'judge_prefeval/check_violation'."""
    text = (_DIR / f"{name}.txt").read_text(encoding="utf-8")
    if _SEP in text:
        text = text.split(_SEP, 1)[1]
    return text
