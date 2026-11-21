"""Gate registry. ``GateContext`` lazily builds the shared embedder / cross-encoder / LLM."""

from __future__ import annotations

from functools import cached_property
from typing import Any

from .base import Gate, ScoringGate, Selection  # noqa: F401
from .inject_all import InjectAllGate
from .llm_select import LLMSelectGate
from .none import NoneGate
from .oracle import OracleGate
from .scope_embed import ScopeEmbedGate
from .scope_ft import ScopeFTGate, ft_model_dir
from .scope_xenc import ScopeXencGate
from .sim_topk import SimRerankGate, SimTopKGate


class GateContext:
    def __init__(self, cfg: Any, llm: Any = None):
        self.cfg = cfg
        self._llm = llm
        self._ft: dict[int, Any] = {}

    @cached_property
    def embedder(self):
        from ..embed import build_embedder

        return build_embedder(self.cfg.embedder_cfg)

    @cached_property
    def cross(self):
        from ..embed import build_cross

        return build_cross(self.cfg.xenc_cfg)

    @property
    def llm(self):
        if self._llm is None:
            from ..llm.engine import build_llm

            self._llm = build_llm(self.cfg)
        return self._llm

    def ft_cross(self, fold: int):
        if fold not in self._ft:
            from ..embed import build_cross
            from ..memory.scope_writer import writer_name

            if self.cfg.xenc_cfg.type == "hash":
                self._ft[fold] = build_cross(self.cfg.xenc_cfg)
            else:
                d = ft_model_dir(self.cfg, writer_name(self.cfg), fold)
                if not (d / "config.json").exists():
                    raise FileNotFoundError(f"S3 model for fold {fold} missing at {d}; run scope_ft first")
                self._ft[fold] = build_cross(self.cfg.xenc_cfg, model_path=str(d))
        return self._ft[fold]


def build_gate(cond: Any, ctx: GateContext, fold: int | None = None) -> Gate:
    g = str(cond.gate)
    gc = ctx.cfg.gate
    if g == "none":
        return NoneGate(cond)
    if g == "inject_all":
        return InjectAllGate(cond)
    if g == "oracle":
        return OracleGate(cond)
    if g == "sim_topk":
        return SimTopKGate(cond, gc, ctx.embedder)
    if g == "sim_rerank":
        return SimRerankGate(cond, gc, ctx.embedder, ctx.cross)
    if g == "scope_embed":
        return ScopeEmbedGate(cond, gc, ctx.embedder)
    if g == "scope_xenc":
        return ScopeXencGate(cond, gc, ctx.cross)
    if g == "scope_ft":
        if fold is None:
            raise ValueError("scope_ft gate is per fold")
        return ScopeFTGate(cond, gc, ctx.ft_cross(fold))
    if g == "llm_select":
        return LLMSelectGate(cond, gc, ctx.llm)
    raise ValueError(f"unknown gate {g}")


def is_fold_specific(cond: Any) -> bool:
    return str(cond.gate) == "scope_ft"
