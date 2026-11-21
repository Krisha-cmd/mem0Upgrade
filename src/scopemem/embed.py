"""Embedders (bi-encoders) and cross-encoders, plus deterministic hash fakes for tests/smoke.

All embeddings are L2-normalised, so cosine similarity is a dot product (section 13 default).
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

import numpy as np

from .llm.fake import content_words


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _feats(text: str) -> list[str]:
    ws = [_stem(w) for w in content_words(text)]
    return ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


class HashEmbedder:
    """Signed feature hashing over stemmed unigrams+bigrams. Deterministic across processes."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash{dim}"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in _feats(t):
                h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
                m[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _normalize(m)


class STEmbedder:
    def __init__(self, hf_id: str, query_prefix: str = "", batch_size: int = 256, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(hf_id, device=device)
        self.name = hf_id
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        texts = [self.query_prefix + t for t in texts] if (is_query and self.query_prefix) else list(texts)
        v = self._m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)


class CachedEmbedder:
    """Memoises text -> vector so each unique string is embedded once per process."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name = inner.name
        self._memo: dict[tuple[str, bool], np.ndarray] = {}

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        miss = list(dict.fromkeys(t for t in texts if (t, is_query) not in self._memo))
        if miss:
            vecs = self.inner.encode(miss, is_query=is_query)
            for t, v in zip(miss, vecs):
                self._memo[(t, is_query)] = v
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._memo[(t, is_query)] for t in texts])


def build_embedder(ecfg: Any) -> CachedEmbedder:
    if ecfg.type == "hash":
        return CachedEmbedder(HashEmbedder(int(ecfg.get("dim", 512))))
    if ecfg.type == "st":
        return CachedEmbedder(STEmbedder(ecfg.hf_id, ecfg.get("query_prefix", "") or "",
                                         int(ecfg.get("batch_size", 256))))
    raise ValueError(ecfg.type)


# --- cross-encoders --------------------------------------------------------------------------
class CrossScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class HashCrossEncoder:
    """Fake CE: cosine of hash embeddings mapped to [0, 1]."""

    def __init__(self, dim: int = 512):
        self._e = HashEmbedder(dim)
        self.name = f"hashce{dim}"

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        a = self._e.encode([p[0] for p in pairs])
        b = self._e.encode([p[1] for p in pairs])
        return ((np.sum(a * b, axis=1) + 1.0) / 2.0).astype(np.float32)


class STCrossEncoder:
    """sentence-transformers CrossEncoder; raw logits passed through a sigmoid -> [0, 1]."""

    def __init__(self, model: str, max_length: int = 512, batch_size: int = 128, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model, max_length=max_length, device=device)
        self.name = str(model)
        self.batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        import torch

        # Identity => raw logits (None would apply the model's default activation, often sigmoid)
        logits = self._m.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=False,
                                 convert_to_numpy=True, activation_fn=torch.nn.Identity())
        logits = np.asarray(logits, dtype=np.float64).reshape(len(pairs), -1)[:, -1]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


class CachedCrossScorer:
    def __init__(self, inner: CrossScorer):
        self.inner = inner
        self.name = inner.name
        self._memo: dict[tuple[str, str], float] = {}

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        miss = list(dict.fromkeys(p for p in pairs if p not in self._memo))
        if miss:
            for p, s in zip(miss, self.inner.score(miss)):
                self._memo[p] = float(s)
        return np.array([self._memo[p] for p in pairs], dtype=np.float32)


def build_cross(xcfg: Any, model_path: str | None = None) -> CachedCrossScorer:
    if xcfg.type == "hash":
        return CachedCrossScorer(HashCrossEncoder(int(xcfg.get("dim", 512))))
    if xcfg.type == "st":
        return CachedCrossScorer(STCrossEncoder(model_path or xcfg.hf_id, int(xcfg.get("max_length", 512)),
                                                int(xcfg.get("batch_size", 128))))
    raise ValueError(xcfg.type)
