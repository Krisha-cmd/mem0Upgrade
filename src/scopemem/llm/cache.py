"""SQLite cache for LLM calls, keyed by hash(model, messages, params). One DB per model, WAL mode.

A re-run of a judge or an aggregate step never repeats a generation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from .engine import Completion, GenParams, LLM, Messages


def cache_key(model: str, messages: Messages, params: GenParams) -> str:
    blob = json.dumps([model, messages, params.key()], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=60, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = list(keys[i:i + 500])
                q = f"SELECT k, v FROM cache WHERE k IN ({','.join('?' * len(chunk))})"
                for k, v in self._conn.execute(q, chunk):
                    out[k] = json.loads(v)
        return out

    def put_many(self, items: dict[str, dict]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
                                   [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()])
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class CachedLLM:
    def __init__(self, inner: LLM, cache: SQLiteCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def generate(self, prompts: Sequence[Messages], params: GenParams) -> list[Completion]:
        keys = [cache_key(self.name, p, params) for p in prompts]
        found = self.cache.get_many(keys)
        miss_idx = [i for i, k in enumerate(keys) if k not in found]
        # de-duplicate identical prompts within the batch
        uniq: dict[str, int] = {}
        for i in miss_idx:
            uniq.setdefault(keys[i], i)
        new: dict[str, dict] = {}
        if uniq:
            outs = self.inner.generate([prompts[i] for i in uniq.values()], params)
            for k, c in zip(uniq.keys(), outs):
                new[k] = {"text": c.text, "prompt_tokens": c.prompt_tokens,
                          "completion_tokens": c.completion_tokens, "meta": c.meta}
            self.cache.put_many(new)
        self.hits += len(keys) - len(miss_idx)
        self.misses += len(uniq)
        res = []
        for k in keys:
            if k in found:
                v = found[k]
                res.append(Completion(v["text"], v["prompt_tokens"], v["completion_tokens"], True, v.get("meta", {})))
            else:
                v = new[k]
                res.append(Completion(v["text"], v["prompt_tokens"], v["completion_tokens"], False, v.get("meta", {})))
        return res
