from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("scopemem")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


class JsonlAppender:
    """Append one JSON line per record and flush+fsync, so a killed job loses at most one line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")

    def _repair_tail(self) -> None:
        """Drop a partial last line left by a kill mid-write, so appends start on a fresh line."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, "rb+") as f:
            data = f.read()
            if data.endswith(b"\n"):
                return
            cut = data.rfind(b"\n") + 1      # 0 if there is no complete line at all
            f.truncate(cut)

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def done_ids(path: str | Path, key: str = "instance_id") -> set[str]:
    """IDs already present in a shard file. A truncated last line (kill mid-write) is ignored."""
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        from .config import repo_root

        return subprocess.check_output(
            ["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def shard_slice(items: list, shard: int, num_shards: int) -> list:
    """Contiguous-stride sharding: item i goes to shard i % num_shards (stable under resume)."""
    if num_shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_shards == shard]
