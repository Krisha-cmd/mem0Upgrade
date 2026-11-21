"""LLM engines. All LLM calls in the project go through ``LLM.generate`` (usually via CachedLLM).

- ``VLLMEngine``: vLLM offline batched generation (``LLM.chat`` over a whole shard).
- ``VLLMServer``: OpenAI-compatible server, used only for Track B (Mem0 needs HTTP).
- ``FakeLLM`` (fake.py): deterministic, CPU-only, for tests / smoke.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

Messages = list[dict[str, str]]


@dataclass
class GenParams:
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 0
    stop: list[str] | None = None

    def key(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens,
                "seed": self.seed, "stop": self.stop}


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def generate(self, prompts: Sequence[Messages], params: GenParams) -> list[Completion]: ...


class VLLMEngine:
    """Offline vLLM engine. Import is deferred so CPU-only installs never touch vllm."""

    def __init__(self, model_cfg: Any):
        from vllm import LLM as _VLLM  # noqa: N811

        self.name = model_cfg.name
        kwargs: dict[str, Any] = dict(
            model=model_cfg.hf_id,
            tensor_parallel_size=int(model_cfg.tensor_parallel_size),
            gpu_memory_utilization=float(model_cfg.gpu_memory_utilization),
            max_model_len=int(model_cfg.max_model_len),
            seed=0,
            trust_remote_code=False,
        )
        if model_cfg.get("quantization"):
            kwargs["quantization"] = model_cfg.quantization
        self._llm = _VLLM(**kwargs)
        self.max_model_len = int(model_cfg.max_model_len)

    def generate(self, prompts: Sequence[Messages], params: GenParams) -> list[Completion]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=params.temperature, max_tokens=params.max_tokens,
                            seed=params.seed, stop=params.stop)
        outs = self._llm.chat(list(prompts), sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(Completion(text=c.text, prompt_tokens=len(o.prompt_token_ids),
                                  completion_tokens=len(c.token_ids),
                                  meta={"finish_reason": c.finish_reason}))
        return res


class VLLMServer:
    """Start ``vllm serve`` inside a job on a port derived from SLURM_JOB_ID; kill on exit."""

    def __init__(self, model_cfg: Any, port: int | None = None, log_path: str | None = None):
        self.model_cfg = model_cfg
        job = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
        self.port = port or (20000 + job % 20000)
        self.log_path = log_path or os.devnull
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "VLLMServer":
        m = self.model_cfg
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", m.hf_id, "--served-model-name", m.name, "--host", "127.0.0.1",
               "--port", str(self.port), "--tensor-parallel-size", str(m.tensor_parallel_size),
               "--gpu-memory-utilization", str(m.gpu_memory_utilization),
               "--max-model-len", str(m.max_model_len)]
        if m.get("quantization"):
            cmd += ["--quantization", m.quantization]
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self._wait_health()
        return self

    def _wait_health(self, timeout_s: int = 1200) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError("vLLM server did not become healthy")

    def __exit__(self, *exc: Any) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


def build_llm(cfg: Any, model_cfg: Any | None = None, cached: bool = True) -> LLM:
    """Factory honouring ``cfg.llm_backend``; wraps in the SQLite cache by default."""
    from ..config import resolve_path
    from .cache import CachedLLM, SQLiteCache
    from .fake import FakeLLM

    mc = model_cfg if model_cfg is not None else cfg.model_cfg
    if cfg.llm_backend == "fake":
        inner: LLM = FakeLLM(name=str(mc.name) if str(mc.name).startswith("fake") else f"fake-{mc.name}")
    elif cfg.llm_backend == "vllm":
        inner = VLLMEngine(mc)
    else:
        raise ValueError(f"unknown llm_backend {cfg.llm_backend}")
    if not cached:
        return inner
    cache = SQLiteCache(resolve_path(cfg.paths.cache) / f"{inner.name}.sqlite")
    return CachedLLM(inner, cache)
