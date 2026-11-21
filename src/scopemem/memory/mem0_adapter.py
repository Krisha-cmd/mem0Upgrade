"""Track B: real Mem0 (mem0ai==2.2.1) wired to a local vLLM OpenAI-compatible server + local HF
embedder + local Qdrant. Ingestion/gating implemented in Phase 6.

Verified at mem0ai 2.2.1 (see docs/DECISIONS.md D-019):
- LLM provider "vllm" with config keys model, vllm_base_url, api_key, temperature, max_tokens
- embedder provider "huggingface" with model (and embedding_dims)
- vector_store provider "qdrant" with path (local mode), on_disk, collection_name, embedding_model_dims
- Memory.add(messages, user_id=..., infer=True, metadata=...), Memory.search(query, top_k=..,
  filters={"user_id": ...}, threshold=0.1 by default -> we pass threshold=0.0), Memory.get_all(filters=...)
- telemetry: posthog by default; disabled with MEM0_TELEMETRY=False (set in slurm/common.sh)
"""

from __future__ import annotations

import os
from typing import Any


def mem0_config(model_name: str, base_url: str, embedder_hf_id: str, embed_dims: int, qdrant_path: str,
                collection: str, max_tokens: int = 1024) -> dict[str, Any]:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    return {
        "llm": {"provider": "vllm", "config": {"model": model_name, "vllm_base_url": base_url,
                                               "api_key": "local", "temperature": 0.0,
                                               "max_tokens": max_tokens}},
        "embedder": {"provider": "huggingface", "config": {"model": embedder_hf_id,
                                                           "embedding_dims": embed_dims}},
        "vector_store": {"provider": "qdrant", "config": {"path": qdrant_path, "on_disk": True,
                                                          "collection_name": collection,
                                                          "embedding_model_dims": embed_dims}},
    }


def main(argv=None) -> None:
    raise SystemExit("Mem0 ingestion (E5) is implemented in Phase 6")


if __name__ == "__main__":
    main()
