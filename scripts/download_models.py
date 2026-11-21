#!/usr/bin/env python
"""Download model weights into $HF_HOME (run on the LOGIN node, or a laptop for staging).

    python scripts/download_models.py --groups core          # what E0-E4/E6 need
    python scripts/download_models.py --groups llama8b qwen32b judge2

Only safetensors + tokenizer/config files are fetched (no .bin / .pth / onnx duplicates), which
keeps the total near the section-5 budget. The HF token is read from ~/.hf_token and never printed.
Gated Llama models fall back to the logged substitute if access is refused (section 13).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

GROUPS = {
    "core": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct-AWQ", "BAAI/bge-base-en-v1.5",
             "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-reranker-v2-m3"],
    "llama8b": ["meta-llama/Llama-3.1-8B-Instruct"],
    "qwen32b": ["Qwen/Qwen2.5-32B-Instruct"],
    "judge2": ["casperhansen/llama-3.3-70b-instruct-awq"],
}
FALLBACK = {"meta-llama/Llama-3.1-8B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3"}
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.py", "1_Pooling/*",
         "sentencepiece*"]
IGNORE = ["original/*", "*.bin", "*.pth", "*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"]


def token() -> str | None:
    p = Path.home() / ".hf_token"
    return p.read_text().strip() if p.exists() else os.environ.get("HF_TOKEN")


def quota_report() -> None:
    home = os.environ.get("HF_HOME", str(Path.home() / "hf_cache"))
    try:
        print(subprocess.run(["du", "-sh", home], capture_output=True, text=True).stdout.strip())
        q = shutil.which("quota")
        if q:
            print(subprocess.run([q, "-u", os.environ.get("USER", "")], capture_output=True, text=True).stdout)
    except Exception as e:  # noqa: BLE001
        print("quota check failed:", e)


def main() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["core"], choices=list(GROUPS))
    a = ap.parse_args()
    tok = token()
    swaps = {}
    for g in a.groups:
        for repo in GROUPS[g]:
            try:
                print(f"== {repo}")
                snapshot_download(repo, allow_patterns=ALLOW, ignore_patterns=IGNORE, token=tok)
            except (GatedRepoError, HfHubHTTPError) as e:
                fb = FALLBACK.get(repo)
                if not fb:
                    raise
                print(f"   cannot download {repo} ({type(e).__name__}); using fallback {fb} — LOG THIS SWAP")
                snapshot_download(fb, allow_patterns=ALLOW, ignore_patterns=IGNORE, token=tok)
                swaps[repo] = fb
            quota_report()
    if swaps:
        print("MODEL SWAPS (record in docs/DECISIONS.md and RESULTS.md):", swaps)


if __name__ == "__main__":
    main()
