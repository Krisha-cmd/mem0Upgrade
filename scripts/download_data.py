#!/usr/bin/env python
"""Fetch the raw datasets into data/raw/ at pinned revisions (login node or laptop).

  PrefEval   github.com/amazon-science/PrefEval @ 50795054b5ff5f418d2b768a331d71e480f93331 (CC BY-NC 4.0)
  PersonaMem HF dataset bowen-upenn/PersonaMem-v1 @ a8076d5608c93ba2a28983cd78aa99b01a163ae7 (MIT)
  LoCoMo     github.com/snap-research/locomo data/locomo10.json (main; sha256 recorded after download)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PREFEVAL_SHA = "50795054b5ff5f418d2b768a331d71e480f93331"
PERSONAMEM_REV = "a8076d5608c93ba2a28983cd78aa99b01a163ae7"
PERSONAMEM_FILES = ["questions_32k.csv", "shared_contexts_32k.jsonl"]
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    pe = RAW / "prefeval_src"
    if not pe.exists():
        subprocess.run(["git", "clone", "https://github.com/amazon-science/PrefEval.git", str(pe)], check=True)
    subprocess.run(["git", "-C", str(pe), "checkout", "-q", PREFEVAL_SHA], check=True)

    from huggingface_hub import hf_hub_download

    pm = RAW / "personamem"
    pm.mkdir(exist_ok=True)
    for f in PERSONAMEM_FILES:
        hf_hub_download("bowen-upenn/PersonaMem-v1", f, repo_type="dataset", revision=PERSONAMEM_REV,
                        local_dir=str(pm))
    lc = RAW / "locomo"
    lc.mkdir(exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, lc / "locomo10.json")
    manifest = {"prefeval_commit": PREFEVAL_SHA, "personamem_revision": PERSONAMEM_REV,
                "locomo_sha256": sha256(lc / "locomo10.json"),
                "personamem_sha256": {f: sha256(pm / f) for f in PERSONAMEM_FILES}}
    (RAW / "raw_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
