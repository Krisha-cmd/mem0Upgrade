#!/bin/bash
#SBATCH --job-name=sm-check-debug
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Check 1: everything imports, and the CPU test suite passes on the cluster.
source slurm/common.sh
python - <<'PY'
import importlib
for m in ["scopemem", "vllm", "torch", "transformers", "sentence_transformers", "mem0", "pydantic",
          "omegaconf", "pandas", "scipy", "statsmodels", "matplotlib"]:
    mod = importlib.import_module(m); print(f"{m:22s} {getattr(mod, '__version__', '?')}")
import torch; print("torch cuda build:", torch.version.cuda)
PY
python -m pytest -q
echo CHECK_DEBUG_OK
