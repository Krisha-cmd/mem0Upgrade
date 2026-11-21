#!/bin/bash
# One-time setup on the CSIS LOGIN node (editing, env, downloads, sbatch only — no compute here).
#   cd ~/scopemem && bash scripts/setup_env_cluster.sh [--skip-env] [--skip-download] [--groups "core"]
# Prerequisite: ~/.hf_token exists with chmod 600 (never paste the token anywhere else).
set -eo pipefail
cd "$(dirname "$0")/.."
SKIP_ENV=0; SKIP_DL=0; DL_GROUPS="core"
while [[ $# -gt 0 ]]; do case "$1" in
  --skip-env) SKIP_ENV=1;; --skip-download) SKIP_DL=1;; --groups) DL_GROUPS="$2"; shift;; esac; shift; done

type module &>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniconda
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
export HF_HOME="$HOME/hf_cache" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
mkdir -p logs results/env "$HF_HOME"

echo "== 0. cluster probe (GRES / QoS / network) -> results/env/cluster_probe.txt"
{
  echo "# $(date -Is)"; sinfo -o "%20N %40G" || true
  scontrol show node csis.cn1 | grep -i gres || true
  sacctmgr show qos format=name,maxwall,maxtresperuser -p || true
  quota -u "$USER" || true
  curl -s -o /dev/null -w "login node internet (huggingface.co): HTTP %{http_code}\n" https://huggingface.co || echo "login node: no internet"
} | tee results/env/cluster_probe.txt
echo ">> Update configs/cluster/csis.yaml (gres_typed, qos_mode) from the probe above if needed."

if [[ $SKIP_ENV -eq 0 ]]; then
  echo "== 1. conda env"
  if conda env list | grep -q "^scopemem "; then conda env update -n scopemem -f environment.yml
  else conda env create -f environment.yml; fi
  conda activate scopemem
  python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
  pip freeze > results/env/pip_freeze.txt
  conda env export -n scopemem > results/env/conda_env.yml
else
  conda activate scopemem
fi

if [[ $SKIP_DL -eq 0 ]]; then
  echo "== 2. downloads (models -> \$HF_HOME, datasets -> data/raw)"
  quota -u "$USER" || true
  python scripts/download_models.py --groups $DL_GROUPS
  python scripts/download_data.py
  du -sh "$HF_HOME" data/raw || true
  quota -u "$USER" || true
fi

echo "== 3. checks (must both pass before anything else)"
J1=$(sbatch --parsable slurm/check_debug.sbatch)
J2=$(sbatch --parsable slurm/check_gpu.sbatch)
echo "submitted debug check $J1 and GPU check $J2; watch with: squeue -u \$USER"
echo "then: grep -l CHECK_DEBUG_OK logs/sm-check-debug_$J1.out && grep -l CHECK_GPU_OK logs/sm-check-gpu_$J2.out"
