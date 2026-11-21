#!/bin/bash
# Use only if the CSIS login node has no internet. Downloads on a laptop, then copies to the cluster.
#   CLUSTER_USER=<you> bash scripts/stage_from_laptop.sh [groups...]
# Needs: python with huggingface_hub, rsync, ssh access to 172.24.16.132 (campus network / VPN).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLUSTER_USER:?set CLUSTER_USER to your cluster username}"
HOST="${CLUSTER_USER}@172.24.16.132"
STAGE="${STAGE_DIR:-$PWD/.stage}"
DL_GROUPS="${*:-core}"
export HF_HOME="$STAGE/hf_cache"
mkdir -p "$HF_HOME"
python scripts/download_models.py --groups $DL_GROUPS
python scripts/download_data.py
rsync -av --partial --progress "$HF_HOME/" "$HOST:hf_cache/"
rsync -av --partial --progress data/raw/ "$HOST:scopemem/data/raw/"
echo "staged. On the cluster: bash scripts/setup_env_cluster.sh --skip-download"
