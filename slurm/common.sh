#!/bin/bash
# Sourced by every ScopeMem job. Never prints secrets.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/scopemem}"
type module &>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load miniconda
module load cuda/12.8
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate scopemem
set -u
export SCOPEMEM_ROOT="$PWD"
export HF_HOME="$HOME/hf_cache"                  # NOT scratch (30-day deletion)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # compute nodes: assume no internet
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 MEM0_TELEMETRY=False
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(cat "$HOME/.hf_token")"; fi
export SCRATCH_RUN="$HOME/scratch/scopemem/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SCRATCH_RUN"
export VLLM_CACHE_ROOT="$SCRATCH_RUN/vllm" TORCHINDUCTOR_CACHE_DIR="$SCRATCH_RUN/inductor" TRITON_CACHE_DIR="$SCRATCH_RUN/triton"
export TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "job=${SLURM_JOB_ID:-none} task=$TASK node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} git=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
cleanup_scratch() { rm -rf "$SCRATCH_RUN"; }   # all outputs are written under $HOME/scopemem
trap cleanup_scratch EXIT

# run_stage: execute the command stored in $SM_CMD_FILE (with $TASK substituted) in the
# background, so a USR1 (sent 5 min before the time limit) can drop a STOP file that run.py
# checks between batches; then wait for the child to exit cleanly.
run_stage() {
  local cmd
  cmd="$(cat "$SM_CMD_FILE")"
  cmd="${cmd//\$TASK/$TASK}"
  echo "cmd: $cmd"
  trap 'echo "caught USR1, requesting clean stop"; touch "$SCRATCH_RUN/STOP"' USR1
  bash -c "$cmd" &
  local pid=$!
  wait "$pid" || true          # returns early when a signal arrives
  wait "$pid"                  # then wait for the real exit status
}
