#!/bin/bash
#SBATCH --job-name=sm-judge
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1        # submit_experiment.py passes the verified GRES string
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00               # overridden from experiment config (E0 measurement x1.5)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@300            # 5 min warning before the time limit
# Usage: SM_CMD_FILE=<file with command> sbatch --array=0-N%T judge.sbatch
source slurm/common.sh
run_stage
