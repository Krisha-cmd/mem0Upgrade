#!/bin/bash
#SBATCH --job-name=sm-cpu
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@300
# Data building, aggregation, stats. No GPU.
source slurm/common.sh
run_stage
