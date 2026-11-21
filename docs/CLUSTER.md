# CSIS cluster (SOP v2)

These facts come from the SOP, and the code is designed around them. Machine-readable copy: `configs/cluster/csis.yaml`.

## Hardware and access
- **Compute nodes:** `csis.cn[1-4]` plus `csis.mn1`. Each has 2× AMD EPYC (128 cores), 256 GB RAM and **2× NVIDIA A100 80 GB (PCIe)**. They run Rocky Linux 8.10, NVIDIA driver 570.158.01 (CUDA ≤ 12.8) and Slurm 23.11.11.
- **Access:** `ssh <username>@172.24.16.132`. You land on the login (master) node. The login node is **only** for editing, environment setup, downloads and `sbatch`. Everything else goes through Slurm.

## Storage
| Path | Quota | Use |
|---|---|---|
| `$HOME` = `/nfs_home/users/<user>` | **300 GB** (`quota -u $USER`) | Code (`~/scopemem`), the conda env, the HF cache (`~/hf_cache`), final results |
| `~/scratch` → `/scratch/users/<user>` | Shared 1 TB; **files older than 30 days are deleted automatically** | Per-job vLLM / inductor / triton caches only (`$SCRATCH_RUN`, removed at job exit) |

All job outputs (`results/`, `cache/`, `models_ft/`) are written straight to `$HOME/scopemem`, so nothing needs copying back from scratch.

## Software
- `module load miniconda` and `module load cuda/12.8`. **Do not use the system Python 3.9 or 3.13.5.**
- The `scopemem` conda env is Python 3.11, created from `environment.yml`. Jobs activate it with:
  ```bash
  module load miniconda
  source /nfs_home/software/miniconda/etc/profile.d/conda.sh
  conda activate scopemem
  ```

## Partitions (per-job limits)
| Partition | Max time | CPUs | RAM | GPUs | Used for |
|---|---|---|---|---|---|
| `debug` | 30 min | 16 | 48 GB | 0 | Import check and pytest (`slurm/check_debug.sbatch`) |
| `cpu-short` | 12 h | 64 | 96 GB | 0 | Data building, aggregation, stats (`slurm/cpu.sbatch`) |
| `gpu-short` | 8 h | 24 | 96 GB | 2 | **Default:** 7B/8B generation, embeddings, 72B-AWQ judge on 1 GPU (`gen_1gpu`, `judge`) |
| `gpu-long` | 12 h | 16 | 80 GB | 2 | 32B backbone and 70B judge with TP=2 (`gen_2gpu`) |
| `gpu-1day` | 24 h | 16 | 64 GB | 2 | Only if a shard cannot be split further |
| `gpu-3day` | 72 h | 16 | 64 GB | 1 | Avoid |

## Per-user limits and how the code respects them
**At most 5 concurrent jobs and 3 GPUs in total.** A sixth job waits with reason `JobHeldUser`. `slurm/submit_experiment.py` handles this as follows:
- 1-GPU arrays are throttled with `%3` and 2-GPU arrays with `%1`.
- GPU arrays are **serialised**: each GPU array depends (`afterany`) on the previous one, so one submission never uses more than 3 GPUs.
- Data dependencies use `afterok`.
- Before each `sbatch`, it waits while 5 or more jobs are running or eligible. Pending jobs that are only waiting on a dependency are not counted. `--no-wait` skips this check.
- Default requests: 1 GPU → `--cpus-per-task=8 --mem=64G`; 2 GPUs → `--cpus-per-task=16 --mem=80G`; CPU → `8 / 32G`.
- `--time` comes from each experiment's stage config, clamped to the partition maximum. After E0, set it to about 1.5× the measured runtime (`docs/BUDGET.md`).
- Every GPU template has `--signal=B:USR1@300`. Five minutes before the time limit, the job drops a STOP file; `run.py` checks for it between batches and exits cleanly. Resubmitting the same array finishes the remaining work, because shards skip instances that are already done.

## Things to verify on the login node (recorded by `scripts/setup_env_cluster.sh` in `results/env/cluster_probe.txt`)
- **GRES string:** run `sinfo -o "%20N %40G"` and `scontrol show node csis.cn1 | grep -i gres`. The SOP's templates use `gpu:a100-80gb:N`, but the SOP's `sinfo` output is truncated (`gpu:a100-8…`). Set `gres_typed` in `csis.yaml`. If sbatch rejects the typed form, the submitter automatically retries with `gpu:N`.
- **QoS:** run `sacctmgr show qos format=name,maxwall,maxtresperuser -p`. One SOP template uses `--qos=short`, while the QoS table names match the partitions. Set `qos_mode: partition` (passes `--qos=<partition>`) or `qos_mode: omit`.
- **Network:** compute nodes are assumed to be offline, and jobs set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. If the login node also has no internet, use `scripts/stage_from_laptop.sh`.

## Monitoring
```bash
squeue -u $USER                                                                   # status
sacct -j <id> -o JobID,JobName,Partition,AllocTRES,Elapsed,MaxRSS,State,ExitCode  # history
seff <id>                                                                         # efficiency: right-size after E0
sprio -u $USER                                                                    # priority
```
Logs are in `logs/<job-name>_<arrayjob>_<task>.{out,err}`. Submissions are recorded in `results/jobs/<exp>_<timestamp>.json`.
