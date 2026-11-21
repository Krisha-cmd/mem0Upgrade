# ScopeMem

*Not Every Preference, Every Time: Applicability-Aware Preference Memory for Token-Efficient Personalisation.*

When a preference is stored, ScopeMem asks an LLM to write a short **scope** for it: the kinds of requests it applies to, three example requests, and an `always_on` flag. At query time, a cheap gate matches the request against these scopes instead of the preference text. Only the applicable preferences go into the prompt.

The comparison set is no memory, full context, inject-all, Mem0-style similarity top-k, similarity with reranking, an LLM selector, and an oracle. Everything runs on local open-weight models through vLLM. No paid APIs are used.

Status: phases 1–3 are built, with the E1 local plumbing run done. The E1 cluster run is pending. Phase 4 and later are not started. See `docs/`.

## Quickstart (laptop, CPU, no GPU)
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # Windows (.venv/bin/... elsewhere)
make test          # pytest, FakeLLM, CPU only, no network (about 5 s)
make smoke         # E0 on 10 instances: FakeLLM + hash embedder, tiny synthetic fixture (about 2 s)

# Real PrefEval data and the E1 plumbing run (small CPU models + FakeLLM scopes)
python scripts/download_data.py
pip install -e ".[cpu-models]"
make data          # build PrefEval-Multi (all N, full + sub500) and print summary stats
make e1-local      # E1 on a 200-instance subset per N: MiniLM embedder, MiniLM CE, FakeLLM scopes
python -m scopemem.aggregate experiment=E1 profile=local
```
On Windows, `make` is available as `C:/msys64/usr/bin/make.exe`.

## Quickstart (CSIS cluster)
```bash
ssh <user>@172.24.16.132
git clone <this repo> ~/scopemem && cd ~/scopemem
# put your HF token in ~/.hf_token (chmod 600); accept the Llama 3.1 licence on HF first
bash scripts/setup_env_cluster.sh                  # probe GRES/QoS, conda env, downloads, 2 check jobs
python slurm/submit_experiment.py E1 --dry-run     # inspect the sbatch lines
python slurm/submit_experiment.py E1               # build -> scopes -> S3 per fold -> retrieval -> aggregate
squeue -u $USER
```
Details (limits, storage, monitoring) are in [docs/CLUSTER.md](docs/CLUSTER.md).

## Layout
- `configs/`: `base.yaml`, `profiles/{smoke,local,cluster}`, `models/`, `embedders/`, `xencs/`, `conditions/` (one file per condition), `experiments/E0..E8` (including each experiment's Slurm stage graph), and `cluster/csis.yaml`.
- `src/scopemem/`:
  - `data/`: PrefEval, PersonaMem and LoCoMo loaders, the PrefEval-Multi builder, and the global preferences.
  - `llm/`: the vLLM engine and server, FakeLLM, the SQLite cache, and token counting.
  - `memory/`: the store, the scope writer, and the Mem0 adapter.
  - `gates/`: every condition and the fold-wise tuner.
  - `eval/`: metrics, stats, and the judge (Phase 4).
  - Top-level modules: `retrieval.py` (E1), `run.py` (generation runner), `aggregate.py`, `plots.py`.
- `slurm/`: `common.sh`, sbatch templates, `submit_experiment.py`, and the two check jobs.
- `docs/`: [DECISIONS](docs/DECISIONS.md), [DATASETS](docs/DATASETS.md), [CLUSTER](docs/CLUSTER.md), [PROTOCOL](docs/PROTOCOL.md), [RESULTS](docs/RESULTS.md).

## Running one thing
```bash
python -m scopemem.retrieval experiment=E1 model=qwen7b run.shard=2 run.num_shards=6
python -m scopemem.run experiment=E2 model=qwen7b task=17       # task = (condition, shard) index
python -m scopemem.aggregate experiment=E1
```
Any config key can be overridden with `key=value`, for example `gate.k_max=5` or `embedder=bge_m3`. Records carry `config_hash` and `git_sha`. The aggregator refuses to mix config hashes within a cell unless `--allow-mixed` is passed.
