# Local targets. On Windows use the msys64 make (C:/msys64/usr/bin/make.exe) or run the
# commands directly. PY defaults to the project venv if it exists.
VENV_PY := $(firstword $(wildcard .venv/Scripts/python.exe .venv/bin/python))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)

.PHONY: test smoke data data-stats scopes-fake e1-local e1-aggregate paper clean-smoke

test:
	$(PY) -m pytest

# E0 on 10 instances, FakeLLM + hash embedder, CPU, no Slurm, no network.
smoke:
	$(PY) -m scopemem.run experiment=E0 profile=smoke

# Build PrefEval-Multi from data/raw/prefeval_src (all N levels, full + sub500 sets).
data:
	$(PY) -m scopemem.data.build_multi

data-stats:
	$(PY) -m scopemem.data.build_multi --stats-only

# E1 plumbing run on a 200-instance subset: MiniLM embedder + small CPU cross-encoder + FakeLLM scopes.
e1-local:
	$(PY) -m scopemem.retrieval experiment=E1 profile=local

e1-aggregate:
	$(PY) -m scopemem.aggregate experiment=E1

# Regenerate every table and figure from raw JSONL.
paper:
	$(PY) -m scopemem.aggregate experiment=all

clean-smoke:
	rm -rf results/raw/E0_smoke results/tables/E0_smoke* cache/fake*
