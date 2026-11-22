#!/usr/bin/env python
"""Run an experiment's Slurm stage graph locally, in order, with the EXACT stage commands from
configs/experiments/<E>.yaml (array tasks run one after another). CPU, FakeLLM, no Slurm.

    python scripts/run_stages_local.py E1                  # profile=smoke (tiny fixture, seconds)
    python scripts/run_stages_local.py E1 --profile local  # real PrefEval, small CPU models (minutes)
    python scripts/run_stages_local.py E1 --stages build,scopes

This exercises the command strings, CLI parsing, profile propagation (via SCOPEMEM_PROFILE, as the
cluster does), stage-to-stage file hand-offs and array indexing. It does not exercise GPUs/vLLM:
that is what `submit_experiment.py <E> --profile demo` on the cluster is for.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "slurm"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--stages", default=None)
    a = ap.parse_args()
    os.environ["SCOPEMEM_PROFILE"] = a.profile
    os.environ["SCOPEMEM_ROOT"] = str(ROOT)

    from scopemem.config import load_config
    from submit_experiment import n_tasks_for  # type: ignore[import-not-found]

    cfg = load_config(a.experiment, a.profile)
    models = list(cfg.experiment.get("models", [cfg.model]))
    stages = [s for s in cfg.experiment.stages if not a.stages or s.name in a.stages.split(",")]
    env = {**os.environ, "PYTHONUTF8": "1"}
    t_all = time.time()
    for st in stages:
        per_model = "{model}" in str(st.cmd)
        for m in (models if per_model else [models[0]]):
            n = n_tasks_for(st, cfg, m)
            for task in range(n):
                cmd = str(st.cmd).replace("{model}", m).replace("{n_shards}", str(n)).replace("$TASK", str(task))
                cmd = cmd.replace("python -m", f'"{sys.executable}" -m')
                label = f"{st.name}" + (f"[{m}]" if per_model else "") + (f" task {task}/{n}" if n > 1 else "")
                print(f"\n>>> {label}: {cmd}", flush=True)
                t0 = time.time()
                r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
                if r.returncode != 0:
                    print(f"\nFAILED at stage {label} (exit {r.returncode}) after {time.time() - t0:.1f}s")
                    sys.exit(r.returncode)
                print(f"<<< {label} ok ({time.time() - t0:.1f}s)")
    print(f"\nall {len(stages)} stages ok in {time.time() - t_all:.1f}s (profile={a.profile})")


if __name__ == "__main__":
    main()
