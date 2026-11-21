#!/usr/bin/env python
"""Expand an experiment config into Slurm job arrays chained with dependencies.

    python slurm/submit_experiment.py E2 [--models qwen7b] [--stages gen,judge] [--after JOBID]
                                         [--dry-run] [--no-wait] [--profile cluster]

Guarantees (CSIS limits: <=5 running jobs, <=3 GPUs per user):
- 1-GPU arrays are throttled %3, 2-GPU arrays %1.
- GPU stages are serialised: each GPU array also depends (afterany) on the previous GPU array, so
  at most one GPU array runs at a time -> never more than 3 GPUs from this submission.
- Data edges use afterok (a failed stage stops its dependants).
- Before each sbatch, waits while >= max_jobs of the user's jobs are running or eligible
  (pending jobs blocked only by a Dependency are not counted). --no-wait skips this.
Job IDs are printed and written to results/jobs/<exp>_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from scopemem.config import load_config, load_named  # noqa: E402

TEMPLATES = {"gpu1": "slurm/gen_1gpu.sbatch", "gpu2": "slurm/gen_2gpu.sbatch", "cpu": "slurm/cpu.sbatch",
             "judge1": "slurm/judge.sbatch", "judge2": "slurm/judge.sbatch", "debug": "slurm/cpu.sbatch"}


def _secs(t: str) -> int:
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return h * 3600 + m * 60 + s


def _clamp_time(t: str, max_t: str) -> str:
    s = min(_secs(t), _secs(max_t))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class Submitter:
    def __init__(self, cluster, dry_run: bool, wait: bool, cmd_dir: Path):
        self.c = cluster
        self.dry = dry_run
        self.wait = wait
        self.cmd_dir = cmd_dir
        self.fake_id = 1000
        self.lines: list[str] = []

    def inflight(self) -> int:
        user = os.environ.get("USER", "")
        out = subprocess.run(["squeue", "-u", user, "-h", "-t", "R,PD", "-o", "%T|%r"],
                             capture_output=True, text=True, check=True).stdout
        return sum(1 for l in out.splitlines() if l and "Dependency" not in l)

    def wait_slot(self) -> None:
        if self.dry or not self.wait:
            return
        while self.inflight() >= int(self.c.limits.max_jobs):
            print(f"  {self.inflight()} jobs in flight (limit {self.c.limits.max_jobs}); waiting 60s ...")
            time.sleep(60)

    def gres(self, n: int) -> list[str]:
        if n == 0:
            return []
        g = self.c.gres_typed if self.c.use_typed_gres else self.c.gres_fallback
        return [f"--gres={g}:{n}"]

    def submit(self, name: str, kind: str, n_tasks: int, cmd: str, time_s: str, deps_ok: list[str],
               deps_any: list[str]) -> str:
        prof = self.c.defaults["gpu2" if kind in ("gpu2", "judge2") else "gpu1" if kind in ("gpu1", "judge1")
                                else "debug" if kind == "debug" else "cpu"]
        part = str(prof.partition)
        limits = self.c.partitions[part]
        args = ["sbatch", "--parsable", f"--job-name=sm-{name}", f"--partition={part}",
                f"--cpus-per-task={prof.cpus}", f"--mem={prof.mem}",
                f"--time={_clamp_time(time_s, str(limits.max_time))}"]
        if self.c.qos_mode == "partition":
            args.append(f"--qos={part}")
        args += self.gres(int(prof.gpus))
        if n_tasks > 1 or prof.gpus:
            args.append(f"--array=0-{n_tasks - 1}%{min(int(prof.throttle), n_tasks)}")
        dep = [f"afterok:{':'.join(deps_ok)}"] if deps_ok else []
        dep += [f"afterany:{':'.join(deps_any)}"] if deps_any else []
        if dep:
            args.append("--dependency=" + ",".join(dep))
        cmd_file = self.cmd_dir / f"{name}.cmd"
        args.append(f"--export=ALL,SM_CMD_FILE={cmd_file}")
        args.append(TEMPLATES[kind])
        line = " ".join(args)
        self.lines.append(line)
        print(line)
        print(f"    cmd: {cmd}")
        if self.dry:
            self.fake_id += 1
            return str(self.fake_id)
        cmd_file.parent.mkdir(parents=True, exist_ok=True)
        cmd_file.write_text(cmd + "\n")
        self.wait_slot()
        out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
        if out.returncode != 0 and "gres" in out.stderr.lower() and self.c.use_typed_gres:
            print("  typed GRES rejected; retrying with fallback", out.stderr.strip())
            self.c.use_typed_gres = False
            return self.submit(name, kind, n_tasks, cmd, time_s, deps_ok, deps_any)
        if out.returncode != 0:
            raise SystemExit(f"sbatch failed: {out.stderr.strip()}")
        return out.stdout.strip().split(";")[0]


def n_tasks_for(stage, cfg, model: str) -> int:
    a = stage.get("array", 1)
    if a == "tasks":
        from scopemem.run import enumerate_tasks

        mcfg = load_config(cfg.experiment.name, None, [f"model={model}"])
        return len(enumerate_tasks(mcfg))
    if a == "folds":
        return int(cfg.data.n_folds)
    if a == "n_levels":
        return len(list(cfg.experiment.n_levels))
    return int(a)


def stage_kind(stage, model_cfg, judge_cfg) -> str:
    k = str(stage.kind)
    if k == "model":
        return {"gpu1": "gpu1", "gpu2": "gpu2", "cpu": "cpu"}[str(model_cfg.slurm)]
    if k == "judge":
        return "judge2" if str(judge_cfg.slurm) == "gpu2" else "judge1"
    return k


def plan(exp: str, models: list[str] | None, stages: list[str] | None, after: str | None, sub: Submitter,
         profile: str | None) -> dict:
    cfg = load_config(exp, profile)
    models = models or list(cfg.experiment.get("models", [cfg.model]))
    judge_cfg = load_named("models", str(cfg.judge))
    jobs: dict[str, str] = {}
    last_once: str | None = after
    last_gpu: str | None = after
    per_model_last: dict[str, str | None] = {m: None for m in models}
    gpu_kinds = {"gpu1", "gpu2", "judge1", "judge2"}
    st_list = [s for s in cfg.experiment.stages if not stages or s.name in stages]
    for st in st_list:
        per_model = "{model}" in str(st.cmd)
        for m in (models if per_model else [None]):
            mc = load_named("models", m) if m else load_named("models", models[0])
            kind = stage_kind(st, mc, judge_cfg)
            n = n_tasks_for(st, cfg, m or models[0])
            cmd = str(st.cmd).replace("{model}", m or models[0]).replace("{n_shards}", str(n))
            if profile:
                cmd += f" profile={profile}"
            deps_ok = [d for d in ([per_model_last.get(m)] if m else []) + [last_once] if d]
            if not per_model:        # once-stage depends on every model chain so far
                deps_ok += [d for d in per_model_last.values() if d]
            deps_ok = sorted(set(deps_ok))
            deps_any = [last_gpu] if (kind in gpu_kinds and last_gpu and last_gpu not in deps_ok) else []
            name = f"{exp}-{st.name}" + (f"-{m}" if m else "")
            jid = sub.submit(name, kind, n, cmd, str(st.get("time", "01:00:00")), deps_ok, deps_any)
            jobs[name] = jid
            if kind in gpu_kinds:
                last_gpu = jid
            if m:
                per_model_last[m] = jid
            else:
                last_once = jid
                per_model_last = {k: None for k in per_model_last}
    return jobs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--models", default=None)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--after", default=None, help="job id that the first stage must wait for")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args(argv)
    cluster = OmegaConf.load(ROOT / "configs" / "cluster" / "csis.yaml")
    ts = time.strftime("%Y%m%d-%H%M%S")
    sub = Submitter(cluster, a.dry_run, not a.no_wait, ROOT / "results" / "jobs" / "cmds" / f"{a.experiment}_{ts}")
    (ROOT / "logs").mkdir(exist_ok=True)
    jobs = plan(a.experiment, a.models.split(",") if a.models else None,
                a.stages.split(",") if a.stages else None, a.after, sub, a.profile)
    print(json.dumps(jobs, indent=1))
    if not a.dry_run:
        out = ROOT / "results" / "jobs" / f"{a.experiment}_{ts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"jobs": jobs, "sbatch": sub.lines}, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
