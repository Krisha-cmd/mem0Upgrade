#!/usr/bin/env python
"""Login-node preflight: find every error you can find WITHOUT a GPU, before submitting.

    python scripts/preflight.py E1                 # checks for one experiment
    python scripts/preflight.py E1 E2 E3           # several
    python scripts/preflight.py all                # E0-E8
    python scripts/preflight.py E1 --tests         # also run pytest (CPU, ~5 s)
    python scripts/preflight.py E1 --check-demo    # after a `--profile demo` run: did every stage write output?

Checks: python env + packages, HF cache settings, raw data, every model the experiment needs is fully
in the local HF cache and its tokenizer loads OFFLINE (exactly as jobs do), condition/config files,
the Slurm stage plan (dry run), Slurm tools. Exit code 1 if anything FAILs.

What this cannot catch (needs a GPU) is covered by the demo run:
    python slurm/submit_experiment.py E1 --profile demo --max-time 01:00:00
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SCOPEMEM_ROOT", str(ROOT))
# huggingface_hub reads these at import time, so set them before anything imports it: check the
# cache the jobs read (slurm/common.sh), offline, exactly as a job would.
HF_HOME_IN_SHELL = os.environ.get("HF_HOME")
os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  [WARN] {msg}")


def section(t: str) -> None:
    print(f"\n== {t}")


# ---------------------------------------------------------------------------------------------
def check_env() -> None:
    section("Python environment")
    v = sys.version_info
    (ok if (v.major, v.minor) == (3, 11) else warn)(f"python {v.major}.{v.minor} at {sys.executable}")
    if "envs/scopemem" not in sys.executable.replace("\\", "/") and ".venv" not in sys.executable:
        warn("not running from the scopemem conda env: run `conda activate scopemem` first")
    for mod in ["omegaconf", "pydantic", "pandas", "numpy", "huggingface_hub", "transformers",
                "sentence_transformers", "torch", "vllm"]:
        try:
            m = importlib.import_module(mod)
            ok(f"import {mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            (warn if mod == "vllm" and sys.platform == "win32" else fail)(f"import {mod}: {type(e).__name__}: {e}")
    try:
        importlib.import_module("scopemem")
        ok("import scopemem")
    except Exception as e:  # noqa: BLE001
        fail(f"import scopemem: {e} (run `pip install -e .` in the env)")


def check_hf_env() -> None:
    section("Hugging Face cache settings (jobs use HF_HOME=$HOME/hf_cache, offline)")
    job_home = str(Path.home() / "hf_cache")
    cur = HF_HOME_IN_SHELL
    if cur and Path(cur).resolve() != Path(job_home).resolve():
        fail(f"HF_HOME={cur} in this shell but jobs read {job_home} (slurm/common.sh). Checking {cur}; "
             f"downloads made with this setting are invisible to jobs.")
    elif not cur:
        warn(f"HF_HOME unset in this shell; checking {job_home} (what jobs use). "
             f"Export it before running download_models.py.")
    else:
        ok(f"HF_HOME={cur}")
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") and importlib.util.find_spec("hf_transfer") is None:
        fail("HF_HUB_ENABLE_HF_TRANSFER is set but hf_transfer is not installed in THIS python: "
             "`unset HF_HUB_ENABLE_HF_TRANSFER` (and remove it from ~/.bashrc)")
    if (Path.home() / ".cache" / "huggingface" / "hub").exists():
        stray = [p.name for p in (Path.home() / ".cache" / "huggingface" / "hub").glob("models--*")]
        if stray:
            warn(f"models also in ~/.cache/huggingface (jobs do NOT read these): {stray[:6]}")


def check_data(cfg, exps) -> None:
    from scopemem.config import resolve_path

    section("Raw data")
    pe = resolve_path(cfg.paths.prefeval_root)
    b = pe / "benchmark_dataset"
    topics = list((b / "explicit_preference").glob("*.json"))
    if len(topics) == 20 and (b / "filtered_inter_turns.json").exists():
        ok(f"PrefEval: 20 topics + filler at {b}")
    else:
        fail(f"PrefEval incomplete at {pe} ({len(topics)} topic files). "
             f"rm -rf {pe} && python scripts/download_data.py")
    kinds = {str(e.kind) for e in exps}
    if "generalisation" in kinds:
        for p in [resolve_path(cfg.paths.personamem_root) / "questions_32k.csv", resolve_path(cfg.paths.locomo_path)]:
            (ok if p.exists() else fail)(f"{p}" + ("" if p.exists() else " missing: python scripts/download_data.py"))


def hf_snapshot(hf_id: str) -> Path | None:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(hf_id, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def check_model(hf_id: str, role: str, llm: bool) -> None:
    p = hf_snapshot(hf_id)
    if p is None:
        fail(f"{role}: {hf_id} not in HF cache -> python scripts/download_models.py (see GROUPS)")
        return
    files = {f.name for f in p.iterdir()}
    broken = [f.name for f in p.iterdir() if f.is_symlink() and not f.exists()]
    missing = [f for f in ["config.json"] if f not in files]
    if not any(f.endswith(".safetensors") for f in files):
        missing.append("*.safetensors")
    if not files & {"tokenizer.json", "vocab.json", "tokenizer.model", "vocab.txt", "sentencepiece.bpe.model"}:
        missing.append("tokenizer files")
    if broken or missing:
        fail(f"{role}: {hf_id} incomplete at {p}: missing={missing} broken={broken}. "
             f"rm -rf {p.parents[1]} and re-download")
        return
    if llm:
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            shards = set(json.loads(idx.read_text())["weight_map"].values())
            lost = sorted(shards - files)
            if lost:
                fail(f"{role}: {hf_id} missing weight shards {lost}")
                return
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf_id)
            tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
        except Exception as e:  # noqa: BLE001
            fail(f"{role}: {hf_id} tokenizer does not load offline: {type(e).__name__}: {e}")
            return
    ok(f"{role}: {hf_id}")


def needed_models(cfg, exp) -> list[tuple[str, str, bool]]:
    """(hf_id, role, is_llm) for everything this experiment's stages load."""
    from omegaconf import OmegaConf

    from scopemem.config import load_named

    out: list[tuple[str, str, bool]] = []
    kinds = {str(s.kind) for s in exp.get("stages", [])}
    for m in exp.get("models", [cfg.model]):
        mc = load_named("models", str(m))
        out.append((str(mc.hf_id), f"backbone {m}", True))
        if mc.get("fallback_hf_id") and hf_snapshot(str(mc.hf_id)) is None and hf_snapshot(str(mc.fallback_hf_id)):
            out[-1] = (str(mc.fallback_hf_id), f"backbone {m} (fallback)", True)
    for w in (exp.get("extra_sweeps") or {}).get("scope_writer", []) or []:
        out.append((str(load_named("models", str(w)).hf_id), f"scope writer {w}", True))
    if cfg.get("scope_writer"):
        out.append((str(load_named("models", str(cfg.scope_writer)).hf_id), "scope writer", True))
    if "judge" in kinds:
        out.append((str(load_named("models", str(cfg.judge)).hf_id), f"judge {cfg.judge}", True))
    for s in exp.get("stages", []):
        if str(s.kind) == "model_from":
            name = str(OmegaConf.select(cfg, str(s.model_from)))
            out.append((str(load_named("models", name).hf_id), f"{s.name} ({name})", True))
    embs = {str(cfg.embedder)} | set(map(str, (exp.get("extra_sweeps") or {}).get("embedder", []) or []))
    for e in sorted(embs):
        ec = load_named("embedders", e)
        if ec.get("hf_id"):
            out.append((str(ec.hf_id), f"embedder {e}", False))
    if cfg.xenc_cfg.get("hf_id"):
        out.append((str(cfg.xenc_cfg.hf_id), f"cross-encoder {cfg.xenc}", False))
    seen, uniq = set(), []
    for t in out:
        if t[0] not in seen:
            seen.add(t[0])
            uniq.append(t)
    return uniq


def check_configs_and_plan(name: str, profile: str | None) -> None:
    from scopemem.config import configs_dir, load_config

    section(f"{name}: configs, conditions, Slurm plan" + (f" (profile={profile})" if profile else ""))
    cfg = load_config(name, profile)
    exp = cfg.experiment
    conds = list(exp.get("conditions", []))
    for part in (exp.get("parts") or {}).values():
        conds += list(part.get("conditions", []))
    missing = sorted({c for c in conds if not (configs_dir() / "conditions" / f"{c}.yaml").exists()})
    (fail if missing else ok)(f"condition files: {'missing ' + str(missing) if missing else len(set(conds))}")
    for ref in ["tuning_from"]:
        if exp.get(ref):
            ok(f"{ref}: {exp.get(ref)} (that experiment must have finished first)")
    sys.path.insert(0, str(ROOT / "slurm"))
    try:
        import io
        from contextlib import redirect_stdout

        from omegaconf import OmegaConf

        sm = importlib.import_module("submit_experiment")
        cluster = OmegaConf.load(ROOT / "configs" / "cluster" / "csis.yaml")
        sub = sm.Submitter(cluster, True, False, ROOT / "results" / "jobs" / "cmds" / "_preflight", profile)
        with redirect_stdout(io.StringIO()):
            jobs = sm.plan(name, None, None, None, sub, profile)
        ok(f"stage plan: {len(jobs)} jobs: {', '.join(jobs)}")
    except BaseException as e:  # noqa: BLE001
        fail(f"stage plan (dry run) failed: {type(e).__name__}: {e}")


def check_slurm() -> None:
    section("Slurm")
    for t in ["sbatch", "squeue", "sacct", "scancel"]:
        (ok if shutil.which(t) else warn)(f"{t} {'found' if shutil.which(t) else 'not found (fine on a laptop)'}")
    if shutil.which("sinfo"):
        out = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True).stdout.split()
        parts = {p.rstrip("*") for p in out}
        need = {"cpu-short", "gpu-short", "gpu-long"}
        (ok if need <= parts else fail)(f"partitions {sorted(need)} " + ("present" if need <= parts else f"missing; have {sorted(parts)}"))


def check_demo_outputs(name: str) -> None:
    from scopemem.config import exp_name, load_config, resolve_path
    from scopemem.memory.scope_writer import writer_name

    section(f"{name} demo outputs")
    cfg = load_config(name, "demo")
    proc = resolve_path(cfg.paths.processed)
    exp = cfg.experiment
    checks: list[tuple[str, list[Path]]] = [("build: prefs + datasets", [proc / "prefs.jsonl", proc / "folds.json"])]
    if str(exp.kind) == "retrieval":
        model = str(cfg.model_cfg.name)
        checks += [
            ("scopes", [proc / "scopes" / f"{writer_name(cfg)}.jsonl"]),
            ("xt_generate/verify/build", [proc / "cross_topic" / "xtopic_items.jsonl"]),
            ("s3_train", [resolve_path(cfg.paths.models_ft) / f"{writer_name(cfg)}__{cfg.xenc}" / f"fold{k}"
                          for k in range(int(cfg.data.n_folds))]),
        ]
        raw = resolve_path(cfg.paths.raw_results) / exp_name(cfg) / model
        for c in exp.conditions:
            checks.append((f"retrieval {c}", [raw / c]))
        checks.append(("aggregate", [resolve_path(cfg.paths.tables) / f"{exp_name(cfg)}_retrieval.csv"]))
    for label, paths in checks:
        bad = [p for p in paths if not p.exists() or (p.is_dir() and not any(p.iterdir()))]
        (fail if bad else ok)(f"{label}" + (f": missing {[str(p.relative_to(ROOT)) for p in bad]}" if bad else ""))
    if str(exp.kind) == "retrieval":
        raw = resolve_path(cfg.paths.raw_results) / exp_name(cfg) / str(cfg.model_cfg.name)
        cells = {(p.parent.name, p.stem) for p in raw.glob("*/*.jsonl")}
        want = len(exp.conditions) * len(exp.data_sets) * len(exp.n_levels)
        (ok if len(cells) >= want else fail)(f"retrieval cells: {len(cells)}/{want} (condition x split x N)")
    logs = sorted((ROOT / "logs").glob(f"sm-{name}-*.err"), key=lambda p: p.stat().st_mtime)[-40:]
    tb = [p.name for p in logs if "Traceback" in p.read_text(errors="ignore")]
    (fail if tb else ok)("recent logs: " + (f"tracebacks in {tb}" if tb else "no tracebacks"))


# ---------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--profile", default=None, help="config profile to check the plan with (e.g. demo)")
    ap.add_argument("--tests", action="store_true", help="also run pytest")
    ap.add_argument("--check-demo", action="store_true", help="verify outputs of a finished demo run")
    a = ap.parse_args()
    names = [f"E{i}" for i in range(9)] if a.experiments == ["all"] else a.experiments

    from scopemem.config import load_config

    if a.check_demo:
        for n in names:
            check_demo_outputs(n)
    else:
        check_env()
        check_hf_env()
        cfgs = [load_config(n, a.profile) for n in names]
        check_data(cfgs[0], [c.experiment for c in cfgs])
        section("Models (offline, as jobs load them)")
        done: set[str] = set()
        for c in cfgs:
            for hf_id, role, llm in needed_models(c, c.experiment):
                if hf_id not in done:
                    done.add(hf_id)
                    check_model(hf_id, f"{c.experiment.name} {role}", llm)
        for n in names:
            check_configs_and_plan(n, a.profile)
        check_slurm()
        if a.tests:
            section("pytest")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
            (ok if r.returncode == 0 else fail)("pytest")

    print(f"\n{len(FAILS)} FAIL, {len(WARNS)} WARN")
    if FAILS:
        print("Fix the FAIL lines before submitting.")
        sys.exit(1)
    if not a.check_demo:
        print("Next: python slurm/submit_experiment.py " + names[0] + " --profile demo --max-time 01:00:00")


if __name__ == "__main__":
    main()
