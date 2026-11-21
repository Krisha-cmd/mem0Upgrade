import importlib.util
import re
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("submit_experiment", ROOT / "slurm" / "submit_experiment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dry(exp, tmp_path, **kw):
    m = _mod()
    cluster = OmegaConf.load(ROOT / "configs" / "cluster" / "csis.yaml")
    sub = m.Submitter(cluster, dry_run=True, wait=False, cmd_dir=tmp_path)
    jobs = m.plan(exp, kw.get("models"), None, None, sub, None)
    return jobs, sub.lines


def test_e2_dry_run_lines(tmp_path):
    jobs, lines = _dry("E2", tmp_path)
    assert list(jobs) == ["E2-gen-qwen7b", "E2-gen-llama8b", "E2-gen-qwen32b",
                          "E2-judge-qwen7b", "E2-judge-llama8b", "E2-judge-qwen32b", "E2-aggregate"]
    gen7, gen8, gen32 = lines[0], lines[1], lines[2]
    for l in (gen7, gen8):
        assert "--partition=gpu-short" in l and "--gres=gpu:a100-80gb:1" in l
        assert re.search(r"--array=0-159%3 ", l)           # 2 data sets x 10 conditions x 8 shards, throttle 3
    assert "--partition=gpu-long" in gen32 and "--gres=gpu:a100-80gb:2" in gen32
    assert re.search(r"--array=0-159%1 ", gen32)            # 2-GPU arrays throttle 1
    # GPU stages are serialised with afterany; judge waits for its own model's gen with afterok
    assert f"afterany:{jobs['E2-gen-qwen7b']}" in gen8
    judge8 = lines[4]
    assert f"afterok:{jobs['E2-gen-llama8b']}" in judge8 and f"afterany:{jobs['E2-judge-qwen7b']}" in judge8
    agg = lines[-1]
    assert "--partition=cpu-short" in agg and "--gres" not in agg
    for j in ("E2-judge-qwen7b", "E2-judge-llama8b", "E2-judge-qwen32b"):
        assert jobs[j] in agg
    assert all("--qos=" in l for l in lines)
    assert all("--time=" in l for l in lines)


def test_never_more_than_three_gpus(tmp_path):
    _, lines = _dry("E2", tmp_path)
    for l in lines:
        g = re.search(r"--gres=\S+:(\d+)", l)
        t = re.search(r"--array=\S+%(\d+)", l)
        if g:
            assert int(g.group(1)) * int(t.group(1)) <= 3


def test_e1_stage_graph(tmp_path):
    jobs, lines = _dry("E1", tmp_path)
    assert list(jobs) == ["E1-build", "E1-scopes-qwen7b", "E1-xt_generate", "E1-xt_verify", "E1-xt_build",
                          "E1-s3_train-qwen7b", "E1-retrieval-qwen7b", "E1-aggregate"]
    by = dict(zip(jobs, lines))
    assert "--partition=gpu-long" in by["E1-xt_generate"] and "a100-80gb:2" in by["E1-xt_generate"]
    assert "--partition=gpu-short" in by["E1-xt_verify"] and "a100-80gb:1" in by["E1-xt_verify"]
    assert "--partition=cpu-short" in by["E1-xt_build"]
    assert f"afterok:{jobs['E1-xt_build']}" in by["E1-s3_train-qwen7b"]
    assert "--array=0-4%3" in by["E1-s3_train-qwen7b"] and "--array=0-5%3" in by["E1-retrieval-qwen7b"]
    assert f"afterok:{jobs['E1-build']}" in by["E1-scopes-qwen7b"]
