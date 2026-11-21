import json

import pytest

from scopemem import run as runmod
from scopemem.utils import read_jsonl
from tests.conftest import make_cfg


def _files(tmp_path):
    return sorted((tmp_path / "raw").glob("E0_smoke/*/*/*.jsonl"))


def test_smoke_records_have_required_fields(tmp_path):
    args = ["experiment=E0", "profile=smoke"] + [
        f"paths.{k}={tmp_path / v}" for k, v in
        [("processed", "processed"), ("cache", "cache"), ("tuning", "tuning"), ("raw_results", "raw"),
         ("models_ft", "ft")]]
    runmod.main(args)
    files = _files(tmp_path)
    assert len(files) == 10                       # one per condition
    for f in files:
        rows = list(read_jsonl(f))
        assert len(rows) == 10
        for r in rows:
            missing = [k for k in runmod.REQUIRED_FIELDS if k not in r]
            assert not missing, (f, missing)
    oracle = list(read_jsonl([f for f in files if "UB_oracle" in str(f)][0]))
    assert all(sorted(r["selected_ids"]) == sorted(r["gold_ids"]) for r in oracle)
    b5 = list(read_jsonl([f for f in files if "B5_llm_select" in str(f)][0]))
    assert all(r["overhead_tokens"] > 0 for r in b5)
    others = [f for f in files if "B5_llm_select" not in str(f)]
    assert all(r["overhead_tokens"] == 0 for f in others for r in read_jsonl(f))


def test_resume_skips_finished_and_stop_exits(tmp_path, monkeypatch):
    base = ["experiment=E0", "profile=smoke", "run.batch_size=3"] + [
        f"paths.{k}={tmp_path / v}" for k, v in
        [("processed", "processed"), ("cache", "cache"), ("tuning", "tuning"), ("raw_results", "raw"),
         ("models_ft", "ft")]]
    # STOP present from the start: exits before writing anything
    monkeypatch.setenv("SCRATCH_RUN", str(tmp_path))
    (tmp_path / "STOP").write_text("")
    with pytest.raises(SystemExit) as e:
        runmod.main(base + ["task=2"])
    assert e.value.code == 0
    (tmp_path / "STOP").unlink()
    runmod.main(base + ["task=2"])                 # B2_all
    f = _files(tmp_path)[0]
    rows = list(read_jsonl(f))
    assert len(rows) == 10
    # simulate a kill mid-run: drop 4 records and a half-written line, then resume
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:6]) + "\n" + lines[6][:20])
    runmod.main(base + ["task=2"])
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip().endswith("}")]
    ids = [r["instance_id"] for r in rows]
    assert len(set(ids)) == 10


def test_task_enumeration(tmp_path):
    cfg = make_cfg(tmp_path, "E2", ["run.n_levels=[25]"])
    tasks = runmod.enumerate_tasks(cfg)
    assert len(tasks) == len(cfg.experiment.data_sets) * len(cfg.experiment.conditions) * int(cfg.experiment.num_shards)
    assert tasks[0] == ("full", "B0_none", 0, 25) and tasks[9] == ("full", "B1_full", 1, 25)
