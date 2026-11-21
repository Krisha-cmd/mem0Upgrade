from __future__ import annotations

import pytest

from scopemem.config import load_config


def make_cfg(tmp_path, experiment="E0", extra=()):
    over = [f"paths.processed={tmp_path / 'processed'}", f"paths.cache={tmp_path / 'cache'}",
            f"paths.tuning={tmp_path / 'tuning'}", f"paths.raw_results={tmp_path / 'raw'}",
            f"paths.models_ft={tmp_path / 'ft'}", f"paths.tables={tmp_path / 'tables'}",
            f"paths.figures={tmp_path / 'figures'}", *extra]
    return load_config(experiment, "smoke", over)


@pytest.fixture
def smoke_cfg(tmp_path):
    return make_cfg(tmp_path)


@pytest.fixture
def tiny_data():
    from scopemem.config import resolve_path
    from scopemem.data.build_multi import load_globals, make_topic_folds
    from scopemem.data.prefeval import list_topics, load_filler, load_items

    root = resolve_path("tests/fixtures/prefeval_tiny")
    items = load_items(root)
    return {"root": root, "items": items, "filler": load_filler(root), "globals": load_globals(),
            "folds": make_topic_folds(list_topics(root), 5, 13)}
