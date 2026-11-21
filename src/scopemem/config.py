"""Config system: base.yaml <- experiments/<E>.yaml <- profiles/<p>.yaml <- CLI dotlist.

Model / embedder / cross-encoder / condition configs are looked up by name and attached
as ``cfg.model_cfg`` etc., so a run's resolved config is fully self-describing and hashable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

# Keys that do not change results and are excluded from config_hash.
_RUNTIME_KEYS = {"run", "paths", "profile", "task", "fold", "stages"}


def repo_root() -> Path:
    env = os.environ.get("SCOPEMEM_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return OmegaConf.load(path)  # type: ignore[return-value]


def load_named(kind: str, name: str) -> DictConfig:
    """kind in {models, embedders, xencs, conditions, experiments, cluster, profiles}."""
    return _load_yaml(configs_dir() / kind / f"{name}.yaml")


def parse_cli(argv: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Split ``key=value`` args into top-level selectors and dotlist overrides."""
    selectors: dict[str, str] = {}
    dotlist: list[str] = []
    for a in argv:
        if "=" not in a:
            raise SystemExit(f"bad argument {a!r}: expected key=value")
        k, v = a.split("=", 1)
        if k in {"experiment", "profile"}:
            selectors[k] = v
        else:
            dotlist.append(a)
    return selectors, dotlist


def load_config(experiment: str | None = None, profile: str | None = None,
                overrides: Iterable[str] = ()) -> DictConfig:
    cfg = _load_yaml(configs_dir() / "base.yaml")
    if experiment:
        exp = load_named("experiments", experiment)
        cfg = OmegaConf.merge(cfg, {"experiment": exp})
    if profile:
        cfg = OmegaConf.merge(cfg, load_named("profiles", profile))
        cfg.profile = profile
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _attach(cfg)
    return cfg


def _attach(cfg: DictConfig) -> None:
    """Resolve named sub-configs. Fake backend forces the fake model card."""
    model = "fake" if cfg.llm_backend == "fake" else cfg.model
    cfg.model_cfg = load_named("models", model)
    if cfg.llm_backend == "fake":
        cfg.model_cfg.name = f"fake-{cfg.model}"
    cfg.embedder_cfg = load_named("embedders", cfg.embedder)
    cfg.xenc_cfg = load_named("xencs", cfg.xenc)


def condition_cfg(name: str) -> DictConfig:
    return load_named("conditions", name)


def config_hash(cfg: DictConfig) -> str:
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    d = {k: v for k, v in d.items() if k not in _RUNTIME_KEYS}
    if isinstance(d.get("experiment"), dict):
        d["experiment"] = {k: v for k, v in d["experiment"].items() if k != "stages"}
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def derive_seed(*parts: Any) -> int:
    """Deterministic sub-seed from the global seed and any identifying parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def exp_name(cfg: DictConfig) -> str:
    return f"{cfg.experiment.name}{cfg.exp_suffix}"
