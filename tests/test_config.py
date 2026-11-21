from scopemem.config import config_hash, derive_seed, load_config


def test_config_hash_ignores_runtime_keys():
    a = load_config("E1", "smoke")
    b = load_config("E1", "smoke", ["run.shard=3", "paths.cache=/elsewhere"])
    c = load_config("E1", "smoke", ["gate.k_max=5"])
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_derive_seed_deterministic():
    assert derive_seed(13, "E1", 2) == derive_seed(13, "E1", 2) != derive_seed(13, "E1", 3)


def test_fake_backend_forces_fake_model_card():
    cfg = load_config("E1", "smoke")
    assert cfg.model_cfg.name == "fake-qwen7b" and cfg.embedder_cfg.type == "hash"
