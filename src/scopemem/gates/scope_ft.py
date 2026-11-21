"""S3: cross-encoder fine-tuned on training-fold topics (topic-fold CV, section 6.4).

Training pairs per fold f (training topics = topics whose fold != f), built from PrefEval items:
  positive  (t.question, scope(t.pref))                       for every item t in training topics
  hard neg  3 x (t.question, scope(p)) with p from OTHER training topics, highest bi-encoder sim
  rand neg  3 x (t.question, scope(p)) with p random from OTHER training topics
  global    with p=0.3, (t.question, scope(random global pref)) as an extra positive (D-017)
1 epoch, BCE-with-logits, lr 2e-5, batch 32 (plain torch loop over the HF model inside the
sentence-transformers CrossEncoder, to stay robust to ST API changes; D-018).

CLI:  python -m scopemem.gates.scope_ft experiment=E1 model=qwen7b fold=0
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import derive_seed, resolve_path
from ..memory.store import PreferenceStore
from .scope_xenc import ScopeXencGate


class ScopeFTGate(ScopeXencGate):
    """Same decision rule as S2; the cross-encoder is the fold's fine-tuned model."""


def ft_model_dir(cfg, writer_name: str, fold: int) -> Path:
    return resolve_path(cfg.paths.models_ft) / f"{writer_name}__{cfg.xenc}" / f"fold{fold}"


def build_training_pairs(items, store: PreferenceStore, folds: dict[str, int], fold: int, embedder,
                         fields: Sequence[str], seed: int, hard_neg: int = 3, rand_neg: int = 3,
                         global_ids: Sequence[str] = (), global_prob: float = 0.3,
                         max_pairs: int | None = None, xtopic_items: Sequence[dict] = ()
                         ) -> list[tuple[str, str, float]]:
    """Queries come from PrefEval items of training topics and, if available, cross-topic items
    whose home AND query topics are both training topics (primary + reserve; D-033)."""
    train_items = [it for it in items if folds[it.topic] != fold]
    train_ids = [it.id for it in train_items]
    recs = {pid: store.record(pid) for pid in train_ids}
    texts = {pid: r.joined(fields) for pid, r in recs.items()}
    rng = random.Random(derive_seed(seed, "s3pairs", fold))
    ids = np.array(train_ids)
    topics = np.array([it.topic for it in train_items])
    pv = embedder.encode([texts[i] for i in train_ids])
    # (query, target pref id, topics whose prefs may not be negatives)
    queries = [(it.question, it.id, {it.topic}) for it in train_items]
    queries += [(x["request"], x["pref_id"], {x["home_topic"], x["query_topic"]}) for x in xtopic_items
                if folds[x["home_topic"]] != fold and folds[x["query_topic"]] != fold and x["pref_id"] in texts]
    qv = embedder.encode([q for q, _, _ in queries], is_query=True)
    pairs: list[tuple[str, str, float]] = []
    for k, (q, pid, excl) in enumerate(queries):
        pairs.append((q, texts[pid], 1.0))
        other = np.where(~np.isin(topics, list(excl)))[0]
        sims = pv[other] @ qv[k]
        hard = other[np.argsort(-sims, kind="stable")[:hard_neg]]
        hs = set(hard.tolist())
        rest = [j for j in other.tolist() if j not in hs]
        rnd = rng.sample(rest, min(rand_neg, len(rest)))
        for j in list(hard) + rnd:
            pairs.append((q, texts[str(ids[j])], 0.0))
        if global_ids and rng.random() < global_prob:
            g = store.record(rng.choice(list(global_ids)))
            pairs.append((q, g.joined(fields), 1.0))
    rng.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def train_cross_encoder(base_model: str, pairs: Sequence[tuple[str, str, float]], out_dir: Path,
                        lr: float = 2e-5, epochs: int = 1, batch_size: int = 32, max_length: int = 512,
                        seed: int = 0) -> dict:
    import torch
    from sentence_transformers import CrossEncoder

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=device)
    model, tok = ce.model, ce.tokenizer
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    steps, total = 0, 0.0
    for _ in range(epochs):
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i + batch_size]
            enc = tok([p[0] for p in b], [p[1] for p in b], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            y = torch.tensor([p[2] for p in b], dtype=torch.float32, device=device)
            logits = model(**enc).logits.view(-1)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            steps += 1
            total += float(loss.item())
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    info = {"steps": steps, "mean_loss": total / max(steps, 1), "n_pairs": len(pairs), "device": device}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=1))
    return info


def train_fold(cfg, fold: int, writer_name: str | None = None) -> Path:
    from ..config import condition_cfg
    from ..data.build_multi import load_globals, load_pref_table
    from ..data.cross_topic import load_xtopic_items
    from ..data.prefeval import load_items
    from ..embed import build_embedder
    from ..memory.scope_writer import scopes_path, writer_name as _writer_name
    from ..memory.store import ScopeBank

    processed = resolve_path(cfg.paths.processed)
    folds = json.loads((processed / "folds.json").read_text())
    writer_name = writer_name or _writer_name(cfg)
    bank = ScopeBank.load(scopes_path(cfg, writer_name))
    store = PreferenceStore(load_pref_table(processed), bank)
    items = load_items(resolve_path(cfg.paths.prefeval_root), "explicit")
    fields = list(condition_cfg("S3_scope_ft").fields)
    emb = build_embedder(cfg.embedder_cfg)
    st = cfg.s3_train
    pairs = build_training_pairs(items, store, folds, fold, emb, fields, int(cfg.seed),
                                 int(st.hard_neg), int(st.rand_neg), [g.id for g in load_globals()],
                                 float(cfg.data.global_prob), st.get("max_train_pairs"),
                                 load_xtopic_items(processed))
    out = ft_model_dir(cfg, writer_name, fold)
    if cfg.xenc_cfg.type == "hash":     # smoke/tests: nothing to train
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_info.json").write_text(json.dumps({"n_pairs": len(pairs), "fake": True}))
        return out
    info = train_cross_encoder(cfg.xenc_cfg.hf_id, pairs, out, float(st.lr), int(st.epochs),
                               int(st.batch_size), int(cfg.xenc_cfg.get("max_length", 512)),
                               derive_seed(cfg.seed, "s3train", fold))
    print(f"fold {fold}: {info}")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    from ..config import load_config, parse_cli
    from ..utils import setup_logging

    setup_logging()
    sel, dot = parse_cli(argv if argv is not None else sys.argv[1:])
    fold = None
    rest = []
    for d in dot:
        if d.startswith("fold="):
            fold = int(d.split("=", 1)[1])
        else:
            rest.append(d)
    cfg = load_config(sel.get("experiment"), sel.get("profile"), rest)
    folds = [fold] if fold is not None else list(range(int(cfg.data.n_folds)))
    for f in folds:
        print(train_fold(cfg, f))


if __name__ == "__main__":
    main()
