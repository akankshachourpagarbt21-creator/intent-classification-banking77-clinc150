"""Shared utilities for the ML vs DL intent-classification study.

Keeps every model family on identical data, splits, subsets, metrics and timing code,
which is the core of the controlled design.
"""
import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED_DIR = OUT / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def enable_quick_mode() -> None:
    """Smoke tests (--quick) must never touch real results: send predictions to a separate folder."""
    global PRED_DIR
    PRED_DIR = OUT / "quick_preds"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["banking77", "clinc150"]


@lru_cache(maxsize=1)
def load_glove(path: str) -> dict:
    """Read the GloVe text file ONCE per process; returns {word: float32 vector}."""
    vecs = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word, rest = line.rstrip().split(" ", 1)
            vecs[word] = np.asarray(rest.split(), dtype="float32")
    return vecs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:        # torch missing or not working: fine for the classical-ML runner
        pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def load_dataset(name: str, verbose: bool = True):
    """Return (train, val, test) DataFrames with columns text,label (int ids) and the label names.

    banking77: official 10,003 train / 3,080 test; 10% of train held out (stratified, fixed
               seed 0) as validation because no official validation split exists.
    clinc150 : in-scope data only (official 15,000 train / 3,000 val / 4,500 test).
    Train/val rows whose normalized text also appears in test are dropped (leakage guard).
    """
    if name == "banking77":
        tr = pd.read_csv(DATA / "banking77_train.csv").rename(columns={"category": "label"})
        te = pd.read_csv(DATA / "banking77_test.csv").rename(columns={"category": "label"})
        tr, va = train_test_split(tr, test_size=0.10, stratify=tr["label"], random_state=0)
    elif name == "clinc150":
        raw = json.load(open(DATA / "clinc150_full.json"))
        mk = lambda rows: pd.DataFrame(rows, columns=["text", "label"])
        tr, va, te = mk(raw["train"]), mk(raw["val"]), mk(raw["test"])
    else:
        raise ValueError(name)

    test_texts = set(te["text"].map(_norm))
    removed = {}
    keep_tr = ~tr["text"].map(_norm).isin(test_texts)
    keep_va = ~va["text"].map(_norm).isin(test_texts)
    removed["train"], removed["val"] = int((~keep_tr).sum()), int((~keep_va).sum())
    tr, va, te = tr[keep_tr].copy(), va[keep_va].copy(), te.copy()

    labels = sorted(tr["label"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    for df in (tr, va, te):
        df["label"] = df["label"].map(lab2id)
    tr, va, te = (d.reset_index(drop=True) for d in (tr, va, te))
    if verbose:
        print(f"[{name}] train={len(tr)} val={len(va)} test={len(te)} classes={len(labels)} "
              f"| leakage rows removed: {removed}")
    return tr, va, te, labels


def nested_subsets(y, fractions, seed: int):
    """Stratified, NESTED subsets: the 5% set is contained in the 10% set, and so on.

    Within each class the order is one random permutation (seeded); a fraction f takes the first
    max(1, round(f * n_class)) items, so every class is always represented.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    perms = {c: rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    out = {}
    for f in fractions:
        idx = np.concatenate([p[: max(1, int(round(f * len(p))))] for p in perms.values()])
        out[f] = np.sort(idx)
    return out


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def measure_latency(predict_one, texts, n_warmup: int = 100, n: int = 1000) -> dict:
    """End-to-end latency (raw text -> label), batch size 1. Returns p50/p95 in milliseconds."""
    texts = list(texts)[: n_warmup + n]
    for t in texts[:n_warmup]:
        predict_one(t)
    times = []
    for t in texts[n_warmup:]:
        s = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - s) * 1000)
    return {"lat_p50_ms": float(np.percentile(times, 50)), "lat_p95_ms": float(np.percentile(times, 95))}


def save_preds(dataset, model, fraction, seed, y_pred) -> None:
    np.save(PRED_DIR / f"{dataset}__{model}__{fraction}__{seed}.npy", np.asarray(y_pred))


def save_truth(dataset, y_true) -> None:
    np.save(PRED_DIR / f"{dataset}__y_true.npy", np.asarray(y_true))


class ResultStore:
    """Append-only CSV so long runs can be resumed after a Colab disconnect."""

    def __init__(self, path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def has(self, dataset, model, fraction, seed) -> bool:
        if self.df.empty:
            return False
        m = (self.df.dataset == dataset) & (self.df.model == model) & \
            (np.isclose(self.df.fraction, fraction)) & (self.df.seed == seed)
        return bool(m.any())

    def add(self, row: dict) -> None:
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
