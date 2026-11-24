"""
Usage:  python run_nn.py --models bilstm textcnn            
        python run_nn.py --quick --models bilstm textcnn    # smoke test
"""
import argparse
import copy
import io
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import common
from common import (DATASETS, FRACTIONS, OUT, ROOT, SEEDS, ResultStore, load_dataset, load_glove,
                    measure_latency, nested_subsets, save_preds, save_truth, scores, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN, EMB_DIM, BATCH, MAX_EPOCHS, PATIENCE = 64, 300, 64, 40, 5
GLOVE_PATH = ROOT / "glove" / "glove.6B.300d.txt"
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")

GRIDS = {   # 8 configs each
    "bilstm": list(itertools.product([128, 256], [0.3, 0.5], [1e-3, 3e-3])),   # hidden, dropout, lr
    "textcnn": list(itertools.product([100, 200], [0.3, 0.5], [1e-3, 3e-3])),  # filters, dropout, lr
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower()) or ["<unk>"]


class Vocab:
    def __init__(self, texts):
        words = sorted({w for t in texts for w in tokenize(t)})
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in tokenize(text)][:MAX_LEN]


def build_embeddings(vocab):
    mat = np.random.normal(0, 0.1, (len(vocab.itos), EMB_DIM)).astype("float32")
    mat[0] = 0
    found = False
    if GLOVE_PATH.exists():
        glove, hits = load_glove(str(GLOVE_PATH)), 0      # file is read once, then cached in memory
        found = True
        for word, i in vocab.stoi.items():
            vec = glove.get(word)
            if vec is not None:
                mat[i] = vec
                hits += 1
        print(f"    GloVe coverage: {hits}/{len(vocab.itos)} vocab words")
    else:
        print(f"    WARNING: {GLOVE_PATH} not found -> random embeddings (results flagged glove=False)")
    return torch.tensor(mat), found


def collate(seqs, min_len=1):
    n = max(max(len(s) for s in seqs), min_len)
    x = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x


class BiLSTM(nn.Module):
    def __init__(self, emb, hidden, dropout, n_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        mask = x != 0
        lengths = mask.sum(1).clamp(min=1).cpu()
        e = self.drop(self.emb(x))
        out, _ = self.lstm(pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values   # masked max-pool over time
        return self.fc(self.drop(out))


class TextCNN(nn.Module):
    KERNELS = (2, 3, 4)

    def __init__(self, emb, filters, dropout, n_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(EMB_DIM, filters, k) for k in self.KERNELS)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(self.KERNELS), n_classes)

    def forward(self, x):
        e = self.drop(self.emb(x)).transpose(1, 2)
        feats = [torch.relu(c(e)).max(2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, 1)))


def build(model, emb, cfg, n_classes):
    size, drop, _ = cfg
    return (BiLSTM if model == "bilstm" else TextCNN)(emb, size, drop, n_classes)


def predict(net, seqs, min_len):
    net.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(seqs), 256):
            x = collate(seqs[i:i + 256], min_len).to(next(net.parameters()).device)
            preds.append(net(x).argmax(1).cpu())
    return torch.cat(preds).numpy()


def fit(model, cfg, emb, tr_seqs, tr_y, va_seqs, va_y, n_classes, max_epochs):
    net = build(model, emb, cfg, n_classes).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=cfg[2])
    min_len = 4 if model == "textcnn" else 1
    y = torch.tensor(tr_y)
    best, best_state, bad, epochs = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(tr_seqs))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x = collate([tr_seqs[j] for j in b], min_len).to(DEVICE)
            loss = nn.functional.cross_entropy(net(x), y[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs += 1
        acc = float((predict(net, va_seqs, min_len) == va_y).mean())
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best, epochs, min_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=["bilstm", "textcnn"])
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or str(OUT / ("quick_nn.csv" if args.quick else "results_nn.csv"))
    if args.quick:
        common.enable_quick_mode()      # smoke tests never write into the real results
    max_epochs = MAX_EPOCHS
    if args.quick:
        args.fractions, args.seeds, max_epochs = [0.10, 1.00], args.seeds[:1], 2
        for m in GRIDS:
            GRIDS[m] = GRIDS[m][:2]

    store = ResultStore(args.out)
    best_path = OUT / "nn_best_params.json"
    best_cfgs = json.load(open(best_path)) if best_path.exists() and not args.quick else {}
    print(f"device={DEVICE}")

    for ds in args.datasets:
        tr, va, te, labels = load_dataset(ds)
        save_truth(ds, te["label"])
        n_classes = len(labels)
        va_y, te_y = va["label"].values, te["label"].values
        for model in args.models:
            key = f"{ds}__{model}"
            if key not in best_cfgs:
                vocab_full = Vocab(tr["text"])
                emb_full, _ = build_embeddings(vocab_full)
                tr_seqs = [vocab_full.encode(t) for t in tr["text"]]
                va_seqs = [vocab_full.encode(t) for t in va["text"]]
                print(f"  tuning {key} on validation (100% train, seed {args.seeds[0]})")
                best = (-1, None)
                for cfg in GRIDS[model]:
                    set_seed(args.seeds[0])
                    _, acc, ep, _ = fit(model, cfg, emb_full, tr_seqs, tr["label"].values,
                                        va_seqs, va_y, n_classes, max_epochs)
                    print(f"    cfg={cfg}: val_acc={acc:.4f} ({ep} epochs)")
                    if acc > best[0]:
                        best = (acc, list(cfg))
                best_cfgs[key] = best[1]
                if not args.quick:
                    json.dump(best_cfgs, open(best_path, "w"), indent=2)
            cfg = tuple(best_cfgs[key])

            for frac in args.fractions:
                for seed in args.seeds:
                    if store.has(ds, model, frac, seed):
                        continue
                    set_seed(seed)
                    idx = nested_subsets(tr["label"], [frac], seed)[frac]
                    sub = tr.iloc[idx]
                    vocab = Vocab(sub["text"])                    # vocabulary from the subset only
                    emb, glove = build_embeddings(vocab)
                    s_seqs = [vocab.encode(t) for t in sub["text"]]
                    v_seqs = [vocab.encode(t) for t in va["text"]]
                    t_seqs = [vocab.encode(t) for t in te["text"]]
                    t0 = time.perf_counter()
                    net, val_acc, epochs, min_len = fit(model, cfg, emb, s_seqs, sub["label"].values,
                                                        v_seqs, va_y, n_classes, max_epochs)
                    train_time = time.perf_counter() - t0
                    pred = predict(net, t_seqs, min_len)
                    row = {"dataset": ds, "model": model, "family": "nn", "fraction": frac,
                           "seed": seed, "n_train": len(sub), "params": json.dumps(list(cfg)),
                           "val_accuracy": val_acc, **scores(te_y, pred), "train_time_s": train_time,
                           "epochs_run": epochs, "device": DEVICE, "glove": glove}
                    if frac >= 1.0 and seed == args.seeds[0]:
                        cpu = copy.deepcopy(net).cpu().eval()
                        prev_threads = torch.get_num_threads()
                        torch.set_num_threads(1)

                        def one(t, cpu=cpu, vocab=vocab, min_len=min_len):
                            with torch.inference_mode():
                                return cpu(collate([vocab.encode(t)], min_len)).argmax(1).item()
                        row.update(measure_latency(one, te["text"]))
                        torch.set_num_threads(prev_threads)
                        buf = io.BytesIO()
                        torch.save(cpu.state_dict(), buf)
                        row["size_mb"] = buf.getbuffer().nbytes / 1e6
                        row["n_params"] = sum(p.numel() for p in cpu.parameters())
                    save_preds(ds, model, frac, seed, pred)
                    store.add(row)
                    print(f"  {ds} {model} f={frac} seed={seed}: acc={row['accuracy']:.4f} "
                          f"f1={row['macro_f1']:.4f} epochs={epochs} train={train_time:.0f}s")


if __name__ == "__main__":
    main()
