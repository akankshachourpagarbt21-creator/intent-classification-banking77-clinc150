"""
# Pretrained Transformer Model: DistilBERT

This experiment uses `distilbert-base-uncased`, a pretrained transformer model from Hugging Face, fine-tuned for intent classification.

# Experimental Protocol

1. Four hyperparameter configurations are tested using different learning rates and batch sizes on the validation set with 100% of the training data.
2. The best configuration is selected and reused across all training fractions and random seeds.
3. Early stopping is applied based on validation accuracy to prevent unnecessary training.
4. The test set is evaluated once for each final trained model.

# Requirements

- Internet access is required for the initial download of DistilBERT from the Hugging Face Hub.
- A GPU is recommended for faster training. The experiment can also run on a suitable CPU, although training will be slower.

### Usage
python run_bert.py          # Run the complete experiment
python run_bert.py --quick  # Run a quick smoke test (1 epoch, smaller grid)

"""
import argparse
import copy
import io
import itertools
import json
import time

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import common
from common import (DATASETS, FRACTIONS, OUT, SEEDS, ResultStore, load_dataset, measure_latency,
                    nested_subsets, save_preds, save_truth, scores, set_seed)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN, MAX_EPOCHS, PATIENCE = 64, 10, 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = list(itertools.product([2e-5, 5e-5], [16, 32]))   # (learning rate, batch size)


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN)


def make_batch(tok, enc, rows):
    feats = [{k: enc[k][i] for k in ("input_ids", "attention_mask")} for i in rows]
    return {k: v.to(DEVICE) for k, v in tok.pad(feats, return_tensors="pt").items()}


@torch.inference_mode()
def predict(model, tok, enc, n, bs=128):
    model.eval()
    out = []
    for i in range(0, n, bs):
        b = make_batch(tok, enc, range(i, min(i + bs, n)))
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            out.append(model(**b).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


def fit(tok, cfg, tr_enc, tr_y, va_enc, va_y, n_classes, max_epochs):
    lr, bs = cfg
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max_epochs * int(np.ceil(len(tr_y) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    y = torch.tensor(tr_y)
    best, best_state, bad, epochs = -1.0, None, 0, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr_y))
        for i in range(0, len(perm), bs):
            rows = perm[i:i + bs].tolist()
            b = make_batch(tok, tr_enc, rows)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = model(**b, labels=y[rows].to(DEVICE)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        epochs += 1
        acc = float((predict(model, tok, va_enc, len(va_y)) == va_y).mean())
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or str(OUT / ("quick_bert.csv" if args.quick else "results_bert.csv"))
    if args.quick:
        common.enable_quick_mode()      # smoke tests never write into the real results
    grid, max_epochs = GRID, MAX_EPOCHS
    if args.quick:
        args.fractions, args.seeds, grid, max_epochs = [0.10, 1.00], args.seeds[:1], GRID[:1], 1

    store = ResultStore(args.out)
    best_path = OUT / "bert_best_params.json"
    best_cfgs = json.load(open(best_path)) if best_path.exists() and not args.quick else {}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"device={DEVICE}")

    for ds in args.datasets:
        tr, va, te, labels = load_dataset(ds)
        save_truth(ds, te["label"])
        n_classes = len(labels)
        va_enc, te_enc = encode(tok, va["text"]), encode(tok, te["text"])
        va_y, te_y = va["label"].values, te["label"].values
        key = f"{ds}__distilbert"
        if key not in best_cfgs:
            print(f"  tuning {key} on validation (100% train)")
            tr_enc, best = encode(tok, tr["text"]), (-1, None)
            for cfg in grid:
                set_seed(args.seeds[0])
                _, acc, ep = fit(tok, cfg, tr_enc, tr["label"].values, va_enc, va_y, n_classes, max_epochs)
                print(f"    lr={cfg[0]} bs={cfg[1]}: val_acc={acc:.4f} ({ep} epochs)")
                if acc > best[0]:
                    best = (acc, list(cfg))
            best_cfgs[key] = best[1]
            if not args.quick:
                json.dump(best_cfgs, open(best_path, "w"), indent=2)
        cfg = tuple(best_cfgs[key])

        for frac in args.fractions:
            for seed in args.seeds:
                if store.has(ds, "distilbert", frac, seed):
                    continue
                set_seed(seed)
                sub = tr.iloc[nested_subsets(tr["label"], [frac], seed)[frac]]
                t0 = time.perf_counter()
                model, val_acc, epochs = fit(tok, cfg, encode(tok, sub["text"]), sub["label"].values,
                                             va_enc, va_y, n_classes, max_epochs)
                train_time = time.perf_counter() - t0
                pred = predict(model, tok, te_enc, len(te_y))
                row = {"dataset": ds, "model": "distilbert", "family": "pretrained", "fraction": frac,
                       "seed": seed, "n_train": len(sub), "params": json.dumps(list(cfg)),
                       "val_accuracy": val_acc, **scores(te_y, pred), "train_time_s": train_time,
                       "epochs_run": epochs, "device": DEVICE}
                if frac >= 1.0 and seed == args.seeds[0]:
                    cpu = copy.deepcopy(model).cpu().eval()
                    prev_threads = torch.get_num_threads()
                    torch.set_num_threads(1)

                    @torch.inference_mode()
                    def one(t, cpu=cpu):
                        enc = tok(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                        return cpu(**enc).logits.argmax(-1).item()
                    row.update(measure_latency(one, te["text"]))
                    torch.set_num_threads(prev_threads)
                    buf = io.BytesIO()
                    torch.save(cpu.state_dict(), buf)
                    row["size_mb"] = buf.getbuffer().nbytes / 1e6
                    row["n_params"] = sum(p.numel() for p in cpu.parameters())
                save_preds(ds, "distilbert", frac, seed, pred)
                store.add(row)
                print(f"  {ds} distilbert f={frac} seed={seed}: acc={row['accuracy']:.4f} "
                      f"f1={row['macro_f1']:.4f} epochs={epochs} train={train_time:.0f}s")
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
