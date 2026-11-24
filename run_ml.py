"""#Classical Machine Learning Models

This experiment uses TF-IDF vectorization with word-level unigrams and bigrams to convert text into numerical features. Three traditional machine learning classifiers are evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

# Experimental Protocol

1. Each model is tuned using up to 8 hyperparameter configurations on the validation set, with the full training dataset.
2. The best-performing configuration is selected and reused across all training data fractions.
3. For each training fraction, a stratified subset of the training data is created, and the model is evaluated on the test set.
4. The full training fraction (100%) is evaluated once because the models are deterministic for a fixed dataset and configuration. Smaller training fractions are evaluated using 5 different random seeds to account for subset selection variability.

# Usage
python run_ml.py          # Run the complete experiment
python run_ml.py --quick  # Run a quick smoke test
"""
import argparse
import json
import pickle
import time
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import common
from common import (DATASETS, FRACTIONS, OUT, SEEDS, ResultStore, load_dataset,
                    measure_latency, nested_subsets, save_preds, save_truth, scores, set_seed)

warnings.filterwarnings("ignore")

GRIDS = {                                   # <= 8 configurations per model
    "nb": ("alpha", [0.01, 0.03, 0.1, 0.3, 1.0]),
    "lr": ("C", [1, 10, 100]),
    "svm": ("C", [0.1, 0.3, 1, 3]),
}


def make_pipe(model: str, value):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    if model == "nb":
        clf = MultinomialNB(alpha=value)
    elif model == "lr":
        clf = LogisticRegression(C=value, max_iter=1000)
    else:
        clf = LinearSVC(C=value)
    return make_pipeline(tfidf, clf)


def tune(model, tr, va):
    name, values = GRIDS[model]
    best = (-1, None)
    for v in values:
        pipe = make_pipe(model, v).fit(tr["text"], tr["label"])
        acc = float((pipe.predict(va["text"]) == va["label"]).mean())
        print(f"    tune {model} {name}={v}: val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, v)
    return {name: best[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(GRIDS))
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true", help="2 fractions, 1 seed, tiny grids")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or str(OUT / ("quick_ml.csv" if args.quick else "results_ml.csv"))
    if args.quick:
        common.enable_quick_mode()      # smoke tests never write into the real results
    if args.quick:
        args.fractions, args.seeds = [0.10, 1.00], args.seeds[:1]
        for m in GRIDS:
            GRIDS[m] = (GRIDS[m][0], GRIDS[m][1][:2])

    store = ResultStore(args.out)
    best_path = OUT / "ml_best_params.json"
    best_params = json.load(open(best_path)) if best_path.exists() and not args.quick else {}

    for ds in args.datasets:
        tr, va, te, labels = load_dataset(ds)
        save_truth(ds, te["label"])
        for model in args.models:
            key = f"{ds}__{model}"
            if key not in best_params:
                print(f"  tuning {key} on validation (100% train)")
                best_params[key] = tune(model, tr, va)
                if not args.quick:
                    json.dump(best_params, open(best_path, "w"), indent=2)
            pname, pval = next(iter(best_params[key].items()))

            for frac in args.fractions:
                seeds = args.seeds[:1] if frac >= 1.0 else args.seeds
                for seed in seeds:
                    if store.has(ds, model, frac, seed):
                        continue
                    set_seed(seed)
                    idx = nested_subsets(tr["label"], [frac], seed)[frac]
                    sub = tr.iloc[idx]
                    pipe = make_pipe(model, pval)
                    t0 = time.perf_counter()
                    pipe.fit(sub["text"], sub["label"])
                    train_time = time.perf_counter() - t0
                    pred = pipe.predict(te["text"])
                    row = {"dataset": ds, "model": model, "family": "ml", "fraction": frac,
                           "seed": seed, "n_train": len(sub), "params": json.dumps(best_params[key]),
                           "val_accuracy": float((pipe.predict(va["text"]) == va["label"]).mean()),
                           **scores(te["label"], pred), "train_time_s": train_time,
                           "epochs_run": np.nan, "device": "cpu"}
                    if frac >= 1.0:   # efficiency metrics measured once, on the full-data model
                        row.update(measure_latency(lambda t: pipe.predict([t]), te["text"]))
                        row["size_mb"] = len(pickle.dumps(pipe)) / 1e6
                        row["n_params"] = np.nan
                    save_preds(ds, model, frac, seed, pred)
                    store.add(row)
                    print(f"  {ds} {model} f={frac} seed={seed}: acc={row['accuracy']:.4f} "
                          f"f1={row['macro_f1']:.4f} train={train_time:.1f}s")


if __name__ == "__main__":
    main()
