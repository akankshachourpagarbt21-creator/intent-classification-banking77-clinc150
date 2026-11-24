
"""Analysis of model performance and comparison across ML and DL approaches.

Combines the result files and saved predictions to generate performance
summaries, learning curves, latency comparisons and statistical test results.

Output files:
  Tables:
    - summary_100pct.csv
    - learning_curve_table.csv
    - significance.csv

  Figures:
    - learning_curves.png
    - accuracy_vs_latency.png

Statistical analysis (for each dataset):
  - Selects the best model from each family using validation accuracy.
    The representative run is selected using the median validation accuracy.
  - Compares the selected ML, NN and DistilBERT models pairwise.
  - Uses paired bootstrap resampling (10,000 samples) to estimate accuracy
    differences and their 95% confidence intervals.
  - Applies the exact McNemar test to compare prediction errors.
  - Uses Holm correction to adjust p-values across the three comparisons.
"""

import glob

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import OUT

TABLES, FIGS = OUT / "tables", OUT / "figures"
TABLES.mkdir(exist_ok=True), FIGS.mkdir(exist_ok=True)
RNG = np.random.RandomState(0)


def load_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(OUT / "results_*.csv")))
    if not files:
        raise SystemExit("No results_*.csv found in outputs/. Run the runners first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summary_100(df):
    full = df[np.isclose(df.fraction, 1.0)]
    g = full.groupby(["dataset", "family", "model"])
    out = g.agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                val_acc=("val_accuracy", "mean"), train_time_s=("train_time_s", "mean"),
                lat_p50_ms=("lat_p50_ms", "max"), lat_p95_ms=("lat_p95_ms", "max"),
                size_mb=("size_mb", "max"), n_params=("n_params", "max"), n_runs=("seed", "count"))
    out = out.reset_index().round(4)
    out.to_csv(TABLES / "summary_100pct.csv", index=False)
    return out


def curve_table(df):
    g = df.groupby(["dataset", "model", "fraction"]).accuracy.agg(["mean", "std", "count"]).reset_index()
    g.round(4).to_csv(TABLES / "learning_curve_table.csv", index=False)
    return g


def plot_curves(curves):
    ds_list = sorted(curves.dataset.unique())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(6 * len(ds_list), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        for model, d in curves[curves.dataset == ds].groupby("model"):
            d = d.sort_values("fraction")
            ax.plot(d.fraction * 100, d["mean"], marker="o", label=model)
            ax.fill_between(d.fraction * 100, d["mean"] - d["std"].fillna(0),
                            d["mean"] + d["std"].fillna(0), alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set(title=f"{ds}: test accuracy vs training data", xlabel="% of training data (log)",
               ylabel="Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=200)


def plot_efficiency(summary):
    s = summary.dropna(subset=["lat_p50_ms"])
    if s.empty:
        return
    fig, axes = plt.subplots(1, s.dataset.nunique(), figsize=(6 * s.dataset.nunique(), 4.2), squeeze=False)
    for ax, (ds, d) in zip(axes[0], s.groupby("dataset")):
        ax.scatter(d.lat_p50_ms, d.acc_mean * 100, s=60)
        for _, r in d.iterrows():
            ax.annotate(r.model, (r.lat_p50_ms, r.acc_mean * 100), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set(xscale="log", title=f"{ds}: accuracy vs CPU latency", xlabel="p50 latency, ms (log)",
               ylabel="Accuracy (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_vs_latency.png", dpi=200)


def representative(df, ds, family):
    """Best model of a family by validation accuracy; seed with the median validation accuracy."""
    d = df[(df.dataset == ds) & (df.family == family) & np.isclose(df.fraction, 1.0)]
    if d.empty:
        return None
    best_model = d.groupby("model").val_accuracy.mean().idxmax()
    runs = d[d.model == best_model].sort_values("val_accuracy")
    row = runs.iloc[len(runs) // 2]
    pred = np.load(OUT / "preds" / f"{ds}__{best_model}__{row.fraction}__{int(row.seed)}.npy")
    return best_model, pred


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def compare(a, b, y):
    ca, cb = (a == y).astype(int), (b == y).astype(int)
    n = len(y)
    idx = RNG.randint(0, n, size=(10_000, n))
    diffs = (ca[idx] - cb[idx]).mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    b_only, c_only = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    p_mc = binomtest(b_only, b_only + c_only, 0.5).pvalue if b_only + c_only else 1.0
    return {"acc_diff_pts": 100 * (ca.mean() - cb.mean()), "ci_low_pts": 100 * lo, "ci_high_pts": 100 * hi,
            "p_bootstrap": p_boot, "p_mcnemar": p_mc, "a_only_correct": b_only, "b_only_correct": c_only}


def significance(df):
    rows = []
    for ds in sorted(df.dataset.unique()):
        y = np.load(OUT / "preds" / f"{ds}__y_true.npy")
        reps = {f: representative(df, ds, f) for f in ("ml", "nn", "pretrained")}
        pairs = [("ml", "nn"), ("ml", "pretrained"), ("nn", "pretrained")]
        block = []
        for fa, fb in pairs:
            if reps[fa] and reps[fb]:
                res = compare(reps[fa][1], reps[fb][1], y)
                block.append({"dataset": ds, "A": reps[fa][0], "B": reps[fb][0], **res})
        if block:
            adj = holm([r["p_mcnemar"] for r in block])
            for r, p in zip(block, adj):
                r["p_mcnemar_holm"] = p
            rows += block
    out = pd.DataFrame(rows).round(4)
    out.to_csv(TABLES / "significance.csv", index=False)
    return out


if __name__ == "__main__":
    results = load_results()
    print(summary_100(results).to_string(index=False))
    curves = curve_table(results)
    plot_curves(curves)
    plot_efficiency(summary_100(results))
    sig = significance(results)
    print("\nPairwise tests (A vs B; positive diff = A better):")
    print(sig.to_string(index=False) if not sig.empty else "need results from >= 2 families")
