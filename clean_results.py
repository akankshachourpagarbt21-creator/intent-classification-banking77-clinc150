"""Clean the result files:  python clean_results.py

What it removes (a .bak backup of every file is saved first):
  1. Smoke-test rows that older versions of the runners wrote into the real results
     (rows from --quick runs trained for only 1-2 epochs).
  2. Runs with an implausible training time, e.g. when the laptop went to sleep mid-run
     (more than 5x the median of its group and over 5 minutes, or over 4 hours).

Afterwards, re-run the same run_*.py commands: the removed runs are done again and
everything already finished is skipped.
"""
import shutil

import pandas as pd

from common import OUT

QUICK_MAX_EPOCHS = {"results_nn.csv": 2, "results_bert.csv": 1}


def clean(path):
    df = pd.read_csv(path)
    n0 = len(df)
    shutil.copy(path, str(path) + ".bak")
    reason = pd.Series("", index=df.index)

    qmax = QUICK_MAX_EPOCHS.get(path.name)
    if qmax is not None and "epochs_run" in df:
        reason[df["epochs_run"] <= qmax] = "smoke-test row"

    median = df.groupby(["dataset", "model", "fraction"])["train_time_s"].transform("median")
    count = df.groupby(["dataset", "model", "fraction"])["train_time_s"].transform("count")
    slow = ((count >= 3) & (df["train_time_s"] > 5 * median) & (df["train_time_s"] > 300)) | \
           (df["train_time_s"] > 4 * 3600)
    reason[slow & (reason == "")] = "implausible training time"

    removed = df[reason != ""].assign(reason=reason[reason != ""])
    df[reason == ""].to_csv(path, index=False)
    print(f"{path.name}: {n0} rows -> {n0 - len(removed)} rows")
    if len(removed):
        cols = [c for c in ["dataset", "model", "fraction", "seed", "epochs_run", "train_time_s", "reason"]
                if c in removed]
        print(removed[cols].to_string(index=False))


if __name__ == "__main__":
    files = sorted(OUT.glob("results_*.csv"))
    if not files:
        print("No results_*.csv files found in outputs/.")
    for f in files:
        clean(f)
    print("\nDone. Re-run your run_*.py commands to redo the removed runs.")
