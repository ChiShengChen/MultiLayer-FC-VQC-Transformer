#!/usr/bin/env python3
"""
XGBoost on MNIST 4 vs 9 (3-seed) with macOS-safe settings.

The default `tree_method='hist'` triggers an OpenMP segfault on macOS with
xgboost 3.x. Using `tree_method='exact'` and a single worker thread avoids
the crash. Linux users can drop these workarounds.

Usage:  python run_xgboost_mnist.py
"""
import os
# Single-threaded to avoid macOS OpenMP conflicts; harmless on Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import statistics
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


CSV = "MNIST_4v9/data/mnist_4v9_pca11.csv"
SEEDS = [42, 123, 7]


def main():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    print(f"Data: {X.shape}  classes: {np.bincount(y).tolist()}")

    accs, f1s = [], []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed)
        Xv, Xte, yv, yte = train_test_split(Xte, yte, test_size=0.50, random_state=seed)

        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            tree_method="exact",  # macOS-safe; avoids OpenMP segfault in `hist`
            nthread=1,
            n_jobs=1,
            verbosity=0,
            use_label_encoder=False,
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        print(f"  seed={seed:3d}  Test Acc={acc:.4f}  F1={f1:.4f}")

    print(
        f"\n3-seed mean ± std:  Acc = {statistics.mean(accs):.4f} ± "
        f"{statistics.stdev(accs):.4f}  |  F1 = "
        f"{statistics.mean(f1s):.4f} ± {statistics.stdev(f1s):.4f}"
    )


if __name__ == "__main__":
    main()
