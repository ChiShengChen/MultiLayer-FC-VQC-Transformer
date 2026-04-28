#!/usr/bin/env python3
"""
Prepare MNIST 4 vs 9 binary classification dataset.
Downloads MNIST via torchvision, filters digits {4, 9}, applies PCA to
reduce 784 pixels -> 12 features (so the 3-qubit tokenization gives 4
tokens, comparable to other tabular benchmarks), and writes a CSV.

Output: data/mnist_4v9_pca12.csv with columns f0..f11, label (0=four, 1=nine)
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_CSV = os.path.join(OUT_DIR, "mnist_4v9_pca11.csv")
N_PCA   = 11  # +1 zero-pad in FullyConnectedVQCs_12t8t6 -> 12 (4 tokens × 3 qubits)
N_PER_CLASS_TRAIN = 1500   # subsample to keep training time tractable
N_PER_CLASS_TEST  = 500
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading MNIST via sklearn fetch_openml (cached locally)...")
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
X_full = mnist.data.astype(np.float32) / 255.0
y_full = mnist.target.astype(np.int64)

# OpenML's MNIST puts the standard 60K/10K split in the first 60K rows
X_tr_full, X_te_full = X_full[:60000], X_full[60000:]
y_tr_full, y_te_full = y_full[:60000], y_full[60000:]

def filter_subsample(X, y, classes=(4, 9), n_per_class=1500, seed=42):
    rng = np.random.default_rng(seed)
    idxs = []
    for c in classes:
        ci = np.where(y == c)[0]
        ci = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
        idxs.append(ci)
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    Xs = X[idx]
    ys = (y[idx] == classes[1]).astype(np.int64)  # 4 -> 0, 9 -> 1
    return Xs, ys

X_tr, y_tr = filter_subsample(X_tr_full, y_tr_full, (4, 9), N_PER_CLASS_TRAIN, SEED)
X_te, y_te = filter_subsample(X_te_full, y_te_full, (4, 9), N_PER_CLASS_TEST,  SEED + 1)
print(f"  train: {X_tr.shape}, class balance: {np.bincount(y_tr)}")
print(f"  test : {X_te.shape}, class balance: {np.bincount(y_te)}")

print(f"Standardizing + PCA -> {N_PCA} components (fit on train)...")
scaler = StandardScaler()
X_tr_std = scaler.fit_transform(X_tr)
X_te_std = scaler.transform(X_te)

pca = PCA(n_components=N_PCA, random_state=SEED, svd_solver="full")
X_tr_pc = pca.fit_transform(X_tr_std)
X_te_pc = pca.transform(X_te_std)
print(f"  explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Combine train+test into a single CSV; the train.py pipeline will re-split.
X_all = np.vstack([X_tr_pc, X_te_pc])
y_all = np.concatenate([y_tr, y_te])

cols = [f"f{i}" for i in range(N_PCA)]
df = pd.DataFrame(X_all, columns=cols)
df["label"] = y_all

df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}: {df.shape}")
print(f"Class balance overall: {df['label'].value_counts().to_dict()}")
