#!/usr/bin/env python3
"""
Barren plateau / trainability analysis: gradient variance vs epoch
across the 4 main VQC architectures on Boston Housing (3 seeds).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs"
FIG_OUT = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/QCE26_Q_FC_Transformer/figures/grad_variance.png"

SEEDS = {
    42:  "20260224_183942_BostonHousing",
    123: "20260414_232059_BostonHousing_seed123",
    7:   "20260415_033734_BostonHousing_seed7",
}

MODELS = {
    "FC-VQC":     "FullyConnectedVQCs_15t5t1_L3_K3",
    "ResNet-VQC": "ResNetVQC_15t5t1_L3_K3",
    "QT":         "QuantumTransformerVQC_L2_K3",
    "FQT":        "FullQuantumTransformerVQC_L2_K3",
}

COLORS = {
    "FC-VQC":     "#1f77b4",
    "ResNet-VQC": "#2ca02c",
    "QT":         "#ff7f0e",
    "FQT":        "#d62728",
}

# Smoothing window
WIN = 50

def load_grad(seed_dir, model_dir):
    p = os.path.join(OUTPUTS, seed_dir, model_dir, "history.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df["epoch"].values, df["grad_variance"].values

def smooth(arr, w=WIN):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left panel: smoothed grad variance vs epoch (mean across seeds)
ax = axes[0]
for label, mdir in MODELS.items():
    all_curves = []
    min_len = None
    for seed, sdir in SEEDS.items():
        res = load_grad(sdir, mdir)
        if res is None:
            print(f"[skip] {label} seed={seed}")
            continue
        ep, gv = res
        all_curves.append(gv)
        min_len = len(gv) if min_len is None else min(min_len, len(gv))
    if not all_curves:
        continue
    trimmed = np.array([c[:min_len] for c in all_curves])
    mean = trimmed.mean(axis=0)
    smoothed = smooth(mean)
    epochs = np.arange(WIN - 1, WIN - 1 + len(smoothed)) + 1
    ax.semilogy(epochs, smoothed, label=label, color=COLORS[label], lw=1.6)

ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient variance (smoothed, log)")
ax.set_title("(a) Trainability over training")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

# Right panel: median grad variance over first 500 epochs (bar with seed std)
ax = axes[1]
labels, medians, stds = [], [], []
for label, mdir in MODELS.items():
    per_seed = []
    for seed, sdir in SEEDS.items():
        res = load_grad(sdir, mdir)
        if res is None:
            continue
        ep, gv = res
        cutoff = min(500, len(gv))
        per_seed.append(np.median(gv[:cutoff]))
    if not per_seed:
        continue
    labels.append(label)
    medians.append(np.mean(per_seed))
    stds.append(np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0)

x = np.arange(len(labels))
bars = ax.bar(x, medians, yerr=stds, capsize=4,
              color=[COLORS[l] for l in labels], alpha=0.85, edgecolor="black")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Median grad variance (epochs 1–500, log)")
ax.set_title("(b) Early-training gradient magnitude")
ax.grid(True, axis="y", which="both", alpha=0.3)

# Annotate
for i, (m, s) in enumerate(zip(medians, stds)):
    ax.text(i, m * 1.5, f"{m:.1e}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {FIG_OUT}")

# Print summary stats for the paper
print("\n=== Summary (median grad variance, epochs 1-500) ===")
for l, m, s in zip(labels, medians, stds):
    print(f"  {l:>10s}: {m:.3e}  ± {s:.2e}")
