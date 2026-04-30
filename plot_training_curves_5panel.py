#!/usr/bin/env python3
"""
5-panel training-curve figure for the paper's main results section.

Panels (a)-(d): the four main architectures on Boston Housing (3-seed mean ± std
                of train and val MSE per epoch).
Panel (e):     ansatz comparison (StronglyEntanglingLayers vs BasicEntanglerLayers
                on ResNet-VQC, Boston, 3-seed mean ± std).

Output: QCE26_Q_FC_Transformer/figures/boston_training_curves.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs"
OUT_PNG = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/QCE26_Q_FC_Transformer/figures/boston_training_curves.png"

# 3-seed Boston main multiseed runs (4 models)
MAIN_SEEDS = {
    42:  "20260224_183942_BostonHousing",
    123: "20260414_232059_BostonHousing_seed123",
    7:   "20260415_033734_BostonHousing_seed7",
}
MAIN_MODELS = [
    ("FC-VQC",     "FullyConnectedVQCs_15t5t1_L3_K3", "#1f77b4"),
    ("ResNet-VQC", "ResNetVQC_15t5t1_L3_K3",          "#2ca02c"),
    ("QT",         "QuantumTransformerVQC_L2_K3",     "#ff7f0e"),
    ("FQT",        "FullQuantumTransformerVQC_L2_K3", "#d62728"),
]

# 3-seed ansatz comparison runs
ANSATZ_SEEDS = {
    42:  "20260430_014607_BostonHousing_seed42",
    123: "20260430_021530_BostonHousing_seed123",
    7:   "20260430_024432_BostonHousing_seed7",
}
ANSATZ_MODELS = [
    ("Strong (720p)", "ResNetVQC_L3_K3",              "#1f77b4"),
    ("Basic (240p)",  "ResNetVQC_L3_K3_ansatzbasic",  "#ff7f0e"),
]

MAX_EPOCH = 5000  # truncate to common range

def load_curves(run_dirs, model_dir):
    """Return arrays (epochs, train_mean, train_std, val_mean, val_std). None if any seed missing."""
    rows = []
    min_len = None
    for rdir in run_dirs.values():
        p = os.path.join(OUTPUTS, rdir, model_dir, "history.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        # Keep only up to MAX_EPOCH
        df = df[df["epoch"] <= MAX_EPOCH]
        rows.append(df)
        min_len = len(df) if min_len is None else min(min_len, len(df))
    rows = [r.iloc[:min_len] for r in rows]
    epochs = rows[0]["epoch"].values
    train = np.array([r["train_mse"].values for r in rows])
    val   = np.array([r["val_mse"].values   for r in rows])
    return epochs, train.mean(0), train.std(0, ddof=1), val.mean(0), val.std(0, ddof=1)

# ─── Build figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 3.6), sharey=False)

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

# Panels (a)-(d): main models
for i, (label, mdir, color) in enumerate(MAIN_MODELS):
    ax = axes[i]
    res = load_curves(MAIN_SEEDS, mdir)
    if res is None:
        ax.set_title(f"{panel_labels[i]} {label} — missing")
        continue
    ep, tr_m, tr_s, va_m, va_s = res
    ax.plot(ep, tr_m, color=color, lw=1.4, label="train")
    ax.fill_between(ep, tr_m - tr_s, tr_m + tr_s, color=color, alpha=0.18)
    ax.plot(ep, va_m, color=color, lw=1.4, ls="--", label="val")
    ax.fill_between(ep, va_m - va_s, va_m + va_s, color=color, alpha=0.10)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    if i == 0:
        ax.set_ylabel("MSE (log, normalised)")
    ax.set_title(f"{panel_labels[i]} {label}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

# Panel (e): ansatz comparison (overlay both ansatz on the same panel)
ax = axes[4]
for label, mdir, color in ANSATZ_MODELS:
    res = load_curves(ANSATZ_SEEDS, mdir)
    if res is None:
        continue
    ep, tr_m, tr_s, va_m, va_s = res
    ax.plot(ep, tr_m, color=color, lw=1.4, label=f"{label} train")
    ax.fill_between(ep, tr_m - tr_s, tr_m + tr_s, color=color, alpha=0.18)
    ax.plot(ep, va_m, color=color, lw=1.4, ls="--", label=f"{label} val")
    ax.fill_between(ep, va_m - va_s, va_m + va_s, color=color, alpha=0.10)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_title(f"{panel_labels[4]} Ansatz: Strong vs Basic")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="upper right", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
