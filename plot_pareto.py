#!/usr/bin/env python3
"""
Parameter-efficiency Pareto plot: 3-seed mean test metric vs parameter count.
Includes 3 regression datasets, MNIST 4 vs 9 binary classification, and the
ansatz-comparison points (Strong/Basic on ResNet-VQC, Boston).

Output: QCE26_Q_FC_Transformer/figures/pareto_r2_vs_params.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/QCE26_Q_FC_Transformer/figures/pareto_r2_vs_params.png"

# ─── Hard-coded 3-seed mean values from main.tex tables ────────────────────
# (mean_metric, n_params, model_label, dataset, family)
# family: 'tree' | 'kernel' | 'mlp' | 'vqc' | 'ansatz_basic'

REG = [
    # Boston
    (.862,  35_000, "CatBoost",   "Boston", "tree"),
    (.845,  95_000, "XGBoost",    "Boston", "tree"),
    (.576,    300,  "KRR",        "Boston", "kernel"),
    (.547,   3500,  "SVR",        "Boston", "kernel"),
    (.696,    218,  "MLP-PM",     "Boston", "mlp"),
    (.753,    721,  "MLP-720",    "Boston", "mlp"),
    (.829,    720,  "FC-VQC",     "Boston", "vqc"),
    (.775,    720,  "ResNet-VQC", "Boston", "vqc"),
    (.742,   1380,  "QT",         "Boston", "vqc"),
    (.705,    855,  "FQT",        "Boston", "vqc"),
    # CA Housing
    (.854,  35_000, "CatBoost",   "CA",     "tree"),
    (.850,  95_000, "XGBoost",    "CA",     "tree"),
    (.773,  14_000, "KRR",        "CA",     "kernel"),
    (.777,   3500,  "SVR",        "CA",     "kernel"),
    (.779,    218,  "MLP-PM",     "CA",     "mlp"),
    (.800,    721,  "MLP-720",    "CA",     "mlp"),
    (.750,    486,  "FC-VQC",     "CA",     "vqc"),
    (.783,    486,  "ResNet-VQC", "CA",     "vqc"),
    (.807,    828,  "QT",         "CA",     "vqc"),
    (.794,    513,  "FQT",        "CA",     "vqc"),
    # Concrete
    (.931,  35_000, "CatBoost",   "Concrete", "tree"),
    (.914,  95_000, "XGBoost",    "Concrete", "tree"),
    (.773,    700,  "KRR",        "Concrete", "kernel"),
    (.621,   3500,  "SVR",        "Concrete", "kernel"),
    (.868,    218,  "MLP-PM",     "Concrete", "mlp"),
    (.867,    721,  "MLP-720",    "Concrete", "mlp"),
    (.774,    486,  "FC-VQC",     "Concrete", "vqc"),
    (.819,    486,  "ResNet-VQC", "Concrete", "vqc"),
    (.853,    828,  "QT",         "Concrete", "vqc"),
    (.780,    513,  "FQT",        "Concrete", "vqc"),
]

MNIST = [
    # 3-seed mean Acc, n_params, label
    (.938, 612, "FC-VQC"),
    (.936, 550, "ResNet-VQC"),
    (.942, 1078, "QT"),
    (.941, 658, "FQT"),
]

ANSATZ = [
    # 3-seed mean R2, n_params, label  (Boston, ResNet-VQC variants)
    (.776, 720, "Strong"),
    (.596, 240, "Basic"),
]

COLORS = {
    "tree":   "#d62728",
    "kernel": "#9467bd",
    "mlp":    "#2ca02c",
    "vqc":    "#1f77b4",
}
MARKERS = {
    "tree":   "D",
    "kernel": "s",
    "mlp":    "P",
    "vqc":    "o",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# ── (a) Regression Pareto: 3 datasets overlaid (color = family) ─────────────
ax = axes[0]
for metric, p, lbl, ds, fam in REG:
    ax.scatter(p, metric, c=COLORS[fam], marker=MARKERS[fam],
               s=70, alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_xscale("log")
ax.set_xlabel("Parameter count (log)")
ax.set_ylabel(r"Test $R^2$")
ax.set_title("(a) Regression Pareto (3 datasets pooled)")
ax.grid(True, alpha=0.3, which="both")
# legend by family
import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([], [], marker=MARKERS[f], color="w",
                  markerfacecolor=COLORS[f], markeredgecolor="black",
                  markersize=8, label=name)
    for f, name in [("tree", "Trees"), ("kernel", "Kernel"),
                    ("mlp", "MLP"), ("vqc", "VQC")]
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

# ── (b) MNIST 4v9 panel ────────────────────────────────────────────────────
ax = axes[1]
labels, xs, ys = zip(*[(l, p, m) for m, p, l in MNIST])
ax.scatter(xs, ys, c=COLORS["vqc"], marker="o", s=110,
           edgecolor="black", linewidth=0.5)
for x, y, l in zip(xs, ys, labels):
    ax.annotate(l, (x, y), xytext=(6, 4), textcoords="offset points",
                fontsize=9)
ax.set_xscale("log")
ax.set_xlim(400, 2000)
ax.set_ylim(0.92, 0.95)
ax.set_xlabel("Parameter count (log)")
ax.set_ylabel("Test accuracy")
ax.set_title("(b) MNIST 4 vs 9")
ax.grid(True, alpha=0.3, which="both")

# ── (c) Ansatz comparison (Boston, ResNet-VQC) ─────────────────────────────
ax = axes[2]
labels, xs, ys = zip(*[(l, p, m) for m, p, l in ANSATZ])
colors = ["#1f77b4", "#ff7f0e"]
for x, y, l, c in zip(xs, ys, labels, colors):
    ax.scatter(x, y, c=c, marker="o", s=140,
               edgecolor="black", linewidth=0.5, label=l)
for x, y, l in zip(xs, ys, labels):
    ax.annotate(l, (x, y), xytext=(8, -3), textcoords="offset points",
                fontsize=10, fontweight="bold")
# Connector line + delta annotation
ax.plot(xs, ys, "k--", alpha=0.4, lw=0.8)
mid_x = np.exp(np.mean(np.log(xs)))
mid_y = np.mean(ys)
ax.annotate(rf"$\Delta R^2 = {ys[0]-ys[1]:+.2f}$",
            xy=(mid_x, mid_y), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, color="gray")
ax.set_xscale("log")
ax.set_xlim(150, 1100)
ax.set_ylim(0.55, 0.82)
ax.set_xlabel("Parameter count (log)")
ax.set_ylabel(r"Test $R^2$")
ax.set_title("(c) Ansatz: Strong vs Basic (ResNet-VQC, Boston)")
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
