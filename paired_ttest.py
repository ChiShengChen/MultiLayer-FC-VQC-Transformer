#!/usr/bin/env python3
"""
Paired t-test (and Wilcoxon signed-rank) for the four main VQC architectures
across 5 datasets × 3 seeds. Pairs are matched by (dataset, seed).

Outputs:
  - paired_ttest_results.csv   per-pair statistics
  - paired_ttest_table.tex     LaTeX-ready snippet
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

OUTPUTS = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs"

# Map (dataset, seed) -> output dir, picked from the canonical multiseed runs
RUNS = {
    ("BostonHousing", 42):  "20260224_183942_BostonHousing",
    ("BostonHousing", 123): "20260414_232059_BostonHousing_seed123",
    ("BostonHousing", 7):   "20260415_033734_BostonHousing_seed7",
    ("CA_Housing", 42):     "20260225_044701_CA_Housing",
    ("CA_Housing", 123):    "20260415_092037_CA_Housing_seed123",
    ("CA_Housing", 7):      "20260417_031213_CA_Housing_seed7",
    ("Concrete", 42):       "20260225_133933_Concrete",
    ("Concrete", 123):      "20260418_103126_Concrete_seed123",
    ("Concrete", 7):        "20260418_133112_Concrete_seed7",
    ("WineQuality_Red", 42):  "20260225_144947_WineQuality_Red",
    ("WineQuality_Red", 123): "20260418_171526_WineQuality_Red_seed123",
    ("WineQuality_Red", 7):   "20260419_002735_WineQuality_Red_seed7",
    ("WineQuality_RedandWhite", 42):  "20260226_030631_WineQuality_RedandWhite",
    ("WineQuality_RedandWhite", 123): "20260419_064550_WineQuality_RedandWhite_seed123",
    ("WineQuality_RedandWhite", 7):   "20260420_025716_WineQuality_RedandWhite_seed7",
}

# Architecture name in the table -> regex matching the model column
MODEL_PATTERNS = {
    "FC-VQC":     r"^FullyConnectedVQCs.*_L\d+_K3$",
    "ResNet-VQC": r"^ResNetVQC.*_L\d+_K3$",
    "QT":         r"^QuantumTransformerVQC_L\d+_K3$",
    "FQT":        r"^FullQuantumTransformerVQC_L\d+_K3$",
}

REG_DATASETS = {"BostonHousing", "CA_Housing", "Concrete"}

def metric_key(dataset):
    return "test_R2" if dataset in REG_DATASETS else "test_Accuracy"

def load_value(dataset, seed, arch_pattern):
    rdir = RUNS.get((dataset, seed))
    if rdir is None:
        return None
    p = os.path.join(OUTPUTS, rdir, "comparison_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    mask = df["model"].astype(str).str.match(arch_pattern)
    if mask.sum() == 0:
        return None
    val = df.loc[mask, metric_key(dataset)].iloc[0]
    return float(val)

# ─── Build matrix: rows = (dataset, seed), cols = architecture ────────────
rows = []
for (ds, sd), _ in RUNS.items():
    rec = {"dataset": ds, "seed": sd}
    for arch, pat in MODEL_PATTERNS.items():
        rec[arch] = load_value(ds, sd, pat)
    rows.append(rec)
df = pd.DataFrame(rows)
print("Per-(dataset,seed) test metric matrix:")
print(df.to_string(index=False))

# ─── Pairwise paired t-tests ──────────────────────────────────────────────
ARCHS = list(MODEL_PATTERNS.keys())
PAIRS = [
    ("FC-VQC", "QT"),       # central efficiency claim
    ("FC-VQC", "FQT"),
    ("FC-VQC", "ResNet-VQC"),
    ("ResNet-VQC", "QT"),
    ("FQT", "QT"),
]

records = []
print("\nPaired tests (across all 15 (dataset,seed) pairs):")
print(f"{'A vs B':<22s} {'mean Δ':>9s} {'std Δ':>9s} {'t':>7s} {'p':>9s} {'W':>7s} {'p_W':>9s}")
for A, B in PAIRS:
    pair_df = df[[A, B, "dataset", "seed"]].dropna()
    a = pair_df[A].values
    b = pair_df[B].values
    diff = a - b
    n = len(diff)
    if n < 2:
        continue
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
    except ValueError:
        w_stat, w_p = (np.nan, np.nan)
    print(f"{A:>10s} vs {B:<8s} {diff.mean():>+9.4f} {diff.std(ddof=1):>9.4f} "
          f"{t_stat:>7.3f} {t_p:>9.4f} {w_stat:>7.1f} {w_p:>9.4f}")
    records.append({
        "A": A, "B": B, "n": n,
        "mean_diff": diff.mean(),
        "std_diff": diff.std(ddof=1),
        "t_stat": t_stat, "t_p": t_p,
        "w_stat": w_stat, "w_p": w_p,
    })

# Per-dataset paired t-test for ALL pairs (5 pairs × 5 datasets = 25 tests)
print("\nPer-dataset paired t-tests (3 seeds, df=2):")
per_ds = []
header = f"{'Dataset':<26s} | " + " | ".join(f"{A[:6]+'-'+B[:6]:>14s}" for A, B in PAIRS)
print(header)
print("-" * len(header))
for ds in df["dataset"].unique():
    sub_full = df[df["dataset"] == ds]
    cells = [f"{ds:<26s}"]
    for A, B in PAIRS:
        sub = sub_full[[A, B]].dropna()
        if len(sub) < 2:
            cells.append(f"{'n/a':>14s}")
            per_ds.append({"dataset": ds, "A": A, "B": B,
                           "mean_diff": np.nan, "t_stat": np.nan, "t_p": np.nan})
            continue
        a, b = sub[A].values, sub[B].values
        diff = a - b
        t_stat, t_p = stats.ttest_rel(a, b)
        sig = "*" if t_p < 0.05 else " "
        cells.append(f"{diff.mean():>+6.3f}/p={t_p:>5.3f}{sig}")
        per_ds.append({"dataset": ds, "A": A, "B": B,
                       "mean_diff": diff.mean(),
                       "std_diff": diff.std(ddof=1),
                       "t_stat": t_stat, "t_p": t_p})
    print(" | ".join(cells))

# Save per-dataset CSV
pd.DataFrame(per_ds).to_csv("paired_ttest_per_dataset.csv", index=False)
print("\nWrote paired_ttest_per_dataset.csv")

# LaTeX table for per-dataset results
tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Per-dataset paired $t$-tests (3 seeds, $\mathrm{df}{=}2$). Each cell shows $\bar{\Delta} = \text{metric}_A - \text{metric}_B$ and the corresponding $p$-value. Test metric is regression $R^2$ (Boston/CA/Concrete) or classification accuracy (Wine~R/R+W). Significant entries at $\alpha{=}0.05$ in \textbf{bold}.}")
tex.append(r"\label{tab:ttest-per-dataset}")
tex.append(r"\small")
ncols = 1 + 2 * len(PAIRS)
colspec = "l" + "rr" * len(PAIRS)
tex.append(r"\begin{tabular}{" + colspec + "}")
tex.append(r"\toprule")
header_top = ["Dataset"]
header_bot = [""]
for A, B in PAIRS:
    short_A = A.replace("-VQC", "").replace("FC", "FC")
    short_B = B.replace("-VQC", "").replace("FC", "FC")
    header_top.append(r"\multicolumn{2}{c}{" + f"{short_A} vs.\\ {short_B}" + "}")
    header_bot.extend([r"$\bar{\Delta}$", r"$p$"])
tex.append(" & ".join(header_top) + r" \\")
tex.append(" & ".join([h for h in header_bot]) + r" \\")
tex.append(r"\midrule")
# Build rows
ds_order = list(df["dataset"].unique())
for ds in ds_order:
    row = [ds.replace("WineQuality_", "Wine ").replace("_", " ")]
    sub_full = df[df["dataset"] == ds]
    for A, B in PAIRS:
        sub = sub_full[[A, B]].dropna()
        if len(sub) < 2:
            row.extend(["n/a", "n/a"])
            continue
        a, b = sub[A].values, sub[B].values
        diff = a - b
        t_stat, t_p = stats.ttest_rel(a, b)
        delta = diff.mean()
        if t_p < 0.05:
            row.append(f"$\\mathbf{{{delta:+.3f}}}$")
            row.append(f"$\\mathbf{{{t_p:.3f}}}$")
        else:
            row.append(f"${delta:+.3f}$")
            row.append(f"${t_p:.3f}$")
    tex.append(" & ".join(row) + r" \\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\vspace{-5pt}")
tex.append(r"\end{table*}")
out_tex2 = "paired_ttest_per_dataset_table.tex"
with open(out_tex2, "w") as f:
    f.write("\n".join(tex))
print(f"Wrote {out_tex2}")

# ─── Save outputs ─────────────────────────────────────────────────────────
out_csv = "paired_ttest_results.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

# LaTeX table snippet for the paper
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Paired statistical tests across 5 datasets $\times$ 3 seeds (15 pairs). $\Delta = \text{metric}_A - \text{metric}_B$ averaged over matched (dataset, seed) pairs. Mixed regression $R^2$ and classification accuracy values. Significance is at the conventional $\alpha{=}0.05$ level.}")
tex.append(r"\label{tab:ttest}")
tex.append(r"\small")
tex.append(r"\begin{tabular}{lrrrrr}")
tex.append(r"\toprule")
tex.append(r"Comparison & $\bar{\Delta}$ & $\sigma_{\Delta}$ & $t_{14}$ & $p$ (paired $t$) & $p$ (Wilcoxon) \\")
tex.append(r"\midrule")
for r in records:
    sig_t = r"^{*}" if r["t_p"] < 0.05 else ""
    sig_w = r"^{*}" if r["w_p"] < 0.05 else ""
    tex.append(f"{r['A']} vs {r['B']} & "
               f"${r['mean_diff']:+.3f}$ & ${r['std_diff']:.3f}$ & "
               f"${r['t_stat']:+.2f}$ & "
               f"${r['t_p']:.3f}{sig_t}$ & ${r['w_p']:.3f}{sig_w}$ \\\\")
tex.append(r"\bottomrule")
tex.append(r"\multicolumn{6}{l}{\scriptsize $^{*}$ significant at $\alpha{=}0.05$. "
           r"Mixed metric (regression $R^2$ on Boston/CA/Concrete; classification "
           r"accuracy on Wine R / Wine R+W).}")
tex.append(r"\end{tabular}")
tex.append(r"\vspace{-5pt}")
tex.append(r"\end{table}")
out_tex = "paired_ttest_table.tex"
with open(out_tex, "w") as f:
    f.write("\n".join(tex))
print(f"Wrote {out_tex}")
