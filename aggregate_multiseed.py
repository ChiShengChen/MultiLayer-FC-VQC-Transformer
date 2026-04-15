#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results.

Scans outputs/ for runs produced by run_multiseed.sh, groups by
(dataset, model), computes mean ± std, generates:
  1. Summary CSV with mean/std per model per dataset
  2. LaTeX table (paper/tables_multiseed.tex)
  3. Training curve comparison plots (paper/figures/training_curves_comparison/)

Usage:
    python aggregate_multiseed.py [--out-dir paper]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# Model display names (short)
MODEL_SHORT = {
    "FullyConnectedVQCs_15t5t1_L3_K3": "FC-VQC",
    "FullyConnectedVQCs_8t3t1_L3_K3": "FC-VQC",
    "FullyConnectedVQCs_12t8t6_L3_K3": "FC-VQC",
    "ResNetVQC_15t5t1_L3_K3": "ResNet-VQC",
    "ResNetVQC_L3_K3": "ResNet-VQC",
    "QuantumTransformerVQC_L2_K3": "QT",
    "FullQuantumTransformerVQC_L2_K3": "FQT",
}

DATASET_SHORT = {
    "BostonHousing": "Boston",
    "CA_Housing": "CA Hous.",
    "Concrete": "Concrete",
    "WineQuality_Red": "Wine-R",
    "WineQuality_RedandWhite": "Wine-RW",
}

DATASET_TASK = {
    "BostonHousing": "regression",
    "CA_Housing": "regression",
    "Concrete": "regression",
    "WineQuality_Red": "classification",
    "WineQuality_RedandWhite": "classification",
}


def is_multiseed_run(run_dir):
    """Check if a run was produced by multiseed configs."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
        models = cfg.get("models", [])
        names = {m.get("name", "") for m in models}
        # Multiseed configs have exactly these 4 model types
        key_models = {"QuantumTransformerVQC", "FullQuantumTransformerVQC"}
        has_vqc = any("FullyConnected" in n or "ResNet" in n for n in names)
        has_transformer = bool(key_models & names)
        return has_vqc and has_transformer and len(models) == 4
    except Exception:
        return False


def collect_results():
    """Collect all multi-seed results grouped by (dataset, model)."""
    # Structure: results[dataset][model_name] = list of dicts
    results = defaultdict(lambda: defaultdict(list))
    histories = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(OUTPUTS.iterdir()):
        if not run_dir.is_dir():
            continue
        if not is_multiseed_run(run_dir):
            continue

        cfg = json.loads((run_dir / "config.json").read_text())
        dataset = cfg["experiment"]
        seed = cfg.get("seed", 42)

        metrics_csv = run_dir / "comparison_metrics.csv"
        if not metrics_csv.exists():
            continue

        df = pd.read_csv(metrics_csv)
        for _, row in df.iterrows():
            model_name = row["model"]
            entry = {"seed": seed, "run_dir": str(run_dir)}

            if DATASET_TASK[dataset] == "regression":
                entry["test_R2"] = row.get("test_R2", np.nan)
                entry["test_RMSE"] = row.get("test_RMSE", np.nan)
            else:
                entry["test_Accuracy"] = row.get("test_Accuracy", np.nan)
                entry["test_F1_macro"] = row.get("test_F1_macro", np.nan)

            entry["n_params"] = row.get("n_params", "")
            results[dataset][model_name].append(entry)

            # Load training history if available
            hist_path = run_dir / model_name / "history.csv"
            if hist_path.exists():
                h = pd.read_csv(hist_path)
                histories[dataset][model_name].append({
                    "seed": seed,
                    "history": h,
                })

    return dict(results), dict(histories)


def make_summary_table(results, out_dir):
    """Generate summary CSV and LaTeX table with mean ± std."""
    rows = []
    for dataset in results:
        for model_name, entries in results[dataset].items():
            short_model = MODEL_SHORT.get(model_name, model_name)
            short_ds = DATASET_SHORT.get(dataset, dataset)
            task = DATASET_TASK[dataset]

            if task == "regression":
                vals = [e["test_R2"] for e in entries if not np.isnan(e.get("test_R2", np.nan))]
                metric_name = "R2"
            else:
                vals = [e["test_Accuracy"] for e in entries if not np.isnan(e.get("test_Accuracy", np.nan))]
                metric_name = "Acc"

            if not vals:
                continue

            n_params = entries[0].get("n_params", "")
            rows.append({
                "dataset": short_ds,
                "model": short_model,
                "metric": metric_name,
                "n_seeds": len(vals),
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "n_params": n_params,
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results" / "multiseed_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    # Generate LaTeX
    _generate_latex(df, out_dir)
    return df


def _generate_latex(df, out_dir):
    """Generate LaTeX table with mean ± std."""
    tex_path = out_dir / "tables_multiseed.tex"

    # Regression table
    reg_df = df[df["metric"] == "R2"]
    cls_df = df[df["metric"] == "Acc"]

    lines = []
    lines.append("% Auto-generated multi-seed results (mean ± std)")
    lines.append("")

    if not reg_df.empty:
        lines.append("% Regression: Test R² (mean ± std over 3 seeds)")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Test $R^2$ across 3 random seeds (mean $\\pm$ std). Best mean per column in \\textbf{bold}.}")
        lines.append("\\label{tab:multiseed-regression}")
        lines.append("\\small")

        datasets = reg_df["dataset"].unique()
        models = reg_df["model"].unique()

        cols = "l" + "r" * len(datasets) + "r"
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header = "Model & " + " & ".join(datasets) + " & \\#params \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for model in ["FC-VQC", "ResNet-VQC", "QT", "FQT"]:
            cells = [model]
            for ds in datasets:
                row = reg_df[(reg_df["dataset"] == ds) & (reg_df["model"] == model)]
                if row.empty:
                    cells.append("--")
                else:
                    r = row.iloc[0]
                    cells.append(f"${r['mean']:.3f} \\pm {r['std']:.3f}$")
            # params from first available
            p_row = reg_df[reg_df["model"] == model]
            params = str(int(p_row.iloc[0]["n_params"])) if not p_row.empty and p_row.iloc[0]["n_params"] else "--"
            cells.append(params)
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if not cls_df.empty:
        lines.append("% Classification: Test Accuracy (mean ± std over 3 seeds)")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Test accuracy across 3 random seeds (mean $\\pm$ std).}")
        lines.append("\\label{tab:multiseed-classification}")
        lines.append("\\small")

        datasets = cls_df["dataset"].unique()
        cols = "l" + "r" * len(datasets) + "r"
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header = "Model & " + " & ".join(datasets) + " & \\#params \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for model in ["FC-VQC", "ResNet-VQC", "QT", "FQT"]:
            cells = [model]
            for ds in datasets:
                row = cls_df[(cls_df["dataset"] == ds) & (cls_df["model"] == model)]
                if row.empty:
                    cells.append("--")
                else:
                    r = row.iloc[0]
                    cells.append(f"${r['mean']:.3f} \\pm {r['std']:.3f}$")
            p_row = cls_df[cls_df["model"] == model]
            params = str(int(p_row.iloc[0]["n_params"])) if not p_row.empty and p_row.iloc[0]["n_params"] else "--"
            cells.append(params)
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table: {tex_path}")


def plot_training_curves(histories, out_dir):
    """Generate training curve comparison plots for representative datasets."""
    fig_dir = out_dir / "figures" / "training_curves_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot for each dataset
    for dataset in histories:
        short_ds = DATASET_SHORT.get(dataset, dataset)
        task = DATASET_TASK[dataset]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: training loss, Right: validation loss
        for model_name in sorted(histories[dataset].keys()):
            short_model = MODEL_SHORT.get(model_name, model_name)
            entries = histories[dataset][model_name]

            all_train = []
            all_val = []
            for e in entries:
                h = e["history"]
                if task == "regression":
                    train_col = "train_mse" if "train_mse" in h.columns else "train_loss"
                    val_col = "val_mse" if "val_mse" in h.columns else "val_loss"
                else:
                    train_col = "train_loss" if "train_loss" in h.columns else "train_mse"
                    val_col = "val_loss" if "val_loss" in h.columns else "val_mse"

                if train_col in h.columns:
                    all_train.append(h[train_col].values)
                if val_col in h.columns:
                    all_val.append(h[val_col].values)

            if all_train:
                min_len = min(len(t) for t in all_train)
                arr = np.array([t[:min_len] for t in all_train])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                epochs = np.arange(1, min_len + 1)

                axes[0].plot(epochs, mean, label=short_model)
                axes[0].fill_between(epochs, mean - std, mean + std, alpha=0.15)

            if all_val:
                min_len = min(len(v) for v in all_val)
                arr = np.array([v[:min_len] for v in all_val])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                epochs = np.arange(1, min_len + 1)

                axes[1].plot(epochs, mean, label=short_model)
                axes[1].fill_between(epochs, mean - std, mean + std, alpha=0.15)

        loss_label = "MSE" if task == "regression" else "Loss"
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel(f"Train {loss_label}")
        axes[0].set_title(f"{short_ds} — Training Loss")
        axes[0].legend()
        axes[0].set_yscale("log")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(f"Val {loss_label}")
        axes[1].set_title(f"{short_ds} — Validation Loss")
        axes[1].legend()
        axes[1].set_yscale("log")

        fig.tight_layout()
        fig_path = fig_dir / f"{dataset}_training_curves.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Training curve plot: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="paper",
                        help="Output directory for tables and figures")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir

    print("Collecting multi-seed results...")
    results, histories = collect_results()

    if not results:
        print("No multi-seed results found. Run run_multiseed.sh first.")
        return

    total_runs = sum(len(v) for d in results.values() for v in d.values())
    print(f"Found {total_runs} total model runs across {len(results)} datasets.\n")

    print("Generating summary table...")
    df = make_summary_table(results, out_dir)
    print()
    print(df.to_string(index=False))
    print()

    print("Generating training curve plots...")
    plot_training_curves(histories, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
