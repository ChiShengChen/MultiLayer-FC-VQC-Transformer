#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results.

Scans outputs/ for ALL runs, extracts the 4 key quantum models
(FC-VQC, ResNet-VQC, QT, FQT), groups by (dataset, model),
keeps the latest run per (dataset, model, seed), computes mean ± std.

Generates:
  1. Summary CSV with mean/std per model per dataset
  2. LaTeX table (paper/tables_multiseed.tex)
  3. Training curve comparison plots (paper/figures/training_curves_comparison/)

Usage:
    python aggregate_multiseed.py [--out-dir paper]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# Key model name patterns to extract
KEY_MODEL_PATTERNS = [
    (re.compile(r"^FullyConnectedVQCs_\d+t\d+t\d+_L\d+_K\d+$"), "FC-VQC"),
    (re.compile(r"^ResNetVQC(_\d+t\d+t\d+)?_L\d+_K\d+$"), "ResNet-VQC"),
    (re.compile(r"^QuantumTransformerVQC_L\d+_K\d+$"), "QT"),
    (re.compile(r"^FullQuantumTransformerVQC_L\d+_K\d+$"), "FQT"),
]

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


def classify_model(model_name):
    """Return short name if model_name matches a key model, else None."""
    for pattern, short in KEY_MODEL_PATTERNS:
        if pattern.match(model_name):
            return short
    return None


def collect_results():
    """Collect results for key models across all runs."""
    # raw[dataset][short_model][(seed, timestamp)] = {metrics, history, ...}
    raw = defaultdict(lambda: defaultdict(dict))

    for run_dir in sorted(OUTPUTS.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        metrics_csv = run_dir / "comparison_metrics.csv"
        if not cfg_path.exists() or not metrics_csv.exists():
            continue

        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue

        dataset = cfg.get("experiment", "")
        if dataset not in DATASET_TASK:
            continue

        seed = cfg.get("seed", 42)
        # Extract timestamp from dir name for "latest" resolution
        dirname = run_dir.name
        ts_match = re.match(r"^(\d{8}_\d{6})", dirname)
        timestamp = ts_match.group(1) if ts_match else dirname

        df = pd.read_csv(metrics_csv)
        for _, row in df.iterrows():
            model_name = row["model"]
            short = classify_model(model_name)
            if short is None:
                continue

            task = DATASET_TASK[dataset]
            entry = {
                "seed": seed,
                "timestamp": timestamp,
                "run_dir": str(run_dir),
                "model_full_name": model_name,
                "n_params": row.get("n_params", ""),
            }

            if task == "regression":
                entry["test_R2"] = row.get("test_R2", np.nan)
                entry["test_RMSE"] = row.get("test_RMSE", np.nan)
            else:
                entry["test_Accuracy"] = row.get("test_Accuracy", np.nan)
                entry["test_F1_macro"] = row.get("test_F1_macro", np.nan)

            # Load training history
            hist_path = run_dir / model_name / "history.csv"
            if hist_path.exists():
                entry["history"] = pd.read_csv(hist_path)

            key = (seed, timestamp)
            # Keep latest timestamp per (dataset, model, seed)
            existing = raw[dataset][short].get(seed)
            if existing is None or timestamp > existing["timestamp"]:
                raw[dataset][short][seed] = entry

    # Flatten: results[dataset][short_model] = list of entries (one per seed)
    results = defaultdict(lambda: defaultdict(list))
    histories = defaultdict(lambda: defaultdict(list))
    for dataset in raw:
        for short_model in raw[dataset]:
            for seed, entry in sorted(raw[dataset][short_model].items()):
                results[dataset][short_model].append(entry)
                if "history" in entry:
                    histories[dataset][short_model].append({
                        "seed": entry["seed"],
                        "history": entry["history"],
                    })

    return dict(results), dict(histories)


def make_summary_table(results, out_dir):
    """Generate summary CSV and LaTeX table with mean ± std."""
    rows = []
    for dataset in results:
        for short_model, entries in results[dataset].items():
            short_ds = DATASET_SHORT.get(dataset, dataset)
            task = DATASET_TASK[dataset]

            if task == "regression":
                vals = [e["test_R2"] for e in entries
                        if not np.isnan(e.get("test_R2", np.nan))]
                metric_name = "R2"
            else:
                vals = [e["test_Accuracy"] for e in entries
                        if not np.isnan(e.get("test_Accuracy", np.nan))]
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
                "seeds": [e["seed"] for e in entries],
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results" / "multiseed_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["seeds"], errors="ignore").to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    _generate_latex(df, out_dir)
    return df


def _generate_latex(df, out_dir):
    """Generate LaTeX table with mean ± std."""
    tex_path = out_dir / "tables_multiseed.tex"
    model_order = ["FC-VQC", "ResNet-VQC", "QT", "FQT"]

    lines = []
    lines.append("% Auto-generated multi-seed results (mean ± std)")
    lines.append("")

    for metric_name, task_label, caption_metric in [
        ("R2", "Regression", "R^2"),
        ("Acc", "Classification", "\\text{Accuracy}"),
    ]:
        sub = df[df["metric"] == metric_name]
        if sub.empty:
            continue

        datasets = list(dict.fromkeys(sub["dataset"]))  # preserve order

        lines.append(f"% {task_label}: Test ${caption_metric}$ (mean ± std)")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        n_col = "n" if metric_name == "R2" else "n"
        lines.append(f"\\caption{{Test ${caption_metric}$ across multiple random seeds "
                      f"(mean $\\pm$ std). Best mean per column in \\textbf{{bold}}.}}")
        lines.append(f"\\label{{tab:multiseed-{task_label.lower()}}}")
        lines.append("\\small")

        cols = "l" + "r" * len(datasets) + "r"
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header = "Model & " + " & ".join(datasets) + " & \\#params \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Find best mean per dataset for bolding
        best = {}
        for ds in datasets:
            ds_sub = sub[sub["dataset"] == ds]
            if not ds_sub.empty:
                best[ds] = ds_sub["mean"].max()

        for model in model_order:
            cells = [model]
            for ds in datasets:
                row = sub[(sub["dataset"] == ds) & (sub["model"] == model)]
                if row.empty:
                    cells.append("--")
                else:
                    r = row.iloc[0]
                    val_str = f"${r['mean']:.3f} \\pm {r['std']:.3f}$"
                    if abs(r['mean'] - best.get(ds, -999)) < 1e-6:
                        val_str = f"$\\mathbf{{{r['mean']:.3f}}} \\pm {r['std']:.3f}$"
                    n_s = int(r['n_seeds'])
                    if n_s < 3:
                        val_str += f"$^{{{n_s}}}$"
                    cells.append(val_str)
            p_row = sub[sub["model"] == model]
            try:
                params = str(int(float(p_row.iloc[0]["n_params"])))
            except (ValueError, IndexError):
                params = "--"
            cells.append(params)
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table: {tex_path}")


def plot_training_curves(histories, out_dir):
    """Generate training curve comparison plots."""
    fig_dir = out_dir / "figures" / "training_curves_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_order = ["FC-VQC", "ResNet-VQC", "QT", "FQT"]
    colors = {"FC-VQC": "#1f77b4", "ResNet-VQC": "#2ca02c",
              "QT": "#ff7f0e", "FQT": "#d62728"}

    for dataset in histories:
        short_ds = DATASET_SHORT.get(dataset, dataset)
        task = DATASET_TASK[dataset]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for short_model in model_order:
            if short_model not in histories[dataset]:
                continue
            entries = histories[dataset][short_model]
            color = colors.get(short_model, None)

            all_train, all_val = [], []
            for e in entries:
                h = e["history"]
                if task == "regression":
                    tc = "train_mse" if "train_mse" in h.columns else "train_loss"
                    vc = "val_mse" if "val_mse" in h.columns else "val_loss"
                else:
                    tc = "train_loss" if "train_loss" in h.columns else "train_mse"
                    vc = "val_loss" if "val_loss" in h.columns else "val_mse"
                if tc in h.columns:
                    all_train.append(h[tc].values)
                if vc in h.columns:
                    all_val.append(h[vc].values)

            for ax_idx, (data_list, label) in enumerate([
                (all_train, "Train"), (all_val, "Val")
            ]):
                if not data_list:
                    continue
                min_len = min(len(d) for d in data_list)
                arr = np.array([d[:min_len] for d in data_list])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                epochs = np.arange(1, min_len + 1)

                axes[ax_idx].plot(epochs, mean, label=short_model, color=color)
                if len(data_list) > 1:
                    axes[ax_idx].fill_between(
                        epochs, mean - std, mean + std, alpha=0.15, color=color)

        loss_label = "MSE" if task == "regression" else "Loss"
        for ax_idx, split in enumerate(["Training", "Validation"]):
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel(f"{split} {loss_label}")
            axes[ax_idx].set_title(f"{short_ds} — {split} Loss")
            axes[ax_idx].legend()
            axes[ax_idx].set_yscale("log")

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

    print("Scanning all runs for key models (FC-VQC, ResNet-VQC, QT, FQT)...")
    results, histories = collect_results()

    if not results:
        print("No results found in outputs/.")
        return

    total = sum(len(v) for d in results.values() for v in d.values())
    seeds_per = {ds: {m: len(v) for m, v in results[ds].items()} for ds in results}
    print(f"Found {total} model results across {len(results)} datasets.")
    for ds in sorted(seeds_per):
        parts = [f"{m}={n}" for m, n in sorted(seeds_per[ds].items())]
        print(f"  {DATASET_SHORT.get(ds, ds)}: {', '.join(parts)}")
    print()

    print("Generating summary table...")
    df = make_summary_table(results, out_dir)
    print()
    print(df.drop(columns=["seeds"], errors="ignore").to_string(index=False))
    print()

    print("Generating training curve plots...")
    plot_training_curves(histories, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
