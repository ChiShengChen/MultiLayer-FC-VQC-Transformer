"""
Summarize experiment results into LaTeX tables for the paper.

Scans outputs/ for the most recent run per (dataset, experiment_type),
parses comparison_metrics.csv files, and generates publication-ready
LaTeX tables organized by:

  1. Main comparison      — quantum vs classical baselines per dataset
  2. Multi-head scaling   — H1 / H2 / H3 ablation
  3. Architecture ablation — no-attention / Type 3 FFN / LayerNorm
  4. Noise robustness     — performance vs depolarizing noise strength

Usage:
    python summarize_results.py [--out tables.tex] [--latest-only]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# ─────────────────────────────────────────────────────────────────────
# Dataset / model classification
# ─────────────────────────────────────────────────────────────────────

REGRESSION_DATASETS = ["BostonHousing", "CA_Housing", "Concrete"]
CLASSIFICATION_DATASETS = ["WineQuality_Red", "WineQuality_RedandWhite"]
ALL_DATASETS = REGRESSION_DATASETS + CLASSIFICATION_DATASETS

DATASET_DISPLAY = {
    "BostonHousing": "Boston",
    "CA_Housing": "CA Housing",
    "Concrete": "Concrete",
    "WineQuality_Red": "Wine (Red)",
    "WineQuality_RedandWhite": "Wine (R+W)",
}

CLASSICAL_MODELS = {
    "SVR_RBF", "KernelRidge_RBF", "Ridge", "LinearRegression",
    "SVC_RBF", "LogisticRegression",
    "XGBRegressor", "CatBoostRegressor", "XGBClassifier", "CatBoostClassifier",
}

MODEL_DISPLAY = {
    "LinearRegression": "Linear",
    "Ridge": "Ridge",
    "KernelRidge_RBF": "KRR (RBF)",
    "SVR_RBF": "SVR (RBF)",
    "SVC_RBF": "SVC (RBF)",
    "LogisticRegression": "LogReg",
    "XGBRegressor": "XGBoost",
    "XGBClassifier": "XGBoost",
    "CatBoostRegressor": "CatBoost",
    "CatBoostClassifier": "CatBoost",
    "MLPRegressor_ParamMatch": "MLP$_{\\text{PM}}$",
    "MLPClassifier_ParamMatch": "MLP$_{\\text{PM}}$",
}


def is_classical(name: str) -> bool:
    base = name.split("_L")[0]  # strip _L2_K3 etc.
    return base in CLASSICAL_MODELS


def is_paramatch_mlp(name: str) -> bool:
    return "ParamMatch" in name


def is_quantum(name: str) -> bool:
    return any(k in name for k in ("VQC", "Transformer", "ResNet"))


def display_name(run_name: str) -> str:
    """Convert raw run name to a paper-friendly label."""
    if run_name in MODEL_DISPLAY:
        return MODEL_DISPLAY[run_name]

    # Strip _L?_K? suffix to look up classical short names
    base = re.sub(r"_L\d+_K\d+.*$", "", run_name)
    if base in MODEL_DISPLAY:
        return MODEL_DISPLAY[base]

    # Quantum models — keep architecture flag suffixes
    name = run_name
    name = name.replace("QuantumTransformerVQC", "QT")
    name = name.replace("FullQuantumTransformerVQC", "FQT")
    name = name.replace("ResNetVQC", "ResNet-VQC")
    name = name.replace("FullyConnectedVQCs", "FC-VQC")
    name = re.sub(r"_L(\d+)_K(\d+)", r" ($L\1$, $K\2$)", name)
    name = name.replace("_H2", ", $H{=}2$")
    name = name.replace("_H3", ", $H{=}3$")
    name = name.replace("_noAttn", ", no-attn")
    name = name.replace("_ffnmultiple", ", Type 3")
    name = name.replace("_LN", ", +LN")
    name = re.sub(r"_noise([\d.]+)", r", $p_d{=}\1$", name)
    return name


# ─────────────────────────────────────────────────────────────────────
# Output discovery
# ─────────────────────────────────────────────────────────────────────

def parse_output_dir(d: Path):
    """Return (timestamp, dataset) from a dir name like '20260410_032628_WineQuality_Red'."""
    m = re.match(r"^(\d{8}_\d{6})_(.+)$", d.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def classify_run(csv_path: Path) -> str:
    """Determine experiment type from the model names in a comparison CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return "unknown"
    names = df["model"].astype(str).tolist()
    has_quantum = any(is_quantum(n) for n in names)
    has_classical = any(is_classical(n.split("_L")[0]) for n in names)
    has_noise = any("noise" in n for n in names)
    has_ablation = any(any(k in n for k in ("noAttn", "ffnmultiple", "_LN")) for n in names)
    has_multihead = any(("_H2" in n or "_H3" in n) for n in names)

    if has_noise:
        return "noise"
    if has_ablation:
        return "ablation"
    if has_multihead and has_quantum:
        return "multihead"
    if has_classical and not has_quantum:
        return "classical"
    if has_quantum and has_classical:
        return "compare"
    if has_quantum:
        return "quantum"
    return "other"


def collect_results():
    """Walk outputs/ and collect ALL runs grouped by dataset.

    Per-model deduplication (latest wins) happens in merge_per_dataset,
    so a model that only appears in an older run is still preserved if
    no newer run includes it.
    """
    by_dataset = defaultdict(list)        # ds → list of (ts, df, dir, exp_type)
    summary_groups = defaultdict(list)    # (ds, exp_type) → list of (ts, dir)
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        ts, ds = parse_output_dir(d)
        if ds is None or ds not in ALL_DATASETS:
            continue
        csv = d / "comparison_metrics.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty or "model" not in df.columns:
            continue
        exp_type = classify_run(csv)
        by_dataset[ds].append((ts, df, d, exp_type))
        summary_groups[(ds, exp_type)].append((ts, d))
    return by_dataset, summary_groups


def merge_per_dataset(by_dataset):
    """Combine all runs for a dataset, keeping the latest row per model.

    Within a single run a model name is unique; across runs we keep the
    most recent occurrence.  Rows whose test metrics are missing or NaN
    are skipped so that a stale run can't shadow a newer valid one.
    """
    merged = {}
    for ds, runs in by_dataset.items():
        frames = []
        for ts, df, d, exp_type in runs:
            df = df.copy()
            df["_ts"] = ts
            df["_exp_type"] = exp_type
            # Drop rows with no usable test metric
            metric_col = ("test_R2" if "test_R2" in df.columns
                          else "test_Accuracy" if "test_Accuracy" in df.columns
                          else None)
            if metric_col is not None:
                df = df[df[metric_col].notna()]
            frames.append(df)
        if not frames:
            continue
        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df.sort_values("_ts").drop_duplicates(
            subset=["model"], keep="last")
        merged[ds] = all_df.reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────
# LaTeX formatting
# ─────────────────────────────────────────────────────────────────────

def fmt(x, prec=3):
    if pd.isna(x) or x == "":
        return "--"
    try:
        return f"{float(x):.{prec}f}"
    except (ValueError, TypeError):
        return "--"


def fmt_int(x):
    if pd.isna(x) or x == "":
        return "--"
    try:
        return f"{int(float(x)):,}"
    except (ValueError, TypeError):
        return "--"


def bold_best(values, lower_is_better=False):
    """Return list with the best value wrapped in \\textbf{}."""
    out = []
    nums = []
    for v in values:
        try:
            nums.append(float(v) if v != "--" else None)
        except (ValueError, TypeError):
            nums.append(None)
    valid = [(i, n) for i, n in enumerate(nums) if n is not None]
    if not valid:
        return values
    best_idx = (min(valid, key=lambda t: t[1]) if lower_is_better
                else max(valid, key=lambda t: t[1]))[0]
    for i, v in enumerate(values):
        out.append(f"\\textbf{{{v}}}" if i == best_idx else v)
    return out


# ─────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────

def build_main_regression_table(merged):
    """Main regression comparison table — one row per model, columns = datasets × (R²)."""
    datasets = REGRESSION_DATASETS
    # Collect all model names that appear in any regression dataset
    model_names = set()
    for ds in datasets:
        if ds in merged:
            for n in merged[ds]["model"]:
                if any(k in n for k in ("Transformer", "ResNet", "FullyConnected")) \
                        or is_classical(n.split("_L")[0]) or is_paramatch_mlp(n):
                    # Skip ablation/noise variants from main table
                    if any(s in n for s in ("noAttn", "ffnmultiple", "_LN", "noise")):
                        continue
                    model_names.add(n)

    # Sort: classical → MLP → quantum
    def sort_key(n):
        base = n.split("_L")[0]
        if base in CLASSICAL_MODELS:
            return (0, n)
        if "ParamMatch" in n:
            return (1, n)
        return (2, n)
    model_names = sorted(model_names, key=sort_key)

    # Build matrix
    rows = []
    r2_cols = {ds: [] for ds in datasets}
    for n in model_names:
        row = {"display": display_name(n)}
        for ds in datasets:
            df = merged.get(ds, pd.DataFrame())
            match = df[df["model"] == n]
            if not match.empty and "test_R2" in match.columns:
                v = fmt(match.iloc[0]["test_R2"])
            else:
                v = "--"
            row[ds] = v
            r2_cols[ds].append(v)
        # Param count (any dataset that has it)
        np_val = "--"
        for ds in datasets:
            df = merged.get(ds, pd.DataFrame())
            match = df[df["model"] == n]
            if not match.empty and "n_params" in match.columns:
                p = match.iloc[0]["n_params"]
                if pd.notna(p) and p != "":
                    np_val = fmt_int(p)
                    break
        row["n_params"] = np_val
        rows.append(row)

    # Bold best per dataset column
    for ds in datasets:
        bolded = bold_best(r2_cols[ds], lower_is_better=False)
        for i, row in enumerate(rows):
            row[ds] = bolded[i]

    # Render
    lines = []
    lines.append("% Main regression comparison (test R²)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Test $R^2$ on regression datasets. "
                 "Best value per column in \\textbf{bold}. "
                 "MLP$_{\\text{PM}}$ is parameter-matched to the median quantum model.}")
    lines.append("\\label{tab:main-regression}")
    lines.append("\\small")
    col_spec = "l" + "r" * (len(datasets) + 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = "Model & " + " & ".join(DATASET_DISPLAY[d] for d in datasets) + " & \\#params \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Group separators
    last_group = -1
    for n, row in zip(model_names, rows):
        base = n.split("_L")[0]
        if base in CLASSICAL_MODELS:
            grp = 0
        elif "ParamMatch" in n:
            grp = 1
        else:
            grp = 2
        if last_group != -1 and grp != last_group:
            lines.append("\\midrule")
        last_group = grp
        cells = [row["display"]] + [row[d] for d in datasets] + [row["n_params"]]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_main_classification_table(merged):
    """Main classification comparison — Acc + F1 columns per dataset."""
    datasets = CLASSIFICATION_DATASETS
    model_names = set()
    for ds in datasets:
        if ds in merged:
            for n in merged[ds]["model"]:
                if any(s in n for s in ("noAttn", "ffnmultiple", "_LN", "noise")):
                    continue
                if (any(k in n for k in ("Transformer", "ResNet", "FullyConnected"))
                        or is_classical(n.split("_L")[0]) or is_paramatch_mlp(n)):
                    model_names.add(n)

    def sort_key(n):
        base = n.split("_L")[0]
        if base in CLASSICAL_MODELS:
            return (0, n)
        if "ParamMatch" in n:
            return (1, n)
        return (2, n)
    model_names = sorted(model_names, key=sort_key)

    # Build value matrix
    rows = []
    cols = {(ds, m): [] for ds in datasets for m in ("acc", "f1")}
    for n in model_names:
        row = {"display": display_name(n)}
        for ds in datasets:
            df = merged.get(ds, pd.DataFrame())
            match = df[df["model"] == n]
            for metric, src in (("acc", "test_Accuracy"), ("f1", "test_F1_macro")):
                v = fmt(match.iloc[0][src]) if (not match.empty and src in match.columns) else "--"
                row[f"{ds}_{metric}"] = v
                cols[(ds, metric)].append(v)
        np_val = "--"
        for ds in datasets:
            df = merged.get(ds, pd.DataFrame())
            match = df[df["model"] == n]
            if not match.empty and "n_params" in match.columns:
                p = match.iloc[0]["n_params"]
                if pd.notna(p) and p != "":
                    np_val = fmt_int(p)
                    break
        row["n_params"] = np_val
        rows.append(row)

    # Bold best per column
    for (ds, m), vals in cols.items():
        bolded = bold_best(vals, lower_is_better=False)
        for i, row in enumerate(rows):
            row[f"{ds}_{m}"] = bolded[i]

    lines = []
    lines.append("% Main classification comparison (test Accuracy / F1)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Test Accuracy and macro-F1 on classification datasets. "
                 "Best value per column in \\textbf{bold}.}")
    lines.append("\\label{tab:main-classification}")
    lines.append("\\small")
    col_spec = "l" + "rr" * len(datasets) + "r"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    grp_header = "Model"
    for d in datasets:
        grp_header += f" & \\multicolumn{{2}}{{c}}{{{DATASET_DISPLAY[d]}}}"
    grp_header += " & \\#params \\\\"
    lines.append(grp_header)
    cmid = ""
    col_idx = 2
    for _ in datasets:
        cmid += f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}"
        col_idx += 2
    lines.append(cmid)
    sub_header = " & " + " & ".join("Acc & F1" for _ in datasets) + " & \\\\"
    lines.append(sub_header)
    lines.append("\\midrule")

    last_group = -1
    for n, row in zip(model_names, rows):
        base = n.split("_L")[0]
        if base in CLASSICAL_MODELS:
            grp = 0
        elif "ParamMatch" in n:
            grp = 1
        else:
            grp = 2
        if last_group != -1 and grp != last_group:
            lines.append("\\midrule")
        last_group = grp
        cells = [row["display"]]
        for ds in datasets:
            cells.append(row[f"{ds}_acc"])
            cells.append(row[f"{ds}_f1"])
        cells.append(row["n_params"])
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_ablation_table(merged):
    """Ablation table — focuses on QT and FQT variants vs default."""
    rows = []
    headers = ["Variant"]
    for ds in ALL_DATASETS:
        headers.append(DATASET_DISPLAY[ds])

    # Variant patterns to detect
    variants = [
        ("QuantumTransformerVQC_L2_K3",                "QT (default, Type 4)"),
        ("QuantumTransformerVQC_L2_K3_noAttn",         "\\quad $-$attention"),
        ("QuantumTransformerVQC_L2_K3_ffnmultiple",    "\\quad Type 3 FFN"),
        ("FullQuantumTransformerVQC_L2_K3",            "FQT (default, Type 4)"),
        ("FullQuantumTransformerVQC_L2_K3_noAttn",     "\\quad $-$attention"),
        ("FullQuantumTransformerVQC_L2_K3_ffnmultiple","\\quad Type 3 FFN"),
        ("FullQuantumTransformerVQC_L2_K3_LN",         "\\quad +LayerNorm"),
    ]

    for run_name, label in variants:
        row = [label]
        for ds in ALL_DATASETS:
            df = merged.get(ds, pd.DataFrame())
            match = df[df["model"] == run_name]
            if match.empty:
                row.append("--")
                continue
            if ds in REGRESSION_DATASETS and "test_R2" in match.columns:
                row.append(fmt(match.iloc[0]["test_R2"]))
            elif "test_Accuracy" in match.columns:
                row.append(fmt(match.iloc[0]["test_Accuracy"]))
            else:
                row.append("--")
        rows.append(row)

    lines = []
    lines.append("% Ablation study")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Architecture ablation. Regression cells show test $R^2$, "
                 "classification cells show test accuracy. "
                 "$-$attention disables self-attention; "
                 "Type 3 FFN replaces fully-connected (Type 4) inter-token mixing with "
                 "circular-shift connectivity; "
                 "+LayerNorm adds LayerNorm to FQT (which is unnormalized by default).}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\small")
    col_spec = "l" + "r" * len(ALL_DATASETS)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    sub = " & " + " & ".join("$R^2$" if d in REGRESSION_DATASETS else "Acc"
                              for d in ALL_DATASETS) + " \\\\"
    lines.append(sub)
    lines.append("\\midrule")
    for i, row in enumerate(rows):
        if i == 3:  # break between QT and FQT groups
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_noise_table(merged):
    """Noise robustness table — performance vs depolarizing strength."""
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05]

    def find_for_level(df, base, p):
        if p == 0.0:
            target = base
        else:
            target = f"{base}_noise{p}"
        match = df[df["model"] == target]
        if match.empty:
            return "--"
        if "test_R2" in match.columns and not pd.isna(match.iloc[0]["test_R2"]):
            return fmt(match.iloc[0]["test_R2"])
        if "test_Accuracy" in match.columns:
            return fmt(match.iloc[0]["test_Accuracy"])
        return "--"

    rows = []
    for base, label in (("QuantumTransformerVQC_L2_K3", "QT"),
                        ("FullQuantumTransformerVQC_L2_K3", "FQT")):
        for ds in ALL_DATASETS:
            df = merged.get(ds, pd.DataFrame())
            row = [label, DATASET_DISPLAY[ds]]
            row += [find_for_level(df, base, p) for p in noise_levels]
            rows.append(row)

    lines = []
    lines.append("% Noise robustness")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Robustness to single-qubit depolarizing noise applied after "
                 "every layer of \\texttt{StronglyEntanglingLayers}. "
                 "Cells show test $R^2$ (regression) or test accuracy (classification).}")
    lines.append("\\label{tab:noise}")
    lines.append("\\small")
    col_spec = "ll" + "r" * len(noise_levels)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    head = "Model & Dataset & " + " & ".join(
        f"$p_d{{=}}{p}$" if p > 0 else "noiseless" for p in noise_levels) + " \\\\"
    lines.append(head)
    lines.append("\\midrule")
    last_model = None
    for row in rows:
        if last_model is not None and row[0] != last_model:
            lines.append("\\midrule")
        last_model = row[0]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_multihead_table(merged):
    """Multi-head scaling — H1 / H2 / H3 across datasets."""
    rows = []
    for base, label in (("QuantumTransformerVQC_L2_K3", "QT"),
                        ("FullQuantumTransformerVQC_L2_K3", "FQT")):
        for n_heads, suffix in ((1, ""), (2, "_H2"), (3, "_H3")):
            row = [label, f"$H{{=}}{n_heads}$"]
            np_val = "--"
            for ds in ALL_DATASETS:
                df = merged.get(ds, pd.DataFrame())
                match = df[df["model"] == base + suffix]
                if match.empty:
                    row.append("--")
                    continue
                if ds in REGRESSION_DATASETS and "test_R2" in match.columns:
                    row.append(fmt(match.iloc[0]["test_R2"]))
                elif "test_Accuracy" in match.columns:
                    row.append(fmt(match.iloc[0]["test_Accuracy"]))
                else:
                    row.append("--")
                if np_val == "--" and "n_params" in match.columns:
                    p = match.iloc[0]["n_params"]
                    if pd.notna(p) and p != "":
                        np_val = fmt_int(p)
            row.append(np_val)
            rows.append(row)

    lines = []
    lines.append("% Multi-head scaling")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Effect of varying the number of attention heads $H$. "
                 "Cells show test $R^2$ (regression) / test accuracy (classification).}")
    lines.append("\\label{tab:multihead}")
    lines.append("\\small")
    col_spec = "ll" + "r" * len(ALL_DATASETS) + "r"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    head = "Model & Heads & " + " & ".join(DATASET_DISPLAY[d] for d in ALL_DATASETS) + " & \\#params \\\\"
    lines.append(head)
    lines.append("\\midrule")
    last_model = None
    for row in rows:
        if last_model is not None and row[0] != last_model:
            lines.append("\\midrule")
        last_model = row[0]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="tables.tex",
                        help="Output .tex file (default: tables.tex)")
    parser.add_argument("--print", action="store_true",
                        help="Also print tables to stdout")
    args = parser.parse_args()

    by_dataset, summary = collect_results()
    if not by_dataset:
        print("No results found in outputs/. Run experiments first.")
        return

    print(f"Discovered runs (latest per (dataset, experiment_type) shown):")
    for (ds, exp), runs in sorted(summary.items()):
        latest_ts, latest_d = max(runs)
        print(f"  {ds:25s} {exp:10s} {latest_ts}  {latest_d.name}  "
              f"({len(runs)} run(s))")
    print()

    merged = merge_per_dataset(by_dataset)

    parts = []
    parts.append("% =================================================================")
    parts.append("% Auto-generated by summarize_results.py")
    parts.append("% =================================================================")
    parts.append("")
    parts.append(build_main_regression_table(merged))
    parts.append("")
    parts.append(build_main_classification_table(merged))
    parts.append("")
    parts.append(build_multihead_table(merged))
    parts.append("")
    parts.append(build_ablation_table(merged))
    parts.append("")
    parts.append(build_noise_table(merged))
    parts.append("")

    out_path = ROOT / args.out
    out_path.write_text("\n".join(parts))
    print(f"Wrote LaTeX tables to {out_path}")

    if args.print:
        print()
        print("\n".join(parts))


if __name__ == "__main__":
    main()
