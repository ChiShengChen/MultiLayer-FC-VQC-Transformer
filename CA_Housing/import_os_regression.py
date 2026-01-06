import os
import re
import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================

# List of folder names to process
FOLDER_NAMES = ["MLPRegressor","SingleVQC_8", "8t3t1", "16t4t1", "24t8t3t1", "32t11t4t1", "40t14t5t1", "48t16t4t1", "56t18t6t2"]   # add more, e.g. ["54t18t6", "another_folder"]

# Pattern templates to detect result files and extract L and D
# Examples:
#   FullyConnectedVQCs_54t18t6_L4_K8_stats.csv
#   SingleVQC_54t18t6_L4_K8_stats.csv
FILE_PATTERNS_TEMPLATES = [
    r"FullyConnectedVQCs_{folder}_L(?P<L>\d+)_K(?P<D>\d+)_stats(?:\.\w+)?$",
    r"{folder}_L(?P<L>\d+)_K(?P<D>\d+)_stats(?:\.\w+)?$",
]

EPOCH_COL = "epoch"
TARGET_EPOCH = 5000

# Metric column names in your CSV files
TRAIN_COL = "Train_R2"
VAL_COL   = "Val_R2"
TEST_COL  = "Test_R2"

# ============================================================
# File discovery
# ============================================================

def find_result_files(folder, folder_name):
    """
    Scan `folder` for files that match ANY of the patterns for the given
    folder_name. Returns a list of (path, L, D).
    """
    compiled_patterns = []
    for tmpl in FILE_PATTERNS_TEMPLATES:
        pattern_str = tmpl.format(folder=re.escape(folder_name))
        compiled_patterns.append(re.compile(pattern_str))

    files_info = []
    for fname in os.listdir(folder):
        for file_pattern in compiled_patterns:
            match = file_pattern.match(fname)
            if match:
                L_val = int(match.group("L"))
                D_val = int(match.group("D"))
                path = os.path.join(folder, fname)
                files_info.append((path, L_val, D_val))
                break  # stop after first matching pattern for this file
    return files_info


# ============================================================
# Processing: create last_epoch_Test_Acc_summary.csv only
# ============================================================

def process_folders(folder_names):
    """
    For each folder name in folder_names:
      - Find matching result files under that directory (both FullyConnectedVQCs_* and SingleVQC_*).
      - For each file, pick TARGET_EPOCH row if exists, otherwise last row.
      - Extract Test_Acc, Train_Acc, Val_Acc for that row.
      - Store in a combined summary with columns:
          Folder, L, D, epoch, Test, Train, Val
    Save a single CSV: last_epoch_Test_Acc_summary.csv
    """
    summary_rows = []

    for folder_name in folder_names:
        folder_path = folder_name  # assumes folder_name is also the path
        if not os.path.isdir(folder_path):
            print(f"Warning: folder '{folder_path}' not found, skipping.")
            continue

        files_info = find_result_files(folder_path, folder_name)
        if not files_info:
            print(f"No matching result files found in '{folder_path}'.")
            continue

        for path, L_val, D_val in files_info:
            print(f"Processing: {path}  (Folder={folder_name}, L={L_val}, D={D_val})")

            # Try comma-separated first, then whitespace
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, delim_whitespace=True)

            if EPOCH_COL not in df.columns:
                print(
                    f"Warning: File {path} does not contain required column "
                    f"'{EPOCH_COL}'. Skipping this file."
                )
                continue

            # Ensure epochs numeric and sorted
            df = df.copy()
            df[EPOCH_COL] = pd.to_numeric(df[EPOCH_COL], errors="coerce")
            df = df.dropna(subset=[EPOCH_COL])
            df = df.sort_values(EPOCH_COL).reset_index(drop=True)

            if df.empty:
                print(f"Warning: File {path} has no valid epoch rows. Skipping.")
                continue

            # Choose the row at TARGET_EPOCH if present, else use last row
            target_df = df[df[EPOCH_COL] == TARGET_EPOCH]
            if target_df.empty:
                print(
                    f"Warning: File {path} does not contain epoch {TARGET_EPOCH}. "
                    f"Using last available epoch instead."
                )
                target_row = df.iloc[-1]
            else:
                target_row = target_df.iloc[-1]

            epoch_used = int(target_row[EPOCH_COL])

            # Safely get Train/Val/Test accuracies (NaN if missing)
            test_val  = float(target_row[TEST_COL])  if TEST_COL  in df.columns else np.nan
            train_val = float(target_row[TRAIN_COL]) if TRAIN_COL in df.columns else np.nan
            val_val   = float(target_row[VAL_COL])   if VAL_COL   in df.columns else np.nan

            summary_rows.append({
                "Folder": folder_name,
                "L": L_val,
                "D": D_val,
                "epoch": epoch_used,
                "Test": test_val,
                "Train": train_val,
                "Val": val_val,
            })

    if not summary_rows:
        print("No data collected. No summary file will be written.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["Folder", "L", "D"]).reset_index(drop=True)

    out_name = "last_epoch_Test_Acc_summary.csv"
    summary_df.to_csv(out_name, index=False)
    print(f"Saved summary to: {out_name}")


# ============================================================
# Main entry
# ============================================================

if __name__ == "__main__":
    process_folders(FOLDER_NAMES)