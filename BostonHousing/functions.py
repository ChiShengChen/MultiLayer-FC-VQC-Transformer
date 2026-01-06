# EE_functions.py

import os
import math
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Generic CSV loader for regression (task-agnostic)
# -------------------------------------------------------------------
def load_regression_csv(
    csv_path: str,
    target_column: str,
    dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Generic local CSV loader for regression.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target_column : str
        Name of the target column in the CSV.
    dtype : str
        Numpy dtype for features and target.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    feature_names : List[str]
        Names of feature columns.
    target_name : str
        Name of the target column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            f"Please place your dataset there or provide the correct path."
        )

    df = pd.read_csv(csv_path)

    # Clean column names (remove whitespace)
    df.columns = [c.strip() for c in df.columns]

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_column]

    X = df[feature_cols].values.astype(dtype)
    y = df[target_column].values.astype(dtype)

    print(f"Loaded regression dataset from {csv_path}")
    print(f"Target column: {target_column}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, target_column


# -------------------------------------------------------------------
# Generic dataset preparation (no task-specific assumptions)
# -------------------------------------------------------------------
def prepare_datasets(
    csv_path: str,
    target_column: str,
    test_size: float = 0.30,
    random_state: int = 42,
    clip_percentile: float = 0,
    feature_range_train = (
        -np.pi + np.finfo(np.float32).eps,
        np.pi - np.finfo(np.float32).eps,
    ),
    batch_size_train: Optional[int] = None,
    batch_size_val: Optional[int] = None,
    batch_size_test: Optional[int] = None,
):
    """
    Load a regression CSV and prepare train/val/test DataLoaders.

    - Loads from `csv_path` using `target_column` as the regression target.
    - Splits into train / (val + test) via `test_size` (default 70/30).
    - Splits the remaining 30% into val/test 50/50 (default 15/15).
    - Scales features and targets to given ranges.

    Returns
    -------
    train_loader, val_loader, test_loader, y_scaler, n_features
    """
    # Feature/target scaling ranges (for quantum-style encodings)
    eps = np.finfo(np.float32).eps
    feature_range = (-np.pi + eps, np.pi - eps)
    target_range = (-np.pi + eps, np.pi - eps)
    # feature_range_train = feature_range# (-2.9, 2.9)

    # ---- Load data from CSV (generic) ----
    X, y, feature_names, target_name = load_regression_csv(
        csv_path=csv_path,
        target_column=target_column,
    )

    # ---- Train / val / test split ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    # ---- Optional clipping ----
    lower_bound_train = np.percentile(X_train, clip_percentile, axis=0)
    upper_bound_train = np.percentile(X_train, 100 - clip_percentile, axis=0)
    X_train = np.clip(X_train, lower_bound_train, upper_bound_train)

    # ---- Scale features ----
    x_scaler = MinMaxScaler(feature_range=feature_range_train)
    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

    X_train_s = np.clip(X_train_s, feature_range[0], feature_range[1])
    X_val_s = np.clip(X_val_s, feature_range[0], feature_range[1])
    X_test_s = np.clip(X_test_s, feature_range[0], feature_range[1])

    # ---- Scale targets ----
    y_scaler = MinMaxScaler(feature_range=target_range)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    y_train_s = np.clip(y_train_s, target_range[0], target_range[1])
    y_val_s = np.clip(y_val_s, target_range[0], target_range[1])
    y_test_s = np.clip(y_test_s, target_range[0], target_range[1])

    # ---- DataLoaders ----
    n_train = X_train_s.shape[0]
    n_val = X_val_s.shape[0]
    n_test = X_test_s.shape[0]

    bs_train = batch_size_train or n_train
    bs_val = batch_size_val or n_val
    bs_test = batch_size_test or n_test

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)),
        batch_size=bs_train,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_s)),
        batch_size=bs_val,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test_s)),
        batch_size=bs_test,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, y_scaler, X.shape[1]


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------
class Trainer:
    """
    Generic training & evaluation helper for regression models.
    Tracks gradient variance for analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        y_scaler,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        sched_factor: float = 0.5,
        sched_patience: int = 20,
        epochs: int = 1000,
        print_every: int = 25,
        eval_interval: int = 1000,
        run_name: str = "",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.y_scaler = y_scaler
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_factor,
                patience=sched_patience,
            )
        else:
            self.scheduler = None

        self.epochs = epochs
        self.print_every = print_every
        self.eval_interval = eval_interval
        self.run_name = run_name

        self.history = {
            "epoch": [],
            "train_mse": [],
            "val_mse": [],
            "grad_variance": [],
        }
        self.checkpoint_stats = {}

    # -----------------------------
    # Internal evaluation helper
    # -----------------------------
    def _eval_loader(self, dl):
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.device)
                pred = self.model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())

        preds = np.vstack(preds)
        trues = np.vstack(trues)

        preds = self.y_scaler.inverse_transform(preds).ravel()
        trues = self.y_scaler.inverse_transform(trues).ravel()

        r2 = r2_score(trues, preds)
        rmse = math.sqrt(mean_squared_error(trues, preds))
        mae = mean_absolute_error(trues, preds)
        return r2, rmse, mae

    def _compute_all_metrics(self):
        r2_tr, rmse_tr, mae_tr = self._eval_loader(self.train_loader)
        r2_va, rmse_va, mae_va = self._eval_loader(self.val_loader)
        r2_te, rmse_te, mae_te = self._eval_loader(self.test_loader)

        return {
            "train": {"R2": r2_tr, "RMSE": rmse_tr, "MAE": mae_tr},
            "val": {"R2": r2_va, "RMSE": rmse_va, "MAE": mae_va},
            "test": {"R2": r2_te, "RMSE": rmse_te, "MAE": mae_te},
        }

    # -----------------------------
    # Training loop
    # -----------------------------
    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            epoch_grad_vars = []

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()

                # Gradient variance
                all_grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        all_grads.append(p.grad.view(-1))
                if all_grads:
                    g_flat = torch.cat(all_grads)
                    g_var = torch.var(g_flat).item()
                    epoch_grad_vars.append(g_var)
                else:
                    epoch_grad_vars.append(0.0)

                self.optimizer.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= len(self.train_loader.dataset)
            avg_grad_var = float(np.mean(epoch_grad_vars)) if epoch_grad_vars else 0.0

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += (
                        self.criterion(self.model(xb), yb).item() * xb.size(0)
                    )
            val_loss /= len(self.val_loader.dataset)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.history["epoch"].append(epoch)
            self.history["train_mse"].append(train_loss)
            self.history["val_mse"].append(val_loss)
            self.history["grad_variance"].append(avg_grad_var)

            if epoch % self.print_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:05d} | "
                    f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
                    f"Grad Var: {avg_grad_var:.2e}"
                )

            if (epoch % self.eval_interval == 0) or (epoch == self.epochs):
                stats = self._compute_all_metrics()
                self.checkpoint_stats[epoch] = stats
                print(
                    f"[Eval @ epoch {epoch}] "
                    f"Test R2={stats['test']['R2']:.3f} | "
                    f"Test RMSE={stats['test']['RMSE']:.3f}"
                )

    # -----------------------------
    # History utilities
    # -----------------------------
    def history_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_history_csv(self, path: str):
        df = self.history_to_dataframe()
        df.to_csv(path, index=False)
        print(f"Saved per-epoch history to {path}")

    def save_stats_csv(self, path: str):
        rows = []
        for epoch, stats in sorted(self.checkpoint_stats.items()):
            rows.append(
                {
                    "epoch": epoch,
                    "Train_R2": stats["train"]["R2"],
                    "Train_RMSE": stats["train"]["RMSE"],
                    "Train_MAE": stats["train"]["MAE"],
                    "Val_R2": stats["val"]["R2"],
                    "Val_RMSE": stats["val"]["RMSE"],
                    "Val_MAE": stats["val"]["MAE"],
                    "Test_R2": stats["test"]["R2"],
                    "Test_RMSE": stats["test"]["RMSE"],
                    "Test_MAE": stats["test"]["MAE"],
                }
            )

        if not rows:
            print("No checkpoint stats to save.")
            return

        df_stats = pd.DataFrame(rows)
        df_stats.to_csv(path, index=False)
        print(f"Saved checkpoint stats to {path}")

    def plot_history(self, path: str | None = None, show: bool = True):
        if not self.history["epoch"]:
            print("No training history to plot.")
            return

        epochs = self.history["epoch"]
        train_mse = self.history["train_mse"]
        val_mse = self.history["val_mse"]

        plt.figure()
        plt.plot(epochs, train_mse, label="Train MSE")
        plt.plot(epochs, val_mse, label="Val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        title = self.run_name if self.run_name else "Training and validation loss"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved training curve plot to {path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_gradient_history(self, path: str | None = None, show: bool = True):
        if not self.history["epoch"]:
            print("No training history to plot.")
            return

        epochs = self.history["epoch"]
        grad_vars = self.history["grad_variance"]

        plt.figure()
        plt.plot(epochs, grad_vars, label="Gradient Variance", color="purple")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Variance of Gradients (Log Scale)")
        title = f"Gradient Dynamics: {self.run_name}" if self.run_name else "Gradient Variance"
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved gradient variance plot to {path}")

        if show:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Saved model parameters to {path}")
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model parameters from {path}")    


# -------------------------------------------------------------------
# Experiment logger
# -------------------------------------------------------------------
class ExperimentLogger:
    def __init__(self):
        self.rows = []

    def log(self, run_name: str, r2: float, rmse: float, mae: float,
            r2_train=None, rmse_train=None, mae_train=None,
            r2_val=None, rmse_val=None, mae_val=None):
        self.rows.append({
            "run_name": run_name,
            "Train_R2": r2_train, "Train_RMSE": rmse_train, "Train_MAE": mae_train,
            "Val_R2": r2_val, "Val_RMSE": rmse_val, "Val_MAE": mae_val,
            "Test_R2": r2, "Test_RMSE": rmse, "Test_MAE": mae,
        })
        print(f"\nLogged results for {run_name}")

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "run_name", "Train_R2", "Train_RMSE", "Train_MAE",
            "Val_R2", "Val_RMSE", "Val_MAE", "Test_R2", "Test_RMSE", "Test_MAE",
        ]
        return pd.DataFrame(self.rows, columns=cols)

    def save_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"\nSaved results to {path}")