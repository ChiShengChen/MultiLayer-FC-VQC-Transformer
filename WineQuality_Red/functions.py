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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


# -------------------------------------------------------------------
# Generic CSV loader for regression (task-agnostic)
# -------------------------------------------------------------------
def load_classification_csv(
    csv_path: str,
    target_column: str,
    dtype: str = "float32",
):
    """
    Generic local CSV loader for classification.

    - Reads CSV
    - Extracts features X and labels y
    - Converts labels to integer class indices 0..C-1

    Returns
    -------
    X : np.ndarray, shape [N, D]
    y : np.ndarray, shape [N] (int labels)
    feature_names : List[str]
    target_name : str
    n_classes : int
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            f"Please place your dataset there or provide the correct path."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_column]

    X = df[feature_cols].values.astype(dtype)
    y_raw = df[target_column].values

    # Map arbitrary labels (e.g. 3,4,5,6,7,8) to 0..C-1
    classes, y_int = np.unique(y_raw, return_inverse=True)
    n_classes = len(classes)

    print(f"Loaded classification dataset from {csv_path}")
    print(f"Target column: {target_column}")
    print(f"Classes (original): {classes}")
    print(f"X shape: {X.shape}, y shape: {y_int.shape}, n_classes={n_classes}")

    return X, y_int.astype(np.int64), feature_cols, target_column, n_classes


# -------------------------------------------------------------------
# Generic dataset preparation (no task-specific assumptions)
# -------------------------------------------------------------------
def prepare_classification_datasets(
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
    Load a classification CSV and prepare train/val/test DataLoaders.

    - Features are scaled similar to the regression version.
    - Labels are integer class indices (no target scaling).
    - Returns no y_scaler.

    Returns
    -------
    train_loader, val_loader, test_loader, n_features, n_classes
    """
    eps = np.finfo(np.float32).eps
    feature_range = (-np.pi + eps, np.pi - eps)

    # ---- Load data ----
    X, y, feature_names, target_name, n_classes = load_classification_csv(
        csv_path=csv_path,
        target_column=target_column,
    )

    # ---- Train / val / test split ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # ---- Optional clipping ----
    lower_bound_train = np.percentile(X_train, clip_percentile, axis=0)
    upper_bound_train = np.percentile(X_train, 100 - clip_percentile, axis=0)
    X_train_p = np.clip(X_train, lower_bound_train, upper_bound_train)

    # ---- Scale features (same style as regression) ----
    x_scaler = MinMaxScaler(feature_range=feature_range_train)
    X_train_s = x_scaler.fit_transform(X_train_p).astype(np.float32)
    X_val_s   = x_scaler.transform(X_val).astype(np.float32)
    X_test_s  = x_scaler.transform(X_test).astype(np.float32)

    X_train_s = np.clip(X_train_s, feature_range[0], feature_range[1])
    X_val_s   = np.clip(X_val_s, feature_range[0], feature_range[1])
    X_test_s  = np.clip(X_test_s, feature_range[0], feature_range[1])

    # ---- Tensors ----
    X_train_t = torch.from_numpy(X_train_s)
    X_val_t   = torch.from_numpy(X_val_s)
    X_test_t  = torch.from_numpy(X_test_s)

    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_val_t   = torch.from_numpy(y_val.astype(np.int64))
    y_test_t  = torch.from_numpy(y_test.astype(np.int64))

    # ---- DataLoaders ----
    n_train = X_train_t.shape[0]
    n_val   = X_val_t.shape[0]
    n_test  = X_test_t.shape[0]

    bs_train = batch_size_train or n_train
    bs_val   = batch_size_val   or n_val
    bs_test  = batch_size_test  or n_test

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=bs_train,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=bs_val,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=bs_test,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, X.shape[1], n_classes


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------

class ClassificationTrainer:
    """
    Training & evaluation helper for classification models.
    Uses CrossEntropyLoss and tracks accuracy + gradient variance.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",           # maximize validation accuracy
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
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "grad_variance": [],
        }
        self.checkpoint_stats = {}

    @torch.no_grad()
    def _eval_loader(self, dl):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in dl:
            xb = xb.to(self.device)
            yb = yb.to(self.device).long()

            logits = self.model(xb)             # [B, C]
            loss = self.criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return avg_loss, acc

    def train(self):
        # best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            total_correct = 0
            total_samples = 0
            epoch_grad_vars = []

            for step, (xb, yb) in enumerate(self.train_loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device).long()

                self.optimizer.zero_grad()
                logits = self.model(xb)         # [B, C]
                loss = self.criterion(logits, yb)
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
                preds = logits.argmax(dim=1)
                total_correct += (preds == yb).sum().item()
                total_samples += xb.size(0)

                if (step + 1) % self.print_every == 0:
                    print(
                        f"Epoch {epoch:05d} Step {step+1} | "
                        f"Batch Loss: {loss.item():.4f}"
                    )

            train_loss /= total_samples
            train_acc = total_correct / total_samples
            avg_grad_var = float(np.mean(epoch_grad_vars)) if epoch_grad_vars else 0.0

            val_loss, val_acc = self._eval_loader(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_acc)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["grad_variance"].append(avg_grad_var)

            if epoch % self.print_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:05d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"Grad Var: {avg_grad_var:.2e}"
                )

            if (epoch % self.eval_interval == 0) or (epoch == self.epochs):
                train_loss_eval, train_acc_eval = self._eval_loader(self.train_loader)
                val_loss_eval, val_acc_eval = self._eval_loader(self.val_loader)
                test_loss_eval, test_acc_eval = self._eval_loader(self.test_loader)
                self.checkpoint_stats[epoch] = {
                    "train": {"loss": train_loss_eval, "acc": train_acc_eval},
                    "val": {"loss": val_loss_eval, "acc": val_acc_eval},
                    "test": {"loss": test_loss_eval, "acc": test_acc_eval},
                }
                print(
                    f"[Eval @ epoch {epoch}] "
                    f"Test Loss={test_loss_eval:.4f} Acc={test_acc_eval:.4f}"
                )

                # if val_acc_eval > best_val_acc:
                #     best_val_acc = val_acc_eval
                #     torch.save(
                #         self.model.state_dict(),
                #         f"{self.run_name}_best.pth"
                #     )

        # best_path = f"{self.run_name}_best.pth"
        # if os.path.exists(best_path):
        #     self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        #     test_loss, test_acc = self._eval_loader(self.test_loader)
        #     print(
        #         f"[Best Model] Test Loss={test_loss:.4f} Acc={test_acc:.4f}"
        #     )

    def history_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_history_csv(self, path: str):
        df = self.history_to_dataframe()
        df.to_csv(path, index=False)
        print(f"Saved per-epoch classification history to {path}")

    def save_stats_csv(self, path: str):
        rows = []
        for epoch, stats in sorted(self.checkpoint_stats.items()):
            rows.append(
                {
                    "epoch": epoch,
                    "Train_Loss": stats["train"]["loss"],
                    "Train_Acc": stats["train"]["acc"],
                    "Val_Loss": stats["val"]["loss"],
                    "Val_Acc": stats["val"]["acc"],
                    "Test_Loss": stats["test"]["loss"],
                    "Test_Acc": stats["test"]["acc"],
                }
            )

        if not rows:
            print("No checkpoint stats to save.")
            return

        df_stats = pd.DataFrame(rows)
        df_stats.to_csv(path, index=False)
        print(f"Saved classification checkpoint stats to {path}")

    def plot_history(self, path: str | None = None, show: bool = True):
        if not self.history["epoch"]:
            print("No training history to plot.")
            return

        epochs = self.history["epoch"]
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]
        train_acc = self.history["train_acc"]
        val_acc = self.history["val_acc"]

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(self.run_name + " (Loss)" if self.run_name else "Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(self.run_name + " (Accuracy)" if self.run_name else "Accuracy")
        plt.legend()

        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved classification training curves to {path}")

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
        title = f"Gradient Dynamics (Cls): {self.run_name}" if self.run_name else "Gradient Variance (Cls)"
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved classification gradient variance plot to {path}")

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

    @torch.no_grad()
    def get_all_preds(self, loader):
        """Helper to get all predictions and true labels from a loader."""
        self.model.eval()
        all_preds = []
        all_trues = []
        for xb, yb in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_trues.append(yb.cpu())
        
        return torch.cat(all_preds).numpy(), torch.cat(all_trues).numpy()

    def plot_confusion_matrix(self, loader, path: str | None = None, show: bool = True):
        """Generates and saves a Confusion Matrix plot."""
        y_pred, y_true = self.get_all_preds(loader)
        
        # Calculate matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotting
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
        
        plt.title(f"Confusion Matrix: {self.run_name}")
        
        if path:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved Confusion Matrix to {path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        # Also return F1 for logging
        return f1_score(y_true, y_pred, average="macro")
    
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