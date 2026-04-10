"""
Classical baselines for fair VQC comparison.

Provides sklearn/xgboost/catboost models alongside a parameter-matched MLP.
Used by train.py for a unified experiment pipeline.

Models (regression):
    SVR_RBF, KernelRidge_RBF, Ridge, LinearRegression, XGBRegressor, CatBoostRegressor,
    MLPRegressor_ParamMatch

Models (classification):
    SVC_RBF, LogisticRegression, XGBClassifier, CatBoostClassifier,
    MLPClassifier_ParamMatch
"""

import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

_CLASSICAL_REGRESSION = {
    "SVR_RBF", "KernelRidge_RBF", "Ridge", "LinearRegression", "XGBRegressor", "CatBoostRegressor",
}
_CLASSICAL_CLASSIFICATION = {
    "SVC_RBF", "LogisticRegression", "XGBClassifier", "CatBoostClassifier",
}
_CLASSICAL_MODELS = _CLASSICAL_REGRESSION | _CLASSICAL_CLASSIFICATION


def is_classical(name: str) -> bool:
    """Check if a model name refers to a classical (non-NN) model."""
    return name in _CLASSICAL_MODELS


def build_classical_model(mc: dict, n_features: int = None, n_classes: int = None):
    """Build a classical (sklearn/xgboost) model from a model-config dict."""
    from sklearn.svm import SVR, SVC
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

    name = mc["name"]

    # ── Regression ──
    if name == "SVR_RBF":
        return SVR(kernel="rbf",
                   C=mc.get("C", 1.0),
                   gamma=mc.get("gamma", "scale"),
                   epsilon=mc.get("epsilon", 0.1))
    if name == "KernelRidge_RBF":
        return KernelRidge(kernel="rbf",
                           alpha=mc.get("alpha", 1.0),
                           gamma=mc.get("gamma", None))
    if name == "Ridge":
        return Ridge(alpha=mc.get("alpha", 1.0))
    if name == "LinearRegression":
        return LinearRegression()

    # ── Classification ──
    if name == "SVC_RBF":
        return SVC(kernel="rbf",
                   C=mc.get("C", 1.0),
                   gamma=mc.get("gamma", "scale"))
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=mc.get("max_iter", 1000))

    # ── XGBoost (optional dependency) ──
    try:
        from xgboost import XGBRegressor, XGBClassifier
    except ImportError:
        raise ImportError(f"xgboost not installed. Run: pip install xgboost")

    if name == "XGBRegressor":
        return XGBRegressor(
            n_estimators=mc.get("n_estimators", 500),
            max_depth=mc.get("max_depth", 6),
            learning_rate=mc.get("xgb_lr", 0.1),
            random_state=42,
        )
    if name == "XGBClassifier":
        return XGBClassifier(
            n_estimators=mc.get("n_estimators", 500),
            max_depth=mc.get("max_depth", 6),
            learning_rate=mc.get("xgb_lr", 0.1),
            random_state=42,
            eval_metric="mlogloss" if (n_classes and n_classes > 2) else "logloss",
        )

    # ── CatBoost (optional dependency) ──
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        raise ImportError(f"catboost not installed. Run: pip install catboost")

    if name == "CatBoostRegressor":
        return CatBoostRegressor(
            iterations=mc.get("iterations", 500),
            depth=mc.get("max_depth", 6),
            learning_rate=mc.get("cb_lr", 0.1),
            random_seed=42,
            verbose=0,
        )
    if name == "CatBoostClassifier":
        return CatBoostClassifier(
            iterations=mc.get("iterations", 500),
            depth=mc.get("max_depth", 6),
            learning_rate=mc.get("cb_lr", 0.1),
            random_seed=42,
            verbose=0,
        )
    raise ValueError(f"Unknown classical model: {name}")


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def _extract_original_scale(loader, x_scaler, y_scaler=None):
    """Convert loader data from [-π,π] back to original scale.

    Returns (X_orig, y_orig).  If y_scaler is None the labels are
    returned as-is (classification).

    NOTE: Iterates the loader in a single pass to keep X and y aligned
    (important when shuffle=True).
    """
    all_x, all_y = [], []
    for xb, yb in loader:
        all_x.append(xb)
        all_y.append(yb)
    X = torch.cat(all_x, 0).numpy()
    y = torch.cat(all_y, 0).numpy()
    X_orig = x_scaler.inverse_transform(X)
    if y_scaler is not None:
        y_orig = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    else:
        y_orig = y.ravel()
    return X_orig.astype(np.float64), y_orig


def train_classical(model, data: dict, model_dir: Path, run_name: str):
    """Fit a classical model on original-scale data with StandardScaler.

    Classical models (SVR, Ridge, XGBoost, CatBoost, …) are trained on
    features processed through sklearn StandardScaler and on un-scaled
    targets — the standard practice for classical ML.  The fitted
    StandardScaler is persisted alongside the model so that prediction
    can apply the same transform.
    """
    x_scaler = data["x_scaler"]
    y_scaler = data.get("y_scaler")  # None for classification

    X_orig, y_orig = _extract_original_scale(
        data["train_loader"], x_scaler, y_scaler)

    cls_scaler = StandardScaler()
    X_train = cls_scaler.fit_transform(X_orig)  # float64 — sklearn standard

    print(f"  Fitting {run_name} (StandardScaler → original-scale targets) ...")
    t0 = time.time()
    model.fit(X_train, y_orig.astype(np.float64))
    elapsed = time.time() - t0
    print(f"  Fit complete ({elapsed:.2f}s)")

    joblib.dump(model, model_dir / "best_model.joblib")
    joblib.dump(cls_scaler, model_dir / "cls_scaler.joblib")
    return model, cls_scaler


# ═══════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════

def predict_classical_regression(model, loader, x_scaler, y_scaler,
                                  cls_scaler):
    """Predict with a sklearn regression model.

    Data in the loader is in [-π,π] quantum encoding.  We inverse-
    transform features back to original scale, apply the classical
    StandardScaler, predict, and return (preds, trues) both in
    original target scale.
    """
    X_orig, y_orig = _extract_original_scale(loader, x_scaler, y_scaler)
    X_cls = cls_scaler.transform(X_orig)  # float64
    preds = model.predict(X_cls).ravel()
    return preds, y_orig


def predict_classical_classification(model, loader, x_scaler, cls_scaler):
    """Predict labels with a sklearn classifier."""
    X_orig, y_orig = _extract_original_scale(loader, x_scaler, y_scaler=None)
    X_cls = cls_scaler.transform(X_orig)  # float64
    preds = model.predict(X_cls).ravel()
    return preds.astype(int), y_orig.astype(int)


# ═══════════════════════════════════════════════════════════════════════
# Parameter-Matched MLP
# ═══════════════════════════════════════════════════════════════════════

class ParamMatchedMLP(nn.Module):
    """MLP whose hidden dim is auto-sized to approximately match a target
    parameter count, for fair capacity comparison against VQCs.

    Architecture: [Linear → ReLU (→ Dropout)] × n_layers, then Linear output.

    Total params = (n_layers-1)·h² + (in_dim + n_layers + out_dim)·h + out_dim
    """

    def __init__(self, in_dim, target_params, n_layers=3, out_dim=1, dropout=0.0):
        super().__init__()
        h = self._solve_hidden_dim(in_dim, n_layers, target_params, out_dim)
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(h, out_dim))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = h

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _solve_hidden_dim(in_dim, n_layers, target, out_dim):
        """Solve for hidden dim h so that total params ≈ target.

        Quadratic: a·h² + b·h + c = 0   where c = out_dim − target
        """
        a = n_layers - 1
        b = in_dim + n_layers + out_dim
        c = out_dim - target

        if a <= 0:  # single hidden layer → linear equation
            h = (target - out_dim) / max(b, 1)
        else:
            disc = b ** 2 - 4 * a * c
            if disc < 0:
                h = target / max(in_dim + out_dim, 1)
            else:
                h = (-b + math.sqrt(disc)) / (2 * a)
        return max(1, round(h))


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def count_params(model) -> int | None:
    """Trainable parameter count (nn.Module) or None (classical)."""
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return None


def make_run_name(mc: dict) -> str:
    """Generate a directory/label name from a model config dict."""
    name = mc["name"]
    if name in _CLASSICAL_MODELS or name.endswith("_ParamMatch"):
        return name
    layers = mc.get("hidden_layers", 3)
    depth = mc.get("vqc_depth", 3)
    n_heads = mc.get("n_heads", 1)
    parts = [name, f"L{layers}", f"K{depth}"]
    if n_heads > 1:
        parts.append(f"H{n_heads}")
    # Ablation suffixes
    if not mc.get("use_attention", True):
        parts.append("noAttn")
    ffn_mode = mc.get("ffn_mode", "fully")
    if ffn_mode != "fully":
        parts.append(f"ffn{ffn_mode}")
    if mc.get("use_layernorm", False):
        parts.append("LN")
    ns = mc.get("noise_strength", 0.0)
    if ns > 0:
        parts.append(f"noise{ns}")
    return "_".join(parts)
