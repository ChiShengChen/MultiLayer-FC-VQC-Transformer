#!/usr/bin/env python3
"""
Config-driven training wrapper for Multi-Layer FC-VQC experiments.

Wraps existing experiment directories without modifying them.
Supports regression (BostonHousing, CA_Housing, Concrete) and
classification (WineQuality_Red, WineQuality_RedandWhite) tasks.

Features:
  - Timestamped output directory per run
  - Best-model checkpointing (by val loss)
  - Multi-model comparison in a single config ("models" array)
  - Prediction vs Ground Truth overlay plot across all models
  - Training curve overlay comparison
  - Aggregated metrics CSV

Usage:
    python train.py --config configs/boston_compare.json
    python train.py --config configs/boston_resnet.json --depth 5 --layers 7
    python train.py --config configs/boston_resnet.json --sweep
    python train.py --list-models BostonHousing
    python train.py --list-models BostonHousing --module models_resnet
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score,
)
import joblib
import classical_models as cm

ROOT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Import
# ═══════════════════════════════════════════════════════════════════════

def _import_module(module_name: str, experiment: str):
    """Import a Python module from an experiment directory (or root for shared modules)."""
    module_path = ROOT_DIR / experiment / f"{module_name}.py"
    if not module_path.exists():
        # Fallback: check root directory for shared modules
        root_path = ROOT_DIR / f"{module_name}.py"
        if root_path.exists():
            module_path = root_path
        else:
            raise FileNotFoundError(
                f"Module not found: {module_path}\n"
                f"Also checked root: {root_path}\n"
                f"Available .py files in {experiment}/: "
                f"{[p.name for p in (ROOT_DIR / experiment).glob('*.py')]}"
            )
    uid = f"_exp_.{experiment}.{module_name}"
    if uid in sys.modules:
        return sys.modules[uid]
    spec = importlib.util.spec_from_file_location(uid, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[uid] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_models(experiment: str, module_name: str = "models"):
    """Return {class_name: class} for every nn.Module subclass in a module."""
    mod = _import_module(module_name, experiment)
    return {
        name: getattr(mod, name)
        for name in dir(mod)
        if isinstance(getattr(mod, name), type)
        and issubclass(getattr(mod, name), nn.Module)
        and getattr(mod, name) is not nn.Module
    }


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_model_configs(cfg: dict) -> list:
    """Extract model config(s) — supports both 'model' and 'models' keys."""
    if "models" in cfg:
        return cfg["models"]
    if "model" in cfg:
        return [cfg["model"]]
    raise ValueError("Config must have a 'model' or 'models' key.")


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Merge CLI flags into the config dict (CLI wins)."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    # depth/layers only apply to single-model configs
    if "model" in cfg:
        if args.depth is not None:
            cfg["model"]["vqc_depth"] = args.depth
        if args.layers is not None:
            cfg["model"]["hidden_layers"] = args.layers
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════

def _build_model_from_mc(mc: dict, experiment: str, n_features: int, n_classes: int = None):
    """Instantiate one model from a model-config dict."""
    name = mc["name"]
    depth = mc.get("vqc_depth", 3)
    layers = mc.get("hidden_layers", 3)
    module = mc.get("module", "models")

    # ── Classical models (sklearn/xgboost) ──
    if cm.is_classical(name):
        return cm.build_classical_model(mc, n_features=n_features, n_classes=n_classes)

    # ── Parameter-matched MLP ──
    if name.endswith("_ParamMatch"):
        target = mc.get("target_params", 200)
        out_dim = n_classes if (n_classes and n_classes > 1) else 1
        return cm.ParamMatchedMLP(in_dim=n_features, target_params=target,
                                   n_layers=layers, out_dim=out_dim,
                                   dropout=mc.get("dropout", 0.0))

    # ── nn.Module from experiment directories ──
    mod = _import_module(module, experiment)
    if not hasattr(mod, name):
        available = sorted(
            n for n in dir(mod)
            if isinstance(getattr(mod, n), type) and issubclass(getattr(mod, n), nn.Module)
        )
        raise ValueError(f"Model '{name}' not found in {experiment}/{module}.py.\nAvailable: {available}")
    cls = getattr(mod, name)

    # Shared models (from shared_models.py) — need n_features
    if module == "shared_models":
        kwargs = dict(n_features=n_features, layers=layers, depth=depth)
        if n_classes is not None and n_classes > 1:
            kwargs["n_classes"] = n_classes
        if mc.get("n_heads", 1) != 1 and "Transformer" in name:
            kwargs["n_heads"] = mc["n_heads"]
        # Ablation / noise parameters
        if "Transformer" in name:
            for key in ("ffn_mode", "use_attention", "use_layernorm",
                        "noise_strength"):
                if key in mc:
                    kwargs[key] = mc[key]
        elif "ResNet" in name:
            if "noise_strength" in mc:
                kwargs["noise_strength"] = mc["noise_strength"]
        return cls(**kwargs)

    if "MLPRegressor" in name:
        return cls(in_dim=n_features, layers=layers, dropout=mc.get("dropout", 0.0))
    if "MLPClassifier" in name:
        return cls(in_dim=n_features, n_classes=n_classes, layers=layers, dropout=mc.get("dropout", 0.0))
    if "CatBoost" in name or "XGBoost" in name:
        return cls(depth)
    return cls(layers=layers, depth=depth)


# legacy wrapper
def build_model(cfg: dict, n_features: int, n_classes: int = None):
    return _build_model_from_mc(cfg["model"], cfg["experiment"], n_features, n_classes)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def _resolve_csv(cfg: dict) -> str:
    p = cfg["data"]["csv_path"]
    if not os.path.isabs(p):
        p = str(ROOT_DIR / cfg["experiment"] / p)
    return p


def load_data(cfg: dict) -> dict:
    fn = _import_module("functions", cfg["experiment"])
    csv = _resolve_csv(cfg)
    target = cfg["data"]["target_column"]
    clip = cfg["data"].get("clip_percentile", 0)
    bs = cfg.get("batch_size", {})

    if cfg["task_type"] == "regression":
        train_ld, val_ld, test_ld, y_scaler, n_feat, x_scaler = fn.prepare_datasets(
            csv_path=csv, target_column=target, clip_percentile=clip,
            batch_size_train=bs.get("train"), batch_size_val=bs.get("val"),
            batch_size_test=bs.get("test"),
        )
        return dict(train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                    y_scaler=y_scaler, x_scaler=x_scaler,
                    n_features=n_feat, n_classes=None)

    if cfg["task_type"] == "classification":
        train_ld, val_ld, test_ld, n_feat, n_cls, x_scaler = fn.prepare_classification_datasets(
            csv_path=csv, target_column=target, clip_percentile=clip,
            batch_size_train=bs.get("train"), batch_size_val=bs.get("val"),
            batch_size_test=bs.get("test"),
        )
        return dict(train_loader=train_ld, val_loader=val_ld, test_loader=test_ld,
                    y_scaler=None, x_scaler=x_scaler,
                    n_features=n_feat, n_classes=n_cls)

    raise ValueError(f"Unknown task_type: {cfg['task_type']!r}")


# ═══════════════════════════════════════════════════════════════════════
# Output Directory
# ═══════════════════════════════════════════════════════════════════════

def create_output_dir(cfg: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = cfg["experiment"]
    out = ROOT_DIR / "outputs" / f"{ts}_{exp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict_regression(model, loader, y_scaler, device):
    """Return (preds_original_scale, trues_original_scale)."""
    model.eval()
    all_p, all_t = [], []
    for xb, yb in loader:
        all_p.append(model(xb.to(device)).cpu().numpy())
        all_t.append(yb.numpy())
    p = np.vstack(all_p)
    t = np.vstack(all_t)
    return y_scaler.inverse_transform(p).ravel(), y_scaler.inverse_transform(t).ravel()


@torch.no_grad()
def _predict_classification(model, loader, device):
    """Return (pred_labels, true_labels)."""
    model.eval()
    all_p, all_t = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        all_p.append(logits.argmax(dim=1).cpu())
        all_t.append(yb)
    return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()


def _regression_metrics(preds, trues):
    return {
        "R2": r2_score(trues, preds),
        "RMSE": math.sqrt(mean_squared_error(trues, preds)),
        "MAE": mean_absolute_error(trues, preds),
    }


def _classification_metrics(preds, trues):
    return {
        "Accuracy": accuracy_score(trues, preds),
        "F1_macro": f1_score(trues, preds, average="macro"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Training Loops with Best-Model Checkpoint
# ═══════════════════════════════════════════════════════════════════════

def _train_regression(model, data, device, tc, model_dir, run_name):
    """Custom regression training loop with best-model saving."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                 weight_decay=tc.get("weight_decay", 0.0))
    scheduler = None
    if tc.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=tc.get("sched_factor", 0.5),
            patience=tc.get("sched_patience", 20),
        )

    epochs = tc["epochs"]
    print_every = tc.get("print_every", 100)
    eval_interval = tc.get("eval_interval", 100)
    y_scaler = data["y_scaler"]

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    history = {"epoch": [], "train_mse": [], "val_mse": [], "grad_variance": []}
    checkpoint_stats = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        grad_vars = []

        for xb, yb in data["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            grad_vars.append(torch.var(torch.cat(grads)).item() if grads else 0.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(data["train_loader"].dataset)
        avg_gv = float(np.mean(grad_vars)) if grad_vars else 0.0

        # ── validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in data["val_loader"]:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(data["val_loader"].dataset)

        if scheduler is not None:
            scheduler.step(val_loss)

        # ── best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            torch.save(best_state, model_dir / "best_model.pt")

        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        history["grad_variance"].append(avg_gv)

        if epoch % print_every == 0 or epoch == 1:
            marker = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:05d} | Train MSE: {train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | Grad Var: {avg_gv:.2e}{marker}")

        if (epoch % eval_interval == 0) or epoch == epochs:
            p_tr, t_tr = _predict_regression(model, data["train_loader"], y_scaler, device)
            p_va, t_va = _predict_regression(model, data["val_loader"], y_scaler, device)
            p_te, t_te = _predict_regression(model, data["test_loader"], y_scaler, device)
            checkpoint_stats[epoch] = {
                "train": _regression_metrics(p_tr, t_tr),
                "val":   _regression_metrics(p_va, t_va),
                "test":  _regression_metrics(p_te, t_te),
            }
            s = checkpoint_stats[epoch]
            print(f"  [Eval @ {epoch}] Test R2={s['test']['R2']:.4f}  "
                  f"RMSE={s['test']['RMSE']:.4f}")

    # ── load best weights ──
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch} (val_mse={best_val_loss:.6f})")

    return model, history, checkpoint_stats, best_epoch


def _train_classification(model, data, device, tc, model_dir, run_name):
    """Custom classification training loop with best-model saving."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                 weight_decay=tc.get("weight_decay", 0.0))
    scheduler = None
    if tc.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=tc.get("sched_factor", 0.5),
            patience=tc.get("sched_patience", 20),
        )

    epochs = tc["epochs"]
    print_every = tc.get("print_every", 100)
    eval_interval = tc.get("eval_interval", 100)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "grad_variance": []}
    checkpoint_stats = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        grad_vars = []

        for xb, yb in data["train_loader"]:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            grad_vars.append(torch.var(torch.cat(grads)).item() if grads else 0.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss /= total
        train_acc = correct / total
        avg_gv = float(np.mean(grad_vars)) if grad_vars else 0.0

        # ── validation ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in data["val_loader"]:
                xb, yb = xb.to(device), yb.to(device).long()
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            torch.save(best_state, model_dir / "best_model.pt")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_variance"].append(avg_gv)

        if epoch % print_every == 0 or epoch == 1:
            marker = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:05d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f}{marker}")

        if (epoch % eval_interval == 0) or epoch == epochs:
            p_te, t_te = _predict_classification(model, data["test_loader"], device)
            m = _classification_metrics(p_te, t_te)
            checkpoint_stats[epoch] = m
            print(f"  [Eval @ {epoch}] Test Acc={m['Accuracy']:.4f}  F1={m['F1_macro']:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")

    return model, history, checkpoint_stats, best_epoch


    # NOTE: _train_booster removed — all classical models now go through
    # cm.train_classical() which uses StandardScaler + original-scale targets.


# ═══════════════════════════════════════════════════════════════════════
# Per-Model Output Saving
# ═══════════════════════════════════════════════════════════════════════

def _save_regression_outputs(history, checkpoint_stats, model_dir, run_name):
    # ── history CSV ──
    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

    # ── metrics CSV ──
    rows = []
    for epoch, s in sorted(checkpoint_stats.items()):
        rows.append({"epoch": epoch,
                      **{f"Train_{k}": v for k, v in s["train"].items()},
                      **{f"Val_{k}": v for k, v in s["val"].items()},
                      **{f"Test_{k}": v for k, v in s["test"].items()}})
    if rows:
        pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)

    # ── training curve plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"], label="Train MSE")
    ax.plot(history["epoch"], history["val_mse"], label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(run_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "training_curve.png", dpi=200)
    plt.close(fig)

    # ── gradient plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["grad_variance"], color="purple")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Variance")
    ax.set_title(f"{run_name} — Gradient Dynamics")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(model_dir / "gradient_history.png", dpi=200)
    plt.close(fig)


def _save_classification_outputs(history, checkpoint_stats, model_dir, run_name):
    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

    rows = []
    for epoch, s in sorted(checkpoint_stats.items()):
        rows.append({"epoch": epoch, **s})
    if rows:
        pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{run_name} Loss"); ax1.legend()
    ax2.plot(history["epoch"], history["train_acc"], label="Train")
    ax2.plot(history["epoch"], history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title(f"{run_name} Accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "training_curve.png", dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Cross-Model Comparison Plots
# ═══════════════════════════════════════════════════════════════════════

def _plot_regression_comparison(results, output_dir):
    """
    Generates:
      1. Pred vs GT scatter (x=GT, y=Pred, diagonal=perfect)
      2. Sorted overlay (samples sorted by GT value)
      3. Training curves overlay
    """
    n = len(results)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))

    # ── 1) Pred vs GT scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    trues = results[0]["test_trues"]
    lo, hi = trues.min(), trues.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1.5, alpha=0.6, label="Perfect (y=x)")
    for i, r in enumerate(results):
        n_p = r.get("n_params")
        p_str = f", {n_p:,}p" if n_p is not None else ""
        ax.scatter(r["test_trues"], r["test_preds"], color=colors[i],
                   alpha=0.45, s=18, label=f"{r['run_name']} (R\u00b2={r['test_metrics']['R2']:.4f}{p_str})")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction vs Ground Truth")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_pred_vs_gt.png", dpi=200)
    plt.close(fig)

    # ── 2) Sorted overlay ──
    sort_idx = np.argsort(trues)
    x_axis = np.arange(len(trues))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, trues[sort_idx], "k-", linewidth=2, label="Ground Truth", zorder=10)
    for i, r in enumerate(results):
        ax.plot(x_axis, r["test_preds"][sort_idx], color=colors[i],
                alpha=0.7, linewidth=1, label=r["run_name"])
    ax.set_xlabel("Sample (sorted by GT)")
    ax.set_ylabel("Value")
    ax.set_title("All Models vs Ground Truth (sorted)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_sorted_overlay.png", dpi=200)
    plt.close(fig)

    # ── 3) Training curves overlay ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, r in enumerate(results):
        if "history" not in r:
            continue
        h = r["history"]
        ax1.plot(h["epoch"], h["train_mse"], color=colors[i], alpha=0.6,
                 label=f"{r['run_name']} train")
        ax1.plot(h["epoch"], h["val_mse"], color=colors[i], linestyle="--",
                 label=f"{r['run_name']} val")
        ax2.plot(h["epoch"], h["val_mse"], color=colors[i], label=r["run_name"])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE"); ax1.set_title("Training Curves")
    ax1.legend(fontsize=6); ax1.set_yscale("log")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val MSE"); ax2.set_title("Validation Loss")
    ax2.legend(fontsize=7); ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_training_curves.png", dpi=200)
    plt.close(fig)

    print(f"\n  Comparison plots saved to {output_dir}/")


def _plot_classification_comparison(results, output_dir):
    """Bar chart comparing accuracy and F1 across models."""
    names = [r["run_name"] for r in results]
    accs = [r["test_metrics"]["Accuracy"] for r in results]
    f1s = [r["test_metrics"]["F1_macro"] for r in results]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 6))
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="steelblue")
    ax.bar(x + w / 2, f1s, w, label="F1 (macro)", color="coral")
    for i in range(len(names)):
        ax.text(x[i] - w / 2, accs[i] + 0.005, f"{accs[i]:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + w / 2, f1s[i] + 0.005, f"{f1s[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set")
    ax.set_ylim(0, 1.08)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_accuracy.png", dpi=200)
    plt.close(fig)

    # ── training curves overlay ──
    n = len(results)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, r in enumerate(results):
        if "history" not in r:
            continue
        h = r["history"]
        ax1.plot(h["epoch"], h["val_loss"], color=colors[i], label=r["run_name"])
        ax2.plot(h["epoch"], h["val_acc"], color=colors[i], label=r["run_name"])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss"); ax1.set_title("Validation Loss"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy"); ax2.set_title("Validation Accuracy"); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_training_curves.png", dpi=200)
    plt.close(fig)

    print(f"\n  Comparison plots saved to {output_dir}/")



def _load_previous_results(resume_dir: Path, task_type: str) -> list:
    """Load previously trained model results from an existing output directory."""
    results = []
    for subdir in sorted(resume_dir.iterdir()):
        if not subdir.is_dir():
            continue
        run_name = subdir.name
        preds_path = subdir / "test_preds.npy"
        trues_path = subdir / "test_trues.npy"
        history_path = subdir / "history.csv"
        if not preds_path.exists() or not trues_path.exists():
            print(f"  ⚠ Skipping {run_name}: missing test_preds.npy / test_trues.npy")
            continue

        result = {"run_name": run_name}
        result["test_preds"] = np.load(preds_path)
        result["test_trues"] = np.load(trues_path)

        if history_path.exists():
            h = pd.read_csv(history_path)
            result["history"] = {col: h[col].tolist() for col in h.columns}

        if task_type == "regression":
            result["test_metrics"] = _regression_metrics(result["test_preds"], result["test_trues"])
            result["train_metrics"] = result["test_metrics"]  # approximate
            result["val_metrics"] = result["test_metrics"]    # approximate
        else:
            result["test_metrics"] = _classification_metrics(result["test_preds"], result["test_trues"])
            result["train_metrics"] = result["test_metrics"]
            result["val_metrics"] = result["test_metrics"]

        print(f"  ✓ Loaded {run_name} from resume dir")
        results.append(result)
    return results



def replot_from_dir(output_dir_str: str):
    """
    Reload best-model weights from an existing output directory,
    re-run inference to produce .npy prediction files, and regenerate
    all comparison plots. No training is performed.
    Reads config.json from the output directory to rebuild models.
    """
    output_dir = Path(output_dir_str)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {output_dir_str}")

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {output_dir_str}")

    with open(config_path) as f:
        cfg = json.load(f)

    task = cfg["task_type"]
    device = torch.device(cfg.get("device", "cpu"))
    data = load_data(cfg)
    model_configs = _get_model_configs(cfg)

    # Discover model subdirs not in config.json (e.g. from earlier runs)
    _config_run_names = {cm.make_run_name(mc) for mc in model_configs}
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        _mc_path = subdir / "model_config.json"
        if subdir.name not in _config_run_names and _mc_path.exists():
            with open(_mc_path) as f:
                _discovered_mc = json.load(f)
            model_configs.append(_discovered_mc)
            _config_run_names.add(subdir.name)

    print(f"\n{'═' * 60}")
    print(f"  Replot: {output_dir}")
    print(f"  Task: {task}  |  Models in config: {len(model_configs)}")
    print(f"{'═' * 60}")

    # Resolve auto target_params for ParamMatch models
    _auto_needed = any(
        mc.get("name", "").endswith("_ParamMatch") and mc.get("target_params") == "auto"
        for mc in model_configs
    )
    if _auto_needed:
        _vqc_counts = []
        for _mc in model_configs:
            _n = _mc["name"]
            if _n in cm._CLASSICAL_MODELS or _n.endswith("_ParamMatch") or "MLP" in _n:
                continue
            _tmp = _build_model_from_mc(_mc, cfg["experiment"], data["n_features"], data["n_classes"])
            _vqc_counts.append(cm.count_params(_tmp))
            del _tmp
        if _vqc_counts:
            _auto_target = int(np.median(_vqc_counts))
            for _mc in model_configs:
                if _mc.get("name", "").endswith("_ParamMatch") and _mc.get("target_params") == "auto":
                    _mc["target_params"] = _auto_target
        else:
            for _mc in model_configs:
                if _mc.get("name", "").endswith("_ParamMatch") and _mc.get("target_params") == "auto":
                    _mc["target_params"] = 200

    results = []
    for mc in model_configs:
        name = mc["name"]
        run_name = cm.make_run_name(mc)
        model_dir = output_dir / run_name

        # Prefer per-model config (has resolved target_params etc.)
        _mc_path = model_dir / "model_config.json"
        if _mc_path.exists():
            with open(_mc_path) as f:
                mc = json.load(f)

        best_pt = model_dir / "best_model.pt"
        best_jl = model_dir / "best_model.joblib"
        if not model_dir.is_dir() or (not best_pt.exists() and not best_jl.exists()):
            print(f"  \u26a0 Skipping {run_name}: no saved model found")
            continue

        # Build model and load best weights
        model = _build_model_from_mc(mc, cfg["experiment"],
                                      data["n_features"], data["n_classes"])
        is_nn = isinstance(model, nn.Module)

        cls_scaler = None
        if best_jl.exists() and not is_nn:
            model = joblib.load(best_jl)
            cls_scaler_path = model_dir / "cls_scaler.joblib"
            if cls_scaler_path.exists():
                cls_scaler = joblib.load(cls_scaler_path)
            else:
                # Legacy model trained on [-π,π] data — retrain scaler
                cls_scaler = cm.StandardScaler()
                X_orig, _ = cm._extract_original_scale(
                    data["train_loader"], data["x_scaler"], data.get("y_scaler"))
                cls_scaler.fit(X_orig)
        elif best_pt.exists() and is_nn:
            model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
        else:
            print(f"  \u26a0 Skipping {run_name}: model/weight type mismatch")
            continue

        n_params = cm.count_params(model)
        result = {"run_name": run_name, "model_config": mc, "n_params": n_params}

        # Load history if available
        history_path = model_dir / "history.csv"
        if history_path.exists():
            h = pd.read_csv(history_path)
            result["history"] = {col: h[col].tolist() for col in h.columns}

        # Run inference and save predictions
        x_scaler = data["x_scaler"]
        if task == "regression":
            y_scaler = data["y_scaler"]
            if is_nn:
                p_tr, t_tr = _predict_regression(model, data["train_loader"], y_scaler, device)
                p_va, t_va = _predict_regression(model, data["val_loader"], y_scaler, device)
                p_te, t_te = _predict_regression(model, data["test_loader"], y_scaler, device)
            else:
                p_tr, t_tr = cm.predict_classical_regression(model, data["train_loader"], x_scaler, y_scaler, cls_scaler)
                p_va, t_va = cm.predict_classical_regression(model, data["val_loader"], x_scaler, y_scaler, cls_scaler)
                p_te, t_te = cm.predict_classical_regression(model, data["test_loader"], x_scaler, y_scaler, cls_scaler)
            result["test_preds"] = p_te
            result["test_trues"] = t_te
            result["train_metrics"] = _regression_metrics(p_tr, t_tr)
            result["val_metrics"] = _regression_metrics(p_va, t_va)
            result["test_metrics"] = _regression_metrics(p_te, t_te)
            np.save(model_dir / "test_preds.npy", p_te)
            np.save(model_dir / "test_trues.npy", t_te)
            print(f"  \u2713 {run_name} \u2014 R2={result['test_metrics']['R2']:.4f}")
        else:
            if is_nn:
                p_tr, t_tr = _predict_classification(model, data["train_loader"], device)
                p_va, t_va = _predict_classification(model, data["val_loader"], device)
                p_te, t_te = _predict_classification(model, data["test_loader"], device)
            else:
                p_tr, t_tr = cm.predict_classical_classification(model, data["train_loader"], x_scaler, cls_scaler)
                p_va, t_va = cm.predict_classical_classification(model, data["val_loader"], x_scaler, cls_scaler)
                p_te, t_te = cm.predict_classical_classification(model, data["test_loader"], x_scaler, cls_scaler)
            result["test_preds"] = p_te
            result["test_trues"] = t_te
            result["train_metrics"] = _classification_metrics(p_tr, t_tr)
            result["val_metrics"] = _classification_metrics(p_va, t_va)
            result["test_metrics"] = _classification_metrics(p_te, t_te)
            np.save(model_dir / "test_preds.npy", p_te)
            np.save(model_dir / "test_trues.npy", t_te)
            print(f"  \u2713 {run_name} \u2014 Acc={result['test_metrics']['Accuracy']:.4f}")

        results.append(result)

    # Pick up orphan subdirs that have .npy but weren't in config
    _processed_names = {r["run_name"] for r in results}
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in _processed_names:
            continue
        preds_path = subdir / "test_preds.npy"
        trues_path = subdir / "test_trues.npy"
        if not preds_path.exists() or not trues_path.exists():
            continue
        run_name = subdir.name
        r = {"run_name": run_name}
        r["test_preds"] = np.load(preds_path)
        r["test_trues"] = np.load(trues_path)
        history_path = subdir / "history.csv"
        if history_path.exists():
            h = pd.read_csv(history_path)
            r["history"] = {col: h[col].tolist() for col in h.columns}
        if task == "regression":
            r["test_metrics"] = _regression_metrics(r["test_preds"], r["test_trues"])
            r["train_metrics"] = r["test_metrics"]
            r["val_metrics"] = r["test_metrics"]
            print(f"  ✓ {run_name} (discovered) — R2={r['test_metrics']['R2']:.4f}")
        else:
            r["test_metrics"] = _classification_metrics(r["test_preds"], r["test_trues"])
            r["train_metrics"] = r["test_metrics"]
            r["val_metrics"] = r["test_metrics"]
            print(f"  ✓ {run_name} (discovered) — Acc={r['test_metrics']['Accuracy']:.4f}")
        results.append(r)

    if not results:
        print("  No models found to replot.")
        return

    # Regenerate comparison outputs
    if task == "regression":
        summary_rows = []
        for r in results:
            row = {"model": r["run_name"], "best_epoch": r.get("best_epoch", "N/A"),
                   "n_params": r.get("n_params", "")}
            for split in ("train", "val", "test"):
                for k, v in r[f"{split}_metrics"].items():
                    row[f"{split}_{k}"] = v
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
        _plot_regression_comparison(results, output_dir)
    else:
        summary_rows = []
        for r in results:
            row = {"model": r["run_name"], "best_epoch": r.get("best_epoch", "N/A"),
                   "n_params": r.get("n_params", "")}
            for split in ("train", "val", "test"):
                for k, v in r[f"{split}_metrics"].items():
                    row[f"{split}_{k}"] = v
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
        _plot_classification_comparison(results, output_dir)

    print(f"\n{'═' * 60}")
    print(f"  Replot done. Updated: {output_dir}")
    print(f"{'═' * 60}\n")

# ═══════════════════════════════════════════════════════════════════════
# Experiment Runner (main entry for new workflow)
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(cfg: dict, resume_dir: str = None):
    """
    Train one or more models, save all outputs to a timestamped directory,
    and generate cross-model comparison plots.
    If resume_dir is given, load previous results from that directory
    and append new model results for combined comparison.
    """
    if resume_dir:
        output_dir = Path(resume_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    else:
        output_dir = create_output_dir(cfg)
    device = torch.device(cfg.get("device", "cpu"))
    task = cfg["task_type"]
    model_configs = _get_model_configs(cfg)

    # Save config snapshot (merge on resume to keep all model refs)
    if resume_dir:
        _cfg_path = output_dir / "config.json"
        if _cfg_path.exists():
            with open(_cfg_path) as f:
                _existing = json.load(f)
            _existing_names = {m["name"] for m in _existing.get("models", [])}
            for _mc in model_configs:
                if _mc["name"] not in _existing_names:
                    _existing.setdefault("models", []).append(_mc)
            with open(_cfg_path, "w") as f:
                json.dump(_existing, f, indent=2)
        else:
            with open(output_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)
    else:
        with open(output_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Experiment: {cfg['experiment']}  |  Task: {task}")
    print(f"  Models: {len(model_configs)}  |  Output: {output_dir}")
    print(f"{'═' * 60}")

    # ── load data once (shared across models) ──
    data = load_data(cfg)

    # ── load previous results if resuming ──
    results = []
    if resume_dir:
        print(f"\n  Resuming from: {output_dir}")
        results = _load_previous_results(output_dir, task)
        print(f"  Loaded {len(results)} previous model(s)\n")

    # ── resolve auto target_params for ParamMatch models ──
    _auto_needed = any(
        mc.get("name", "").endswith("_ParamMatch") and mc.get("target_params") == "auto"
        for mc in model_configs
    )
    if _auto_needed:
        _vqc_counts = []
        for _mc in model_configs:
            _n = _mc["name"]
            if _n in cm._CLASSICAL_MODELS or _n.endswith("_ParamMatch") or "MLP" in _n:
                continue
            _tmp = _build_model_from_mc(_mc, cfg["experiment"], data["n_features"], data["n_classes"])
            _vqc_counts.append(cm.count_params(_tmp))
            del _tmp
        if _vqc_counts:
            _auto_target = int(np.median(_vqc_counts))
            for _mc in model_configs:
                if _mc.get("name", "").endswith("_ParamMatch") and _mc.get("target_params") == "auto":
                    _mc["target_params"] = _auto_target
            print(f"  ParamMatch target: {_auto_target:,} params (median of {len(_vqc_counts)} VQC models)")
        else:
            # No VQC models to match against — use sensible default
            _default = 200
            for _mc in model_configs:
                if _mc.get("name", "").endswith("_ParamMatch") and _mc.get("target_params") == "auto":
                    _mc["target_params"] = _default
            print(f"  ParamMatch target: {_default:,} params (default — no VQC models in config)")

    for midx, mc in enumerate(model_configs, 1):
        name = mc["name"]
        run_name = cm.make_run_name(mc)

        # Skip models already loaded from resume dir
        _existing_names = {r["run_name"] for r in results}
        if run_name in _existing_names:
            print(f"\n{'─' * 60}")
            print(f"  [{midx}/{len(model_configs)}] {run_name} — already trained, skipping")
            print(f"{'─' * 60}")
            continue
        model_dir = output_dir / run_name
        model_dir.mkdir(exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"  [{midx}/{len(model_configs)}] {run_name}")
        print(f"{'─' * 60}")

        model = _build_model_from_mc(mc, cfg["experiment"],
                                      data["n_features"], data["n_classes"])
        is_nn = isinstance(model, nn.Module)

        # Print param count
        n_params = cm.count_params(model)
        if n_params is not None:
            print(f"  Parameters: {n_params:,}")
        else:
            print(f"  Classical model ({model.__class__.__name__})")

        # Save per-model config (for replot reliability)
        with open(model_dir / "model_config.json", "w") as f:
            json.dump(mc, f, indent=2)

        result = {"run_name": run_name, "model_config": mc, "n_params": n_params}

        cls_scaler = None  # only set for classical models
        if not is_nn:
            model, cls_scaler = cm.train_classical(model, data, model_dir, run_name)
        elif task == "regression":
            model, hist, ckpt, best_ep = _train_regression(
                model, data, device, cfg["training"], model_dir, run_name)
            _save_regression_outputs(hist, ckpt, model_dir, run_name)
            result["history"] = hist
            result["best_epoch"] = best_ep
        else:
            model, hist, ckpt, best_ep = _train_classification(
                model, data, device, cfg["training"], model_dir, run_name)
            _save_classification_outputs(hist, ckpt, model_dir, run_name)
            result["history"] = hist
            result["best_epoch"] = best_ep

        # ── compute final metrics with best weights ──
        x_scaler = data["x_scaler"]
        if task == "regression":
            y_scaler = data["y_scaler"]
            if is_nn:
                p_tr, t_tr = _predict_regression(model, data["train_loader"], y_scaler, device)
                p_va, t_va = _predict_regression(model, data["val_loader"], y_scaler, device)
                p_te, t_te = _predict_regression(model, data["test_loader"], y_scaler, device)
            else:
                p_tr, t_tr = cm.predict_classical_regression(model, data["train_loader"], x_scaler, y_scaler, cls_scaler)
                p_va, t_va = cm.predict_classical_regression(model, data["val_loader"], x_scaler, y_scaler, cls_scaler)
                p_te, t_te = cm.predict_classical_regression(model, data["test_loader"], x_scaler, y_scaler, cls_scaler)
            result["test_preds"] = p_te
            result["test_trues"] = t_te
            result["train_metrics"] = _regression_metrics(p_tr, t_tr)
            result["val_metrics"] = _regression_metrics(p_va, t_va)
            result["test_metrics"] = _regression_metrics(p_te, t_te)

            np.save(model_dir / "test_preds.npy", p_te)
            np.save(model_dir / "test_trues.npy", t_te)

            print(f"  \u25ba Best Test  R2={result['test_metrics']['R2']:.4f}  "
                  f"RMSE={result['test_metrics']['RMSE']:.4f}  "
                  f"MAE={result['test_metrics']['MAE']:.4f}")
        else:
            if is_nn:
                p_tr, t_tr = _predict_classification(model, data["train_loader"], device)
                p_va, t_va = _predict_classification(model, data["val_loader"], device)
                p_te, t_te = _predict_classification(model, data["test_loader"], device)
            else:
                p_tr, t_tr = cm.predict_classical_classification(model, data["train_loader"], x_scaler, cls_scaler)
                p_va, t_va = cm.predict_classical_classification(model, data["val_loader"], x_scaler, cls_scaler)
                p_te, t_te = cm.predict_classical_classification(model, data["test_loader"], x_scaler, cls_scaler)
            result["test_preds"] = p_te
            result["test_trues"] = t_te
            result["train_metrics"] = _classification_metrics(p_tr, t_tr)
            result["val_metrics"] = _classification_metrics(p_va, t_va)
            result["test_metrics"] = _classification_metrics(p_te, t_te)

            np.save(model_dir / "test_preds.npy", p_te)
            np.save(model_dir / "test_trues.npy", t_te)

            print(f"  \u25ba Best Test  Acc={result['test_metrics']['Accuracy']:.4f}  "
                  f"F1={result['test_metrics']['F1_macro']:.4f}")

        results.append(result)

    # ═══════════════════════════════════════════════════════════════
    # Cross-model outputs
    # ═══════════════════════════════════════════════════════════════
    if task == "regression":
        summary_rows = []
        for r in results:
            row = {"model": r["run_name"], "best_epoch": r.get("best_epoch", "N/A"),
                   "n_params": r.get("n_params", "")}
            for split in ("train", "val", "test"):
                for k, v in r[f"{split}_metrics"].items():
                    row[f"{split}_{k}"] = v
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
        _plot_regression_comparison(results, output_dir)
    else:
        summary_rows = []
        for r in results:
            row = {"model": r["run_name"], "best_epoch": r.get("best_epoch", "N/A"),
                   "n_params": r.get("n_params", "")}
            for split in ("train", "val", "test"):
                for k, v in r[f"{split}_metrics"].items():
                    row[f"{split}_{k}"] = v
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
        _plot_classification_comparison(results, output_dir)

    print(f"\n{'═' * 60}")
    print(f"  Done. All outputs in: {output_dir}")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════
# Legacy: train_single / run_sweep (backward compatible)
# ═══════════════════════════════════════════════════════════════════════

def train_single(cfg: dict):
    """Legacy single-model training (no timestamped dir). Use run_experiment instead."""
    run_experiment(cfg)


def run_sweep(cfg: dict):
    sw = cfg.get("sweep", {})
    depths = sw.get("depths", list(range(1, 10, 2)))
    layers_list = sw.get("layers", list(range(1, 10, 2)))

    total = len(depths) * len(layers_list)
    print(f"\n>>> Sweep: {len(depths)} depths × {len(layers_list)} layers = {total} runs\n")

    for idx, d in enumerate(depths):
        for jdx, l in enumerate(layers_list):
            run_cfg = copy.deepcopy(cfg)
            # Apply to all model configs
            for mc in _get_model_configs(run_cfg):
                mc["vqc_depth"] = d
                mc["hidden_layers"] = l
            n = idx * len(layers_list) + jdx + 1
            print(f"\n[{n}/{total}] depth={d}, layers={l}")
            try:
                run_experiment(run_cfg)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def cmd_list_models(experiment: str, module: str = None):
    """Print every nn.Module subclass available in an experiment."""
    exp_dir = ROOT_DIR / experiment
    if not exp_dir.is_dir():
        print(f"Directory not found: {exp_dir}")
        return

    targets = [module] if module else [p.stem for p in sorted(exp_dir.glob("models*.py"))]

    print(f"\n  Models in {experiment}/")
    print(f"  {'─' * 50}")
    for mod_name in targets:
        models = discover_models(experiment, mod_name)
        for cls_name in sorted(models):
            sig = "(layers, depth)"
            if "MLP" in cls_name:
                sig = "(in_dim, [n_classes,] layers, dropout)"
            elif "Boost" in cls_name or "XGBoost" in cls_name:
                sig = "(depth)"
            print(f"    {mod_name:20s} :: {cls_name}{sig}")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Config-driven training for Multi-Layer FC-VQC experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python train.py --config configs/boston_compare.json          # multi-model comparison
  python train.py --config configs/boston_resnet.json            # single model
  python train.py --config configs/boston_resnet.json --depth 5 --layers 7
  python train.py --config configs/boston_resnet.json --sweep
  python train.py --config configs/boston_transformers.json --resume-dir outputs/YYYYMMDD_BostonHousing/
  python train.py --replot outputs/YYYYMMDD_BostonHousing/     # regenerate .npy + comparison from best weights
  python train.py --list-models BostonHousing
  python train.py --list-models BostonHousing --module models_resnet
""",
    )
    p.add_argument("--config", type=str, help="Path to JSON config file")
    p.add_argument("--depth", type=int, help="Override model.vqc_depth (single-model only)")
    p.add_argument("--layers", type=int, help="Override model.hidden_layers (single-model only)")
    p.add_argument("--epochs", type=int, help="Override training.epochs")
    p.add_argument("--lr", type=float, help="Override training.lr")
    p.add_argument("--sweep", action="store_true", help="Grid search over sweep.depths × sweep.layers")
    p.add_argument("--list-models", metavar="EXPERIMENT", help="List models in an experiment directory")
    p.add_argument("--module", type=str, help="Filter --list-models to a specific module file")
    p.add_argument("--resume-dir", type=str,
                    help="Resume from existing output dir: load previous results, train new models, regenerate comparison")
    p.add_argument("--replot", type=str, metavar="OUTPUT_DIR",
                    help="Reload best weights from output dir, re-infer predictions, regenerate comparison plots (no training)")

    args = p.parse_args()

    if args.replot:
        replot_from_dir(args.replot)
        return

    if args.list_models:
        cmd_list_models(args.list_models, args.module)
        return

    if not args.config:
        if not args.replot:
            p.print_help()
        return

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.sweep:
        run_sweep(cfg)
    else:
        run_experiment(cfg, resume_dir=args.resume_dir)


if __name__ == "__main__":
    main()
