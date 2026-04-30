#!/usr/bin/env python3
"""
NISQ hardware-emulation demo: a single FC-VQC inference pass on Boston Housing
under a depolarizing-noise model whose strength is calibrated to typical IBM
Eagle-class device 2-qubit gate error rates (median ~5e-3 to 1e-2).

We do *not* execute on the IBM Quantum cloud here (no queue, no transpilation
overhead, reproducible offline) — this is a calibrated noise-model emulation,
explicitly noted as such in the paper.

Output:
  - QCE26_Q_FC_Transformer/figures/hw_demo.png  (single bar/scatter figure)
  - prints R²/RMSE for noiseless vs noise-emulated inference
"""
import os, sys, math, copy, functools
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt

ROOT = "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "BostonHousing"))
from models import (
    FullyConnectedVQCs_15t5t1,
    q_NtoN_Strong_function, q_Nto1_Strong_function,
)

OUT_PNG = os.path.join(ROOT, "QCE26_Q_FC_Transformer/figures/hw_demo.png")
CKPT = os.path.join(
    ROOT,
    "outputs/20260224_183942_BostonHousing/FullyConnectedVQCs_15t5t1_L3_K3/best_model.pt",
)

# IBM Eagle-class device-calibrated noise rate (median 2-qubit ECR error).
# Reference: ibm_brisbane / ibm_kyoto calibration data, late 2024 (~5e-3 to 1e-2 per ECR).
NOISE_P = 0.005

# ─── Build noisy QNode functions (depolarizing after every SEL layer) ────
def q_NtoN_Strong_noisy(x, weights, n_class, noise_strength=NOISE_P):
    n_qub = int(weights.shape[-2])
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    n_layers = weights.shape[0]
    for li in range(n_layers):
        qml.StronglyEntanglingLayers(weights[li:li+1], wires=range(n_qub))
        for w in range(n_qub):
            qml.DepolarizingChannel(noise_strength, wires=w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]


def q_Nto1_Strong_noisy(x, weights, n_class, noise_strength=NOISE_P):
    n_qub = int(weights.shape[-2])
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    n_layers = weights.shape[0]
    for li in range(n_layers):
        qml.StronglyEntanglingLayers(weights[li:li+1], wires=range(n_qub))
        for w in range(n_qub):
            qml.DepolarizingChannel(noise_strength, wires=w)
    return [qml.expval(qml.PauliZ(0))]


def patch_to_noisy(model, p):
    """Re-bind FC-VQC's QNodes to default.mixed devices with depolarizing noise."""
    dev3 = qml.device("default.mixed", wires=3, shots=None)
    dev5 = qml.device("default.mixed", wires=5, shots=None)
    nton3 = functools.partial(q_NtoN_Strong_noisy, noise_strength=p)
    nto13 = functools.partial(q_Nto1_Strong_noisy, noise_strength=p)
    nto15 = functools.partial(q_Nto1_Strong_noisy, noise_strength=p)
    model.dev_Q3 = dev3
    model.dev_Q5 = dev5
    model.quantum_net_Q3_3to3 = qml.QNode(nton3, dev3, interface="torch")
    model.quantum_net_Q3_3to1 = qml.QNode(nto13, dev3, interface="torch")
    model.quantum_net_Q5_5to1 = qml.QNode(nto15, dev5, interface="torch")


# ─── Load Boston test data via the official BostonHousing pipeline ──────
from functions import prepare_datasets

train_ld, val_ld, test_ld, y_scaler, n_feat, x_scaler = prepare_datasets(
    csv_path=os.path.join(ROOT, "BostonHousing/data/boston_housing.csv"),
    target_column="MEDV",
    clip_percentile=0.04,
    random_state=42,
)
xb_list, yb_list = [], []
for x, y in test_ld:
    xb_list.append(x); yb_list.append(y)
xb = torch.cat(xb_list, dim=0)
yb_norm = torch.cat(yb_list, dim=0).numpy().reshape(-1, 1)
yb = y_scaler.inverse_transform(yb_norm).flatten()
print(f"Boston test set: {xb.shape}, target range [{yb.min():.1f}, {yb.max():.1f}]")
ys = y_scaler  # alias for downstream

# ─── Load FC-VQC checkpoint ──────────────────────────────────────────────
torch.manual_seed(42)
model = FullyConnectedVQCs_15t5t1(layers=3, depth=3)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

def infer(model, xb):
    with torch.no_grad():
        pred_norm = model(xb).squeeze().cpu().numpy()
    pred = ys.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    rmse = float(np.sqrt(np.mean((pred - yb) ** 2)))
    ss_res = np.sum((yb - pred) ** 2)
    ss_tot = np.sum((yb - yb.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return pred, r2, rmse

print("\nRunning noiseless inference...")
pred_clean, r2_clean, rmse_clean = infer(model, xb)
print(f"  Noiseless: R² = {r2_clean:.4f}, RMSE = {rmse_clean:.3f}")

results_noisy = {}
for p in [0.001, 0.005]:
    print(f"\nPatching to noisy QNodes (p = {p})...")
    # Re-load fresh state because patching mutates the model attributes
    model_n = FullyConnectedVQCs_15t5t1(layers=3, depth=3)
    model_n.load_state_dict(state)
    model_n.eval()
    patch_to_noisy(model_n, p)
    pred_n, r2_n, rmse_n = infer(model_n, xb)
    print(f"  p={p}: R² = {r2_n:.4f}, RMSE = {rmse_n:.3f}")
    results_noisy[p] = (pred_n, r2_n, rmse_n)

# ─── Plot 2-panel: scatter + R² vs noise bar ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) Scatter overlay
ax = axes[0]
lo = float(min(yb.min(), pred_clean.min())) - 1
hi = float(max(yb.max(), pred_clean.max())) + 1
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="ideal")
ax.scatter(yb, pred_clean, s=30, c="#1f77b4", alpha=0.75, edgecolor="black",
           linewidth=0.3, label=f"Noiseless ($R^2{{=}}{r2_clean:.3f}$)")
p_color = {0.001: "#ff7f0e", 0.005: "#d62728"}
p_marker = {0.001: "s", 0.005: "^"}
for p, (pred_n, r2_n, _) in results_noisy.items():
    ax.scatter(yb, pred_n, s=30, c=p_color[p], alpha=0.7, edgecolor="black",
               linewidth=0.3, marker=p_marker[p],
               label=f"$p_d{{=}}{p}$ ($R^2{{=}}{r2_n:.3f}$)")
ax.set_xlabel("Ground truth (MEDV)")
ax.set_ylabel("Prediction")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_title("(a) Prediction vs ground truth")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(True, alpha=0.3)

# (b) R² vs noise level bar
ax = axes[1]
labels = ["noiseless"] + [f"$p_d{{=}}{p}$" for p in sorted(results_noisy.keys())]
vals = [r2_clean] + [results_noisy[p][1] for p in sorted(results_noisy.keys())]
colors = ["#1f77b4"] + [p_color[p] for p in sorted(results_noisy.keys())]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", lw=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02 if v >= 0 else v - 0.05,
            f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.set_ylabel(r"Test $R^2$")
ax.set_title("(b) Inference $R^2$ under emulated noise")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
print(f"\nNote: this is *noise-model emulation*, calibrated to median IBM Eagle/Heron")
print(f"2-qubit gate error rates (~5e-3) and single-qubit error rates (~1e-3 to 5e-4).")
print(f"Actual queue execution on IBM Quantum is left to future work.")
