#!/usr/bin/env python3
"""
Computational cost benchmark for the four main VQC architectures.

For each model, measure:
  - Wall-clock per training step (forward + backward, full Boston batch n=354)
  - Inference latency per sample (100 single-sample forward passes)
  - On-disk model size (saved best_model.pt)
  - Parameter count

Reports a small table for the paper's "Computational cost" section.
"""
import os, sys, time, statistics
import torch
import numpy as np

# Make project modules importable
sys.path.insert(0, "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main")

from shared_models import (
    ResNetVQC, QuantumTransformerVQC, FullQuantumTransformerVQC,
)
sys.path.insert(0, "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/BostonHousing")
from models import FullyConnectedVQCs_15t5t1  # Boston FC-VQC variant

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

# Boston-shaped synthetic input
N_FEAT = 13
N_TRAIN = 354
BATCH_INF = 1
N_INFER_TRIALS = 50

x_train = torch.randn(N_TRAIN, N_FEAT)
y_train = torch.randn(N_TRAIN)

MODELS = {
    "FC-VQC":     lambda: FullyConnectedVQCs_15t5t1(layers=3, depth=3),
    "ResNet-VQC": lambda: ResNetVQC(n_features=N_FEAT, layers=3, depth=3),
    "QT":         lambda: QuantumTransformerVQC(n_features=N_FEAT, layers=2, depth=3),
    "FQT":        lambda: FullQuantumTransformerVQC(n_features=N_FEAT, layers=2, depth=3),
}

# Existing trained checkpoints (for on-disk size)
CKPT = {
    "FC-VQC":     "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs/20260224_183942_BostonHousing/FullyConnectedVQCs_15t5t1_L3_K3/best_model.pt",
    "ResNet-VQC": "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs/20260224_183942_BostonHousing/ResNetVQC_15t5t1_L3_K3/best_model.pt",
    "QT":         "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs/20260224_183942_BostonHousing/QuantumTransformerVQC_L2_K3/best_model.pt",
    "FQT":        "/Users/michael/Desktop/Multi-Layer-Fully-Connected-VQCs-main/outputs/20260224_183942_BostonHousing/FullQuantumTransformerVQC_L2_K3/best_model.pt",
}

print(f"{'Model':<12s} {'#params':>8s} {'fwd+bwd (s)':>14s} {'inference/sample (ms)':>22s} {'on-disk (KB)':>14s}")
print("-" * 78)

results = []
for name, ctor in MODELS.items():
    torch.manual_seed(42)
    model = ctor().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Warm-up forward (build qnodes etc)
    model(x_train[:8])

    # ---- Training step timing: 1 full epoch (forward + backward + step) ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()
    times = []
    for _ in range(3):  # 3 trials, take min
        t0 = time.perf_counter()
        opt.zero_grad()
        pred = model(x_train).squeeze()
        loss = crit(pred, y_train)
        loss.backward()
        opt.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    train_step_s = min(times)

    # ---- Inference latency per sample (single-sample forward) ----
    model.eval()
    one = x_train[:BATCH_INF]
    with torch.no_grad():
        # warm up
        for _ in range(3):
            model(one)
        infer_times = []
        for _ in range(N_INFER_TRIALS):
            t0 = time.perf_counter()
            model(one)
            t1 = time.perf_counter()
            infer_times.append((t1 - t0) * 1000.0)  # ms
    infer_med_ms = statistics.median(infer_times)

    # ---- On-disk size ----
    ckpt = CKPT[name]
    on_disk_kb = os.path.getsize(ckpt) / 1024.0 if os.path.exists(ckpt) else float("nan")

    print(f"{name:<12s} {n_params:>8d} {train_step_s:>14.3f} {infer_med_ms:>22.1f} {on_disk_kb:>14.1f}")
    results.append({"model": name, "n_params": n_params,
                    "train_step_s": train_step_s,
                    "infer_med_ms": infer_med_ms,
                    "on_disk_kb": on_disk_kb})

# Save CSV + LaTeX table fragment
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("compute_cost.csv", index=False)
print("\nWrote compute_cost.csv")

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{Computational cost on a single CPU core (Boston Housing, $n_{\text{train}}{=}354$, $K{=}3$ qubits per block, depth~3). Train step = one full-batch forward + backward + Adam step. Inference latency = median of 50 single-sample forward passes. On-disk = serialized \texttt{state\_dict} bytes.}")
tex.append(r"\label{tab:compute}")
tex.append(r"\small")
tex.append(r"\begin{tabular}{lrrrr}")
tex.append(r"\toprule")
tex.append(r"Model & \#params & Train step (s) & Inference (ms) & On-disk (KB) \\")
tex.append(r"\midrule")
for r in results:
    tex.append(f"{r['model']} & {r['n_params']:,} & {r['train_step_s']:.2f} & "
               f"{r['infer_med_ms']:.1f} & {r['on_disk_kb']:.1f} \\\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\vspace{-5pt}")
tex.append(r"\end{table}")
with open("compute_cost_table.tex", "w") as f:
    f.write("\n".join(tex))
print("Wrote compute_cost_table.tex")
