"""
Expressibility analysis for Q3 VQC projections vs classical linear projections.

Computes the expressibility metric from Sim et al. (2019):
    Expr = D_KL( P_VQC(F) || P_Haar(F) )
where F is the fidelity between pairs of random states.

Usage:
    python expressibility_analysis.py [--n_samples 5000] [--depths 1 2 3 4 5]
"""

import argparse
import math
import numpy as np
import pennylane as qml
from scipy.stats import entropy
from collections import OrderedDict
import matplotlib.pyplot as plt


def haar_fidelity_pdf(f, n_qubits):
    """Analytical PDF of fidelity for Haar-random states on n_qubits."""
    N = 2 ** n_qubits
    return (N - 1) * (1 - f) ** (N - 2)


def compute_fidelities_vqc(n_qubits, depth, n_samples, circuit_fn="strong"):
    """Sample fidelities from random VQC parameter pairs."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(params):
        qml.AngleEmbedding(np.zeros(n_qubits), wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.state()

    fidelities = []
    for _ in range(n_samples):
        theta1 = np.random.uniform(0, 2 * np.pi, (depth, n_qubits, 3))
        theta2 = np.random.uniform(0, 2 * np.pi, (depth, n_qubits, 3))
        psi1 = circuit(theta1)
        psi2 = circuit(theta2)
        f = np.abs(np.vdot(psi1, psi2)) ** 2
        fidelities.append(f)
    return np.array(fidelities)


def compute_fidelities_linear(n_qubits, n_samples):
    """Sample fidelities from random linear projections (classical baseline).

    A linear projection W*x + b maps input to output of the same dimension.
    We use random Gaussian weights and compute cosine-similarity-based
    "fidelity" to measure the effective state diversity.
    """
    dim_in = n_qubits
    dim_out = n_qubits
    fidelities = []
    x = np.random.randn(dim_in)
    x = x / np.linalg.norm(x)

    for _ in range(n_samples):
        W1 = np.random.randn(dim_out, dim_in)
        W2 = np.random.randn(dim_out, dim_in)
        out1 = W1 @ x
        out2 = W2 @ x
        out1 = out1 / (np.linalg.norm(out1) + 1e-12)
        out2 = out2 / (np.linalg.norm(out2) + 1e-12)
        f = np.abs(np.dot(out1, out2)) ** 2
        fidelities.append(f)
    return np.array(fidelities)


def expressibility_kl(fidelities, n_qubits, n_bins=75):
    """Compute KL divergence between empirical fidelity distribution and Haar."""
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1),
                                   density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Haar reference
    haar = np.array([haar_fidelity_pdf(f, n_qubits) for f in bin_centers])
    haar = haar / haar.sum()  # normalize to probability

    # Empirical
    p = hist * bin_width
    p = p / p.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    haar = haar + eps
    p = p / p.sum()
    haar = haar / haar.sum()

    return entropy(p, haar)


def main():
    parser = argparse.ArgumentParser(description="VQC Expressibility Analysis")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n_qubits", type=int, default=3,
                        help="Number of qubits (default: 3, matching Q3 VQC)")
    parser.add_argument("--output", type=str, default="expressibility_results.png")
    args = parser.parse_args()

    n_q = args.n_qubits
    results = OrderedDict()

    print(f"Expressibility analysis: {n_q} qubits, {args.n_samples} samples")
    print(f"Depths: {args.depths}")
    print()

    # VQC expressibility at various depths
    for d in args.depths:
        print(f"  Computing VQC depth={d} ...", end=" ", flush=True)
        fids = compute_fidelities_vqc(n_q, d, args.n_samples)
        kl = expressibility_kl(fids, n_q)
        results[f"VQC depth={d}"] = {"fidelities": fids, "kl": kl}
        print(f"KL={kl:.6f}")

    # Classical linear projection baseline
    print(f"  Computing Linear projection ...", end=" ", flush=True)
    fids_lin = compute_fidelities_linear(n_q, args.n_samples)
    kl_lin = expressibility_kl(fids_lin, n_q)
    results["Linear proj."] = {"fidelities": fids_lin, "kl": kl_lin}
    print(f"KL={kl_lin:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fidelity histograms
    ax = axes[0]
    bins = np.linspace(0, 1, 76)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    haar = np.array([haar_fidelity_pdf(f, n_q) for f in bin_centers])
    haar = haar / haar.sum() * len(bin_centers)
    ax.plot(bin_centers, haar, "k--", linewidth=2, label="Haar random", zorder=10)

    for label, data in results.items():
        ax.hist(data["fidelities"], bins=bins, density=True, alpha=0.4, label=label)
    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Density")
    ax.set_title(f"Fidelity Distributions ({n_q} qubits)")
    ax.legend(fontsize=8)

    # Right: KL divergence bar chart
    ax = axes[1]
    labels = list(results.keys())
    kls = [results[k]["kl"] for k in labels]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(range(len(labels)), kls, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("KL Divergence (lower = more expressive)")
    ax.set_title("Expressibility Comparison")
    ax.set_yscale("log")
    for bar, kl in zip(bars, kls):
        ax.text(bar.get_x() + bar.get_width() / 2, kl * 1.1,
                f"{kl:.4f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"\nPlot saved to {args.output}")

    # Print summary table
    print(f"\n{'Model':<25s} {'KL Divergence':>15s}")
    print("-" * 42)
    for label, data in results.items():
        print(f"{label:<25s} {data['kl']:>15.6f}")


if __name__ == "__main__":
    main()
