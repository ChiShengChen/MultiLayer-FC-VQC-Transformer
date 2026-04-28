#!/usr/bin/env bash
# =================================================================
# MNIST 4 vs 9 binary classification, 3-seed
# 4 main models (FC-VQC, ResNet-VQC, QT, FQT)
# Subsampled to 1500 train / 500 test per class, PCA -> 12 features
# =================================================================
set -e

SEEDS=(42 123 7)
CFG="configs/mnist_4v9_multiseed.json"

# Auto-prepare CSV if missing
if [ ! -f "MNIST_4v9/data/mnist_4v9_pca11.csv" ]; then
    echo "MNIST CSV not found; running prepare_data.py..."
    python MNIST_4v9/prepare_data.py
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MNIST 4 vs 9 (3-seed): 4 models × 3 seeds = 12 runs    ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    # Skip if a completed run for this seed already exists
    existing=""
    for d in outputs/*_MNIST_4v9_seed${seed}; do
        if [ -d "$d" ] && [ -f "$d/comparison_metrics.csv" ]; then
            existing="$d"
            break
        fi
    done
    if [ -n "$existing" ]; then
        echo "  ⏭ Skipping seed=$seed (found $existing)"
        continue
    fi

    echo ""
    echo "━━━ MNIST 4 vs 9 | seed=$seed ━━━"
    python train.py --config "$CFG" --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MNIST 4 vs 9 multiseed complete.                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
