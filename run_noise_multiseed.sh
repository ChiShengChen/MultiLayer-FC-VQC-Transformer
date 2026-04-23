#!/usr/bin/env bash
# =================================================================
# Noise 3-seed validation (FQT only, Boston Housing)
# 3 seeds × 4 models (noiseless + 3 noise levels) = 12 runs
# Noiseless baseline included so seeds match exactly
# Estimated: ~6 hours total
# =================================================================
set -e

SEEDS=(42 123 7)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Noise 3-seed: FQT × {noiseless, p=.001, .005, .01}     ║"
echo "║  3 seeds × 4 models = 12 runs (~6 hours)                ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "━━━ Boston FQT noise study | seed=$seed (4 models) ━━━"
    python train.py --config configs/boston_noise_fqt_only.json --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Noise 3-seed runs complete.                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
