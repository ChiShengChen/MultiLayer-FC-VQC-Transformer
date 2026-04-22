#!/usr/bin/env bash
# =================================================================
# FQT+LayerNorm 3-seed validation on classification datasets
# Validates practical recommendation (iv): "add LayerNorm to FQT"
# 2 datasets × 3 seeds = 6 runs
# =================================================================
set -e

# Seed 42 already has results from prior ablation runs; only run new seeds.
SEEDS=(123 7)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FQT+LN classification × 2 new seeds (42 already done)   ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "━━━ Wine Red | FQT+LN | seed=$seed ━━━"
    python train.py --config configs/wine_red_fqt_ln.json --seed "$seed"

    echo ""
    echo "━━━ Wine R+W | FQT+LN | seed=$seed ━━━"
    python train.py --config configs/wine_redwhite_fqt_ln.json --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FQT+LN 3-seed runs complete.                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
