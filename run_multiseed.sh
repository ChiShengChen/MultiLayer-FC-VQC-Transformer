#!/usr/bin/env bash
# =================================================================
# Multi-seed experiment runner
# 4 key models × 5 datasets × 3 seeds = 60 runs
# Seeds: 42, 123, 7
# =================================================================
set -e

SEEDS=(42 123 7)

CONFIGS=(
    configs/boston_multiseed.json
    configs/ca_housing_multiseed.json
    configs/concrete_multiseed.json
    configs/wine_red_multiseed.json
    configs/wine_redwhite_multiseed.json
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Multi-seed experiment: 5 datasets × 3 seeds × 4 models ║"
echo "║  Total runs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Config: $cfg  |  Seed: $seed"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python train.py --config "$cfg" --seed "$seed"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All multi-seed runs complete.                           ║"
echo "║  Run: python aggregate_multiseed.py to collect results.  ║"
echo "╚══════════════════════════════════════════════════════════╝"
