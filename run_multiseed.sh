#!/usr/bin/env bash
# =================================================================
# Multi-seed experiment runner
# 4 key models × 5 datasets × 3 seeds = 60 runs
# Seeds: 42, 123, 7
# =================================================================
set -e

# Seed 42 already has results from prior runs; only run new seeds.
SEEDS=(123 7)

CONFIGS=(
    configs/boston_multiseed.json
    configs/ca_housing_multiseed.json
    configs/concrete_multiseed.json
    configs/wine_red_multiseed.json
    configs/wine_redwhite_multiseed.json
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Multi-seed experiment: 5 datasets × 2 new seeds × 4 models ║"
echo "║  Total runs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))  (seed 42 already done)                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

for cfg in "${CONFIGS[@]}"; do
    # Extract experiment name from config
    exp=$(python3 -c "import json; print(json.load(open('$cfg'))['experiment'])")
    for seed in "${SEEDS[@]}"; do
        # Skip if a completed run already exists for this (experiment, seed)
        existing=$(find outputs/ -maxdepth 1 -type d -name "*_${exp}_seed${seed}" 2>/dev/null | head -1)
        if [ -n "$existing" ] && [ -f "${existing}/comparison_metrics.csv" ]; then
            echo ""
            echo "  ⏭ Skipping $exp seed=$seed (found ${existing})"
            continue
        fi
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Config: $cfg  |  Seed: $seed"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python train.py --config "$cfg" --seed "$seed"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All multi-seed runs complete. Aggregating results...    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
python aggregate_multiseed.py
echo ""
echo "Done. Results in paper/results/ and paper/figures/training_curves_comparison/"
