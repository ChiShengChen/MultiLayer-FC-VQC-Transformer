#!/usr/bin/env bash
# =================================================================
# Ansatz comparison: StronglyEntanglingLayers vs BasicEntanglerLayers
# Architecture: ResNet-VQC on Boston Housing (smallest dataset)
# 2 ansatz × 3 seeds = 6 runs
# =================================================================
set -e

SEEDS=(42 123 7)
CFG="configs/boston_ansatz_compare.json"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ansatz comparison (Strong vs Basic) on ResNet-VQC      ║"
echo "║  3 seeds × 2 ansatz = 6 runs                            ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    existing=""
    for d in outputs/*_BostonHousing_seed${seed}; do
        if [ -d "$d" ] && [ -f "$d/comparison_metrics.csv" ] && \
           grep -q "ansatzbasic" "$d/comparison_metrics.csv" 2>/dev/null; then
            existing="$d"
            break
        fi
    done
    if [ -n "$existing" ]; then
        echo "  ⏭ Skipping seed=$seed (found $existing)"
        continue
    fi

    echo ""
    echo "━━━ Boston ResNet-VQC ansatz comparison | seed=$seed ━━━"
    python train.py --config "$CFG" --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ansatz comparison complete.                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
