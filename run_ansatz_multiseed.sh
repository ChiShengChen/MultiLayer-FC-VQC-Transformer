#!/usr/bin/env bash
# =================================================================
# Ansatz comparison: BasicEntanglerLayers across all 4 architectures
# (Strong is already covered by the main multiseed runs.)
# Architectures: FC-VQC, ResNet-VQC, QT, FQT on Boston Housing
# 4 models × 3 seeds = 12 runs (~3-5 hours total on a laptop CPU)
# =================================================================
set -e

SEEDS=(42 123 7)
CFG="configs/boston_ansatz_compare.json"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ansatz comparison: Basic across 4 architectures        ║"
echo "║  3 seeds × 4 models = 12 runs                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    # Skip if a 4-model basic-ansatz run already exists for this seed
    existing=""
    for d in outputs/*_BostonHousing_seed${seed}; do
        if [ -d "$d" ] && [ -f "$d/comparison_metrics.csv" ]; then
            n=$(grep -c "ansatzbasic" "$d/comparison_metrics.csv" 2>/dev/null || echo 0)
            if [ "$n" -ge 4 ]; then
                existing="$d"
                break
            fi
        fi
    done
    if [ -n "$existing" ]; then
        echo "  ⏭ Skipping seed=$seed (found $existing)"
        continue
    fi

    echo ""
    echo "━━━ Boston ansatz comparison (Basic × 4 models) | seed=$seed ━━━"
    python train.py --config "$CFG" --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ansatz comparison complete.                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
