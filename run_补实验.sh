#!/usr/bin/env bash
# =================================================================
# Supplementary experiments for reviewer response
# 1. MLP-720 on CA Housing + Concrete
# 2. Boston ablation × 3 seeds
# =================================================================
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Supplementary experiments                               ║"
echo "║  Part 1: MLP-720 baselines (2 runs)                     ║"
echo "║  Part 2: Boston ablation × 3 seeds (15 runs)            ║"
echo "║  Total: 17 runs                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Part 1: MLP-720 on CA Housing + Concrete ──
echo ""
echo "━━━ Part 1: MLP-720 baselines ━━━"

echo "  Running MLP-720 on CA Housing..."
python train.py --config configs/ca_housing_mlp720.json

echo "  Running MLP-720 on Concrete..."
python train.py --config configs/concrete_mlp720.json

# ── Part 2: Boston ablation × 3 seeds ──
echo ""
echo "━━━ Part 2: Boston ablation × 3 seeds ━━━"

SEEDS=(42 123 7)
for seed in "${SEEDS[@]}"; do
    # Skip if already completed
    existing=$(find outputs/ -maxdepth 1 -type d -name "*_BostonHousing_seed${seed}" 2>/dev/null | while read d; do
        cfg="$d/config.json"
        if [ -f "$cfg" ] && grep -q "use_attention" "$cfg" 2>/dev/null && [ -f "$d/comparison_metrics.csv" ]; then
            echo "$d"
            break
        fi
    done)
    if [ -n "$existing" ]; then
        echo "  ⏭ Skipping Boston ablation seed=$seed (found $existing)"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Boston ablation  |  Seed: $seed  (5 models)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python train.py --config configs/boston_ablation.json --seed "$seed"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All supplementary experiments complete.                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
