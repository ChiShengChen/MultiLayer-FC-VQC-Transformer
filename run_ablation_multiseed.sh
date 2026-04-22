#!/usr/bin/env bash
# =================================================================
# Ablation 3-seed validation for ALL datasets
# Seed 42 already exists; only run seeds 123 and 7
# 5 datasets × 2 seeds × 5 variants per config = 50 model runs
# =================================================================
set -e

SEEDS=(123 7)

CONFIGS=(
    configs/boston_ablation.json
    configs/ca_housing_ablation.json
    configs/concrete_ablation.json
    configs/wine_red_ablation.json
    configs/wine_redwhite_ablation.json
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ablation × 2 new seeds × 5 datasets (seed 42 done)     ║"
echo "║  Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} )) config runs (5 models each)          ║"
echo "╚══════════════════════════════════════════════════════════╝"

for cfg in "${CONFIGS[@]}"; do
    ds=$(python3 -c "import json; print(json.load(open('$cfg'))['experiment'])")
    for seed in "${SEEDS[@]}"; do
        # Skip if already completed
        existing=$(find outputs/ -maxdepth 1 -type d -name "*_${ds}_seed${seed}" 2>/dev/null | while read d; do
            c="$d/config.json"
            if [ -f "$c" ] && grep -q "use_attention" "$c" 2>/dev/null && [ -f "$d/comparison_metrics.csv" ]; then
                echo "$d"
                break
            fi
        done)
        if [ -n "$existing" ]; then
            echo "  ⏭ Skipping $ds ablation seed=$seed (found $existing)"
            continue
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  $ds ablation | Seed: $seed (5 models)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python train.py --config "$cfg" --seed "$seed"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All ablation multi-seed runs complete.                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
