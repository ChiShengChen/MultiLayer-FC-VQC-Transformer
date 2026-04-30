#!/usr/bin/env bash
# =================================================================
# Multi-head attention 3-seed completion
# QT/FQT × {H=2, H=3} on 5 datasets — seed=42 already done in main
# multiseed runs (20260224..20260226). Only seed=123 and seed=7 missing.
# =================================================================
set -e

SEEDS=(123 7)
# Set DATASETS via env var; default to Boston only (fastest).
# Examples:
#   bash run_multihead_multiseed.sh                       # Boston only (~4-8h)
#   DATASETS="BostonHousing Concrete WineQuality_Red" \
#     bash run_multihead_multiseed.sh                     # 3 small datasets
#   DATASETS="BostonHousing Concrete WineQuality_Red WineQuality_RedandWhite CA_Housing" \
#     bash run_multihead_multiseed.sh                     # all 5 (~2-3 days)
if [ -n "${DATASETS_OVERRIDE:-}" ]; then
    read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
elif [ -n "${DATASETS:-}" ]; then
    read -r -a DATASETS <<< "$DATASETS"
else
    DATASETS=("BostonHousing")
fi
cfg_for_dataset() {
    case "$1" in
        BostonHousing)            echo "configs/boston_transformers_mh.json" ;;
        Concrete)                 echo "configs/concrete_transformers_mh.json" ;;
        WineQuality_Red)          echo "configs/wine_red_transformers_mh.json" ;;
        WineQuality_RedandWhite)  echo "configs/wine_redwhite_transformers_mh.json" ;;
        CA_Housing)               echo "configs/ca_housing_transformers_mh.json" ;;
        *) echo "" ;;
    esac
}

N_DS=${#DATASETS[@]}
N_RUNS=$((4 * N_DS * 2))
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Multi-head 3-seed completion                            ║"
echo "║  Datasets: ${DATASETS[*]}"
echo "║  4 models × $N_DS dataset(s) × 2 seeds = $N_RUNS runs"
echo "║  (seed=42 already done in original main multiseed runs)  ║"
echo "╚══════════════════════════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        cfg=$(cfg_for_dataset "$ds")
        if [ -z "$cfg" ]; then
            echo "  ⚠ Unknown dataset '$ds' — skipping"
            continue
        fi
        # Skip if a 4-model multi-head run already exists for this (ds, seed)
        existing=""
        for d in outputs/*_${ds}_seed${seed}; do
            if [ -d "$d" ] && [ -f "$d/comparison_metrics.csv" ]; then
                if grep -q "_H[23]" "$d/comparison_metrics.csv" 2>/dev/null; then
                    existing="$d"
                    break
                fi
            fi
        done
        if [ -n "$existing" ]; then
            echo "  ⏭ Skipping $ds seed=$seed (found $existing)"
            continue
        fi

        echo ""
        echo "━━━ $ds multi-head | seed=$seed ━━━"
        python train.py --config "$cfg" --seed "$seed"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Multi-head multiseed complete.                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
