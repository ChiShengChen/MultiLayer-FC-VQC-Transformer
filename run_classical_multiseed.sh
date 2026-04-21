#!/usr/bin/env bash
# =================================================================
# Multi-seed classical baselines (CatBoost, XGBoost, KRR, SVR, MLP)
# 3 seeds × 5 datasets × 2 configs (classical + boosting) = 30 runs
# These run fast (no VQC simulation)
# =================================================================
set -e

SEEDS=(42 123 7)

# Regression datasets
REG_DATASETS=(
    "boston:configs/boston_classical.json:configs/boston_boosting.json"
    "ca_housing:configs/ca_housing_classical.json:configs/ca_housing_boosting.json"
    "concrete:configs/concrete_classical.json:configs/concrete_boosting.json"
)

# Classification datasets
CLS_DATASETS=(
    "wine_red:configs/wine_red_classical.json:configs/wine_red_boosting.json"
    "wine_rw:configs/wine_redwhite_classical.json:configs/wine_redwhite_boosting.json"
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Classical baselines × 3 seeds × 5 datasets              ║"
echo "╚══════════════════════════════════════════════════════════╝"

for entry in "${REG_DATASETS[@]}" "${CLS_DATASETS[@]}"; do
    IFS=':' read -r name classical boosting <<< "$entry"
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "━━━ $name | seed=$seed ━━━"
        python train.py --config "$classical" --seed "$seed"
        python train.py --config "$boosting" --seed "$seed"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All classical multi-seed runs complete.                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
