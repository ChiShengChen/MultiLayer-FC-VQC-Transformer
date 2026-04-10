#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Ablation study: Attention contribution, FFN connectivity, LayerNorm
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate qml 2>/dev/null || true

DATASETS=(
    "boston_ablation"
    "ca_housing_ablation"
    "concrete_ablation"
    "wine_red_ablation"
    "wine_redwhite_ablation"
)

for ds in "${DATASETS[@]}"; do
    echo "══════════════════════════════════════"
    echo "  Ablation: ${ds}"
    echo "══════════════════════════════════════"
    python train.py --config "configs/${ds}.json"
done

echo ""
echo "All ablation experiments complete."
