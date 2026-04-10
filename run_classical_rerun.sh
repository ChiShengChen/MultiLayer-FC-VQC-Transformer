#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Re-run ALL classical baselines with fixed StandardScaler pipeline
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate qml 2>/dev/null || true

CONFIGS=(
    "boston_classical"
    "boston_boosting"
    "ca_housing_classical"
    "ca_housing_boosting"
    "concrete_classical"
    "concrete_boosting"
    "wine_red_classical"
    "wine_red_boosting"
    "wine_redwhite_classical"
    "wine_redwhite_boosting"
)

for cfg in "${CONFIGS[@]}"; do
    echo "══════════════════════════════════════"
    echo "  Classical re-run: ${cfg}"
    echo "══════════════════════════════════════"
    python train.py --config "configs/${cfg}.json"
done

echo ""
echo "All classical baselines re-run complete."
