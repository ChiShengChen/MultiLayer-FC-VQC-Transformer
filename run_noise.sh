#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Noise robustness: depolarizing noise at various strengths
# NOTE: Noisy simulation is ~5-10x slower than noiseless.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate qml 2>/dev/null || true

DATASETS=(
    "boston_noise"
    "ca_housing_noise"
    "concrete_noise"
    "wine_red_noise"
    "wine_redwhite_noise"
)

for ds in "${DATASETS[@]}"; do
    echo "══════════════════════════════════════"
    echo "  Noise study: ${ds}"
    echo "══════════════════════════════════════"
    python train.py --config "configs/${ds}.json"
done

echo ""
echo "All noise experiments complete."
