#!/bin/bash
# Run multi-head transformers (n_heads=2,3) on completed datasets
# Usage: nohup bash run_transformers_mh.sh > run_transformers_mh.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
echo "=== Started at $(date) ==="

# BostonHousing (4 new transformer variants: H2, H3 × Route A, B)
BOSTON_DIR="outputs/20260224_183942_BostonHousing"
echo ">>> BostonHousing multi-head transformers..."
python -u train.py --config configs/boston_transformers_mh.json --resume-dir "$BOSTON_DIR"
echo ">>> BostonHousing replot..."
python -u train.py --replot "$BOSTON_DIR"
echo ">>> BostonHousing DONE at $(date)"

# CA_Housing (4 new transformer variants)
CA_DIR="outputs/20260225_044701_CA_Housing"
echo ">>> CA_Housing multi-head transformers..."
python -u train.py --config configs/ca_housing_transformers_mh.json --resume-dir "$CA_DIR"
echo ">>> CA_Housing replot..."
python -u train.py --replot "$CA_DIR"
echo ">>> CA_Housing DONE at $(date)"

echo "=== ALL DONE at $(date) ==="
