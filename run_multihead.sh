#!/bin/bash
# Run multi-head Transformer variants (H2, H3) for all 5 datasets
# Resume into existing output dirs so comparison plots merge all models
# Usage: nohup bash run_multihead.sh > run_multihead.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
echo "=== Multi-Head Runs Started at $(date) ==="

# 1. BostonHousing
BH_DIR=$(ls -td outputs/*BostonHousing* | head -1)
echo ">>> BostonHousing multi-head (resume into $BH_DIR)..."
python -u train.py --config configs/boston_transformers_mh.json --resume-dir "$BH_DIR"
echo ">>> BostonHousing replot..."
python -u train.py --replot "$BH_DIR"
echo ">>> BostonHousing DONE at $(date)"

# 2. CA_Housing
CA_DIR=$(ls -td outputs/*CA_Housing* | head -1)
echo ">>> CA_Housing multi-head (resume into $CA_DIR)..."
python -u train.py --config configs/ca_housing_transformers_mh.json --resume-dir "$CA_DIR"
echo ">>> CA_Housing replot..."
python -u train.py --replot "$CA_DIR"
echo ">>> CA_Housing DONE at $(date)"

# 3. Concrete
CO_DIR=$(ls -td outputs/*Concrete* | head -1)
echo ">>> Concrete multi-head (resume into $CO_DIR)..."
python -u train.py --config configs/concrete_transformers_mh.json --resume-dir "$CO_DIR"
echo ">>> Concrete replot..."
python -u train.py --replot "$CO_DIR"
echo ">>> Concrete DONE at $(date)"

# 4. WineQuality_Red
WR_DIR=$(ls -td outputs/*WineQuality_Red* | grep -v RedandWhite | head -1)
echo ">>> WineQuality_Red multi-head (resume into $WR_DIR)..."
python -u train.py --config configs/wine_red_transformers_mh.json --resume-dir "$WR_DIR"
echo ">>> WineQuality_Red replot..."
python -u train.py --replot "$WR_DIR"
echo ">>> WineQuality_Red DONE at $(date)"

# 5. WineQuality_RedandWhite
WRW_DIR=$(ls -td outputs/*RedandWhite* | head -1)
echo ">>> WineQuality_RedandWhite multi-head (resume into $WRW_DIR)..."
python -u train.py --config configs/wine_redwhite_transformers_mh.json --resume-dir "$WRW_DIR"
echo ">>> WineQuality_RedandWhite replot..."
python -u train.py --replot "$WRW_DIR"
echo ">>> WineQuality_RedandWhite DONE at $(date)"

echo "=== ALL Multi-Head Runs DONE at $(date) ==="
