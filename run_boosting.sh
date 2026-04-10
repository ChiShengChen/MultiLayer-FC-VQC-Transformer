#!/bin/bash
# Run XGBoost + CatBoost for all 5 datasets, resume into existing output dirs
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1  # Prevent segfault from xgboost+torch OpenMP conflict on macOS
echo "=== Boosting Runs Started at $(date) ==="

# 1. BostonHousing
BH_DIR=$(ls -td outputs/*BostonHousing* | head -1)
echo ">>> BostonHousing boosting (resume into $BH_DIR)..."
python -u train.py --config configs/boston_boosting.json --resume-dir "$BH_DIR"
python -u train.py --replot "$BH_DIR"
echo ">>> BostonHousing DONE at $(date)"

# 2. CA_Housing
CA_DIR=$(ls -td outputs/*CA_Housing* | head -1)
echo ">>> CA_Housing boosting (resume into $CA_DIR)..."
python -u train.py --config configs/ca_housing_boosting.json --resume-dir "$CA_DIR"
python -u train.py --replot "$CA_DIR"
echo ">>> CA_Housing DONE at $(date)"

# 3. Concrete
CO_DIR=$(ls -td outputs/*Concrete* | head -1)
echo ">>> Concrete boosting (resume into $CO_DIR)..."
python -u train.py --config configs/concrete_boosting.json --resume-dir "$CO_DIR"
python -u train.py --replot "$CO_DIR"
echo ">>> Concrete DONE at $(date)"

# 4. WineQuality_Red
WR_DIR=$(ls -td outputs/*WineQuality_Red* | grep -v RedandWhite | head -1)
echo ">>> WineQuality_Red boosting (resume into $WR_DIR)..."
python -u train.py --config configs/wine_red_boosting.json --resume-dir "$WR_DIR"
python -u train.py --replot "$WR_DIR"
echo ">>> WineQuality_Red DONE at $(date)"

# 5. WineQuality_RedandWhite
WRW_DIR=$(ls -td outputs/*RedandWhite* | head -1)
echo ">>> WineQuality_RedandWhite boosting (resume into $WRW_DIR)..."
python -u train.py --config configs/wine_redwhite_boosting.json --resume-dir "$WRW_DIR"
python -u train.py --replot "$WRW_DIR"
echo ">>> WineQuality_RedandWhite DONE at $(date)"

echo "=== ALL Boosting Runs DONE at $(date) ==="
