#!/bin/bash
# Resume: Wine Red classical+replot, then full RedandWhite
# Usage: nohup bash run_all.sh > run_all.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
echo "=== Started at $(date) ==="

# 1. WineQuality_Red — compare already done, resume classical + replot
WR_DIR=$(ls -td outputs/*WineQuality_Red* | grep -v RedandWhite | head -1)
echo ">>> WineQuality_Red classical baselines (resume into $WR_DIR)..."
python -u train.py --config configs/wine_red_classical.json --resume-dir "$WR_DIR"
echo ">>> WineQuality_Red replot..."
python -u train.py --replot "$WR_DIR"
echo ">>> WineQuality_Red DONE at $(date)"

# 2. WineQuality_RedandWhite — full run
echo ">>> WineQuality_RedandWhite compare (5 models)..."
python -u train.py --config configs/wine_redwhite_compare.json
WRW_DIR=$(ls -td outputs/*RedandWhite* | head -1)
echo ">>> WineQuality_RedandWhite classical baselines..."
python -u train.py --config configs/wine_redwhite_classical.json --resume-dir "$WRW_DIR"
echo ">>> WineQuality_RedandWhite replot..."
python -u train.py --replot "$WRW_DIR"
echo ">>> WineQuality_RedandWhite DONE at $(date)"

echo "=== ALL DONE at $(date) ==="
