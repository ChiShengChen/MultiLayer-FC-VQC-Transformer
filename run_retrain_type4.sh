#!/bin/bash
# Retrain all quantum models affected by Type 3→Type 4 migration
# Usage: nohup bash run_retrain_type4.sh > run_retrain_type4.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
echo "=== Type 4 Retrain Started at $(date) ==="

# --- Output directories ---
BOSTON_DIR="outputs/20260224_183942_BostonHousing"
CA_DIR="outputs/20260225_044701_CA_Housing"
CONCRETE_DIR="outputs/20260225_133933_Concrete"
WINERED_DIR="outputs/20260225_144947_WineQuality_Red"
WINERW_DIR="outputs/20260226_030631_WineQuality_RedandWhite"

# --- Step 1: Delete old model subdirs that need retraining ---
echo ">>> Removing old model dirs to force retraining..."

# BostonHousing: FC-VQC + ResNet + Transformers (H1)
rm -rf "$BOSTON_DIR/FullyConnectedVQCs_15t5t1_L3_K3"
rm -rf "$BOSTON_DIR/ResNetVQC_15t5t1_L3_K3"
rm -rf "$BOSTON_DIR/QuantumTransformerVQC_L2_K3"
rm -rf "$BOSTON_DIR/FullQuantumTransformerVQC_L2_K3"
# Multi-head
rm -rf "$BOSTON_DIR/QuantumTransformerVQC_L2_K3_H2"
rm -rf "$BOSTON_DIR/QuantumTransformerVQC_L2_K3_H3"
rm -rf "$BOSTON_DIR/FullQuantumTransformerVQC_L2_K3_H2"
rm -rf "$BOSTON_DIR/FullQuantumTransformerVQC_L2_K3_H3"

# CA_Housing: ResNet + Transformers (H1)
rm -rf "$CA_DIR/ResNetVQC_L3_K3"
rm -rf "$CA_DIR/QuantumTransformerVQC_L2_K3"
rm -rf "$CA_DIR/FullQuantumTransformerVQC_L2_K3"
# Multi-head
rm -rf "$CA_DIR/QuantumTransformerVQC_L2_K3_H2"
rm -rf "$CA_DIR/QuantumTransformerVQC_L2_K3_H3"
rm -rf "$CA_DIR/FullQuantumTransformerVQC_L2_K3_H2"
rm -rf "$CA_DIR/FullQuantumTransformerVQC_L2_K3_H3"

# Concrete: ResNet + Transformers (H1)
rm -rf "$CONCRETE_DIR/ResNetVQC_L3_K3"
rm -rf "$CONCRETE_DIR/QuantumTransformerVQC_L2_K3"
rm -rf "$CONCRETE_DIR/FullQuantumTransformerVQC_L2_K3"
# Multi-head
rm -rf "$CONCRETE_DIR/QuantumTransformerVQC_L2_K3_H2"
rm -rf "$CONCRETE_DIR/QuantumTransformerVQC_L2_K3_H3"
rm -rf "$CONCRETE_DIR/FullQuantumTransformerVQC_L2_K3_H2"
rm -rf "$CONCRETE_DIR/FullQuantumTransformerVQC_L2_K3_H3"

# WineRed: FC-VQC + ResNet + Transformers (H1)
rm -rf "$WINERED_DIR/FullyConnectedVQCs_12t8t6_L3_K3"
rm -rf "$WINERED_DIR/ResNetVQC_L3_K3"
rm -rf "$WINERED_DIR/QuantumTransformerVQC_L2_K3"
rm -rf "$WINERED_DIR/FullQuantumTransformerVQC_L2_K3"
# Multi-head
rm -rf "$WINERED_DIR/QuantumTransformerVQC_L2_K3_H2"
rm -rf "$WINERED_DIR/QuantumTransformerVQC_L2_K3_H3"
rm -rf "$WINERED_DIR/FullQuantumTransformerVQC_L2_K3_H2"
rm -rf "$WINERED_DIR/FullQuantumTransformerVQC_L2_K3_H3"

# WineRW: FC-VQC + ResNet + Transformers (H1)
rm -rf "$WINERW_DIR/FullyConnectedVQCs_12t8t6_L3_K3"
rm -rf "$WINERW_DIR/ResNetVQC_L3_K3"
rm -rf "$WINERW_DIR/QuantumTransformerVQC_L2_K3"
rm -rf "$WINERW_DIR/FullQuantumTransformerVQC_L2_K3"
# Multi-head
rm -rf "$WINERW_DIR/QuantumTransformerVQC_L2_K3_H2"
rm -rf "$WINERW_DIR/QuantumTransformerVQC_L2_K3_H3"
rm -rf "$WINERW_DIR/FullQuantumTransformerVQC_L2_K3_H2"
rm -rf "$WINERW_DIR/FullQuantumTransformerVQC_L2_K3_H3"

echo ">>> Old model dirs removed."

# --- Step 2: Retrain base models (H1) ---
echo ""
echo ">>> [1/10] BostonHousing base models..."
python -u train.py --config configs/boston_retrain_type4.json --resume-dir "$BOSTON_DIR"
echo ">>> BostonHousing base DONE at $(date)"

echo ""
echo ">>> [2/10] CA_Housing base models..."
python -u train.py --config configs/ca_housing_retrain_type4.json --resume-dir "$CA_DIR"
echo ">>> CA_Housing base DONE at $(date)"

echo ""
echo ">>> [3/10] Concrete base models..."
python -u train.py --config configs/concrete_retrain_type4.json --resume-dir "$CONCRETE_DIR"
echo ">>> Concrete base DONE at $(date)"

echo ""
echo ">>> [4/10] WineQuality_Red base models..."
python -u train.py --config configs/wine_red_retrain_type4.json --resume-dir "$WINERED_DIR"
echo ">>> WineRed base DONE at $(date)"

echo ""
echo ">>> [5/10] WineQuality_RedandWhite base models..."
python -u train.py --config configs/wine_redwhite_retrain_type4.json --resume-dir "$WINERW_DIR"
echo ">>> WineRW base DONE at $(date)"

# --- Step 3: Retrain multi-head transformers ---
echo ""
echo ">>> [6/10] BostonHousing multi-head transformers..."
python -u train.py --config configs/boston_transformers_mh.json --resume-dir "$BOSTON_DIR"
echo ">>> BostonHousing MH DONE at $(date)"

echo ""
echo ">>> [7/10] CA_Housing multi-head transformers..."
python -u train.py --config configs/ca_housing_transformers_mh.json --resume-dir "$CA_DIR"
echo ">>> CA_Housing MH DONE at $(date)"

echo ""
echo ">>> [8/10] Concrete multi-head transformers..."
python -u train.py --config configs/concrete_transformers_mh.json --resume-dir "$CONCRETE_DIR"
echo ">>> Concrete MH DONE at $(date)"

echo ""
echo ">>> [9/10] WineQuality_Red multi-head transformers..."
python -u train.py --config configs/wine_red_transformers_mh.json --resume-dir "$WINERED_DIR"
echo ">>> WineRed MH DONE at $(date)"

echo ""
echo ">>> [10/10] WineQuality_RedandWhite multi-head transformers..."
python -u train.py --config configs/wine_redwhite_transformers_mh.json --resume-dir "$WINERW_DIR"
echo ">>> WineRW MH DONE at $(date)"

# --- Step 4: Replot all ---
echo ""
echo ">>> Replotting all datasets..."
python -u train.py --replot "$BOSTON_DIR"
python -u train.py --replot "$CA_DIR"
python -u train.py --replot "$CONCRETE_DIR"
python -u train.py --replot "$WINERED_DIR"
python -u train.py --replot "$WINERW_DIR"

echo ""
echo "=== ALL DONE at $(date) ==="
