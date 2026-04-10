# Multi-Layer Fully-Connected VQCs

Quantum machine learning research code exploring **multi-layer fully-connected
Variational Quantum Circuits (VQCs)**, **residual (ResNet-style) VQCs**, and
**Quantum Transformer VQCs** on tabular regression and classification
benchmarks.

## Architectures

All models are implemented in per-experiment `models.py` files and the shared
[shared_models.py](shared_models.py):

- **Fully-Connected VQCs** (`FullyConnectedVQCs_*`) — Multi-layer FC-VQCs with
  configurable qubit layouts (e.g. `16t4t1`, `15t5t2`, `52t18t6t1`).
- **ResNet VQC** — VQC with residual / skip connections
  ([shared_models.py](shared_models.py): `ResNetVQC`).
- **Quantum Transformer VQCs** — Two routes:
  - `QuantumTransformerVQC` (Route A, hybrid classical-quantum attention)
  - `FullQuantumTransformerVQC` (Route B, fully quantum attention)
- **Classical baselines** ([classical_models.py](classical_models.py)) —
  MLP, XGBoost, CatBoost, LightGBM, Random Forest, etc.

## Datasets / Experiments

Each experiment lives in its own directory and can be driven from the
config-based training entry point:

| Directory | Task | Target |
|-----------|------|--------|
| [BostonHousing/](BostonHousing/) | regression | MEDV |
| [CA_Housing/](CA_Housing/) | regression | MedHouseVal |
| [Concrete/](Concrete/) | regression | compressive strength |
| [WineQuality_Red/](WineQuality_Red/) | classification | quality |
| [WineQuality_RedandWhite/](WineQuality_RedandWhite/) | classification | quality |
| [Option_Portfolio/](Option_Portfolio/) | regression | portfolio value |

## Usage

### Config-driven training

[train.py](train.py) is a config-driven wrapper that runs one or more models
on a dataset, performs best-checkpointing by validation loss, and writes a
timestamped run directory under `outputs/` with metrics CSV, prediction-vs-GT
plots, and training-curve overlays.

```bash
# Compare multiple models in one run
python train.py --config configs/boston_compare.json

# Run a single ResNet VQC with overrides
python train.py --config configs/boston_resnet.json --depth 5 --layers 7

# Depth/layer sweep
python train.py --config configs/boston_resnet.json --sweep

# List available models in an experiment module
python train.py --list-models BostonHousing
python train.py --list-models BostonHousing --module models_resnet
```

Configs live in [configs/](configs/) and are organized by `<dataset>_<kind>.json`,
where `<kind>` is one of: `compare`, `ablation`, `boosting`, `classical`,
`noise`, `resnet`, `retrain_type4`, `transformers`, `transformers_mh`.

### Batch experiment scripts

- [run_all.sh](run_all.sh) — full comparison across datasets
- [run_ablation.sh](run_ablation.sh) — depth / layer ablations
- [run_boosting.sh](run_boosting.sh) — gradient-boosting baselines
- [run_classical_rerun.sh](run_classical_rerun.sh) — classical baseline reruns
- [run_noise.sh](run_noise.sh) — noise-robustness experiments
- [run_multihead.sh](run_multihead.sh) / [run_transformers_mh.sh](run_transformers_mh.sh) —
  multi-head Quantum Transformer sweeps
- [run_retrain_type4.sh](run_retrain_type4.sh) — Type-4 VQC retraining

### Analysis utilities

- [expressibility_analysis.py](expressibility_analysis.py) — expressibility
  measurement for the VQC ansätze (see
  [expressibility_results.png](expressibility_results.png))
- [summarize_results.py](summarize_results.py) — aggregate run metrics across
  `outputs/` into summary tables
- [tables.tex](tables.tex) — LaTeX tables for the paper

## Project layout

```
.
├── train.py                  # config-driven training entry point
├── shared_models.py          # ResNet VQC + Quantum Transformer VQCs
├── classical_models.py       # classical baselines (MLP / XGB / CatBoost / ...)
├── summarize_results.py      # cross-run metric aggregation
├── expressibility_analysis.py
├── configs/                  # per-experiment JSON configs
├── BostonHousing/            # dataset-specific models + legacy main.py
├── CA_Housing/
├── Concrete/
├── WineQuality_Red/
├── WineQuality_RedandWhite/
├── Option_Portfolio/
└── run_*.sh                  # batch experiment drivers
```

## Requirements

The code uses PyTorch, PennyLane (for quantum circuit simulation), scikit-learn,
pandas, matplotlib, and the boosting libraries (`xgboost`, `catboost`,
`lightgbm`). Install via your preferred environment manager.

## Notes

- Run outputs (`outputs/`) and cache directories are git-ignored.
- Experiment modules retain their original `main.py` / `main_v2.py` /
  `main_transformer.py` entry points; [train.py](train.py) wraps them without
  modifying them.
