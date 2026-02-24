# main_transformer.py
# Entry point for Quantum Transformer models on BostonHousing

import os
import time
import numpy as np
import torch

from functions import load_regression_csv, prepare_datasets, Trainer
from models import QuantumTransformerVQC, FullQuantumTransformerVQC

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "boston_housing.csv")
TARGET_COLUMN = "MEDV"

# Batch sizes (None => full batch)
BATCH_SIZE_TRAIN = None
BATCH_SIZE_VAL = None
BATCH_SIZE_TEST = None

TRAINING_PARAMS = {
    "lr": 0.005,
    "weight_decay": 0.0,
    "use_scheduler": False,
    "epochs": 10000,
    "print_every": 100,
    "eval_interval": 100,
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(vqc_depth: int, hidden_layers: int):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1) Load data
    X, y, feature_names, target_name = load_regression_csv(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
    )

    # 2) Prepare loaders
    train_loader, val_loader, test_loader, y_scaler, n_features = prepare_datasets(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
        clip_percentile=0.04,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        batch_size_test=BATCH_SIZE_TEST,
    )

    # 3) Model selection
    # model = QuantumTransformerVQC(layers=hidden_layers, depth=vqc_depth).to(device)   # Route A: Hybrid
    model = FullQuantumTransformerVQC(layers=hidden_layers, depth=vqc_depth).to(device)  # Route B: Full Quantum

    model_name = model.__class__.__name__
    run_name = f"{model_name}_L{hidden_layers}_K{vqc_depth}"
    print(f"===== Running {run_name} =====")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params}, Trainable: {n_trainable}")

    # 4) Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        y_scaler=y_scaler,
        device=device,
        run_name=run_name,
        **TRAINING_PARAMS,
    )

    start = time.time()
    trainer.train()
    print("Running Time:", time.time() - start)

    # 5) Save
    trainer.save_model(f"{run_name}_final_params.pt")
    trainer.save_history_csv(f"{run_name}_history.csv")
    trainer.save_stats_csv(f"{run_name}_stats.csv")
    trainer.plot_history(f"{run_name}_history.png", show=False)
    trainer.plot_gradient_history(f"{run_name}_gradients.png", show=False)


if __name__ == "__main__":
    # depth=3, layers=2
    main(3, 2)
