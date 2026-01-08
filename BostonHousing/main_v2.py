# main.py

import os
import time
import numpy as np
import torch

from functions import load_regression_csv, prepare_datasets, Trainer
from models_v2 import MLPRegressor, FullyConnectedVQCs_16t4t1, FullyConnectedVQCs_16t4t1_orig, FullyConnectedVQCs_15t5t1, FullyConnectedVQCs_15t5t2, FullyConnectedVQCs_26t9t3t1, FullyConnectedVQCs_39t13t5t1, FullyConnectedVQCs_52t18t6t1, FullyConnectedVQCs_52t18t6t2, CatBoostModel, XGBoostModel

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# >>> Change these two when you switch to another regression task <<<
CSV_PATH = os.path.join(DATA_DIR, "boston_housing.csv")
TARGET_COLUMN = "MEDV"   # e.g. "Heating_Load", or any other regression target

# Batch sizes (None ⇒ full batch)
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

    # ----------------------------
    # 1) Load data (generic CSV loader in EE_functions)
    # ----------------------------
    X, y, feature_names, target_name = load_regression_csv(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
    )

    # ----------------------------
    # 2) Prepare loaders (generic in EE_functions)
    # ----------------------------
    train_loader, val_loader, test_loader, y_scaler, n_features = prepare_datasets(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
        clip_percentile = 0.04, # 0 for no clip
        # feature_range_train = (-range, range), # determine the input value range 
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        batch_size_test=BATCH_SIZE_TEST,
    )

    # ----------------------------
    # 3) Model selection
    # ----------------------------
    model = FullyConnectedVQCs_16t4t1(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_16t4t1_orig(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_15t5t1(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_15t5t2(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_26t9t3t1(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_39t13t5t1(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_52t18t6t1(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_52t18t6t2(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = SingleVQC_8(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = MLPRegressor(in_dim=n_features, layers=hidden_layers, dropout=0.0).to(device)
    # model = CatBoostModel(vqc_depth)
    # model = XGBoostModel(vqc_depth)

    model_name = model.__class__.__name__
    run_name = f"{model_name}_L{hidden_layers}_K{vqc_depth}"
    print(f"===== Running {run_name} =====")

    # ----------------------------
    # 4) Special path for CatBoost / XGBoost
    # ----------------------------
    if model_name in ["CatBoostModel", "XGBoostModel"]:
        from sklearn.metrics import r2_score, mean_squared_error

        print(f"Model is {model_name}: using its own.fit instead of Trainer.")

        # Get full train set as dense tensors
        X_train_list, y_train_list = [], []
        for xb, yb in train_loader:
            X_train_list.append(xb)
            y_train_list.append(yb)

        X_train = torch.cat(X_train_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)

        print(f"Fitting {model_name}...")
        start = time.time()
        model.fit(X_train, y_train)
        print(f"Fitting Time: {time.time() - start:.2f}s")

        def evaluate_booster(loader, dataset_name="Test"):
            preds, trues = [], []
            for xb, yb in loader:
                pred = model(xb)
                preds.append(pred)
                trues.append(yb)

            preds = torch.cat(preds).detach().cpu().numpy()
            trues = torch.cat(trues).detach().cpu().numpy()

            preds_inv = y_scaler.inverse_transform(preds).ravel()
            trues_inv = y_scaler.inverse_transform(trues).ravel()

            r2 = r2_score(trues_inv, preds_inv)
            rmse = np.sqrt(mean_squared_error(trues_inv, preds_inv))
            print(f"{dataset_name} R2: {r2:.4f} | {dataset_name} RMSE: {rmse:.4f}")
            return r2, rmse

        print("\n--- Booster Results ---")
        evaluate_booster(train_loader, "Train")
        evaluate_booster(val_loader, "Val")
        evaluate_booster(test_loader, "Test")
        return

    # ----------------------------
    # 5) Gradient‑based training path
    # ----------------------------
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

    # after training is done
    trainer.save_model(f"{run_name}_final_params.pt")

    # Save histories and plots
    trainer.save_history_csv(f"{run_name}_history.csv")
    trainer.save_stats_csv(f"{run_name}_stats.csv")
    trainer.plot_history(f"{run_name}_history.png", show=False)
    trainer.plot_gradient_history(f"{run_name}_gradients.png", show=False)

    # # Load pre-trained model weights
    # trainer.load_model(f"{run_name}_final_params.pt")   # make sure this file exists in the same folder
    # r2_tr, rmse_tr, mae_tr = trainer._eval_loader(trainer.train_loader)
    # r2_va, rmse_va, mae_va = trainer._eval_loader(trainer.val_loader)
    # r2_te, rmse_te, mae_te = trainer._eval_loader(trainer.test_loader)
    # print("\n--- Loaded model performance ---")
    # print(f"Train: R2={r2_tr:.4f}, RMSE={rmse_tr:.4f}, MAE={mae_tr:.4f}")
    # print(f"Val  : R2={r2_va:.4f}, RMSE={rmse_va:.4f}, MAE={mae_va:.4f}")
    # print(f"Test : R2={r2_te:.4f}, RMSE={rmse_te:.4f}, MAE={mae_te:.4f}")


if __name__ == "__main__": 
    main(3,3)

    # # Example: single combination; extend to sweeps as needed
    # for depth in range(1, 10, 2):
    #     for layers in range(1, 10, 2):
    #         main(depth, layers)