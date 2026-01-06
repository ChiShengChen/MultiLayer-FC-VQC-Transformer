# main.py

import os
import time
import numpy as np
import torch

from functions import load_classification_csv, prepare_classification_datasets, ClassificationTrainer
from models import MLPClassifier, SingleVQC_11t6, FullyConnectedVQCs_12t8t6, FullyConnectedVQCs_22t8t6, FullyConnectedVQCs_33t12t8t6, FullyConnectedVQCs_44t15t10t8t6, FullyConnectedVQCs_54t18t6, FullyConnectedVQCs_66t24t8t6, CatBoostClassifierModel, XGBoostClassifierModel

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# >>> Change these two when you switch to another regression task <<<
CSV_PATH = os.path.join(DATA_DIR, "wine-quality-red.csv")
TARGET_COLUMN = "quality"   # e.g. "Heating_Load", or any other regression target

# Batch sizes (None ⇒ full batch)
BATCH_SIZE_TRAIN = None
BATCH_SIZE_VAL = None
BATCH_SIZE_TEST = None

TRAINING_PARAMS = {
    "lr": 0.005,
    "weight_decay": 0,
    "use_scheduler": False,
    "epochs": 5000,
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
    X, y, feature_names, target_name, n_classes = load_classification_csv(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
    )
    # ----------------------------
    # 2) Prepare loaders (generic in EE_functions)
    # ----------------------------
    train_loader, val_loader, test_loader, n_features, n_classes = prepare_classification_datasets(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
        # clip_percentile = 0.0, # 0 for no clip
        # feature_range_train = (-1.0, 1.0), # determine the input value range 
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        batch_size_test=BATCH_SIZE_TEST,
    )

    # ----------------------------
    # 3) Model selection
    # ----------------------------
    # model = SingleVQC_11t6(layers=hidden_layers, depth=vqc_depth).to(device)
    model = FullyConnectedVQCs_12t8t6(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_22t8t6(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_33t12t8t6(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = FullyConnectedVQCs_44t15t10t8t6(layers=hidden_layers, depth=vqc_depth).to(device)    
    # model = FullyConnectedVQCs_54t18t6(layers=hidden_layers, depth=vqc_depth).to(device)    
    # model = FullyConnectedVQCs_66t24t8t6(layers=hidden_layers, depth=vqc_depth).to(device)
    # model = MLPClassifier(in_dim=n_features, n_classes=n_classes, layers=hidden_layers, dropout=0.0).to(device)
    # model = CatBoostClassifierModel(vqc_depth)
    # model = XGBoostClassifierModel(vqc_depth)


    model_name = model.__class__.__name__
    run_name = f"{model_name}_L{hidden_layers}_K{vqc_depth}"
    print(f"===== Running {run_name} =====")

    # ----------------------------
    # 4) Special path for CatBoost / XGBoost
    # ----------------------------
    if model_name in ["CatBoostClassifierModel", "XGBoostClassifierModel"]:
        from sklearn.metrics import accuracy_score, f1_score

        print(f"Model is {model_name}: using its own.fit instead of ClassificationTrainer.")

        # gather full train set
        X_train_list, y_train_list = [], []
        for xb, yb in train_loader:
            X_train_list.append(xb)
            y_train_list.append(yb)

        X_train = torch.cat(X_train_list, dim=0).cpu().numpy()
        y_train = torch.cat(y_train_list, dim=0).cpu().numpy()

        print(f"Fitting {model_name}...")
        start = time.time()
        model.fit(X_train, y_train)
        print(f"Fitting Time: {time.time() - start:.2f}s")

        def evaluate_booster(loader, dataset_name="Test"):
            preds, trues = [], []
            for xb, yb in loader:
                xb_np = xb.detach().cpu().numpy()
                yb_np = yb.detach().cpu().numpy()
                pred_classes = model.model.predict(xb_np)  # note: underlying.model
                preds.append(pred_classes)
                trues.append(yb_np)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            acc = accuracy_score(trues, preds)
            f1_macro = f1_score(trues, preds, average="macro")
            print(f"{dataset_name} Acc: {acc:.4f} | {dataset_name} F1(macro): {f1_macro:.4f}")
            return acc, f1_macro

        print("\n--- Booster Classification Results ---")
        evaluate_booster(train_loader, "Train")
        evaluate_booster(val_loader, "Val")
        evaluate_booster(test_loader, "Test")
        return

    # ----------------------------
    # 5) Gradient‑based training path
    # ----------------------------
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
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

    # Generate the plot on the Test Set
    test_f1 = trainer.plot_confusion_matrix(
        loader=test_loader, 
        path=f"{run_name}_confusion_matrix.png", 
        show=False
    )
    print(f"Final Test F1-Score (Macro): {test_f1:.4f}")

    # # Load pre-trained model weights
    # trainer.load_model(f"{run_name}_final_params.pt")   # make sure this file exists in the same folder
    # loss_tr, acc_tr = trainer._eval_loader(trainer.train_loader)
    # loss_va, acc_va = trainer._eval_loader(trainer.val_loader)
    # loss_te, acc_te = trainer._eval_loader(trainer.test_loader)
    # print("\n--- Loaded model performance ---")
    # print(f"Train Loss = {loss_tr:.4f}, Acc = {acc_tr:.4f}")
    # print(f"Val Loss = {loss_va:.4f}, Acc = {acc_va:.4f}")
    # print(f"Test Loss = {loss_te:.4f}, Acc = {acc_te:.4f}")


if __name__ == "__main__":
    main(3,3)
    # # Example: single combination; extend to sweeps as needed
    # for depth in range(1, 10, 2):
    #     for layers in range(1, 10, 2):
    #         main(depth, layers)