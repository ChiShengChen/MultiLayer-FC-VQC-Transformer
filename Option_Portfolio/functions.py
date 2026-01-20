import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import Dataset
from datetime import datetime
import joblib  # For saving Tree models

# Tree Model Imports
try:
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:
    print("Warning: XGBoost or CatBoost not installed. Tree models will fail.")

import models

# --- MCSimulation and PortfolioValuation classes remain unchanged ---
class MCSimulation:
    def __init__(self, portfolio_, data_, r_, T_, steps_, N_):
        self.portfolio = portfolio_
        self.data = data_
        self.r = r_
        self.T = T_
        self.steps = steps_
        self.N = N_
        self.dt = T_ / steps_
        self.X_paths = None
        self.X_features = None

    def generate_paths(self):
        X_paths = []
        X_dW = []
        for S, q, sigma in self.data:
            dW = np.random.normal(size=(self.steps, self.N))
            increments = (self.r - q - 0.5 * sigma ** 2) * self.dt + sigma * np.sqrt(self.dt) * dW
            ST = np.full((1, self.N), np.log(S))
            ST = np.concatenate((ST, increments), axis=0)
            ST = np.cumsum(ST, axis=0)
            ST = np.exp(ST) 
            X_paths.append(ST)
            X_dW.append(dW)
        self.X_paths = np.array([np.array(xi) for xi in X_paths])
        self.X_paths = np.transpose(self.X_paths, (2, 1, 0)) 

        # Global Normalization
        mean_global = self.X_paths.mean(axis=(0, 1), keepdims=True) 
        std_global  = self.X_paths.std(axis=(0, 1), keepdims=True)
        self.X_features = (self.X_paths - mean_global) / (std_global + 1e-8)
        self.X_features = np.clip(self.X_features, -3, 3)

        self.X_dW = np.array([np.array(xi_dw) for xi_dw in X_dW])
        self.X_dW = np.transpose(X_dW, (2,1,0)) 
        return self.X_paths, self.X_features, self.X_dW

class PortfolioValuation:
    def __init__(self, price_, portfolio_, data_, r_, T_):
        self.price = price_
        self.portfolio = portfolio_
        self.data = data_
        self.r = r_
        self.dt = T_ /(price_.shape[1]-1)
        self.t = np.linspace(T_, 0, price_.shape[1]) 

    def valuation(self):
        total_value = 0
        for i, contract in enumerate(self.portfolio):
            if contract[2] == 'c':
                total_value += self.black_scholes_call(
                    self.price[:, :, contract[0] - 1], contract[1], self.t, self.r,
                    self.data[int(contract[0]) - 1][1], self.data[int(contract[0]) - 1][2])
            elif contract[2] == 'p':
                total_value += self.black_scholes_put(
                    self.price[:, :, contract[0] - 1], contract[1], self.t, self.r,
                    self.data[int(contract[0]) - 1][1], self.data[int(contract[0]) - 1][2])
            elif contract[2] == '-c':
                total_value -= self.black_scholes_call(
                    self.price[:, :, contract[0] - 1], contract[1], self.t, self.r,
                    self.data[int(contract[0]) - 1][1], self.data[int(contract[0]) - 1][2])
            elif contract[2] == '-p':
                total_value -= self.black_scholes_put(
                    self.price[:, :, contract[0] - 1], contract[1], self.t, self.r,
                    self.data[int(contract[0]) - 1][1], self.data[int(contract[0]) - 1][2])
        return total_value

    def black_scholes_call(self, S_, K_, T_, r_, q_, sigma_):
        S_no_last = S_[:, :-1]
        T_no_last = T_[:-1]
        d1 = (np.log(S_no_last / K_) + (r_ - q_ + sigma_ ** 2 / 2) * T_no_last) / (sigma_ * np.sqrt(T_no_last))
        d2 = d1 - sigma_ * np.sqrt(T_no_last)
        call_no_last = S_no_last * np.exp(-q_ * T_no_last) * norm.cdf(d1) - K_ * np.exp(-r_ * T_no_last) * norm.cdf(d2)
        last_col = np.maximum(S_[:, -1] - K_, 0).reshape(S_.shape[0], 1)
        return np.concatenate([call_no_last, last_col], axis=1)

    def black_scholes_put(self, S_, K_, T_, r_, q_, sigma_):
        S_no_last = S_[:, :-1]
        T_no_last = T_[:-1]
        d1 = (np.log(S_no_last / K_) + (r_ - q_ + sigma_ ** 2 / 2) * T_no_last) / (sigma_ * np.sqrt(T_no_last))
        d2 = d1 - sigma_ * np.sqrt(T_no_last)
        put_no_last = K_ * np.exp(-r_ * T_no_last) * norm.cdf(-d2) - S_no_last * np.exp(-q_ * T_no_last) * norm.cdf(-d1)
        last_col = np.maximum(K_ - S_[:, -1], 0).reshape(S_.shape[0], 1)
        return np.concatenate([put_no_last, last_col], axis=1)


class ModelRunner:
    def __init__(self, params, train_mode=True, directory_name=None):
        self.params = params
        self.train_mode = train_mode
        self.directory_name = directory_name
        self.device = torch.device(params.get('device', 'cpu'))
        
        self.portfolio = params['portfolio']
        self.data = params['data']
        self.r = params['r']
        self.T = params['T']
        self.num_steps = params['num_steps']
        self.num_simulations = params['num_simulations']
        self.portfolio_name = params['portfolio_name']
        
        self.input_dim = len(self.data)
        self.output_dim = len(self.portfolio)
        self.model_list = params.get('TrainingModel', ['DNN'])
        
        self.batch_size = params.get('batch_size', 64)
        self.epochs = params.get('epochs', 100)
        self.lr = params.get('learning_rate', 1e-3)
        self.num_layers = params.get('num_layers', 2)
        self.hidden_size = params.get('hidden_size', 64)
        self.vqc_depth = params.get('vqc_depth', 3)

        self.results = {}
        self.directory = None
        self.cost_histories = {} 
        self.full_histories = {}

        if self.train_mode:
            self.directory_name = self._create_directory_name()
            self._create_directory(self.directory_name)
        else:
            if self.directory_name is None: raise ValueError("directory_name required for loading.")
            self.directory = self.directory_name

        print(f"1. Generating Simulation Paths (Input Dim: {self.input_dim}, Output Dim: {self.output_dim})...")
        np.random.seed(42)
        simulator = MCSimulation(self.portfolio, self.data, self.r, self.T, self.num_steps, self.num_simulations)
        
        self.S_paths_numpy, self.S_features_numpy, _ = simulator.generate_paths()
        payoff_matrix_T = self._compute_vector_payoffs()

        # Keep Tensors for PyTorch, but we will access numpy versions for Trees
        self.train_inputs = torch.from_numpy(self.S_features_numpy).float().to(self.device)
        self.train_targets = self._generate_vector_targets(payoff_matrix_T)

        print("2. Calculating Ground Truth (Analytical)...")
        pv = PortfolioValuation(self.S_paths_numpy, self.portfolio, self.data, self.r, self.T)
        self.y_true_sum = pv.valuation()

    def _compute_vector_payoffs(self):
        ST = self.S_paths_numpy[:, -1, :]
        payoffs = np.zeros((self.num_simulations, self.output_dim))
        for i, port in enumerate(self.portfolio):
            asset_idx = int(port[0]) - 1
            K, opt_type = port[1], port[2]
            S_final = ST[:, asset_idx]
            if opt_type == 'c': payoffs[:, i] = np.maximum(S_final - K, 0)
            elif opt_type == 'p': payoffs[:, i] = np.maximum(K - S_final, 0)
            elif opt_type == '-c': payoffs[:, i] = -np.maximum(S_final - K, 0)
            elif opt_type == '-p': payoffs[:, i] = -np.maximum(K - S_final, 0)
        return payoffs

    def _generate_vector_targets(self, payoff_matrix_T):
        targets = np.zeros((self.num_simulations, self.num_steps + 1, self.output_dim))
        time_grid = np.linspace(0, self.T, self.num_steps + 1)
        for t_idx in range(self.num_steps + 1):
            t = time_grid[t_idx]
            df = np.exp(-self.r * (self.T - t))
            targets[:, t_idx, :] = payoff_matrix_T * df
        return torch.from_numpy(targets).float().to(self.device)

    def run(self):
        for model_name in self.model_list:
            if self.train_mode: self.train_single_model(model_name)
            else: self.load_predict_single_model(model_name)
        return self.results, self.directory

    def _get_model_instance(self, model_name):
        """
        Instantiates models. 
        - Tree models are created directly here.
        - PyTorch models are looked up in models.py
        """
        # 1. Handle Tree Models (Direct Instantiation)
        if 'XGBoost' in model_name:
            # Note: We use self.lr to set the learning rate from params
            return MultiOutputRegressor(XGBRegressor(
                n_estimators=self.epochs, 
                max_depth=self.vqc_depth, 
                learning_rate=self.lr, 
                n_jobs=-1
            ))
        
        if 'CatBoost' in model_name:
            return MultiOutputRegressor(CatBoostRegressor(
                iterations=self.epochs, 
                depth=self.vqc_depth, 
                learning_rate=self.lr, 
                verbose=0, 
                loss_function='RMSE',
                allow_const_label=True,      # 1. Allow constant 0 targets
                bootstrap_type='Bernoulli',  # 2. Fix: Use Bernoulli instead of MVS
                subsample=0.8                # 3. Standard sampling rate
            ))

        # 2. Handle PyTorch Models (Look up in models.py)
        if hasattr(models, model_name): 
            model_class = getattr(models, model_name)
        else: 
            raise ValueError(f"Model class '{model_name}' not found in models.py")

        if 'QNN' in model_name:
            return model_class(self.input_dim, self.vqc_depth, self.output_dim, self.num_layers, self.num_steps).to(self.device)
        elif 'DNN' in model_name:
            return model_class(self.input_dim, self.hidden_size, self.output_dim, self.num_layers, self.num_steps).to(self.device)
        else:
            return model_class(self.input_dim, self.hidden_size, self.output_dim, self.num_layers, self.num_steps).to(self.device)

    def train_single_model(self, model_name):
        print(f"\n=== Training {model_name} (Out: {self.output_dim}) ===")
        start_time = time.time()
        model = self._get_model_instance(model_name)
        
        # --- BRANCH: Check if model is PyTorch or Tree ---
        if isinstance(model, nn.Module):
            # === PYTORCH TRAINING LOOP ===
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            num_batches = self.num_simulations // self.batch_size
            cost_history = []
            history = {"epoch": [], "train_mse": [], "grad_variance": []}
            
            model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                epoch_grad_vars = []
                permutation = torch.randperm(self.num_simulations)
                for i in range(0, self.num_simulations, self.batch_size):
                    indices = permutation[i : i + self.batch_size]
                    batch_x, batch_y = self.train_inputs[indices], self.train_targets[indices]
                    
                    optimizer.zero_grad()
                    preds = model(batch_x)
                    if preds.shape != batch_y.shape: preds = preds.reshape(batch_y.shape)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    
                    all_grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                    if all_grads: epoch_grad_vars.append(torch.var(torch.cat(all_grads)).item())

                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / (num_batches + 1)
                avg_grad_var = np.mean(epoch_grad_vars) if epoch_grad_vars else 0.0
                cost_history.append(avg_loss)
                history["epoch"].append(epoch + 1)
                history["train_mse"].append(avg_loss)
                history["grad_variance"].append(avg_grad_var)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"[{model_name}] Epoch {epoch+1}/{self.epochs} | MSE: {avg_loss:.6f}")

            self.cost_histories[model_name] = cost_history
            self.full_histories[model_name] = history
            
            save_path = os.path.join(self.directory, f"{model_name}.pt")
            torch.save(model.state_dict(), save_path)
            
            # Save History CSV/Plots only for Neural Networks
            self.save_history_csv(model_name)
            self.plot_gradient_history(model_name)

        else:
            # === TREE MODEL TRAINING (sklearn/xgboost style) ===
            # 1. Flatten Data: (Sims, Steps+1, D) -> (Sims*(Steps+1), D)
            # Tree models don't take 3D tensors. We treat every (path, time) pair as a sample.
            X_flat = self.train_inputs.cpu().numpy().reshape(-1, self.input_dim)
            y_flat = self.train_targets.cpu().numpy().reshape(-1, self.output_dim)
            
            print(f"[{model_name}] Flattening data: {self.train_inputs.shape} -> {X_flat.shape}")
            print(f"[{model_name}] Fitting Tree Ensemble...")
            
            model.fit(X_flat, y_flat)
            
            save_path = os.path.join(self.directory, f"{model_name}.joblib")
            joblib.dump(model, save_path)
            
            # Dummy history for compatibility
            self.cost_histories[model_name] = [] 

        print(f"Model saved to {save_path}")
        training_time = time.time() - start_time
        self._predict_and_store(model, model_name, training_time)

    def load_predict_single_model(self, model_name):
        print(f"Loading model {model_name}...")
        
        # Check for PyTorch file (.pt) or Joblib file (.joblib)
        pt_path = os.path.join(self.directory, f"{model_name}.pt")
        joblib_path = os.path.join(self.directory, f"{model_name}.joblib")
        
        if os.path.exists(pt_path):
            model = self._get_model_instance(model_name)
            model.load_state_dict(torch.load(pt_path, map_location=self.device))
        elif os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
        else:
            print(f"Warning: Model file not found for {model_name}. Skipping.")
            return

        self._predict_and_store(model, model_name, time_taken=0)

    def _predict_and_store(self, model, model_name, time_taken):
        # Handle Prediction differences
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                preds = model(self.train_inputs)
                if preds.shape != self.train_targets.shape: preds = preds.reshape(self.train_targets.shape)
                y_pred_vector = np.maximum(preds.cpu().numpy(), 0)
        else:
            # Tree Model Prediction
            X_flat = self.train_inputs.cpu().numpy().reshape(-1, self.input_dim)
            preds_flat = model.predict(X_flat)
            # Reshape back to 3D: (Sims, Steps+1, Out)
            y_pred_vector = preds_flat.reshape(self.num_simulations, self.num_steps + 1, self.output_dim)
            y_pred_vector = np.maximum(y_pred_vector, 0)
        
        y_pred_sum = np.sum(y_pred_vector, axis=2)
        abs_mae_per_step = np.mean(np.abs(self.y_true_sum - y_pred_sum), axis=0)
        mean_true_val = np.mean(self.y_true_sum, axis=0)
        rel_mae_per_step = abs_mae_per_step / (mean_true_val + 1e-8)
        
        overall_abs_mae = np.mean(abs_mae_per_step)
        overall_rel_mae = np.mean(rel_mae_per_step)

        print(f"[{model_name}] Overall Relative MAE: {overall_rel_mae:.6f}")

        self.results[model_name] = {
            'y_pred_sum': y_pred_sum,
            'y_pred_full': y_pred_vector,
            'abs_mae_per_step': abs_mae_per_step,
            'rel_mae_per_step': rel_mae_per_step,
            'overall_abs_mae': overall_abs_mae,
            'overall_rel_mae': overall_rel_mae,
            'cost': self.cost_histories.get(model_name, []),
            'time': time_taken
        }
        if 'y_true_sum' not in self.results: self.results['y_true_sum'] = self.y_true_sum

    # --- Reporting Methods (unchanged) ---
    def save_history_csv(self, model_name=None):
        if model_name is not None or not self.full_histories: return
        print("Generating history CSV...")
        max_epochs = 0
        for h in self.full_histories.values():
            if h["epoch"]: max_epochs = max(max_epochs, max(h["epoch"]))
        combined_df = pd.DataFrame({"epoch": list(range(1, max_epochs + 1))})
        for name, history in self.full_histories.items():
            df_model = pd.DataFrame(history).rename(columns={"train_mse": f"{name}_mse", "grad_variance": f"{name}_grad_variance"})
            combined_df = pd.merge(combined_df, df_model, on="epoch", how="left")
        combined_df.to_csv(os.path.join(self.directory, "history.csv"), index=False)

    def plot_gradient_history(self, model_name=None):
        if model_name is not None or not self.full_histories: return
        plt.figure(figsize=(10, 6))
        for name, history in self.full_histories.items():
            plt.plot(history["epoch"], history["grad_variance"], label=name)
        plt.yscale("log")
        plt.title("Gradient Dynamics: All Models")
        plt.legend()
        plt.savefig(os.path.join(self.directory, "grad_variance.png"), bbox_inches="tight")
        plt.close()

    def save_summary_to_csv(self):
        y_true = self.results['y_true_sum']
        num_timesteps = y_true.shape[1]
        model_keys = [k for k in self.results.keys() if k not in ['y_true_sum']]
        summary_rows = []
        for t in range(num_timesteps):
            row = {'time_step': t, 'mean_true_portfolio_val': np.mean(y_true[:, t])}
            for m_key in model_keys:
                row[f'{m_key}_mean_val'] = np.mean(self.results[m_key]['y_pred_sum'][:, t])
                row[f'{m_key}_abs_mae'] = self.results[m_key]['abs_mae_per_step'][t]
                row[f'{m_key}_rel_mae'] = self.results[m_key]['rel_mae_per_step'][t]
            summary_rows.append(row)
        overall_row = {'time_step': 'overall', 'mean_true_portfolio_val': np.mean(y_true)}
        for m_key in model_keys:
            overall_row[f'{m_key}_mean_val'] = np.mean(self.results[m_key]['y_pred_sum'])
            overall_row[f'{m_key}_abs_mae'] = self.results[m_key]['overall_abs_mae']
            overall_row[f'{m_key}_rel_mae'] = self.results[m_key]['overall_rel_mae']
        pd.DataFrame(summary_rows + [overall_row]).to_csv(os.path.join(self.directory, "summary_results.csv"), index=False)

    def plot_relative_mae(self):
        model_keys = [k for k in self.results.keys() if k not in ['y_true_sum']]
        plt.figure(figsize=(10, 6))
        for m_key in model_keys:
            plt.plot(self.results[m_key]['rel_mae_per_step'], label=m_key, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Relative MAE')
        plt.legend()
        plt.savefig(os.path.join(self.directory, 'mae_plot.png'))
        plt.close()

    def save_and_plot_costs(self):
        if not self.cost_histories: return
        plt.figure(figsize=(10, 6))
        for name, history in self.cost_histories.items():
            if history: plt.plot(history, label=name)
        plt.title("Training Loss")
        plt.savefig(os.path.join(self.directory, "cost_plot.png"))
        plt.close()

    def save_full_predictions(self):
        pd.DataFrame(self.results['y_true_sum']).to_csv(os.path.join(self.directory, "y_true_sum.csv"), index=False)
        model_keys = [k for k in self.results.keys() if k not in ['y_true_sum']]
        for m_key in model_keys:
            pd.DataFrame(self.results[m_key]['y_pred_sum']).to_csv(os.path.join(self.directory, f"y_pred_sum_{m_key}.csv"), index=False)
        print("Predictions saved.")
    
    def _create_directory_name(self):
        return f"{self.portfolio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    def _create_directory(self, directory_name):
        if not os.path.exists(directory_name): os.makedirs(directory_name)
        self.directory = directory_name