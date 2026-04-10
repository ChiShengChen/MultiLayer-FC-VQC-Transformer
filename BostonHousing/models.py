import numpy as np
import torch
from torch import nn
import pennylane as qml
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
import torch.nn.functional as F

# ----------------------------
# Model
# ----------------------------

class XGBoostModel(nn.Module):
    def __init__(self, depth):
        super().__init__()
        
        # Configure GPU if available
        # XGBoost uses 'cuda' for GPU support in modern versions
        # if isinstance(device, str):
        #     dev_type = device
        # else:
        #     dev_type = device.type
            
        tree_method = "hist" # Standard efficient method
        # if dev_type == 'cuda':
        #     device_arg = "cuda"
        # else:
        #     device_arg = "cpu"
        
        # print(f"Initializing XGBoost with device={device_arg}")

        self.model = XGBRegressor(
            n_estimators=10000,
            # learning_rate=0.02,
            max_depth=depth,            # Tree Depth
            # reg_lambda=3,          # L2 Regularization (like l2_leaf_reg)
            tree_method=tree_method,
            device="cpu",
            # early_stopping_rounds=50,   # Stop if validation doesn't improve
            # verbosity=0                 # Silent training
        )
        self.is_fitted = False

    def fit(self, X, y, eval_set=None):
        # Convert Torch -> Numpy
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.cpu().numpy()
        if y.ndim > 1: y = y.ravel()
            
        # Handle Validation Set for Early Stopping
        eval_list = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, torch.Tensor): X_val = X_val.cpu().numpy()
            if isinstance(y_val, torch.Tensor): y_val = y_val.cpu().numpy()
            if y_val.ndim > 1: y_val = y_val.ravel()
            eval_list = [(X_val, y_val)]

        self.model.fit(X, y, eval_set=eval_list, verbose=False)
        self.is_fitted = True

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("XGBoost must be fitted before prediction.")
        
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        preds = self.model.predict(x)
        return torch.from_numpy(preds).float().view(-1, 1)


class CatBoostModel(nn.Module):
    # Added 'l2_leaf_reg' to arguments
    def __init__(self, depth=6):
        super().__init__()
        
        # # Check device type
        # if isinstance(device, str):
        #     device_type = device
        # else:
        #     device_type = device.type
        # task_type = "GPU" if device_type == 'cuda' else "CPU"
        
        self.model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.02,
            depth=depth,                # Dynamic Depth
            l2_leaf_reg=3,    # Dynamic L2 Regularization
            loss_function='RMSE',
            verbose=0,
            task_type="CPU",
            allow_writing_files=False
        )
        self.is_fitted = False

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if y.ndim > 1:
            y = y.ravel()  
        self.model.fit(X, y)
        self.is_fitted = True

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("CatBoost must be fitted before prediction.")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        preds = self.model.predict(x)
        return torch.from_numpy(preds).float().view(-1, 1)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, layers, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, 64)

        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(64, 64))

        self.output_layer = nn.Linear(64, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.input_layer(x)))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.output_layer(x)


def q_Nto1_Strong_function(x, weights, n_class):
    """
    x        : (n_qubits,) or (batch, n_qubits)
    weights  : (L, n_qubits, 3) for StronglyEntanglingLayers
    n_class  : number of Z readouts (<= n_qubits)

    Returns: a plain Python tuple of expvals.
             PennyLane will convert it to a Torch tensor when interface="torch",
             or we'll coerce it in forward() if it doesn't.
    """
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub, "n_class cannot exceed n_qubits"

    # 1) Feature embedding
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")

    # 2) Variational body
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))

    # 3) Readout (plain tuple avoids autograd-numpy conversions)
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(1)]
    return exp_vals


def q_Nto2_Strong_function(x, weights, n_class):
    """
    x        : (n_qubits,) or (batch, n_qubits)
    weights  : (L, n_qubits, 3) for StronglyEntanglingLayers
    n_class  : number of Z readouts (<= n_qubits)

    Returns: a plain Python tuple of expvals.
             PennyLane will convert it to a Torch tensor when interface="torch",
             or we'll coerce it in forward() if it doesn't.
    """
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub, "n_class cannot exceed n_qubits"

    # 1) Feature embedding
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")

    # 2) Variational body
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))

    # 3) Readout (plain tuple avoids autograd-numpy conversions)
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(2)]
    return exp_vals


def q_NtoN_Strong_function(x, weights, n_class):
    """
    x        : (n_qubits,) or (batch, n_qubits)
    weights  : (L, n_qubits, 3) for StronglyEntanglingLayers
    n_class  : number of Z readouts (<= n_qubits)

    Returns: a plain Python tuple of expvals.
             PennyLane will convert it to a Torch tensor when interface="torch",
             or we'll coerce it in forward() if it doesn't.
    """
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub, "n_class cannot exceed n_qubits"

    # 1) Feature embedding
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")

    # 2) Variational body
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))

    # 3) Readout (plain tuple avoids autograd-numpy conversions)
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
    return exp_vals


class SingleVQC_8(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        n_qubits = 8
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q8 = qml.device("default.qubit", wires=8, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q8 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q8, interface="torch")  

        self.theta_Q8 = nn.ParameterList([
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, n_qubits, 3))
                    for _ in range(self.layers)])

    def forward(self, x_):
        H = x_
        # Layer: Q8 8 to 8
        for i in range(self.layers):
            H = torch.stack(self.quantum_net_Q8(H,self.theta_Q8[i], 8)).T.float()

        return H[:,4:5] * self.multiplier


class FullyConnectedVQCs_16t4t1_orig(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q4_qubits = 4
        Q4_layers = self.layers + 3
        Q4_blocks = 4
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q4 = qml.device("default.qubit", wires=4, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q4_4to4 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q4, interface="torch")  
        self.quantum_net_Q4_4to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q4, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q4_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q4_qubits, 3))
                    for _ in range(Q4_blocks)])
                for __ in range(Q4_layers)])

    def _qcall_Q4_4to4(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q4_4to4(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                       

    def _qcall_Q4_4to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q4_4to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                              

    def _quantum_layer_Q4_4to4(
        # Q4 4to4 here
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 4

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  

        # Build next-layer inputs (each (B,4))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 4-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            next_inputs = []
            ni = torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0], blocks[3][:, 0]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1], blocks[3][:, 1]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2], blocks[3][:, 2]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 3], blocks[1][:, 3], blocks[2][:, 3], blocks[3][:, 3]], dim=1)
            next_inputs.append(ni)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to4(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def _quantum_layer_Q4_16to4(
            # Q4 16t4 here
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 4

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  

        next_inputs = []
        ni = torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0], blocks[3][:, 0]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1], blocks[3][:, 1]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2], blocks[3][:, 2]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 3], blocks[1][:, 3], blocks[2][:, 3], blocks[3][:, 3]], dim=1)
        next_inputs.append(ni)       

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to1(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out     

    def _quantum_layer_Q4_4to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        # mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 3."
        # n_blocks = D // 4   

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to1(H_in, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out     

    def _quantum_layer_Q4_4to1(
        # Q4 4t1 here
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        # mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 4
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  # each (B,3)  

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  
        return H_out   

    def forward(self, x_):
        batch_size = x_.shape[0]
        # Create zero columns for the first and last positions
        zeros_first = torch.zeros(batch_size, 1)
        zeros_last1  = torch.zeros(batch_size, 1)
        zeros_last2  = torch.zeros(batch_size, 1)
        # Concatenate along the feature dimension (dim=1)
        x  = torch.cat([zeros_first, x_, zeros_last1, zeros_last2], dim=1)
        
        B, D = x.shape
        assert D == 16, f"Expected (B,15) input, got (B,{D})"

        H = x
        # Layer 1: Q4 4 VQCs 16 to 16
        H = self._quantum_layer_Q4_4to4(H,thetas_layer=self.theta_Q4_list[0],n_qubits=4, mode="first")  # (B,15)
        for i in range (self.layers):
            H = self._quantum_layer_Q4_4to4(H,thetas_layer=self.theta_Q4_list[i+3],n_qubits=4, mode="multiple")  # (B,15)

        # Layer 2: Q4 4 VQCs 16 to 4 
        H = self._quantum_layer_Q4_16to4(H,thetas_layer=self.theta_Q4_list[1],n_qubits=4)  # (B,15)  
        H = self._quantum_layer_Q4_4to1(H,thetas_layer=self.theta_Q4_list[2],n_qubits=4)  # (B,15)  

        return H * self.multiplier


class FullyConnectedVQCs_16t4t1(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q4_qubits = 4
        Q4_layers = self.layers + 3
        Q4_blocks = 4
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q4 = qml.device("default.qubit", wires=4, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q4_4to4 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q4, interface="torch")  
        self.quantum_net_Q4_4to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q4, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q4_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q4_qubits, 3))
                    for _ in range(Q4_blocks)])
                for __ in range(Q4_layers)])

    def _qcall_Q4_4to4(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q4_4to4(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                       

    def _qcall_Q4_4to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q4_4to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                              

    def _quantum_layer_Q4_16to16(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 4."
        n_blocks = D // 4

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  

        # Build next-layer inputs (each (B,4))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 4-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            next_inputs = []
            ni = torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0], blocks[3][:, 0]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1], blocks[3][:, 1]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2], blocks[3][:, 2]], dim=1)
            next_inputs.append(ni)
            ni = torch.stack([blocks[0][:, 3], blocks[1][:, 3], blocks[2][:, 3], blocks[3][:, 3]], dim=1)
            next_inputs.append(ni)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to4(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*4)
        return H_out  

    def _quantum_layer_Q4_16to4(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 4-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 4."
        n_blocks = D // 4

        # Rebuild current blocks (B,4) from H_in
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  

        next_inputs = []
        ni = torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0], blocks[3][:, 0]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1], blocks[3][:, 1]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2], blocks[3][:, 2]], dim=1)
        next_inputs.append(ni)
        ni = torch.stack([blocks[0][:, 3], blocks[1][:, 3], blocks[2][:, 3], blocks[3][:, 3]], dim=1)
        next_inputs.append(ni)       

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to1(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  
        return H_out     

    def _quantum_layer_Q4_4to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 4,
        ) -> torch.Tensor:

        assert n_qubits == 4, "This implementation is fixed to 4-qubit blocks."
        B, D = H_in.shape
        assert D % 4 == 0, f"Input width {D} must be divisible by 4."
        n_blocks = D // 4
        blocks = [H_in[:, 4*b:4*(b+1)] for b in range(n_blocks)]  # each (B,4)  

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q4_4to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  
        return H_out   
   

    def forward(self, x_):
        H = x_
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # Adding 3 zero valuew to make it 16 values 
        H = torch.cat([zeros_col, zeros_col, H, zeros_col], dim=1)  # (batch, 16) 

        # Layer 1: Q4 4 VQCs 16 to 16
        H = self._quantum_layer_Q4_16to16(H,thetas_layer=self.theta_Q4_list[2],n_qubits=4, mode="first")  # (B,16)
        for i in range (self.layers):
            H = self._quantum_layer_Q4_16to16(H,thetas_layer=self.theta_Q4_list[i+3],n_qubits=4, mode="multiple")  # (B,16)

        # Layer 2: Q4 4 VQCs 16 to 4 
        H = self._quantum_layer_Q4_16to4(H,thetas_layer=self.theta_Q4_list[1],n_qubits=4)  # (B,4)  
        H = self._quantum_layer_Q4_4to1(H,thetas_layer=self.theta_Q4_list[0],n_qubits=4)  # (B,1)  

        return H * self.multiplier


class FullyConnectedVQCs_15t5t1(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 2
        Q3_blocks = 5
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q5 = qml.device("default.qubit", wires=5, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q5_5to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q5, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q5 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 5, 3))

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # Generalized fully-connected: transpose-based for any n_blocks
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                B = blocks[0].shape[0]
                stacked = torch.stack(blocks, dim=1)       # (B, n_blocks, 3)
                transposed = stacked.transpose(1, 2)        # (B, 3, n_blocks)
                flat = transposed.reshape(B, -1)             # (B, 3*n_blocks)
                next_inputs = [flat[:, 3*i:3*(i+1)] for i in range(n_blocks)]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # Adding 2 zero values to make it 15 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 15)  

        # Layer 0: Q3 VQCs 15 to 15
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3,mode="first")  # (B,15)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+2],n_qubits=3,mode="fully")  # (B,15)

        # Q3 15 to 5
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,5) 

        # Q5 5 to 1
        H = torch.stack(self.quantum_net_Q5_5to1(H, self.theta_Q5, 5)).T.float()

        return H * self.multiplier


class FullyConnectedVQCs_15t5t2(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 5
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])         

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # True dense mapping for 3 blocks only: each next VQC sees ALL 9 features
            assert n_blocks == 3, "Fully-connected mode expects exactly 3 previous blocks (D=9)."
            next_inputs = [
                torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0]], dim=1),
                torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1]], dim=1),
                torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2]], dim=1),
            ]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # Adding 2 zero values to make it 15 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 15)  

        # Q3 VQCs 15 to 
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,15)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,15)

        # Q3 15 to 5
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,5) 

        # Adding a zero value to make it 6 values 
        H = torch.cat([zeros_col, H], dim=1)  # (batch, 6)  

        # Q3 6 to 2  
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,2)
        H = H[:, 0:2].mean(dim=1).unsqueeze(1)

        return H * self.multiplier


class FullyConnectedVQCs_26t9t3t1(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 4
        Q3_blocks = 9
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # True dense mapping for 3 blocks only: each next VQC sees ALL 9 features
            assert n_blocks == 3, "Fully-connected mode expects exactly 3 previous blocks (D=9)."
            next_inputs = [
                torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0]], dim=1),
                torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1]], dim=1),
                torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2]], dim=1),
            ]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # Adding a zero value to make it 27 values 
        H = torch.cat([zeros_col, H], dim=1)  # (batch, 27)  

        # Layer 0: Q3 VQCs 27 to 27
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[3],n_qubits=3,mode="first")  # (B,27)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+4],n_qubits=3,mode="multiple")  # (B,27)

        # Q3 27 to 9
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3)  # (B,9) 
 
        # Q3 9 to 3
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,3) 
        # Q3 3 to 1
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,1) 

        return H * self.multiplier


class FullyConnectedVQCs_39t15t5t1(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 13
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q5 = qml.device("default.qubit", wires=5, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q5_5to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q5, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q5 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 5, 3))

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # True dense mapping for 3 blocks only: each next VQC sees ALL 9 features
            assert n_blocks == 3, "Fully-connected mode expects exactly 3 previous blocks (D=9)."
            next_inputs = [
                torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0]], dim=1),
                torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1]], dim=1),
                torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2]], dim=1),
            ]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    

        # Layer 0: Q3 VQCs 39 to 39
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,39)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,39)

        # Q3 39 to 13
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,13) 

        # Adding 2 zero values to make it 15 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 15)  

        # Q3 15 to 5
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,5) 

        # Q5 5 to 1
        H = torch.stack(self.quantum_net_Q5_5to1(H, self.theta_Q5, 5)).T.float()

        return H * self.multiplier




class FullyConnectedVQCs_52t18t6t2(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 4
        Q3_blocks = 18
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])         

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # True dense mapping for 3 blocks only: each next VQC sees ALL 9 features
            assert n_blocks == 3, "Fully-connected mode expects exactly 3 previous blocks (D=9)."
            next_inputs = [
                torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0]], dim=1),
                torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1]], dim=1),
                torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2]], dim=1),
            ]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # Adding 2 zero values to make it 15 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 54)  

        # Q3 VQCs 54 to 54
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[3],n_qubits=3,mode="first")  # (B,54)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+4],n_qubits=3,mode="multiple")  # (B,54)

        # Q3 54 to 18
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3)  # (B,18) 
        # Q3 18 to 6
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,6)  
        # Q3 6 to 2  
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,2)
        H = H[:, 0:2].mean(dim=1).unsqueeze(1)

        return H * self.multiplier
    


class FullyConnectedVQCs_52t18t6t1(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 18
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q6 = qml.device("default.qubit", wires=6, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q6_6to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q6, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q6 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 6, 3))

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block      

    def _qcall_Q3_3to1(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to1(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block               

    def _quantum_layer_Q3_3to3(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3)

        # Build next-layer inputs (each (B,3))
        if mode == "first":
            # For the first layer, the "next inputs" are literally the current 3-wide chunks.
            next_inputs = blocks
        elif mode == "multiple":
            if n_blocks == 1:
                next_inputs = [blocks[0]]
            else:
                next_inputs = []
                for i in range(n_blocks):
                    if i == 0:
                        # [block0[1], block0[2], block1[0]]
                        ni = torch.stack([blocks[n_blocks-1][:, 2], blocks[0][:, 1], blocks[1][:, 0]], dim=1)
                    elif i == n_blocks - 1:
                        # [block(n-2)[2], block(n-1)[0], block(n-1)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[0][:, 0]], dim=1)
                    else:
                        # [block(i-1)[2], block(i)[0], block(i)[1]]
                        ni = torch.stack([blocks[i-1][:, 2], blocks[i][:, 1], blocks[i+1][:, 0]], dim=1)
                    next_inputs.append(ni)
        elif mode == "fully":
            # True dense mapping for 3 blocks only: each next VQC sees ALL 9 features
            assert n_blocks == 3, "Fully-connected mode expects exactly 3 previous blocks (D=9)."
            next_inputs = [
                torch.stack([blocks[0][:, 0], blocks[1][:, 0], blocks[2][:, 0]], dim=1),
                torch.stack([blocks[0][:, 1], blocks[1][:, 1], blocks[2][:, 1]], dim=1),
                torch.stack([blocks[0][:, 2], blocks[1][:, 2], blocks[2][:, 2]], dim=1),
            ]
   
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'multiple'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out      

    def _quantum_layer_Q3_3to1(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        assert n_qubits == 3, "This implementation is fixed to 3-qubit blocks."
        B, D = H_in.shape
        assert D % 3 == 0, f"Input width {D} must be divisible by 3."
        n_blocks = D // 3

        # Rebuild current blocks (B,3) from H_in
        blocks = [H_in[:, 3*b:3*(b+1)] for b in range(n_blocks)]  # each (B,3) 

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_Q3_3to1(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)  
        # Adding 2 zero values to make it 15 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 54)    

        # Layer 0: Q3 VQCs 54 to 54
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,54)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,54)

        # Q3 54 to 18
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,18) 

        # Q3 18 to 6
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,6) 

        # Q5 6 to 1
        H = torch.stack(self.quantum_net_Q6_6to1(H, self.theta_Q6, 6)).T.float()

        return H * self.multiplier

# ============================================================
# Hybrid Quantum Transformer (Route A)
# ============================================================

class QuantumTransformerVQC(nn.Module):
    """
    Hybrid Quantum Transformer using VQC blocks as Q/K/V projections and FFN.

    Maps the FC-VQC architecture to a transformer:
    - Tokens : 5 blocks of 3 features each (Q3 VQCs)
    - Q/K/V : Independent VQC per token ("first" connectivity)
    - Attn  : Classical softmax(Q·K^T / sqrt(d)) → weighted V
    - FFN   : Independent VQC per token (position-wise)
    - Skip  : Residual connection + LayerNorm after each sub-layer

    Architecture:
        Input(13) → pad to 15 → 5 tokens × dim 3
        → [Transformer Layer × N]:
              Self-Attention(Q/K/V via Q3 VQCs, classical softmax)
              + Residual + LayerNorm
              FFN(Q3 VQCs, position-wise)
              + Residual + LayerNorm
        → Reduce: Q3 15→5 (3to1)
        → Final : Q5 5→1
        → × (π − ε)
    """

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi - eps

        Q3_qubits = 3
        n_tokens = 5
        shots = None

        # Quantum devices
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q5 = qml.device("default.qubit", wires=5, shots=shots)

        # QNodes (stateless — params passed as args)
        self.qnode_3to3 = qml.QNode(q_NtoN_Strong_function, self.dev_Q3, interface="torch")
        self.qnode_3to1 = qml.QNode(q_Nto1_Strong_function, self.dev_Q3, interface="torch")
        self.qnode_5to1 = qml.QNode(q_Nto1_Strong_function, self.dev_Q5, interface="torch")

        # === Per Transformer Layer: Q, K, V projections + FFN ===
        self.theta_Q = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        self.theta_K = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        self.theta_V = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        self.theta_FFN = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        # LayerNorm after attention and FFN
        self.attn_norm = nn.ModuleList([nn.LayerNorm(n_tokens * Q3_qubits) for _ in range(layers)])
        self.ffn_norm  = nn.ModuleList([nn.LayerNorm(n_tokens * Q3_qubits) for _ in range(layers)])

        # === Readout ===
        self.theta_reduce = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
            for _ in range(n_tokens)])

        self.theta_final = nn.Parameter(0.01 * torch.randn(depth, 5, 3))

    # ----------------------------------------------------------------
    # QNode call helpers (same pattern as existing FC-VQC models)
    # ----------------------------------------------------------------
    def _qcall_3to3(self, blocks, thetas):
        """Run Q3 NtoN VQC on each block. blocks: list of (B,3), thetas: ParameterList."""
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to3(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_3to1(self, blocks, thetas):
        """Run Q3 Nto1 VQC on each block. blocks: list of (B,3), thetas: ParameterList."""
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to1(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    # ----------------------------------------------------------------
    # Transformer sub-layers
    # ----------------------------------------------------------------
    def _project(self, H, thetas):
        """Q/K/V projection: (B, 15) → (B, 5, 3) via 5 independent Q3 VQCs."""
        blocks = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        out_blocks = self._qcall_3to3(blocks, thetas)  # 5 × (B, 3)
        return torch.stack(out_blocks, dim=1)           # (B, 5, 3)

    def _ffn(self, H, thetas):
        """Position-wise FFN: (B, 15) → (B, 15) via 5 independent Q3 VQCs."""
        blocks = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        out_blocks = self._qcall_3to3(blocks, thetas)  # 5 × (B, 3)
        return torch.cat(out_blocks, dim=1)              # (B, 15)

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, x_):
        B = x_.shape[0]

        # Pad 13 → 15
        zeros_col = torch.zeros(B, 1, device=x_.device, dtype=x_.dtype)
        H = torch.cat([zeros_col, x_, zeros_col], dim=1)  # (B, 15)

        # === Transformer Layers ===
        for l in range(self.layers):
            # --- Self-Attention ---
            Q = self._project(H, self.theta_Q[l])   # (B, 5, 3)
            K = self._project(H, self.theta_K[l])   # (B, 5, 3)
            V = self._project(H, self.theta_V[l])   # (B, 5, 3)

            # Attention: softmax(Q·K^T / √d) · V
            scores = torch.bmm(Q, K.transpose(1, 2)) / (3.0 ** 0.5)  # (B, 5, 5)
            attn_weights = F.softmax(scores, dim=-1)                   # (B, 5, 5)
            attn_out = torch.bmm(attn_weights, V)                     # (B, 5, 3)
            attn_out = attn_out.reshape(B, 15)                         # (B, 15)

            H = self.attn_norm[l](H + attn_out)  # residual + norm

            # --- Feed-Forward ---
            ffn_out = self._ffn(H, self.theta_FFN[l])  # (B, 15)
            H = self.ffn_norm[l](H + ffn_out)          # residual + norm

        # === Readout ===
        # Q3 15→5 (3to1)
        blocks = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        reduce_outs = self._qcall_3to1(blocks, self.theta_reduce)  # 5 × (B, 1)
        H = torch.cat(reduce_outs, dim=1)                          # (B, 5)

        # Q5 5→1
        H = torch.stack(self.qnode_5to1(H, self.theta_final, 5)).T.float()  # (B, 1)

        return H * self.multiplier


# ============================================================
# Full Quantum Transformer (Route B) — no softmax, no LayerNorm
# ============================================================

class FullQuantumTransformerVQC(nn.Module):
    """
    Full Quantum Transformer — all operations are quantum circuits.
    No classical softmax, no LayerNorm.

    Quantum Attention:
        Transpose the token matrix (5 tokens × 3 dim → 3 groups × 5 tokens)
        → 3 Q5 VQCs with StronglyEntanglingLayers (cross-token entanglement)
        → Transpose back (3 groups × 5 → 5 tokens × 3)
        + Residual

    Quantum FFN (FC-VQC):
        5 Q3 VQCs with "multiple" (circular-shift) connectivity
        + Residual

    Architecture:
        Input(13) → pad to 15 = 5 tokens × dim 3
        → Stem: 5 Q3 VQCs (mode="first", per-token feature extraction)
        → [Full Quantum Transformer Layer × N]:
              Quantum Attention  (Q5, cross-token) + Residual
              Quantum FFN        (Q3, FC-VQC "multiple") + Residual
        → Reduce: Q3 15→5  (3to1)
        → Final : Q5 5→1
        → × (π − ε)
    """

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi - eps

        Q3_qubits = 3
        Q5_qubits = 5
        n_tokens  = 5       # 15 features / 3 qubits
        n_groups  = 3       # feature dim per token
        shots = None

        # Quantum devices
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q5 = qml.device("default.qubit", wires=5, shots=shots)

        # QNodes (stateless — params passed as args)
        self.qnode_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")
        self.qnode_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")
        self.qnode_Q5_5to5 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q5, interface="torch")
        self.qnode_Q5_5to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q5, interface="torch")

        # === Stem: 5 Q3 VQCs (mode="first") ===
        self.theta_stem = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
            for _ in range(n_tokens)])

        # === Per Transformer Layer ===
        # Quantum Attention: 3 Q5 VQCs (one per feature-dimension group)
        self.theta_attn = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q5_qubits, 3))
                for _ in range(n_groups)])
            for _ in range(layers)])

        # Quantum FFN: 5 Q3 VQCs ("multiple" connectivity)
        self.theta_ffn = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        # === Readout ===
        self.theta_reduce = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, Q3_qubits, 3))
            for _ in range(n_tokens)])

        self.theta_final = nn.Parameter(0.01 * torch.randn(depth, 5, 3))

    # ----------------------------------------------------------------
    # QNode call helpers
    # ----------------------------------------------------------------
    def _qcall_Q5_5to5(self, groups, thetas):
        """Run Q5 NtoN VQC on each group. groups: list of (B,5)."""
        outs = []
        for i in range(len(groups)):
            out = torch.stack(self.qnode_Q5_5to5(groups[i], thetas[i], 5)).T.float()
            outs.append(out)
        return outs

    def _qcall_Q3_3to3(self, blocks, thetas):
        """Run Q3 NtoN VQC on each block. blocks: list of (B,3)."""
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_Q3_3to3(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_Q3_3to1(self, blocks, thetas):
        """Run Q3 Nto1 VQC on each block. blocks: list of (B,3)."""
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_Q3_3to1(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    # ----------------------------------------------------------------
    # Quantum Attention: transpose → Q5 VQCs → transpose back
    # ----------------------------------------------------------------
    def _quantum_attention(self, H, thetas):
        """
        Cross-token entanglement via transposed Q5 VQCs.

        (B,15) → reshape (B,5,3) → transpose (B,3,5)
        → 3 Q5 VQCs (each sees one feature from ALL 5 tokens)
        → transpose back (B,5,3) → flatten (B,15)

        The StronglyEntanglingLayers inside each Q5 VQC create
        data-dependent entanglement across all 5 tokens — this IS
        the quantum analogue of self-attention.
        """
        B = H.shape[0]
        tokens     = H.reshape(B, 5, 3)              # (B, 5, 3)
        transposed = tokens.transpose(1, 2)           # (B, 3, 5)

        groups   = [transposed[:, g, :] for g in range(3)]  # 3 × (B, 5)
        attended = self._qcall_Q5_5to5(groups, thetas)       # 3 × (B, 5)

        result = torch.stack(attended, dim=1)         # (B, 3, 5)
        result = result.transpose(1, 2)                # (B, 5, 3)
        return result.reshape(B, 15)                   # (B, 15)

    # ----------------------------------------------------------------
    # Quantum FFN: FC-VQC with "multiple" (circular-shift) connectivity
    # ----------------------------------------------------------------
    def _quantum_ffn(self, H, thetas):
        """
        FC-VQC feed-forward with circular-shift inter-block connectivity.

        Each output block[i] receives:
          [blocks[i-1][:,2],  blocks[i][:,1],  blocks[i+1][:,0]]
        (wrapping at boundaries)
        """
        blocks   = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        n_blocks = 5

        next_inputs = []
        for i in range(n_blocks):
            if i == 0:
                ni = torch.stack([blocks[n_blocks-1][:, 2],
                                  blocks[0][:, 1],
                                  blocks[1][:, 0]], dim=1)
            elif i == n_blocks - 1:
                ni = torch.stack([blocks[i-1][:, 2],
                                  blocks[i][:, 1],
                                  blocks[0][:, 0]], dim=1)
            else:
                ni = torch.stack([blocks[i-1][:, 2],
                                  blocks[i][:, 1],
                                  blocks[i+1][:, 0]], dim=1)
            next_inputs.append(ni)

        out_blocks = self._qcall_Q3_3to3(next_inputs, thetas)  # 5 × (B, 3)
        return torch.cat(out_blocks, dim=1)                     # (B, 15)

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, x_):
        B = x_.shape[0]

        # Pad 13 → 15
        zeros_col = torch.zeros(B, 1, device=x_.device, dtype=x_.dtype)
        H = torch.cat([zeros_col, x_, zeros_col], dim=1)  # (B, 15)

        # Stem: independent Q3 VQCs (mode="first")
        blocks    = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        stem_outs = self._qcall_Q3_3to3(blocks, self.theta_stem)  # 5 × (B, 3)
        H = torch.cat(stem_outs, dim=1)                          # (B, 15)

        # Full Quantum Transformer layers
        for l in range(self.layers):
            H = H + self._quantum_attention(H, self.theta_attn[l])  # + residual
            H = H + self._quantum_ffn(H, self.theta_ffn[l])         # + residual

        # Readout: Q3 15→5 (3to1)
        blocks      = [H[:, 3*b : 3*(b+1)] for b in range(5)]
        reduce_outs = self._qcall_Q3_3to1(blocks, self.theta_reduce)  # 5 × (B, 1)
        H = torch.cat(reduce_outs, dim=1)                            # (B, 5)

        # Final: Q5 5→1
        H = torch.stack(self.qnode_Q5_5to1(H, self.theta_final, 5)).T.float()

        return H * self.multiplier
