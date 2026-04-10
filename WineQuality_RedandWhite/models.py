import numpy as np
import torch
from torch import nn
import pennylane as qml
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
import torch.nn.functional as F

# ----------------------------
# Model
# ----------------------------
class CatBoostClassifierModel(nn.Module):
    def __init__(self, depth=6, iterations=1000, learning_rate=0.1):
        """
        Wrapper around CatBoostClassifier to behave like a torch.nn.Module.
        Used for classification baselines (trained via its own.fit).
        """
        super().__init__()

        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="MultiClass",
            verbose=0,
            task_type="CPU",
            allow_writing_files=False,
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
        """
        Optional torch-style forward.
        Returns class probabilities [N, num_classes] as a tensor.
        In main.py booster path you will typically call self.model.predict.
        """
        if not self.is_fitted:
            raise RuntimeError("CatBoostClassifierModel must be fitted before prediction.")

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        probs = self.model.predict_proba(x)  # list/array [N][C]
        probs = np.asarray(probs, dtype=np.float32)
        return torch.from_numpy(probs)


class XGBoostClassifierModel(nn.Module):
    def __init__(self, depth=6, n_estimators=1000, learning_rate=0.1):
        """
        Wrapper around XGBClassifier to behave like a torch.nn.Module.
        Used for classification baselines (trained via its own.fit()).
        """
        super().__init__()

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth,
            tree_method="hist",
            device="cpu",
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.is_fitted = False

    def fit(self, X, y, eval_set=None):
        # Convert Torch -> Numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if y.ndim > 1:
            y = y.ravel()

        eval_list = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy()
            if y_val.ndim > 1:
                y_val = y_val.ravel()
            eval_list = [(X_val, y_val)]

        self.model.fit(X, y, eval_set=eval_list, verbose=False)
        self.is_fitted = True

    def forward(self, x):
        """
        Optional torch-style forward.
        Returns class probabilities [N, num_classes] as a tensor.
        In main.py booster path you'll usually call self.model.predict for labels.
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoostClassifierModel must be fitted before prediction.")

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        probs = self.model.predict_proba(x)  # [N, C]
        return torch.from_numpy(probs).float()


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, layers, dropout=0.2):
        """
        Simple MLP classifier:
        - same structure as MLPRegressor, but final layer outputs n_classes logits.
        """
        super().__init__()
        self.input_layer = nn.Linear(in_dim, 64)

        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(64, 64))

        # classification head: logits for each class
        self.output_layer = nn.Linear(64, n_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.input_layer(x)))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        # return logits; ClassificationTrainer will apply CrossEntropyLoss
        H = self.output_layer(x)
        return H



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


def q_3Nto1_Strong_function(x, weights, n_class: int = 3):
    """
    x        : shape (9,) or (B, 9) -> [x0,x1,x2 | x3,x4,x5 | x6,x7,x8]
               3 angles per qubit: RX, RY, RZ.
    weights  : (L, n_qubits, 3)   (or broadcastable to that)
    n_class  : number of Z readouts (<= 3 and <= n_qubits)

    Returns  : list of n_class expvals (⟨Z0⟩, ⟨Z1⟩, ⟨Z2⟩) or batched thereof
    """
    n_qub = int(weights.shape[-2])
    assert n_qub >= 3, "Need at least 3 qubits"
    assert 1 <= n_class <= min(5, n_qub), "n_class must be 1..min(3, n_qubits)"

    # --- Encode 3 features per qubit (keep batch dim with x[..., k]) ---
    for q in range(3):
        b = 3 * q
        qml.RX(x[..., b + 0], wires=q)   # per-qubit RX
        qml.RY(x[..., b + 1], wires=q)   # per-qubit RY
        qml.RZ(x[..., b + 2], wires=q)   # per-qubit RZ

    # --- Variational body ---
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))

    # --- Readout: first n_class qubits along Z ---
    return [qml.expval(qml.PauliZ(i)) for i in range(1)]


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

class SingleVQC_12t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        n_qubits = 12
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q12 = qml.device("default.qubit", wires=12, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q12 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q12, interface="torch")  

        self.theta_Q12 = nn.ParameterList([
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, n_qubits, 3))
                    for _ in range(self.layers)])

    def forward(self, x_):
        H = x_
        # Layer: Q12 12 to 12
        for i in range(self.layers):
            H = torch.stack(self.quantum_net_Q12(H,self.theta_Q12[i], 12)).T.float()
        
        H = H[:,3:9]

        return H * self.multiplier


class FullyConnectedVQCs_12t8t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 2
        Q3_blocks = 4
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q8 = qml.device("default.qubit", wires=8, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to2 = qml.QNode(q_Nto2_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q8_8to8 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q8, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q8 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 8, 3))

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
        return outs      

    def _qcall_Q3_3to2(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to2(batch_in[i], theta[i], n_qubits)).T.float()
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

    def _quantum_layer_Q3_3to2(
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
        out_blocks = self._qcall_Q3_3to2(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        # batch_size = H.size(0)
        # zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    
        # # Adding a zero value to make it 12 values 
        # H = torch.cat([zeros_col, H], dim=1)  # (batch, 12)  

        # Layer 0: Q3 VQCs 12 to 12
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3,mode="first")  # (B,12)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+2],n_qubits=3,mode="fully")  # (B,12)

        # Q3 12 to 8
        H = self._quantum_layer_Q3_3to2(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,8) 

        # Q8 8 to 6
        H = torch.stack(self.quantum_net_Q8_8to8(H, self.theta_Q8, 8)).T.float()
        H = H[:,1:7]

        return H * self.multiplier


class FullyConnectedVQCs_24t8t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 2
        Q3_blocks = 8
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q8 = qml.device("default.qubit", wires=8, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q8_8to8 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q8, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q8 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 8, 3))

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
        # batch_size = H.size(0)
        # zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)   
        # # Adding a zero value to make it 24 values 
        # H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 24)   

        # Layer 0: Q3 VQCs 24 to 24
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3,mode="first")  # (B,24)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+2],n_qubits=3,mode="multiple")  # (B,24)

        # Q3 24 to 8
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,8) 

        # Q8 8 to 6
        H = torch.stack(self.quantum_net_Q8_8to8(H, self.theta_Q8, 8)).T.float()
        H = H[:,1:7]

        return H * self.multiplier


class FullyConnectedVQCs_36t12t8t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 12
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q8 = qml.device("default.qubit", wires=8, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to2 = qml.QNode(q_Nto2_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q8_8to8 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q8, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q8 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 8, 3))

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
        return outs      

    def _qcall_Q3_3to2(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to2(batch_in[i], theta[i], n_qubits)).T.float()
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

    def _quantum_layer_Q3_3to2(
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
        out_blocks = self._qcall_Q3_3to2(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        # batch_size = H.size(0)
        # zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)    

        # Layer 0: Q3 VQCs 36 to 36
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,36)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,36)

        # Q3 36 to 12
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,12)  

        # Q3 12 to 8
        H = self._quantum_layer_Q3_3to2(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,8) 

        # Q8 8 to 6
        H = torch.stack(self.quantum_net_Q8_8to8(H, self.theta_Q8, 8)).T.float()
        H = H[:,1:7]

        return H * self.multiplier


class FullyConnectedVQCs_48t16t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 16
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
        return outs             

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

        # Layer 0: Q3 VQCs 48 to 48
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,48)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,48)

        # Q3 48 to 16
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,16)  

        # # Adding a zero value to make it 18 values 
        H = torch.cat([zeros_col, H, zeros_col], dim=1)  # (batch, 18)  

        # Q3 18 to 6
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,6) 

        return H * self.multiplier


class FullyConnectedVQCs_60t20t7t6(nn.Module):

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 20
        shots=None

        # Devices (one is fine; we use same dev for all QNodes)
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)
        self.dev_Q7 = qml.device("default.qubit", wires=7, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to2 = qml.QNode(q_Nto2_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q3_3to1 = qml.QNode(q_Nto1_Strong_function,  self.dev_Q3, interface="torch")  
        self.quantum_net_Q7_7to7 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q7, interface="torch")  

        # ---- Trainable weights per block (shape (L, n_qubits, 3) for SEL) ----

        self.theta_Q3_list = nn.ModuleList([
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, Q3_qubits, 3))
                    for _ in range(Q3_blocks)])
                for __ in range(Q3_layers)])

        self.theta_Q7 = nn.Parameter(0.01 * torch.randn(self.vqc_depth, 7, 3))

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
        return outs      

    def _qcall_Q3_3to2(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to2(batch_in[i], theta[i], n_qubits)).T.float()
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

    def _quantum_layer_Q3_3to2(
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
        out_blocks = self._qcall_Q3_3to2(blocks, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*3)
        return H_out  

    def forward(self, x_):
        H = x_    
        batch_size = H.size(0)
        zeros_col = torch.zeros(batch_size, 1, device=H.device, dtype=H.dtype)   

        # Layer 0: Q3 VQCs 60 to 60
        H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[2],n_qubits=3,mode="first")  # (B,60)
        for i in range (self.layers):
            H = self._quantum_layer_Q3_3to3(H,thetas_layer=self.theta_Q3_list[i+3],n_qubits=3,mode="multiple")  # (B,60)

        # Q3 60 to 20
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[1],n_qubits=3)  # (B,20)  
  
        # # Adding a zero value to make it 21 values 
        H = torch.cat([zeros_col, H], dim=1)  # (batch, 21)  
        # Q3 21 to 7
        H = self._quantum_layer_Q3_3to1(H,thetas_layer=self.theta_Q3_list[0],n_qubits=3)  # (B,7)  

        # Q8 7 to 6
        H = torch.stack(self.quantum_net_Q7_7to7(H, self.theta_Q7, 7)).T.float()
        H = H[:,1:7]

        return H * self.multiplier

