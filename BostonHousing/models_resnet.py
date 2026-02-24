import numpy as np
import torch
from torch import nn
import pennylane as qml


# ----------------------------
# Quantum Functions
# ----------------------------

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


# ----------------------------
# ResNet VQC Models
# ----------------------------

class ResNetVQC_15t5t1(nn.Module):
    """
    ResNet-style VQC for 13-feature datasets (e.g. BostonHousing).
    Same architecture as FullyConnectedVQCs_15t5t1 but with residual/skip
    connections in the hidden layers:
        H_new = VQC(H, mode="multiple") + H
    
    Architecture:
        Input(13) → pad to 15 → Stem(15→15, first) 
        → [Residual Block × layers: H = VQC(H, multiple) + H]
        → Reduce(15→5, 3to1) → Final(5→1)
        → Output × (π-ε)
    """

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

        # Stem: Q3 VQCs 15→15 (mode="first")
        H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[0], n_qubits=3, mode="first")  # (B,15)

        # Residual blocks with skip connections
        for i in range(self.layers):
            H_res = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[i+1], n_qubits=3, mode="multiple")  # (B,15)
            H = H_res + H  # ← residual/skip connection

        # Reduce: Q3 15→5
        H = self._quantum_layer_Q3_3to1(H, thetas_layer=self.theta_Q3_list[self.layers+1], n_qubits=3)  # (B,5) 

        # Final: Q5 5→1
        H = torch.stack(self.quantum_net_Q5_5to1(H, self.theta_Q5, 5)).T.float()

        return H * self.multiplier


class ResNetVQC_9t3t1(nn.Module):
    """
    ResNet-style VQC for 8-feature datasets (e.g. Concrete, CA_Housing).
    Same concept as FullyConnectedVQCs with residual/skip connections:
        H_new = VQC(H, mode="fully") + H
    
    Architecture:
        Input(8) → pad to 9 → Stem(9→9, first) 
        → [Residual Block × layers: H = VQC(H, fully) + H]
        → Reduce(9→3, 3to1) → Reduce(3→1, 3to1)
        → Output × (π-ε)
    """

    def __init__(self, layers, depth):
        super().__init__()
        self.layers = layers
        self.vqc_depth = depth
        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi-eps
        Q3_qubits = 3
        Q3_layers = self.layers + 3
        Q3_blocks = 3
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
        # Adding 1 zero value to make it 9 values 
        H = torch.cat([zeros_col, H], dim=1)  # (batch, 9)  

        # Stem: Q3 VQCs 9→9 (mode="first")
        H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[0], n_qubits=3, mode="first")  # (B,9)

        # Residual blocks with skip connections
        for i in range(self.layers):
            H_res = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[i+1], n_qubits=3, mode="fully")  # (B,9)
            H = H_res + H  # ← residual/skip connection

        # Reduce: Q3 9→3
        H = self._quantum_layer_Q3_3to1(H, thetas_layer=self.theta_Q3_list[self.layers+1], n_qubits=3)  # (B,3) 

        # Reduce: Q3 3→1
        H = self._quantum_layer_Q3_3to1(H, thetas_layer=self.theta_Q3_list[self.layers+2], n_qubits=3)  # (B,1) 

        return H * self.multiplier
