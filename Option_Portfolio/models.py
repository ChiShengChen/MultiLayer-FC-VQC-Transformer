# models.py
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import itertools
from math import comb


class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, timesteps):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size =int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.T_max = int(timesteps) + 1
        self.fc1_list = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size) for _ in range(self.T_max)])
        self.fc2_list = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.T_max)])
        self.fc3_list = nn.ModuleList([nn.Linear(self.hidden_size, self.output_size) for _ in range(self.T_max)])

    def forward(self, input_data_):
        u = input_data_
        B, T, D = input_data_.shape
        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            u_t = torch.relu(self.fc1_list[t](u_t))
            for i in range(self.num_layers):
                u_t = torch.relu(self.fc2_list[t](u_t))
            u_t = self.fc3_list[t](u_t)
            outputs.append(u_t.unsqueeze(1))

        y = torch.stack(outputs, dim=1)                 # (B, T, output_size)
        if y.size(-1) == 1:
            y = y.squeeze(-1)                           # (B, T)
        return y.squeeze()


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
    # return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_class))


class QNN_Q3(nn.Module):
    def __init__(self, input_size, vqc_depth, output_size, num_layers, timesteps):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_qubits = 3 # 3-qubit-VQC per block
        self.vqc_depth = vqc_depth
        self.n_blocks = int(self.input_size / self.n_qubits)
        self.T_max = int(timesteps) + 1
        self.n_layers = int(num_layers)
        shots=None

        # VQC
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.theta_Q3_list = nn.ModuleList([
            nn.ModuleList([  # per time t
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, self.n_qubits, 3))
                    for _ in range(self.n_blocks)])
                for __ in range(self.n_layers)])
            for ___ in range(self.T_max)])

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size) with input_size divisible by 3.
        Applies n_layers of 3-qubit block layers per time step using self._quantum_layer.
        """
        B, T, D_in = x.shape
        assert D_in == self.input_size and self.input_size % 3 == 0
        assert self.n_qubits == 3, "Model configured for 3-qubit blocks."

        outputs_over_time = []

        for t in range(T):
            H = x[:, t, :]  # (B, D_in)

            # Layer 0
            # H = self._quantum_layer(H, self.θ_list[t][0], n_qubits=3, mode='first')
            H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[t][0], n_qubits=3, mode="first")

            # Intermediate / final layers
            for l in range(1, self.n_layers):
                H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[t][l], n_qubits=3, mode="multiple")

            outputs_over_time.append(H.unsqueeze(1))  # (B, 1, D_in)

        y = torch.cat(outputs_over_time, dim=1)        # x: (B, T, output_size)

        if y.size(-1) == 1:
            y = y.squeeze(-1)
        return y


class QNN_Q3_Parallel(nn.Module):
    def __init__(self, input_size, vqc_depth, output_size, num_layers, timesteps):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_qubits = 3 # 3-qubit-VQC per block
        self.vqc_depth = vqc_depth
        self.n_blocks = int(self.input_size / self.n_qubits)
        self.T_max = int(timesteps) + 1
        self.n_layers = int(num_layers)
        shots=None

        # VQC
        self.dev_Q3 = qml.device("default.qubit", wires=3, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_Q3_3to3 = qml.QNode(q_NtoN_Strong_function,  self.dev_Q3, interface="torch")  
        self.theta_Q3_list = nn.ModuleList([
            nn.ModuleList([  # per time t
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, self.n_qubits, 3))
                    for _ in range(self.n_blocks)])
                for __ in range(self.n_layers)])
            for ___ in range(self.T_max)])

    def _qcall_Q3_3to3(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_Q3_3to3(batch_in[i], theta[i], n_qubits)).T.float()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size) with input_size divisible by 3.
        Applies n_layers of 3-qubit block layers per time step using self._quantum_layer.
        """
        B, T, D_in = x.shape
        assert D_in == self.input_size and self.input_size % 3 == 0
        assert self.n_qubits == 3, "Model configured for 3-qubit blocks."

        outputs_over_time = []

        for t in range(T):
            H = x[:, t, :]  # (B, D_in)

            # Layer 0
            # H = self._quantum_layer(H, self.θ_list[t][0], n_qubits=3, mode='first')
            H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[t][0], n_qubits=3, mode="first")

            # Intermediate / final layers
            for l in range(1, self.n_layers):
                H = self._quantum_layer_Q3_3to3(H, thetas_layer=self.theta_Q3_list[t][l], n_qubits=3, mode="first") # 'first' to do paprallel 

            outputs_over_time.append(H.unsqueeze(1))  # (B, 1, D_in)

        y = torch.cat(outputs_over_time, dim=1)        # x: (B, T, output_size)

        if y.size(-1) == 1:
            y = y.squeeze(-1)
        return y


class QNN_QSquared(nn.Module):
    def __init__(self, input_size, vqc_depth, output_size, num_layers, timesteps):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_qubits = int(np.sqrt(self.input_size))
        self.vqc_depth = vqc_depth
        self.n_blocks = int(self.input_size / self.n_qubits)
        self.T_max = int(timesteps) + 1
        self.n_layers = int(num_layers)
        shots=None

        assert self.input_size / self.n_qubits == self.n_qubits, "Input size must equal n_qubits."
        assert self.input_size == self.output_size, "Input size must equal output size."

        # VQC
        self.dev_QN = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        # θ_list[t][l][d] -> Parameter[vqc_depth, n_qubits, 3]
        self.quantum_net_QN_NtoN = qml.QNode(q_NtoN_Strong_function,  self.dev_QN, interface="torch")  
        self.theta_QN_list = nn.ModuleList([
            nn.ModuleList([  # per time t
                nn.ParameterList([  # per layer l
                    nn.Parameter(0.01 * torch.randn(self.vqc_depth, self.n_qubits, 3))
                    for _ in range(self.n_blocks)])
                for __ in range(self.n_layers)])
            for ___ in range(self.T_max)])

    def _qcall_QN_NtoN(self, batch_in: torch.Tensor, theta: torch.Tensor, n_qubits: int) -> torch.Tensor:

        B = len(batch_in)
        outs = []
        for i in range(B):
            out = torch.stack(self.quantum_net_QN_NtoN(batch_in[i], theta[i], n_qubits)).T.float()
            outs.append(out)
        return outs                      # enforce 3 outputs per block   

    def _quantum_layer_QSquared(
        self,
        H_in: torch.Tensor,                 # (B, D) — concatenated features coming into THIS layer
        thetas_layer,                       # self.θ_list[t][l] — ParameterList of length n_blocks
        n_qubits: int = 3,
        mode: str = "first",
        ) -> torch.Tensor:

        n_blocks = self.n_blocks
        assert n_qubits == self.n_qubits, "This implementation is fixed to n-qubit blocks."
        assert n_blocks == self.n_qubits, "This implementation is fixed to n-qubit blocks."

        # Rebuild current blocks (B,n_qubits) from H_in
        blocks = [H_in[:, n_qubits * b : n_qubits * (b+1)] for b in range(n_blocks)]  

        # Build next-layer inputs (each (B,N))
        if mode == "first":
            next_inputs = blocks
        elif mode == "fully":
            next_inputs = []
            for j in range(n_qubits):
                # take column j from every block and stack along dim=1
                ni = torch.stack([blocks[i][:, j] for i in range(n_qubits)], dim=1)
                next_inputs.append(ni)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'first'|'fully'.")         

        # Run VQCs for this layer with the given thetas
        out_blocks = self._qcall_QN_NtoN(next_inputs, thetas_layer, n_qubits)
        H_out = torch.cat(out_blocks, dim=1)  # (B, n_blocks*n_qubits)
        return H_out      

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D_in = x.shape
        outputs_over_time = []

        for t in range(T):
            H = x[:, t, :]  # (B, D_in)

            # Layer 0
            # H = self._quantum_layer(H, self.θ_list[t][0], n_qubits=3, mode='first')
            H = self._quantum_layer_QSquared(H, thetas_layer=self.theta_QN_list[t][0], n_qubits=self.n_qubits, mode="first")

            # Intermediate / final layers
            for l in range(1, self.n_layers):
                H = self._quantum_layer_QSquared(H, thetas_layer=self.theta_QN_list[t][l], n_qubits=self.n_qubits, mode="fully")

            outputs_over_time.append(H.unsqueeze(1))  # (B, 1, D_in)

        y = torch.cat(outputs_over_time, dim=1)        # x: (B, T, output_size)

        if y.size(-1) == 1:
            y = y.squeeze(-1)
        return y

