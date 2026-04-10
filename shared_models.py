"""
Generalized quantum models that adapt to any n_features / n_classes.

Provides:
  - ResNetVQC(n_features, layers, depth, n_classes=None)
  - QuantumTransformerVQC(n_features, layers, depth, n_classes=None)
  - FullQuantumTransformerVQC(n_features, layers, depth, n_classes=None)

Regression (n_classes=None): output scalar × (π-ε)
Classification (n_classes>0): quantum backbone + Linear(n_tokens, n_classes) head
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pennylane as qml


# ═══════════════════════════════════════════════════════════════════════
# Quantum Circuit Functions
# ═══════════════════════════════════════════════════════════════════════

def q_NtoN_Strong_function(x, weights, n_class):
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]


def q_Nto1_Strong_function(x, weights, n_class):
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qub))
    return [qml.expval(qml.PauliZ(0))]


# ── Noisy circuit variants (depolarizing channel after each layer) ──

def q_NtoN_Strong_noisy(x, weights, n_class, noise_strength=0.01):
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    n_layers = weights.shape[0]
    for layer_idx in range(n_layers):
        qml.StronglyEntanglingLayers(
            weights[layer_idx:layer_idx+1], wires=range(n_qub))
        for w in range(n_qub):
            qml.DepolarizingChannel(noise_strength, wires=w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]


def q_Nto1_Strong_noisy(x, weights, n_class, noise_strength=0.01):
    n_qub = int(weights.shape[-2])
    assert n_class <= n_qub
    qml.AngleEmbedding(x, wires=range(n_qub), rotation="Y")
    n_layers = weights.shape[0]
    for layer_idx in range(n_layers):
        qml.StronglyEntanglingLayers(
            weights[layer_idx:layer_idx+1], wires=range(n_qub))
        for w in range(n_qub):
            qml.DepolarizingChannel(noise_strength, wires=w)
    return [qml.expval(qml.PauliZ(0))]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _compute_padding(n_features):
    """Return (padded_size, n_tokens, pad_left, pad_right)."""
    n_tokens = math.ceil(n_features / 3)
    padded = n_tokens * 3
    total_pad = padded - n_features
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return padded, n_tokens, pad_left, pad_right


def _pad_input(x, pad_left, pad_right):
    """Pad input tensor with zeros: (B, n_features) → (B, padded)."""
    B = x.shape[0]
    parts = []
    if pad_left > 0:
        parts.append(torch.zeros(B, pad_left, device=x.device, dtype=x.dtype))
    parts.append(x)
    if pad_right > 0:
        parts.append(torch.zeros(B, pad_right, device=x.device, dtype=x.dtype))
    return torch.cat(parts, dim=1)


def _make_qnodes(n_wires, noise_strength=0.0):
    """Build QNode pair (NtoN, Nto1) — noisy if noise_strength > 0."""
    if noise_strength > 0:
        dev = qml.device("default.mixed", wires=n_wires, shots=None)
        import functools
        nton_fn = functools.partial(q_NtoN_Strong_noisy,
                                     noise_strength=noise_strength)
        nto1_fn = functools.partial(q_Nto1_Strong_noisy,
                                     noise_strength=noise_strength)
    else:
        dev = qml.device("default.qubit", wires=n_wires, shots=None)
        nton_fn = q_NtoN_Strong_function
        nto1_fn = q_Nto1_Strong_function
    qnode_nton = qml.QNode(nton_fn, dev, interface="torch")
    qnode_nto1 = qml.QNode(nto1_fn, dev, interface="torch")
    return dev, qnode_nton, qnode_nto1


def _build_multiple_inputs(blocks, n_blocks):
    """Circular-shift connectivity for 'multiple' mode."""
    if n_blocks == 1:
        return [blocks[0]]
    next_inputs = []
    for i in range(n_blocks):
        prev_idx = (i - 1) % n_blocks
        next_idx = (i + 1) % n_blocks
        ni = torch.stack([blocks[prev_idx][:, 2],
                          blocks[i][:, 1],
                          blocks[next_idx][:, 0]], dim=1)
        next_inputs.append(ni)
    return next_inputs


def _build_fully_inputs(blocks):
    """Fully-connected (transpose) mode for any number of blocks.

    Stacks all blocks into (B, n_blocks, 3), transposes to (B, 3, n_blocks),
    then flattens and re-chunks into n_blocks groups of 3.
    Each output VQC sees qubits from multiple different blocks.
    For n_blocks=3 this is identical to the original per-position transpose.
    """
    n_blocks = len(blocks)
    if n_blocks == 1:
        return [blocks[0]]
    B = blocks[0].shape[0]
    stacked = torch.stack(blocks, dim=1)          # (B, n_blocks, 3)
    transposed = stacked.transpose(1, 2)            # (B, 3, n_blocks)
    flat = transposed.reshape(B, -1)                 # (B, 3 * n_blocks)
    return [flat[:, 3*i:3*(i+1)] for i in range(n_blocks)]


# ═══════════════════════════════════════════════════════════════════════
# ResNetVQC — Generalized
# ═══════════════════════════════════════════════════════════════════════

class ResNetVQC(nn.Module):
    """
    Generalized ResNet-style VQC for any feature count.

    Architecture:
        Input(n_features) → pad to n_tokens*3
        → Stem(Q3, mode="first")
        → [Residual Block × layers]: H = Q3_VQC(H, mode) + H
        → Reduce: Q3 3to1 → n_tokens values
        → Final: Q{n_tokens}→1 (regression) or Linear (classification)
    """

    def __init__(self, n_features, layers, depth, n_classes=None,
                 noise_strength=0.0):
        super().__init__()
        self.n_features = n_features
        self.layers = layers
        self.vqc_depth = depth
        self.n_classes = n_classes
        self.noise_strength = noise_strength

        padded, n_tokens, pad_left, pad_right = _compute_padding(n_features)
        self.padded = padded
        self.n_tokens = n_tokens
        self.pad_left = pad_left
        self.pad_right = pad_right

        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi - eps

        # Residual mode — always fully-connected (Type 4)
        self.res_mode = "fully"

        # Number of Q3 layer groups:
        #   stem(1) + residual(layers) + reduce steps
        if n_tokens <= 3:
            n_reduce = 2  # two 3to1 steps: n_tokens*3→n_tokens→? then 3→1
            # Actually: 9→3 (3to1 per token), then 3→1 (one 3to1)
        else:
            n_reduce = 1  # one 3to1 step, then Q{n_tokens}→1
        self.n_reduce = n_reduce
        n_q3_layers = 1 + layers + n_reduce

        # Quantum devices (noisy if noise_strength > 0)
        self.dev_Q3, self.qnode_3to3, self.qnode_3to1 = _make_qnodes(
            3, noise_strength)

        # Q3 parameters: [layer_group][block] → (depth, 3, 3)
        blocks_per_layer = []
        blocks_per_layer.append(n_tokens)  # stem
        for _ in range(layers):
            blocks_per_layer.append(n_tokens)
        # Reduce layers
        blocks_per_layer.append(n_tokens)  # first reduce: n_tokens*3 → n_tokens
        if n_reduce == 2:
            # n_tokens must be 3 here (since n_tokens <= 3 and we pad to multiple of 3)
            blocks_per_layer.append(3)  # second reduce: 3 → 1 (but actually it's a single 3to1)

        self.theta_Q3_list = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                for _ in range(nb)])
            for nb in blocks_per_layer])

        # Final readout (only for n_tokens > 3, regression)
        if n_tokens > 3 and n_classes is None:
            self.dev_final, _, self.qnode_final = _make_qnodes(
                n_tokens, noise_strength)
            self.theta_final = nn.Parameter(0.01 * torch.randn(depth, n_tokens, 3))

        # Classification head
        if n_classes is not None:
            self.cls_head = nn.Linear(n_tokens, n_classes)

    def _qcall_3to3(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to3(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_3to1(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to1(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _layer_3to3(self, H, thetas, mode="first"):
        blocks = [H[:, 3*b:3*(b+1)] for b in range(H.shape[1] // 3)]
        if mode == "first":
            inputs = blocks
        elif mode == "multiple":
            inputs = _build_multiple_inputs(blocks, len(blocks))
        elif mode == "fully":
            inputs = _build_fully_inputs(blocks)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        out_blocks = self._qcall_3to3(inputs, thetas)
        return torch.cat(out_blocks, dim=1)

    def _layer_3to1(self, H, thetas):
        n_blocks = H.shape[1] // 3
        blocks = [H[:, 3*b:3*(b+1)] for b in range(n_blocks)]
        out_blocks = self._qcall_3to1(blocks, thetas)
        return torch.cat(out_blocks, dim=1)

    def forward(self, x_):
        H = _pad_input(x_, self.pad_left, self.pad_right)
        layer_idx = 0

        # Stem
        H = self._layer_3to3(H, self.theta_Q3_list[layer_idx], mode="first")
        layer_idx += 1

        # Residual blocks
        for _ in range(self.layers):
            H_res = self._layer_3to3(H, self.theta_Q3_list[layer_idx], mode=self.res_mode)
            H = H_res + H
            layer_idx += 1

        # Reduce: n_tokens*3 → n_tokens
        H = self._layer_3to1(H, self.theta_Q3_list[layer_idx])
        layer_idx += 1

        if self.n_reduce == 2:
            # Second reduce: 3 → 1
            H = self._layer_3to1(H, self.theta_Q3_list[layer_idx])
            layer_idx += 1

        # Output
        if self.n_classes is not None:
            return self.cls_head(H)
        elif self.n_tokens > 3:
            H = torch.stack(self.qnode_final(H, self.theta_final, self.n_tokens)).T.float()
            return H * self.multiplier
        else:
            # n_tokens <= 3, already reduced to 1
            return H * self.multiplier


# ═══════════════════════════════════════════════════════════════════════
# QuantumTransformerVQC — Hybrid (Generalized)
# ═══════════════════════════════════════════════════════════════════════

class QuantumTransformerVQC(nn.Module):
    """
    Generalized Hybrid Quantum Transformer (multi-head support).

    Architecture:
        Input(n_features) → pad to n_tokens*3
        → [Transformer Layer × layers]:
              Multi-Head: n_heads × (Q/K/V via n_tokens Q3 VQCs)
              Classical softmax attention per head
              Concat heads → Linear projection
              + Residual + LayerNorm
              FFN via Q3 VQCs (circular-shift)
              + Residual + LayerNorm
        → Reduce: Q3 3to1 → n_tokens
        → Q{n_tokens}→1 (regression) or Linear (classification)
    """

    def __init__(self, n_features, layers, depth, n_classes=None, n_heads=1,
                 ffn_mode="fully", use_attention=True, noise_strength=0.0):
        super().__init__()
        self.n_features = n_features
        self.layers = layers
        self.vqc_depth = depth
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.ffn_mode = ffn_mode          # "fully" (Type 4) or "multiple" (Type 3)
        self.use_attention = use_attention  # False → FFN-only ablation
        self.noise_strength = noise_strength

        padded, n_tokens, pad_left, pad_right = _compute_padding(n_features)
        self.padded = padded
        self.n_tokens = n_tokens
        self.pad_left = pad_left
        self.pad_right = pad_right

        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi - eps

        # Quantum devices (noisy if noise_strength > 0)
        self.dev_Q3, self.qnode_3to3, self.qnode_3to1 = _make_qnodes(
            3, noise_strength)

        # Per-layer Q, K, V projections (only if attention is used)
        if use_attention:
            if n_heads == 1:
                self.theta_Q = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                        for _ in range(n_tokens)])
                    for _ in range(layers)])
                self.theta_K = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                        for _ in range(n_tokens)])
                    for _ in range(layers)])
                self.theta_V = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                        for _ in range(n_tokens)])
                    for _ in range(layers)])
            else:
                self.theta_Q = nn.ModuleList([
                    nn.ModuleList([
                        nn.ParameterList([
                            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                            for _ in range(n_tokens)])
                        for _ in range(n_heads)])
                    for _ in range(layers)])
                self.theta_K = nn.ModuleList([
                    nn.ModuleList([
                        nn.ParameterList([
                            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                            for _ in range(n_tokens)])
                        for _ in range(n_heads)])
                    for _ in range(layers)])
                self.theta_V = nn.ModuleList([
                    nn.ModuleList([
                        nn.ParameterList([
                            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                            for _ in range(n_tokens)])
                        for _ in range(n_heads)])
                    for _ in range(layers)])

            # Multi-head output projection
            if n_heads > 1:
                self.W_O = nn.ModuleList([
                    nn.Linear(padded * n_heads, padded, bias=False)
                    for _ in range(layers)])

            # Attention LayerNorm
            self.attn_norm = nn.ModuleList([nn.LayerNorm(padded) for _ in range(layers)])

        # Per-layer: FFN projections (shared across heads)
        self.theta_FFN = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        # FFN LayerNorm
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(padded) for _ in range(layers)])

        # Readout
        self.theta_reduce = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
            for _ in range(n_tokens)])

        if n_classes is None:
            self.dev_final, _, self.qnode_final = _make_qnodes(
                n_tokens, noise_strength)
            self.theta_final = nn.Parameter(0.01 * torch.randn(depth, n_tokens, 3))
        else:
            self.cls_head = nn.Linear(n_tokens, n_classes)

    def _qcall_3to3(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to3(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_3to1(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to1(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _project(self, H, thetas):
        """Q/K/V projection: (B, padded) → (B, n_tokens, 3)."""
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        out_blocks = self._qcall_3to3(blocks, thetas)
        return torch.stack(out_blocks, dim=1)  # (B, n_tokens, 3)

    def _ffn(self, H, thetas):
        """FFN with configurable connectivity: (B, padded) → (B, padded)."""
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        if self.ffn_mode == "fully":
            inputs = _build_fully_inputs(blocks)
        elif self.ffn_mode == "multiple":
            inputs = _build_multiple_inputs(blocks, len(blocks))
        else:
            inputs = blocks  # "first" — independent per-token
        out_blocks = self._qcall_3to3(inputs, thetas)
        return torch.cat(out_blocks, dim=1)

    def forward(self, x_):
        B = x_.shape[0]
        H = _pad_input(x_, self.pad_left, self.pad_right)

        for l in range(self.layers):
            if self.use_attention:
                # Multi-Head Self-Attention
                head_outs = []
                for h in range(self.n_heads):
                    th_Q = self.theta_Q[l] if self.n_heads == 1 else self.theta_Q[l][h]
                    th_K = self.theta_K[l] if self.n_heads == 1 else self.theta_K[l][h]
                    th_V = self.theta_V[l] if self.n_heads == 1 else self.theta_V[l][h]
                    Q = self._project(H, th_Q)  # (B, n_tokens, 3)
                    K = self._project(H, th_K)
                    V = self._project(H, th_V)

                    scores = torch.bmm(Q, K.transpose(1, 2)) / (3.0 ** 0.5)
                    attn_weights = F.softmax(scores, dim=-1)
                    head_outs.append(torch.bmm(attn_weights, V).reshape(B, self.padded))

                if self.n_heads > 1:
                    attn_out = self.W_O[l](torch.cat(head_outs, dim=-1))
                else:
                    attn_out = head_outs[0]

                H = self.attn_norm[l](H + attn_out)

            # FFN
            ffn_out = self._ffn(H, self.theta_FFN[l])
            H = self.ffn_norm[l](H + ffn_out)

        # Reduce: Q3 3to1 per token → n_tokens values
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        reduce_outs = self._qcall_3to1(blocks, self.theta_reduce)
        H = torch.cat(reduce_outs, dim=1)  # (B, n_tokens)

        if self.n_classes is not None:
            return self.cls_head(H)
        else:
            H = torch.stack(self.qnode_final(H, self.theta_final, self.n_tokens)).T.float()
            return H * self.multiplier


# ═══════════════════════════════════════════════════════════════════════
# FullQuantumTransformerVQC — Generalized
# ═══════════════════════════════════════════════════════════════════════

class FullQuantumTransformerVQC(nn.Module):
    """
    Generalized Full Quantum Transformer (multi-head support).

    Architecture:
        Input(n_features) → pad to n_tokens*3
        → Stem: n_tokens Q3 VQCs (mode="first")
        → [Layer × layers]:
              Multi-Head Quantum Attention: n_heads × (transpose → 3 Q{n_tokens} VQCs → transpose back)
              Concat heads → Linear projection + residual
              Quantum FFN: Q3 VQCs (circular-shift) + residual
        → Reduce: Q3 3to1 → n_tokens
        → Q{n_tokens}→1 (regression) or Linear (classification)
    """

    def __init__(self, n_features, layers, depth, n_classes=None, n_heads=1,
                 ffn_mode="fully", use_attention=True, use_layernorm=False,
                 noise_strength=0.0):
        super().__init__()
        self.n_features = n_features
        self.layers = layers
        self.vqc_depth = depth
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.ffn_mode = ffn_mode          # "fully" (Type 4) or "multiple" (Type 3)
        self.use_attention = use_attention  # False → FFN-only ablation
        self.use_layernorm = use_layernorm  # True → add LayerNorm (ablation)
        self.noise_strength = noise_strength

        padded, n_tokens, pad_left, pad_right = _compute_padding(n_features)
        self.padded = padded
        self.n_tokens = n_tokens
        self.pad_left = pad_left
        self.pad_right = pad_right

        eps = np.finfo(np.float32).eps
        self.multiplier = np.pi - eps

        # Quantum devices (noisy if noise_strength > 0)
        self.dev_Q3, self.qnode_3to3, self.qnode_3to1 = _make_qnodes(
            3, noise_strength)
        self.dev_Qnt, self.qnode_Nt_NtoN, self.qnode_Nt_Nto1 = _make_qnodes(
            n_tokens, noise_strength)

        # Stem: n_tokens Q3 VQCs
        self.theta_stem = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
            for _ in range(n_tokens)])

        # Per-layer Quantum Attention params (only if attention is used)
        if use_attention:
            if n_heads == 1:
                self.theta_attn = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(0.01 * torch.randn(depth, n_tokens, 3))
                        for _ in range(3)])
                    for _ in range(layers)])
            else:
                self.theta_attn = nn.ModuleList([
                    nn.ModuleList([
                        nn.ParameterList([
                            nn.Parameter(0.01 * torch.randn(depth, n_tokens, 3))
                            for _ in range(3)])
                        for _ in range(n_heads)])
                    for _ in range(layers)])

            # Multi-head output projection
            if n_heads > 1:
                self.W_O = nn.ModuleList([
                    nn.Linear(padded * n_heads, padded, bias=False)
                    for _ in range(layers)])

        self.theta_ffn = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(0.01 * torch.randn(depth, 3, 3))
                for _ in range(n_tokens)])
            for _ in range(layers)])

        # Optional LayerNorm (ablation study)
        if use_layernorm:
            if use_attention:
                self.attn_norm = nn.ModuleList([nn.LayerNorm(padded) for _ in range(layers)])
            self.ffn_norm = nn.ModuleList([nn.LayerNorm(padded) for _ in range(layers)])

        # Readout
        self.theta_reduce = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(depth, 3, 3))
            for _ in range(n_tokens)])

        if n_classes is None:
            self.theta_final = nn.Parameter(0.01 * torch.randn(depth, n_tokens, 3))
        else:
            self.cls_head = nn.Linear(n_tokens, n_classes)

    def _qcall_3to3(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to3(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_3to1(self, blocks, thetas):
        outs = []
        for i in range(len(blocks)):
            out = torch.stack(self.qnode_3to1(blocks[i], thetas[i], 3)).T.float()
            outs.append(out)
        return outs

    def _qcall_Nt_NtoN(self, groups, thetas):
        outs = []
        for i in range(len(groups)):
            out = torch.stack(self.qnode_Nt_NtoN(groups[i], thetas[i], self.n_tokens)).T.float()
            outs.append(out)
        return outs

    def _quantum_attention(self, H, thetas):
        """Cross-token entanglement via transposed Q{n_tokens} VQCs."""
        B = H.shape[0]
        tokens = H.reshape(B, self.n_tokens, 3)      # (B, n_tokens, 3)
        transposed = tokens.transpose(1, 2)            # (B, 3, n_tokens)

        groups = [transposed[:, g, :] for g in range(3)]  # 3 × (B, n_tokens)
        attended = self._qcall_Nt_NtoN(groups, thetas)     # 3 × (B, n_tokens)

        result = torch.stack(attended, dim=1)          # (B, 3, n_tokens)
        result = result.transpose(1, 2)                 # (B, n_tokens, 3)
        return result.reshape(B, self.padded)           # (B, padded)

    def _quantum_ffn(self, H, thetas):
        """FC-VQC FFN with configurable connectivity."""
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        if self.ffn_mode == "fully":
            inputs = _build_fully_inputs(blocks)
        elif self.ffn_mode == "multiple":
            inputs = _build_multiple_inputs(blocks, len(blocks))
        else:
            inputs = blocks
        out_blocks = self._qcall_3to3(inputs, thetas)
        return torch.cat(out_blocks, dim=1)

    def forward(self, x_):
        B = x_.shape[0]
        H = _pad_input(x_, self.pad_left, self.pad_right)

        # Stem
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        stem_outs = self._qcall_3to3(blocks, self.theta_stem)
        H = torch.cat(stem_outs, dim=1)

        # Transformer layers
        for l in range(self.layers):
            if self.use_attention:
                # Multi-Head Quantum Attention
                head_outs = []
                for h in range(self.n_heads):
                    th = self.theta_attn[l] if self.n_heads == 1 else self.theta_attn[l][h]
                    head_outs.append(self._quantum_attention(H, th))

                if self.n_heads > 1:
                    attn_out = self.W_O[l](torch.cat(head_outs, dim=-1))
                else:
                    attn_out = head_outs[0]

                H = H + attn_out  # residual
                if self.use_layernorm:
                    H = self.attn_norm[l](H)

            ffn_out = self._quantum_ffn(H, self.theta_ffn[l])
            H = H + ffn_out  # residual
            if self.use_layernorm:
                H = self.ffn_norm[l](H)

        # Reduce: Q3 3to1 → n_tokens values
        blocks = [H[:, 3*b:3*(b+1)] for b in range(self.n_tokens)]
        reduce_outs = self._qcall_3to1(blocks, self.theta_reduce)
        H = torch.cat(reduce_outs, dim=1)  # (B, n_tokens)

        if self.n_classes is not None:
            return self.cls_head(H)
        else:
            H = torch.stack(self.qnode_Nt_Nto1(H, self.theta_final, self.n_tokens)).T.float()
            return H * self.multiplier
