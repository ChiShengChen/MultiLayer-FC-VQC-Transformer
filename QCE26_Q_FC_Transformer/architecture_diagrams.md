# Architecture Diagrams Reference

ASCII references for drawing publication-quality figures.
Color suggestion: **quantum blocks** (blue/purple), **classical components** (green/gray).

---

## FC-VQC (Multi-Layer Fully-Connected)

Visual focus: funnel shape with decreasing qubit count per layer.

```
Input x (13 features)
    │
    ▼
┌─────────────────────────────┐
│  Linear Encoding (13 → 16)  │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   VQC Block 1 (16 qubits)  │
│   StronglyEntanglingLayers  │
│   depth=3                   │
│   ┌─┐ ┌─┐ ┌─┐     ┌──┐    │
│   │q₁│─│q₂│─│q₃│─···─│q₁₆│ │
│   └─┘ └─┘ └─┘     └──┘    │
│         Measure all         │
└─────────────┬───────────────┘
              ▼ (16 values)
┌─────────────────────────────┐
│  Type-4: Linear (16 → 4)   │  ← fully-connected inter-block
│  Type-3: Circular shift     │  ← or fixed permutation
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   VQC Block 2 (4 qubits)   │
│   StronglyEntanglingLayers  │
│   depth=3                   │
│   ┌─┐ ┌─┐ ┌─┐ ┌─┐         │
│   │q₁│─│q₂│─│q₃│─│q₄│      │
│   └─┘ └─┘ └─┘ └─┘         │
│         Measure all         │
└─────────────┬───────────────┘
              ▼ (4 values)
┌─────────────────────────────┐
│     Linear (4 → 1)         │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   VQC Block 3 (1 qubit)    │
│   depth=3                   │
│         ┌─┐                 │
│         │q₁│                │
│         └─┘                 │
│        Measure              │
└─────────────┬───────────────┘
              ▼
         Output ŷ
```

---

## ResNet-VQC (Residual)

Visual focus: skip connections (draw as dashed arrows).

```
Input x
    │
    ▼
┌──────────────┐
│ Linear Enc.  │
└──────┬───────┘
       ▼
       z₀
       │
       ├──────────────────┐
       ▼                  │ skip
┌──────────────┐          │
│  VQC Block 1 │          │
│  (n qubits)  │          │
└──────┬───────┘          │
       ▼                  │
       ⊕ ◄───────────────┘   z₁ = VQC₁(z₀) + z₀
       │
       ├──────────────────┐
       ▼                  │ skip
┌──────────────┐          │
│  VQC Block 2 │          │
│  (n qubits)  │          │
└──────┬───────┘          │
       ▼                  │
       ⊕ ◄───────────────┘   z₂ = VQC₂(z₁) + z₁
       │
       ├──────────────────┐
       ▼                  │ skip (with Linear if dim changes)
┌──────────────┐   ┌──────┴──────┐
│  VQC Block 3 │   │ Linear proj │
│  (m qubits)  │   │  (n → m)    │
└──────┬───────┘   └──────┬──────┘
       ▼                  │
       ⊕ ◄───────────────┘   z₃ = VQC₃(z₂) + W·z₂
       │
       ▼
    Output ŷ
```

---

## QT — Quantum Transformer (Route A: Hybrid)

Visual focus: classical MHA (green) + quantum FFN (blue).

```
Input x (d features)
    │
    ▼
┌───────────────────────────────────────────┐
│           VQC Encoder (d qubits)          │
│  AngleEmbedding → StronglyEntanglingLayers│
│              Measure all                  │
└───────────────────┬───────────────────────┘
                    │ embeddings e ∈ ℝᵈ
          ┌─────────┴─────────┐
          │                   │  residual
          ▼                   │
┌───────────────────┐         │
│  Classical MHA    │         │
│  ┌─────┐ ┌─────┐ │         │
│  │Head₁│ │Head₂│ │  (H heads)
│  │Q K V│ │Q K V│ │         │
│  └──┬──┘ └──┬──┘ │         │
│     └───┬───┘    │         │
│     Concat+Proj  │         │
└────────┬─────────┘         │
         ▼                   │
         ⊕ ◄─────────────────┘
         │
         ▼
    LayerNorm
         │
         ├────────────────────┐
         ▼                    │  residual
┌────────────────────┐        │
│   VQC FFN          │        │
│   (quantum block)  │        │
│   Encode → Entangle│        │
│   → Measure        │        │
└────────┬───────────┘        │
         ▼                    │
         ⊕ ◄─────────────────┘
         │
         ▼
    Linear → Output ŷ
```

---

## FQT — Fully Quantum Transformer (Route B)

Visual focus: everything is quantum (blue/purple). Highlight the absence of LayerNorm by default.

```
Input x (d features)
    │
    ▼
┌───────────────────────────────────────────┐
│           VQC Encoder (d qubits)          │
│  AngleEmbedding → StronglyEntanglingLayers│
│              Measure all                  │
└───────────────────┬───────────────────────┘
                    │ embeddings e ∈ ℝᵈ
          ┌─────────┴─────────┐
          │                   │  residual
          ▼                   │
┌─────────────────────────┐   │
│  Quantum Attention      │   │
│  ┌────────────────────┐ │   │
│  │VQC_Q  VQC_K  VQC_V│ │   │
│  │(each a separate    │ │   │
│  │ parameterized VQC) │ │   │
│  └────────┬───────────┘ │   │
│      scores = Q·Kᵀ/√d  │   │
│      softmax → ·V       │   │
└────────────┬────────────┘   │
             ▼                │
             ⊕ ◄──────────────┘
             │
             ▼
   ┌──── [no LayerNorm by default] ────┐
   │     [+LN variant adds it here]    │
   └──────────┬────────────────────────┘
              │
              ├────────────────────┐
              ▼                    │  residual
┌──────────────────────┐           │
│   VQC FFN            │           │
│   (quantum block)    │           │
│   Encode → Entangle  │           │
│   → Measure          │           │
└──────────┬───────────┘           │
           ▼                       │
           ⊕ ◄─────────────────────┘
           │
           ▼
      Linear → Output ŷ
```

---

## QT vs FQT Side-by-Side

```
        QT (Route A)                    FQT (Route B)
   ┌─────────────────┐            ┌─────────────────┐
   │   VQC Encoder   │            │   VQC Encoder   │
   └────────┬────────┘            └────────┬────────┘
            ▼                              ▼
   ┌─────────────────┐            ┌─────────────────┐
   │ CLASSICAL  MHA  │ ← green   │  QUANTUM   MHA  │ ← blue
   │ (Linear Q,K,V)  │           │ (VQC_Q, VQC_K,  │
   │                 │           │  VQC_V)          │
   └────────┬────────┘            └────────┬────────┘
        ⊕ + residual                   ⊕ + residual
            ▼                              ▼
       LayerNorm ✓                  [no LN by default]
            ▼                              ▼
   ┌─────────────────┐            ┌─────────────────┐
   │    VQC FFN      │ ← blue    │    VQC FFN      │ ← blue
   └────────┬────────┘            └────────┬────────┘
        ⊕ + residual                   ⊕ + residual
            ▼                              ▼
         Output                         Output
```

---

## Ablation Variants Quick Reference

```
Variant         Attention     FFN          LayerNorm
────────────────────────────────────────────────────
QT default      Classical     VQC (Type4)  ✓
QT −attn        ✗ removed     VQC (Type4)  ✓
QT T3 FFN       Classical     VQC (Type3)  ✓
FQT default     Quantum       VQC (Type4)  ✗
FQT −attn       ✗ removed     VQC (Type4)  ✗
FQT T3 FFN      Quantum       VQC (Type3)  ✗
FQT +LN         Quantum       VQC (Type4)  ✓ added
```

---

## Drawing Tips

- Use the same color scheme across all diagrams:
  - **Blue/purple** = quantum components (VQC blocks, quantum attention)
  - **Green/gray** = classical components (Linear, Classical MHA, LayerNorm)
- FC-VQC: emphasize the **funnel shape** (16 → 4 → 1 qubits)
- ResNet-VQC: emphasize **dashed skip arrows**
- QT vs FQT: draw **side-by-side** with color difference in the attention block
- Consider a combined 4-panel figure (one per architecture) for the paper
