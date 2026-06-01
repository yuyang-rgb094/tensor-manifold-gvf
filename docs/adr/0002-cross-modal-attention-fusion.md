# ADR-0002: Cross-Modal Attention as Tensor Manifold Projection Approximation

## Status

Proposed

## Context

The theoretical framework proves that the academic literature graph is a discrete projection of a tensor manifold (Theorem 3.2 in the paper). In v1, this projection is implemented via gated residual connection + linear projection, which works for two channels (semantic + diffusion signature) but doesn't scale to four channels.

With four independent channels (ADR-0001), we need a fusion mechanism that:
1. Preserves the theoretical coherence of tensor manifold projection
2. Supports task-specific channel weighting
3. Is trainable and adaptable to new tasks

Three options were considered:

### Option A: Pure Tensor Product Fusion
Construct a 4th-order tensor via tensor product of all channel embeddings, then apply relation-aware low-dimensional projection. Theoretically elegant but causes dimension explosion (768 × 256 × 256 × 64 ≈ 10^9 before projection).

### Option B: Cross-Modal Attention (CLIP/BLIP-style)
Each channel has independent Q/K/V projections. Cross-attention heads exchange information between channels. Task-specific heads learn per-task weights. Flexible but requires training and has tension with the analytical, matrix-free theoretical style.

### Option C: Hybrid — Tensor Manifold Theory + Cross-Attention Implementation
Theoretically prove that four-channel fusion is equivalent to tensor manifold projection. Engineering-wise, use cross-attention to approximate this projection because analytical solutions are infeasible at four-channel scale.

## Decision

**Adopt Option C (Hybrid approach).**

The theoretical narrative remains: "four-channel fusion is a tensor manifold projection." The engineering implementation uses cross-modal attention as a learnable approximation.

Formally, we define:
- Channel embeddings: $h_1, h_2, h_3, h_4 \in \mathbb{R}^{d_i}$
- Cross-attention fusion: $f(h_1, h_2, h_3, h_4) = \text{CrossAttn}(h_1, h_2, h_3, h_4; \theta)$
- Theorem (to be proven): $f$ approximates the tensor manifold projection $P: \mathcal{M} \to \mathbb{R}^D$ with bounded error $\|f - P\| < \epsilon$

## Consequences

### Positive
- Preserves theoretical coherence with the paper's framework
- Cross-attention is well-understood, with mature implementations (PyTorch nn.MultiheadAttention)
- Task-specific heads naturally provide "channel-level weight adjustability"
- Can be pre-trained self-supervised, then fine-tuned per task

### Negative
- The approximation theorem needs formal proof (ongoing work)
- Cross-attention introduces trainable parameters, moving away from the "matrix-free, analytical" philosophy of v1
- Training requires GPU resources not needed in v1

### Mitigations
- The approximation error can be bounded empirically even before formal proof
- Cross-attention is the de facto standard in modern multi-modal AI — this aligns the system with industry practice
- Pre-training can be done offline; inference is still O(1) per query
