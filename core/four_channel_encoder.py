"""Four-channel fusion encoder.

Fuses semantic, metadata, topology, and temporal channel embeddings
into a unified manifold embedding using cross-modal attention and
gated residual fusion.  Reuses internal components from
:class:`HierarchicalManifoldEncoder`.

See ADR-0002 for the architectural rationale (tensor manifold theory
+ cross-attention engineering approximation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .manifold_encoder import (
    ManifoldGatedResidual,
    ManifoldNormalizer,
    RelationAwareProjection,
)


class FourChannelFusionEncoder(nn.Module):
    """Four-channel fusion encoder.

    Architecture:
        1. Each channel projected to unified ``hidden_dim``
        2. Cross-attention: semantic <-> topology, metadata <-> temporal
        3. Concatenate four channels -> gated residual fusion
        4. Relation-aware projection -> manifold normalization

    Parameters
    ----------
    semantic_dim : int
        Input dimension from semantic channel (default 1024 for BGE-M3).
    metadata_dim : int
        Input dimension from metadata channel (default 256).
    topology_dim : int
        Input dimension from topology channel (default 256).
    temporal_dim : int
        Input dimension from temporal channel (default 64).
    hidden_dim : int
        Unified intermediate dimension for all channels.
    output_dim : int
        Final manifold embedding dimension.
    num_relations : int
        Number of relation types for relation-aware projection.
    num_attention_heads : int
        Number of attention heads for cross-modal attention.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        metadata_dim: int = 256,
        topology_dim: int = 256,
        temporal_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_relations: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.metadata_dim = metadata_dim
        self.topology_dim = topology_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ------------------------------------------------------------------
        # Step 1: Channel projections to unified hidden_dim
        # ------------------------------------------------------------------
        self.sem_proj = nn.Linear(semantic_dim, hidden_dim)
        self.meta_proj = nn.Linear(metadata_dim, hidden_dim)
        self.topo_proj = nn.Linear(topology_dim, hidden_dim)
        self.temp_proj = nn.Linear(temporal_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Step 2: Cross-modal attention pairs
        # ------------------------------------------------------------------
        # semantic <-> topology (captures how citation topology influences relevance)
        self.sem_topo_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        # metadata <-> temporal (captures how research themes evolve over time)
        self.meta_temp_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms for cross-attention outputs
        self.sem_topo_ln = nn.LayerNorm(hidden_dim)
        self.meta_temp_ln = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # Step 3: Gated residual fusion of all four channels
        # ------------------------------------------------------------------
        # Concatenate 4 * hidden_dim, fuse to hidden_dim via MLP with gating
        combined_dim = hidden_dim * 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # Step 4: Relation-aware projection + manifold normalization
        # ------------------------------------------------------------------
        self.relation_proj = RelationAwareProjection(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_relations=num_relations,
        )
        self.normalizer = ManifoldNormalizer()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        semantic: torch.Tensor,
        metadata: torch.Tensor,
        topology: torch.Tensor,
        temporal: torch.Tensor,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Four-channel fusion forward pass.

        Parameters
        ----------
        semantic : torch.Tensor
            Shape ``(N, semantic_dim)``.
        metadata : torch.Tensor
            Shape ``(N, metadata_dim)``.
        topology : torch.Tensor
            Shape ``(N, topology_dim)``.
        temporal : torch.Tensor
            Shape ``(N, temporal_dim)``.
        relation_ids : Optional[torch.Tensor]
            Shape ``(N,)`` — per-node relation type ids.

        Returns
        -------
        torch.Tensor
            Shape ``(N, output_dim)`` — L2-normalized manifold embeddings.
        """
        # Step 1: Project to unified hidden_dim
        sem_h = self.sem_proj(semantic)       # (N, hidden_dim)
        meta_h = self.meta_proj(metadata)    # (N, hidden_dim)
        topo_h = self.topo_proj(topology)    # (N, hidden_dim)
        temp_h = self.temp_proj(temporal)    # (N, hidden_dim)

        # Step 2: Cross-modal attention
        # semantic <-> topology
        sem_h_attn, _ = self.sem_topo_cross_attn(
            query=sem_h, key=topo_h, value=topo_h
        )
        sem_h = self.sem_topo_ln(sem_h + sem_h_attn)  # residual connection

        # metadata <-> temporal
        meta_h_attn, _ = self.meta_temp_cross_attn(
            query=meta_h, key=temp_h, value=temp_h
        )
        meta_h = self.meta_temp_ln(meta_h + meta_h_attn)  # residual connection

        # Step 3: Concatenate and gated fusion
        concat = torch.cat([sem_h, meta_h, topo_h, temp_h], dim=-1)  # (N, 4*hidden_dim)
        gate = self.fusion_gate(concat)       # (N, hidden_dim)
        fused = self.fusion_mlp(concat)       # (N, hidden_dim)
        fused = gate * fused + (1 - gate) * fused.mean(dim=-1, keepdim=True).expand_as(fused)
        fused = self.fusion_ln(fused)

        # Step 4: Relation-aware projection + normalization
        projected = self.relation_proj(fused, relation_ids)  # (N, output_dim)
        normalized = self.normalizer(projected)  # (N, output_dim)

        return normalized

    # ------------------------------------------------------------------
    # Convenience: numpy in/out
    # ------------------------------------------------------------------

    def encode_all_from_channels(
        self,
        semantic: np.ndarray,
        metadata: np.ndarray,
        topology: np.ndarray,
        temporal: np.ndarray,
        relation_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode from numpy channel outputs to numpy manifold embeddings.

        Parameters
        ----------
        semantic : np.ndarray
            Shape ``(N, semantic_dim)``.
        metadata : np.ndarray
            Shape ``(N, metadata_dim)``.
        topology : np.ndarray
            Shape ``(N, topology_dim)``.
        temporal : np.ndarray
            Shape ``(N, temporal_dim)``.
        relation_ids : Optional[np.ndarray]
            Shape ``(N,)`` — integer relation type ids.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)`` — L2-normalized manifold embeddings.
        """
        self.eval()
        with torch.no_grad():
            sem_t = torch.tensor(semantic, dtype=torch.float32)
            meta_t = torch.tensor(metadata, dtype=torch.float32)
            topo_t = torch.tensor(topology, dtype=torch.float32)
            temp_t = torch.tensor(temporal, dtype=torch.float32)

            rel_t = None
            if relation_ids is not None:
                rel_t = torch.tensor(relation_ids, dtype=torch.long)

            output = self.forward(sem_t, meta_t, topo_t, temp_t, rel_t)
            return output.numpy().astype(np.float32)
