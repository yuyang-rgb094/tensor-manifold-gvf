"""
Hierarchical Manifold Encoder with gated residual connections,
relation-aware projections, and manifold normalization.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldGatedResidual(nn.Module):
    """Gated residual fusion of signature and semantic features.

    gate = sigmoid(W_proj @ concat(sig, sem))
    fused = gate * sig + (1 - gate) * sem
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.proj = nn.Linear(input_dim * 2, hidden_dim)
        self.gate_act = nn.Sigmoid()

    def forward(
        self,
        signature: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        signature : torch.Tensor
            Shape (..., d) -- diffusion signature.
        semantic : torch.Tensor
            Shape (..., d) -- semantic embedding.

        Returns
        -------
        torch.Tensor
            Shape (..., d_out) -- fused representation.
        """
        concat = torch.cat([signature, semantic], dim=-1)
        gate = self.gate_act(self.proj(concat))
        return gate * signature + (1 - gate) * semantic


class RelationAwareProjection(nn.Module):
    """Relation-aware linear projection.

    output = base_W @ input + relation_offsets[relation_type]

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    num_relations : int
        Number of distinct relation types.
    """

    def __init__(self, input_dim: int, output_dim: int, num_relations: int = 4):
        super().__init__()
        self.base_W = nn.Linear(input_dim, output_dim, bias=True)
        self.relation_offsets = nn.Embedding(num_relations, output_dim)
        # Initialize offsets to zero so behavior is base_W at init
        nn.init.zeros_(self.relation_offsets.weight)

    def forward(
        self,
        x: torch.Tensor,
        relation_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (..., input_dim).
        relation_ids : Optional[torch.LongTensor]
            Shape (...,) -- integer relation type ids.
            If ``None``, uses relation 0.

        Returns
        -------
        torch.Tensor
            Shape (..., output_dim).
        """
        out = self.base_W(x)
        if relation_ids is not None:
            offset = self.relation_offsets(relation_ids)
            out = out + offset
        else:
            out = out + self.relation_offsets.weight[0]
        return out


class ManifoldNormalizer(nn.Module):
    """L2 normalization projected onto the manifold."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2 normalize along the last dimension."""
        return F.normalize(x, p=2, dim=-1, eps=self.eps)


class HierarchicalManifoldEncoder(nn.Module):
    """Hierarchical encoder that fuses diffusion signatures with semantic
    embeddings using gated residuals and relation-aware projections.

    Parameters
    ----------
    semantic_dim : int
        Dimension of input semantic embeddings.
    signature_dim : int
        Dimension of input diffusion signatures.
    hidden_dim : int
        Intermediate hidden dimension.
    output_dim : int
        Final output embedding dimension.
    num_relations : int
        Number of relation types for relation-aware projection.
    """

    def __init__(
        self,
        semantic_dim: int,
        signature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_relations: int = 4,
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.signature_dim = signature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Align signature and semantic to common hidden dim
        self.sig_proj = nn.Linear(signature_dim, hidden_dim)
        self.sem_proj = nn.Linear(semantic_dim, hidden_dim)

        # Gated residual fusion
        self.gated_fusion = ManifoldGatedResidual(hidden_dim * 2, hidden_dim)

        # Relation-aware projection
        self.relation_proj = RelationAwareProjection(
            hidden_dim, output_dim, num_relations
        )

        # Manifold normalizer
        self.normalizer = ManifoldNormalizer()

    def _get_primary_relation_type(
        self,
        edge_types: Optional[List[str]] = None,
        relation_vocab: Optional[Dict[str, int]] = None,
    ) -> int:
        """Determine the primary relation type id for a node.

        Parameters
        ----------
        edge_types : Optional[List[str]]
            Edge types connected to the node.
        relation_vocab : Optional[Dict[str, int]]
            Mapping from relation name to integer id.

        Returns
        -------
        int
            Relation type id (defaults to 0).
        """
        if edge_types and relation_vocab:
            # Return the most frequent relation type
            from collections import Counter
            counts = Counter(edge_types)
            most_common = counts.most_common(1)[0][0]
            return relation_vocab.get(most_common, 0)
        return 0

    def encode_node(
        self,
        semantic: torch.Tensor,
        signature: torch.Tensor,
        relation_id: int = 0,
    ) -> torch.Tensor:
        """Encode a single node.

        Parameters
        ----------
        semantic : torch.Tensor
            Shape (semantic_dim,).
        signature : torch.Tensor
            Shape (signature_dim,).
        relation_id : int
            Primary relation type id.

        Returns
        -------
        torch.Tensor
            Shape (output_dim,).
        """
        # Project to hidden dim
        sig_h = self.sig_proj(signature)       # (hidden_dim,)
        sem_h = self.sem_proj(semantic)        # (hidden_dim,)

        # Gated residual fusion
        fused = self.gated_fusion(sig_h, sem_h)  # (hidden_dim,)

        # Relation-aware projection
        rel_id = torch.tensor([relation_id], device=fused.device)
        projected = self.relation_proj(fused.unsqueeze(0), rel_id)  # (1, output_dim)

        # Manifold normalization
        return self.normalizer(projected).squeeze(0)  # (output_dim,)

    def encode_all(
        self,
        semantics: torch.Tensor,
        signatures: torch.Tensor,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode all nodes.

        Parameters
        ----------
        semantics : torch.Tensor
            Shape (N, semantic_dim).
        signatures : torch.Tensor
            Shape (N, signature_dim).
        relation_ids : Optional[torch.Tensor]
            Shape (N,) -- per-node relation type ids.

        Returns
        -------
        torch.Tensor
            Shape (N, output_dim).
        """
        N = semantics.shape[0]
        if relation_ids is None:
            relation_ids = torch.zeros(N, dtype=torch.long, device=semantics.device)

        sig_h = self.sig_proj(signatures)     # (N, hidden_dim)
        sem_h = self.sem_proj(semantics)      # (N, hidden_dim)

        fused = self.gated_fusion(sig_h, sem_h)  # (N, hidden_dim)
        projected = self.relation_proj(fused, relation_ids)  # (N, output_dim)
        return self.normalizer(projected)  # (N, output_dim)

    def encode_batch(
        self,
        semantics: torch.Tensor,
        signatures: torch.Tensor,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized batch encoding (alias for encode_all).

        Parameters
        ----------
        semantics : torch.Tensor
            Shape (B, N, semantic_dim) or (N, semantic_dim).
        signatures : torch.Tensor
            Shape (B, N, signature_dim) or (N, signature_dim).
        relation_ids : Optional[torch.Tensor]
            Shape (B, N) or (N,).

        Returns
        -------
        torch.Tensor
            Same batch structure as input.
        """
        if semantics.dim() == 3:
            # Batch mode: (B, N, d) -> reshape, encode, restore
            B, N, _ = semantics.shape
            sem_flat = semantics.reshape(B * N, -1)
            sig_flat = signatures.reshape(B * N, -1)
            if relation_ids is not None:
                rel_flat = relation_ids.reshape(B * N)
            else:
                rel_flat = None

            encoded_flat = self.encode_all(sem_flat, sig_flat, rel_flat)
            return encoded_flat.reshape(B, N, self.output_dim)
        else:
            return self.encode_all(semantics, signatures, relation_ids)
