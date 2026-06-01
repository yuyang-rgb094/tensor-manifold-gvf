"""Task-Specific Attention Heads for Four-Channel Fusion.

Each task (semantic_retrieval, citation_analysis, author_disambiguation,
trend_analysis) has its own attention head with learned channel weights.

See ADR-0006 for architectural rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# Task-Specific Channel Weight Presets
# ======================================================================

# Default channel weights for each task type
# These are soft biases; the network can learn to adjust them
_TASK_CHANNEL_WEIGHTS = {
    "semantic_retrieval": {
        "semantic": 0.40,
        "metadata": 0.20,
        "topology": 0.25,
        "temporal": 0.15,
    },
    "citation_analysis": {
        "semantic": 0.20,
        "metadata": 0.15,
        "topology": 0.45,
        "temporal": 0.20,
    },
    "author_disambiguation": {
        "semantic": 0.25,
        "metadata": 0.40,
        "topology": 0.15,
        "temporal": 0.20,
    },
    "trend_analysis": {
        "semantic": 0.25,
        "metadata": 0.20,
        "topology": 0.20,
        "temporal": 0.35,
    },
}


# ======================================================================
# TaskSpecificAttentionHead
# ======================================================================

class TaskSpecificAttentionHead(nn.Module):
    """Task-specific attention head with learned channel weights.

    Each task has its own set of parameters, including:
    - Channel importance weights (learnable, initialized from presets)
    - Cross-attention between channel pairs
    - Output projection

    Parameters
    ----------
    task_name : str
        One of "semantic_retrieval", "citation_analysis",
        "author_disambiguation", "trend_analysis".
    hidden_dim : int
        Dimension of channel embeddings (all channels projected to this).
    output_dim : int
        Output embedding dimension.
    num_attention_heads : int
        Number of heads for cross-attention.
    dropout : float
        Dropout rate.
    """

    TASK_NAMES = [
        "semantic_retrieval",
        "citation_analysis",
        "author_disambiguation",
        "trend_analysis",
    ]

    def __init__(
        self,
        task_name: str,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if task_name not in self.TASK_NAMES:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Valid tasks: {self.TASK_NAMES}"
            )

        self.task_name = task_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Channel weights attribute (for interface compatibility)
        self.channel_weights: Optional[Dict[str, float]] = None

        # ------------------------------------------------------------------
        # Channel importance weights (learnable)
        # ------------------------------------------------------------------
        preset_weights = _TASK_CHANNEL_WEIGHTS[task_name]
        initial_weights = torch.tensor([
            preset_weights["semantic"],
            preset_weights["metadata"],
            preset_weights["topology"],
            preset_weights["temporal"],
        ], dtype=torch.float32)

        # Store as learnable parameters (log space for positivity)
        self._channel_weight_logits = nn.Parameter(
            torch.log(initial_weights + 1e-8)
        )

        # ------------------------------------------------------------------
        # Cross-attention between channel pairs
        # ------------------------------------------------------------------
        self.sem_topo_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.meta_temp_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.sem_topo_ln = nn.LayerNorm(hidden_dim)
        self.meta_temp_ln = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        self.output_ln = nn.LayerNorm(output_dim)

    # ------------------------------------------------------------------
    # Channel weights
    # ------------------------------------------------------------------

    def get_channel_weights(self) -> Dict[str, float]:
        """Get current channel weights (softmax-normalized).

        Returns
        -------
        Dict[str, float]
            Mapping from channel name to weight (sums to 1).
        """
        weights = torch.softmax(self._channel_weight_logits, dim=0)
        result = {
            "semantic": float(weights[0].detach()),
            "metadata": float(weights[1].detach()),
            "topology": float(weights[2].detach()),
            "temporal": float(weights[3].detach()),
        }
        self.channel_weights = result
        return result

    def _apply_channel_weights(
        self,
        sem: torch.Tensor,
        meta: torch.Tensor,
        topo: torch.Tensor,
        temp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply channel importance weights."""
        weights = torch.softmax(self._channel_weight_logits, dim=0)
        w_sem, w_meta, w_topo, w_temp = weights

        # Scale each channel by sqrt(weight) to preserve variance
        scale_sem = torch.sqrt(w_sem)
        scale_meta = torch.sqrt(w_meta)
        scale_topo = torch.sqrt(w_topo)
        scale_temp = torch.sqrt(w_temp)

        return (
            sem * scale_sem,
            meta * scale_meta,
            topo * scale_topo,
            temp * scale_temp,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        semantic: torch.Tensor,
        metadata: torch.Tensor,
        topology: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        semantic : torch.Tensor
            Shape (N, hidden_dim).
        metadata : torch.Tensor
            Shape (N, hidden_dim).
        topology : torch.Tensor
            Shape (N, hidden_dim).
        temporal : torch.Tensor
            Shape (N, hidden_dim).

        Returns
        -------
        torch.Tensor
            Shape (N, output_dim), L2-normalized.
        """
        # Apply channel weights
        sem_w, meta_w, topo_w, temp_w = self._apply_channel_weights(
            semantic, metadata, topology, temporal
        )

        # Cross-attention: semantic <-> topology
        sem_attn, _ = self.sem_topo_attn(
            query=sem_w, key=topo_w, value=topo_w
        )
        sem_fused = self.sem_topo_ln(sem_w + sem_attn)

        # Cross-attention: metadata <-> temporal
        meta_attn, _ = self.meta_temp_attn(
            query=meta_w, key=temp_w, value=temp_w
        )
        meta_fused = self.meta_temp_ln(meta_w + meta_attn)

        # Concatenate and project
        concat = torch.cat([sem_fused, meta_fused, topo_w, temp_w], dim=-1)
        output = self.output_proj(concat)
        output = self.output_ln(output)

        # L2 normalize
        norms = torch.norm(output, dim=1, keepdim=True)
        output = output / torch.clamp(norms, min=1e-8)

        return output

    def forward_with_attention(
        self,
        semantic: torch.Tensor,
        metadata: torch.Tensor,
        topology: torch.Tensor,
        temporal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with attention weight extraction.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - output: (N, output_dim)
            - attention_info: dict with channel_weights and cross_attention_weights
        """
        # Apply channel weights
        sem_w, meta_w, topo_w, temp_w = self._apply_channel_weights(
            semantic, metadata, topology, temporal
        )

        # Cross-attention with weights
        sem_attn, sem_attn_weights = self.sem_topo_attn(
            query=sem_w, key=topo_w, value=topo_w, need_weights=True
        )
        sem_fused = self.sem_topo_ln(sem_w + sem_attn)

        meta_attn, meta_attn_weights = self.meta_temp_attn(
            query=meta_w, key=temp_w, value=temp_w, need_weights=True
        )
        meta_fused = self.meta_temp_ln(meta_w + meta_attn)

        # Concatenate and project
        concat = torch.cat([sem_fused, meta_fused, topo_w, temp_w], dim=-1)
        output = self.output_proj(concat)
        output = self.output_ln(output)

        # L2 normalize
        norms = torch.norm(output, dim=1, keepdim=True)
        output = output / torch.clamp(norms, min=1e-8)

        attention_info = {
            "channel_weights": self.get_channel_weights(),
            "cross_attention_weights": {
                "semantic_topology": sem_attn_weights.detach().cpu().numpy(),
                "metadata_temporal": meta_attn_weights.detach().cpu().numpy(),
            },
        }

        return output, attention_info


# ======================================================================
# Task Registry and Factory
# ======================================================================

TASK_REGISTRY: Dict[str, type] = {
    "semantic_retrieval": TaskSpecificAttentionHead,
    "citation_analysis": TaskSpecificAttentionHead,
    "author_disambiguation": TaskSpecificAttentionHead,
    "trend_analysis": TaskSpecificAttentionHead,
}


def create_task_head(
    task_name: str,
    hidden_dim: int = 64,
    output_dim: int = 32,
    **kwargs,
) -> TaskSpecificAttentionHead:
    """Factory function to create a task-specific attention head.

    Parameters
    ----------
    task_name : str
        Task type.
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension.

    Returns
    -------
    TaskSpecificAttentionHead
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Valid tasks: {list(TASK_REGISTRY.keys())}"
        )

    return TaskSpecificAttentionHead(
        task_name=task_name,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        **kwargs,
    )


# ======================================================================
# Per-Task Sensitivity Computation
# ======================================================================

def compute_task_sensitivity_report(
    task_name: str,
    hidden_dim: int = 64,
    output_dim: int = 32,
    n_samples: int = 10,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    """Compute channel sensitivity report for a task.

    Sensitivity measures how much the output changes when each channel
    is perturbed by a small amount.

    Parameters
    ----------
    task_name : str
        Task type.
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension.
    n_samples : int
        Number of random samples to average over.
    epsilon : float
        Perturbation magnitude.

    Returns
    -------
    Dict[str, Any]
        Report with channel sensitivities and statistics.
    """
    head = create_task_head(task_name, hidden_dim, output_dim)
    head.eval()

    rng = np.random.default_rng(42)
    sensitivities: Dict[str, float] = {}

    for channel_name in ["semantic", "metadata", "topology", "temporal"]:
        total_change = 0.0

        for _ in range(n_samples):
            # Random channel inputs
            sem = torch.tensor(rng.standard_normal((1, hidden_dim)), dtype=torch.float32)
            meta = torch.tensor(rng.standard_normal((1, hidden_dim)), dtype=torch.float32)
            topo = torch.tensor(rng.standard_normal((1, hidden_dim)), dtype=torch.float32)
            temp = torch.tensor(rng.standard_normal((1, hidden_dim)), dtype=torch.float32)

            with torch.no_grad():
                baseline = head.forward(sem, meta, topo, temp).numpy()

                # Perturb target channel
                perturbation = epsilon * torch.tensor(
                    rng.standard_normal((1, hidden_dim)), dtype=torch.float32
                )

                if channel_name == "semantic":
                    perturbed = head.forward(sem + perturbation, meta, topo, temp).numpy()
                elif channel_name == "metadata":
                    perturbed = head.forward(sem, meta + perturbation, topo, temp).numpy()
                elif channel_name == "topology":
                    perturbed = head.forward(sem, meta, topo + perturbation, temp).numpy()
                elif channel_name == "temporal":
                    perturbed = head.forward(sem, meta, topo, temp + perturbation).numpy()

            change = np.linalg.norm(perturbed - baseline)
            total_change += change

        sensitivities[channel_name] = float(total_change / n_samples)

    # Find most/least sensitive
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    return {
        "task_name": task_name,
        "channel_sensitivities": sensitivities,
        "most_sensitive_channel": sorted_sens[0][0],
        "least_sensitive_channel": sorted_sens[-1][0],
        "sensitivity_ratio": sorted_sens[0][1] / (sorted_sens[-1][1] + 1e-8),
    }
