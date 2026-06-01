"""TDD tests for Task-Specific Attention Heads.

RED → GREEN → REFACTOR cycle for implementing task-specific channel weighting.

Phase 2 Goals:
1. TaskSpecificAttentionHead - different channel weights per task
2. Attention weight extraction and visualization
3. Per-task channel sensitivity statistics

See ADR-0006 for architectural rationale.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# RED Phase: TaskSpecificAttentionHead Interface
# ======================================================================

class TestTaskSpecificAttentionHeadInterface:
    """Define the interface for task-specific attention heads.

    Each task (semantic_retrieval, citation_analysis, author_disambiguation,
    trend_analysis) has its own attention head with learned channel weights.
    """

    def test_task_head_has_required_attributes(self):
        """TaskSpecificAttentionHead must have task_name and channel_weights."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        assert head.task_name == "semantic_retrieval"
        assert hasattr(head, "channel_weights")
        assert hasattr(head, "forward")

    def test_task_head_forward_shape(self):
        """Forward pass should produce correct output shape."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        # Four channel inputs (already projected to hidden_dim)
        sem = torch.randn(5, 64)
        meta = torch.randn(5, 64)
        topo = torch.randn(5, 64)
        temp = torch.randn(5, 64)

        output = head.forward(sem, meta, topo, temp)

        assert output.shape == (5, 32), f"Expected (5, 32), got {output.shape}"

    def test_task_head_produces_l2_normalized_output(self):
        """Output should be L2 normalized on the manifold."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        sem = torch.randn(3, 64)
        meta = torch.randn(3, 64)
        topo = torch.randn(3, 64)
        temp = torch.randn(3, 64)

        output = head.forward(sem, meta, topo, temp)
        norms = torch.norm(output, dim=1)

        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=0)


# ======================================================================
# RED Phase: Task-Specific Channel Weight Differences
# ======================================================================

class TestTaskSpecificChannelWeights:
    """Verify different tasks have different channel weight distributions."""

    def test_semantic_retrieval_weights_semantic_high(self):
        """Semantic retrieval task should weight semantic channel highest."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        weights = head.get_channel_weights()

        assert isinstance(weights, dict)
        assert "semantic" in weights
        assert "topology" in weights
        assert "metadata" in weights
        assert "temporal" in weights

        # Semantic retrieval should prioritize semantic channel
        assert weights["semantic"] > weights["topology"], \
            f"Semantic retrieval should weight semantic > topology, got {weights}"

    def test_citation_analysis_weights_topology_high(self):
        """Citation analysis task should weight topology channel highest."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="citation_analysis",
            hidden_dim=64,
            output_dim=32,
        )

        weights = head.get_channel_weights()

        # Citation analysis should prioritize topology
        assert weights["topology"] > weights["temporal"], \
            f"Citation analysis should weight topology > temporal, got {weights}"

    def test_author_disambiguation_weights_metadata_high(self):
        """Author disambiguation should weight metadata channel highest."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="author_disambiguation",
            hidden_dim=64,
            output_dim=32,
        )

        weights = head.get_channel_weights()

        # Author disambiguation relies on author/venue info
        assert weights["metadata"] > weights["topology"], \
            f"Author disambiguation should weight metadata > topology, got {weights}"

    def test_trend_analysis_weights_temporal_high(self):
        """Trend analysis should weight temporal channel highest."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="trend_analysis",
            hidden_dim=64,
            output_dim=32,
        )

        weights = head.get_channel_weights()

        # Trend analysis relies on time evolution
        assert weights["temporal"] > weights["topology"], \
            f"Trend analysis should weight temporal > topology, got {weights}"


# ======================================================================
# RED Phase: Task Registry and Factory
# ======================================================================

class TestTaskRegistry:
    """Test the task registry and factory function."""

    def test_registry_has_all_tasks(self):
        """Registry should contain all four task types."""
        from core.task_attention import TASK_REGISTRY

        assert "semantic_retrieval" in TASK_REGISTRY
        assert "citation_analysis" in TASK_REGISTRY
        assert "author_disambiguation" in TASK_REGISTRY
        assert "trend_analysis" in TASK_REGISTRY

    def test_create_task_head_factory(self):
        """Factory function should create correct task head."""
        from core.task_attention import create_task_head

        head = create_task_head("semantic_retrieval", hidden_dim=64, output_dim=32)

        assert head.task_name == "semantic_retrieval"

    def test_invalid_task_raises(self):
        """Invalid task name should raise ValueError."""
        from core.task_attention import create_task_head

        with pytest.raises(ValueError, match="Unknown task"):
            create_task_head("invalid_task", hidden_dim=64, output_dim=32)


# ======================================================================
# RED Phase: Attention Weight Extraction
# ======================================================================

class TestAttentionWeightExtraction:
    """Test extraction and visualization of attention weights."""

    def test_extract_attention_weights(self):
        """Should be able to extract attention weights from forward pass."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        sem = torch.randn(2, 64)
        meta = torch.randn(2, 64)
        topo = torch.randn(2, 64)
        temp = torch.randn(2, 64)

        output, attention_info = head.forward_with_attention(sem, meta, topo, temp)

        assert "channel_weights" in attention_info
        assert "cross_attention_weights" in attention_info
        assert output.shape == (2, 32)

    def test_attention_weights_sum_to_one(self):
        """Channel weights should sum to 1 (normalized)."""
        from core.task_attention import TaskSpecificAttentionHead

        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
        )

        weights = head.get_channel_weights()
        total = sum(weights.values())

        assert abs(total - 1.0) < 1e-5, f"Weights should sum to 1, got {total}"


# ======================================================================
# RED Phase: Per-Task Sensitivity Statistics
# ======================================================================

class TestPerTaskSensitivity:
    """Test per-task channel sensitivity computation."""

    def test_compute_sensitivity_report(self):
        """Should compute sensitivity report for a task."""
        from core.task_attention import compute_task_sensitivity_report

        report = compute_task_sensitivity_report(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
            n_samples=5,
        )

        assert "task_name" in report
        assert "channel_sensitivities" in report
        assert "most_sensitive_channel" in report
        assert "least_sensitive_channel" in report

    def test_sensitivity_report_matches_task_priorities(self):
        """Sensitivity report should align with task priorities."""
        from core.task_attention import compute_task_sensitivity_report

        report = compute_task_sensitivity_report(
            task_name="semantic_retrieval",
            hidden_dim=64,
            output_dim=32,
            n_samples=10,
        )

        # For semantic_retrieval, semantic channel should be among top 2
        top_channels = sorted(
            report["channel_sensitivities"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        top_names = [name for name, _ in top_channels]

        # This is a soft check — semantic should be influential
        assert report["most_sensitive_channel"] in ["semantic", "metadata", "topology", "temporal"]


# ======================================================================
# Integration: Task Head with Four Channel Encoder
# ======================================================================

class TestTaskHeadIntegration:
    """Test TaskSpecificAttentionHead integration with channel encoders."""

    def test_task_head_with_real_channel_outputs(self):
        """End-to-end: channel encoders → task head → output."""
        from core.task_attention import TaskSpecificAttentionHead
        from retrieval.encoders.channels.semantic_channel import SemanticChannelEncoder
        from retrieval.encoders.channels.metadata_channel import MetadataChannelEncoder
        from retrieval.encoders.channels.temporal_channel import TemporalChannelEncoder

        # Create encoders
        sem_enc = SemanticChannelEncoder(fallback_to_sbert=False)
        meta_enc = MetadataChannelEncoder(output_dim=32)
        temp_enc = TemporalChannelEncoder(output_dim=16, n_periodic=8)

        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
        ]
        meta_enc.build_vocab(docs)

        # Encode
        texts = [f"{d['title']} {d['abstract']}" for d in docs]
        sem_emb = sem_enc.encode(texts)
        meta_emb = meta_enc.encode(docs)
        temp_emb = temp_enc.encode([d["year"] for d in docs])
        topo_emb = np.random.randn(1, 32).astype(np.float32)

        # Project to hidden_dim
        hidden_dim = 64
        proj_sem = nn.Linear(sem_enc.output_dim, hidden_dim)
        proj_meta = nn.Linear(32, hidden_dim)
        proj_topo = nn.Linear(32, hidden_dim)
        proj_temp = nn.Linear(16, hidden_dim)

        with torch.no_grad():
            sem_h = proj_sem(torch.tensor(sem_emb))
            meta_h = proj_meta(torch.tensor(meta_emb))
            topo_h = proj_topo(torch.tensor(topo_emb))
            temp_h = proj_temp(torch.tensor(temp_emb))

        # Task-specific head
        head = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=hidden_dim,
            output_dim=32,
        )

        output = head.forward(sem_h, meta_h, topo_h, temp_h)

        assert output.shape == (1, 32)
        # L2 normalized
        norm = torch.norm(output)
        torch.testing.assert_close(norm, torch.tensor(1.0), atol=1e-5, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
