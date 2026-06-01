"""TDD evaluation tests for cross-modal attention fusion.

These tests verify that the cross-modal attention mechanism actually
captures meaningful interactions between channels, not just shape correctness.

RED → GREEN → REFACTOR cycle:
1. Write failing test for desired behavior
2. Implement minimal code to pass
3. Refactor while keeping tests green
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.four_channel_encoder import FourChannelFusionEncoder


# ======================================================================
# RED Phase: Cross-Modal Attention Effectiveness
# ======================================================================

class TestCrossModalAttentionEffectiveness:
    """Verify cross-modal attention captures meaningful channel interactions.

    These tests go beyond shape checking — they verify that:
    1. Semantic-topology interaction affects the output
    2. Metadata-temporal interaction affects the output
    3. Different channel combinations produce different outputs
    """

    @pytest.fixture
    def fusion_encoder(self):
        return FourChannelFusionEncoder(
            semantic_dim=64,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=64,
            output_dim=32,
        )

    def test_semantic_change_affects_output(self, fusion_encoder):
        """Changing semantic input should change the fused output."""
        meta = np.random.randn(1, 32).astype(np.float32)
        topo = np.random.randn(1, 32).astype(np.float32)
        temp = np.random.randn(1, 16).astype(np.float32)

        sem1 = np.random.randn(1, 64).astype(np.float32)
        sem2 = sem1 + 2.0  # significantly different

        out1 = fusion_encoder.encode_all_from_channels(sem1, meta, topo, temp)
        out2 = fusion_encoder.encode_all_from_channels(sem2, meta, topo, temp)

        # Outputs should be different (not identical)
        assert not np.allclose(out1, out2, atol=1e-5), \
            "Semantic change should affect fused output"

    def test_topology_change_affects_output(self, fusion_encoder):
        """Changing topology input should change the fused output."""
        sem = np.random.randn(1, 64).astype(np.float32)
        meta = np.random.randn(1, 32).astype(np.float32)
        temp = np.random.randn(1, 16).astype(np.float32)

        topo1 = np.random.randn(1, 32).astype(np.float32)
        topo2 = topo1 + 2.0

        out1 = fusion_encoder.encode_all_from_channels(sem, meta, topo1, temp)
        out2 = fusion_encoder.encode_all_from_channels(sem, meta, topo2, temp)

        assert not np.allclose(out1, out2, atol=1e-5), \
            "Topology change should affect fused output"

    def test_cross_modal_interaction_not_additive(self, fusion_encoder):
        """Cross-modal attention should produce non-additive fusion.

        If fusion were purely additive (out = f(sem) + f(meta) + f(topo) + f(temp)),
        then changing one channel would have the same effect regardless of other channels.
        Cross-modal attention breaks this additivity.
        """
        sem1 = np.random.randn(1, 64).astype(np.float32)
        sem2 = sem1 + 1.0
        meta_a = np.random.randn(1, 32).astype(np.float32)
        meta_b = meta_a + 1.0
        topo = np.random.randn(1, 32).astype(np.float32)
        temp = np.random.randn(1, 16).astype(np.float32)

        # Effect of semantic change with meta_a
        out_a1 = fusion_encoder.encode_all_from_channels(sem1, meta_a, topo, temp)
        out_a2 = fusion_encoder.encode_all_from_channels(sem2, meta_a, topo, temp)
        delta_a = out_a2 - out_a1

        # Effect of same semantic change with meta_b
        out_b1 = fusion_encoder.encode_all_from_channels(sem1, meta_b, topo, temp)
        out_b2 = fusion_encoder.encode_all_from_channels(sem2, meta_b, topo, temp)
        delta_b = out_b2 - out_b1

        # If purely additive, delta_a == delta_b
        # Cross-modal attention should make them different
        assert not np.allclose(delta_a, delta_b, atol=1e-4), \
            "Cross-modal attention should create non-additive interactions"

    def test_all_channels_contribute(self, fusion_encoder):
        """Each channel should contribute to the final output.

        Zeroing out any channel should change the output.
        """
        sem = np.random.randn(1, 64).astype(np.float32)
        meta = np.random.randn(1, 32).astype(np.float32)
        topo = np.random.randn(1, 32).astype(np.float32)
        temp = np.random.randn(1, 16).astype(np.float32)

        baseline = fusion_encoder.encode_all_from_channels(sem, meta, topo, temp)

        # Zero out semantic
        out_no_sem = fusion_encoder.encode_all_from_channels(
            np.zeros_like(sem), meta, topo, temp
        )
        assert not np.allclose(baseline, out_no_sem, atol=1e-5), \
            "Semantic channel should contribute"

        # Zero out topology
        out_no_topo = fusion_encoder.encode_all_from_channels(
            sem, meta, np.zeros_like(topo), temp
        )
        assert not np.allclose(baseline, out_no_topo, atol=1e-5), \
            "Topology channel should contribute"


# ======================================================================
# GREEN Phase: Attention Weight Extractability
# ======================================================================

class TestAttentionWeightExtractability:
    """Verify attention weights can be extracted for interpretability.

    For task-specific channel weighting (ADR-0006), we need to:
    1. Extract attention weights from cross-modal attention layers
    2. Analyze which channels are most influential for different inputs
    """

    @pytest.fixture
    def fusion_encoder(self):
        return FourChannelFusionEncoder(
            semantic_dim=64,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=64,
            output_dim=32,
        )

    def test_encoder_has_attention_layers(self, fusion_encoder):
        """Fusion encoder should expose attention layer references."""
        assert hasattr(fusion_encoder, "sem_topo_cross_attn"), \
            "Missing semantic-topology cross attention"
        assert hasattr(fusion_encoder, "meta_temp_cross_attn"), \
            "Missing metadata-temporal cross attention"

    def test_can_run_attention_with_weights(self, fusion_encoder):
        """Should be able to run forward pass and get attention weights."""
        sem = torch.randn(1, 64)
        meta = torch.randn(1, 32)
        topo = torch.randn(1, 32)
        temp = torch.randn(1, 16)

        # Project to hidden_dim
        sem_h = fusion_encoder.sem_proj(sem)
        topo_h = fusion_encoder.topo_proj(topo)

        # Run cross-attention with weights
        attn_out, attn_weights = fusion_encoder.sem_topo_cross_attn(
            query=sem_h, key=topo_h, value=topo_h, need_weights=True
        )

        assert attn_weights is not None, "Should return attention weights"
        assert attn_weights.shape[0] == 1, "Batch size should match"
        # attn_weights shape: (batch, num_heads, seq_len_q, seq_len_k)
        # For single vectors: (1, num_heads, 1, 1)


# ======================================================================
# REFACTOR Phase: Channel Sensitivity Analysis
# ======================================================================

class TestChannelSensitivity:
    """Analyze how sensitive the fusion is to each channel.

    This informs task-specific channel weighting:
    - For semantic retrieval: semantic channel should have high sensitivity
    - For citation analysis: topology channel should have high sensitivity
    """

    @pytest.fixture
    def fusion_encoder(self):
        return FourChannelFusionEncoder(
            semantic_dim=64,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=64,
            output_dim=32,
        )

    def _compute_channel_sensitivity(self, encoder, channel_name, n_samples=10):
        """Compute average output change when perturbing one channel."""
        rng = np.random.default_rng(42)
        total_change = 0.0

        for _ in range(n_samples):
            sem = rng.standard_normal((1, 64)).astype(np.float32)
            meta = rng.standard_normal((1, 32)).astype(np.float32)
            topo = rng.standard_normal((1, 32)).astype(np.float32)
            temp = rng.standard_normal((1, 16)).astype(np.float32)

            baseline = encoder.encode_all_from_channels(sem, meta, topo, temp)

            # Perturb the target channel
            epsilon = 0.1
            if channel_name == "semantic":
                sem_perturbed = sem + epsilon * rng.standard_normal((1, 64)).astype(np.float32)
                perturbed = encoder.encode_all_from_channels(sem_perturbed, meta, topo, temp)
            elif channel_name == "topology":
                topo_perturbed = topo + epsilon * rng.standard_normal((1, 32)).astype(np.float32)
                perturbed = encoder.encode_all_from_channels(sem, meta, topo_perturbed, temp)
            elif channel_name == "metadata":
                meta_perturbed = meta + epsilon * rng.standard_normal((1, 32)).astype(np.float32)
                perturbed = encoder.encode_all_from_channels(sem, meta_perturbed, topo, temp)
            elif channel_name == "temporal":
                temp_perturbed = temp + epsilon * rng.standard_normal((1, 16)).astype(np.float32)
                perturbed = encoder.encode_all_from_channels(sem, meta, topo, temp_perturbed)

            change = np.linalg.norm(perturbed - baseline)
            total_change += change

        return total_change / n_samples

    def test_all_channels_have_nonzero_sensitivity(self, fusion_encoder):
        """All channels should have measurable impact on output."""
        for channel in ["semantic", "topology", "metadata", "temporal"]:
            sensitivity = self._compute_channel_sensitivity(fusion_encoder, channel)
            assert sensitivity > 1e-6, f"{channel} channel has zero sensitivity"

    def test_channel_sensitivity_distribution(self, fusion_encoder):
        """Analyze sensitivity distribution across channels.

        This is informational — records current behavior, not a hard assertion.
        The distribution informs task-specific channel weighting strategy.
        """
        sem_sens = self._compute_channel_sensitivity(fusion_encoder, "semantic")
        topo_sens = self._compute_channel_sensitivity(fusion_encoder, "topology")
        meta_sens = self._compute_channel_sensitivity(fusion_encoder, "metadata")
        temp_sens = self._compute_channel_sensitivity(fusion_encoder, "temporal")

        sorted_sens = sorted([
            ("semantic", sem_sens),
            ("topology", topo_sens),
            ("metadata", meta_sens),
            ("temporal", temp_sens),
        ], key=lambda x: x[1], reverse=True)

        # Log the distribution for analysis
        print(f"\nChannel sensitivity distribution: {sorted_sens}")

        # Soft check: all sensitivities should be within reasonable range
        # (not dominated by a single channel)
        max_sens = sorted_sens[0][1]
        min_sens = sorted_sens[-1][1]
        ratio = max_sens / (min_sens + 1e-8)
        assert ratio < 10.0, \
            f"One channel dominates too much (ratio={ratio:.2f}), distribution: {sorted_sens}"


# ======================================================================
# Integration: End-to-End with Real Channel Encoders
# ======================================================================

class TestEndToEndWithRealEncoders:
    """Test fusion with actual channel encoder outputs, not random vectors."""

    def test_fusion_with_fallback_encoders(self):
        """End-to-end test using fallback channel encoders."""
        from retrieval.encoders.channels.semantic_channel import SemanticChannelEncoder
        from retrieval.encoders.channels.metadata_channel import MetadataChannelEncoder
        from retrieval.encoders.channels.temporal_channel import TemporalChannelEncoder

        # Create encoders
        sem_enc = SemanticChannelEncoder(fallback_to_sbert=False)
        meta_enc = MetadataChannelEncoder(output_dim=32)
        temp_enc = TemporalChannelEncoder(output_dim=16, n_periodic=8)

        # Sample documents
        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
            {"id": "p2", "title": "Graph Neural Networks", "abstract": "GNN for citations",
             "authors": ["Bob"], "keywords": ["GNN"], "venue": "ICML", "year": 2021},
        ]

        meta_enc.build_vocab(docs)

        # Encode
        texts = [f"{d['title']} {d['abstract']}" for d in docs]
        sem_emb = sem_enc.encode(texts)
        meta_emb = meta_enc.encode(docs)
        temp_emb = temp_enc.encode([d["year"] for d in docs])
        topo_emb = np.random.randn(2, 32).astype(np.float32)  # mock

        # Fuse
        fusion = FourChannelFusionEncoder(
            semantic_dim=sem_enc.output_dim,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=64,
            output_dim=32,
        )

        manifold_emb = fusion.encode_all_from_channels(
            sem_emb, meta_emb, topo_emb, temp_emb
        )

        assert manifold_emb.shape == (2, 32)
        # L2 normalized
        norms = np.linalg.norm(manifold_emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
