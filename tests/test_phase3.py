"""TDD tests for Phase 3: Task head integration, weight learning, and visualization.

These tests verify:
1. TaskSpecificAttentionHead integration into UnifiedRetriever
2. Learning channel weights from training data
3. Attention weight visualization tools
"""

from __future__ import annotations

import sys
import os
import json
from typing import Dict, Any

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Phase 3.1: Task Head Integration into UnifiedRetriever
# ======================================================================

class TestTaskHeadIntegrationIntoRetriever:
    """Test that UnifiedRetriever supports task-specific attention heads."""

    def test_retriever_accepts_task_name(self):
        """UnifiedRetriever should accept a task_name parameter."""
        from retrieval.retriever import UnifiedRetriever

        retriever = UnifiedRetriever(
            embedding_dim=384,
            manifold_dim=32,
            index_type="brute",
            task_name="semantic_retrieval",
        )

        assert retriever.task_name == "semantic_retrieval"

    def test_retriever_has_task_heads(self):
        """UnifiedRetriever should have a _task_heads dict."""
        from retrieval.retriever import UnifiedRetriever

        retriever = UnifiedRetriever(
            embedding_dim=384,
            manifold_dim=32,
            index_type="brute",
            task_name="semantic_retrieval",
        )

        # Should initialize task heads when four-channel is enabled
        assert hasattr(retriever, "_task_heads")

    def test_retriever_four_channel_with_task_head(self):
        """UnifiedRetriever with four-channel mode should use task head for search."""
        from retrieval.retriever import UnifiedRetriever

        # Create a simple config with channels enabled
        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {
                "hidden_dim": 64,
                "output_dim": 32,
                "num_attention_heads": 4,
            },
        }

        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
        ]

        retriever = UnifiedRetriever(
            manifold_dim=32,
            index_type="brute",
            channels_config=channels_config,
            task_name="semantic_retrieval",
        )
        # _channels_enabled is set inside _init_components() which is called by build()
        retriever.build(docs)

        assert retriever._channels_enabled
        assert "semantic_retrieval" in retriever._task_heads

    def test_retriever_default_task_is_semantic_retrieval(self):
        """Default task should be semantic_retrieval."""
        from retrieval.retriever import UnifiedRetriever

        retriever = UnifiedRetriever(
            embedding_dim=384,
            manifold_dim=32,
            index_type="brute",
        )

        assert retriever.task_name == "semantic_retrieval"


# ======================================================================
# Phase 3.2: Learning Channel Weights from Training Data
# ======================================================================

class TestChannelWeightLearning:
    """Test learning channel weights from labeled training data."""

    def test_learner_initializes(self):
        """ChannelWeightLearner should initialize with training data."""
        from core.channel_weight_learner import ChannelWeightLearner

        # Sample training data: (query_embedding, doc_embedding, label)
        # where label=1 for relevant, 0 for irrelevant
        train_data = [
            {
                "semantic": np.random.randn(64).astype(np.float32),
                "metadata": np.random.randn(32).astype(np.float32),
                "topology": np.random.randn(32).astype(np.float32),
                "temporal": np.random.randn(16).astype(np.float32),
                "label": 1.0,
            }
            for _ in range(10)
        ]

        learner = ChannelWeightLearner(
            hidden_dim=64,
            output_dim=32,
            learning_rate=0.001,
        )

        assert learner is not None
        assert hasattr(learner, "train")

    def test_learner_updates_weights(self):
        """Training should update channel weights."""
        from core.channel_weight_learner import ChannelWeightLearner

        # All channels use hidden_dim=32 as per TaskSpecificAttentionHead docstring:
        # "Dimension of channel embeddings (all channels projected to this)"
        train_data = [
            {
                "semantic": np.random.randn(32).astype(np.float32),
                "metadata": np.random.randn(32).astype(np.float32),
                "topology": np.random.randn(32).astype(np.float32),
                "temporal": np.random.randn(32).astype(np.float32),
                "label": 1.0,
            }
            for _ in range(20)
        ]

        learner = ChannelWeightLearner(hidden_dim=32, output_dim=32)

        initial_weights = learner.get_global_weights().copy()

        # Train for a few steps
        learner.train(train_data, epochs=5)

        updated_weights = learner.get_global_weights()

        # Weights should have changed
        assert not np.allclose(initial_weights, updated_weights, atol=1e-4)

    def test_learner_produces_learned_task_head(self):
        """After training, learner should produce a trained task head."""
        from core.channel_weight_learner import ChannelWeightLearner

        # All channels use hidden_dim=32 per TaskSpecificAttentionHead docstring
        train_data = [
            {
                "semantic": np.random.randn(32).astype(np.float32),
                "metadata": np.random.randn(32).astype(np.float32),
                "topology": np.random.randn(32).astype(np.float32),
                "temporal": np.random.randn(32).astype(np.float32),
                "label": 1.0 if i % 2 == 0 else 0.0,
            }
            for i in range(30)
        ]

        learner = ChannelWeightLearner(hidden_dim=32, output_dim=32)
        learner.train(train_data, epochs=10)

        task_head = learner.get_learned_task_head("semantic_retrieval")

        assert task_head is not None
        assert task_head.task_name == "semantic_retrieval"

    def test_learner_saves_and_loads(self):
        """Learner should support save/load for persistence."""
        from core.channel_weight_learner import ChannelWeightLearner
        import tempfile

        learner = ChannelWeightLearner(hidden_dim=64, output_dim=32)
        learner.train([], epochs=0)  # Just initialize

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            learner.save(temp_path)
            loaded_learner = ChannelWeightLearner.load(temp_path)

            original_weights = learner.get_global_weights()
            loaded_weights = loaded_learner.get_global_weights()

            np.testing.assert_allclose(original_weights, loaded_weights, rtol=1e-5)
        finally:
            os.unlink(temp_path)


# ======================================================================
# Phase 3.3: Attention Weight Visualization
# ======================================================================

class TestAttentionVisualization:
    """Test attention weight visualization tools."""

    def test_visualizer_initializes(self):
        """VisualizationToolkit should initialize."""
        from core.visualization import AttentionVisualizer

        viz = AttentionVisualizer()
        assert viz is not None

    def test_plot_channel_weights_returns_figure(self):
        """plot_channel_weights should return a matplotlib Figure."""
        from core.visualization import AttentionVisualizer
        from matplotlib.figure import Figure

        viz = AttentionVisualizer()

        weights = {
            "semantic": 0.40,
            "metadata": 0.20,
            "topology": 0.25,
            "temporal": 0.15,
        }

        fig = viz.plot_channel_weights(
            weights,
            task_name="semantic_retrieval",
            output_path=None,  # Don't save, just return
        )

        assert isinstance(fig, Figure)

    def test_plot_attention_heatmap(self):
        """plot_attention_heatmap should return a matplotlib Figure."""
        from core.visualization import AttentionVisualizer
        from matplotlib.figure import Figure

        viz = AttentionVisualizer()

        # Sample attention weights from cross-attention
        attn_weights = np.random.rand(4, 4).astype(np.float32)
        attn_weights = attn_weights / attn_weights.sum(axis=1, keepdims=True)

        fig = viz.plot_attention_heatmap(
            attn_weights,
            title="Semantic-Topology Attention",
            output_path=None,
        )

        assert isinstance(fig, Figure)

    def test_plot_sensitivity_comparison(self):
        """plot_sensitivity_comparison should compare multiple tasks."""
        from core.visualization import AttentionVisualizer
        from matplotlib.figure import Figure

        viz = AttentionVisualizer()

        reports = [
            {
                "task_name": "semantic_retrieval",
                "channel_sensitivities": {
                    "semantic": 0.5, "metadata": 0.2,
                    "topology": 0.2, "temporal": 0.1,
                },
            },
            {
                "task_name": "citation_analysis",
                "channel_sensitivities": {
                    "semantic": 0.1, "metadata": 0.1,
                    "topology": 0.6, "temporal": 0.2,
                },
            },
        ]

        fig = viz.plot_sensitivity_comparison(reports, output_path=None)
        assert isinstance(fig, Figure)

    def test_export_sensitivity_report_json(self):
        """export_sensitivity_report should save JSON report."""
        from core.visualization import AttentionVisualizer
        import tempfile

        viz = AttentionVisualizer()

        report = {
            "task_name": "semantic_retrieval",
            "channel_sensitivities": {
                "semantic": 0.5, "metadata": 0.2,
                "topology": 0.2, "temporal": 0.1,
            },
            "most_sensitive_channel": "semantic",
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            viz.export_sensitivity_report(report, temp_path)

            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert loaded["task_name"] == "semantic_retrieval"
            assert "channel_sensitivities" in loaded
        finally:
            os.unlink(temp_path)


# ======================================================================
# Phase 3.4: Qdrant Integration Readiness Check
# ======================================================================

class TestQdrantIntegrationReadiness:
    """Test if conditions are met for Qdrant integration."""

    def test_qdrant_client_available(self):
        """Should detect if qdrant-client is installed and skip if not."""
        try:
            import qdrant_client
        except ImportError:
            pytest.skip("qdrant_client not installed — readiness check deferred")

        # If installed, verify it's importable
        assert qdrant_client is not None

    def test_named_vectors_config_structure(self):
        """UnifiedRetriever should support named vectors config."""
        from retrieval.retriever import UnifiedRetriever

        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {"hidden_dim": 64, "output_dim": 32},
        }

        retriever = UnifiedRetriever(
            manifold_dim=32,
            channels_config=channels_config,
            task_name="semantic_retrieval",
        )

        # Should expose channel output dimensions
        assert hasattr(retriever, "_channel_encoders")


# ======================================================================
# Integration: End-to-End with Task Heads
# ======================================================================

class TestEndToEndWithTaskHeads:
    """End-to-end test with task-specific attention heads."""

    def test_full_pipeline_with_task_head(self):
        """Test complete pipeline: channels → fusion → task head → index."""
        from retrieval.retriever import UnifiedRetriever
        from retrieval.encoders.channels.semantic_channel import SemanticChannelEncoder
        from retrieval.encoders.channels.metadata_channel import MetadataChannelEncoder
        from retrieval.encoders.channels.temporal_channel import TemporalChannelEncoder
        from core.four_channel_encoder import FourChannelFusionEncoder
        from core.task_attention import create_task_head

        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
            {"id": "p2", "title": "Graph Neural Networks", "abstract": "GNN for citations",
             "authors": ["Bob"], "keywords": ["GNN"], "venue": "ICML", "year": 2021},
        ]

        # Encode
        sem_enc = SemanticChannelEncoder(fallback_to_sbert=False)
        meta_enc = MetadataChannelEncoder(output_dim=32)
        temp_enc = TemporalChannelEncoder(output_dim=16, n_periodic=8)
        meta_enc.build_vocab(docs)

        texts = [f"{d['title']} {d['abstract']}" for d in docs]
        sem_emb = sem_enc.encode(texts)
        meta_emb = meta_enc.encode(docs)
        temp_emb = temp_enc.encode([d["year"] for d in docs])
        topo_emb = np.random.randn(2, 32).astype(np.float32)

        # Fusion
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

        # Task head
        task_head = create_task_head(
            "semantic_retrieval",
            hidden_dim=32,  # fusion output dim
            output_dim=32,
        )

        task_emb = task_head.forward(
            torch.tensor(manifold_emb[:, :32]),
            torch.tensor(manifold_emb[:, :32]),
            torch.tensor(manifold_emb[:, :32]),
            torch.tensor(manifold_emb[:, :32]),
        )

        assert task_emb.shape == (2, 32)
        # L2 normalized
        norms = torch.norm(task_emb, dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
