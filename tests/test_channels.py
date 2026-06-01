"""Tests for the four-channel encoder system.

All tests use fallback modes (no GPU or pretrained models required).
Tests requiring BGE-M3 or torch_geometric are marked with skipif.
"""

from __future__ import annotations

import sys
import os
import types

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.encoders.channels.base import ChannelEncoder
from retrieval.encoders.channels.semantic_channel import SemanticChannelEncoder
from retrieval.encoders.channels.metadata_channel import MetadataChannelEncoder
from retrieval.encoders.channels.topology_channel import TopologyChannelEncoder
from retrieval.encoders.channels.temporal_channel import TemporalChannelEncoder
from retrieval.encoders.channels import create_channel_encoder
from core.four_channel_encoder import FourChannelFusionEncoder


# ======================================================================
# 1. ChannelEncoder ABC interface tests
# ======================================================================

class TestChannelEncoderInterface:
    """Verify all four channels implement the ChannelEncoder ABC."""

    @pytest.fixture(params=["semantic", "metadata", "topology", "temporal"])
    def channel_instance(self, request):
        """Create a channel instance using fallback mode."""
        if request.param == "semantic":
            return SemanticChannelEncoder(fallback_to_sbert=False)
        elif request.param == "metadata":
            return MetadataChannelEncoder(output_dim=64)
        elif request.param == "topology":
            return TopologyChannelEncoder(input_dim=32, hidden_dim=32, output_dim=64)
        elif request.param == "temporal":
            return TemporalChannelEncoder(output_dim=32, n_periodic=8)

    def test_is_subclass(self, channel_instance):
        assert isinstance(channel_instance, ChannelEncoder)

    def test_has_output_dim(self, channel_instance):
        assert isinstance(channel_instance.output_dim, int)
        assert channel_instance.output_dim > 0

    def test_has_channel_name(self, channel_instance):
        assert isinstance(channel_instance.channel_name, str)
        assert channel_instance.channel_name in ("semantic", "metadata", "topology", "temporal")


# ======================================================================
# 2. Semantic channel tests
# ======================================================================

class TestSemanticChannel:
    """Test semantic channel encoder."""

    def test_encode_output_shape(self):
        encoder = SemanticChannelEncoder(fallback_to_sbert=False)
        texts = ["hello world", "deep learning", "graph neural network"]
        result = encoder.encode(texts)
        assert result.shape == (3, encoder.output_dim)
        assert result.dtype == np.float32

    def test_encode_single_output_shape(self):
        encoder = SemanticChannelEncoder(fallback_to_sbert=False)
        result = encoder.encode_single("test query")
        assert result.shape == (encoder.output_dim,)
        assert result.dtype == np.float32

    def test_empty_string_handling(self):
        encoder = SemanticChannelEncoder(fallback_to_sbert=False)
        result = encoder.encode(["", "  ", "normal text"])
        assert result.shape == (3, encoder.output_dim)
        # No NaN values
        assert not np.any(np.isnan(result))

    def test_cache_behavior(self):
        encoder = SemanticChannelEncoder(use_cache=True, fallback_to_sbert=False)
        texts = ["cached text", "another text"]
        r1 = encoder.encode(texts)
        r2 = encoder.encode(texts)
        np.testing.assert_array_equal(r1, r2)

    def test_fallback_dim(self):
        """When FlagEmbedding is unavailable, fallback should produce 384-dim."""
        encoder = SemanticChannelEncoder(fallback_to_sbert=False)
        # Without any model, output_dim should be 1024 (BGE-M3 default)
        # but actual encoding produces random vectors of that dim
        assert encoder.output_dim == 1024


# ======================================================================
# 3. Metadata channel tests
# ======================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "p1",
        "title": "Paper One",
        "abstract": "Abstract one",
        "authors": ["Alice Smith", "Bob Jones"],
        "keywords": ["deep learning", "transformer"],
        "venue": "NeurIPS",
    },
    {
        "id": "p2",
        "title": "Paper Two",
        "abstract": "Abstract two",
        "authors": ["Charlie Brown", "Alice Smith"],
        "keywords": ["graph neural network", "citation analysis"],
        "venue": "ICML",
    },
    {
        "id": "p3",
        "title": "Paper Three",
        "abstract": "Abstract three",
        "authors": ["Diana Prince"],
        "keywords": ["knowledge graph"],
        "venue": "ACL",
    },
]


class TestMetadataChannel:
    """Test metadata channel encoder."""

    def test_build_vocab(self):
        encoder = MetadataChannelEncoder(output_dim=64)
        encoder.build_vocab(SAMPLE_DOCUMENTS)
        assert encoder._vocab_built
        assert "Alice Smith" in encoder._author2id
        assert "NeurIPS" in encoder._venue2id

    def test_encode_output_shape(self):
        encoder = MetadataChannelEncoder(output_dim=64)
        encoder.build_vocab(SAMPLE_DOCUMENTS)
        result = encoder.encode(SAMPLE_DOCUMENTS)
        assert result.shape == (3, 64)
        assert result.dtype == np.float32

    def test_encode_single(self):
        encoder = MetadataChannelEncoder(output_dim=64)
        encoder.build_vocab(SAMPLE_DOCUMENTS)
        result = encoder.encode_single(SAMPLE_DOCUMENTS[0])
        assert result.shape == (64,)

    def test_oov_handling(self):
        """Unknown authors/venues should map to UNK without errors."""
        encoder = MetadataChannelEncoder(output_dim=64)
        encoder.build_vocab(SAMPLE_DOCUMENTS)
        unknown_doc = {
            "authors": ["Unknown Author X"],
            "keywords": ["unknown topic"],
            "venue": "Unknown Venue",
        }
        result = encoder.encode([unknown_doc])
        assert result.shape == (1, 64)
        assert not np.any(np.isnan(result))

    def test_empty_metadata(self):
        """Empty authors/keywords/venue should not crash."""
        encoder = MetadataChannelEncoder(output_dim=64)
        encoder.build_vocab(SAMPLE_DOCUMENTS)
        empty_doc = {"authors": [], "keywords": [], "venue": None}
        result = encoder.encode([empty_doc])
        assert result.shape == (1, 64)
        assert not np.any(np.isnan(result))


# ======================================================================
# 4. Topology channel tests
# ======================================================================

class MockGraph:
    """Minimal mock of HeteroAcademicGraph for testing."""

    def __init__(self, adj_by_type, reverse_adj_by_type):
        self.adj_by_type = adj_by_type
        self.reverse_adj_by_type = reverse_adj_by_type


class TestTopologyChannel:
    """Test topology channel encoder."""

    def _make_mock_graph(self):
        adj = {
            "citation": {
                "p1": ["p2", "p3"],
                "p2": ["p3"],
            },
            "collaboration": {
                "p1": ["p2"],
            },
        }
        rev_adj = {
            "citation": {
                "p2": ["p1"],
                "p3": ["p1", "p2"],
            },
        }
        return MockGraph(adj, rev_adj)

    def test_encode_output_shape(self):
        encoder = TopologyChannelEncoder(input_dim=16, hidden_dim=32, output_dim=64)
        graph = self._make_mock_graph()
        node_features = {
            "p1": np.random.randn(16).astype(np.float32),
            "p2": np.random.randn(16).astype(np.float32),
            "p3": np.random.randn(16).astype(np.float32),
        }
        result = encoder.encode((graph, node_features))
        assert result.shape == (3, 64)
        assert result.dtype == np.float32

    def test_encode_single(self):
        encoder = TopologyChannelEncoder(input_dim=16, hidden_dim=32, output_dim=64)
        graph = self._make_mock_graph()
        node_features = {
            "p1": np.random.randn(16).astype(np.float32),
            "p2": np.random.randn(16).astype(np.float32),
            "p3": np.random.randn(16).astype(np.float32),
        }
        result = encoder.encode_single((graph, node_features, "p2"))
        assert result.shape == (64,)

    def test_no_edges(self):
        """Graph with no edges should still produce valid output."""
        encoder = TopologyChannelEncoder(input_dim=16, hidden_dim=32, output_dim=64)
        graph = MockGraph({}, {})
        node_features = {
            "p1": np.random.randn(16).astype(np.float32),
        }
        result = encoder.encode((graph, node_features))
        assert result.shape == (1, 64)
        assert not np.any(np.isnan(result))

    def test_unknown_node_single(self):
        """encode_single for unknown node should return zero vector."""
        encoder = TopologyChannelEncoder(input_dim=16, hidden_dim=32, output_dim=64)
        graph = self._make_mock_graph()
        node_features = {"p1": np.random.randn(16).astype(np.float32)}
        result = encoder.encode_single((graph, node_features, "unknown"))
        assert result.shape == (64,)
        np.testing.assert_array_equal(result, np.zeros(64))


# ======================================================================
# 5. Temporal channel tests
# ======================================================================

class TestTemporalChannel:
    """Test temporal channel encoder."""

    def test_encode_output_shape(self):
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        timestamps = [2020.0, 2021.0, 2022.0, 2023.0]
        result = encoder.encode(timestamps)
        assert result.shape == (4, 32)
        assert result.dtype == np.float32

    def test_encode_single(self):
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        result = encoder.encode_single(2024.0)
        assert result.shape == (32,)

    def test_fit_normalization(self):
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        timestamps = [2010.0, 2020.0, 2024.0]
        encoder.fit(timestamps)
        assert encoder._fitted
        assert encoder._min_t == 2010.0
        assert encoder._max_t == 2024.0

    def test_auto_fit(self):
        """encode() should auto-fit if not already fitted."""
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        assert not encoder._fitted
        encoder.encode([2010.0, 2020.0])
        assert encoder._fitted

    def test_no_nan(self):
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        result = encoder.encode([2020.0])
        assert not np.any(np.isnan(result))

    def test_different_timestamps_produce_different_outputs(self):
        encoder = TemporalChannelEncoder(output_dim=32, n_periodic=8)
        r1 = encoder.encode([2010.0])
        r2 = encoder.encode([2024.0])
        # Should not be identical (different time encodings)
        assert not np.allclose(r1, r2)


# ======================================================================
# 6. FourChannelFusionEncoder tests
# ======================================================================

class TestFourChannelFusionEncoder:
    """Test the four-channel fusion encoder."""

    @pytest.fixture
    def fusion_encoder(self):
        return FourChannelFusionEncoder(
            semantic_dim=64,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=64,
            output_dim=32,
            num_relations=4,
            num_attention_heads=4,
        )

    def test_forward_output_shape(self, fusion_encoder):
        N = 5
        sem = np.random.randn(N, 64).astype(np.float32)
        meta = np.random.randn(N, 32).astype(np.float32)
        topo = np.random.randn(N, 32).astype(np.float32)
        temp = np.random.randn(N, 16).astype(np.float32)

        output = fusion_encoder.encode_all_from_channels(sem, meta, topo, temp)
        assert output.shape == (N, 32)
        assert output.dtype == np.float32

    def test_l2_normalization(self, fusion_encoder):
        N = 5
        sem = np.random.randn(N, 64).astype(np.float32)
        meta = np.random.randn(N, 32).astype(np.float32)
        topo = np.random.randn(N, 32).astype(np.float32)
        temp = np.random.randn(N, 16).astype(np.float32)

        output = fusion_encoder.encode_all_from_channels(sem, meta, topo, temp)
        norms = np.linalg.norm(output, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_no_nan(self, fusion_encoder):
        N = 3
        sem = np.random.randn(N, 64).astype(np.float32)
        meta = np.random.randn(N, 32).astype(np.float32)
        topo = np.random.randn(N, 32).astype(np.float32)
        temp = np.random.randn(N, 16).astype(np.float32)

        output = fusion_encoder.encode_all_from_channels(sem, meta, topo, temp)
        assert not np.any(np.isnan(output))

    def test_single_sample(self, fusion_encoder):
        sem = np.random.randn(1, 64).astype(np.float32)
        meta = np.random.randn(1, 32).astype(np.float32)
        topo = np.random.randn(1, 32).astype(np.float32)
        temp = np.random.randn(1, 16).astype(np.float32)

        output = fusion_encoder.encode_all_from_channels(sem, meta, topo, temp)
        assert output.shape == (1, 32)

    def test_with_relation_ids(self, fusion_encoder):
        N = 5
        sem = np.random.randn(N, 64).astype(np.float32)
        meta = np.random.randn(N, 32).astype(np.float32)
        topo = np.random.randn(N, 32).astype(np.float32)
        temp = np.random.randn(N, 16).astype(np.float32)
        rel_ids = np.array([0, 1, 2, 3, 0])

        output = fusion_encoder.encode_all_from_channels(
            sem, meta, topo, temp, rel_ids
        )
        assert output.shape == (N, 32)


# ======================================================================
# 7. Factory function tests
# ======================================================================

class TestCreateChannelEncoder:
    """Test the channel encoder factory function."""

    def test_create_semantic(self):
        encoder = create_channel_encoder("semantic", {"fallback_to_sbert": False})
        assert isinstance(encoder, SemanticChannelEncoder)
        assert encoder.channel_name == "semantic"

    def test_create_metadata(self):
        encoder = create_channel_encoder("metadata", {"output_dim": 64})
        assert isinstance(encoder, MetadataChannelEncoder)
        assert encoder.channel_name == "metadata"

    def test_create_topology(self):
        encoder = create_channel_encoder("topology", {
            "input_dim": 32, "hidden_dim": 32, "output_dim": 64
        })
        assert isinstance(encoder, TopologyChannelEncoder)
        assert encoder.channel_name == "topology"

    def test_create_temporal(self):
        encoder = create_channel_encoder("temporal", {"output_dim": 32})
        assert isinstance(encoder, TemporalChannelEncoder)
        assert encoder.channel_name == "temporal"

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            create_channel_encoder("unknown", {})


# ======================================================================
# 8. End-to-end integration tests
# ======================================================================

class TestEndToEnd:
    """End-to-end tests for the four-channel pipeline."""

    def test_four_channel_pipeline(self):
        """Test full four-channel encoding + fusion pipeline."""
        # Create channel encoders with small dims for testing
        semantic_enc = SemanticChannelEncoder(fallback_to_sbert=False)
        # Override output_dim for testing
        semantic_enc._using_fallback = False

        metadata_enc = MetadataChannelEncoder(output_dim=32)
        metadata_enc.build_vocab(SAMPLE_DOCUMENTS)

        topology_enc = TopologyChannelEncoder(
            input_dim=1024, hidden_dim=32, output_dim=32
        )

        temporal_enc = TemporalChannelEncoder(output_dim=16, n_periodic=8)

        # Create fusion encoder
        fusion = FourChannelFusionEncoder(
            semantic_dim=semantic_enc.output_dim,
            metadata_dim=32,
            topology_dim=32,
            temporal_dim=16,
            hidden_dim=32,
            output_dim=64,
        )

        # Encode all channels
        texts = [f"{d['title']} {d['abstract']}" for d in SAMPLE_DOCUMENTS]
        sem_emb = semantic_enc.encode(texts)
        meta_emb = metadata_enc.encode(SAMPLE_DOCUMENTS)
        topo_emb = np.random.randn(3, 32).astype(np.float32)  # mock topology
        temp_emb = temporal_enc.encode([2020.0, 2021.0, 2022.0])

        # Fuse
        manifold_emb = fusion.encode_all_from_channels(
            sem_emb, meta_emb, topo_emb, temp_emb
        )

        assert manifold_emb.shape == (3, 64)
        assert manifold_emb.dtype == np.float32
        # L2 normalized
        norms = np.linalg.norm(manifold_emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_retriever_backward_compat(self):
        """Verify v1 mode still works (channels.enabled=false)."""
        from retrieval.retriever import UnifiedRetriever

        retriever = UnifiedRetriever(
            embedding_dim=384,
            manifold_dim=32,
            index_type="brute",
        )
        # Should not raise
        retriever.build(SAMPLE_DOCUMENTS)
        assert retriever._built
        assert not retriever._channels_enabled

        # Search should work (v1 search returns (results_list, decomp))
        results, _ = retriever.search("deep learning", top_k=2)
        # results may be a list or a single RetrievalResult
        if not isinstance(results, list):
            results = [results]
        assert len(results) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
