"""TDD tests for Phase 4: Qdrant adapter and UnifiedRetriever integration.

These tests verify:
1. QdrantStore adapter implements the storage interface
2. Four-channel named vectors are correctly stored and retrieved
3. UnifiedRetriever can route build/search through Qdrant
4. Payload filtering works with Qdrant filters
5. Docker Compose configuration is valid
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, List

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check qdrant_client availability
try:
    import qdrant_client
    from qdrant_client import QdrantClient, models
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _QDRANT_AVAILABLE,
    reason="qdrant_client not installed"
)


# ======================================================================
# Phase 4.1: QdrantStore Adapter Interface
# ======================================================================

class TestQdrantStoreInterface:
    """Test that QdrantStore implements the expected storage interface."""

    def test_qdrant_store_class_exists(self):
        """QdrantStore class should be importable."""
        from retrieval.index.qdrant_store import QdrantStore
        assert QdrantStore is not None

    def test_qdrant_store_initializes_with_config(self):
        """QdrantStore should accept connection config."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_collection",
            host="localhost",
            port=6333,
        )
        assert store.collection_name == "test_collection"

    def test_qdrant_store_creates_collection_with_named_vectors(self):
        """QdrantStore should create a collection with four named vectors."""
        from retrieval.index.qdrant_store import QdrantStore

        # Use in-memory Qdrant for testing
        store = QdrantStore(
            collection_name="test_named_vectors",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        # Verify collection exists
        info = store.client.get_collection("test_named_vectors")
        assert info is not None
        # Named vectors should be configured
        assert info.config.params.vectors is not None


# ======================================================================
# Phase 4.2: Four-Channel Named Vector Storage
# ======================================================================

class TestFourChannelStorage:
    """Test storing and retrieving four-channel named vectors."""

    def test_upsert_documents_with_named_vectors(self):
        """Should upsert documents with four named vectors + payload."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_upsert",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        # Sample documents with four-channel embeddings
        documents = [
            {
                "id": "p1",
                "title": "Deep Learning",
                "abstract": "Neural networks",
                "authors": ["Alice"],
                "year": 2020,
                "venue": "NeurIPS",
                "keywords": ["DL"],
            },
            {
                "id": "p2",
                "title": "Graph Neural Networks",
                "abstract": "GNN for citations",
                "authors": ["Bob"],
                "year": 2021,
                "venue": "ICML",
                "keywords": ["GNN"],
            },
        ]

        channel_embeddings = {
            "semantic": np.random.randn(2, 64).astype(np.float32),
            "metadata": np.random.randn(2, 32).astype(np.float32),
            "topology": np.random.randn(2, 32).astype(np.float32),
            "temporal": np.random.randn(2, 16).astype(np.float32),
        }

        store.upsert(documents, channel_embeddings)

        # Verify count
        assert store.count() == 2

    def test_search_by_named_vector(self):
        """Should search using a specific named vector channel."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_search",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        documents = [
            {"id": "p1", "title": "Paper A", "year": 2020, "venue": "NeurIPS"},
            {"id": "p2", "title": "Paper B", "year": 2021, "venue": "ICML"},
            {"id": "p3", "title": "Paper C", "year": 2022, "venue": "NeurIPS"},
        ]

        channel_embeddings = {
            "semantic": np.random.randn(3, 64).astype(np.float32),
            "metadata": np.random.randn(3, 32).astype(np.float32),
            "topology": np.random.randn(3, 32).astype(np.float32),
            "temporal": np.random.randn(3, 16).astype(np.float32),
        }

        store.upsert(documents, channel_embeddings)

        # Search by semantic channel
        query_vector = channel_embeddings["semantic"][0:1]  # Use first doc as query
        results = store.search(
            query_vector=query_vector,
            channel="semantic",
            top_k=2,
        )

        assert len(results) >= 1
        assert "id" in results[0]
        assert "score" in results[0]

    def test_search_with_payload_filter(self):
        """Should support payload-based filtering (year, venue)."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_filter",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        documents = [
            {"id": "p1", "title": "Paper A", "year": 2020, "venue": "NeurIPS"},
            {"id": "p2", "title": "Paper B", "year": 2023, "venue": "ICML"},
            {"id": "p3", "title": "Paper C", "year": 2024, "venue": "NeurIPS"},
        ]

        channel_embeddings = {
            "semantic": np.random.randn(3, 64).astype(np.float32),
            "metadata": np.random.randn(3, 32).astype(np.float32),
            "topology": np.random.randn(3, 32).astype(np.float32),
            "temporal": np.random.randn(3, 16).astype(np.float32),
        }

        store.upsert(documents, channel_embeddings)

        # Search with year filter
        query_vector = np.random.randn(1, 64).astype(np.float32)
        results = store.search(
            query_vector=query_vector,
            channel="semantic",
            top_k=10,
            filter_year=(2023, 2025),
        )

        # All results should have year >= 2023
        for r in results:
            assert r.get("payload", {}).get("year", 0) >= 2023

    def test_search_by_multiple_channels(self):
        """Should support searching across different named vectors."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_multi_channel",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        documents = [
            {"id": "p1", "title": "Paper A", "year": 2020},
            {"id": "p2", "title": "Paper B", "year": 2021},
        ]

        channel_embeddings = {
            "semantic": np.random.randn(2, 64).astype(np.float32),
            "metadata": np.random.randn(2, 32).astype(np.float32),
            "topology": np.random.randn(2, 32).astype(np.float32),
            "temporal": np.random.randn(2, 16).astype(np.float32),
        }

        store.upsert(documents, channel_embeddings)

        # Search by topology channel
        query_vector = np.random.randn(1, 32).astype(np.float32)
        results = store.search(
            query_vector=query_vector,
            channel="topology",
            top_k=2,
        )

        assert len(results) >= 1

    def test_delete_document(self):
        """Should support deleting documents by ID."""
        from retrieval.index.qdrant_store import QdrantStore

        store = QdrantStore(
            collection_name="test_delete",
            host=":memory:",
        )

        channel_dims = {
            "semantic": 64,
            "metadata": 32,
            "topology": 32,
            "temporal": 16,
        }
        store.create_collection(channel_dims)

        documents = [
            {"id": "p1", "title": "Paper A", "year": 2020},
            {"id": "p2", "title": "Paper B", "year": 2021},
        ]

        channel_embeddings = {
            "semantic": np.random.randn(2, 64).astype(np.float32),
            "metadata": np.random.randn(2, 32).astype(np.float32),
            "topology": np.random.randn(2, 32).astype(np.float32),
            "temporal": np.random.randn(2, 16).astype(np.float32),
        }

        store.upsert(documents, channel_embeddings)
        assert store.count() == 2

        store.delete(["p1"])
        assert store.count() == 1


# ======================================================================
# Phase 4.3: UnifiedRetriever Qdrant Integration
# ======================================================================

class TestRetrieverQdrantIntegration:
    """Test UnifiedRetriever integration with Qdrant backend."""

    def test_retriever_uses_qdrant_when_configured(self):
        """UnifiedRetriever should use Qdrant when qdrant config is provided."""
        from retrieval.retriever import UnifiedRetriever

        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {"hidden_dim": 64, "output_dim": 32},
            "qdrant": {
                "enabled": True,
                "collection_name": "test_retriever_qdrant",
                "host": ":memory:",
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
        retriever.build(docs)

        # Should have a Qdrant store
        assert hasattr(retriever, "_qdrant_store")
        assert retriever._qdrant_store is not None

    def test_retriever_stores_four_channels_in_qdrant(self):
        """Build should store all four channels as named vectors."""
        from retrieval.retriever import UnifiedRetriever

        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {"hidden_dim": 64, "output_dim": 32},
            "qdrant": {
                "enabled": True,
                "collection_name": "test_four_channels",
                "host": ":memory:",
            },
        }

        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
            {"id": "p2", "title": "Graph Neural Networks", "abstract": "GNN for citations",
             "authors": ["Bob"], "keywords": ["GNN"], "venue": "ICML", "year": 2021},
        ]

        retriever = UnifiedRetriever(
            manifold_dim=32,
            index_type="brute",
            channels_config=channels_config,
            task_name="semantic_retrieval",
        )
        retriever.build(docs)

        # Qdrant store should have all documents
        assert retriever._qdrant_store.count() == 2

    def test_retriever_search_uses_qdrant(self):
        """Search should route through Qdrant when configured."""
        from retrieval.retriever import UnifiedRetriever

        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {"hidden_dim": 64, "output_dim": 32},
            "qdrant": {
                "enabled": True,
                "collection_name": "test_search_qdrant",
                "host": ":memory:",
            },
        }

        docs = [
            {"id": "p1", "title": "Deep Learning", "abstract": "Neural networks",
             "authors": ["Alice"], "keywords": ["DL"], "venue": "NeurIPS", "year": 2020},
            {"id": "p2", "title": "Graph Neural Networks", "abstract": "GNN for citations",
             "authors": ["Bob"], "keywords": ["GNN"], "venue": "ICML", "year": 2021},
        ]

        retriever = UnifiedRetriever(
            manifold_dim=32,
            index_type="brute",
            channels_config=channels_config,
            task_name="semantic_retrieval",
        )
        retriever.build(docs)

        # Search should return results
        results = retriever.search("deep learning", top_k=2)
        assert len(results) >= 1
        assert results[0].id in ["p1", "p2"]

    def test_retriever_fallback_when_qdrant_unavailable(self):
        """Should fallback to in-memory index when Qdrant is not available."""
        from retrieval.retriever import UnifiedRetriever

        channels_config = {
            "enabled": True,
            "semantic": {"fallback_to_sbert": False},
            "metadata": {"output_dim": 32},
            "topology": {"input_dim": 1024, "hidden_dim": 32, "output_dim": 32},
            "temporal": {"output_dim": 16},
            "fusion": {"hidden_dim": 64, "output_dim": 32},
            "qdrant": {
                "enabled": True,
                "collection_name": "test_fallback",
                "host": "nonexistent:99999",  # Unreachable host
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
        # Should not raise — fallback to in-memory index
        retriever.build(docs)
        assert retriever._built is True


# ======================================================================
# Phase 4.4: Docker Compose Configuration
# ======================================================================

class TestDockerComposeConfig:
    """Test Docker Compose configuration for Qdrant deployment."""

    def test_docker_compose_file_exists(self):
        """docker-compose.yml should exist in project root."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        assert os.path.isfile(compose_path), f"Not found: {compose_path}"

    def test_docker_compose_has_qdrant_service(self):
        """docker-compose.yml should define a qdrant service."""
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path, "r") as f:
            content = f.read()

        assert "qdrant" in content.lower()
        # Should specify port
        assert "6333" in content

    def test_docker_compose_valid_yaml(self):
        """docker-compose.yml should be valid YAML."""
        import yaml
        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert "services" in config or "version" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
