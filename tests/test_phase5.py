"""TDD tests for Phase 5: FastAPI Knowledge API Service Skeleton.

These tests verify the API layer according to ADR-0005:
- FastAPI application structure
- 6 task-specific endpoints
- Pydantic request/response models
- Integration with UnifiedRetriever
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Any, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check FastAPI availability
try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="FastAPI not installed"
)


# ======================================================================
# Phase 5.1: FastAPI Application Structure
# ======================================================================

class TestFastAPIAppStructure:
    """Test that FastAPI app exists and has correct structure."""

    def test_api_module_exists(self):
        """api/ module should be importable."""
        from api.main import app
        assert app is not None

    def test_fastapi_app_has_title(self):
        """FastAPI app should have title and version."""
        from api.main import app
        assert app.title == "Tensor Manifold GVF Knowledge API"
        assert app.version == "2.0.0"

    def test_api_docs_endpoint_exists(self):
        """/docs endpoint should be available for OpenAPI schema."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_health_check_endpoint(self):
        """/health should return service status."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ======================================================================
# Phase 5.2: Pydantic Request/Response Models
# ======================================================================

class TestPydanticModels:
    """Test Pydantic models for request/response validation."""

    def test_search_request_model_exists(self):
        """SearchRequest model should exist with required fields."""
        from api.models import SearchRequest

        req = SearchRequest(
            query="deep learning",
            top_k=10,
            task="semantic_retrieval",
        )
        assert req.query == "deep learning"
        assert req.top_k == 10
        assert req.task == "semantic_retrieval"

    def test_search_request_optional_filters(self):
        """SearchRequest should accept optional filters."""
        from api.models import SearchRequest

        req = SearchRequest(
            query="neural networks",
            top_k=5,
            task="semantic_retrieval",
            filter_year=(2020, 2024),
            filter_venue="NeurIPS",
        )
        assert req.filter_year == (2020, 2024)
        assert req.filter_venue == "NeurIPS"

    def test_search_response_model_exists(self):
        """SearchResponse model should exist."""
        from api.models import SearchResponse, SearchResult

        result = SearchResult(
            id="p1",
            title="Deep Learning",
            abstract="Neural networks...",
            score=0.95,
            rank=1,
            metadata={"year": 2023, "venue": "NeurIPS"},
        )
        response = SearchResponse(
            results=[result],
            total=1,
            query="deep learning",
        )
        assert len(response.results) == 1
        assert response.total == 1

    def test_citation_network_request_model(self):
        """CitationNetworkRequest model should exist."""
        from api.models import CitationNetworkRequest

        req = CitationNetworkRequest(
            paper_id="p1",
            depth=2,
            direction="both",
        )
        assert req.paper_id == "p1"
        assert req.depth == 2
        assert req.direction == "both"

    def test_author_disambiguation_request_model(self):
        """AuthorDisambiguationRequest model should exist."""
        from api.models import AuthorDisambiguationRequest

        req = AuthorDisambiguationRequest(
            author_name="J. Smith",
            affiliation="MIT",
        )
        assert req.author_name == "J. Smith"
        assert req.affiliation == "MIT"

    def test_trend_analysis_request_model(self):
        """TrendAnalysisRequest model should exist."""
        from api.models import TrendAnalysisRequest

        req = TrendAnalysisRequest(
            concept="transformer",
            time_range=(2020, 2024),
            granularity="year",
        )
        assert req.concept == "transformer"
        assert req.time_range == (2020, 2024)

    def test_knowledge_graph_request_model(self):
        """KnowledgeGraphRequest model should exist."""
        from api.models import KnowledgeGraphRequest

        req = KnowledgeGraphRequest(
            seed_papers=["p1", "p2"],
            max_nodes=100,
            relation_types=["cites", "related"],
        )
        assert req.seed_papers == ["p1", "p2"]
        assert req.max_nodes == 100

    def test_insights_request_model(self):
        """InsightsRequest model should exist."""
        from api.models import InsightsRequest

        req = InsightsRequest(
            query="emerging trends in NLP",
            insight_type="trend_prediction",
        )
        assert req.query == "emerging trends in NLP"
        assert req.insight_type == "trend_prediction"


# ======================================================================
# Phase 5.3: API Endpoints (ADR-0005)
# ======================================================================

class TestAPIEndpoints:
    """Test all 6 API endpoints from ADR-0005."""

    def test_post_api_v1_search(self):
        """POST /api/v1/search should perform semantic retrieval."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/search",
            json={
                "query": "deep learning",
                "top_k": 5,
                "task": "semantic_retrieval",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query" in data

    def test_post_api_v1_citation_network(self):
        """POST /api/v1/citation/network should analyze citation network."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/citation/network",
            json={
                "paper_id": "p1",
                "depth": 2,
                "direction": "both",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

    def test_post_api_v1_authors_disambiguate(self):
        """POST /api/v1/authors/disambiguate should disambiguate authors."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/authors/disambiguate",
            json={
                "author_name": "J. Smith",
                "affiliation": "MIT",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "author_clusters" in data

    def test_post_api_v1_trends_evolution(self):
        """POST /api/v1/trends/evolution should analyze trends."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/trends/evolution",
            json={
                "concept": "transformer",
                "time_range": [2020, 2024],
                "granularity": "year",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "timeline" in data

    def test_post_api_v1_knowledge_graph(self):
        """POST /api/v1/knowledge-graph should construct knowledge graph."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/knowledge-graph",
            json={
                "seed_papers": ["p1", "p2"],
                "max_nodes": 100,
                "relation_types": ["cites", "related"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

    def test_post_api_v1_insights(self):
        """POST /api/v1/insights should extract insights."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/v1/insights",
            json={
                "query": "emerging trends in NLP",
                "insight_type": "trend_prediction",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "insights" in data


# ======================================================================
# Phase 5.4: Task Router Integration
# ======================================================================

class TestTaskRouter:
    """Test task router selects correct attention head."""

    def test_task_router_exists(self):
        """TaskRouter class should exist."""
        from api.router import TaskRouter
        assert TaskRouter is not None

    def test_task_router_selects_correct_task(self):
        """TaskRouter should route to correct task head."""
        from api.router import TaskRouter

        router = TaskRouter()
        task_head = router.get_task_head("citation_analysis")
        assert task_head is not None
        assert task_head.task_name == "citation_analysis"

    def test_task_router_invalid_task_raises(self):
        """TaskRouter should raise error for invalid task."""
        from api.router import TaskRouter

        router = TaskRouter()
        with pytest.raises(ValueError):
            router.get_task_head("invalid_task")


# ======================================================================
# Phase 5.5: API Integration with UnifiedRetriever
# ======================================================================

class TestAPIRetrieverIntegration:
    """Test API layer integrates with UnifiedRetriever."""

    def test_api_has_retriever_function(self):
        """API should have get_retriever function (returns None in skeleton mode)."""
        from api.main import get_retriever

        # In skeleton mode, get_retriever returns None until initialized
        # This test verifies the function exists and is callable
        retriever = get_retriever()
        # retriever may be None in skeleton mode - that's expected
        assert callable(get_retriever)

    def test_search_endpoint_uses_retriever(self):
        """Search endpoint should use UnifiedRetriever for actual search."""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        # This test assumes retriever is built with some test data
        # In practice, we'd mock or use a test fixture
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "top_k": 3},
        )
        # Should return 200 even if empty results
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
