"""Pydantic models for Knowledge API request/response validation.

See ADR-0005 for API endpoint specifications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ======================================================================
# Search Endpoint Models
# ======================================================================

class SearchRequest(BaseModel):
    """Request model for semantic search endpoint.
    
    Parameters
    ----------
    query : str
        Natural language query text.
    top_k : int
        Number of results to return (default: 10).
    task : str
        Task type for task-specific attention head.
        One of: "semantic_retrieval", "citation_analysis", 
        "author_disambiguation", "trend_analysis".
    filter_year : tuple[int, int], optional
        Year range filter [min_year, max_year].
    filter_venue : str, optional
        Venue name filter (case-insensitive substring match).
    """
    query: str = Field(..., description="Natural language query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    task: str = Field(
        default="semantic_retrieval",
        description="Task type for attention head selection"
    )
    filter_year: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Year range filter [min, max]"
    )
    filter_venue: Optional[str] = Field(
        default=None,
        description="Venue name filter"
    )


class SearchResult(BaseModel):
    """Single search result item."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    rank: int = Field(..., ge=1, description="Result rank")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (year, venue, authors, keywords)"
    )


class SearchResponse(BaseModel):
    """Response model for semantic search endpoint."""
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., ge=0, description="Total number of results")
    query: str = Field(..., description="Original query")


# ======================================================================
# Citation Network Endpoint Models
# ======================================================================

class CitationNetworkRequest(BaseModel):
    """Request model for citation network analysis."""
    paper_id: str = Field(..., description="Seed paper ID")
    depth: int = Field(default=2, ge=1, le=5, description="Citation graph depth")
    direction: str = Field(
        default="both",
        description="Citation direction: 'in', 'out', or 'both'"
    )


class CitationNetworkNode(BaseModel):
    """Node in citation network graph."""
    id: str
    title: str
    year: Optional[int] = None
    citation_count: int = 0


class CitationNetworkEdge(BaseModel):
    """Edge in citation network graph."""
    source: str
    target: str
    relation: str = "cites"


class CitationNetworkResponse(BaseModel):
    """Response model for citation network analysis."""
    nodes: List[CitationNetworkNode]
    edges: List[CitationNetworkEdge]
    seed_paper: str


# ======================================================================
# Author Disambiguation Endpoint Models
# ======================================================================

class AuthorDisambiguationRequest(BaseModel):
    """Request model for author entity disambiguation."""
    author_name: str = Field(..., description="Author name to disambiguate")
    affiliation: Optional[str] = Field(None, description="Optional affiliation hint")


class AuthorCluster(BaseModel):
    """Cluster of papers for a disambiguated author entity."""
    cluster_id: str
    author_name: str
    affiliation: Optional[str] = None
    paper_ids: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class AuthorDisambiguationResponse(BaseModel):
    """Response model for author disambiguation."""
    author_clusters: List[AuthorCluster]
    query_name: str


# ======================================================================
# Trend Analysis Endpoint Models
# ======================================================================

class TrendAnalysisRequest(BaseModel):
    """Request model for research trend analysis."""
    concept: str = Field(..., description="Research concept to analyze")
    time_range: Tuple[int, int] = Field(..., description="Year range [start, end]")
    granularity: str = Field(
        default="year",
        description="Time granularity: 'year', 'month'"
    )


class TrendPoint(BaseModel):
    """Single point in trend timeline."""
    year: int
    count: int
    avg_citations: float
    top_papers: List[str]


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""
    timeline: List[TrendPoint]
    concept: str
    growth_rate: float


# ======================================================================
# Knowledge Graph Endpoint Models
# ======================================================================

class KnowledgeGraphRequest(BaseModel):
    """Request model for knowledge graph construction."""
    seed_papers: List[str] = Field(..., min_length=1, description="Seed paper IDs")
    max_nodes: int = Field(default=100, ge=10, le=1000)
    relation_types: List[str] = Field(
        default=["cites", "related"],
        description="Relation types to include"
    )


class KnowledgeGraphNode(BaseModel):
    """Node in knowledge graph."""
    id: str
    type: str  # "paper", "author", "concept"
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphEdge(BaseModel):
    """Edge in knowledge graph."""
    source: str
    target: str
    relation: str
    weight: Optional[float] = None


class KnowledgeGraphResponse(BaseModel):
    """Response model for knowledge graph construction."""
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]
    seed_papers: List[str]


# ======================================================================
# Insights Endpoint Models
# ======================================================================

class InsightsRequest(BaseModel):
    """Request model for insight extraction."""
    query: str = Field(..., description="Research question or topic")
    insight_type: str = Field(
        default="trend_prediction",
        description="Type of insight: 'trend_prediction', 'gap_analysis', 'inspiration'"
    )


class InsightItem(BaseModel):
    """Single insight item."""
    type: str
    description: str
    supporting_papers: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class InsightsResponse(BaseModel):
    """Response model for insight extraction."""
    insights: List[InsightItem]
    query: str
