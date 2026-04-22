"""
Tensor Manifold GVF - Unified Retriever Module

Provides unified retrieval over academic knowledge graphs using tensor
manifold embeddings, Grassmann vector fields, and CP decomposition.

Submodules
----------
encoders
    Pluggable embedding encoder interface (SBERT, BGE-M3, etc.).
index
    Pluggable vector index backends (FAISS, HNSWLIB, brute-force).
pipeline
    Tensor signature construction, manifold projection, and decomposition.
retriever
    UnifiedRetriever — the main pipeline orchestrator.
result_formatter
    Output formatting (JSON, Markdown, table, detailed).
"""

from .retriever import RetrievalResult, UnifiedRetriever
from .result_formatter import ResultFormatter
from .encoders import EmbeddingEncoder, SentenceTransformerEncoder, create_encoder
from .index import VectorIndex, FAISSVectorIndex, HNSWVectorIndex, BruteForceIndex, create_index
from .pipeline import ManifoldProjector, SignatureBuilder, TensorDecomposer, DecompositionResult

__all__ = [
    # Main retriever
    "RetrievalResult",
    "UnifiedRetriever",
    "ResultFormatter",
    # Encoders
    "EmbeddingEncoder",
    "SentenceTransformerEncoder",
    "create_encoder",
    # Index
    "VectorIndex",
    "FAISSVectorIndex",
    "HNSWVectorIndex",
    "BruteForceIndex",
    "create_index",
    # Pipeline
    "ManifoldProjector",
    "SignatureBuilder",
    "TensorDecomposer",
    "DecompositionResult",
]
