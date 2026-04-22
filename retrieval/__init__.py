"""
Tensor Manifold GVF - Unified Retriever Module

Provides unified retrieval over academic knowledge graphs using tensor
manifold embeddings, Grassmann vector fields, and CP decomposition.
"""

from .retriever import RetrievalResult, UnifiedRetriever
from .result_formatter import ResultFormatter

__all__ = ["RetrievalResult", "UnifiedRetriever", "ResultFormatter"]
