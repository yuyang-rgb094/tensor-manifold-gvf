"""
Vector index abstraction layer for tensor-manifold-gvf retrieval.

Provides a unified interface for building, searching, and managing
vector similarity indices across multiple backends (FAISS, HNSWLIB,
brute-force numpy).
"""

from .base import VectorIndex
from .brute_index import BruteForceIndex
from .faiss_index import FAISSVectorIndex
from .hnswlib_index import HNSWVectorIndex

__all__ = [
    "VectorIndex",
    "FAISSVectorIndex",
    "HNSWVectorIndex",
    "BruteForceIndex",
    "create_index",
]


def create_index(index_type: str, dim: int = None, **kwargs) -> VectorIndex:
    """Factory function for creating vector index instances.

    Args:
        index_type: Type of index to create. Supported values are
            ``"faiss"``, ``"hnswlib"``, and ``"brute"`` (default).
        dim: Embedding dimensionality (kept for API compatibility;
            individual backends may ignore it).
        **kwargs: Additional keyword arguments forwarded to the
            concrete index constructor.

    Returns:
        A :class:`VectorIndex` instance of the requested type.

    Raises:
        ValueError: If *index_type* is not recognised.
    """
    if index_type == "faiss":
        return FAISSVectorIndex(**kwargs)
    elif index_type == "hnswlib":
        return HNSWVectorIndex(**kwargs)
    elif index_type in ("brute", "numpy", "bruteforce"):
        return BruteForceIndex(**kwargs)
    else:
        raise ValueError(
            f"Unknown index type '{index_type}'. "
            f"Supported types: 'faiss', 'hnswlib', 'brute'."
        )
