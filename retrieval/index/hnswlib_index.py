"""
HNSWLIB-backed vector index implementation.

Provides approximate nearest-neighbour search via the ``hnswlib``
library.  If ``hnswlib`` is not installed, :meth:`build` raises
:exc:`ImportError` with a descriptive message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import VectorIndex

logger = logging.getLogger("tensor_manifold_gvf.retrieval.index")

try:
    import hnswlib

    _HAS_HNSWLIB = True
except ImportError:  # pragma: no cover
    _HAS_HNSWLIB = False


class HNSWVectorIndex(VectorIndex):
    """Vector index backed by HNSWLIB (Hierarchical Navigable Small World).

    Args:
        ef_construction: Construction-time beam width.  Higher values
            yield better recall at the cost of slower indexing.
        M: Maximum number of connections per node in the HNSW graph.
        space: Distance metric -- ``"ip"`` (inner product, default) or
            ``"l2"`` (squared Euclidean).
    """

    def __init__(
        self,
        ef_construction: int = 200,
        ef_search: int = 128,
        M: int = 16,
        space: str = "ip",
    ) -> None:
        if space not in ("ip", "l2"):
            raise ValueError(f"space must be 'ip' or 'l2', got '{space}'")
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._M = M
        self._space = space

        self._index: Optional[object] = None
        self._dim: Optional[int] = None
        self._n_total: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray) -> None:
        """Build the HNSW index from *embeddings*.

        Args:
            embeddings: 2-D float array of shape ``(N, D)``.

        Raises:
            ImportError: If ``hnswlib`` is not installed.
        """
        if not _HAS_HNSWLIB:
            logger.warning(
                "hnswlib is not installed; cannot build HNSWVectorIndex. "
                "Install with: pip install hnswlib"
            )
            raise ImportError(
                "hnswlib is not installed. Install with: pip install hnswlib"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._dim = embeddings.shape[1]
        N = embeddings.shape[0]

        index = hnswlib.Index(space=self._space, dim=self._dim)
        index.init_index(
            max_elements=N,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        index.add_items(embeddings)

        self._index = index
        self._n_total = N
        logger.info(
            "Built HNSW index: N=%d, d=%d, ef_construction=%d, M=%d, space=%s",
            N, self._dim, self._ef_construction, self._M, self._space,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the *top_k* nearest neighbours.

        Args:
            query: 1-D ``(D,)`` or 2-D ``(Q, D)`` query vector(s).
            top_k: Number of results to return.

        Returns:
            ``(scores, indices)`` -- each of shape ``(Q, top_k)`` or
            ``(top_k,)``.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        # Clamp top_k to available elements
        actual_k = min(top_k, self._n_total)
        if actual_k <= 0:
            actual_k = 1

        # Set ef_search to be at least actual_k, preferably larger
        ef = max(self._ef_search, actual_k * 2)
        self._index.set_ef(ef)

        single = query.ndim == 1
        if single:
            query = query.reshape(1, -1)

        query = np.ascontiguousarray(query, dtype=np.float32)

        # hnswlib returns (labels, distances); we swap to (scores, indices)
        # and negate distances for L2 so that higher = more similar.
        labels, distances = self._index.knn_query(query, k=actual_k)

        if self._space == "l2":
            # Convert to similarity: negate so higher is better
            scores = -distances
        else:
            scores = distances

        # Pad with -1 if we need more results than available
        if actual_k < top_k:
            pad_len = top_k - actual_k
            labels = np.pad(labels, ((0, 0), (0, pad_len)), constant_values=-1)
            scores = np.pad(scores, ((0, 0), (0, pad_len)), constant_values=-np.inf)

        if single:
            scores = scores[0]
            labels = labels[0]

        return scores, labels

    # ------------------------------------------------------------------
    # Incremental add
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray) -> None:
        """Add new vectors to the existing index.

        Args:
            embeddings: 2-D float array of shape ``(M, D)``.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add_items(embeddings)
        self._n_total += embeddings.shape[0]
        logger.debug(
            "Added %d vectors to HNSW index (total: %d)",
            embeddings.shape[0], self._n_total,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the HNSW index to disk.

        Args:
            path: Destination file path or directory.
                  If directory, saves as "hnsw_index.bin within it.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")
        path_obj = Path(path)
        if path_obj.is_dir():
            path_obj = str(path_obj / "hnsw_index.bin")
        else:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(path)
        logger.info("Saved HNSW index to %s", path)

    def load(self, path: str) -> None:
        """Load an HNSW index from disk.

        Args:
            path: Source file path or directory.
                  If directory, loads "hnsw_index.bin" within it.

        Raises:
            ImportError: If ``hnswlib`` is not installed.
        """
        if not _HAS_HNSWLIB:
            raise ImportError(
                "hnswlib is not installed. Install with: pip install hnswlib"
            )
        path_obj = Path(path)
        if path_obj.is_dir():
            candidate = path_obj / "hnsw_index.bin"
            if candidate.exists():
                path = str(candidate)
        self._index = hnswlib.Index(space=self._space, dim=self._dim or 0)
        self._index.load_index(path)
        self._n_total = self._index.get_current_count()
        logger.info(
            "Loaded HNSW index from %s (total vectors: %d)",
            path, self._n_total,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_total(self) -> int:
        """Total number of indexed vectors."""
        return self._n_total
