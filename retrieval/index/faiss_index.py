"""
FAISS-backed vector index implementation.

Merges two strategies found in the codebase:
- ``IndexFlatIP`` / ``IndexFlatL2`` for small datasets (from
  ``retriever.py``).
- ``IndexIVFPQ`` for large datasets (from ``core/riemannian_index.py``).

If FAISS is not installed, :meth:`build` raises :exc:`ImportError` with
a descriptive message.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .base import VectorIndex

logger = logging.getLogger("tensor_manifold_gvf.retrieval.index")

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


class FAISSVectorIndex(VectorIndex):
    """Vector index backed by Facebook AI Similarity Search (FAISS).

    The constructor does **not** import FAISS eagerly.  The import is
    deferred to :meth:`build` so that the class can be instantiated
    safely even when FAISS is absent (e.g. for configuration or testing
    purposes).

    Args:
        metric: Distance metric -- ``"ip"`` (inner product, default) or
            ``"l2"`` (Euclidean distance).
        use_ivf_pq: If ``True`` and the dataset size exceeds
            ``nlist * 10``, an ``IndexIVFPQ`` is used.  Otherwise a
            flat index is used regardless of this flag.
        nlist: Number of inverted-list clusters for IVF.
        m_pq: Number of sub-quantisers for Product Quantisation.
        nbits: Number of bits per sub-quantiser.
        nprobe: Number of clusters to probe at query time for IVF.
    """

    def __init__(
        self,
        metric: str = "ip",
        use_ivf_pq: bool = False,
        nlist: int = 100,
        m_pq: int = 8,
        nbits: int = 8,
        nprobe: int = 10,
    ) -> None:
        if metric not in ("ip", "l2"):
            raise ValueError(f"metric must be 'ip' or 'l2', got '{metric}'")
        self._metric = metric
        self._use_ivf_pq = use_ivf_pq
        self._nlist = nlist
        self._m_pq = m_pq
        self._nbits = nbits
        self._nprobe = nprobe

        self._index: Optional[object] = None
        self._n_total: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray) -> None:
        """Build the FAISS index from *embeddings*.

        Strategy:
        - If ``use_ivf_pq`` is ``False`` **or** ``N <= nlist * 10``:
          use a flat exact index (``IndexFlatIP`` or ``IndexFlatL2``).
        - If ``use_ivf_pq`` is ``True`` **and** ``N > nlist * 10``:
          use ``IndexIVFPQ`` with the configured parameters.

        Args:
            embeddings: 2-D float array of shape ``(N, D)``.

        Raises:
            ImportError: If the ``faiss`` package is not installed.
        """
        if not _HAS_FAISS:
            logger.warning(
                "faiss is not installed; cannot build FAISSVectorIndex. "
                "Install with: pip install faiss-cpu"
            )
            raise ImportError(
                "faiss is not installed. Install with: pip install faiss-cpu"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        d = embeddings.shape[1]
        N = embeddings.shape[0]

        if self._use_ivf_pq and N > self._nlist * 10:
            # IVF + PQ for large datasets (from core/riemannian_index.py)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(
                quantizer, d, self._nlist, self._m_pq, self._nbits
            )
            index.nprobe = self._nprobe
            if not index.is_trained:
                index.train(embeddings)
            index.add(embeddings)
            logger.info(
                "Built FAISS IVF+PQ index: N=%d, d=%d, nlist=%d, "
                "m_pq=%d, nbits=%d, nprobe=%d",
                N, d, self._nlist, self._m_pq, self._nbits, self._nprobe,
            )
        else:
            # Flat index for small datasets (from retriever.py)
            if self._metric == "l2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            logger.info(
                "Built FAISS flat index: N=%d, d=%d, metric=%s",
                N, d, self._metric,
            )

        self._index = index
        self._n_total = N

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

        single = query.ndim == 1
        if single:
            query = query.reshape(1, -1)

        query = np.ascontiguousarray(query, dtype=np.float32)
        scores, indices = self._index.search(query, top_k)

        if single:
            scores = scores[0]
            indices = indices[0]

        return scores, indices

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
        self._index.add(embeddings)
        self._n_total += embeddings.shape[0]
        logger.debug(
            "Added %d vectors to FAISS index (total: %d)",
            embeddings.shape[0], self._n_total,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the FAISS index to disk.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")
        faiss.write_index(self._index, path)
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: str) -> None:
        """Load a FAISS index from disk.

        Args:
            path: Source file path.

        Raises:
            ImportError: If ``faiss`` is not installed.
        """
        if not _HAS_FAISS:
            raise ImportError(
                "faiss is not installed. Install with: pip install faiss-cpu"
            )
        self._index = faiss.read_index(path)
        self._n_total = self._index.ntotal
        logger.info(
            "Loaded FAISS index from %s (total vectors: %d)",
            path, self._n_total,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_total(self) -> int:
        """Total number of indexed vectors."""
        return self._n_total
