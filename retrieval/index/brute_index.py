"""
Brute-force vector index backed by pure NumPy.

Computes exact cosine similarity via dot product (assuming embeddings
are L2-normalised).  No external dependencies beyond NumPy.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .base import VectorIndex

logger = logging.getLogger("tensor_manifold_gvf.retrieval.index")


class BruteForceIndex(VectorIndex):
    """Exact nearest-neighbour search using NumPy dot-product similarity.

    This index assumes that embeddings are L2-normalised so that the
    dot product equals cosine similarity.  It has no external
    dependencies beyond NumPy and serves as a reliable fallback when
    FAISS or HNSWLIB are unavailable.

    Args:
        dtype: NumPy dtype used to store embeddings internally.
            Defaults to ``np.float32``.
    """

    def __init__(self, dtype: np.dtype = np.float32) -> None:
        self._dtype = dtype
        self._embeddings: np.ndarray | None = None
        self._n_total: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray) -> None:
        """Build the brute-force index from *embeddings*.

        Args:
            embeddings: 2-D array of shape ``(N, D)``.
        """
        self._embeddings = np.asarray(embeddings, dtype=self._dtype)
        self._n_total = self._embeddings.shape[0]
        logger.info(
            "Built brute-force index: N=%d, d=%d",
            self._n_total,
            self._embeddings.shape[1],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the *top_k* nearest neighbours.

        Computes cosine similarity via dot product.

        Args:
            query: 1-D ``(D,)`` or 2-D ``(Q, D)`` query vector(s).
            top_k: Number of results to return.

        Returns:
            ``(scores, indices)`` -- each of shape ``(Q, top_k)`` or
            ``(top_k,)``.  *scores* are dot-product similarities
            (higher is more similar).

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._embeddings is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        single = query.ndim == 1
        if single:
            query = query.reshape(1, -1)

        query = np.asarray(query, dtype=self._dtype)
        # (Q, D) @ (D, N) -> (Q, N)
        scores = query @ self._embeddings.T

        # Partition + sort to get top_k efficiently
        k = min(top_k, scores.shape[1])
        # argpartition is O(N); then sort only the top-k
        top_k_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        # Gather the top-k scores
        rows = np.arange(scores.shape[0])[:, None]
        top_k_scores = scores[rows, top_k_idx]
        # Sort within the top-k (descending)
        sorted_order = np.argsort(-top_k_scores, axis=1)
        top_k_scores = np.take_along_axis(top_k_scores, sorted_order, axis=1)
        top_k_idx = np.take_along_axis(top_k_idx, sorted_order, axis=1)

        if single:
            top_k_scores = top_k_scores[0]
            top_k_idx = top_k_idx[0]

        return top_k_scores, top_k_idx

    # ------------------------------------------------------------------
    # Incremental add
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray) -> None:
        """Add new vectors to the existing index.

        Args:
            embeddings: 2-D array of shape ``(M, D)``.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._embeddings is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        new_embeddings = np.asarray(embeddings, dtype=self._dtype)
        self._embeddings = np.vstack([self._embeddings, new_embeddings])
        self._n_total += new_embeddings.shape[0]
        logger.debug(
            "Added %d vectors to brute-force index (total: %d)",
            new_embeddings.shape[0], self._n_total,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save embeddings to a ``.npy`` file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._embeddings is None:
            raise RuntimeError("Index has not been built. Call build() first.")
        np.save(path, self._embeddings)
        logger.info("Saved brute-force index to %s", path)

    def load(self, path: str) -> None:
        """Load embeddings from a ``.npy`` file.

        Args:
            path: Source file path.
        """
        self._embeddings = np.load(path).astype(self._dtype)
        self._n_total = self._embeddings.shape[0]
        logger.info(
            "Loaded brute-force index from %s (total vectors: %d)",
            path, self._n_total,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_total(self) -> int:
        """Total number of indexed vectors."""
        return self._n_total
