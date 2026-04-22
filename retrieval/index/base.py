"""
Abstract base class for vector similarity indices.

All concrete index implementations (FAISS, HNSWLIB, brute-force) must
subclass :class:`VectorIndex` and implement the required interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class VectorIndex(ABC):
    """Abstract base class defining the vector index interface.

    Subclasses provide concrete implementations backed by different
    approximate / exact nearest-neighbor libraries.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        """Build the index from a matrix of embeddings.

        Args:
            embeddings: 2-D array of shape ``(N, D)`` where *N* is the
                number of vectors and *D* is the dimensionality.
        """

    @abstractmethod
    def search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for the *top_k* nearest neighbors.

        Args:
            query: 1-D array of shape ``(D,)`` or 2-D array of shape
                ``(Q, D)`` for batch queries.
            top_k: Number of nearest neighbors to return.

        Returns:
            A tuple ``(scores, indices)`` each of shape ``(Q, top_k)``
            (or ``(top_k,)`` for a single query).  *scores* contains
            similarity scores (higher is more similar for inner-product
            metrics, lower for L2).
        """

    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Incrementally add new vectors to the index.

        The index must already have been built via :meth:`build` before
        calling this method.

        Args:
            embeddings: 2-D array of shape ``(M, D)`` with the new
                vectors to insert.
        """

    # ------------------------------------------------------------------
    # Optional interface (default: raise NotImplementedError)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index to *path*.

        Args:
            path: Filesystem path for the serialised index.

        Raises:
            NotImplementedError: If the backend does not support
                serialisation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support save()."
        )

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*.

        Args:
            path: Filesystem path of the serialised index.

        Raises:
            NotImplementedError: If the backend does not support
                deserialisation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support load()."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_total(self) -> int:
        """Return the total number of indexed vectors."""
