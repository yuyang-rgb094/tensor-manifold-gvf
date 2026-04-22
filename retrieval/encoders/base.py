"""Abstract base class for embedding encoders.

All concrete encoder implementations (sentence-transformers, OpenAI, etc.)
must inherit from :class:`EmbeddingEncoder` and implement its abstract
interface so that downstream components (retriever, index builder) can
swap encoders transparently.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np


class EmbeddingEncoder(abc.ABC):
    """Abstract interface for text embedding encoders.

    Subclasses must implement :meth:`encode`, :meth:`encode_single`, and
    the :attr:`embedding_dim` property.  The default :attr:`supports_sparse`
    and :attr:`model_name` are provided as concrete properties so that
    every encoder can be inspected uniformly.
    """

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into dense embeddings.

        Parameters
        ----------
        texts : List[str]
            The text strings to encode.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), embedding_dim)`` with dtype
            ``float32`` (or ``float64`` depending on the backend).
        """

    @abc.abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string into a 1-D embedding vector.

        Parameters
        ----------
        text : str
            The text string to encode.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(embedding_dim,)``.
        """

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this encoder."""

    # ------------------------------------------------------------------
    # Concrete helpers / introspection
    # ------------------------------------------------------------------

    @property
    def supports_sparse(self) -> bool:
        """Whether this encoder can produce sparse embeddings (default ``False``)."""
        return False

    @property
    def model_name(self) -> str:
        """Human-readable identifier for the underlying model (default ``\"unknown\"``)."""
        return "unknown"
