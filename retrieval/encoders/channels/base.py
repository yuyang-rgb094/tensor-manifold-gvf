"""Abstract base class for channel encoders in the four-channel architecture.

Each channel (semantic, metadata, topology, temporal) implements this interface
independently, producing embeddings in its own optimal space.  Channels are
fused downstream by :class:`FourChannelFusionEncoder`.

See ADR-0001 for the architectural rationale.
"""

from __future__ import annotations

import abc
from typing import Any, List

import numpy as np


class ChannelEncoder(abc.ABC):
    """Abstract interface for a single channel encoder.

    Unlike :class:`EmbeddingEncoder` which is text-only, each channel may
    accept different input types:

    * **semantic**  – ``List[str]`` (title + abstract text)
    * **metadata**  – ``List[Dict[str, Any]]`` (authors, keywords, venue)
    * **topology**  – ``Tuple[HeteroAcademicGraph, Dict[str, np.ndarray]]``
    * **temporal**  – ``List[float]`` (timestamps or year values)

    Concrete subclasses must document their accepted input type.
    """

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def encode(self, inputs: Any) -> np.ndarray:
        """Encode a batch of inputs into channel embeddings.

        Parameters
        ----------
        inputs : Any
            Channel-specific input (see class docstring for per-channel types).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, output_dim)`` with dtype ``float32``.
        """

    @abc.abstractmethod
    def encode_single(self, input_data: Any) -> np.ndarray:
        """Encode a single input into a 1-D channel embedding.

        Parameters
        ----------
        input_data : Any
            A single channel-specific input.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(output_dim,)``.
        """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the channel embeddings."""

    @property
    @abc.abstractmethod
    def channel_name(self) -> str:
        """Canonical channel identifier (e.g. ``\"semantic\"``, ``\"topology\"``)."""
