"""Temporal encoder for time-aware node representations.

Encodes timestamps into vector representations using normalization
and sinusoidal encoding schemes, supporting temporal similarity computation.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


class TemporalEncoder:
    """Encodes timestamps into vector representations.

    Supports scalar normalization and sinusoidal encoding for capturing
    temporal patterns in academic graph data.
    """

    def __init__(
        self,
        output_dim: int = 128,
        min_year: float = 1900.0,
        max_year: float = 2030.0,
    ):
        """Initialize the temporal encoder.

        Args:
            output_dim: Dimensionality of the output temporal embedding.
            min_year: Minimum year for normalization.
            max_year: Maximum year for normalization.
        """
        self.output_dim = output_dim
        self.min_year = min_year
        self.max_year = max_year

    def normalize(self, year: float) -> float:
        """Normalize a year value to [0, 1] range.

        Args:
            year: Year value to normalize.

        Returns:
            Normalized value in [0, 1].
        """
        if self.max_year == self.min_year:
            return 0.0
        return (year - self.min_year) / (self.max_year - self.min_year)

    def encode_scalar(self, year: float) -> np.ndarray:
        """Encode a year as a scalar (normalized) value.

        Args:
            year: Year value to encode.

        Returns:
            numpy array of shape (1,) with the normalized value.
        """
        return np.array([self.normalize(year)])

    def encode_sinusoidal(self, year: float) -> np.ndarray:
        """Encode a year using sinusoidal position encoding.

        Uses multiple frequency bands to capture temporal patterns,
        similar to Transformer positional encoding.

        Args:
            year: Year value to encode.

        Returns:
            numpy array of shape (output_dim,).
        """
        normalized = self.normalize(year)
        dim_indices = np.arange(self.output_dim)
        # Use even/odd frequency bands
        frequencies = 1.0 / (10000.0 ** (2.0 * (dim_indices // 2) / self.output_dim))
        angles = normalized * 2.0 * np.pi * frequencies

        # Apply sin to even indices, cos to odd indices
        encodings = np.zeros(self.output_dim)
        encodings[0::2] = np.sin(angles[0::2])
        encodings[1::2] = np.cos(angles[1::2])

        return encodings

    def encode_batch(
        self,
        years: List[float],
        method: str = "sinusoidal",
    ) -> np.ndarray:
        """Encode a batch of years.

        Args:
            years: List of year values.
            method: Encoding method, either 'scalar' or 'sinusoidal'.

        Returns:
            numpy array of shape (n_years, output_dim) for sinusoidal,
            or (n_years, 1) for scalar.

        Raises:
            ValueError: If method is not recognized.
        """
        if method == "scalar":
            return np.array([self.encode_scalar(y) for y in years])
        elif method == "sinusoidal":
            return np.array([self.encode_sinusoidal(y) for y in years])
        else:
            raise ValueError(
                f"Unknown encoding method: {method}. "
                "Use 'scalar' or 'sinusoidal'."
            )

    def compute_temporal_similarity(
        self,
        year1: float,
        year2: float,
        decay_alpha: float = 0.3,
    ) -> float:
        """Compute temporal similarity between two years.

        Uses exponential decay based on the absolute year difference.

        Args:
            year1: First year.
            year2: Second year.
            decay_alpha: Decay rate controlling how quickly similarity
                         decreases with temporal distance.

        Returns:
            Similarity score in [0, 1].
        """
        diff = abs(year1 - year2)
        return np.exp(-decay_alpha * diff)
