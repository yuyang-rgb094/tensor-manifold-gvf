"""Temporal signal channel encoder using Time2Vec.

Encodes publication timestamps into temporal embeddings using learnable
Time2Vec encoding with periodic and linear components, plus discrete
year/month embeddings.

Input type: ``List[float]`` (timestamps, e.g. year as float like 2024.0).

Output shape: ``(N, output_dim)`` (default 64).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChannelEncoder

logger = logging.getLogger(__name__)


class Time2Vec(nn.Module):
    """Learnable time encoding with periodic and linear components.

    tau(t) = [sin(w1*t + b1), ..., sin(wk*t + bk), w0*t + b0]

    Parameters
    ----------
    n_periodic : int
        Number of periodic (sinusoidal) components.
    """

    def __init__(self, n_periodic: int = 32):
        super().__init__()
        self.n_periodic = n_periodic
        # Learnable parameters for periodic components
        self.w_periodic = nn.Parameter(torch.randn(1, n_periodic) * 0.1)
        self.b_periodic = nn.Parameter(torch.randn(1, n_periodic) * 0.1)
        # Learnable parameters for linear (trend) component
        self.w_linear = nn.Parameter(torch.randn(1, 1) * 0.1)
        self.b_linear = nn.Parameter(torch.randn(1, 1) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : torch.Tensor
            Shape ``(batch, 1)`` — normalized timestamps in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_periodic + 1)``
        """
        # Periodic components: sin(w*t + b)
        periodic = torch.sin(
            torch.matmul(t, self.w_periodic) + self.b_periodic
        )  # (batch, n_periodic)
        # Linear component: w*t + b
        linear = torch.matmul(t, self.w_linear) + self.b_linear  # (batch, 1)
        return torch.cat([periodic, linear], dim=-1)


class TemporalChannelEncoder(ChannelEncoder, nn.Module):
    """Temporal signal channel encoder using Time2Vec.

    Combines:
    1. Time2Vec continuous encoding (periodic + linear)
    2. Discrete year embedding
    3. Discrete month embedding

    Parameters
    ----------
    output_dim : int
        Final output dimension.
    n_periodic : int
        Number of periodic components in Time2Vec.
    year_vocab_size : int
        Number of discrete year buckets (e.g. 1900-2100 = 200).
    month_vocab_size : int
        Number of discrete month buckets (13 = 1-12 + padding).
    year_embed_dim : int
        Dimension of year embedding.
    month_embed_dim : int
        Dimension of month embedding.
    """

    CHANNEL_NAME = "temporal"

    def __init__(
        self,
        output_dim: int = 64,
        n_periodic: int = 32,
        year_vocab_size: int = 200,
        month_vocab_size: int = 13,
        year_embed_dim: int = 16,
        month_embed_dim: int = 8,
    ):
        nn.Module.__init__(self)
        ChannelEncoder.__init__(self)

        self._output_dim = output_dim
        self._n_periodic = n_periodic

        # Time2Vec: n_periodic + 1 dimensions
        self.time2vec = Time2Vec(n_periodic)

        # Discrete embeddings
        self.year_embedding = nn.Embedding(year_vocab_size, year_embed_dim)
        self.month_embedding = nn.Embedding(month_vocab_size, month_embed_dim)

        # Projection: (n_periodic + 1) + year_embed_dim + month_embed_dim -> output_dim
        t2v_dim = n_periodic + 1
        proj_input_dim = t2v_dim + year_embed_dim + month_embed_dim
        self.proj = nn.Linear(proj_input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Normalization state
        self._min_t: Optional[float] = None
        self._max_t: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def fit(self, timestamps: List[float]) -> None:
        """Compute normalization range from timestamps.

        Parameters
        ----------
        timestamps : List[float]
            Raw timestamp values (e.g. publication years).
        """
        if not timestamps:
            self._min_t = 0.0
            self._max_t = 1.0
            self._fitted = True
            return

        self._min_t = float(min(timestamps))
        self._max_t = float(max(timestamps))
        # Avoid division by zero
        if self._max_t == self._min_t:
            self._max_t = self._min_t + 1.0
        self._fitted = True
        logger.info(
            "TemporalChannel: fitted normalization range [%.1f, %.1f]",
            self._min_t,
            self._max_t,
        )

    def _normalize(self, timestamps: List[float]) -> torch.Tensor:
        """Normalize timestamps to [0, 1]."""
        if not self._fitted:
            self.fit(timestamps)

        t = np.array(timestamps, dtype=np.float32)
        t_norm = (t - self._min_t) / (self._max_t - self._min_t)
        t_norm = np.clip(t_norm, 0.0, 1.0)
        return torch.tensor(t_norm, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def _extract_year_month(self, timestamps: List[float]) -> tuple:
        """Extract discrete year and month from timestamps.

        Assumes timestamps are year values (e.g. 2024.0) or epoch seconds.
        """
        years = []
        months = []
        for t in timestamps:
            if t > 1e9:
                # Epoch seconds — convert to year/month
                import datetime
                dt = datetime.datetime.fromtimestamp(t)
                years.append(max(0, dt.year - 1900))
                months.append(dt.month)
            else:
                # Year as float (e.g. 2024.5)
                year_int = int(t)
                month_float = (t - year_int) * 12 + 1
                years.append(max(0, year_int - 1900))
                months.append(int(min(12, max(1, month_float))))
        return (
            torch.tensor(years, dtype=torch.long),
            torch.tensor(months, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # ChannelEncoder interface
    # ------------------------------------------------------------------

    def encode(self, inputs: List[float]) -> np.ndarray:
        """Encode a list of timestamps.

        Parameters
        ----------
        inputs : List[float]
            Timestamp values (years as float, or epoch seconds).

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``.
        """
        self.eval()
        with torch.no_grad():
            # Normalize timestamps
            t_norm = self._normalize(inputs)  # (N, 1)

            # Time2Vec encoding
            t2v = self.time2vec(t_norm)  # (N, n_periodic + 1)

            # Discrete year/month embeddings
            year_ids, month_ids = self._extract_year_month(inputs)
            year_emb = self.year_embedding(year_ids)  # (N, year_embed_dim)
            month_emb = self.month_embedding(month_ids)  # (N, month_embed_dim)

            # Concatenate and project
            combined = torch.cat([t2v, year_emb, month_emb], dim=-1)
            projected = self.proj(combined)
            normalized = self.layer_norm(projected)

            return normalized.numpy().astype(np.float32)

    def encode_single(self, input_data: float) -> np.ndarray:
        """Encode a single timestamp.

        Parameters
        ----------
        input_data : float
            A single timestamp value.

        Returns
        -------
        np.ndarray
            Shape ``(output_dim,)``.
        """
        return self.encode([input_data])[0]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def channel_name(self) -> str:
        return self.CHANNEL_NAME
