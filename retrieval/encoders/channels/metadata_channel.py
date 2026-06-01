"""Structured metadata channel encoder.

Encodes authors, keywords, and venue information into a unified metadata
embedding using a combination of learnable entity embeddings and shared
semantic encoding.

Input type: ``List[Dict[str, Any]]`` where each dict contains:
    - ``authors``: ``List[str]``
    - ``keywords``: ``List[str]``
    - ``venue``: ``Optional[str]``

Output shape: ``(N, output_dim)`` (default 256).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChannelEncoder

logger = logging.getLogger(__name__)

_UNK_ID = 0  # Unknown / out-of-vocabulary token index


class MetadataChannelEncoder(ChannelEncoder, nn.Module):
    """Structured metadata channel encoder.

    Three sub-encoders:
    1. **Keywords** – shared BGE-M3 encoding, mean-pooled
    2. **Authors** – learnable EmbeddingBag (handles large vocabularies)
    3. **Venue** – learnable Embedding (bounded vocabulary)

    Concatenated and projected to ``output_dim``.

    Parameters
    ----------
    output_dim : int
        Final output dimension.
    author_vocab_size : int
        Maximum author vocabulary size.
    venue_vocab_size : int
        Maximum venue vocabulary size.
    semantic_encoder : Optional[Any]
        Shared semantic encoder instance (e.g. SemanticChannelEncoder).
        If ``None``, keywords are encoded via simple hash embedding.
    keyword_dim : int
        Dimension of keyword embeddings (must match semantic encoder output).
    author_embed_dim : int
        Dimension of author embedding vectors.
    venue_embed_dim : int
        Dimension of venue embedding vectors.
    """

    CHANNEL_NAME = "metadata"

    def __init__(
        self,
        output_dim: int = 256,
        author_vocab_size: int = 100_000,
        venue_vocab_size: int = 50_000,
        semantic_encoder: Optional[Any] = None,
        keyword_dim: int = 1024,
        author_embed_dim: int = 128,
        venue_embed_dim: int = 128,
    ):
        nn.Module.__init__(self)
        ChannelEncoder.__init__(self)

        self._output_dim = output_dim
        self._author_vocab_size = author_vocab_size
        self._venue_vocab_size = venue_vocab_size
        self._semantic_encoder = semantic_encoder
        self._keyword_dim = keyword_dim
        self._author_embed_dim = author_embed_dim
        self._venue_embed_dim = venue_embed_dim

        # Learnable entity embeddings
        self.author_embedding = nn.EmbeddingBag(
            author_vocab_size, author_embed_dim, mode="mean", padding_idx=0
        )
        self.venue_embedding = nn.Embedding(
            venue_vocab_size, venue_embed_dim, padding_idx=0
        )

        # Projection: keyword_dim + author_embed_dim + venue_embed_dim -> output_dim
        proj_input_dim = keyword_dim + author_embed_dim + venue_embed_dim
        self.proj = nn.Linear(proj_input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Vocabulary mappings (built during build_vocab)
        self._author2id: Dict[str, int] = {}
        self._venue2id: Dict[str, int] = {}
        self._vocab_built = False

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def build_vocab(self, documents: List[Dict[str, Any]]) -> None:
        """Build entity-to-ID mappings from document metadata.

        Must be called before :meth:`encode` for meaningful results.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            Document dicts containing ``authors``, ``keywords``, ``venue``.
        """
        author_set: set = set()
        venue_set: set = set()

        for doc in documents:
            for author in doc.get("authors", []):
                author_set.add(str(author))
            venue = doc.get("venue")
            if venue:
                venue_set.add(str(venue))

        # Sort for determinism
        for i, author in enumerate(sorted(author_set), start=1):
            if i < self._author_vocab_size:
                self._author2id[author] = i

        for i, venue in enumerate(sorted(venue_set), start=1):
            if i < self._venue_vocab_size:
                self._venue2id[venue] = i

        self._vocab_built = True
        logger.info(
            "MetadataChannel: built vocab — %d authors, %d venues",
            len(self._author2id),
            len(self._venue2id),
        )

    def _get_author_ids(self, authors: List[str]) -> torch.Tensor:
        """Convert author names to embedding IDs."""
        ids = [self._author2id.get(str(a), _UNK_ID) for a in authors]
        if not ids:
            ids = [_UNK_ID]
        return torch.tensor([ids], dtype=torch.long)

    def _get_venue_id(self, venue: Optional[str]) -> torch.Tensor:
        """Convert venue name to embedding ID."""
        vid = self._venue2id.get(str(venue), _UNK_ID) if venue else _UNK_ID
        return torch.tensor([vid], dtype=torch.long)

    # ------------------------------------------------------------------
    # ChannelEncoder interface
    # ------------------------------------------------------------------

    def encode(self, inputs: List[Dict[str, Any]]) -> np.ndarray:
        """Encode a batch of document metadata.

        Parameters
        ----------
        inputs : List[Dict[str, Any]]
            Each dict should contain ``authors``, ``keywords``, ``venue``.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``.
        """
        self.eval()
        all_features = []

        with torch.no_grad():
            for doc in inputs:
                # 1. Keywords encoding
                kw_vec = self._encode_keywords(doc.get("keywords", []))
                # 2. Author embedding
                author_ids = self._get_author_ids(doc.get("authors", []))
                author_vec = self.author_embedding(author_ids).squeeze(0)  # (author_embed_dim,)
                # 3. Venue embedding
                venue_id = self._get_venue_id(doc.get("venue"))
                venue_vec = self.venue_embedding(venue_id).squeeze(0)  # (venue_embed_dim,)

                combined = torch.cat([kw_vec, author_vec, venue_vec], dim=-1)
                projected = self.proj(combined)
                normalized = self.layer_norm(projected)
                all_features.append(normalized.numpy())

        return np.array(all_features, dtype=np.float32)

    def encode_single(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Encode a single document's metadata.

        Parameters
        ----------
        input_data : Dict[str, Any]
            Dict containing ``authors``, ``keywords``, ``venue``.

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

    # ------------------------------------------------------------------
    # Internal keyword encoding
    # ------------------------------------------------------------------

    def _encode_keywords(self, keywords: List[str]) -> torch.Tensor:
        """Encode keywords into a fixed-dim vector.

        Strategy:
        - If semantic_encoder is available, encode each keyword and mean-pool
        - Otherwise, use a simple hash-based projection
        """
        if not keywords:
            return torch.zeros(self._keyword_dim, dtype=torch.float32)

        if self._semantic_encoder is not None:
            # Use shared semantic encoder
            kw_embeddings = self._semantic_encoder.encode(keywords)
            kw_tensor = torch.tensor(kw_embeddings, dtype=torch.float32)
            return kw_tensor.mean(dim=0)
        else:
            # Hash-based fallback: deterministic projection
            rng = np.random.default_rng(42)
            hash_proj = rng.standard_normal(
                (self._keyword_dim, 32), dtype=np.float32
            )
            # Simple character-level hash
            kw_vecs = []
            for kw in keywords:
                h = hash(kw) % (2**31)
                indices = [(h + i) % 32 for i in range(8)]
                one_hot = np.zeros(32, dtype=np.float32)
                for idx in indices:
                    one_hot[idx] = 1.0
                projected = one_hot @ hash_proj.T
                kw_vecs.append(projected)
            return torch.tensor(
                np.mean(kw_vecs, axis=0), dtype=torch.float32
            )
