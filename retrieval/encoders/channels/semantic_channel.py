"""Semantic content channel encoder using BGE-M3.

Encodes title + abstract text into dense embeddings via BGE-M3 (1024-dim).
Falls back to SentenceTransformer (384-dim) when FlagEmbedding is unavailable.

Input type: ``List[str]`` (title + abstract concatenated text).
Output shape: ``(N, 1024)`` with BGE-M3, ``(N, 384)`` with SBERT fallback.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .base import ChannelEncoder

logger = logging.getLogger(__name__)

# BGE-M3 dense embedding dimension
_BGE_M3_DIM = 1024
# SBERT fallback dimension (all-MiniLM-L6-v2)
_SBERT_FALLBACK_DIM = 384


class SemanticChannelEncoder(ChannelEncoder):
    """Semantic content channel using BGE-M3.

    Parameters
    ----------
    model_name : str
        BGE-M3 model identifier (default ``"BAAI/bge-m3"``).
    batch_size : int
        Batch size for encoding.
    use_fp16 : bool
        Use half-precision inference when available.
    use_cache : bool
        Cache encoded vectors in memory.
    device : Optional[str]
        Device string (``"cpu"``, ``"cuda:0"``, etc.). ``None`` for auto.
    fallback_to_sbert : bool
        Fall back to SentenceTransformer when FlagEmbedding is unavailable.
    """

    CHANNEL_NAME = "semantic"

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 12,
        use_fp16: bool = True,
        use_cache: bool = True,
        device: Optional[str] = None,
        fallback_to_sbert: bool = True,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._use_fp16 = use_fp16
        self._use_cache = use_cache
        self._device = device
        self._fallback_to_sbert = fallback_to_sbert

        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self._fallback_encoder = None
        self._using_fallback = False

        self._ensure_model_loaded()

    # ------------------------------------------------------------------
    # Internal model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazily load BGE-M3 or fall back to SentenceTransformer."""
        if self._model is not None or self._fallback_encoder is not None:
            return

        # Try BGE-M3 via FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel

            devices = [self._device] if self._device else None
            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=self._use_fp16,
                devices=devices,
            )
            logger.info(
                "SemanticChannel: loaded BGE-M3 '%s' (dim=%d)",
                self._model_name,
                _BGE_M3_DIM,
            )
            return
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed; "
                "BGE-M3 unavailable for semantic channel."
            )
        except Exception as exc:
            logger.warning("Failed to load BGE-M3: %s", exc)

        # Fallback to SentenceTransformer
        if self._fallback_to_sbert:
            try:
                from sentence_transformers import SentenceTransformer

                self._fallback_encoder = SentenceTransformer(
                    "all-MiniLM-L6-v2", device=self._device or "cpu"
                )
                self._using_fallback = True
                logger.info(
                    "SemanticChannel: fell back to SentenceTransformer "
                    "'all-MiniLM-L6-v2' (dim=%d)",
                    _SBERT_FALLBACK_DIM,
                )
            except ImportError:
                logger.error(
                    "Neither FlagEmbedding nor sentence-transformers available. "
                    "Semantic channel will produce random embeddings."
                )
        else:
            logger.error(
                "FlagEmbedding unavailable and fallback disabled. "
                "Semantic channel will produce random embeddings."
            )

    # ------------------------------------------------------------------
    # ChannelEncoder interface
    # ------------------------------------------------------------------

    def encode(self, inputs: List[str]) -> np.ndarray:
        """Encode a list of text strings.

        Parameters
        ----------
        inputs : List[str]
            Text strings (typically title + abstract concatenated).

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``.
        """
        # Check cache
        if self._use_cache:
            cached_results = []
            uncached_indices = []
            uncached_texts = []
            for i, text in enumerate(inputs):
                if text in self._cache:
                    cached_results.append((i, self._cache[text]))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            if not uncached_texts:
                # All cached
                results = np.zeros((len(inputs), self.output_dim), dtype=np.float32)
                for i, vec in cached_results:
                    results[i] = vec
                return results

            encoded = self._encode_batch(uncached_texts)
            for idx, text, vec in zip(uncached_indices, uncached_texts, encoded):
                self._cache[text] = vec
                cached_results.append((idx, vec))

            results = np.zeros((len(inputs), self.output_dim), dtype=np.float32)
            for i, vec in cached_results:
                results[i] = vec
            return results

        return self._encode_batch(inputs)

    def encode_single(self, input_data: str) -> np.ndarray:
        """Encode a single text string.

        Parameters
        ----------
        input_data : str
            A single text string.

        Returns
        -------
        np.ndarray
            Shape ``(output_dim,)``.
        """
        if self._use_cache and input_data in self._cache:
            return self._cache[input_data]
        result = self.encode([input_data])[0]
        if self._use_cache:
            self._cache[input_data] = result
        return result

    @property
    def output_dim(self) -> int:
        if self._using_fallback:
            return _SBERT_FALLBACK_DIM
        return _BGE_M3_DIM

    @property
    def channel_name(self) -> str:
        return self.CHANNEL_NAME

    # ------------------------------------------------------------------
    # Internal encoding
    # ------------------------------------------------------------------

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Core batch encoding logic."""
        # Replace empty strings
        safe_texts = [t if t.strip() else "empty document" for t in texts]

        if self._model is not None:
            # BGE-M3 encoding
            result = self._model.encode(
                safe_texts,
                batch_size=self._batch_size,
                max_length=8192,
            )
            dense = result["dense_vecs"]
            if hasattr(dense, "numpy"):
                dense = dense.numpy()
            return dense.astype(np.float32)

        if self._fallback_encoder is not None:
            # SentenceTransformer fallback
            embeddings = self._fallback_encoder.encode(
                safe_texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)

        # No model available — produce random normalized vectors
        logger.warning(
            "No embedding model available; producing random embeddings."
        )
        dim = self.output_dim
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(safe_texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return vecs / norms
