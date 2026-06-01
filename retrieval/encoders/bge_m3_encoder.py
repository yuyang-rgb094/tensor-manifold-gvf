"""BGE-M3 encoder adapter for the EmbeddingEncoder interface.

Wraps :class:`SemanticChannelEncoder` to conform to the
:class:`EmbeddingEncoder` ABC, enabling BGE-M3 to be used as a
drop-in replacement for SentenceTransformer in the v1 pipeline.

Only importable when ``FlagEmbedding`` is installed.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import EmbeddingEncoder
from .channels.semantic_channel import SemanticChannelEncoder


class BGEM3Encoder(EmbeddingEncoder):
    """BGE-M3 encoder conforming to the EmbeddingEncoder interface.

    Parameters
    ----------
    model_name : str
        BGE-M3 model identifier (default ``"BAAI/bge-m3"``).
    batch_size : int
        Batch size for encoding.
    use_fp16 : bool
        Use half-precision inference.
    use_cache : bool
        Cache encoded vectors in memory.
    device : str
        Device string (``"cpu"``, ``"cuda:0"``, etc.).
    fallback_to_sbert : bool
        Fall back to SentenceTransformer when FlagEmbedding is unavailable.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 12,
        use_fp16: bool = True,
        use_cache: bool = True,
        device: str = "cpu",
        fallback_to_sbert: bool = True,
    ):
        self._channel = SemanticChannelEncoder(
            model_name=model_name,
            batch_size=batch_size,
            use_fp16=use_fp16,
            use_cache=use_cache,
            device=device,
            fallback_to_sbert=fallback_to_sbert,
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._channel.encode(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self._channel.encode_single(text)

    @property
    def embedding_dim(self) -> int:
        return self._channel.output_dim

    @property
    def supports_sparse(self) -> bool:
        return True  # BGE-M3 supports sparse embeddings

    @property
    def model_name(self) -> str:
        return self._channel._model_name
