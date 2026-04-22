"""Sentence-Transformers embedding encoder.

Provides a concrete :class:`EmbeddingEncoder` backed by the
``sentence-transformers`` library.  Features include:

* In-memory embedding cache to avoid redundant computation.
* ``encode_documents`` helper that merges *title* + *abstract* from
  document dictionaries (mirrors the logic in ``retriever.py``).
* Graceful fallback to **random normalised embeddings** when
  ``sentence_transformers`` is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import EmbeddingEncoder

logger = logging.getLogger("tensor_manifold_gvf.encoders")

# Default embedding dimension used when falling back to random embeddings.
_DEFAULT_FALLBACK_DIM = 384


class SentenceTransformerEncoder(EmbeddingEncoder):
    """Embedding encoder powered by *sentence-transformers*.

    Parameters
    ----------
    model_name : str
        Name or local path of a pre-trained sentence-transformers model.
        Defaults to ``"all-MiniLM-L6-v2"``.
    batch_size : int
        Number of texts processed per forward pass.
    max_length : int
        Maximum token length accepted by the tokenizer.
    device : str | None
        Torch device string (e.g. ``"cuda"``, ``"cpu"``).  When *None*
        the library auto-selects the best available device.
    use_cache : bool
        If *True*, previously computed embeddings are stored in an
        in-memory dictionary and reused on subsequent calls.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
        max_length: int = 512,
        device: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._use_cache = use_cache

        # In-memory cache: text -> 1-D np.ndarray
        self._cache: Dict[str, np.ndarray] = {}

        # Lazily loaded model reference.
        self._model: Any = None
        self._fallback_mode = False

        # Try to import and load the model eagerly so that configuration
        # errors surface early.
        self._ensure_model_loaded()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load the sentence-transformers model on first access.

        If the library is not available the encoder enters *fallback mode*
        and will generate random normalised embeddings instead.
        """
        if self._model is not None or self._fallback_mode:
            return

        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            kwargs: Dict[str, Any] = {}
            if self._device is not None:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self._model_name, **kwargs)
            logger.info(
                "Loaded sentence-transformers model '%s' (dim=%d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        except ImportError:
            logger.warning(
                "sentence_transformers is not installed; "
                "falling back to random embeddings (dim=%d). "
                "Install it with: pip install sentence-transformers",
                _DEFAULT_FALLBACK_DIM,
            )
            self._fallback_mode = True

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic random normalised embeddings.

        This keeps the encoder usable for integration tests and
        prototyping even when ``sentence_transformers`` is absent.
        """
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(texts), self.embedding_dim))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # EmbeddingEncoder interface
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into dense embeddings.

        Empty or whitespace-only strings are replaced with
        ``"empty document"`` to avoid degenerate encodings.

        Parameters
        ----------
        texts : List[str]
            Text strings to encode.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), embedding_dim)`` with dtype
            ``float32``.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Normalise inputs.
        normalised = [t.strip() if t and t.strip() else "empty document" for t in texts]

        # Resolve cache hits.
        if self._use_cache:
            results: List[Optional[np.ndarray]] = [None] * len(normalised)
            to_encode: List[str] = []
            indices_to_encode: List[int] = []

            for i, text in enumerate(normalised):
                cached = self._cache.get(text)
                if cached is not None:
                    results[i] = cached
                else:
                    to_encode.append(text)
                    indices_to_encode.append(i)

            if to_encode:
                new_embeddings = self._encode_batch(to_encode)
                for idx, text, emb in zip(indices_to_encode, to_encode, new_embeddings):
                    self._cache[text] = emb
                    results[idx] = emb

            return np.asarray(results, dtype=np.float32)

        return self._encode_batch(normalised)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Low-level batch encode without caching."""
        self._ensure_model_loaded()

        if self._fallback_mode:
            return self._encode_fallback(texts)

        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string into a 1-D embedding vector.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(embedding_dim,)``.
        """
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Document-level convenience
    # ------------------------------------------------------------------

    def encode_documents(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Encode document dictionaries by merging *title* and *abstract*.

        Each document is expected to be a mapping with optional ``"title"``
        and ``"abstract"`` keys.  The two fields are joined with a single
        space, mirroring the logic in
        :meth:`~tensor_manifold_gvf.retrieval.retriever.UnifiedRetriever._sbert_encode`.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List of document dictionaries.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(documents), embedding_dim)``.
        """
        texts: List[str] = []
        for doc in documents:
            parts = [doc.get("title", ""), doc.get("abstract", "")]
            text = " ".join(p for p in parts if p).strip()
            texts.append(text if text else "empty document")
        return self.encode(texts)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Actual embedding dimensionality.

        Returns the real dimension reported by the model once loaded,
        or ``_DEFAULT_FALLBACK_DIM`` when in fallback mode.
        """
        self._ensure_model_loaded()
        if self._fallback_mode:
            return _DEFAULT_FALLBACK_DIM
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    @property
    def model_name(self) -> str:
        """Name or path of the underlying sentence-transformers model."""
        return self._model_name

    @property
    def supports_sparse(self) -> bool:
        """Sentence-Transformers only produces dense embeddings."""
        return False
