"""SBERT-based text encoder for academic graph nodes.

Wraps sentence-transformers for encoding node text (title + keywords + abstract)
into dense vector representations, with caching support.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ..data.oag_schema import Node


class SBERTEncoder:
    """Sentence-BERT encoder for generating node embeddings.

    Wraps sentence-transformers models and provides caching for
    efficient repeated encoding of the same texts.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
        max_length: int = 512,
        device: str = "cuda",
    ):
        """Initialize the SBERT encoder.

        Args:
            model_name: Name or path of the sentence-transformers model.
            batch_size: Batch size for encoding.
            max_length: Maximum sequence length for tokenization.
            device: Device to run inference on ('cuda' or 'cpu').
        """
        warnings.warn(
            "models.sbert_encoder.SBERTEncoder is deprecated. "
            "Use retrieval.encoders.SentenceTransformerEncoder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._cache: Dict[str, np.ndarray] = {}

    def encode_text(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into dense embeddings.

        Args:
            texts: Single text string or list of text strings.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_nodes(
        self,
        nodes: List[Node],
        preprocessor: Any = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of nodes into embeddings.

        Uses the preprocessor to extract text from node attributes,
        then encodes the resulting text strings.

        Args:
            nodes: List of Node objects.
            preprocessor: Optional TextPreprocessor instance. If None,
                          uses raw 'title' attribute.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n_nodes, embedding_dim).
        """
        texts = []
        for node in nodes:
            if preprocessor is not None:
                text = preprocessor.preprocess(node.attributes)
            else:
                text = node.attributes.get("title", "")
            texts.append(text)

        return self.encode_text(texts, show_progress=show_progress)

    def encode_with_cache(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) with caching to avoid redundant computation.

        Args:
            texts: Single text string or list of text strings.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        to_encode = []
        indices_to_encode = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
            else:
                results.append(None)
                to_encode.append(text)
                indices_to_encode.append(i)

        if to_encode:
            new_embeddings = self.encode_text(to_encode, show_progress=show_progress)
            for idx, text, emb in zip(indices_to_encode, to_encode, new_embeddings):
                self._cache[text] = emb
                results[idx] = emb

        return np.array(results)

    def save_cache(self, filepath: Union[str, Path]) -> None:
        """Save the encoding cache to disk.

        Args:
            filepath: Path to save the cache file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._cache, f)

    def load_cache(self, filepath: Union[str, Path]) -> None:
        """Load the encoding cache from disk.

        Args:
            filepath: Path to the cache file.
        """
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, "rb") as f:
                self._cache = pickle.load(f)
