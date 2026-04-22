"""Encoder abstraction layer for tensor-manifold-gvf.

This package provides a pluggable encoder interface so that downstream
components (retriever, index builder) are decoupled from any specific
embedding backend.

Quick start::

    from tensor_manifold_gvf.retrieval.encoders import create_encoder

    encoder = create_encoder({"type": "sentence_transformer"})
    vectors = encoder.encode(["hello world"])

The :func:`create_encoder` factory inspects the ``"type"`` key of a
configuration dictionary and returns the appropriate
:class:`~tensor_manifold_gvf.retrieval.encoders.base.EmbeddingEncoder`
subclass instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import EmbeddingEncoder
from .sentence_transformer_encoder import SentenceTransformerEncoder

__all__ = [
    "EmbeddingEncoder",
    "SentenceTransformerEncoder",
    "create_encoder",
]

logger = logging.getLogger("tensor_manifold_gvf.encoders")

# Registry mapping encoder type identifiers to concrete classes.
_ENCODER_REGISTRY: Dict[str, type] = {
    "sentence_transformer": SentenceTransformerEncoder,
    "sbert": SentenceTransformerEncoder,
}


def create_encoder(config: Dict[str, Any]) -> EmbeddingEncoder:
    """Factory: build an :class:`EmbeddingEncoder` from a config dict.

    The configuration dictionary **must** contain a ``"type"`` key whose
    value matches one of the registered encoder identifiers (e.g.
    ``"sentence_transformer"`` or ``"sbert"``).  All remaining keys are
    forwarded as keyword arguments to the encoder constructor.

    Example configuration::

        {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 64,
            "use_cache": True,
        }

    Parameters
    ----------
    config : Dict[str, Any]
        Encoder configuration.  Required key: ``"type"``.

    Returns
    -------
    EmbeddingEncoder
        An initialised encoder instance.

    Raises
    ------
    ValueError
        If ``"type"`` is missing or not recognised.
    """
    encoder_type = config.get("type")
    if not encoder_type:
        raise ValueError(
            "Encoder config must contain a 'type' key. "
            f"Available types: {sorted(_ENCODER_REGISTRY.keys())}"
        )

    cls = _ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(
            f"Unknown encoder type '{encoder_type}'. "
            f"Available types: {sorted(_ENCODER_REGISTRY.keys())}"
        )

    # Strip the 'type' key so it is not passed to the constructor.
    kwargs = {k: v for k, v in config.items() if k != "type"}

    logger.info("Creating encoder '%s' with kwargs=%s", encoder_type, kwargs)
    return cls(**kwargs)
