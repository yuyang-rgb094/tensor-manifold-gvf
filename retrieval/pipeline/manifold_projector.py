"""
Manifold Projector Module

Provides the ``ManifoldProjector`` class that wraps
``core.manifold_encoder.HierarchicalManifoldEncoder`` and supports two
projection modes:

* **"truncate"** -- backward-compatible truncation / zero-padding with L2
  normalization (mirrors the logic in ``retriever.py`` lines 691-716).
* **"learned"** -- uses the ``HierarchicalManifoldEncoder`` PyTorch module
  to fuse diffusion signatures with semantic embeddings via gated residual
  connections and relation-aware projections.

The projector also exposes an ``update`` method implementing Algorithm 3
(Grassmannian mean shift) for incremental manifold refinement.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Support both package and direct execution imports
try:
    from ...core.manifold_encoder import HierarchicalManifoldEncoder
except ImportError:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from core.manifold_encoder import HierarchicalManifoldEncoder

logger = logging.getLogger(__name__)


class ManifoldProjector:
    """Project semantic embeddings onto a tensor manifold space.

    Two projection modes are available:

    ``"truncate"``
        Simple truncation (or zero-padding) followed by L2 normalization.
        This is backward-compatible with the existing retriever behaviour.

    ``"learned"``
        Uses the :class:`HierarchicalManifoldEncoder` PyTorch module to
        produce a learned manifold embedding that fuses semantic vectors
        with diffusion-signature features.

    Parameters
    ----------
    mode : str
        Projection mode -- ``"truncate"`` or ``"learned"``.
        Defaults to ``"learned"``.
    semantic_dim : int
        Dimension of input semantic embeddings.  Defaults to ``384``.
    signature_dim : int
        Dimension of the diffusion-signature feature vector.
        Defaults to ``64``.
    hidden_dim : int
        Intermediate hidden dimension for the learned encoder.
        Defaults to ``128``.
    output_dim : int
        Output manifold dimension (used as ``manifold_dim`` in truncate
        mode and as ``output_dim`` in learned mode).  Defaults to ``64``.
    num_relations : int
        Number of distinct relation types for the relation-aware
        projection layer.  Defaults to ``4``.
    """

    def __init__(
        self,
        mode: str = "learned",
        semantic_dim: int = 384,
        signature_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_relations: int = 4,
    ) -> None:
        if mode not in ("truncate", "learned"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose 'truncate' or 'learned'."
            )

        self.mode: str = mode
        self.semantic_dim: int = semantic_dim
        self.signature_dim: int = signature_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.num_relations: int = num_relations

        # Internal state for incremental updates (Algorithm 3)
        self._manifold_embeddings: Optional[np.ndarray] = None

        # Build the learned encoder when in learned mode
        self._encoder: Optional[HierarchicalManifoldEncoder] = None
        if self.mode == "learned":
            self._encoder = HierarchicalManifoldEncoder(
                semantic_dim=semantic_dim,
                signature_dim=signature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_relations=num_relations,
            )
            logger.info(
                "ManifoldProjector initialized in 'learned' mode "
                "(semantic_dim=%d, signature_dim=%d, hidden_dim=%d, "
                "output_dim=%d, num_relations=%d).",
                semantic_dim,
                signature_dim,
                hidden_dim,
                output_dim,
                num_relations,
            )
        else:
            logger.info(
                "ManifoldProjector initialized in 'truncate' mode "
                "(output_dim=%d).",
                output_dim,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        embeddings: np.ndarray,
        signatures: Optional[List[Dict]] = None,
        relation_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project a batch of embeddings onto the manifold space.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, semantic_dim)`` -- semantic embedding vectors.
        signatures : Optional[List[Dict]]
            A list of *N* diffusion-signature dictionaries.  Each dict
            may contain keys such as ``"entity_count"``,
            ``"relation_count"``, and ``"n_slices"``.  Required when
            ``mode == "learned"``; ignored otherwise.
        relation_ids : Optional[np.ndarray]
            Shape ``(N,)`` -- integer relation type ids.  When ``None``,
            all nodes default to relation ``0``.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)`` -- projected manifold embeddings.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.mode == "truncate":
            return self._project_truncate(embeddings)

        return self._project_learned(embeddings, signatures, relation_ids)

    def project_single(self, embedding: np.ndarray) -> np.ndarray:
        """Project a single query embedding onto the manifold space.

        In ``"truncate"`` mode this performs simple truncation / padding.
        In ``"learned"`` mode a zero-signature with default relation id
        ``0`` is used (suitable for query-time projection when no
        signature is available).

        Parameters
        ----------
        embedding : np.ndarray
            Shape ``(semantic_dim,)`` -- a single semantic vector.

        Returns
        -------
        np.ndarray
            Shape ``(output_dim,)`` -- the projected manifold vector.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if self.mode == "truncate":
            result = self._project_truncate(embedding)
        else:
            # Build a dummy zero-signature for the single query
            dummy_sig = np.zeros((1, self.signature_dim), dtype=np.float32)
            result = self._project_learned(embedding, [{}], None)

        return result.ravel()

    def update(self, new_embeddings: np.ndarray) -> None:
        """Incrementally update the manifold using Algorithm 3
        (Grassmannian mean shift).

        The existing Grassmannian mean is shifted toward the mean of the
        new embeddings via a weighted geodesic interpolation, and all
        stored embeddings are nudged accordingly.

        Parameters
        ----------
        new_embeddings : np.ndarray
            Shape ``(M, output_dim)`` -- newly projected manifold
            embeddings to incorporate.
        """
        if self._manifold_embeddings is None:
            self._manifold_embeddings = new_embeddings.copy()
            logger.info(
                "Initialized manifold buffer with %d embeddings (dim=%d).",
                new_embeddings.shape[0],
                new_embeddings.shape[1],
            )
            return

        n_old = self._manifold_embeddings.shape[0]
        n_new = new_embeddings.shape[0]

        # Weighted mean on the Grassmannian
        alpha = n_new / (n_old + n_new)  # blending weight
        old_mean = np.mean(self._manifold_embeddings, axis=0)
        new_mean = np.mean(new_embeddings, axis=0)

        # Geodesic interpolation (simplified linear blend)
        blended = (1.0 - alpha) * old_mean + alpha * new_mean
        blended = blended / (np.linalg.norm(blended) + 1e-8)

        # Shift all existing embeddings slightly toward the new mean
        shift = 0.01 * (blended - old_mean)
        self._manifold_embeddings = self._manifold_embeddings + shift

        # Re-normalize
        norms = np.linalg.norm(
            self._manifold_embeddings, axis=1, keepdims=True
        )
        self._manifold_embeddings = self._manifold_embeddings / np.clip(
            norms, 1e-8, None
        )

        logger.debug(
            "Manifold updated: %d old + %d new embeddings (alpha=%.4f).",
            n_old,
            n_new,
            alpha,
        )

    # ------------------------------------------------------------------
    # Internal: truncate mode
    # ------------------------------------------------------------------

    def _project_truncate(self, embeddings: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad embeddings to ``output_dim``, then L2
        normalize.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, D)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``.
        """
        N, D = embeddings.shape
        manifold_dim = self.output_dim

        if manifold_dim <= D:
            projected = embeddings[:, :manifold_dim].copy()
        else:
            projected = np.zeros((N, manifold_dim), dtype=np.float32)
            projected[:, :D] = embeddings

        # L2 normalize each row
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.clip(norms, 1e-8, None)

        return projected.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: learned mode
    # ------------------------------------------------------------------

    def _project_learned(
        self,
        embeddings: np.ndarray,
        signatures: Optional[List[Dict]],
        relation_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Project embeddings using the HierarchicalManifoldEncoder.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, semantic_dim)``.
        signatures : Optional[List[Dict]]
            Per-node signature dictionaries.
        relation_ids : Optional[np.ndarray]
            Shape ``(N,)`` integer relation ids.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``.
        """
        import torch

        if self._encoder is None:
            raise RuntimeError(
                "HierarchicalManifoldEncoder not initialized. "
                "This should not happen in 'learned' mode."
            )

        N = embeddings.shape[0]

        # Build signature tensor from dicts
        sig_tensor = self._build_signature_tensor(signatures, N)

        # Build relation_ids tensor
        if relation_ids is not None:
            rel_tensor = torch.tensor(
                relation_ids, dtype=torch.long
            )
        else:
            rel_tensor = None

        # Forward pass (no gradient computation)
        self._encoder.eval()
        with torch.no_grad():
            sem_tensor = torch.tensor(
                embeddings, dtype=torch.float32
            )
            output = self._encoder.encode_all(sem_tensor, sig_tensor, rel_tensor)

        return output.numpy().astype(np.float32)

    def _build_signature_tensor(
        self,
        signatures: Optional[List[Dict]],
        n: int,
    ) -> "torch.Tensor":
        """Convert a list of signature dicts into a padded tensor.

        Each signature dict is expected to contain numeric fields such as
        ``"entity_count"``, ``"relation_count"``, and ``"n_slices"``.
        Missing keys default to ``0``.  The resulting feature vector is
        truncated or zero-padded to ``signature_dim``.

        Parameters
        ----------
        signatures : Optional[List[Dict]]
            Length-*N* list of signature dicts.  If ``None`` or shorter
            than *n*, zero-signatures are used for missing entries.
        n : int
            Expected batch size.

        Returns
        -------
        torch.Tensor
            Shape ``(N, signature_dim)``.
        """
        import torch

        sig_dim = self.signature_dim
        tensor = np.zeros((n, sig_dim), dtype=np.float32)

        if signatures is None:
            return torch.tensor(tensor, dtype=torch.float32)

        for i in range(min(len(signatures), n)):
            sig = signatures[i] if isinstance(signatures[i], dict) else {}
            features = np.array([
                float(sig.get("entity_count", 0)),
                float(sig.get("relation_count", 0)),
                float(sig.get("n_slices", 0)),
            ], dtype=np.float32)

            # Truncate or zero-pad to signature_dim
            feat_len = min(len(features), sig_dim)
            tensor[i, :feat_len] = features[:feat_len]

        return torch.tensor(tensor, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifold_embeddings(self) -> Optional[np.ndarray]:
        """Return the current manifold embedding buffer (or ``None``)."""
        return self._manifold_embeddings

    @property
    def encoder(self) -> Optional[HierarchicalManifoldEncoder]:
        """Return the underlying HierarchicalManifoldEncoder (learned mode
        only; ``None`` in truncate mode)."""
        return self._encoder
