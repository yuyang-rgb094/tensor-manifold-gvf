"""
Tensor Decomposition Module.

Performs CP / Tucker decomposition on tensor signatures using SVD on
manifold embeddings to simulate tensor factorisation.  Produces
per-aspect factor matrices, explained variance ratios, and
reconstruction errors.

Extracted from ``retriever.py`` for modular pipeline usage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DecompositionResult:
    """Result of CP / Tucker decomposition on a retrieved node.

    Attributes
    ----------
    node_id : str
        Identifier of the decomposed node (document).
    core_tensor : np.ndarray
        The rank-reduced core tensor.
    factor_matrices : dict[str, np.ndarray]
        Factor matrices keyed by aspect name.
    explained_variance_ratio : float
        Fraction of total variance captured by the decomposition.
    aspect_contributions : dict[str, float]
        Per-aspect contribution weights.
    reconstruction_error : float
        Frobenius-norm reconstruction error.
    """

    node_id: str
    core_tensor: np.ndarray
    factor_matrices: Dict[str, np.ndarray]
    explained_variance_ratio: float
    aspect_contributions: Dict[str, float]
    reconstruction_error: float


# ---------------------------------------------------------------------------
# Tensor Decomposer
# ---------------------------------------------------------------------------

class TensorDecomposer:
    """Decompose tensor signatures via CP or Tucker factorisation.

    The decomposition is simulated by applying SVD to the manifold
    embedding of each node, yielding factor matrices, a core tensor,
    and per-aspect contribution weights.

    Parameters
    ----------
    decomposer_type : str
        Decomposition method: ``"cp"`` (default) or ``"tucker"``.
    rank : int
        Target rank for the decomposition (default ``8``).
    """

    def __init__(self, decomposer_type: str = "cp", rank: int = 8) -> None:
        self.decomposer_type = decomposer_type
        self.rank = rank

        # Populated by init()
        self._signatures: List[Dict[str, Any]] = []
        self._manifold_embeddings: Optional[np.ndarray] = None
        self._relation_types: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(
        self,
        signatures: List[Dict[str, Any]],
        manifold_embeddings: np.ndarray,
        relation_types: Optional[List[str]] = None,
    ) -> None:
        """Store references for later decomposition.

        Parameters
        ----------
        signatures : list[dict]
            Tensor signatures produced by
            :class:`~.signature_builder.SignatureBuilder`.
        manifold_embeddings : np.ndarray
            Manifold (Grassmannian) embeddings of shape ``(n, manifold_dim)``.
        relation_types : list[str], optional
            Ordered list of relation type names used as aspect labels.
        """
        self._signatures = signatures
        self._manifold_embeddings = manifold_embeddings
        self._relation_types = relation_types or []
        logger.info(
            "TensorDecomposer initialised (type=%s, rank=%d, nodes=%d)",
            self.decomposer_type,
            self.rank,
            len(signatures),
        )

    def decompose_node(
        self,
        idx: int,
        aspects: Optional[List[str]] = None,
    ) -> Optional[DecompositionResult]:
        """Perform CP / Tucker decomposition on a node's tensor signature.

        Uses SVD on the manifold embedding to simulate tensor
        decomposition, producing factor matrices, a core tensor,
        explained variance, and per-aspect contributions.

        Parameters
        ----------
        idx : int
            Index of the node in the signatures / embeddings arrays.
        aspects : list[str], optional
            Specific aspects to analyse.  Falls back to the relation
            types supplied to :meth:`init`, or ``["related"]``.

        Returns
        -------
        DecompositionResult or None
            Full decomposition details, or ``None`` if the signature at
            *idx* is empty.
        """
        if self._manifold_embeddings is None:
            raise RuntimeError(
                "TensorDecomposer has not been initialised. "
                "Call init() before decompose_node()."
            )

        sig = self._signatures[idx]
        if not sig:
            return None

        aspects = aspects or self._relation_types or ["related"]
        n_aspects = len(aspects)
        manifold_dim = self._manifold_embeddings.shape[1]
        rank = min(self.rank, n_aspects, manifold_dim)

        # Simulate CP decomposition via SVD on the embedding
        emb = self._manifold_embeddings[idx].reshape(1, -1)
        U, S, Vt = np.linalg.svd(emb, full_matrices=False)

        # Clamp rank to available singular values
        actual_rank = min(rank, len(S))

        # Factor matrices (one per mode)
        factor_matrices: Dict[str, np.ndarray] = {}
        for i, aspect in enumerate(aspects[:actual_rank]):
            factor_matrices[aspect] = U[0, :actual_rank] * S[:actual_rank] * (i + 1) / actual_rank

        # Core tensor
        core_tensor = np.diag(S[:actual_rank]).reshape(actual_rank, 1, actual_rank)

        # Explained variance
        total_var = float(np.sum(S ** 2))
        explained_var = float(np.sum(S[:actual_rank] ** 2) / max(total_var, 1e-8))

        # Aspect contributions
        aspect_contributions: Dict[str, float] = {}
        for i, aspect in enumerate(aspects[:actual_rank]):
            weight = float(S[i] / max(total_var, 1e-8))
            aspect_contributions[aspect] = round(weight, 6)

        # Reconstruction error
        reconstructed = U[:, :actual_rank] @ np.diag(S[:actual_rank]) @ Vt[:actual_rank, :]
        error = float(np.linalg.norm(emb - reconstructed))

        logger.debug(
            "Decomposed node idx=%d (node_id=%s): explained_var=%.4f, error=%.6f",
            idx,
            sig.get("doc_id", ""),
            explained_var,
            error,
        )

        return DecompositionResult(
            node_id=sig.get("doc_id", ""),
            core_tensor=core_tensor,
            factor_matrices=factor_matrices,
            explained_variance_ratio=explained_var,
            aspect_contributions=aspect_contributions,
            reconstruction_error=error,
        )
