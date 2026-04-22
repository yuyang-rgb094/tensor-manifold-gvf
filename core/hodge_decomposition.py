"""
Hodge Decomposition on graph manifolds.

Decomposes node features into exact, co-exact, and harmonic components
using a local approximation strategy for efficiency.
"""

from typing import Dict, List, Optional

import numpy as np

from .dec_operators import DiscreteExteriorCalculus


class HodgeDecomposition:
    """Hodge decomposition of graph node features.

    Uses a local approximation:
        - exact     = neighbor mean (gradient component)
        - co-exact  = original - exact (curl component)
        - harmonic  = global mean (topological invariant)

    Parameters
    ----------
    dec : DiscreteExteriorCalculus
        Pre-built DEC operator instance.
    compensation_strength : float
        Scaling factor applied to the co-exact component during
        error compensation.  Default 0.5.
    """

    def __init__(
        self,
        dec: DiscreteExteriorCalculus,
        compensation_strength: float = 0.5,
    ):
        self.dec = dec
        self.compensation_strength = compensation_strength

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        node_features: np.ndarray,
        node_ids: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Decompose node features into exact / co-exact / harmonic.

        Parameters
        ----------
        node_features : np.ndarray
            Shape (N, d) -- feature matrix.  N may be a subset of the
            full graph when *node_ids* is provided.
        node_ids : Optional[List[int]]
            Global node ids corresponding to rows of *node_features*.
            If given, a local index mapping is built so that DEC
            neighbor lookups are translated correctly.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: ``"exact"``, ``"coexact"``, ``"harmonic"``.
            Each value has shape (N, d).
        """
        local_idx = self._build_local_index(node_ids, node_features.shape[0])

        N, d = node_features.shape
        exact = np.zeros_like(node_features)
        harmonic = np.mean(node_features, axis=0, keepdims=True)  # (1, d)

        for local_i in range(N):
            global_id = node_ids[local_i] if node_ids is not None else local_i
            topo_feat = self.dec.extract_topological_feature(
                global_id, node_features, local_idx=local_idx
            )
            exact[local_i] = topo_feat

        coexact = node_features - exact

        return {
            "exact": exact,
            "coexact": coexact,
            "harmonic": np.broadcast_to(harmonic, (N, d)).copy(),
        }

    def error_compensation(
        self,
        raw_signature: np.ndarray,
        node_id: int,
        node_features: np.ndarray,
        local_idx: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Compensate a raw signature using the co-exact component.

        Extracts the local subgraph features around *node_id*, performs
        Hodge decomposition, and returns the co-exact component scaled
        by *compensation_strength*.

        Parameters
        ----------
        raw_signature : np.ndarray
            Shape (d,) -- the raw diffusion signature to compensate.
        node_id : int
            Global node id.
        node_features : np.ndarray
            Shape (N, d) -- node feature matrix (full or local).
        local_idx : Optional[Dict[int, int]]
            Mapping from global node ids to local indices into
            *node_features*.

        Returns
        -------
        np.ndarray
            Shape (d,) -- compensated signature.
        """
        # Collect local subgraph: node_id + its neighbors
        neighbors = self.dec._adj.get(node_id, [])
        subgraph_ids = [node_id] + neighbors

        # Map to local indices if not provided
        if local_idx is None:
            local_idx = {nid: idx for idx, nid in enumerate(subgraph_ids)}

        # Extract subgraph features
        sub_features_list = []
        valid_ids = []
        for nid in subgraph_ids:
            if nid in local_idx:
                sub_features_list.append(node_features[local_idx[nid]])
                valid_ids.append(nid)

        if not valid_ids:
            return raw_signature

        sub_features = np.stack(sub_features_list, axis=0)

        # Decompose the local subgraph
        decomp = self.decompose(sub_features, node_ids=valid_ids)
        coexact = decomp["coexact"]

        # The co-exact component for the target node (first in valid_ids)
        target_local = 0  # node_id is first in valid_ids
        compensation = coexact[target_local] * self.compensation_strength

        return raw_signature + compensation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_local_index(
        node_ids: Optional[List[int]],
        n: int,
    ) -> Optional[Dict[int, int]]:
        """Build a global-to-local index mapping.

        Parameters
        ----------
        node_ids : Optional[List[int]]
            Global ids.  If ``None``, returns ``None`` (identity mapping).
        n : int
            Expected number of entries (used only for validation).

        Returns
        -------
        Optional[Dict[int, int]]
            Mapping {global_id: local_index} or ``None``.
        """
        if node_ids is None:
            return None
        return {gid: idx for idx, gid in enumerate(node_ids)}
