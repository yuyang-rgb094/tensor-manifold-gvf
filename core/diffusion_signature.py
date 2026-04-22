"""
Diffusion signature computation on graph manifolds.

Combines k-hop neighborhood aggregation with DEC topological features
and Hodge decomposition error compensation to produce robust node
signatures.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dec_operators import DiscreteExteriorCalculus
from .hodge_decomposition import HodgeDecomposition


@dataclass
class SignatureConfig:
    """Configuration for diffusion signature computation."""

    k_hops: int = 3
    """Number of hops for neighborhood aggregation."""

    decay_factor: float = 0.5
    """Exponential decay factor per hop."""

    lambda_sem: float = 0.7
    """Weight for semantic similarity in edge weight computation."""

    nu_temporal: float = 0.3
    """Weight for temporal similarity in edge weight computation."""

    hybrid_alpha: float = 0.6
    """Blending weight: alpha * diffusion + (1-alpha) * topological."""

    compensation_strength: float = 0.5
    """Strength of Hodge error compensation."""

    temporal_scale: float = 86400.0
    """Scale (in seconds) for temporal normalization. Default 1 day."""


class DiffusionSignatureUpdater:
    """Compute and update diffusion signatures for graph nodes.

    Parameters
    ----------
    dec : DiscreteExteriorCalculus
        Pre-built DEC operator instance.
    sem_vectors : Dict[int, np.ndarray]
        Semantic embedding vectors keyed by node id.  The keys of this
        dict (not ``graph.nodes``) are used to determine ``node_ids``
        and ``node_idx``, which correctly handles subgraphs.
    timestamps : Optional[Dict[int, float]]
        Per-node timestamps (epoch seconds).
    config : SignatureConfig
        Hyperparameters.
    """

    def __init__(
        self,
        dec: DiscreteExteriorCalculus,
        sem_vectors: Dict[int, np.ndarray],
        timestamps: Optional[Dict[int, float]] = None,
        config: Optional[SignatureConfig] = None,
    ):
        self.dec = dec
        self.sem_vectors = sem_vectors
        self.timestamps = timestamps or {}
        self.config = config or SignatureConfig()

        # Build node index from sem_vectors keys (handles subgraphs)
        self.node_ids: List[int] = sorted(sem_vectors.keys())
        self.node_idx: Dict[int, int] = {nid: i for i, nid in enumerate(self.node_ids)}

        # Build feature matrix aligned with self.node_ids
        self.node_features = np.stack(
            [sem_vectors[nid] for nid in self.node_ids], axis=0
        )  # (N, d)

        # Hodge decomposition
        self.hodge = HodgeDecomposition(
            dec, compensation_strength=self.config.compensation_strength
        )

        # Cached signatures
        self._signatures: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_k_order_neighbors(
        self,
        node_id: int,
        k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """BFS to collect k-hop neighbors with decay weights.

        Parameters
        ----------
        node_id : int
            Source node.
        k : Optional[int]
            Hop count.  Defaults to ``config.k_hops``.

        Returns
        -------
        List[Tuple[int, float]]
            List of ``(neighbor_id, weight)`` pairs.  Weight decays as
            ``decay_factor ** hop``.
        """
        k = k or self.config.k_hops
        visited = {node_id}
        queue: deque = deque()
        # (node, hop)
        queue.append((node_id, 0))
        result: List[Tuple[int, float]] = []

        while queue:
            current, hop = queue.popleft()
            if hop > 0:
                weight = self.config.decay_factor ** hop
                result.append((current, weight))

            if hop < k:
                for neighbor in self.dec._adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, hop + 1))

        return result

    def compute_edge_weight(
        self,
        u: int,
        v: int,
    ) -> float:
        """Compute hybrid edge weight between two nodes.

        w(u, v) = lambda * cosine_sem(u, v) + nu * temporal_sim(u, v)

        Parameters
        ----------
        u, v : int
            Node ids.

        Returns
        -------
        float
            Edge weight in [0, 1].
        """
        sem_sim = self._cosine_similarity(u, v)
        temp_sim = self._temporal_similarity(u, v)

        return (
            self.config.lambda_sem * sem_sim
            + self.config.nu_temporal * temp_sim
        )

    def update_signature(self, node_id: int) -> np.ndarray:
        """Compute the diffusion signature for a single node.

        Steps:
            1. Collect k-hop neighbors with decay.
            2. Compute DEC topological feature.
            3. Form hybrid signature (weighted diffusion + topological).
            4. Apply Hodge error compensation.

        Parameters
        ----------
        node_id : int
            Target node id.

        Returns
        -------
        np.ndarray
            Shape (d,) -- the updated signature.
        """
        # Step 1: k-hop neighbors
        neighbors = self.get_k_order_neighbors(node_id)

        # Step 2: DEC topological feature
        topo_feat = self.dec.extract_topological_feature(
            node_id, self.node_features, local_idx=self.node_idx
        )

        # Step 3: Hybrid signature
        if neighbors:
            # Weighted sum of neighbor semantic vectors
            weighted_sum = np.zeros_like(self.node_features[0])
            total_weight = 0.0
            for nid, w in neighbors:
                if nid in self.node_idx:
                    feat = self.node_features[self.node_idx[nid]]
                    edge_w = self.compute_edge_weight(node_id, nid)
                    weighted_sum += w * edge_w * feat
                    total_weight += w * edge_w

            if total_weight > 0:
                diffusion_component = weighted_sum / total_weight
            else:
                diffusion_component = self.node_features[
                    self.node_idx[node_id]
                ]
        else:
            diffusion_component = self.node_features[self.node_idx[node_id]]

        alpha = self.config.hybrid_alpha
        hybrid = alpha * diffusion_component + (1 - alpha) * topo_feat

        # Step 4: Hodge error compensation
        compensated = self.hodge.error_compensation(
            raw_signature=hybrid,
            node_id=node_id,
            node_features=self.node_features,
            local_idx=self.node_idx,
        )

        self._signatures[node_id] = compensated
        return compensated

    def update_all(self) -> Dict[int, np.ndarray]:
        """Update signatures for all nodes in the index.

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping {node_id: signature}.
        """
        for nid in self.node_ids:
            self.update_signature(nid)
        return dict(self._signatures)

    def incremental_update(
        self,
        updated_node_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """Incrementally update signatures for a subset of nodes.

        Parameters
        ----------
        updated_node_ids : List[int]
            Node ids whose signatures need recomputation.

        Returns
        -------
        Dict[int, np.ndarray]
            Updated signatures for the requested nodes.
        """
        results = {}
        for nid in updated_node_ids:
            if nid in self.node_idx:
                sig = self.update_signature(nid)
                results[nid] = sig
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, u: int, v: int) -> float:
        """Cosine similarity between semantic vectors of u and v."""
        if u not in self.sem_vectors or v not in self.sem_vectors:
            return 0.0
        vec_u = self.sem_vectors[u]
        vec_v = self.sem_vectors[v]
        norm_u = np.linalg.norm(vec_u)
        norm_v = np.linalg.norm(vec_v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(vec_u, vec_v) / (norm_u * norm_v))

    def _temporal_similarity(self, u: int, v: int) -> float:
        """Temporal similarity: 1 / (1 + |dt| / scale)."""
        t_u = self.timestamps.get(u)
        t_v = self.timestamps.get(v)
        if t_u is None or t_v is None:
            return 0.0
        dt = abs(t_u - t_v)
        return 1.0 / (1.0 + dt / self.config.temporal_scale)

    def _normalize_temporal(self, timestamps: Dict[int, float]) -> Dict[int, float]:
        """Normalize timestamps to [0, 1] range.

        Parameters
        ----------
        timestamps : Dict[int, float]
            Raw timestamps.

        Returns
        -------
        Dict[int, float]
            Normalized timestamps.
        """
        if not timestamps:
            return timestamps
        values = list(timestamps.values())
        t_min = min(values)
        t_max = max(values)
        span = t_max - t_min
        if span == 0:
            return {k: 0.0 for k in timestamps}
        return {k: (v - t_min) / span for k, v in timestamps.items()}
