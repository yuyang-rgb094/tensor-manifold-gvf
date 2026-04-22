"""
Similarity decomposition into semantic, graph, and temporal components.

Provides interpretable similarity breakdowns for tensor manifold
graph-vector fusion retrieval.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DecomposedSimilarity:
    """Decomposed similarity between a query and a candidate node.

    Attributes
    ----------
    query_id : int
        Query node id.
    candidate_id : int
        Candidate node id.
    text_semantic : float
        Cosine similarity on semantic embeddings V_sem.
    total : float
        Cosine similarity on full signatures S.
    graph : float
        Graph structure contribution: total - text_semantic.
    graph_by_type : Dict[str, float]
        Graph contribution split by edge type (direct edges + 2-hop
        shared neighbors).
    temporal : float
        Temporal similarity: 1 / (1 + |delta_t|).
    """

    query_id: int
    candidate_id: int
    text_semantic: float = 0.0
    total: float = 0.0
    graph: float = 0.0
    graph_by_type: Dict[str, float] = field(default_factory=dict)
    temporal: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Convert to a plain dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "candidate_id": self.candidate_id,
            "text_semantic": self.text_semantic,
            "total": self.total,
            "graph": self.graph,
            "graph_by_type": dict(self.graph_by_type),
            "temporal": self.temporal,
        }


class SimilarityDecomposer:
    """Decompose similarity between node pairs into interpretable components.

    Parameters
    ----------
    sem_vectors : Dict[int, np.ndarray]
        Semantic embedding vectors keyed by node id.
    signatures : Dict[int, np.ndarray]
        Full diffusion signature vectors keyed by node id.
    timestamps : Optional[Dict[int, float]]
        Per-node timestamps (epoch seconds).
    adj : Optional[Dict[int, List[int]]]
        Adjacency list.  If ``None``, graph components will be zero.
    edge_types : Optional[Dict[Tuple[int, int], str]]
        Mapping from (u, v) edge pairs to type labels.
    """

    def __init__(
        self,
        sem_vectors: Dict[int, np.ndarray],
        signatures: Dict[int, np.ndarray],
        timestamps: Optional[Dict[int, float]] = None,
        adj: Optional[Dict[int, List[int]]] = None,
        edge_types: Optional[Dict[Tuple[int, int], str]] = None,
    ):
        self.sem_vectors = sem_vectors
        self.signatures = signatures
        self.timestamps = timestamps or {}
        self.adj = adj or {}
        self.edge_types = edge_types or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        query_id: int,
        candidate_id: int,
    ) -> DecomposedSimilarity:
        """Decompose similarity between two nodes.

        Components:
            - text_semantic = cosine(V_sem(query), V_sem(candidate))
            - total         = cosine(S(query), S(candidate))
            - graph         = total - text_semantic
            - graph_by_type = split by edge type counts
              (direct edges + 2-hop shared neighbors)
            - temporal      = 1 / (1 + |delta_t|)

        Parameters
        ----------
        query_id : int
        candidate_id : int

        Returns
        -------
        DecomposedSimilarity
        """
        # Text semantic similarity
        text_semantic = self._cosine(
            self.sem_vectors.get(query_id),
            self.sem_vectors.get(candidate_id),
        )

        # Total signature similarity
        total = self._cosine(
            self.signatures.get(query_id),
            self.signatures.get(candidate_id),
        )

        # Graph contribution
        graph = total - text_semantic

        # Graph by type
        graph_by_type = self._count_edge_types_between(query_id, candidate_id)

        # Temporal similarity
        temporal = self._temporal_similarity(query_id, candidate_id)

        return DecomposedSimilarity(
            query_id=query_id,
            candidate_id=candidate_id,
            text_semantic=text_semantic,
            total=total,
            graph=graph,
            graph_by_type=graph_by_type,
            temporal=temporal,
        )

    def to_dict(
        self,
        query_id: int,
        candidate_id: int,
    ) -> Dict[str, object]:
        """Convenience: decompose and return as dict.

        Parameters
        ----------
        query_id : int
        candidate_id : int

        Returns
        -------
        Dict[str, object]
        """
        return self.decompose(query_id, candidate_id).to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_edge_types_between(
        self,
        u: int,
        v: int,
    ) -> Dict[str, float]:
        """Count edge types between u and v.

        Considers:
            1. Direct edges between u and v.
            2. 2-hop shared neighbors (edges u-w and v-w for common w).

        The contribution of each type is proportional to the count.

        Parameters
        ----------
        u, v : int
            Node ids.

        Returns
        -------
        Dict[str, float]
            Edge type -> normalized contribution.
        """
        type_counts: Dict[str, int] = defaultdict(int)

        # Direct edges
        neighbors_u = set(self.adj.get(u, []))
        neighbors_v = set(self.adj.get(v, []))

        for w in neighbors_u:
            if w == v:
                # Direct edge u-v
                etype = self.edge_types.get((u, v)) or self.edge_types.get((v, u)) or "unknown"
                type_counts[etype] += 1

        # 2-hop shared neighbors
        shared = neighbors_u & neighbors_v
        for w in shared:
            etype_uw = self.edge_types.get((u, w)) or self.edge_types.get((w, u)) or "unknown"
            etype_vw = self.edge_types.get((v, w)) or self.edge_types.get((w, v)) or "unknown"
            type_counts[etype_uw] += 1
            type_counts[etype_vw] += 1

        # Normalize to sum to 1.0
        total_count = sum(type_counts.values())
        if total_count == 0:
            return {}

        return {k: v / total_count for k, v in type_counts.items()}

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _temporal_similarity(self, u: int, v: int) -> float:
        """Temporal similarity: 1 / (1 + |delta_t|)."""
        t_u = self.timestamps.get(u)
        t_v = self.timestamps.get(v)
        if t_u is None or t_v is None:
            return 0.0
        dt = abs(t_u - t_v)
        return 1.0 / (1.0 + dt)
