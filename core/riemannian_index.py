"""
Riemannian metric and manifold index for tensor-field-guided retrieval.

Combines FAISS approximate nearest neighbor search with Riemannian
metric reranking for high-quality similarity retrieval on manifold-
structured embeddings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class RiemannianMetricConfig:
    """Configuration for the Riemannian manifold index."""

    dimension: int = 128
    """Embedding dimension."""

    metric_type: str = "ip"
    """Distance metric: ``'ip'`` (inner product) or ``'l2'``."""

    nlist: int = 100
    """Number of Voronoi cells for IVF index."""

    nprobe: int = 10
    """Number of cells to probe at query time."""

    m_pq: int = 8
    """Number of sub-quantizers for PQ compression."""

    nbits: int = 8
    """Bits per sub-quantizer."""

    use_gpu: bool = False
    """Whether to use FAISS GPU resources."""

    rerank_factor: int = 5
    """Over-fetch factor for Riemannian reranking (e.g. 5x)."""

    rebuild_threshold: float = 0.2
    """Fraction of new nodes that triggers index rebuild."""

    density_threshold: float = 0.8
    """Density threshold for dynamic traversal control."""

    temperature: float = 0.1
    """Temperature for Riemannian distance kernel."""


class TemporalRiemannianMetric:
    """Riemannian metric tensor with temporal decay.

    g_{ij}(t) = exp(-|t_i - t_j| / tau) * delta_{ij}

    Parameters
    ----------
    config : RiemannianMetricConfig
        Metric configuration.
    """

    def __init__(self, config: Optional[RiemannianMetricConfig] = None):
        self.config = config or RiemannianMetricConfig()

    def compute_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t_x: Optional[float] = None,
        t_y: Optional[float] = None,
    ) -> float:
        """Compute Riemannian distance between two embeddings.

        d_R(x, y) = ||x - y||^2 * temporal_decay

        Parameters
        ----------
        x, y : np.ndarray
            Shape (d,).
        t_x, t_y : Optional[float]
            Timestamps for temporal decay.

        Returns
        -------
        float
            Riemannian distance.
        """
        euclidean_sq = float(np.sum((x - y) ** 2))

        if t_x is not None and t_y is not None:
            dt = abs(t_x - t_y)
            temporal_decay = np.exp(-dt / (self.config.temperature * 86400.0))
        else:
            temporal_decay = 1.0

        return euclidean_sq * temporal_decay

    def compute_distance_batch(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        t_query: Optional[float] = None,
        t_candidates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch Riemannian distance computation.

        Parameters
        ----------
        query : np.ndarray
            Shape (d,).
        candidates : np.ndarray
            Shape (N, d).
        t_query : Optional[float]
        t_candidates : Optional[np.ndarray]
            Shape (N,).

        Returns
        -------
        np.ndarray
            Shape (N,) -- distances.
        """
        diffs = candidates - query[np.newaxis, :]
        euclidean_sq = np.sum(diffs ** 2, axis=1)

        if t_query is not None and t_candidates is not None:
            dt = np.abs(t_candidates - t_query)
            temporal_decay = np.exp(-dt / (self.config.temperature * 86400.0))
        else:
            temporal_decay = 1.0

        return euclidean_sq * temporal_decay


class TemporalRiemannianManifoldIndex:
    """FAISS-backed index with Riemannian metric reranking.

    Parameters
    ----------
    config : RiemannianMetricConfig
        Index configuration.
    """

    def __init__(self, config: Optional[RiemannianMetricConfig] = None):
        self.config = config or RiemannianMetricConfig()
        self.metric = TemporalRiemannianMetric(self.config)

        self._embeddings: Optional[np.ndarray] = None  # (N, d)
        self._timestamps: Optional[np.ndarray] = None   # (N,)
        self._node_ids: List[int] = []
        self._faiss_index = None
        self._total_added: int = 0

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,
        node_ids: Optional[List[int]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> None:
        """Build the FAISS index from embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, d).  Must be detached and on CPU.
        node_ids : Optional[List[int]]
            External node ids.  Defaults to range(N).
        timestamps : Optional[np.ndarray]
            Shape (N,).
        """
        # Ensure numpy on CPU
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        if timestamps is not None and hasattr(timestamps, "detach"):
            timestamps = timestamps.detach().cpu().numpy()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        N, d = embeddings.shape

        assert d == self.config.dimension, (
            f"Embedding dim {d} != config dimension {self.config.dimension}"
        )

        self._embeddings = embeddings
        self._timestamps = timestamps
        self._node_ids = node_ids if node_ids is not None else list(range(N))
        self._total_added = N

        self._build_faiss_index(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build the appropriate FAISS index.

        Uses IVF+PQ for large datasets (> nlist * 10), FlatIP otherwise.
        """
        if not HAS_FAISS:
            raise RuntimeError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        d = embeddings.shape[1]
        N = embeddings.shape[0]

        if N > self.config.nlist * 10:
            # IVF + PQ for large datasets
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(
                quantizer,
                d,
                self.config.nlist,
                self.config.m_pq,
                self.config.nbits,
            )
            index.nprobe = self.config.nprobe
            if not index.is_trained:
                index.train(embeddings)
            index.add(embeddings)
        else:
            # Flat index for small datasets
            if self.config.metric_type == "l2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(embeddings)

        self._faiss_index = index

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_knn(
        self,
        query: np.ndarray,
        k: int = 10,
        t_query: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors with Riemannian reranking.

        First retrieves ``k * rerank_factor`` candidates via FAISS,
        then reranks using the Riemannian metric.

        Parameters
        ----------
        query : np.ndarray
            Shape (d,).
        k : int
            Number of final results.
        t_query : Optional[float]
            Query timestamp for temporal decay.

        Returns
        -------
        List[Tuple[int, float]]
            List of ``(node_id, riemannian_distance)`` pairs, sorted
            by ascending distance.
        """
        if hasattr(query, "detach"):
            query = query.detach().cpu().numpy()
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)

        # Over-fetch for reranking
        fetch_k = min(k * self.config.rerank_factor, self._total_added)
        distances, indices = self._faiss_index.search(query, fetch_k)

        # Collect candidates
        candidates = []
        for i in range(fetch_k):
            idx = int(indices[0][i])
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            node_id = self._node_ids[idx]
            emb = self._embeddings[idx]
            t_cand = (
                float(self._timestamps[idx])
                if self._timestamps is not None
                else None
            )
            r_dist = self.metric.compute_distance(
                query.flatten(), emb, t_query, t_cand
            )
            candidates.append((node_id, r_dist, idx))

        # Sort by Riemannian distance
        candidates.sort(key=lambda x: x[1])

        return [(nid, dist) for nid, dist, _ in candidates[:k]]

    # ------------------------------------------------------------------
    # Dynamic maintenance
    # ------------------------------------------------------------------

    def dynamic_traversal(
        self,
        new_embeddings: np.ndarray,
        new_node_ids: Optional[List[int]] = None,
        new_timestamps: Optional[np.ndarray] = None,
    ) -> bool:
        """Check density/threshold and rebuild if needed.

        Parameters
        ----------
        new_embeddings : np.ndarray
            New embeddings to potentially add.
        new_node_ids : Optional[List[int]]
        new_timestamps : Optional[np.ndarray]

        Returns
        -------
        bool
            ``True`` if the index was rebuilt.
        """
        if hasattr(new_embeddings, "detach"):
            new_embeddings = new_embeddings.detach().cpu().numpy()

        n_new = new_embeddings.shape[0]
        ratio = n_new / max(self._total_added, 1)

        if ratio >= self.config.rebuild_threshold:
            # Rebuild with all data
            all_emb = np.vstack([self._embeddings, new_embeddings])
            all_ids = self._node_ids + (
                new_node_ids if new_node_ids else list(range(
                    self._total_added, self._total_added + n_new
                ))
            )
            all_ts = None
            if self._timestamps is not None and new_timestamps is not None:
                all_ts = np.concatenate([self._timestamps, new_timestamps])

            self.build_index(all_emb, all_ids, all_ts)
            return True

        return False

    def incremental_add(
        self,
        embeddings: np.ndarray,
        node_ids: Optional[List[int]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> None:
        """Add new embeddings to the index incrementally.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (M, d).  Must be detached and on CPU.
        node_ids : Optional[List[int]]
        timestamps : Optional[np.ndarray]
        """
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        if timestamps is not None and hasattr(timestamps, "detach"):
            timestamps = timestamps.detach().cpu().numpy()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        M = embeddings.shape[0]

        new_ids = node_ids if node_ids else list(
            range(self._total_added, self._total_added + M)
        )

        self._node_ids.extend(new_ids)
        self._embeddings = np.vstack([self._embeddings, embeddings])
        if timestamps is not None:
            if self._timestamps is not None:
                self._timestamps = np.concatenate([self._timestamps, timestamps])
            else:
                self._timestamps = timestamps.copy()

        self._total_added += M
        self._faiss_index.add(embeddings)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_embedding(self, node_id: int) -> Optional[np.ndarray]:
        """Retrieve the embedding for a node.

        Parameters
        ----------
        node_id : int

        Returns
        -------
        Optional[np.ndarray]
            Shape (d,) or ``None`` if not found.
        """
        if node_id not in self._node_ids:
            return None
        idx = self._node_ids.index(node_id)
        return self._embeddings[idx].copy()
