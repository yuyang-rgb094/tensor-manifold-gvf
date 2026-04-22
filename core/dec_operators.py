"""
Discrete Exterior Calculus (DEC) operators on graph manifolds.

Implements boundary operators, Hodge star operators, and exterior derivatives
for computing topological features on graph-structured data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags


class DiscreteExteriorCalculus:
    """Discrete Exterior Calculus operators on a graph.

    The graph is represented as an edge list with optional edge types and weights.
    Nodes are identified by integer indices.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    edges : List[Tuple[int, int]]
        List of directed edges (u, v).
    edge_weights : Optional[List[float]]
        Weight for each edge. Defaults to 1.0.
    edge_types : Optional[List[str]]
        Type label for each edge (e.g. "citation", "coauthor").
    """

    def __init__(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int]],
        edge_weights: Optional[List[float]] = None,
        edge_types: Optional[List[str]] = None,
    ):
        self.num_nodes = num_nodes
        self.edges = edges
        self.num_edges = len(edges)

        if edge_weights is None:
            self.edge_weights = [1.0] * self.num_edges
        else:
            self.edge_weights = list(edge_weights)

        self.edge_types = edge_types  # Optional[str] per edge

        # Build adjacency structures
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        self._edge_type_adj: Dict[str, Dict[int, List[int]]] = {}
        for idx, (u, v) in enumerate(self.edges):
            self._adj[u].append(v)
            self._adj[v].append(u)  # undirected neighborhood
            if self.edge_types is not None:
                etype = self.edge_types[idx]
                if etype not in self._edge_type_adj:
                    self._edge_type_adj[etype] = {i: [] for i in range(num_nodes)}
                self._edge_type_adj[etype][u].append(v)
                self._edge_type_adj[etype][v].append(u)

        # Build operators
        self._boundary_op: Optional[csr_matrix] = None
        self._hodge_star_0: Optional[csr_matrix] = None
        self._hodge_star_1: Optional[csr_matrix] = None
        self._degree_cache: Optional[Dict[int, int]] = None

        self._build_boundary_operator()
        self._build_hodge_star_operators()
        self._build_degree_cache()

    # ------------------------------------------------------------------
    # Operator construction
    # ------------------------------------------------------------------

    def _build_boundary_operator(self) -> None:
        """Build sparse boundary operator d_1 (|E| x |V|).

        For each directed edge e_{uv}, the boundary is:
            d_1(e_{uv}) = v - u
        which corresponds to a row with +1 at column v and -1 at column u.
        """
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for e_idx, (u, v) in enumerate(self.edges):
            # d_1(e_uv) = v - u
            rows.append(e_idx)
            cols.append(v)
            data.append(1.0)
            rows.append(e_idx)
            cols.append(u)
            data.append(-1.0)

        self._boundary_op = csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_edges, self.num_nodes),
        )

    def _build_hodge_star_operators(self) -> None:
        """Build Hodge star operators star_0 and star_1.

        star_0: diagonal matrix with node degrees on the diagonal.
        star_1: diagonal matrix with edge weights on the diagonal.
        """
        # star_0: node degrees
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        for u, v in self.edges:
            degrees[u] += 1.0
            degrees[v] += 1.0
        self._hodge_star_0 = diags(degrees, format="csr")

        # star_1: edge weights
        self._hodge_star_1 = diags(
            self.edge_weights, format="csr", shape=(self.num_edges, self.num_edges)
        )

    def _build_degree_cache(self) -> None:
        """Cache node degrees for fast lookup."""
        self._degree_cache = {}
        for i in range(self.num_nodes):
            self._degree_cache[i] = len(self._adj[i])

    # ------------------------------------------------------------------
    # Exterior derivative
    # ------------------------------------------------------------------

    def exterior_derivative(
        self,
        node_features: np.ndarray,
        edge_type: Optional[str] = None,
    ) -> np.ndarray:
        """Compute exterior derivative d_0: 0-form -> 1-form.

        d_0(f)(e_{uv}) = f(v) - f(u)

        Parameters
        ----------
        node_features : np.ndarray
            Shape (|V|, d) -- node-level 0-form values.
        edge_type : Optional[str]
            If given, compute d_0 only over edges of this type.

        Returns
        -------
        np.ndarray
            Shape (|E|, d) or (|E_type|, d) -- edge-level 1-form values.
        """
        if edge_type is not None:
            return self._exterior_derivative_by_type(node_features, edge_type)

        # Matrix form: d_0(f) = boundary_op @ f  (|E| x d)
        return self._boundary_op.dot(node_features)

    def _exterior_derivative_by_type(
        self,
        node_features: np.ndarray,
        edge_type: str,
    ) -> np.ndarray:
        """Compute exterior derivative filtered by edge type.

        Parameters
        ----------
        node_features : np.ndarray
            Shape (|V|, d).
        edge_type : str
            Edge type to filter by.

        Returns
        -------
        np.ndarray
            Shape (|E_type|, d).
        """
        if edge_type not in self._edge_type_adj:
            raise ValueError(f"Unknown edge type: {edge_type}")

        results: List[np.ndarray] = []
        for e_idx, (u, v) in enumerate(self.edges):
            if self.edge_types[e_idx] == edge_type:
                results.append(node_features[v] - node_features[u])

        if not results:
            return np.empty((0, node_features.shape[1]), dtype=node_features.dtype)

        return np.stack(results, axis=0)

    # ------------------------------------------------------------------
    # Hodge star
    # ------------------------------------------------------------------

    def hodge_star(self, k_form: np.ndarray, k: int) -> np.ndarray:
        """Apply Hodge star operator.

        Parameters
        ----------
        k_form : np.ndarray
            k-form values. Shape (|V|, d) for k=0, (|E|, d) for k=1.
        k : int
            Form degree (0 or 1).

        Returns
        -------
        np.ndarray
            Hodge-dual form values, same shape as input.
        """
        if k == 0:
            return self._hodge_star_0.dot(k_form)
        elif k == 1:
            return self._hodge_star_1.dot(k_form)
        else:
            raise ValueError(f"Unsupported form degree k={k}. Must be 0 or 1.")

    # ------------------------------------------------------------------
    # Topological feature extraction
    # ------------------------------------------------------------------

    def extract_topological_feature(
        self,
        node_id: int,
        node_features: np.ndarray,
        local_idx: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Extract topological feature for a single node.

        Computes the mean of (phi(u) - phi(v)) over all neighbors u of v,
        which is the average exterior derivative at the node.  O(d) per
        neighbor, O(d * deg(v)) total.

        Parameters
        ----------
        node_id : int
            Target node identifier (global index).
        node_features : np.ndarray
            Shape (|V|, d) -- all node feature vectors.
        local_idx : Optional[Dict[int, int]]
            Mapping from global node ids to local (contiguous) indices.
            If provided, node_id is interpreted as a global id and is
            translated via local_idx before accessing node_features.

        Returns
        -------
        np.ndarray
            Shape (d,) -- mean topological feature vector.
        """
        neighbors = self._adj.get(node_id, [])
        if not neighbors:
            d = node_features.shape[1]
            return np.zeros(d, dtype=node_features.dtype)

        diffs = []
        for u in neighbors:
            u_feat = node_features[local_idx[u]] if local_idx else node_features[u]
            v_feat = node_features[local_idx[node_id]] if local_idx else node_features[node_id]
            diffs.append(u_feat - v_feat)

        return np.mean(diffs, axis=0)

    # ------------------------------------------------------------------
    # Degree accessors
    # ------------------------------------------------------------------

    def get_node_degree(self, node_id: int) -> int:
        """Return the degree of a single node."""
        return self._degree_cache.get(node_id, 0)

    def get_all_degrees(self) -> Dict[int, int]:
        """Return a copy of the full degree cache {node_id: degree}."""
        return dict(self._degree_cache)
