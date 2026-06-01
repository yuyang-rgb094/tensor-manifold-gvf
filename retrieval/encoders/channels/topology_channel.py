"""Topology channel encoder using GraphSAGE.

Encodes citation graph structure into topology-aware embeddings using
GraphSAGE (inductive, supports new nodes).  Falls back to simple
networkx-based neighbor averaging when torch_geometric is unavailable.

Input type: ``Tuple[HeteroAcademicGraph, Dict[str, np.ndarray]]``
    - ``HeteroAcademicGraph``: the academic literature graph
    - ``Dict[str, np.ndarray]``: node_id -> initial feature vector

Output shape: ``(N, output_dim)`` (default 256).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import ChannelEncoder

logger = logging.getLogger(__name__)

# Check torch_geometric availability
_HAS_TORCH_GEOMETRIC = False
try:
    from torch_geometric.nn import SAGEConv
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    logger.info("torch_geometric not installed; topology channel will use networkx fallback.")


class TopologyChannelEncoder(ChannelEncoder, nn.Module):
    """Topology channel encoder using GraphSAGE.

    Parameters
    ----------
    input_dim : int
        Dimension of initial node features (e.g. BGE-M3 dense dim = 1024).
    hidden_dim : int
        Hidden dimension for GraphSAGE layers.
    output_dim : int
        Final output dimension.
    num_layers : int
        Number of GraphSAGE convolution layers.
    num_neighbors : int
        Number of neighbors to sample per layer (for large graphs).
    edge_types : Optional[List[str]]
        Edge types to include (default: citation and collaboration).
    dropout : float
        Dropout rate between layers.
    use_diffusion_signatures : bool
        Whether to use DiffusionSignatureUpdater output as additional features.
    """

    CHANNEL_NAME = "topology"

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        num_neighbors: int = 10,
        edge_types: Optional[List[str]] = None,
        dropout: float = 0.3,
        use_diffusion_signatures: bool = False,
    ):
        nn.Module.__init__(self)
        ChannelEncoder.__init__(self)

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._num_layers = num_layers
        self._num_neighbors = num_neighbors
        self._edge_types = edge_types or ["citation", "collaboration"]
        self._dropout = dropout
        self._use_diffusion_signatures = use_diffusion_signatures

        if _HAS_TORCH_GEOMETRIC:
            self._build_gnn_layers()
        else:
            self._gnn_layers = None
            logger.info("TopologyChannel: using networkx fallback (no torch_geometric)")

        # Projection from hidden_dim to output_dim
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def _build_gnn_layers(self) -> None:
        """Build GraphSAGE convolution layers."""
        self._gnn_layers = nn.ModuleList()
        # First layer: input_dim -> hidden_dim
        self._gnn_layers.append(SAGEConv(self._input_dim, self._hidden_dim))
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(self._num_layers - 2):
            self._gnn_layers.append(SAGEConv(self._hidden_dim, self._hidden_dim))
        # Last layer: hidden_dim -> hidden_dim (proj handles output_dim)
        if self._num_layers >= 2:
            self._gnn_layers.append(SAGEConv(self._hidden_dim, self._hidden_dim))

        self._dropout_layer = nn.Dropout(self._dropout)
        self._relu = nn.ReLU()

    # ------------------------------------------------------------------
    # ChannelEncoder interface
    # ------------------------------------------------------------------

    def encode(self, inputs: Any) -> np.ndarray:
        """Encode graph topology for all nodes.

        Parameters
        ----------
        inputs : Tuple[HeteroAcademicGraph, Dict[str, np.ndarray]]
            - graph: HeteroAcademicGraph instance
            - node_features: Dict mapping node_id -> feature vector (np.ndarray)

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)`` where N is the number of nodes
            in the graph.
        """
        graph, node_features = inputs

        if _HAS_TORCH_GEOMETRIC and self._gnn_layers is not None:
            return self._encode_gnn(graph, node_features)
        else:
            return self._encode_networkx_fallback(graph, node_features)

    def encode_single(self, input_data: Any) -> np.ndarray:
        """Encode topology for a single node.

        Parameters
        ----------
        input_data : Tuple[HeteroAcademicGraph, Dict[str, np.ndarray], str]
            - graph: HeteroAcademicGraph instance
            - node_features: Dict mapping node_id -> feature vector
            - node_id: The target node ID

        Returns
        -------
        np.ndarray
            Shape ``(output_dim,)``.
        """
        graph, node_features, node_id = input_data
        all_emb = self.encode((graph, node_features))
        # Find index of node_id
        node_ids = sorted(node_features.keys())
        if node_id in node_ids:
            idx = node_ids.index(node_id)
            return all_emb[idx]
        # Return zero vector for unknown nodes
        return np.zeros(self._output_dim, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def channel_name(self) -> str:
        return self.CHANNEL_NAME

    # ------------------------------------------------------------------
    # GraphSAGE encoding
    # ------------------------------------------------------------------

    def _encode_gnn(
        self,
        graph: Any,
        node_features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Encode using GraphSAGE via torch_geometric."""
        self.eval()
        with torch.no_grad():
            pyg_data = self._graph_to_pyg_data(graph, node_features)
            x, edge_index = pyg_data["x"], pyg_data["edge_index"]

            for i, conv in enumerate(self._gnn_layers):
                x = conv(x, edge_index)
                if i < len(self._gnn_layers) - 1:
                    x = self._relu(x)
                    x = self._dropout_layer(x)

            # Project to output_dim
            x = self.proj(x)
            x = self.layer_norm(x)
            return x.numpy().astype(np.float32)

    def _graph_to_pyg_data(
        self,
        graph: Any,
        node_features: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Convert HeteroAcademicGraph to PyG-compatible tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``x``: (N, input_dim) node feature matrix
            ``edge_index``: (2, E) edge index in COO format
        """
        node_ids = sorted(node_features.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        # Build node feature matrix
        feat_dim = next(iter(node_features.values())).shape[0] if node_features else self._input_dim
        x = np.zeros((N, feat_dim), dtype=np.float32)
        for nid, feat in node_features.items():
            if nid in node_id_to_idx:
                x[node_id_to_idx[nid]] = feat

        # Build edge index from selected edge types
        src_list: List[int] = []
        dst_list: List[int] = []

        adj = graph.adj_by_type if hasattr(graph, "adj_by_type") else {}
        reverse_adj = graph.reverse_adj_by_type if hasattr(graph, "reverse_adj_by_type") else {}

        for edge_type in self._edge_types:
            # Forward edges
            if edge_type in adj:
                for src_id, targets in adj[edge_type].items():
                    src_idx = node_id_to_idx.get(src_id)
                    if src_idx is None:
                        continue
                    for tgt_id in targets:
                        tgt_idx = node_id_to_idx.get(tgt_id)
                        if tgt_idx is not None:
                            src_list.append(src_idx)
                            dst_list.append(tgt_idx)

            # Reverse edges (make undirected for better message passing)
            if edge_type in reverse_adj:
                for src_id, targets in reverse_adj[edge_type].items():
                    src_idx = node_id_to_idx.get(src_id)
                    if src_idx is None:
                        continue
                    for tgt_id in targets:
                        tgt_idx = node_id_to_idx.get(tgt_id)
                        if tgt_idx is not None:
                            src_list.append(tgt_idx)
                            dst_list.append(src_idx)

        if not src_list:
            # No edges — return isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": edge_index,
        }

    # ------------------------------------------------------------------
    # NetworkX fallback
    # ------------------------------------------------------------------

    def _encode_networkx_fallback(
        self,
        graph: Any,
        node_features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Fallback encoding using simple neighbor averaging via networkx.

        1-hop neighbor feature averaging with decay.
        """
        import networkx as nx

        node_ids = sorted(node_features.keys())
        N = len(node_ids)
        feat_dim = self._input_dim

        # Build a simple undirected NetworkX graph
        G = nx.Graph()
        for nid in node_ids:
            G.add_node(nid)

        adj = graph.adj_by_type if hasattr(graph, "adj_by_type") else {}
        reverse_adj = graph.reverse_adj_by_type if hasattr(graph, "reverse_adj_by_type") else {}

        for edge_type in self._edge_types:
            if edge_type in adj:
                for src_id, targets in adj[edge_type].items():
                    for tgt_id in targets:
                        if src_id in G and tgt_id in G:
                            G.add_edge(src_id, tgt_id)
            if edge_type in reverse_adj:
                for src_id, targets in reverse_adj[edge_type].items():
                    for tgt_id in targets:
                        if src_id in G and tgt_id in G:
                            G.add_edge(src_id, tgt_id)

        # Compute 1-hop neighbor averages
        output = np.zeros((N, self._output_dim), dtype=np.float32)
        feat_matrix = np.zeros((N, feat_dim), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            if nid in node_features:
                feat_matrix[i] = node_features[nid]

        # Simple projection from feat_dim to hidden_dim
        rng = np.random.default_rng(42)
        proj_w = rng.standard_normal((feat_dim, self._hidden_dim)).astype(np.float32) * 0.01
        projected = feat_matrix @ proj_w  # (N, hidden_dim)

        for i, nid in enumerate(node_ids):
            neighbors = list(G.neighbors(nid))
            if neighbors:
                neighbor_indices = [node_ids.index(n) for n in neighbors if n in node_ids]
                if neighbor_indices:
                    neighbor_feats = projected[neighbor_indices].mean(axis=0)
                    # Blend self + neighbors
                    blended = 0.7 * projected[i] + 0.3 * neighbor_feats
                else:
                    blended = projected[i]
            else:
                blended = projected[i]

            # Project to output_dim
            proj_out = rng.standard_normal((self._hidden_dim, self._output_dim)).astype(np.float32) * 0.01
            output[i] = blended @ proj_out

        return output
