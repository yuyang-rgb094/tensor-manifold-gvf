"""OAG (Open Academic Graph) schema definitions.

Defines node types, edge types, and graph data structures
for heterogeneous academic knowledge graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    """Types of nodes in the academic graph."""
    PAPER = "paper"
    AUTHOR = "author"
    JOURNAL = "journal"
    DISCIPLINE = "discipline"
    INSTITUTION = "institution"
    PROJECT = "project"


class EdgeType(str, Enum):
    """Types of edges in the academic graph."""
    CITATION = "citation"
    COLLABORATION = "collaboration"
    AFFILIATION = "affiliation"
    PUBLISHES_IN = "publishes_in"
    BELONGS_TO = "belongs_to"
    WORKS_ON = "works_on"
    FUNDED_BY = "funded_by"


@dataclass
class Node:
    """Represents a single node in the heterogeneous academic graph.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: Type of the node (e.g., paper, author).
        attributes: Dictionary of node attributes (title, abstract, etc.).
    """
    node_id: str
    node_type: NodeType
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Represents a directed edge in the heterogeneous academic graph.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        edge_type: Type of the edge (e.g., citation, collaboration).
        weight: Edge weight (default 1.0).
        timestamp: Optional timestamp for temporal edges.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    timestamp: Optional[float] = None


@dataclass
class HeteroAcademicGraph:
    """Heterogeneous academic knowledge graph with typed nodes and edges.

    Maintains adjacency structures indexed by edge type for efficient
    heterogeneous neighborhood traversal.

    Attributes:
        nodes: Dictionary mapping node_id -> Node.
        edges: List of Edge objects.
        adj_by_type: Nested dict: edge_type -> {source_id -> [target_ids]}.
        reverse_adj_by_type: Nested dict: edge_type -> {target_id -> [source_ids]}.
        node_type_map: Dictionary mapping node_id -> NodeType.
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    adj_by_type: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    reverse_adj_by_type: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    node_type_map: Dict[str, NodeType] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node object to add.
        """
        self.nodes[node.node_id] = node
        self.node_type_map[node.node_id] = node.node_type

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph and update adjacency structures.

        Args:
            edge: Edge object to add.
        """
        self.edges.append(edge)
        etype = edge.edge_type.value

        # Forward adjacency
        if etype not in self.adj_by_type:
            self.adj_by_type[etype] = {}
        if edge.source_id not in self.adj_by_type[etype]:
            self.adj_by_type[etype][edge.source_id] = []
        self.adj_by_type[etype][edge.source_id].append(edge.target_id)

        # Reverse adjacency
        if etype not in self.reverse_adj_by_type:
            self.reverse_adj_by_type[etype] = {}
        if edge.target_id not in self.reverse_adj_by_type[etype]:
            self.reverse_adj_by_type[etype][edge.target_id] = []
        self.reverse_adj_by_type[etype][edge.target_id].append(edge.source_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[str]:
        """Get neighbor node IDs for a given node.

        Args:
            node_id: ID of the query node.
            edge_type: Optional edge type filter. If None, returns neighbors
                       across all edge types.
            direction: 'outgoing' (default), 'incoming', or 'both'.

        Returns:
            List of neighbor node IDs.
        """
        neighbors = []

        if direction in ("outgoing", "both"):
            if edge_type is not None:
                adj = self.adj_by_type.get(edge_type, {})
                neighbors.extend(adj.get(node_id, []))
            else:
                for adj in self.adj_by_type.values():
                    neighbors.extend(adj.get(node_id, []))

        if direction in ("incoming", "both"):
            if edge_type is not None:
                radj = self.reverse_adj_by_type.get(edge_type, {})
                neighbors.extend(radj.get(node_id, []))
            else:
                for radj in self.reverse_adj_by_type.values():
                    neighbors.extend(radj.get(node_id, []))

        return neighbors

    def get_degree(self, node_id: str, edge_type: Optional[str] = None) -> int:
        """Get the degree of a node.

        Args:
            node_id: ID of the query node.
            edge_type: Optional edge type filter.

        Returns:
            Total degree (in + out) for the node.
        """
        return len(self.get_neighbors(node_id, edge_type=edge_type, direction="both"))

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph.

        Returns:
            Number of nodes.
        """
        return len(self.nodes)

    def get_edge_count(self) -> int:
        """Get total number of edges in the graph.

        Returns:
            Number of edges.
        """
        return len(self.edges)

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type.

        Args:
            node_type: NodeType to filter by.

        Returns:
            List of Node objects matching the type.
        """
        return [
            node for node in self.nodes.values()
            if node.node_type == node_type
        ]
