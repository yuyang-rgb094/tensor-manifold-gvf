"""Graph utility functions for heterogeneous academic graphs.

Provides neighborhood traversal, edge type analysis, and graph
statistics computation for the OAG data structures.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..data.oag_schema import EdgeType, HeteroAcademicGraph


def get_k_order_neighbors(
    graph: HeteroAcademicGraph,
    node_id: str,
    k: int = 3,
    edge_type: Optional[str] = None,
) -> Dict[int, Set[str]]:
    """Get k-order neighbors of a node using BFS.

    Args:
        graph: The heterogeneous academic graph.
        node_id: Starting node ID.
        k: Maximum order of neighbors to collect.
        edge_type: Optional edge type filter. If None, considers all types.

    Returns:
        Dictionary mapping order (1..k) to sets of neighbor node IDs.
    """
    neighbors_by_order: Dict[int, Set[str]] = {}
    visited = {node_id}
    current_level = {node_id}

    for order in range(1, k + 1):
        next_level: Set[str] = set()
        for nid in current_level:
            nbrs = graph.get_neighbors(nid, edge_type=edge_type, direction="both")
            for nbr in nbrs:
                if nbr not in visited:
                    visited.add(nbr)
                    next_level.add(nbr)
        neighbors_by_order[order] = next_level
        current_level = next_level

    return neighbors_by_order


def get_edge_types_between(
    graph: HeteroAcademicGraph,
    source_id: str,
    target_id: str,
) -> List[str]:
    """Get all edge types between two nodes.

    Args:
        graph: The heterogeneous academic graph.
        source_id: Source node ID.
        target_id: Target node ID.

    Returns:
        List of edge type strings connecting the two nodes.
    """
    edge_types = []

    for etype, adj in graph.adj_by_type.items():
        targets = adj.get(source_id, [])
        if target_id in targets:
            edge_types.append(etype)

    for etype, radj in graph.reverse_adj_by_type.items():
        sources = radj.get(target_id, [])
        if source_id in sources:
            if etype not in edge_types:
                edge_types.append(etype)

    return edge_types


def get_primary_relation_type(
    graph: HeteroAcademicGraph,
    source_id: str,
    target_id: str,
) -> Optional[str]:
    """Get the primary (most frequent) relation type between two nodes.

    Args:
        graph: The heterogeneous academic graph.
        source_id: Source node ID.
        target_id: Target node ID.

    Returns:
        The most frequent edge type, or None if no edge exists.
    """
    type_counts: Dict[str, int] = defaultdict(int)

    for edge in graph.edges:
        if (edge.source_id == source_id and edge.target_id == target_id) or \
           (edge.source_id == target_id and edge.target_id == source_id):
            type_counts[edge.edge_type.value] += 1

    if not type_counts:
        return None

    return max(type_counts, key=type_counts.get)


def calculate_graph_density(graph: HeteroAcademicGraph) -> float:
    """Calculate the density of the graph.

    Density = 2 * |E| / (|V| * (|V| - 1)) for undirected interpretation.

    Args:
        graph: The heterogeneous academic graph.

    Returns:
        Graph density value in [0, 1].
    """
    n = graph.get_node_count()
    if n <= 1:
        return 0.0

    m = graph.get_edge_count()
    max_edges = n * (n - 1)
    return 2.0 * m / max_edges


def get_average_degree(graph: HeteroAcademicGraph) -> float:
    """Calculate the average degree of the graph.

    Args:
        graph: The heterogeneous academic graph.

    Returns:
        Average degree across all nodes.
    """
    n = graph.get_node_count()
    if n == 0:
        return 0.0

    total_degree = sum(graph.get_degree(nid) for nid in graph.nodes)
    return total_degree / n
