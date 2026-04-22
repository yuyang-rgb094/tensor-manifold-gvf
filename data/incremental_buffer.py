"""Incremental buffer for staging graph updates.

Buffers new nodes and edges before committing them to the main graph,
enabling efficient batch processing of incremental updates.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .oag_schema import Edge, Node


class IncrementalBuffer:
    """Buffer for staging incremental graph updates.

    Accumulates new nodes and edges, and flushes them in batch to the
    target graph when requested.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the incremental buffer.

        Args:
            max_size: Maximum number of items (nodes + edges) before
                      automatic flush is recommended.
        """
        self._pending_nodes: List[Node] = []
        self._pending_edges: List[Edge] = []
        self.max_size = max_size

    def add_node(self, node: Node) -> None:
        """Add a node to the buffer.

        Args:
            node: Node object to buffer.
        """
        self._pending_nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the buffer.

        Args:
            edge: Edge object to buffer.
        """
        self._pending_edges.append(edge)

    def add_batch(
        self,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
    ) -> None:
        """Add a batch of nodes and edges to the buffer.

        Args:
            nodes: Optional list of Node objects.
            edges: Optional list of Edge objects.
        """
        if nodes:
            self._pending_nodes.extend(nodes)
        if edges:
            self._pending_edges.extend(edges)

    def flush(self) -> Tuple[List[Node], List[Edge]]:
        """Flush all pending nodes and edges from the buffer.

        Returns:
            Tuple of (nodes, edges) lists. The buffer is cleared after flush.
        """
        nodes = self._pending_nodes
        edges = self._pending_edges
        self._pending_nodes = []
        self._pending_edges = []
        return nodes, edges

    def pending_count(self) -> int:
        """Get the total number of pending items in the buffer.

        Returns:
            Sum of pending nodes and edges.
        """
        return len(self._pending_nodes) + len(self._pending_edges)

    def is_empty(self) -> bool:
        """Check whether the buffer is empty.

        Returns:
            True if no pending nodes or edges.
        """
        return len(self._pending_nodes) == 0 and len(self._pending_edges) == 0
