"""OAG (Open Academic Graph) data loader.

Supports loading academic graph data from JSON Lines or JSON array formats,
with incremental update capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .oag_schema import (
    Edge,
    EdgeType,
    HeteroAcademicGraph,
    Node,
    NodeType,
)


class OAGLoader:
    """Loader for heterogeneous academic graph data.

    Supports JSON Lines (.jsonl) and JSON array (.json) file formats.
    Extracts nodes and edges from academic paper records.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OAG loader.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    def load_graph(self, filepath: Union[str, Path]) -> HeteroAcademicGraph:
        """Load a graph from a JSON Lines or JSON array file.

        Args:
            filepath: Path to the data file (.jsonl or .json).

        Returns:
            HeteroAcademicGraph populated with nodes and edges.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        graph = HeteroAcademicGraph()

        if filepath.suffix == ".jsonl":
            self._load_jsonl(filepath, graph)
        elif filepath.suffix == ".json":
            self._load_json(filepath, graph)
        else:
            raise ValueError(
                f"Unsupported file format: {filepath.suffix}. "
                "Use .jsonl or .json"
            )

        return graph

    def load_incremental(
        self,
        filepath: Union[str, Path],
        existing_graph: HeteroAcademicGraph,
    ) -> HeteroAcademicGraph:
        """Incrementally load new data into an existing graph.

        Args:
            filepath: Path to the new data file.
            existing_graph: Existing graph to update.

        Returns:
            Updated HeteroAcademicGraph.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        if filepath.suffix == ".jsonl":
            self._load_jsonl(filepath, existing_graph)
        elif filepath.suffix == ".json":
            self._load_json(filepath, existing_graph)
        else:
            raise ValueError(
                f"Unsupported file format: {filepath.suffix}. "
                "Use .jsonl or .json"
            )

        return existing_graph

    def _load_jsonl(
        self,
        filepath: Path,
        graph: HeteroAcademicGraph,
    ) -> None:
        """Load data from a JSON Lines file.

        Args:
            filepath: Path to the .jsonl file.
            graph: Graph to populate.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._extract_relations_from_papers(record, graph)

    def _load_json(
        self,
        filepath: Path,
        graph: HeteroAcademicGraph,
    ) -> None:
        """Load data from a JSON array file.

        Args:
            filepath: Path to the .json file.
            graph: Graph to populate.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for record in data:
                self._extract_relations_from_papers(record, graph)
        elif isinstance(data, dict):
            self._extract_relations_from_papers(data, graph)
        else:
            raise ValueError("JSON file must contain an array or object.")

    def _extract_relations_from_papers(
        self,
        record: Dict[str, Any],
        graph: HeteroAcademicGraph,
    ) -> None:
        """Extract nodes and edges from a paper record.

        Creates paper, author, journal, and institution nodes, along with
        citation, collaboration, affiliation, and publishes_in edges.

        Args:
            record: Dictionary containing paper data.
            graph: Graph to populate.
        """
        # Extract paper node
        paper_id = record.get("id") or record.get("paper_id", "")
        if not paper_id:
            return

        paper_attrs = {
            "title": record.get("title", ""),
            "abstract": record.get("abstract", ""),
            "keywords": record.get("keywords", []),
            "year": record.get("year"),
            "venue": record.get("venue", ""),
        }
        paper_node = Node(
            node_id=str(paper_id),
            node_type=NodeType.PAPER,
            attributes=paper_attrs,
        )
        graph.add_node(paper_node)

        # Collect paper_ids to avoid dict modification during iteration
        paper_ids = [str(paper_id)]

        # Extract author nodes and edges
        authors = record.get("authors", [])
        for author_info in authors:
            if isinstance(author_info, dict):
                author_id = str(author_info.get("id", ""))
                author_name = author_info.get("name", "")
            else:
                author_id = str(author_info)
                author_name = str(author_info)

            if not author_id:
                continue

            author_node = Node(
                node_id=author_id,
                node_type=NodeType.AUTHOR,
                attributes={"name": author_name},
            )
            graph.add_node(author_node)

            # Author writes paper edge (works_on)
            graph.add_edge(Edge(
                source_id=author_id,
                target_id=str(paper_id),
                edge_type=EdgeType.WORKS_ON,
            ))

        # Extract collaboration edges between co-authors
        author_ids = []
        for author_info in authors:
            if isinstance(author_info, dict):
                aid = str(author_info.get("id", ""))
            else:
                aid = str(author_info)
            if aid:
                author_ids.append(aid)

        for i in range(len(author_ids)):
            for j in range(i + 1, len(author_ids)):
                graph.add_edge(Edge(
                    source_id=author_ids[i],
                    target_id=author_ids[j],
                    edge_type=EdgeType.COLLABORATION,
                ))

        # Extract journal/venue node
        venue = record.get("venue", "")
        journal_id = record.get("journal_id", "")
        if journal_id:
            journal_node = Node(
                node_id=str(journal_id),
                node_type=NodeType.JOURNAL,
                attributes={"name": venue or record.get("journal_name", "")},
            )
            graph.add_node(journal_node)
            graph.add_edge(Edge(
                source_id=str(paper_id),
                target_id=str(journal_id),
                edge_type=EdgeType.PUBLISHES_IN,
            ))

        # Extract institution nodes and affiliation edges
        institutions = record.get("institutions", [])
        for inst_info in institutions:
            if isinstance(inst_info, dict):
                inst_id = str(inst_info.get("id", ""))
                inst_name = inst_info.get("name", "")
            else:
                inst_id = str(inst_info)
                inst_name = str(inst_info)

            if not inst_id:
                continue

            inst_node = Node(
                node_id=inst_id,
                node_type=NodeType.INSTITUTION,
                attributes={"name": inst_name},
            )
            graph.add_node(inst_node)

            # Link authors to institution
            for aid in author_ids:
                graph.add_edge(Edge(
                    source_id=aid,
                    target_id=inst_id,
                    edge_type=EdgeType.AFFILIATION,
                ))

        # Extract citation edges
        references = record.get("references", [])
        for ref_id in references:
            ref_id_str = str(ref_id)
            graph.add_edge(Edge(
                source_id=str(paper_id),
                target_id=ref_id_str,
                edge_type=EdgeType.CITATION,
            ))

        # Extract discipline nodes
        fos = record.get("fos", []) or record.get("fields_of_study", [])
        for field_info in fos:
            if isinstance(field_info, dict):
                disc_name = field_info.get("name", "")
            else:
                disc_name = str(field_info)

            if not disc_name:
                continue

            disc_id = f"disc_{disc_name.lower().replace(' ', '_')}"
            disc_node = Node(
                node_id=disc_id,
                node_type=NodeType.DISCIPLINE,
                attributes={"name": disc_name},
            )
            graph.add_node(disc_node)
            graph.add_edge(Edge(
                source_id=str(paper_id),
                target_id=disc_id,
                edge_type=EdgeType.BELONGS_TO,
            ))

        # Extract project/funding nodes
        projects = record.get("projects", [])
        for proj_info in projects:
            if isinstance(proj_info, dict):
                proj_id = str(proj_info.get("id", ""))
                proj_name = proj_info.get("name", "")
            else:
                proj_id = str(proj_info)
                proj_name = str(proj_info)

            if not proj_id:
                continue

            proj_node = Node(
                node_id=proj_id,
                node_type=NodeType.PROJECT,
                attributes={"name": proj_name},
            )
            graph.add_node(proj_node)
            graph.add_edge(Edge(
                source_id=str(paper_id),
                target_id=proj_id,
                edge_type=EdgeType.FUNDED_BY,
            ))
