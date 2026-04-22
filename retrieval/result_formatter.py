"""
Result Formatter for Tensor Manifold GVF.

Provides static methods to format RetrievalResult lists into
JSON, plain-text tables, Markdown, and detailed reports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .retriever import RetrievalResult


class ResultFormatter:
    """Format retrieval results into various output formats."""

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(
        results: List[RetrievalResult],
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """
        Serialize results to a JSON string.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieval results to format.
        indent : int
            JSON indentation level.
        ensure_ascii : bool
            Whether to escape non-ASCII characters.

        Returns
        -------
        str
            JSON-formatted string.
        """
        data = [r.to_dict() for r in results]
        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

    # ------------------------------------------------------------------
    # Plain-text table
    # ------------------------------------------------------------------

    @staticmethod
    def to_table(
        results: List[RetrievalResult],
        max_abstract: int = 60,
        show_metadata: bool = True,
    ) -> str:
        """
        Format results as a plain-text aligned table.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieval results to format.
        max_abstract : int
            Maximum characters of abstract to display.
        show_metadata : bool
            Whether to include year/venue columns.

        Returns
        -------
        str
            Plain-text table string.
        """
        if not results:
            return "(no results)"

        # Column widths
        rank_w = 5
        score_w = 8
        id_w = max(len(r.id) for r in results) + 2
        title_w = max(len(r.title) for r in results) + 2
        abs_w = min(max_abstract, max(len(r.abstract) for r in results) + 2)
        meta_w = 30 if show_metadata else 0

        # Header
        header_parts = [
            f"{'Rank':<{rank_w}}",
            f"{'Score':<{score_w}}",
            f"{'ID':<{id_w}}",
            f"{'Title':<{title_w}}",
            f"{'Abstract':<{abs_w}}",
        ]
        if show_metadata:
            header_parts.append(f"{'Year/Venue':<{meta_w}}")
        header = " | ".join(header_parts)

        sep = "-+-".join("-" * len(p) for p in header.split(" | "))

        lines = [header, sep]
        for r in results:
            abstract_short = (
                r.abstract[: max_abstract - 3] + "..."
                if len(r.abstract) > max_abstract
                else r.abstract
            )
            row_parts = [
                f"{r.rank:<{rank_w}}",
                f"{r.score:<{score_w}.4f}",
                f"{r.id:<{id_w}}",
                f"{r.title:<{title_w}}",
                f"{abstract_short:<{abs_w}}",
            ]
            if show_metadata:
                year = r.metadata.get("year", "")
                venue = r.metadata.get("venue", "")
                meta_str = f"{year} | {venue}" if year or venue else ""
                row_parts.append(f"{meta_str:<{meta_w}}")
            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(
        results: List[RetrievalResult],
        max_abstract: int = 120,
    ) -> str:
        """
        Format results as a Markdown table.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieval results to format.
        max_abstract : int
            Maximum characters of abstract to display.

        Returns
        -------
        str
            Markdown table string.
        """
        if not results:
            return "*No results found.*"

        lines = [
            "| Rank | Score | ID | Title | Abstract | Year | Venue |",
            "|------|-------|----|-------|----------|------|-------|",
        ]
        for r in results:
            abstract_short = (
                r.abstract[: max_abstract - 3] + "..."
                if len(r.abstract) > max_abstract
                else r.abstract
            )
            # Escape pipe characters in text
            abstract_short = abstract_short.replace("|", "\\|")
            title = r.title.replace("|", "\\|")
            year = r.metadata.get("year", "")
            venue = r.metadata.get("venue", "").replace("|", "\\|")
            lines.append(
                f"| {r.rank} | {r.score:.4f} | {r.id} | {title} "
                f"| {abstract_short} | {year} | {venue} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Detailed report
    # ------------------------------------------------------------------

    @staticmethod
    def to_detailed(
        results: List[RetrievalResult],
        show_decomposition: bool = True,
        show_related: bool = True,
    ) -> str:
        """
        Format results as a detailed human-readable report.

        Each result is rendered as a block with full metadata,
        optional decomposition info, and related nodes.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieval results to format.
        show_decomposition : bool
            Whether to include tensor decomposition details.
        show_related : bool
            Whether to include related node information.

        Returns
        -------
        str
            Detailed report string.
        """
        if not results:
            return "No results found."

        blocks: List[str] = []
        for r in results:
            lines = [
                f"--- Result #{r.rank}  (score: {r.score:.4f}) ---",
                f"  ID:      {r.id}",
                f"  Title:   {r.title}",
                f"  Abstract: {r.abstract}",
            ]

            # Metadata
            meta = r.metadata
            if meta.get("authors"):
                authors = ", ".join(str(a) for a in meta["authors"])
                lines.append(f"  Authors: {authors}")
            if meta.get("year"):
                lines.append(f"  Year:    {meta['year']}")
            if meta.get("venue"):
                lines.append(f"  Venue:   {meta['venue']}")
            if meta.get("keywords"):
                keywords = ", ".join(str(k) for k in meta["keywords"])
                lines.append(f"  Keywords: {keywords}")

            # Decomposition
            if show_decomposition and r.decomposition:
                decomp = r.decomposition
                lines.append("  [Decomposition]")
                lines.append(
                    f"    Explained variance: "
                    f"{decomp.get('explained_variance_ratio', 0):.4f}"
                )
                lines.append(
                    f"    Reconstruction error: "
                    f"{decomp.get('reconstruction_error', 0):.6f}"
                )
                if r.aspect_scores:
                    lines.append("    Aspect contributions:")
                    for aspect, weight in r.aspect_scores.items():
                        lines.append(f"      - {aspect}: {weight:.4f}")

            # Related nodes
            if show_related and r.related_nodes:
                lines.append("  [Related Nodes]")
                for node in r.related_nodes:
                    lines.append(
                        f"    - {node.get('id', '?')}: "
                        f"{node.get('title', '?')} "
                        f"({node.get('relation_type', '?')})"
                    )

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)
