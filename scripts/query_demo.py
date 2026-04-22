#!/usr/bin/env python3
"""
Interactive Query Demo for Tensor Manifold GVF.

Commands:
    text <query>   - Text-based retrieval
    node <id>      - Node-based retrieval with decomposition
    top [k]        - Show top-k results from last query
    quit / exit    - Exit

Usage:
    python scripts/query_demo.py --index retriever_state.json
    python scripts/query_demo.py --data papers.json --citations citations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import UnifiedRetriever
from retrieval.result_formatter import ResultFormatter

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("query_demo")


def build_from_data(data_path: str, citations_path: Optional[str] = None) -> UnifiedRetriever:
    """Build a retriever from data files on the fly."""
    from scripts.build_index import load_oag_data, load_citations

    documents = load_oag_data(data_path)
    relations = load_citations(citations_path) if citations_path else []

    retriever = UnifiedRetriever(config={"index_type": "brute"})
    retriever.build(documents, relations=relations)
    return retriever


def print_banner():
    """Print the interactive banner."""
    print()
    print("=" * 60)
    print("  Tensor Manifold GVF - Interactive Query Demo")
    print("=" * 60)
    print()
    print("Commands:")
    print("  text <query>       Text-based retrieval")
    print("  node <id>          Node retrieval with decomposition")
    print("  top [k]            Show top-k from last results (default 10)")
    print("  format <fmt>       Set output format: table / markdown / detailed")
    print("  save <path>        Save last results to JSON file")
    print("  stats              Show index statistics")
    print("  help               Show this help message")
    print("  quit / exit        Exit")
    print()


def cmd_text(retriever: UnifiedRetriever, query: str, fmt: str) -> List:
    """Execute a text query."""
    print(f"\nSearching: \"{query}\" ...")
    results = retriever.search(query, top_k=20)
    _print_results(results, fmt)
    return results


def cmd_node(retriever: UnifiedRetriever, node_id: str, fmt: str) -> tuple:
    """Execute a node query with decomposition."""
    print(f"\nDecomposing node: \"{node_id}\" ...")
    results, decomp = retriever.search_with_decomposition(node_id, top_k=10)

    if not results:
        print(f"  No results found for node '{node_id}'.")
        if node_id not in retriever._id_to_idx:
            print(f"  Node '{node_id}' does not exist in the index.")
            available = list(retriever._id_to_idx.keys())[:10]
            print(f"  Available nodes: {available}")
        return results, decomp

    if decomp:
        print(f"\n  Decomposition (type={retriever.decomposer_type}, "
              f"rank={retriever.rank}):")
        print(f"    Explained variance: {decomp.explained_variance_ratio:.4f}")
        print(f"    Reconstruction error: {decomp.reconstruction_error:.6f}")
        print(f"    Aspect contributions:")
        for aspect, weight in decomp.aspect_contributions.items():
            bar = "#" * int(weight * 200)
            print(f"      {aspect:<20s} {weight:.4f}  {bar}")

    print(f"\n  Related nodes:")
    _print_results(results, fmt)
    return results, decomp


def cmd_top(last_results: List, k: int, fmt: str) -> None:
    """Show top-k results from last query."""
    if not last_results:
        print("No previous results. Run a query first.")
        return
    _print_results(last_results[:k], fmt)


def cmd_stats(retriever: UnifiedRetriever) -> None:
    """Show index statistics."""
    print(f"\nIndex Statistics:")
    print(f"  Total documents:   {len(retriever)}")
    print(f"  SBERT model:       {retriever.sbert_model}")
    print(f"  Embedding dim:     {retriever.embedding_dim}")
    print(f"  Manifold dim:      {retriever.manifold_dim}")
    print(f"  Index type:        {retriever.index_type}")
    print(f"  Decomposer:        {retriever.decomposer_type} (rank={retriever.rank})")
    print(f"  Relation types:    {retriever._relation_types}")
    print(f"  Built:             {retriever._built}")


def _print_results(results: list, fmt: str) -> None:
    """Print results in the specified format."""
    if not results:
        print("  (no results)")
        return

    if fmt == "markdown":
        print(ResultFormatter.to_markdown(results))
    elif fmt == "detailed":
        print(ResultFormatter.to_detailed(results))
    else:
        print(ResultFormatter.to_table(results))


def interactive_loop(retriever: UnifiedRetriever):
    """Run the interactive REPL."""
    print_banner()
    cmd_stats(retriever)

    last_results = []
    current_fmt = "table"

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        elif cmd == "help":
            print_banner()

        elif cmd == "text" and arg:
            last_results = cmd_text(retriever, arg, current_fmt)

        elif cmd == "node" and arg:
            last_results, _ = cmd_node(retriever, arg, current_fmt)

        elif cmd == "top":
            k = int(arg) if arg.isdigit() else 10
            cmd_top(last_results, k, current_fmt)

        elif cmd == "format" and arg:
            if arg in ("table", "markdown", "detailed"):
                current_fmt = arg
                print(f"Output format set to: {current_fmt}")
            else:
                print(f"Unknown format '{arg}'. Use: table, markdown, detailed")

        elif cmd == "save" and arg:
            if last_results:
                json_str = ResultFormatter.to_json(last_results)
                Path(arg).write_text(json_str, encoding="utf-8")
                print(f"Results saved to {arg}")
            else:
                print("No results to save. Run a query first.")

        elif cmd == "stats":
            cmd_stats(retriever)

        elif cmd == "text" and not arg:
            print("Usage: text <query>")
        elif cmd == "node" and not arg:
            print("Usage: node <id>")
        elif cmd == "save" and not arg:
            print("Usage: save <path>")
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive query demo for Tensor Manifold GVF"
    )
    parser.add_argument(
        "--index", "-i",
        type=str,
        default=None,
        help="Path to saved retriever state (JSON)",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to document data (builds index on the fly)",
    )
    parser.add_argument(
        "--citations",
        type=str,
        default=None,
        help="Path to citations JSON",
    )

    args = parser.parse_args()

    # Load or build retriever
    if args.index:
        logger.info("Loading retriever from %s ...", args.index)
        retriever = UnifiedRetriever.from_json(args.index)
    elif args.data:
        logger.info("Building retriever from %s ...", args.data)
        retriever = build_from_data(args.data, args.citations)
    else:
        print("Error: Provide --index or --data.")
        sys.exit(1)

    # Run interactive loop
    interactive_loop(retriever)


if __name__ == "__main__":
    main()
