#!/usr/bin/env python3
"""
Build Index Script for Tensor Manifold GVF.

Usage:
    python scripts/build_index.py --config config.yaml --data oag_data.json
    python scripts/build_index.py --data examples/self_published_papers/papers.json \
        --citations examples/self_published_papers/citations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import UnifiedRetriever
from retrieval.result_formatter import ResultFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found; using defaults.", config_path)
        return {}

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed; cannot read %s", config_path)
            return {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_oag_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load OAG-format data.

    Supports:
    - Raw OAG JSON (list of paper records)
    - Simplified JSON format (list of dicts with id, title, abstract, etc.)
    - JSONL format (one JSON object per line)
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Detect format by file extension
    suffix = path.suffix.lower()
    is_jsonl = suffix == ".jsonl"

    if is_jsonl:
        # Load JSONL format: one JSON object per line
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON on line %d: %s", line_num, e)
    else:
        # Load JSON format (existing behavior)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # OAG format may wrap papers in a key
            for key in ("papers", "data", "records", "results"):
                if key in data:
                    data = data[key]
                    break

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of documents, got {type(data)}")

    # Normalize fields
    normalized = []
    for item in data:
        doc = {
            "id": str(item.get("id", item.get("paper_id", ""))),
            "title": str(item.get("title", item.get("name", ""))),
            "abstract": str(item.get("abstract", item.get("summary", ""))),
            "year": item.get("year", item.get("pub_year")),
            "authors": item.get("authors", item.get("author_names", [])),
            "venue": item.get("venue", item.get("journal", item.get("conference", ""))),
            "keywords": item.get("keywords", item.get("tags", item.get("concepts", []))),
        }
        if doc["id"]:
            normalized.append(doc)

    logger.info("Loaded %d documents from %s", len(normalized), data_path)
    return normalized


def load_citations(citations_path: str) -> List[Dict[str, Any]]:
    """Load citation relationships from JSON or JSONL file."""
    path = Path(citations_path)
    if not path.exists():
        logger.warning("Citations file %s not found; no relations loaded.", citations_path)
        return []

    # Detect format by file extension
    suffix = path.suffix.lower()
    is_jsonl = suffix == ".jsonl"

    if is_jsonl:
        # Load JSONL format: one JSON object per line
        citations = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    citations.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON on line %d: %s", line_num, e)
    else:
        # Load JSON format (existing behavior)
        with open(path, "r", encoding="utf-8") as f:
            citations = json.load(f)

    # Normalize to {source, target, type} format
    relations = []
    for cite in citations:
        relations.append({
            "source": str(cite.get("source", cite.get("citing", ""))),
            "target": str(cite.get("target", cite.get("cited", ""))),
            "type": cite.get("type", cite.get("relation", "cites")),
        })

    logger.info("Loaded %d citation relationships from %s", len(relations), citations_path)
    return relations


def demo_query(retriever: UnifiedRetriever, query: str = "graph neural network") -> None:
    """Run a demo query and print results."""
    print(f"\n{'='*60}")
    print(f"Demo Query: \"{query}\"")
    print(f"{'='*60}")

    results = retriever.search(query, top_k=5)
    print(ResultFormatter.to_markdown(results))

    print(f"\n{ResultFormatter.to_table(results, max_abstract=50)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval index for Tensor Manifold GVF"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to document data (OAG JSON or simplified JSON)",
    )
    parser.add_argument(
        "--citations",
        type=str,
        default=None,
        help="Path to citations JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the built retriever state (JSON)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="graph neural network",
        help="Demo query string",
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default=None,
        help="SBERT model name or path",
    )
    parser.add_argument(
        "--manifold-dim",
        type=int,
        default=64,
        help="Manifold embedding dimension",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="brute",
        choices=["faiss", "hnswlib", "brute"],
        help="Index backend type",
    )
    parser.add_argument(
        "--decomposer-type",
        type=str,
        default="cp",
        choices=["cp", "tucker"],
        help="Tensor decomposition method",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Decomposition rank",
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip demo query after building",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW: ef_construction parameter",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=128,
        help="HNSW: ef_search parameter",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=16,
        help="HNSW: M parameter (max connections per node)",
    )
    parser.add_argument(
        "--hnsw-space",
        type=str,
        default="ip",
        choices=["ip", "l2"],
        help="HNSW: distance metric",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config) if args.config else {}

    # Override config with CLI args (support both flat and nested structure)
    if args.sbert_model:
        config["sbert_model"] = args.sbert_model
        # Also update nested structure
        if "encoder" not in config:
            config["encoder"] = {}
        config["encoder"]["model_name"] = args.sbert_model
    if args.manifold_dim:
        config["manifold_dim"] = args.manifold_dim
        if "manifold" not in config:
            config["manifold"] = {}
        config["manifold"]["output_dim"] = args.manifold_dim
    if args.index_type:
        config["index_type"] = args.index_type
        if "index" not in config:
            config["index"] = {}
        config["index"]["type"] = args.index_type
    if args.decomposer_type:
        config["decomposer_type"] = args.decomposer_type
        if "decomposer" not in config:
            config["decomposer"] = {}
        config["decomposer"]["type"] = args.decomposer_type
    if args.rank:
        config["rank"] = args.rank
        if "decomposer" not in config:
            config["decomposer"] = {}
        config["decomposer"]["rank"] = args.rank
    if args.ef_construction:
        config["ef_construction"] = args.ef_construction
        if "index" not in config:
            config["index"] = {}
        if "hnswlib" not in config["index"]:
            config["index"]["hnswlib"] = {}
        config["index"]["hnswlib"]["ef_construction"] = args.ef_construction
    if args.ef_search:
        config["ef_search"] = args.ef_search
        if "index" not in config:
            config["index"] = {}
        if "hnswlib" not in config["index"]:
            config["index"]["hnswlib"] = {}
        config["index"]["hnswlib"]["ef_search"] = args.ef_search
    if args.M:
        config["M"] = args.M
        if "index" not in config:
            config["index"] = {}
        if "hnswlib" not in config["index"]:
            config["index"]["hnswlib"] = {}
        config["index"]["hnswlib"]["M"] = args.M
    if args.hnsw_space:
        config["hnsw_space"] = args.hnsw_space
        if "index" not in config:
            config["index"] = {}
        if "hnswlib" not in config["index"]:
            config["index"]["hnswlib"] = {}
        config["index"]["hnswlib"]["space"] = args.hnsw_space

    # Load data
    documents = load_oag_data(args.data)
    relations = load_citations(args.citations) if args.citations else []

    if not documents:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    # Initialize retriever
    logger.info("Initializing UnifiedRetriever with config: %s", config)
    retriever = UnifiedRetriever(config=config)

    # Build index
    logger.info("Building retrieval index ...")
    t_start = time.time()
    retriever.build(documents, relations=relations)
    t_build = time.time() - t_start

    print(f"\nBuild completed in {t_build:.2f} seconds")
    print(f"  Documents indexed: {len(retriever)}")
    print(f"  Manifold dim:      {retriever.manifold_dim}")
    print(f"  Index type:        {retriever.index_type}")
    print(f"  Decomposer:        {retriever.decomposer_type} (rank={retriever.rank})")

    # Save state
    if args.output:
        retriever.to_json(args.output)
        print(f"  Saved to:          {args.output}")

    # Demo query
    if not args.no_demo:
        demo_query(retriever, args.query)


if __name__ == "__main__":
    main()
