#!/usr/bin/env python3
"""
Incremental Update Script for Tensor Manifold GVF.

Demonstrates Algorithm 3 (Incremental Manifold Update) by:
  1. Loading an existing index
  2. Adding new papers via a buffer
  3. Performing incremental update
  4. Verifying the updated index

Usage:
    python scripts/incremental_update.py \
        --index retriever_state.json \
        --new-data new_papers.json \
        --new-citations new_citations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import UnifiedRetriever
from retrieval.result_formatter import ResultFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("incremental_update")


def load_documents(path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("papers", "data", "records"):
            if key in data:
                data = data[key]
                break
    return data if isinstance(data, list) else [data]


def load_relations(path: str) -> List[Dict[str, Any]]:
    """Load citation relations from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        citations = json.load(f)
    relations = []
    for cite in citations:
        relations.append({
            "source": str(cite.get("source", cite.get("citing", ""))),
            "target": str(cite.get("target", cite.get("cited", ""))),
            "type": cite.get("type", cite.get("relation", "cites")),
        })
    return relations


def verify_update(
    retriever: UnifiedRetriever,
    new_doc_ids: List[str],
    query: str = "graph neural network",
) -> Dict[str, Any]:
    """
    Verify the incremental update by checking:
    1. New documents are present in the index
    2. Queries return results including new documents
    3. Decomposition works on new nodes
    """
    report: Dict[str, Any] = {"checks": []}

    # Check 1: New documents in index
    missing = []
    found = []
    for doc_id in new_doc_ids:
        if doc_id in retriever._id_to_idx:
            found.append(doc_id)
        else:
            missing.append(doc_id)
    report["checks"].append({
        "name": "new_docs_in_index",
        "passed": len(missing) == 0,
        "found": len(found),
        "missing": missing,
    })
    logger.info(
        "Check 1 - New docs in index: %d/%d found",
        len(found), len(found) + len(missing),
    )

    # Check 2: Query returns results
    results = retriever.search(query, top_k=10)
    result_ids = {r.id for r in results}
    new_in_results = result_ids & set(new_doc_ids)
    report["checks"].append({
        "name": "query_returns_results",
        "passed": len(results) > 0,
        "n_results": len(results),
        "new_docs_in_top10": list(new_in_results),
    })
    logger.info(
        "Check 2 - Query returns %d results, %d are new docs",
        len(results), len(new_in_results),
    )

    # Check 3: Decomposition on new nodes
    decomp_ok = 0
    for doc_id in new_doc_ids[:3]:  # Check first 3 new docs
        _, decomp = retriever.search_with_decomposition(doc_id, top_k=3)
        if decomp is not None:
            decomp_ok += 1
    report["checks"].append({
        "name": "decomposition_on_new_nodes",
        "passed": decomp_ok > 0,
        "decomp_success": decomp_ok,
        "decomp_tried": min(3, len(new_doc_ids)),
    })
    logger.info(
        "Check 3 - Decomposition: %d/%d new nodes decomposed successfully",
        decomp_ok, min(3, len(new_doc_ids)),
    )

    # Check 4: Index size
    report["checks"].append({
        "name": "index_size",
        "passed": len(retriever) > 0,
        "total_docs": len(retriever),
    })
    logger.info("Check 4 - Total documents in index: %d", len(retriever))

    report["all_passed"] = all(c["passed"] for c in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Incremental update demo for Tensor Manifold GVF"
    )
    parser.add_argument(
        "--index", "-i",
        type=str,
        required=True,
        help="Path to existing retriever state (JSON)",
    )
    parser.add_argument(
        "--new-data",
        type=str,
        required=True,
        help="Path to new papers JSON",
    )
    parser.add_argument(
        "--new-citations",
        type=str,
        default=None,
        help="Path to new citations JSON",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save updated retriever state",
    )
    parser.add_argument(
        "--verify-query",
        type=str,
        default="graph neural network",
        help="Query string for verification",
    )

    args = parser.parse_args()

    # Step 1: Load existing index
    print(f"\n{'='*60}")
    print("Step 1: Loading existing index")
    print(f"{'='*60}")
    logger.info("Loading retriever from %s ...", args.index)
    retriever = UnifiedRetriever.from_json(args.index)
    n_before = len(retriever)
    print(f"  Loaded: {n_before} documents")

    # Step 2: Load new papers
    print(f"\n{'='*60}")
    print("Step 2: Loading new papers")
    print(f"{'='*60}")
    new_docs = load_documents(args.new_data)
    new_rels = load_relations(args.new_citations) if args.new_citations else []
    new_doc_ids = [doc["id"] for doc in new_docs]
    print(f"  New documents: {len(new_docs)}")
    print(f"  New relations: {len(new_rels)}")
    print(f"  New doc IDs:   {new_doc_ids}")

    if not new_docs:
        print("No new documents to add. Exiting.")
        sys.exit(0)

    # Step 3: Incremental update
    print(f"\n{'='*60}")
    print("Step 3: Performing incremental update (Algorithm 3)")
    print(f"{'='*60}")
    t_start = time.time()
    stats = retriever.incremental_update(new_docs, relations=new_rels)
    t_update = time.time() - t_start
    print(f"  Update time:    {t_update:.4f} s")
    print(f"  Documents added: {stats['n_added']}")
    print(f"  Total documents: {stats['total']}")

    # Step 4: Verify
    print(f"\n{'='*60}")
    print("Step 4: Verifying update")
    print(f"{'='*60}")
    report = verify_update(retriever, new_doc_ids, args.verify_query)

    print(f"\nVerification Report:")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check}")

    all_passed = report["all_passed"]
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")

    # Demo query
    print(f"\n{'='*60}")
    print("Demo query after update")
    print(f"{'='*60}")
    results = retriever.search(args.verify_query, top_k=5)
    print(ResultFormatter.to_markdown(results))

    # Save updated state
    if args.output:
        retriever.to_json(args.output)
        print(f"\nUpdated retriever saved to: {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
