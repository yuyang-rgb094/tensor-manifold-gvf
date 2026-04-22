#!/usr/bin/env python3
"""
Benchmark Script for Tensor Manifold GVF.

Measures:
  - Index build time and memory usage
  - Query latency (p50, p95, p99)
  - Incremental update performance
  - Decomposition overhead

Usage:
    python scripts/benchmark.py --data papers.json --citations citations.json
    python scripts/benchmark.py --data papers.json --iterations 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import UnifiedRetriever
from retrieval.result_formatter import ResultFormatter

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def load_documents(path: str) -> List[Dict[str, Any]]:
    """Load documents from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("papers", "data", "records"):
            if key in data:
                data = data[key]
                break
    return data if isinstance(data, list) else [data]


def load_relations(path: str) -> List[Dict[str, Any]]:
    """Load relations from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        citations = json.load(f)
    relations = []
    for cite in citations:
        relations.append({
            "source": str(cite.get("source", "")),
            "target": str(cite.get("target", "")),
            "type": cite.get("type", "cites"),
        })
    return relations


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_build(
    documents: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Benchmark index build time and memory."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Index Build")
    print("=" * 60)

    tracemalloc.start()
    t_start = time.perf_counter()

    retriever = UnifiedRetriever(config=config)
    retriever.build(documents, relations=relations)

    t_build = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "n_documents": len(documents),
        "n_relations": len(relations),
        "build_time_s": round(t_build, 4),
        "memory_current_mb": round(current_mem / 1024 / 1024, 2),
        "memory_peak_mb": round(peak_mem / 1024 / 1024, 2),
    }

    print(f"  Documents:      {result['n_documents']}")
    print(f"  Relations:      {result['n_relations']}")
    print(f"  Build time:     {result['build_time_s']:.4f} s")
    print(f"  Memory (curr):  {result['memory_current_mb']:.2f} MB")
    print(f"  Memory (peak):  {result['memory_peak_mb']:.2f} MB")

    return retriever, result


def benchmark_query(
    retriever: UnifiedRetriever,
    queries: List[str],
    iterations: int = 50,
) -> Dict[str, Any]:
    """Benchmark query latency."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Query Latency")
    print("=" * 60)

    latencies: List[float] = []
    top_k = 10

    # Warm-up
    for q in queries[:2]:
        retriever.search(q, top_k=top_k)

    for i in range(iterations):
        query = queries[i % len(queries)]
        t0 = time.perf_counter()
        retriever.search(query, top_k=top_k)
        latencies.append(time.perf_counter() - t0)

    result = {
        "iterations": iterations,
        "n_queries": len(queries),
        "top_k": top_k,
        "latency_mean_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "latency_p50_ms": round(percentile(latencies, 50) * 1000, 2),
        "latency_p95_ms": round(percentile(latencies, 95) * 1000, 2),
        "latency_p99_ms": round(percentile(latencies, 99) * 1000, 2),
        "latency_min_ms": round(min(latencies) * 1000, 2),
        "latency_max_ms": round(max(latencies) * 1000, 2),
        "qps": round(iterations / sum(latencies), 2),
    }

    print(f"  Iterations:     {result['iterations']}")
    print(f"  Queries:        {result['n_queries']}")
    print(f"  Top-K:          {result['top_k']}")
    print(f"  Mean latency:   {result['latency_mean_ms']:.2f} ms")
    print(f"  P50 latency:    {result['latency_p50_ms']:.2f} ms")
    print(f"  P95 latency:    {result['latency_p95_ms']:.2f} ms")
    print(f"  P99 latency:    {result['latency_p99_ms']:.2f} ms")
    print(f"  Min latency:    {result['latency_min_ms']:.2f} ms")
    print(f"  Max latency:    {result['latency_max_ms']:.2f} ms")
    print(f"  Throughput:     {result['qps']:.2f} QPS")

    return result


def benchmark_decomposition(
    retriever: UnifiedRetriever,
    n_nodes: int = 5,
) -> Dict[str, Any]:
    """Benchmark decomposition overhead."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Decomposition")
    print("=" * 60)

    doc_ids = list(retriever._id_to_idx.keys())[:n_nodes]
    latencies: List[float] = []

    for doc_id in doc_ids:
        t0 = time.perf_counter()
        _, decomp = retriever.search_with_decomposition(doc_id, top_k=5)
        latencies.append(time.perf_counter() - t0)

    result = {
        "n_nodes": len(doc_ids),
        "latency_mean_ms": round(sum(latencies) / len(latencies) * 1000, 2)
        if latencies else 0,
        "latency_max_ms": round(max(latencies) * 1000, 2) if latencies else 0,
    }

    print(f"  Nodes tested:   {result['n_nodes']}")
    print(f"  Mean latency:   {result['latency_mean_ms']:.2f} ms")
    print(f"  Max latency:    {result['latency_max_ms']:.2f} ms")

    return result


def benchmark_incremental(
    retriever: UnifiedRetriever,
    documents: List[Dict[str, Any]],
    n_batches: int = 3,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """Benchmark incremental update performance."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Incremental Update")
    print("=" * 60)

    batch_latencies: List[float] = []
    total_docs_added = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(documents))
        if start >= len(documents):
            break

        batch_docs = documents[start:end]
        t0 = time.perf_counter()
        stats = retriever.incremental_update(batch_docs)
        batch_latencies.append(time.perf_counter() - t0)
        total_docs_added += stats["n_added"]

    result = {
        "n_batches": len(batch_latencies),
        "total_docs_added": total_docs_added,
        "batch_size": batch_size,
        "latency_mean_ms": round(
            sum(batch_latencies) / len(batch_latencies) * 1000, 2
        )
        if batch_latencies else 0,
        "latency_per_doc_ms": round(
            sum(batch_latencies) / max(total_docs_added, 1) * 1000, 2
        ),
    }

    print(f"  Batches:        {result['n_batches']}")
    print(f"  Docs added:     {result['total_docs_added']}")
    print(f"  Batch size:     {result['batch_size']}")
    print(f"  Mean batch ms:  {result['latency_mean_ms']:.2f} ms")
    print(f"  Per-doc ms:     {result['latency_per_doc_ms']:.2f} ms")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Tensor Manifold GVF"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to document data JSON",
    )
    parser.add_argument(
        "--citations",
        type=str,
        default=None,
        help="Path to citations JSON",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=50,
        help="Number of query iterations",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save benchmark results JSON",
    )
    parser.add_argument(
        "--skip-incremental",
        action="store_true",
        help="Skip incremental update benchmark",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip decomposition benchmark",
    )

    args = parser.parse_args()

    # Load data
    documents = load_documents(args.data)
    relations = load_relations(args.citations) if args.citations else []

    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    config = {"index_type": "brute", "manifold_dim": 64}
    all_results: Dict[str, Any] = {
        "config": config,
        "data_source": args.data,
        "iterations": args.iterations,
    }

    # Benchmark 1: Build
    retriever, build_result = benchmark_build(documents, relations, config)
    all_results["build"] = build_result

    # Benchmark 2: Query
    queries = [
        "graph neural network",
        "tensor decomposition",
        "information retrieval",
        "knowledge graph embedding",
        "manifold learning",
    ]
    # Add some document titles as queries
    for doc in documents[:5]:
        queries.append(doc.get("title", ""))

    query_result = benchmark_query(retriever, queries, args.iterations)
    all_results["query"] = query_result

    # Benchmark 3: Decomposition
    if not args.skip_decomposition:
        decomp_result = benchmark_decomposition(retriever)
        all_results["decomposition"] = decomp_result

    # Benchmark 4: Incremental update
    if not args.skip_incremental and len(documents) > 10:
        # Use last portion of documents as "new" data
        split = len(documents) - min(15, len(documents) // 2)
        incremental_docs = documents[split:]
        # Rebuild with first portion only
        retriever2 = UnifiedRetriever(config=config)
        retriever2.build(documents[:split], relations=relations)
        inc_result = benchmark_incremental(retriever2, incremental_docs)
        all_results["incremental"] = inc_result

    # Summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Build time:       {all_results['build']['build_time_s']:.4f} s")
    print(f"  Peak memory:      {all_results['build']['memory_peak_mb']:.2f} MB")
    print(f"  Query P50:        {all_results['query']['latency_p50_ms']:.2f} ms")
    print(f"  Query P99:        {all_results['query']['latency_p99_ms']:.2f} ms")
    print(f"  Throughput:       {all_results['query']['qps']:.2f} QPS")
    if "decomposition" in all_results:
        print(f"  Decomp mean:      {all_results['decomposition']['latency_mean_ms']:.2f} ms")
    if "incremental" in all_results:
        print(f"  Incr per-doc:     {all_results['incremental']['latency_per_doc_ms']:.2f} ms")

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
