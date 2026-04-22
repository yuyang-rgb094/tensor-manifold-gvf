"""End-to-end tests for UnifiedRetriever."""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import UnifiedRetriever
from retrieval.result_formatter import ResultFormatter

EXAMPLES_DIR = PROJECT_ROOT / "examples" / "self_published_papers"

def _load_example_data():
    papers_path = EXAMPLES_DIR / "papers.json"
    citations_path = EXAMPLES_DIR / "citations.json"
    with open(papers_path) as f:
        papers = json.load(f)
    with open(citations_path) as f:
        citations = json.load(f)
    return papers, citations

def test_build_and_search():
    papers, citations = _load_example_data()
    retriever = UnifiedRetriever(config={"index_type": "brute", "manifold_mode": "truncate"})
    retriever.build(papers, relations=citations)

    assert len(retriever) == len(papers)
    assert retriever._built

    results = retriever.search("graph neural network", top_k=5)
    assert len(results) > 0
    assert results[0].rank == 1
    assert results[0].score > 0

def test_search_with_decomposition():
    papers, citations = _load_example_data()
    retriever = UnifiedRetriever(config={"index_type": "brute", "manifold_mode": "truncate"})
    retriever.build(papers, relations=citations)

    doc_id = papers[0]["id"]
    results, decomp = retriever.search_with_decomposition(doc_id, top_k=3)
    assert decomp is not None
    assert decomp.node_id == doc_id

def test_incremental_update():
    papers, citations = _load_example_data()
    retriever = UnifiedRetriever(config={"index_type": "brute", "manifold_mode": "truncate"})

    # Build with first half
    half = len(papers) // 2
    retriever.build(papers[:half], relations=citations)
    n_before = len(retriever)

    # Incremental update
    stats = retriever.incremental_update(papers[half:])
    assert stats["n_added"] == len(papers) - half
    assert len(retriever) == len(papers)

    # Verify search still works
    results = retriever.search("neural", top_k=3)
    assert len(results) > 0

def test_serialization():
    papers, citations = _load_example_data()
    retriever = UnifiedRetriever(config={"index_type": "brute", "manifold_mode": "truncate"})
    retriever.build(papers, relations=citations)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    retriever.to_json(path)

    retriever2 = UnifiedRetriever.from_json(path)
    assert len(retriever2) == len(papers)

    results = retriever2.search("graph", top_k=3)
    assert len(results) > 0

    Path(path).unlink()

def test_result_formatter():
    papers, citations = _load_example_data()
    retriever = UnifiedRetriever(config={"index_type": "brute", "manifold_mode": "truncate"})
    retriever.build(papers, relations=citations)
    results = retriever.search("graph", top_k=3)

    # Test all formats
    json_str = ResultFormatter.to_json(results)
    assert '"rank"' in json_str

    table_str = ResultFormatter.to_table(results)
    assert "Rank" in table_str

    md_str = ResultFormatter.to_markdown(results)
    assert "| Rank |" in md_str

    detailed_str = ResultFormatter.to_detailed(results)
    assert "Result #1" in detailed_str

if __name__ == "__main__":
    test_build_and_search()
    print("PASS: test_build_and_search")
    test_search_with_decomposition()
    print("PASS: test_search_with_decomposition")
    test_incremental_update()
    print("PASS: test_incremental_update")
    test_serialization()
    print("PASS: test_serialization")
    test_result_formatter()
    print("PASS: test_result_formatter")
    print("\nAll retriever tests passed!")
