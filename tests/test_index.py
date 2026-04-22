"""Tests for retrieval.index module."""
import sys
from pathlib import Path
import numpy as np
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.index import BruteForceIndex, create_index

def _make_embeddings(n=100, d=64):
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)

def test_brute_build_search():
    idx = BruteForceIndex()
    emb = _make_embeddings()
    idx.build(emb)
    assert idx.n_total == 100

    query = emb[0:1]
    scores, indices = idx.search(query, top_k=5)
    assert indices[0][0] == 0  # self should be top-1
    assert scores.shape == (1, 5)

def test_brute_add():
    idx = BruteForceIndex()
    emb = _make_embeddings(50)
    idx.build(emb)
    new_emb = _make_embeddings(10)
    idx.add(new_emb)
    assert idx.n_total == 60

def test_brute_save_load():
    idx = BruteForceIndex()
    emb = _make_embeddings()
    idx.build(emb)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
    idx.save(path)

    idx2 = BruteForceIndex()
    idx2.load(path)
    assert idx2.n_total == 100

    query = emb[0:1]
    s1, i1 = idx.search(query, top_k=3)
    s2, i2 = idx2.search(query, top_k=3)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2)
    Path(path).unlink()

def test_create_index_factory():
    idx = create_index("brute")
    assert isinstance(idx, BruteForceIndex)
    idx = create_index("numpy")
    assert isinstance(idx, BruteForceIndex)

if __name__ == "__main__":
    test_brute_build_search()
    print("PASS: test_brute_build_search")
    test_brute_add()
    print("PASS: test_brute_add")
    test_brute_save_load()
    print("PASS: test_brute_save_load")
    test_create_index_factory()
    print("PASS: test_create_index_factory")
    print("\nAll index tests passed!")
