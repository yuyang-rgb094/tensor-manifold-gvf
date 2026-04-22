"""Tests for retrieval.pipeline module."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.pipeline import ManifoldProjector, SignatureBuilder, TensorDecomposer

def test_signature_builder():
    builder = SignatureBuilder(embedding_dim=384)
    docs = [
        {"id": "1", "title": "Paper A", "abstract": "Abstract A", "authors": ["Alice"], "keywords": ["ml"]},
        {"id": "2", "title": "Paper B", "abstract": "Abstract B", "authors": ["Bob"], "venue": "CVPR"},
    ]
    relations = [
        {"source": "1", "target": "2", "type": "cites"},
    ]
    rel_types = builder.get_relation_types(relations)
    assert rel_types == ["cites"]

    sigs = builder.build(docs, relations)
    assert len(sigs) == 2
    assert sigs[0]["doc_id"] == "1"
    assert sigs[0]["entity_count"] == 2  # Alice + ml
    assert sigs[1]["entity_count"] == 2  # Bob + CVPR (venue)

def test_manifold_projector_truncate():
    proj = ManifoldProjector(mode="truncate", semantic_dim=384, output_dim=64)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 384)).astype(np.float32)
    sigs = [{"entity_count": 1, "relation_count": 1, "slices": []} for _ in range(5)]

    result = proj.project(emb, sigs)
    assert result.shape == (5, 64)
    # Check L2 normalization
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

def test_manifold_projector_single():
    proj = ManifoldProjector(mode="truncate", semantic_dim=384, output_dim=64)
    rng = np.random.default_rng(42)
    query = rng.standard_normal(384).astype(np.float32)

    result = proj.project_single(query)
    assert result.shape == (64,)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6

def test_manifold_projector_update():
    proj = ManifoldProjector(mode="truncate", semantic_dim=384, output_dim=64)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 384)).astype(np.float32)
    sigs = [{"entity_count": 1, "relation_count": 1, "slices": []} for _ in range(5)]

    result1 = proj.project(emb, sigs)
    proj.update(result1)
    # After update, internal state should be modified
    assert proj._manifold_embeddings is not None

def test_decomposer():
    decomp = TensorDecomposer(decomposer_type="cp", rank=4)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((5, 64)).astype(np.float32)
    sigs = [{"doc_id": str(i), "entities": [], "relations": [], "shape": (0,1,64), "slices": [], "entity_count": 0, "relation_count": 1} for i in range(5)]

    decomp.init(sigs, emb, ["cites"])
    result = decomp.decompose_node(0)
    assert result is not None
    assert result.node_id == "0"
    assert result.explained_variance_ratio > 0
    assert len(result.aspect_contributions) > 0

if __name__ == "__main__":
    test_signature_builder()
    print("PASS: test_signature_builder")
    test_manifold_projector_truncate()
    print("PASS: test_manifold_projector_truncate")
    test_manifold_projector_single()
    print("PASS: test_manifold_projector_single")
    test_manifold_projector_update()
    print("PASS: test_manifold_projector_update")
    test_decomposer()
    print("PASS: test_decomposer")
    print("\nAll pipeline tests passed!")
