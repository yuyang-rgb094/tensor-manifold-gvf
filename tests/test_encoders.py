"""Tests for retrieval.encoders module."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.encoders import SentenceTransformerEncoder, create_encoder

def test_encoder_create():
    encoder = create_encoder({"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"})
    assert encoder is not None
    assert encoder.embedding_dim > 0

def test_encoder_encode_single():
    encoder = SentenceTransformerEncoder(use_cache=False)
    result = encoder.encode_single("test query")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == encoder.embedding_dim

def test_encoder_encode_batch():
    encoder = SentenceTransformerEncoder(use_cache=False)
    texts = ["hello world", "test query", "another text"]
    result = encoder.encode(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3
    assert result.shape[1] == encoder.embedding_dim

def test_encoder_encode_documents():
    encoder = SentenceTransformerEncoder(use_cache=False)
    docs = [
        {"id": "1", "title": "Test Paper", "abstract": "This is a test abstract."},
        {"id": "2", "title": "Another Paper", "abstract": "Another abstract."},
    ]
    result = encoder.encode_documents(docs)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2

def test_encoder_cache():
    encoder = SentenceTransformerEncoder(use_cache=True)
    text = "cache test"
    r1 = encoder.encode_single(text)
    r2 = encoder.encode_single(text)
    np.testing.assert_array_equal(r1, r2)

if __name__ == "__main__":
    test_encoder_create()
    print("PASS: test_encoder_create")
    test_encoder_encode_single()
    print("PASS: test_encoder_encode_single")
    test_encoder_encode_batch()
    print("PASS: test_encoder_encode_batch")
    test_encoder_encode_documents()
    print("PASS: test_encoder_encode_documents")
    test_encoder_cache()
    print("PASS: test_encoder_cache")
    print("\nAll encoder tests passed!")
