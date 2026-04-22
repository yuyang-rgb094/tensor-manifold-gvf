"""Mathematical utility functions for tensor manifold operations.

Provides vector similarity, normalization, and distance computations
used throughout the tensor manifold GVF pipeline.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector, shape (d,).
        b: Second vector, shape (d,).

    Returns:
        Cosine similarity score in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query and a batch of candidates.

    Args:
        query: Query vector, shape (d,).
        candidates: Candidate vectors, shape (n, d).

    Returns:
        numpy array of shape (n,) with similarity scores.
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.zeros(len(candidates))

    candidate_norms = np.linalg.norm(candidates, axis=1)
    # Avoid division by zero
    safe_norms = np.where(candidate_norms == 0.0, 1.0, candidate_norms)
    similarities = candidates @ query / (safe_norms * query_norm)
    # Set similarity to 0 for zero-norm candidates
    similarities[candidate_norms == 0.0] = 0.0
    return similarities


def normalize_to_manifold(
    vectors: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """Normalize vectors to lie on a hypersphere manifold.

    Projects each vector onto the surface of a hypersphere with the
    given radius.

    Args:
        vectors: Input vectors, shape (n, d) or (d,).
        radius: Radius of the target hypersphere.

    Returns:
        Normalized vectors with unit norm scaled by radius.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0.0:
            return np.zeros_like(vectors)
        return vectors / norm * radius

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms * radius


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        a: First vector, shape (d,).
        b: Second vector, shape (d,).

    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(a - b))


def batch_euclidean_distance(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances between a query and a batch of candidates.

    Args:
        query: Query vector, shape (d,).
        candidates: Candidate vectors, shape (n, d).

    Returns:
        numpy array of shape (n,) with distances.
    """
    diff = candidates - query[np.newaxis, :]
    return np.linalg.norm(diff, axis=1)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors to unit length.

    Args:
        vectors: Input vectors, shape (n, d) or (d,).

    Returns:
        L2-normalized vectors.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0.0:
            return np.zeros_like(vectors)
        return vectors / norm

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms
