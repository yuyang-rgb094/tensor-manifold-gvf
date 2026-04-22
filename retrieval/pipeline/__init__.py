"""
Tensor Manifold GVF - Retrieval Pipeline Module

Provides manifold projection and retrieval pipeline components for
projecting semantic embeddings onto tensor manifold spaces.
"""

from .manifold_projector import ManifoldProjector
from .signature_builder import SignatureBuilder
from .decomposer import TensorDecomposer, DecompositionResult

__all__ = [
    "ManifoldProjector",
    "SignatureBuilder",
    "TensorDecomposer",
    "DecompositionResult",
]
