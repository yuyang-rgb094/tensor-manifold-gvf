"""
Core algorithm modules for Tensor Manifold GVF.

Contains the mathematical foundations:
- DiscreteExteriorCalculus (DEC operators)
- HodgeDecomposition
- DiffusionSignatureUpdater
- HierarchicalManifoldEncoder
- SimilarityDecomposer
"""
from .dec_operators import DiscreteExteriorCalculus
from .hodge_decomposition import HodgeDecomposition
from .diffusion_signature import DiffusionSignatureUpdater
from .manifold_encoder import HierarchicalManifoldEncoder
from .similarity_decomposition import SimilarityDecomposer, DecomposedSimilarity

__all__ = [
    "DiscreteExteriorCalculus",
    "HodgeDecomposition",
    "DiffusionSignatureUpdater",
    "HierarchicalManifoldEncoder",
    "SimilarityDecomposer",
    "DecomposedSimilarity",
]
