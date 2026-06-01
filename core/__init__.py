"""
Core algorithm modules for Tensor Manifold GVF.

Contains the mathematical foundations:
- DiscreteExteriorCalculus (DEC operators)
- HodgeDecomposition
- DiffusionSignatureUpdater
- HierarchicalManifoldEncoder
- FourChannelFusionEncoder (v2)
- TaskSpecificAttentionHead (v2 phase 2)
- SimilarityDecomposer
"""
from .dec_operators import DiscreteExteriorCalculus
from .hodge_decomposition import HodgeDecomposition
from .diffusion_signature import DiffusionSignatureUpdater
from .manifold_encoder import HierarchicalManifoldEncoder
from .four_channel_encoder import FourChannelFusionEncoder
from .task_attention import (
    TaskSpecificAttentionHead,
    create_task_head,
    compute_task_sensitivity_report,
    TASK_REGISTRY,
)
from .similarity_decomposition import SimilarityDecomposer, DecomposedSimilarity

__all__ = [
    "DiscreteExteriorCalculus",
    "HodgeDecomposition",
    "DiffusionSignatureUpdater",
    "HierarchicalManifoldEncoder",
    "FourChannelFusionEncoder",
    "TaskSpecificAttentionHead",
    "create_task_head",
    "compute_task_sensitivity_report",
    "TASK_REGISTRY",
    "SimilarityDecomposer",
    "DecomposedSimilarity",
]
