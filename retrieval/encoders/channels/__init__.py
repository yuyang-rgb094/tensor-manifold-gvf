"""Four-channel encoder package.

Provides the abstract base class, concrete channel implementations,
and a factory function for creating channel encoders from configuration.

Channel encoders:
- :class:`SemanticChannelEncoder` — BGE-M3 text embedding (1024-dim)
- :class:`MetadataChannelEncoder` — entity embedding (256-dim)
- :class:`TopologyChannelEncoder` — GraphSAGE graph embedding (256-dim)
- :class:`TemporalChannelEncoder` — Time2Vec temporal encoding (64-dim)
"""

from .base import ChannelEncoder
from .semantic_channel import SemanticChannelEncoder
from .metadata_channel import MetadataChannelEncoder
from .topology_channel import TopologyChannelEncoder
from .temporal_channel import TemporalChannelEncoder

__all__ = [
    "ChannelEncoder",
    "SemanticChannelEncoder",
    "MetadataChannelEncoder",
    "TopologyChannelEncoder",
    "TemporalChannelEncoder",
    "create_channel_encoder",
]

# ------------------------------------------------------------------
# Channel registry
# ------------------------------------------------------------------

_CHANNEL_REGISTRY = {
    "semantic": SemanticChannelEncoder,
    "metadata": MetadataChannelEncoder,
    "topology": TopologyChannelEncoder,
    "temporal": TemporalChannelEncoder,
}


def create_channel_encoder(
    channel_name: str,
    config: dict,
) -> ChannelEncoder:
    """Factory function: create a channel encoder from configuration.

    Parameters
    ----------
    channel_name : str
        One of ``"semantic"``, ``"metadata"``, ``"topology"``, ``"temporal"``.
    config : dict
        Configuration dict with keyword arguments for the encoder constructor.
        The ``"type"`` key (if present) is consumed and not passed through.

    Returns
    -------
    ChannelEncoder
        An initialized channel encoder instance.

    Raises
    ------
    ValueError
        If *channel_name* is not recognized.
    """
    if channel_name not in _CHANNEL_REGISTRY:
        raise ValueError(
            f"Unknown channel '{channel_name}'. "
            f"Available: {list(_CHANNEL_REGISTRY.keys())}"
        )

    cls = _CHANNEL_REGISTRY[channel_name]
    # Remove 'type' key if present (it's the channel name, not a constructor arg)
    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls(**kwargs)
