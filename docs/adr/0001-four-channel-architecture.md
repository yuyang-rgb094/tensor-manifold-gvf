# ADR-0001: Four-Channel Multi-Modal Architecture

## Status

Proposed

## Context

The v1 system encodes all document information (title + abstract) through a single SBERT model into a 384-dimensional vector, then fuses topology via gated residual connection. This causes two problems:

1. **Semantic dominance**: The abstract's semantic signal overwhelms citation topology in cosine similarity, causing citation-coupled papers to scatter in vector space.
2. **Fixed modality weighting**: All tasks (semantic retrieval, author disambiguation, school clustering, trend analysis) use the same vector with the same implicit modality weights, making the system brittle across diverse tasks.

## Decision

Adopt a **four-channel architecture** where each information source is encoded independently into its optimal embedding space:

| Channel | Information Sources | Encoding Method | Output Dimension |
|---------|-------------------|-----------------|-----------------|
| Semantic Content | title + abstract | LLM embedding (pluggable) | 768-1024 (model-dependent) |
| Structured Metadata | keywords + venue + authors | Entity/KG embedding | 128-256 |
| Topology | citation graph | GNN (e.g., GraphSAGE) | 128-256 |
| Temporal Signal | pub_time + update_time | RoPE-style time encoding | 16-64 |

Channels are fused via cross-modal attention (see ADR-0002).

## Consequences

### Positive
- Each channel operates in its optimal embedding space
- Citation topology is no longer subordinated to semantic content
- Task-specific attention heads can learn per-task channel weights
- Embedding model upgrades in one channel don't affect others

### Negative
- Increased system complexity: 4 encoding pipelines instead of 1
- Higher storage cost: multiple vectors per document
- Requires GNN infrastructure for the topology channel (currently absent)
- Training cross-modal attention requires task-specific labeled data

### Mitigations
- Use Qdrant's named vectors to store each channel independently
- Start with pre-trained GNN models (e.g., GraphSAGE on citation graph)
- Use self-supervised pre-training for cross-modal attention, fine-tune per task
