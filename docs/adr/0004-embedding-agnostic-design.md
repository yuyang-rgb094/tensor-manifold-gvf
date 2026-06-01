# ADR-0004: Embedding-Agnostic Design with BGE-M3 Default

## Status

Proposed

## Context

v1 hardcodes `all-MiniLM-L6-v2` (384-dim, English-focused). v2 targets AI companies that may have their own embedding infrastructure (OpenAI, Cohere, Jina, custom models).

Additionally, the user mentioned DeepSeek and OpenAI Embedding as candidates. Key facts:
- **DeepSeek does not have a dedicated embedding API product** as of 2025-2026. Using DeepSeek chat models for embedding via last hidden state is suboptimal.
- OpenAI `text-embedding-3-large` supports Matryoshka Representation Learning (MRL) for variable dimensions, which is interesting but locks into a single vendor.
- BGE-M3 (BAAI) is open-source, multilingual, supports dense+sparse+colbert retrieval, and is fine-tunable.

## Decision

**Design an embedding-agnostic encoder interface. Default to BGE-M3 for the semantic content channel.**

The encoder interface:
```python
class EmbeddingEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
```

Supported implementations:
1. **BGE-M3** (default): 1024-dim, multilingual, fine-tunable, free
2. **OpenAI text-embedding-3-large**: 3072-dim (or reduced via MRL), API-based
3. **Sentence-Transformers** (backward compat): any SBERT-compatible model
4. **Custom**: user implements the interface with their own model

Each channel can use a different encoder:
- Semantic Content Channel: BGE-M3 (default) or OpenAI
- Structured Metadata Channel: Entity embedding model (e.g., fastText or learned entity embeddings)
- Topology Channel: GNN (not an "embedding encoder" per se, but follows the same interface pattern)
- Temporal Channel: Fixed mathematical encoding (no learned encoder needed)

## Consequences

### Positive
- Users can swap embedding models without changing any downstream code
- BGE-M3's multilingual support enables non-English academic literature
- BGE-M3's dense+sparse hybrid retrieval can improve recall
- No vendor lock-in

### Negative
- BGE-M3 is significantly larger than MiniLM (568M vs 22M params), requiring more GPU memory
- Different embedding models produce vectors in different spaces — cross-model comparison is meaningless
- The topology and temporal channels don't use traditional embedding models, so the "agnostic" design is really only for the semantic channel

### Mitigations
- BGE-M3 inference is still fast on a single A100 (~5ms/query for 1024-dim)
- Document which encoder was used for each index build (stored in Qdrant collection metadata)
- The topology and temporal channels have their own encoder abstractions (GNN encoder, time encoder) that follow the same interface pattern
