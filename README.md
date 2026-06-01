# Tensor Manifold-Based Graph-Vector Fusion (TMGVF) v2.0

[English](#v20-features) | [中文](#v20-特性)

> 📄 **Paper**: [Tensor Manifold-Based Graph-Vector Fusion for AI-Native Academic Literature Retrieval](https://arxiv.org/abs/2604.16416) (arXiv:2604.16416)
>
> 📥 [PDF](paper/2604.16416.pdf) | 💻 [GitHub Repository](https://github.com/yuyang-rgb094/tensor-manifold-gvf)

---

<a id="v20-features"></a>

## 🔥 v2.0 — Major Architecture Upgrade (2026)

### What's New in v2.0

v2.0 is a complete architecture redesign targeting **AI Researchers, Builders, and Agent technology companies**, addressing three core problems of v1:

| Problem | v1 | v2.0 |
|---------|-----|------|
| **Semantic Dilution** | Abstract dominates embeddings (1/2~2/3) | Four independent channels, cross-attention fusion |
| **Fixed Channel Weights** | Single preset for all tasks | Task-specific attention heads with learned weights |
| **Memory-only Index** | FAISS/HNSWLIB in-memory | Qdrant named vectors with persistent storage |

### v2.0 Architecture

```
Four-Channel Encoder
├── Semantic Channel   → BGE-M3 (1024d, multilingual) or SBERT fallback
├── Metadata Channel   → Author EmbeddingBag + Keyword BGE-M3 mean
├── Topology Channel   → GraphSAGE (torch_geometric) or networkx fallback
└── Temporal Channel  → Time2Vec (learnable sin + linear)

        ↓ Cross-Modal Attention ↓

Task-Specific Attention Head (4 presets)
├── semantic_retrieval   (sem:0.40, meta:0.20, topo:0.25, temp:0.15)
├── citation_analysis    (sem:0.20, meta:0.15, topo:0.45, temp:0.20)
├── author_disambiguation(sem:0.25, meta:0.40, topo:0.15, temp:0.20)
└── trend_analysis      (sem:0.25, meta:0.20, topo:0.20, temp:0.35)

        ↓ Learned Channel Weights ↓

Qdrant Named Vectors (persistent, multi-channel search)
├── semantic:  1024d Cosine  ← BGE-M3
├── metadata:   256d Cosine   ← EmbeddingBag
├── topology:    32d Cosine   ← GraphSAGE
└── temporal:    32d Cosine   ← Time2Vec
```

### v2.0 Quick Start

```bash
# Install
pip install -e .

# v1 backward-compatible mode (SBERT + FAISS, default)
python scripts/build_index.py --data papers.json --citations citations.json

# v2 four-channel mode (requires channels_config)
python scripts/build_index.py --config config/channels.yaml --data papers.json
```

### v2.0 Python API

```python
from retrieval import UnifiedRetriever

# v1 mode (backward compatible)
retriever = UnifiedRetriever(manifold_dim=64, index_type="faiss")
retriever.build(documents, relations=citations)
results = retriever.search("graph neural network", top_k=10)

# v2 four-channel mode
retriever = UnifiedRetriever(
    manifold_dim=64,
    index_type="brute",
    channels_config={
        "enabled": True,
        "semantic": {"model": "BGE-M3"},
        "metadata": {"output_dim": 256},
        "topology": {"input_dim": 1024, "hidden_dim": 64},
        "temporal": {"output_dim": 32},
        "fusion": {"hidden_dim": 128},
        "qdrant": {"enabled": True, "host": "localhost", "port": 6333},
    },
    task_name="citation_analysis",  # Use task-specific attention head
)
retriever.build(documents, relations=citations, graph=hetero_graph)
results = retriever.search("transformer architecture", top_k=10)
```

### v2.0 REST API

```bash
# Start the Knowledge API service
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Semantic search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning", "top_k": 5, "task": "semantic_retrieval"}'

# Citation network analysis
curl -X POST http://localhost:8000/api/v1/citation/network \
  -d '{"paper_id": "p1", "depth": 2, "direction": "both"}'
```

API docs: http://localhost:8000/docs

### v2.0 Dependencies

```
# Core (required)
numpy>=1.20.0
torch>=2.1.0
sentence-transformers>=2.2.2

# Four-channel encoders (optional, auto-fallback if missing)
FlagEmbedding>=1.2.0      # BGE-M3 semantic encoder
torch_geometric>=2.5.0   # GraphSAGE topology encoder

# Vector store (optional)
qdrant-client>=1.7.0     # Qdrant named vector storage

# Knowledge API (optional)
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
```

### v2.0 Project Structure

```
tensor_manifold_gvf/
├── api/                        # FastAPI Knowledge API (ADR-0005)
│   ├── main.py                 # Application + 6 endpoints
│   ├── models.py               # Pydantic request/response models
│   └── router.py               # TaskRouter
├── core/
│   ├── four_channel_encoder.py # Cross-modal attention fusion
│   ├── task_attention.py       # Task-specific attention heads
│   ├── channel_weight_learner.py # Learned channel weights
│   └── visualization.py        # Attention weight visualization
├── retrieval/
│   ├── encoders/
│   │   ├── channels/           # Four channel encoders
│   │   │   ├── semantic_channel.py   # BGE-M3 / SBERT
│   │   │   ├── metadata_channel.py   # EmbeddingBag
│   │   │   ├── topology_channel.py   # GraphSAGE / networkx
│   │   │   └── temporal_channel.py   # Time2Vec
│   │   └── qdrant_store.py     # Qdrant named vector adapter
│   └── retriever.py            # UnifiedRetriever (v1 + v2)
├── docs/
│   ├── adr/                   # Architecture Decision Records
│   │   ├── 0001-four-channel-architecture.md
│   │   ├── 0002-cross-modal-attention-fusion.md
│   │   ├── 0003-qdrant-vector-store.md
│   │   ├── 0004-embedding-agnostic-design.md
│   │   ├── 0005-knowledge-api-service.md
│   │   └── 0006-task-specific-attention-heads.md
│   └── ROADMAP.md             # Future development roadmap
├── tests/
│   ├── test_channels.py        # Channel encoder tests
│   ├── test_cross_modal_attention.py
│   ├── test_task_attention.py
│   ├── test_phase3.py          # Phase 3 TDD (task heads, learner, viz)
│   ├── test_phase4.py          # Phase 4 TDD (Qdrant integration)
│   └── test_phase5.py          # Phase 5 TDD (FastAPI API)
└── docker-compose.yml          # Qdrant local deployment
```

### Test Coverage

```bash
pytest tests/ -v
# 141 passed ✅
```

---

*For v1 (legacy) architecture, see the [`v1` branch](https://github.com/yuyang-rgb094/tensor-manifold-gvf/tree/v1).*

---

<a id="original-paper"></a>

---

## Original Paper Description

> 📄 **Paper**: [Tensor Manifold-Based Graph-Vector Fusion for AI-Native Academic Literature Retrieval](https://arxiv.org/abs/2604.16416) (arXiv:2604.16416)

This is a Python implementation of the **Tensor Manifold-Based Graph-Vector Fusion** framework for AI-native academic literature retrieval, based on the theoretical foundation that academic literature graphs are discrete projections of tensor manifolds. This project unifies text vectors and graph topology into the same geometric space, supporting heterogeneous node types (papers, authors, journals, institutions, disciplines, projects) and multiple edge types (citations, collaborations, affiliations).

### Core Algorithms

- **Tensor Signature Construction**: Builds third-order tensors from entity-relation triples
- **Grassmann Vector Field Retrieval**: Projects queries onto Grassmann manifold for geometric search
- **Incremental Manifold Update**: Updates index with new documents without full rebuild
- **Tensor Decomposition**: CP and Tucker decomposition for multi-aspect analysis

### License

MIT License — see [LICENSE](LICENSE)

---

<a id="中文"></a>

## 张量流形图向量融合 (TMGVF) v2.0

> 📄 **论文**: [arXiv:2604.16416](https://arxiv.org/abs/2604.16416)

### v2.0 主要升级

| 问题 | v1 | v2.0 |
|------|-----|------|
| **语义稀释** | Abstract 占向量 1/2~2/3 | 四通道独立编码 + 交叉注意力融合 |
| **固定通道权重** | 所有任务用同一套权重 | 任务特定注意力头 + 可学习权重 |
| **内存索引** | FAISS/HNSWLIB 仅内存 | Qdrant Named Vectors 持久化存储 |

### 快速开始

```bash
# 安装
pip install -e .

# 启动 Qdrant (可选)
docker compose up -d

# 启动 Knowledge API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档: http://localhost:8000/docs

### 依赖安装

```bash
# 基础依赖
pip install -e .

# 四通道编码器 (可选，有自动降级)
pip install "FlagEmbedding>=1.2.0" "torch_geometric>=2.5.0"

# 向量存储 (可选)
pip install "qdrant-client>=1.7.0"

# Knowledge API (可选)
pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0"
```

### 下一步开发

详见 [docs/ROADMAP.md](docs/ROADMAP.md) — 包含 P2 功能开发指引和社区贡献指南。

---

*License: MIT — Copyright (c) 2026 yuyang-rgb094*
