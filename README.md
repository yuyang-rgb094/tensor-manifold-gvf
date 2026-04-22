# Tensor Manifold-Based Graph-Vector Fusion / 张量流形图向量融合

[English](#english) | [中文](#中文)

> 📄 **Paper**: [Tensor Manifold-Based Graph-Vector Fusion for AI-Native Academic Literature Retrieval](https://arxiv.org/abs/2604.16416) (arXiv:2604.16416)
>
> 📥 [PDF](paper/2604.16416.pdf) | 💻 [GitHub Repository](https://github.com/yuyang-rgb094/tensor-manifold-gvf)

---

<a id="english"></a>

## Tensor Manifold-Based Graph-Vector Fusion (TMGVF)

A Python implementation of the **Tensor Manifold-Based Graph-Vector Fusion** framework for AI-native academic literature retrieval, based on the theoretical foundation that academic literature graphs are discrete projections of tensor manifolds. This project unifies text vectors and graph topology into the same geometric space, supporting heterogeneous node types (papers, authors, journals, institutions, disciplines, projects) and multiple edge types (citations, collaborations, affiliations).

### Features

- **Tensor Signature Construction (Algorithm 1)**: Builds third-order tensors from entity-relation triples in knowledge graphs
- **Grassmann Vector Field Retrieval (Algorithm 2)**: Projects queries onto the Grassmann manifold for geometrically-aware similarity search
- **Incremental Manifold Update (Algorithm 3)**: Efficiently updates the index with new documents without full rebuild
- **Tensor Decomposition**: CP and Tucker decomposition for multi-aspect retrieval and analysis
- **Multiple Index Backends**: FAISS, HNSWLIB, and brute-force similarity search
- **OAG Format Support**: Compatible with Open Academic Graph data format
- **Flexible Output**: JSON, plain-text tables, Markdown, and detailed reports

### Dependencies

```
numpy>=1.20.0
sentence-transformers>=2.0.0   # optional; random embeddings used as fallback
faiss-cpu>=1.7.0               # optional; brute-force used as fallback
hnswlib>=0.7.0                 # optional; brute-force used as fallback
pyyaml>=6.0                    # optional; for YAML config files
```

### Project Structure

```
tensor_manifold_gvf/
├── retrieval/
│   ├── __init__.py              # Module exports
│   ├── retriever.py             # UnifiedRetriever, RetrievalResult
│   └── result_formatter.py      # ResultFormatter (JSON/table/Markdown/detailed)
├── scripts/
│   ├── build_index.py           # Build retrieval index from data
│   ├── query_demo.py            # Interactive query REPL
│   ├── incremental_update.py    # Incremental update with verification
│   └── benchmark.py             # Performance benchmarking
├── tests/
│   └── __init__.py
├── examples/
│   └── self_published_papers/
│       ├── papers.json          # 10 sample academic papers
│       └── citations.json       # 19 citation relationships
└── README.md
```

### Quick Start

#### 1. Build Index

```bash
python scripts/build_index.py \
    --data examples/self_published_papers/papers.json \
    --citations examples/self_published_papers/citations.json \
    --output retriever_state.json
```

#### 2. Interactive Query

```bash
python scripts/query_demo.py --index retriever_state.json
```

Available commands in the interactive REPL:
- `text <query>` -- Text-based retrieval
- `node <id>` -- Node-based retrieval with full decomposition
- `top [k]` -- Show top-k results from last query
- `format <fmt>` -- Switch output format (table / markdown / detailed)
- `save <path>` -- Save results to JSON
- `stats` -- Show index statistics
- `quit` -- Exit

#### 3. Incremental Update

```bash
python scripts/incremental_update.py \
    --index retriever_state.json \
    --new-data new_papers.json \
    --new-citations new_citations.json \
    --output updated_retriever.json
```

#### 4. Benchmark

```bash
python scripts/benchmark.py \
    --data examples/self_published_papers/papers.json \
    --citations examples/self_published_papers/citations.json \
    --iterations 100 \
    --output benchmark_results.json
```

### Python API

```python
from retrieval import UnifiedRetriever, ResultFormatter

# Build retriever
retriever = UnifiedRetriever(
    sbert_model="all-MiniLM-L6-v2",
    manifold_dim=64,
    index_type="faiss",
    decomposer_type="cp",
    rank=8,
)

retriever.build(documents, relations=citations)

# Text search
results = retriever.search("graph neural network", top_k=10)
print(ResultFormatter.to_markdown(results))

# Node search with decomposition
results, decomp = retriever.search_with_decomposition("paper_001", top_k=5)
if decomp:
    print(f"Explained variance: {decomp.explained_variance_ratio:.4f}")
    print(f"Aspect contributions: {decomp.aspect_contributions}")

# Incremental update
stats = retriever.incremental_update(new_documents, new_relations=new_relations)
print(f"Added {stats['n_added']} docs in {stats['update_time_s']}s")

# Save / Load
retriever.to_json("index.json")
retriever = UnifiedRetriever.from_json("index.json")
```

### Algorithm Descriptions

#### Algorithm 1: Tensor Signature Construction

For each document *d* in the knowledge graph, construct a third-order tensor
T_d in R^{n_e x n_r x d} where n_e is the number of entities, n_r is the
number of relation types, and d is the embedding dimension. Entities are
extracted from author names, keywords, and venue. Each relation slice
encodes the connectivity pattern for a specific relation type.

#### Algorithm 2: Grassmann Vector Field Retrieval

1. Encode the query text using SBERT to obtain a dense vector q
2. Project q onto the Grassmann manifold Gr(k, d) via orthonormalization
3. Define a vector field V on the manifold that points toward semantically
   similar documents
4. Follow the vector field to retrieve the top-k nearest neighbors
5. Re-rank results using aspect-weighted scores from decomposition

#### Algorithm 3: Incremental Manifold Update

When new documents arrive:
1. Encode new documents with SBERT
2. Build tensor signatures for new entries
3. Project new signatures onto the existing manifold
4. Compute the Grassmannian mean shift: blend old and new means with
   weight alpha = n_new / (n_old + n_new)
5. Shift existing embeddings toward the updated mean
6. Extend the similarity index with new manifold embeddings

#### Tensor Decomposition

The system supports two decomposition methods:

- **CP Decomposition**: Factorizes the tensor into a sum of rank-one tensors,
  providing interpretable aspect contributions for each relation type.
- **Tucker Decomposition**: Factorizes the tensor into a core tensor and
  factor matrices along each mode, capturing multi-way interactions.

Decomposition results include explained variance ratio, reconstruction error,
and per-aspect contribution weights.

### OAG Format Support

The system supports loading data in Open Academic Graph (OAG) format. Documents
should be JSON files containing records with the following fields (field name
aliases are supported):

| Field | Aliases | Description |
|-------|---------|-------------|
| `id` | `paper_id` | Unique paper identifier |
| `title` | `name` | Paper title |
| `abstract` | `summary` | Paper abstract |
| `year` | `pub_year` | Publication year |
| `authors` | `author_names` | List of author names |
| `venue` | `journal`, `conference` | Publication venue |
| `keywords` | `tags`, `concepts` | List of keywords |

Citation files should contain entries with `source` (or `citing`) and `target`
(or `cited`) fields specifying citation relationships.

### License

MIT License

---

<a id="中文"></a>

## 张量流形图向量融合 (TMGVF)

基于论文 [Tensor Manifold-Based Graph-Vector Fusion for AI-Native Academic Literature Retrieval](https://arxiv.org/abs/2604.16416) (arXiv:2604.16416) 的 Python 实现。基于学术文献图是张量流形的离散投影这一理论基础，将文本向量和图拓扑统一到同一几何空间，支持论文、作者、期刊、机构、领域、项目等多类型异构节点，以及引用、合作、隶属等多种边类型。

### 特性

- **张量签名构建（算法 1）**：从知识图谱中的实体-关系三元组构建三阶张量
- **Grassmann 向量场检索（算法 2）**：将查询投影到 Grassmann 流形上进行几何感知的相似度搜索
- **增量流形更新（算法 3）**：高效更新索引，无需全量重建
- **张量分解**：支持 CP 和 Tucker 分解，用于多维度检索与分析
- **多种索引后端**：FAISS、HNSWLIB 和暴力相似度搜索
- **OAG 格式支持**：兼容开放学术图谱数据格式
- **灵活输出**：JSON、纯文本表格、Markdown 和详细报告

### 依赖

```
numpy>=1.20.0
sentence-transformers>=2.0.0   # 可选；未安装时使用随机嵌入
faiss-cpu>=1.7.0               # 可选；未安装时使用暴力搜索
hnswlib>=0.7.0                 # 可选；未安装时使用暴力搜索
pyyaml>=6.0                    # 可选；用于 YAML 配置文件
```

### 项目结构

```
tensor_manifold_gvf/
├── retrieval/
│   ├── __init__.py              # 模块导出
│   ├── retriever.py             # UnifiedRetriever, RetrievalResult
│   └── result_formatter.py      # ResultFormatter (JSON/表格/Markdown/详细)
├── scripts/
│   ├── build_index.py           # 从数据构建检索索引
│   ├── query_demo.py            # 交互式查询 REPL
│   ├── incremental_update.py    # 增量更新与验证
│   └── benchmark.py             # 性能基准测试
├── tests/
│   └── __init__.py
├── examples/
│   └── self_published_papers/
│       ├── papers.json          # 10 篇示例学术论文
│       └── citations.json       # 19 条引用关系
└── README.md
```

### 快速开始

#### 1. 构建索引

```bash
python scripts/build_index.py \
    --data examples/self_published_papers/papers.json \
    --citations examples/self_published_papers/citations.json \
    --output retriever_state.json
```

#### 2. 交互式查询

```bash
python scripts/query_demo.py --index retriever_state.json
```

交互式 REPL 支持的命令：
- `text <查询>` -- 文本检索
- `node <ID>` -- 节点检索（含完整分解）
- `top [k]` -- 显示上次查询的前 k 个结果
- `format <格式>` -- 切换输出格式（table / markdown / detailed）
- `save <路径>` -- 保存结果到 JSON
- `stats` -- 显示索引统计信息
- `quit` -- 退出

#### 3. 增量更新

```bash
python scripts/incremental_update.py \
    --index retriever_state.json \
    --new-data new_papers.json \
    --new-citations new_citations.json \
    --output updated_retriever.json
```

#### 4. 性能测试

```bash
python scripts/benchmark.py \
    --data examples/self_published_papers/papers.json \
    --citations examples/self_published_papers/citations.json \
    --iterations 100 \
    --output benchmark_results.json
```

### Python API 示例

```python
from retrieval import UnifiedRetriever, ResultFormatter

# 构建检索器
retriever = UnifiedRetriever(
    sbert_model="all-MiniLM-L6-v2",
    manifold_dim=64,
    index_type="faiss",
    decomposer_type="cp",
    rank=8,
)

retriever.build(documents, relations=citations)

# 文本搜索
results = retriever.search("graph neural network", top_k=10)
print(ResultFormatter.to_markdown(results))

# 节点搜索与分解
results, decomp = retriever.search_with_decomposition("paper_001", top_k=5)
if decomp:
    print(f"解释方差比: {decomp.explained_variance_ratio:.4f}")
    print(f"维度贡献: {decomp.aspect_contributions}")

# 增量更新
stats = retriever.incremental_update(new_documents, new_relations=new_relations)
print(f"新增 {stats['n_added']} 篇文档，耗时 {stats['update_time_s']}s")

# 保存 / 加载
retriever.to_json("index.json")
retriever = UnifiedRetriever.from_json("index.json")
```

### 算法说明

#### 算法 1：张量签名构建

对知识图谱中的每个文档 d，构建三阶张量 T_d 属于 R^{n_e x n_r x d}，
其中 n_e 为实体数量，n_r 为关系类型数量，d 为嵌入维度。实体从作者名、
关键词和发表场所中提取。每个关系切片编码特定关系类型的连接模式。

#### 算法 2：Grassmann 向量场检索

1. 使用 SBERT 编码查询文本，获得稠密向量 q
2. 通过正交化将 q 投影到 Grassmann 流形 Gr(k, d)
3. 在流形上定义向量场 V，指向语义相似的文档
4. 沿向量场检索 top-k 近邻
5. 使用分解得到的维度权重对结果进行重排序

#### 算法 3：增量流形更新

当新文档到达时：
1. 使用 SBERT 编码新文档
2. 为新条目构建张量签名
3. 将新签名投影到现有流形上
4. 计算 Grassmann 均值漂移：以 alpha = n_new / (n_old + n_new) 混合新旧均值
5. 将现有嵌入向更新后的均值方向偏移
6. 使用新流形嵌入扩展相似度索引

#### 张量分解

系统支持两种分解方法：

- **CP 分解**：将张量分解为一秩张量之和，提供每种关系类型的可解释维度贡献
- **Tucker 分解**：将张量分解为核心张量和各模式因子矩阵，捕获多路交互

分解结果包括解释方差比、重建误差和各维度贡献权重。

### OAG 格式支持

系统支持加载开放学术图谱 (OAG) 格式的数据。文档应为 JSON 文件，包含以下字段（支持字段别名）：

| 字段 | 别名 | 说明 |
|------|------|------|
| `id` | `paper_id` | 唯一论文标识符 |
| `title` | `name` | 论文标题 |
| `abstract` | `summary` | 论文摘要 |
| `year` | `pub_year` | 发表年份 |
| `authors` | `author_names` | 作者列表 |
| `venue` | `journal`, `conference` | 发表场所 |
| `keywords` | `tags`, `concepts` | 关键词列表 |

引用文件应包含 `source`（或 `citing`）和 `target`（或 `cited`）字段，指定引用关系。

### 许可证

MIT License
