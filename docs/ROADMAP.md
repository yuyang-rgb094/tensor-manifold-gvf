# Tensor Manifold GVF 后续开发路线图

> 本文档说明项目当前状态及后续开发优先级建议，供开发者社区参考。

## 当前状态总览

| 阶段 | 功能 | 状态 | 测试覆盖 |
|------|------|------|----------|
| Phase 1 | 四通道编码器 (BGE-M3 + EmbeddingBag + GraphSAGE + Time2Vec) | ✅ 完成 | 44 tests |
| Phase 2 | 跨模态注意力融合 | ✅ 完成 | 9 tests |
| Phase 3 | 任务特定注意力头 + 通道权重学习 + 可视化 | ✅ 完成 | 15 tests |
| Phase 4 | Qdrant Named Vectors 存储 | ✅ 完成 | 15 tests |
| Phase 5 | FastAPI Knowledge API 骨架 | ✅ 完成 | 23 tests |
| **P2** | **任务头训练管道** | 🟡 骨架 | 需补充 |
| **P2** | **知识图谱构建端点** | 🟡 Mock | 需实现 |
| **P2** | **Insights提取端点** | 🟡 Mock | 需实现 |

**当前总测试数**: 141 passed

---

## P2 功能开发建议

### 功能1: Task-specific Attention Heads 训练管道

#### 当前状态
- ✅ `ChannelWeightLearner` - 完整实现（训练、保存、加载）
- ✅ `TaskSpecificAttentionHead` - 4任务类型完整实现
- ✅ `TaskRouter` - API层路由实现
- ⚠️ `UnifiedRetriever` - 初始化了任务头但未在搜索中实际调用
- ❌ 训练脚本/管道 - 缺失

#### 建议策略: **社区开发（Good First Issue）**

**原因**:
1. 核心算法已实现，仅需"胶水代码"
2. 工作量适中（2-3天）
3. 适合熟悉 PyTorch 的开发者入门

**开发指引**:

**任务 A: 完成 UnifiedRetriever 任务头集成**
```python
# 文件: retrieval/retriever.py
# 方法: _search_four_channel()

# 当前: 直接返回融合后的向量，未使用任务头
query_manifold = self._fusion_encoder.encode_all_from_channels(...)

# 需要: 通过当前任务头处理
if self._current_task_head is not None:
    query_manifold = self._current_task_head(
        torch.from_numpy(query_manifold).unsqueeze(0)
    ).detach().numpy()
```

**任务 B: 创建训练脚本**
```python
# 建议文件: scripts/train_task_heads.py

# 输入: 带标注的训练数据
# 格式: {"semantic": vec, "metadata": vec, "topology": vec, "temporal": vec, "label": 1.0, "task": "citation_analysis"}

# 输出: 训练好的任务头权重文件
# 路径: checkpoints/task_heads/{task_name}.pt

# 关键类: ChannelWeightLearner (已实现)
```

**验收标准**:
- [ ] `UnifiedRetriever.search()` 支持 `task` 参数切换任务头
- [ ] 训练脚本可运行并保存模型
- [ ] 新增 5+ 个测试覆盖训练流程

---

### 功能2: 知识图谱构建端点

#### 当前状态
- ✅ `HeteroAcademicGraph` - 完整异构图数据结构
- ✅ 图工具函数 (BFS遍历、密度计算等)
- ✅ API端点骨架 (`/api/v1/knowledge-graph`)
- ❌ 图谱构建算法 - 未实现（当前返回空edges）

#### 建议策略: **核心团队优先开发**

**原因**:
1. 需要设计图扩展算法（从种子论文到相关实体）
2. 涉及多跳关系推理，算法复杂度较高
3. 影响 API 核心功能体验

**开发指引**:

**算法设计**:
```python
# 建议文件: core/knowledge_graph_builder.py

class KnowledgeGraphBuilder:
    def build_from_seeds(
        self,
        seed_papers: List[str],
        max_nodes: int = 100,
        relation_types: List[str] = ["cites", "related"]
    ) -> Tuple[List[Node], List[Edge]]:
        """
        从种子论文构建知识图谱
        
        策略:
        1. 种子论文作为初始节点
        2. BFS扩展: 通过引用关系(cites)找到相关论文
        3. 添加作者、机构、期刊等元数据节点
        4. 计算边权重（基于共现频率、语义相似度）
        5. 当节点数达到max_nodes时停止
        """
        pass
```

**与现有代码集成**:
```python
# 文件: api/main.py
# 端点: POST /api/v1/knowledge-graph

from core.knowledge_graph_builder import KnowledgeGraphBuilder

@app.post("/api/v1/knowledge-graph")
async def knowledge_graph(request: KnowledgeGraphRequest) -> KnowledgeGraphResponse:
    builder = KnowledgeGraphBuilder(graph=hetero_graph)
    nodes, edges = builder.build_from_seeds(
        seed_papers=request.seed_papers,
        max_nodes=request.max_nodes,
        relation_types=request.relation_types,
    )
    return KnowledgeGraphResponse(nodes=nodes, edges=edges, ...)
```

**验收标准**:
- [ ] 从2个种子论文可扩展到50+节点
- [ ] 支持3种以上关系类型（cites, coauthor, same_venue）
- [ ] 边权重计算合理（0-1范围）
- [ ] 响应时间 < 2s（100节点）

---

### 功能3: Insights 提取端点

#### 当前状态
- ✅ API端点骨架 (`/api/v1/insights`)
- ❌ 洞察生成逻辑 - 未实现（当前返回硬编码响应）

#### 建议策略: **分阶段实现**

**阶段1: 基于规则的Insights（社区可参与）**
- 趋势预测: 基于时间序列统计
- Gap分析: 基于关键词共现缺失检测
- 工作量: 2-3天

**阶段2: LLM增强Insights（核心团队）**
- 使用GPT-4/Claude生成研究建议
- 需要设计Prompt工程和结果验证
- 工作量: 5-7天

**开发指引**:

**阶段1实现示例**:
```python
# 建议文件: core/insights_engine.py

class InsightsEngine:
    def generate_trend_prediction(
        self,
        query: str,
        papers: List[Paper]
    ) -> InsightItem:
        """基于时间序列统计预测趋势"""
        # 1. 提取query相关论文的时间分布
        # 2. 计算年增长率
        # 3. 识别新兴关键词
        # 4. 生成趋势描述
        pass
    
    def generate_gap_analysis(
        self,
        query: str,
        papers: List[Paper]
    ) -> InsightItem:
        """基于关键词共现识别研究空白"""
        # 1. 构建关键词共现矩阵
        # 2. 识别低频但高潜力的关键词组合
        # 3. 生成gap描述
        pass
```

**验收标准**:
- [ ] 支持3种insight_type（trend_prediction, gap_analysis, inspiration）
- [ ] 每个insight附带支撑论文列表
- [ ] confidence分数基于数据支撑度计算

---

## 开发者上手路径

### 路径A: 熟悉代码库（1天）
1. 阅读 `CONTEXT.md` 理解领域概念
2. 阅读 `docs/adr/` 了解架构决策
3. 运行测试: `pytest tests/ -v`
4. 启动API: `uvicorn api.main:app --reload`

### 路径B: 第一个贡献（Task Heads训练）
1. 阅读 `core/channel_weight_learner.py`
2. 阅读 `core/task_attention.py`
3. 实现 `scripts/train_task_heads.py`
4. 提交PR（包含测试）

### 路径C: 核心功能开发（知识图谱）
1. 阅读 `data/oag_schema.py` 理解图结构
2. 阅读 `utils/graph_utils.py` 了解现有工具
3. 设计 `core/knowledge_graph_builder.py`
4. 在 `api/main.py` 中替换mock实现
5. 性能测试和优化

---

## 技术债务与注意事项

### 已知问题
1. **FastAPI生命周期警告**: 使用`@app.on_event`已弃用，建议迁移到`lifespan`
2. **Qdrant连接失败降级**: 当前降级到内存索引，但无持久化警告
3. **四通道编码器内存占用**: BGE-M3 (1024d) + GraphSAGE在大数据量时需注意GPU内存

### 扩展建议
1. **缓存层**: 高频查询结果可缓存到Redis
2. **异步处理**: 知识图谱构建可改为后台任务（Celery）
3. **监控**: 添加Prometheus指标收集

---

## 联系与协作

- **Issue标签**: `good-first-issue`, `help-wanted`, `core-feature`
- **讨论区**: 使用GitHub Discussions进行设计讨论
- **PR要求**: 必须包含测试，通过CI（141 tests）

---

*最后更新: 2026-06-01*
*版本: v2.0.0-skeleton*
