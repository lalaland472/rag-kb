# RAG-KB — 个人 AI 论文知识库问答系统

> **Month 6 Capstone · Week 21-24 | 180天 AI 学习计划**
> 从零手写的 RAG 系统，把 20+ 篇 AI 论文建成可检索的个人知识库，用自然语言提问，返回带来源引用的答案。含 Self-RAG 启发的高级问答（严格/对话双模式 + 按需检索 + rerank 精排）。

## 📌 本项目涵盖（W21-24）

本项目对应学习计划 Month 6 的完整四周，从数据层到可用的问答产品：

| 阶段 | 主题 | 核心内容 | 状态 |
|------|------|---------|:---:|
| **W21** | 数据层 + 索引 | PDF解析 → Semantic Chunking(p=10) → BGE-small-zh(512d) → FAISS HNSW | ✅ |
| **W22** | RAPTOR 整合 | DeepSeek 抽象摘要 → 摘要索引 → Flat + RAPTOR 混合检索（双模式） | ✅ |
| **W23** | 问答 + 生成 | generate_answer + 引用链 + CLI + 20条测试 + 三组对比 + **Self-RAG 落地** | ✅ |
| **W24** | 收尾 | README / Benchmark / 简历条目 | ✅ |

## 🏗️ 技术架构

```
文档导入 (PyMuPDF)
  → Semantic Chunking (p=10, per-document)
  → BGE-small-zh-v1.5 编码 (512维)      [chunk_pipeline.py, ingest.py]
  → FAISS HNSW 索引 (M=16)              [data/index/index.faiss]
  → RAPTOR 摘要索引                     [build_raptor_index.py]
  → 双模式检索: Flat叶子 / 混合加权RRF   [retriever.py]
  → Cross-encoder 重排 (bge-reranker)   [retriever.py, W23 优化]
  → 生成 (DeepSeek) + 来源引用链        [generate_answer.py, citation_pipeline.py]
  → 交互 CLI（strict/chat 双风格）       [query.py]
```

## 🧪 双模式检索（W22 核心成果）

| 模式 | 原理 | 适合场景 | 验证结果 |
|------|------|---------|---------|
| **flat**（默认）| 纯 FAISS 叶子精确召回 + 可选 rerank | 精确句子/细节匹配 | **R@5 = 0.96** |
| **hybrid** | Flat + RAPTOR 摘要加权 RRF ×3.0 | 中文主题性、跨篇检索 | **8/8 查询 top-1** |

## ✨ 高级问答（W23 + Self-RAG 启发）

`query.py` 从"检索+生成"升级为带 Self-RAG 思想的交互式问答：

| 能力 | 对应 Self-RAG | 说明 |
|------|--------------|------|
| `--style strict` | **IsSup 拉满** | 每个可核验事实点都要有引用依据，末尾附忠实性核对 |
| `--style chat` | **IsUse 拉高** | 允许结合自身知识适度发散，回答更完整自然 |
| `--retrieval-check` | **Retrieve on-demand** | 判断是否需检索，纯常识/闲聊 query 直接 LLM-only，省耗时 |
| `--no-rerank` | rerank 路由决策 | flat 默认 rerank；口语跨篇 query 可手动关闭 |

**rerank 实测（flat，20 条测试集）**：完整命中 10→12（+20%），引用纯净度大幅提升；hybrid 模式不开（负优化）。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install pymupdf sentence-transformers faiss-cpu numpy scikit-learn umap-learn requests
```

### 2. 配置
复制 `config/credentials.env.template` 为 `config/credentials.env` 并填入 DeepSeek API key。

### 3. 构建索引（W21-22，按序执行）
```bash
python3 scripts/chunk_pipeline.py     # 分块
python3 scripts/ingest.py             # BGE 编码 + FAISS HNSW
python3 scripts/build_raptor_index.py # RAPTOR 摘要索引
```

### 4. 交互式问答（W23）
```bash
# 单次查询（默认 flat + rerank）
python3 scripts/query.py "Self-RAG 的反思 token 有哪两类？"

# 严格模式（每句有依据 + 忠实核对）
python3 scripts/query.py "RAG Survey 讲了哪三阶段？" --style strict

# 对话模式（允许发散）
python3 scripts/query.py "DPO 和 RLHF 区别？" --style chat

# 按需检索（纯常识 query 跳过检索，省耗时）
python3 scripts/query.py "一年有几个季节" --retrieval-check

# 交互会话
python3 scripts/query.py --interactive
```

## 🕸️ GraphRAG 演示（W24 Day 3）

面向**整个语料库**的全局问答演示——普通 RAG 检索"片段"，GraphRAG 归纳"森林"。
配套可视化图谱：`data/graph/graph.html` 浏览器打开可交互浏览实体图谱与社区。

```bash
python3 scripts/graphrag_demo.py            # 完整四幕
python3 scripts/graphrag_demo.py --quick    # 只跑全局问答对比（省时）
```

演示四幕：
- **Act 0 图谱概览**：实体节点/关系边/社区/论文规模
- **Act 1 社区地图**：语料库自动拆出的技术主题社区
- **Act 2 全局问答对比**：GraphRAG vs RAG(flat) 同屏对比
- **Act 3 Map-Reduce 透明**：选社区 → 逐社区答 → 汇总，全程可见

实际效果（2026-09-01 实测，模型 deepseek-v4-flash）：

| 全局问题 | RAG(flat) | GraphRAG |
|------|:---:|:---:|
| Q1 语料库整体主题 | ❌ 空答（62s）| ✅ 完整架构脉络（17s）|
| Q2 RAG 演进与变体 | ✅ 三阶段 | ✅ 演进+自反思+图增强 |
| Q3 大模型对齐方法 | ✅ RLHF+DPO | ✅ 更结构化（含两者关系）|
| Q4 组织知识方式 | ⚠️ 无法完整回答 | ✅ 多层覆盖（架构/优化/外部知识/评估）|

**亮点**：Q1 上 flat 直接空答，GraphRAG 给出结构化全局答案——全局归纳性问题正是普通 RAG 失效、GraphRAG 的主场。

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `scripts/chunk_pipeline.py` | Semantic Chunking 分块（p=10）|
| `scripts/ingest.py` | BGE 编码 + FAISS HNSW 索引构建 |
| `scripts/build_raptor_index.py` / `clean_summary_index.py` | RAPTOR 摘要索引构建/清洗 |
| `scripts/abstract_summary.py` | DeepSeek 抽象摘要生成 |
| `scripts/retriever.py` | 双模式检索器 + Cross-encoder rerank |
| `scripts/citation_pipeline.py` | 来源引用链（chunk → 文件 → 页面）|
| `scripts/generate_answer.py` | 生成模块（strict/chat + Retrieve-on-demand + 忠实核对）|
| `scripts/query.py` | 交互式问答 CLI |
| `scripts/demo.py` | RAG-KB 演示（flat/hybrid 五场景）|
| `scripts/graphrag_demo.py` | **GraphRAG 专项演示**（图谱概览 + 社区地图 + 全局问答对比 + Map-Reduce 透明）|
| `scripts/evaluate_recall.py` | W21 Recall@5 验证 |
| `scripts/compare_modes.py` | W23 三组对比（LLM-only vs RAG vs RAPTOR）|
| `scripts/test_queries.py` | 20 条 query 端到端回归测试 |

## 📊 评测结果

| 维度 | 结果 |
|------|------|
| 检索召回 | Flat R@5 = 0.96；Hybrid 8/8 top-1 |
| 三组对比（Day5）| RAPTOR 召回最全(1.1/条)、可溯源 100%；RAG 核心价值=可验证 |
| rerank 回归（flat）| 完整命中 10→12 (+20%)，引用纯净度提升，B4 口语跨篇需关 |
| 平均耗时 | 单次问答 ~8s（开放模型 + 本地检索）|

## 🧠 学习路径

180天 AI 学习计划 Month 6：RAG-KB Capstone（本项目）→ **Month 7 Agent 工程化**

## 📚 参考论文

- RAG Survey (2024) — RAG 全景（Naive/Advanced/Modular）
- Lost in the Middle (2023) — 长上下文位置效应
- RAPTOR (2024) — 递归摘要树
- Self-RAG (2023) — 检索反思（本项目落地 strict/chat/on-demand）

---
📚 Month 6 · Week 21-24 · RAG-KB Capstone · 2026-08-29
