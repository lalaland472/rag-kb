# RAG-KB — 个人 AI 论文知识库问答系统

> **Month 6 Capstone · Week 21-22 | 150天 AI 学习计划**
> 从零手写的 RAG 系统，把 20+ 篇 AI 论文建成可检索的个人知识库，用自然语言提问，返回带来源引用的答案。

## 📌 本项目涵盖（W21 + W22）

本项目对应学习计划 Month 6 的前两周，完整实现一个 RAG 知识库的两个阶段：

| 阶段 | 主题 | 核心内容 | 状态 |
|------|------|---------|:---:|
| **W21** | 数据层 + 索引 | PDF解析 → Semantic Chunking(p=10) → BGE-small-zh(512d) → FAISS HNSW | ✅ |
| **W22** | RAPTOR 整合 | DeepSeek 抽象摘要 → 摘要索引 → Flat + RAPTOR 混合检索（双模式） | ✅ |

## 🏗️ 技术架构

```
文档导入 (PyMuPDF)
  → Semantic Chunking (p=10, per-document)
  → BGE-small-zh-v1.5 编码 (512维)      [scripts/chunk_pipeline.py, ingest.py]
  → FAISS HNSW 索引 (M=16)              [data/index/index.faiss]
  → RAPTOR 摘要索引                     [build_raptor_index.py, abstract_summary.py]
  → 双模式检索: Flat叶子 / 混合加权RRF   [retriever.py]
  → 来源引用                           [metadata.json 映射]
```

## 🧪 双模式检索（W22 核心成果）

在同一查询上，两种模式各自主场最优：

| 模式 | 原理 | 适合场景 | 验证结果 |
|------|------|---------|---------|
| **flat**（默认）| 纯 FAISS 叶子精确召回 | 精确句子/细节匹配 | **R@5 = 0.96**（验收达标）|
| **hybrid** | Flat + RAPTOR 摘要加权 RRF / ×3.0 | 中文主题性问题、跨篇检索 | **8/8 查询 top-1** |

**验证示例**：
```bash
# 精确召回（flat）
python3 retriever.py "哪篇论文讲了给 LLM 加反思和自评的机制？" --mode flat
# 主题检索（hybrid）→ Self_RAG top-1
python3 retriever.py "哪篇论文讲了给 LLM 加反思和自评的机制？" --mode hybrid
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install pymupdf sentence-transformers faiss-cpu numpy scikit-learn umap-learn requests
```

### 2. 获取论文 PDF（本项目不携带 PDF，自行下载后放入 data/papers/）
```bash
python3 scripts/download_papers.py   # 或手动放入 data/papers/*.pdf
```

### 3. 处理管道（按序执行）
```bash
# W21: 分块 + BGE 编码 + FAISS HNSW 索引
python3 scripts/chunk_pipeline.py
python3 scripts/ingest.py

# W22: 构建 RAPTOR 摘要索引（需 DeepSeek API，见 config/credentials.env 模板）
python3 scripts/build_raptor_index.py   # 或 clean_summary_index.py（轻量清洗）
python3 scripts/abstract_summary.py     # 抽象摘要（可选用真实 LLM）

# 检索
python3 scripts/retriever.py "你的问题" --mode flat
python3 scripts/retriever.py "你的问题" --mode hybrid --debug
```

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `scripts/chunk_pipeline.py` | Semantic Chunking 分块（p=10）|
| `scripts/ingest.py` | BGE 编码 + FAISS HNSW 索引构建 |
| `scripts/build_raptor_index.py` | RAPTOR 摘要索引构建（聚类+摘要）|
| `scripts/clean_summary_index.py` | 轻量清洗摘要索引（滤致谢/版权垃圾）|
| `scripts/abstract_summary.py` | DeepSeek 抽象摘要生成（替换 extractive）|
| `scripts/retriever.py` | 双模式检索器（Flat / Hybrid）|
| `scripts/evaluate_recall.py` | W21 Recall@5 验证（0.91）|
| `scripts/accept_raptor.py` | W22 验收评测 |

## ⚙️ 配置

复制 `config/credentials.env.template` 为 `config/credentials.env` 并填入你的 DeepSeek API key。
（生产环境请用环境变量，勿提交 key）

## 📊 已知约束 / 工程决策

- **1GB 内存约束**：对 4521 chunks 全量跑 UMAP 逐篇聚类会内存耗尽（thrash）→ 改用「轻量清洗」只编码摘要节点。
- **双模式取舍**：Flat 保精确句子召回，Hybrid 保主题检索；由调用方按场景选择（详见架构）。

## 🧠 学习路径

150天 AI 学习计划 Month 6：
- Month 1-5: AI基础 → 微调 → DPO → RAG → Agent → 量化 → 推理加速 → 部署 → 评估 → FAISS → 高级检索 → 长文档
- **Month 6 (当前)**: RAG-KB Capstone（本项目）→ Agent 工程化

## 📚 参考论文

- Lost in the Middle (2023) — 长上下文信息位置效应（prompt 组织指导）
- RAPTOR (2024) — 递归摘要树
- Self-RAG (2023) — 检索反思
- RAG Survey (2024) — RAG 全景

---
📚 Month 6 · Week 21-22 · RAG-KB Capstone · 2026-08-27
