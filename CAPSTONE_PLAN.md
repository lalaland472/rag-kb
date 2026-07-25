# RAG-KB: 个人知识库问答助手

## 产品定位

把你学过的所有 AI 资料（论文、笔记、代码注释）放进一个知识库，用自然语言问，AI 帮你找到答案并附来源引用。

## 核心功能

1. **文档导入**：指定目录，自动扫描 PDF/Markdown/纯文本，分块建索引
2. **智能问答**：自然语言提问 → 检索相关章节 → LLM 生成回答
3. **来源引用**：每段回答附带来源（哪篇文档、哪个段落）

## 技术架构

```
文档导入(PyMuPDF)
  → Semantic Chunking(p=10, per-document)
  → BGE-small-zh-v1.5 (512维)
  → FAISS HNSW (M=16)
  → RAPTOR 树 (跨篇聚类, UMAP+GMM)
  → Collapsed Tree 检索
  → Cross-encoder 重排序 (bge-reranker-base, 可选)
  → Qwen2.5-7B-Instruct 生成
  → 来源引用
```

## 项目结构

```
rag-kb/
├── data/                  # 20+ 篇资料
├── index/                 # FAISS 索引 + RAPTOR 树
├── src/
│   ├── ingest.py          # 扫描目录 → 分块 → 建索引
│   ├── query.py           # 提问 → 检索 → 生成 → 来源
│   └── config.py          # 模型路径、参数配置
├── README.md
└── requirements.txt
```

## Month 6 开发计划 (Week 21-24)

### Week 21: 数据层 + 索引构建
- 整理 20+ 篇资料, Semantic Chunking + BGE embedding
- FAISS HNSW 索引 + 来源追踪
- 索引持久化

### Week 22: RAPTOR 树整合
- 整合 W19 raptor_pipeline.py
- 跨篇聚类 + 摘要生成(Qwen0.5B)
- RAPTOR vs Flat 检索对比

### Week 23: 问答 + 生成
- Qwen7B 生成 + 来源引用
- CLI 交互式问答
- 端到端测试

### Week 24: 收尾 + 展示
- README 完善 + 性能报告
- GitHub 推送 + 简历描述
