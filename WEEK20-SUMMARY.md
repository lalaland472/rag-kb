# Week 20 — RAG 技术全景图 + Capstone 方案设计

Date: 2026-07-25 | 完成状态: ✅

## 产出文件
- `RAG_PANORAMA.md`: RAG 全链路技术全景图
- `CAPSTONE_PLAN.md`: RAG-KB 个人知识库产品方案
- `full_rag_pipeline.py`: 完整 RAG 管道（Embed → Search → Rerank → Generate）

## Day 1-2: RAG 技术全景图

### 全链路骨架
```
Query改写(W18) → Embedding(W8,19) → 分块(W19) → 索引(W17) → 检索(全程) → 重排序(W8) → 生成(W8)
```

### 关键 trade-off 矩阵
| 决策点 | 选项 | 个人知识库选择 | 原因 |
|--------|------|:--:|------|
| 分块 | Naive vs Semantic | Semantic(p=8-15) | W19 Day4 R@5=97% |
| 索引 | Flat vs HNSW vs IVF | HNSW(M=16) | 百万级+高召回 |
| 检索 | 单Query vs Multi-Query | 单Query | 个人场景优先效率 |
| 树索引 | Flat vs RAPTOR | Flat默认+RAPTOR实验 | W19 Day5结论 |
| 重排序 | 有 vs 无 | 可选(大文档有收益) | W20 Day3-4 |

### Month 5 仓库分工
| 仓库 | 周 | 覆盖环节 |
|------|-----|---------|
| rag-knowledge-bot | W8 | 全链路基础 |
| faiss-benchmark | W17 | 向量索引层 |
| retrieval-strategies | W18 | 查询改写+融合 |
| longdoc-rag | W19 | 分块+树检索 |

## Day 3-4: 全链路补缺

### full_rag_pipeline.py 实验结论
- LLM-only: RAG=音乐软件（幻觉严重）
- RAG+检索: 答案准确引用文档
- 耗时拆解: Embed+Search+Rerank <20ms, Generate ~6.4s (99%)
- Cross-encoder: 小文档收益不明显

## Day 5-6: Capstone 方案设计

### RAG-KB 产品定位
个人 AI 知识库问答助手：20+篇论文 → 自然语言提问 → 来源引用回答

### 技术选型
PyMuPDF → Semantic Chunking → BGE-zh(512d) → FAISS HNSW → RAPTOR树 → Collapsed Tree → Qwen7B生成

### Month 6 开发计划
| 周 | 内容 |
|-----|------|
| W21 | 数据层+索引构建 |
| W22 | RAPTOR树整合（最难，W19 Day5翻车过） |
| W23 | 问答+Qwen7B生成 |
| W24 | 收尾+展示+GitHub |

### RAPTOR 成功标准
Recall@5 >= Flat的90%，且摘要命中率 >= 20%
