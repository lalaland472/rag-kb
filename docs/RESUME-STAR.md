# RAG-KB 简历项目条目（STAR 法则）

> W24 Day 4 · 2026-08-29 · 求职作品集第二块
> 配合 Month 7 的 Agent 工程化产品，形成完整作品集

---

## 项目一句话

**RAG-KB — 从零手写的个人 AI 论文知识库问答系统**：把 23 篇 AI 论文建成可检索知识库，自然语言提问返回带来源引用的答案；并借鉴 Self-RAG 思想实现"严格/对话"双模式、按需检索与重排序。

---

## STAR 条目（可直接放入简历）

### S（情境）
学习 LLM 全栈时发现：RAG 是解决 LLM 幻觉、支撑"可验证回答"的关键，但主流框架（LangChain/LlamaIndex）封装过深，难以理解底层检索原理。决定**从零手写**一套完整 RAG 系统，作为求职作品集的工程深度证明。

### T（任务）
独立设计并实现一个生产可用的个人知识库问答产品：检索要准（大论文库）、回答要可信（带来源引用）、要能处理中文口语化问题和跨论文主题比较。

### A（行动）
**从零手写完整 RAG 管道，不依赖 RAG 框架**（LangChain/LlamaIndex）：
- **索引**：PyMuPDF 解析 → Semantic Chunking(p=10) → BGE-small-zh(512d) → FAISS HNSW(M=16) → 4521 chunks / 23 篇论文
- **多层次检索**：实现 **双模式**——Flat 叶子精确召回 + RAPTOR 递归摘要树混合加权 RRF（×3.0），解决"精确术语 vs 中文主题"两类查询
- **精排**：接入 Cross-encoder（bge-reranker）重排，压掉"语义接近但话题不同"的无关片段；并通过回归测试确定 rerank **按模式路由**（flat 开 +20%、hybrid 关避免 -13%）
- **生成**：DeepSeek 生成 + 来源引用链（chunk→文件→页码）
- **Self-RAG 思想落地**：`--strict`（IsSup 拉满，每句有依据+忠实核对）/ `--chat`（IsUse 放开，引用准确上发散）/ `--retrieval-check`（Retrieve on-demand，常识问题跳过检索省 75% 耗时）

### R（结果）
- **检索**：Flat R@5 = 0.91，MRR 0.81
- **问答**：hybrid 端到端完整命中 75%，0 异常，平均 ~8s/问，**100% 可溯源**（带页码引用）
- **三组对比**（LLM-only vs RAG vs RAPTOR）：RAG 可溯源 100% vs LLM-only 仅 5%；RAPTOR 召回最全（1.1/条）
- **rerank 优化**：flat 完整命中 +20%，引用纯净度大幅提升（问 Self-RAG 不再混入 BERT 段落）
- **Demo**：`scripts/demo.py` 一键演示 5 类代表性查询
- 全流程代码自查：检索错位分析、忠实性核对、on-demand 预判，验证"可验证回答"定位

---

## 技术栈关键词（简历技能栏）

Python · RAG · FAISS · BGE (sentence-transformers) · PyMuPDF · Cross-encoder rerank · RAPTOR · Self-RAG(Reflexion思想) · DeepSeek API · 系统设计（多模式检索、成本路由决策）

---

## 突出亮点（可单独强调）

1. **"可验证回答"定位**：RAG 的核心价值不是"答得更好"而是"可溯源、不凭记忆裸奔"——三组对比数据支撑
2. **rerank 按模式路由**：不是无脑加 rerank，而是用回归测试数据定出"flat 开 / hybrid 关"的决策规则
3. **Self-RAG 落地**：把论文思想（IsSup/IsUse/on-demand）翻译成真实可用的 CLI 参数
4. **从零不依赖框架**：所有检索/分块/重排/生成链路手写，展现工程深度

---

📚 W24 Day 4 · 简历条目 · 2026-08-29
GitHub: https://github.com/lalaland472/rag-kb
