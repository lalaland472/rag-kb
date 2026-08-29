# W23 Day 6 — Self-RAG 精读 + 代码落地：三模式问答

> 日期：2026-08-29 · 精读 Self-RAG (Asai et al., 2023) 后落地到 rag-kb
> 对应 learning-record 0007 · 参考卡 self-rag-reflection-tokens.html

## 一、落地内容

给 rag-kb 加了 Self-RAG 启发的三处能力：

### 1. 双生成模式（IsSup vs IsUse 权重切换）
- `--style strict` → **IsSup 拉满**：STRICT_SYSTEM_PROMPT，"每个可核验事实点都要有引用依据"，无依据标注；末尾附忠实性核对
- `--style chat` → **IsUse 拉高**：CHAT_SYSTEM_PROMPT，允许结合自身知识适度发散，目标"对用户最有用"

### 2. 忠实性核对（迷你版 IsSup）
`check_faithfulness()`：strict 模式下，用 LLM 检查回答里每个可核验事实点是否有片段支撑，无依据的逐条标「⚠️ 无资料依据」，全部有依据则返回 None

### 3. Retrieve on-demand（按需检索预判）
`check_retrieval_needed()`：判断 query 需不需要检索
- 纯常识/闲聊 → 走 LLM-only，跳过检索 + HyDE，省耗时
- 知识型 → 正常检索

## 二、验证结果

### 盲区验证（任务 2）：「一年有几个季节」
| 场景 | 结果 |
|------|------|
| 修复前 | 「根据提供的资料无法完整回答」——被 RAG 的"只答资料内"prompt 绑架 |
| 修复后 `--retrieval-check` | 「一年有四个季节：春、夏、秋、冬。」⏱2.0s · ⚡跳过检索 · chunk=0 |

**关键发现**：Retrieve on-demand 不只是省时间，更是避免 prompt 冲突——常识问题该放模型自己答，不该套"只答资料内"的严谨规则。

### 三模式对比（任务 1）：「Self-RAG 的反思 token 有哪两类」
| 模式 | 耗时 | 特点 |
|------|:---:|------|
| strict | 7.9s | 首句「根据论文片段」，只给 [3] 依据的结论，末尾⚠️已忠实核对 |
| chat | 6.1s | [3] 基础上发散补充"[1] 推理时生成、能按任务调整行为"，语气自然 |
| 默认 | 6.9s | 基准 |

## 三、暴露的残留问题（记入改进）
1. **引用不精准**：flat top-5 会带进 BERT/Lost-in-the-Middle 等不太相关片段 → 需提升检索召回精度
2. **忠实核对透明度**：核对结果"全部有依据"时只显示标记，看不到核对了什么 → 可加 verbose 选项输出核对详单

## 四、文件改动
- `scripts/generate_answer.py`：加 STRICT/CHAT/LLM_ONLY prompt + check_faithfulness + check_retrieval_needed + style/retrieval_check/llm_only 参数
- `scripts/query.py`：加 --style / --retrieval-check / --llm-only 参数 + /style、/rc 交互命令 + 输出标记（⚡跳过检索 / ⚠️已忠实核对）

---
📚 W23 Day 6 · 2026-08-29

## 五、残留问题修复：引用召回不精准（Cross-encoder 重排）

**问题**：flat 模式 top-5 混入不相关片段（问 Self-RAG 却召回 BERT MLM 训练段落、Lost-in-the-Middle）
**根因**：BGE-small 向量召回对"中英混合 query"判别有限，把"语义接近但话题不同"的片段误判相近
**修复**：落地架构图预留的 Cross-encoder 重排（bge-reranker-base，本地已缓存）
- 向量召回 top-20 → reranker 逐个算 query↔chunk 匹配分 → 重排取 top-5
- 验证差异：BERT 块32 的 rerank 分 0.0007（几乎为0），被彻底压出 top-5
- 端到端：strict 问答引用从"Self_RAG+BERT+LostInMiddle"收敛为"全 Self_RAG"

**改动文件**：
- retriever.py：+_rerank_chunks() / +chunk_text()（内置缓存，消除循环导入）/ retrieve()、hybrid_retrieve() 加 rerank、rerank_top 参数
- generate_answer.py：flat 模式默认开启 rerank

**注意**：rerank 首次加载模型较慢（端到端 40.9s），后续调用有缓存

## 六、flat 回归测试：rerank 量化验证（rerank vs 基线）

用 Day 4 的 20 条测试集重跑，对比 无rerank 基线 vs 带rerank：

### 核心指标
| 指标 | 基线(无rerank) | rerank | 变化 |
|------|:---:|:---:|:---:|
| 完全命中+有内容 | 10/20 | **12/20** | +20% |
| 未命中 | 2 | **1** | 改善 |
| 平均耗时 | 8.1s | 33.0s | +4x（成本）|

### rerank 明确修复的 5 条（基线没命中）
| ID | 基线 | rerank | 说明 |
|----|:---:|:---:|------|
| C1 | 0% | 67% | 高效微调多论文，补上 GPTQ+LoRA |
| C2 | 50% | 100% | 多智能体协作，补上 AutoGen |
| D1 | 50% | 100% | RAPTOR vs GraphRAG，双锚定 |
| D3 | 50% | 100% | DPO vs RLHF，补上关键论文 |
| E3 | 50% | 100% | embedding 向量化 |

### 引用纯净度提升（"引用召回不精准"修复的直接证据）
- E1：混入 FlashAttention → 纯 Generative_Agents + Attention
- D1：混 RAG_Survey → 纯 GraphRAG
- D4：混 RAG_Lewis/GraphRAG → 纯 Self_RAG

### rerank 的代价：B4 反而退化（67%→33%）
- B4（中文口语跨篇）：rerank 全选 MT_Bench（评估类），跑偏
- 根因：口语化多主题 query，reranker 把"评估/判断"字面语义误判高相关
- **结论**：rerank 擅长精确术语/单主题（A,C,D类受益）；口语跨篇（B类）用纯向量更好

### 应对：加 --no-rerank 逃生舱
- `python3 query.py "..." --no-rerank`：关闭 rerank，回归纯向量
- 验证：B4 用 --no-rerank 恢复完整回答（召回 RAPTOR+RAG Survey+Self RAG）
- 默认策略：rerank on（flat/detect 默认），口语跨篇 query 手动关

### 综合建议
rerank 是"引用精准度 vs 耗时 + 口语跨篇兼容性"的权衡。推荐：
- 精确/单主题：rerank on（收益大）
- 口语/跨篇/快速闲聊：--no-rerank 或 --retrieval-check
- 未来可做按 query 类型自动路由（类似 Self-RAG on-demand 决策）
