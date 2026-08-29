# RAG-KB 性能报告 / Benchmark

> W24 Day 2 · 2026-08-29 · 汇总 W21-W23 全部评测数据
> 覆盖：检索召回 / 延迟 / RAPTOR vs Flat / 三组对比 / rerank 回归 / 幻觉分析

---

## 一、检索性能（W21 验证）

来源：`scripts/evaluate_recall.py` · 100 条 query

| 指标 | 值 |
|------|:---:|
| **Recall@5** | **0.91** |
| MRR | 0.808 |
| 平均首命中位置 | 1.35 |
| 平均检索耗时 | 160ms |

**混淆分析**：少量跨论文混淆（如 Lost_in_the_Middle↔Generative_Agents、BERT↔ReAct），多因主题相近的论文（记忆/Transformer）语义重叠。引入 rerank 后可进一步缓解。

---

## 二、RAPTOR vs Flat（W22，8 条主题查询）

来源：`scripts/accept_raptor.py` · W22 验收

| 模式 | top-1 命中 | 说明 |
|------|:---:|------|
| **Hybrid（Flat+RAPTOR）** | **8/8** | 摘要加权 RRF ×3.0，主题检索强 |
| Flat（纯叶子） | ~5/8 | 精确句子召回好，主题泛化弱 |

**结论**：RAPTOR 摘要节点把分散全文浓缩成主题，对"中文主题性/跨篇"查询显著优于纯 Flat。双模式各有主场。

---

## 三、端到端测试（W23 Day 4-6，20 条 query）

覆盖 5 类（A 精确术语/B 口语/C 跨篇/D 比较/E 边界）。

### 3.1 基线（无 rerank，Day 4）
| 模式 | 完全命中 | 平均耗时 |
|------|:---:|:---:|
| flat | 10/20 | 8.1s |
| hybrid | **15/20** | 8.0s |

### 3.2 加 rerank 后（Day 6）
| 模式 | 完全命中 | 平均耗时 | 变化 |
|------|:---:|:---:|:---:|
| flat + rerank | **12/20** | 33.0s | +20% 命中 |
| hybrid + rerank | 13/20 | 14.9s | **-13% 命中** |

**rerank 路由结论**：
- **flat + rerank 推荐**（+20%，修引用精准度，压掉无关片段）
- **hybrid 不开 rerank**（破坏 RAPTOR 摘要融合，反而降）
- 口语跨篇 query（B4 类）两种都易跑偏，建议 `--no-rerank` 或 `--retrieval-check`

---

## 四、三组对比（W23 Day 5）

来源：`scripts/compare_modes.py` · 同 prompt 三模式

| 模式 | 平均命中/条 | 可溯源 | 诚实承认不足 | 幻觉风险 | 耗时 |
|------|:---:|:---:|:---:|:---:|:---:|
| LLM-only | 0.8 | 5% | 20% | 0%* | 5.9s |
| RAG（flat） | 0.9 | **100%** | 35% | 10% | 6.9s |
| RAPTOR（hybrid）| **1.1** | **100%** | 15% | 5% | 6.7s |

*LLM-only 的"0 幻觉"是评估盲区（无来源可验，非真没有）。

**核心结论**：
1. **RAPTOR 召回最全**（1.1/条，主题/比较类命中多篇）
2. **可溯源是 RAG 的决定性优势**（LLM-only 仅 5% 可验证）
3. **flat 最诚实**（35% 明说资料不足），但答不全；hybrid 靠摘要缓解

---

## 五、幻觉案例分析（Day 5 flat E3）

**问题**："解释什么是 embedding 向量化" → flat 回答给了定义，但**归因到错误片段**（BERT 某段），命中 0 且没承认不足。
→ 源自**检索错位导致的"伪可靠"**：模型基于不相关片段顺水推舟，比直接承认不足更危险。
→ 对策：strict 模式的忠实核对（W23 Day 6 落地）可标记这类"无资料依据"事实点。

---

## 六、Self-RAG 增强验证（W23 Day 6）

| 能力 | 验证结果 |
|------|---------|
| `--retrieval-check`（on-demand）| 纯常识 query 跳过检索，8s→2s，避免 prompt 冲突 bug（已修复）|
| `--style strict`（IsSup）| 每句有依据 + 忠实核对 |
| `--style chat`（IsUse）| 引用准确基础上放开发散，回答更完整 |
| 双模式实测 | chat 能补出 default 不敢说的全称（如 RAPTOR 定义），引用仍准确 |

---

## 七、关键工程决策（1GB 内存约束）

- 4521 chunks 全量 UMAP 聚类会内存耗尽（thrash）→ 轻量清洗只编码摘要节点
- 双模式由调用方按场景选择；rerank 按模式路由（flat 开 / hybrid 关）

---

## 八、总结

1. **RAG-KB 可用**：hybrid 完整命中 75%，Flat R@5=0.91，带页码引用
2. **双模式正确**：精确术语用 flat，主题/跨篇用 hybrid
3. **rerank 要路由**：flat 开（+20%）、hybrid 关（-13%）
4. **价值定位**：RAG 的核心价值是"**可验证、不凭记忆裸奔**"，不只是"答得更好"

---
📚 W24 Day 2 · Benchmark · 2026-08-29
