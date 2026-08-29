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
