# 📦 待推送 GitHub 打包说明 — 2026-09-02

> 用途：为后续 `git push` 到 GitHub 做准备。已本地整理好 commit，推送确认后执行。
> 主推仓库：**rag-kb**（W24 补测 + GraphRAG 完整落地）

---

## 1️⃣ 主推：rag-kb（3 个 commit 待 push ✅ 已就绪）

**Remote**: `https://github.com/lalaland472/rag-kb.git`（branch: main）
**待 push 数**: origin/main .. HEAD = **3 commits**

| commit | 内容 |
|--------|------|
| `a7d3686` | **W24 GraphRAG 完整落地**：全套图索引脚本(entities/community_summary/query_global) + data/graph 图谱数据&可视化 + 全局问答 Map-Reduce 基准 + README 补充 GraphRAG 演示章节（+5440 行，18 文件）|
| `9f0f4cd` | **W24 补测 final**：方法二升级四标准(head-to-head 新增 diversity 0-10) — 总分 31.9 vs 23.8(+8.1)，7胜1平负，diversity +3.0 |
| `f12c98b` | **W24 补测**：方法二 head-to-head 三标准盲评(judge=deepseek-v4-pro 关闭 thinking) — 8题全胜，总分 24.6 vs 17.6 |

**本次 W24 更新的关键成果（供 README/commit message/作品集引用）：**
- **GraphRAG 全局问答落地**：脚本 `scripts/graphrag_*.py`（实体图谱抽取→社区摘要→Map-Reduce 全局问答）
- **方法一覆盖度基准**：cover 62% vs flat 45%（+18pt），耗时 15.7s vs 35.4s（减半）
- **方法二 head-to-head 四标准**（忠实/完整/可溯源/多样）：总分 31.9 vs 23.8（+8.1），**7 胜 1 平负**；多样性 +3.0 为最大差距之一；BENCHMARK.md 8.5 节
- **可视化**：`data/graph/graph.html`（交互式实体图谱/社区）+ `graph.png`
- **方法论坑（记录在案）**：deepseek-v4-pro 是推理模型，judge 需关 thinking；弱 judge 有冗长偏好假象

**新增文件清单（已暂存 & commit a7d3686）：**
- `scripts/`: accept_global, graphrag_entities, graphrag_normalize, graphrag_community_summary, graphrag_query_global, graphrag_benchmark, graphrag_head2head, graphrag_demo, graphrag_visualize, graphrag_export_html, graphrag_apply_normalize
- `data/graph/`: entities.json, entities_normalized.json, community_summaries.json, global_benchmark.json, head2head.json, graph.html, graph.png
- `.gitignore`: 新增 `lib/`（第三方前端库不入库）

**⚠️ 未纳入**: 之前的 `lib/` 第三方前端运行时库（可 npm/CDN 恢复，故排除）。

**✔️ 无敏感文件确认**: 已核验 staged 内容无 credentials/*.env/*.pdf/硬编码 key。

---

## 2️⃣ 待办：agent-month7（W25 起，本地已 2 commit，暂无 remote）

本地已有（需在 GitHub 建仓 + 加 remote 后 push）：
- `7f0595e` W25 Day1: function_call.py（标准 Function Calling 骨架）
- `254ae29` W25 Day2: tool_registry.py（@tool 装饰器+反射 schema+参数校验+超时）

**⚠️ 待办动作**: 尚未 add remote（`git remote -v` 为空）。需先建 GitHub 仓库（如 `lalaland472/agent-month7`）再 `git remote add` + push。

---

## 3️⃣ 复核：W17-19 等遗留仓库（均已 push ✅ 0 待 push）

| 仓库 | remote | 待 push |
|------|--------|:---:|
| retrieval-strategies-lab | github.../retrieval-strategies-lab.git | 0 ✅ |
| week19 (longdoc-rag) | github.../longdoc-rag.git | 0 ✅ |
| vector-db-benchmark | github.../vector-db-benchmark.git | 0 ✅ |
| inference-benchmark | *(见下警告)* | 0 ✅ |
| model-eval-lab | *(见下警告)* | 0 ✅ |
| model-serving-api | *(见下警告)* | 0 ✅ |
| quantization-lab | github.../quantization-lab.git | 0 ✅ |

**🚨 安全警告（重要）**: `inference-benchmark` / `model-eval-lab` / `model-serving-api` 三个仓库的 `.git/config` remote URL 里**硬编码了 GitHub token（`ghp_...`）**！这是凭据泄露隐患（任何能读该机器或误 push 远程日志的人都可能看到）。建议：后续把这三个 remote 改成**不带 token** 的 URL（用 ssh 或 credential helper），并去 GitHub **吊销/轮换**这几个已暴露 token。

---

## ✅ 推荐的推送顺序（确认后执行）
1. **rag-kb**: `git push origin main`（3 commits，主成果）
2. **agent-month7**: 建 GitHub 仓 → add remote → push Day1/Day2
3. （修 token 问题）三个 W15-16 仓库 remote 去 token + 轮换凭据
