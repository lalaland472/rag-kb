#!/usr/bin/env python3
"""
GraphRAG 落地 Step ④ —— 全局问答（Reduce 阶段，全局归纳）
用社区摘要回答"面向整个语料库"的全局问题。

流程（Map-Reduce）：
1. Map:  给定全局问题 → 找出最相关的大社区（关键词/语义匹配摘要）
2. Map:  逐个读相关社区摘要 → 生成部分回答
3. Reduce: 汇总部分回答 → 生成最终全局答案

这也正是普通 RAG 失效、而 GraphRAG 擅长的场景。

用法：python3 scripts/graphrag_query_global.py "你的全局问题"
"""
import json, os, re, urllib.request, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES = os.path.join(BASE, "data", "graph", "community_summaries.json")

def load_cfg():
    cfg = {}
    with open(os.path.join(BASE, "config", "credentials.env")) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg
CFG = load_cfg()
API_URL = CFG["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
AUTH = "Bearer " + CFG["LLM_API_KEY"]
MODEL = CFG.get("LLM_MODEL", "deepseek-v4-flash")

def call_llm(prompt, max_tokens=500, temperature=0.2):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

# ── 关键词匹配找相关社区（轻量方案，够用）──
KEYWORD_TOPICS = {
    "retriev": ["RAG", "faiss", "vect"],
    "rag": ["rag"],
    "fine-tun": ["lora", "adapt"],
    "quantiz": ["gptq", "bitwidth", "quant"],
    "attention": ["attention", "transformer", "flashattention"],
    "alignment": ["rlhf", "dpo", "preference", "feedback"],
    "agent": ["agent", "autogen", "react", "tool"],
    "reason": ["reason", "react", "thought", "chain"],
    "memory": ["memory", "generative agents"],
    "serve": ["vllm", "pagedattention", "kv cache", "serving"],
    "measure": ["benchmark", "metric", "evaluate", "judge"],
    "graphrag": ["graphrag", "graph"],
    "embed": ["embedding", "dense", "faiss", "vector"],
    "translate": ["translation", "machine translation"],
    "math": ["math", "python"],
}

def select_communities(question, summaries, top_n=5):
    """按关键词相关性给社区打分，返回 top_n 相关社区。"""
    q = question.lower()
    scored = []
    for cid, c in summaries.items():
        text = " ".join(c["members"]).lower()
        score = 0
        for kw, keys in KEYWORD_TOPICS.items():
            if kw in q:
                for k in keys:
                    if k in text:
                        score += 2
        # 叠加：问题词直接出现在成员里
        for token in re.findall(r"[a-z]{3,}", q):
            if token in text:
                score += 1
        if score > 0:
            scored.append((score, cid))
    scored.sort(reverse=True)
    # 如果关键词没匹配到，退回按社区规模取 top（覆盖最广）
    if not scored:
        scored = sorted(((len(c["members"]), cid) for cid, c in summaries.items()),
                       reverse=True)[:top_n]
    selected = [cid for _, cid in scored[:top_n]]
    return selected

def score_partials(query, partials):
    """给每个 partial 回答打 0-100 的"有帮助度"分数（论文 3.1.6）。

    返回 [(score, partial), ...] 列表，score 为 0 表示内容低质/空话，应被过滤。
    """
    scored = []
    import time
    for i, p in enumerate(partials):
        if not p.strip():
            scored.append((0, p))  # 空 partial 直接 0 分
            continue
        prompt = (
            "你是检索质量评估助手。下面是一个 LLM 针对全局问题生成的「部分回答」，"
            "来源于知识图谱中某个社区摘要。请评估这个部分回答对回答全局问题的「有帮助度」，"
            "打一个 0-100 的分数。\n"
            "评分标准：\n"
            "- 90-100：直接、实质性地提供了问题所需的可用信息\n"
            "- 50-89：提供了一些相关信息，但不够关键或不够具体\n"
            "- 1-49：内容牵强、绕圈子，仅微弱相关或泛泛而谈\n"
            "- 0：完全没有提供任何有用信息（空话、「本社区不包含相关信息」等）\n\n"
            f"全局问题: {query}\n\n"
            f"部分回答: {p}\n\n"
            "只输出一个 0-100 的整数分数，不要输出其他内容。\n"
        )
        score = 0
        # deepseek-v4-flash 在 max_tokens<=50 时会直接返回空，必须给够输出空间
        for _ in range(3):
            try:
                out = call_llm(prompt, max_tokens=120, temperature=0.0).strip()
                # 提取数字（容忍换行/前后解释文字）
                import re as _re
                m = _re.search(r"(\d{1,3})", out)
                if m:
                    score = max(0, min(100, int(m.group(1))))
                    break
            except Exception:
                pass
            time.sleep(1.0)
        scored.append((score, p))
    return scored


def reduce_final(query, partials, max_tokens=700, min_score=20, top_n=None):
    """Reduce: 汇总部分回答，生成结构化全局答案（对齐论文 3.1.6）。

    论文流程：Helpfulness 打分(0-100) → 分数 0 过滤 → 降序 → 迭代塞满 token 上限。
    本实现：打分 → 低于 min_score 过滤 → 降序 → 截断到 top_n(token 预算) → 汇总。

    参数：
      min_score: 低于此分数的 partial 视为低质/空话，丢弃（论文用 0，这里用阈值更稳）
      top_n:     最多保留几个 partial 进入最终 Reduce（对应 token 上限截断）
    """
    if not partials:
        return ""
    # 1. Helpfulness 打分
    scored = score_partials(query, partials)
    # 2. 过滤低质 + 3. 降序
    kept = sorted([(s, p) for s, p in scored if s >= min_score and p.strip()],
                  key=lambda x: x[0], reverse=True)
    # 4. 截断到 token 预算（top_n 相当于在固定窗口下塞满即止）
    if top_n:
        kept = kept[:top_n]
    parts = "\n\n".join(
        f"[主题{i+1}｜帮助度{s}] {p}" for i, (s, p) in enumerate(kept) if p.strip()
    )
    if not parts.strip():
        # 全被过滤：退而求其次保留原始 partial 避免无输出
        parts = "\n\n".join(f"[主题{i+1}] {p}" for i, p in enumerate(partials) if p.strip())
    prompt = (
        "你是知识图谱总结助手。下面是从『社区摘要』得到的对同一全局问题的多个部分回答，"
        "每个代表语料库的一个技术主题，已按帮助度降序排列。请整合成一份连贯、有条理的最终答案，"
        "覆盖主要主题并说明它们之间的关系。用中文，分点。\n\n"
        f"全局问题: {query}\n\n"
        f"各主题的部分回答:\n{parts}\n\n"
        "最终整合答案:\n"
    )
    import time
    for attempt in range(3):
        try:
            out = call_llm(prompt, max_tokens=max_tokens)
            if out.strip():
                return out.strip()
        except Exception:
            pass
        time.sleep(1.5)
    return parts  # 兜底：至少返回部分回答

def map_partial(query, summary):
    """Map: 读单个社区摘要，生成部分回答。（带重试）"""
    prompt = (
        "你是知识图谱问答助手。基于下面这个『社区摘要』（描述语料库中一个技术主题），"
        "回答全局问题。只从这个社区的角度回答，讲清楚该社区与问题的关系。\n\n"
        f"全局问题: {query}\n\n"
        f"社区摘要: {summary}\n\n"
        "部分回答（从本社区角度，1-2句）:\n"
    )
    import time
    for attempt in range(3):
        try:
            out = call_llm(prompt, max_tokens=250)
            if out.strip():
                return out.strip()
        except Exception:
            pass
        time.sleep(1.5)
    return ""

def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/graphrag_query_global.py \"全局问题\"")
        return
    query = sys.argv[1]

    d = json.load(open(SUMMARIES))
    summaries = d["communities"]
    print(f"全局问答 | 模型={MODEL} | 社区摘要 {len(summaries)} 个\n")
    print(f"❓ 全局问题: {query}\n")

    # 1. 选相关社区
    sel = select_communities(query, summaries)
    sel_sorted = sorted(sel, key=lambda c: len(summaries[c]["members"]), reverse=True)
    print(f"🎯 选中 {len(sel_sorted)} 个相关社区: {sel_sorted}")

    # 2. Map：逐社区部分回答
    partials = []
    for i, cid in enumerate(sel_sorted):
        c = summaries[cid]["members"]
        summ = summaries[cid]["summary"]
        print(f"  Map[{i+1}/{len(sel_sorted)}] 社区{cid} ({len(c)}节点)...")
        try:
            p = map_partial(query, summ)
            partials.append(p)
            print(f"    -> {p[:60]}...")
        except Exception as e:
            print(f"    ❌ {e}")

    # 3. Reduce：汇总
    print(f"\n🔄 Reduce：汇总 {len(partials)} 个部分回答 ...")
    final = reduce_final(query, partials)
    print("\n" + "=" * 50)
    print("📋 最终全局答案:")
    print("=" * 50)
    print(final)

if __name__ == "__main__":
    main()
