#!/usr/bin/env python3
"""
GraphRAG 落地 —— 实体规范化 应用层
把人工确认的合并映射表应用到 entities.json 的所有三元组。

映射规则分两层：
- 安全规则层：论文尾缀去重、复数/大小写变体（机器可安全处理）
- 人工决策层：跨词同概念（RAG/Transformer/GPT 家族），用户逐组拍板

输出：data/graph/entities_normalized.json（三元组已用规范实体重写）
"""
import json, os, re
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "data", "graph", "entities.json")
OUT = os.path.join(BASE, "data", "graph", "entities_normalized.json")

# ── 人工决策层：跨词同概念合并（用户 3 个决策）──
MANUAL_MAP = {
    # 决策1：RAG paradigm 归 RAG，但 RAG 变体保留独立
    "RAG paradigm": "RAG",
    "RAG_Lewis": "RAG",
    "RAG_Survey": "RAG",
    # 决策2：Attention Is All You Need 归 Transformer
    "Attention Is All You Need": "Transformer",
    # 决策3：GPT 通用叫法归 GPT，代际保留
    "GPT models": "GPT",
    "OpenAI GPT": "GPT",
    # 论文名尾缀 → 概念名
    "GPTQ_2023": "GPTQ",
    "LoRA_2021": "LoRA",
    "vLLM_2023": "vLLM",
    "MT_Bench": "MT-Bench",
    "Tree_of_Thoughts": "Tree of Thoughts",
    # 大小写统一
    "generative agents": "Generative Agents",
}

# ── 安全规则：复数/拼写变体（机器处理，不误伤）──
def safe_rules(e):
    e = e.strip()
    # 去论文尾缀 _YYYY
    e = re.sub(r'_\d{4,}$', '', e)
    # 精确复数映射（白名单，不盲切）
    PLURAL = {
        "LLMs": "LLM", "memories": "memory", "languages": "language",
        "tasks": "task", "tools": "tool", "experiences": "experience",
        "entities": "entity",
        "conversable agents": "conversable agent",
        "chat assistants": "chat assistant",
        "human inputs": "human input",
        "LLM applications": "LLM application",
        "multiple agents": "multi-agent",
        "language models": "large language model",
        "pre-trained language models": "pre-trained language model",
        "long-context language models": "long-context language model",
        "recurrent neural networks": "recurrent neural network",
        "operating systems": "operating system",
        "human demonstrations": "human demonstration",
        "high-dimensional features": "high-dimensional feature",
        "traditional benchmarks": "benchmark",
        "cloze tasks": "cloze task",
        "task-specific datasets": "task-specific dataset",
        "task-specific architectures": "task-specific architecture",
        "pre-trained model weights": "pre-trained model weights",
    }
    if e in PLURAL:
        return PLURAL[e]
    return e

def normalize(e):
    e = safe_rules(e)          # 安全层先做
    return MANUAL_MAP.get(e, e)  # 人工层覆盖

def main():
    d = json.load(open(ENT))
    docs = d["documents"]

    # 统计映射效果
    applied = defaultdict(int)
    total_triples = 0
    normalized = {}
    for doc_id, triples in docs.items():
        new_ts = []
        for a, r, b in triples:
            na, nb = normalize(a), normalize(b)
            if na != a: applied[a] += 1
            if nb != b: applied[b] += 1
            total_triples += 1
            new_ts.append((na, r, nb))
        normalized[doc_id] = new_ts

    print(f"总三元组: {total_triples}")
    print(f"被规范化改写的实体: {len(applied)}")

    # 收集规范化后的唯一实体
    ents = set()
    for ts in normalized.values():
        for a, r, b in ts: ents.add(a); ents.add(b)

    out = {"model": d["model"], "documents": normalized,
           "errors": d.get("errors", {}), "entity_map": MANUAL_MAP}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)

    print(f"\n唯一实体: {len(ents)} (规范化前应高于此)")
    print(f"\n=== 实际应用的映射 ===")
    for src, cnt in sorted(applied.items(), key=lambda x: -x[1]):
        print(f"  {src} → {normalize(src)}  ({cnt} 处)")

    print(f"\n✅ 已保存 → {OUT}")

if __name__ == "__main__":
    main()
