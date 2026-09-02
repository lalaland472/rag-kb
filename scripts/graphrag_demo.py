#!/usr/bin/env python3
"""
W24 Day 3 — graphrag_demo.py：GraphRAG 专项演示脚本

与 demo.py（RAG flat/hybrid）互补，集中展示「面向整个语料库」的全局归纳能力。
GraphRAG 的核心价值：普通 RAG 检索片段 → 只见树木；图谱 + 社区摘要 + Map-Reduce → 俯瞰森林。

演示四幕：
  Act 0. 图谱概览       实体图谱规模（节点/边/社区）+ 可视化入口
  Act 1. 社区地图       语料库被拆成哪些技术社区（从摘要自动归纳）
  Act 2. 全局问答对比   GraphRAG vs RAG(flat) 同屏对比，展示覆盖差异
  Act 3. Map-Reduce 透明 选中哪些社区 → 逐社区部分回答 → 汇总，全程可见

用法：
  python3 scripts/graphrag_demo.py             # 完整四幕
  python3 scripts/graphrag_demo.py --quick     # 只跑 Act 2（问答对比），省时
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

import graphrag_query_global as gq

GRAPH_HTML = os.path.join(BASE, "data", "graph", "graph.html")
SUMMARIES = os.path.join(BASE, "data", "graph", "community_summaries.json")
ENTITIES = os.path.join(BASE, "data", "graph", "entities_normalized.json")

# ── 标志性全局问题（"整个语料库"级别，正是 GraphRAG 主场）──
GLOBAL_QUESTIONS = [
    ("Q1", "这个语料库整体讲了哪些 AI 技术主题？它们之间是什么关系？"),
    ("Q2", "检索增强生成(RAG)这个方向经历了怎样的演进？有哪些变体？"),
    ("Q3", "大模型对齐（让模型符合人类偏好）的主流方法有哪些？"),
    ("Q4", "从组织知识的角度，语料库覆盖了哪些方式？它们各有何优劣？"),
]


def _load():
    sums = json.load(open(SUMMARIES))["communities"]
    ent = json.load(open(ENTITIES))
    return sums, ent


# ── Act 0：图谱概览 ──
def act0(sums, ent):
    print("=" * 68)
    print("🎬 Act 0 · 知识图谱概览")
    print("=" * 68)
    triples = [t for ts in ent["documents"].values() for t in ts]
    nodes = sorted({x for t in triples for x in (t[0], t[2])})
    edges = list({(a, b) for a, r, b in triples})
    print(f"  📊 实体节点: {len(nodes)} 个")
    print(f"  🔗 关系边:   {len(edges)} 条")
    print(f"  🏘️  社区:     {len(sums)} 个技术主题")
    print(f"  📄 论文:      {len(ent['documents'])} 篇（20+ 语料库）")
    if os.path.exists(GRAPH_HTML):
        print(f"  🌐 可视化:   data/graph/graph.html （浏览器打开，可缩放/拖拽/点节点）")
    print()


# ── Act 1：社区地图 ──
def act1(sums):
    print("=" * 68)
    print("🎬 Act 1 · 知识社区地图（语料库被拆成这些技术主题）")
    print("=" * 68)
    for cid in sorted(sums, key=lambda x: int(x[0])):
        c = sums[cid]
        members = c["members"]
        # 用摘要首句 + 成员代表词概括
        summ = c["summary"].replace("\n", " ")
        core = members[0] if members else ""
        extra = f"+{len(members)-1}" if len(members) > 1 else ""
        print(f"  [C{cid}] {core}{extra} → {summ[:52]}...")
    print()


# ── Act 2：全局问答对比 ──
def _answer_flat(q):
    from generate_answer import generate_answer
    res = generate_answer(q, mode="flat", k=5, max_tokens=400)
    return res["answer"]


def _answer_graphrag(q, sums):
    sel = sorted(gq.select_communities(q, sums),
                 key=lambda c: len(sums[c]["members"]), reverse=True)
    partials = []
    for cid in sel:
        p = gq.map_partial(q, sums[cid]["summary"])
        if p.strip():
            partials.append(p)
    final = gq.reduce_final(q, partials) if partials else ""
    return final, sel


def act2(sums, quick):
    print("=" * 68)
    print("🎬 Act 2 · 全局问答：GraphRAG vs RAG(flat) 覆盖度对比")
    print("=" * 68)
    for qid, q in GLOBAL_QUESTIONS:
        print(f"\n{'─'*68}\n❓ {qid} | {q}\n{'─'*68}")
        # flat
        try:
            t0 = time.time()
            af = _answer_flat(q)
            lf = time.time() - t0
            print(f"\n  📖 RAG(flat)   ({lf:.1f}s):")
            print(f"  {af[:300]}")
        except Exception as e:
            print(f"\n  ⚠️ flat 失败: {e}")
        # graphrag
        try:
            t0 = time.time()
            ag, sel = _answer_graphrag(q, sums)
            lg = time.time() - t0
            print(f"\n  🕸️ GraphRAG   ({lg:.1f}s, 选中 {len(sel)} 社区 {sel}):")
            print(f"  {ag[:300]}")
        except Exception as e:
            print(f"\n  ⚠️ graphrag 失败: {e}")
        print()


# ── Act 3：Map-Reduce 透明展示 ──
def act3(sums):
    print("=" * 68)
    print("🎬 Act 3 · Map-Reduce 如何工作（透明展示 Q2）")
    print("=" * 68)
    q = GLOBAL_QUESTIONS[1][1]  # Q2 RAG 演进
    sel = sorted(gq.select_communities(q, sums),
                 key=lambda c: len(sums[c]["members"]), reverse=True)
    print(f"\n  ❓ 全局问题: {q}\n")
    print(f"  🌍 第1步 Map(选社区): 按关键词/语义相关性选中 {len(sel)} 个社区")
    for cid in sel:
        core = sums[cid]["members"][0] if sums[cid]["members"] else "?"
        print(f"     • C{cid} ({core} 等 {len(sums[cid]['members'])} 节点)")
    print(f"\n  🔀 第2步 Map(逐社区答): 每个社区生成一个「部分回答」，只讲本社区与问题的关系")
    partials = []
    for cid in sel:
        p = gq.map_partial(q, sums[cid]["summary"])
        partials.append(p)
        print(f"     C{cid} → {p[:70]}...")
    print(f"\n  🧩 第3步 Reduce(汇总): 把 {len(partials)} 个部分回答整合成一份连贯全局答案")
    final = gq.reduce_final(q, partials)
    print(f"     → {final[:280]}...")
    print()


def main():
    quick = "--quick" in sys.argv
    sums, ent = _load()
    print("🕸️  RAG-KB · GraphRAG 专项演示（面向整个语料库的全局问答）")
    print(f"    模型={gq.MODEL} | 社区 {len(sums)} 个 | 快速模式={'开' if quick else '关'}")
    act0(sums, ent)
    if quick:
        act2(sums, quick)
    else:
        act1(sums)
        act2(sums, quick)
        act3(sums)
    print("\n✅ 演示完成。GraphRAG 核心卖点：普通 RAG 检索「片段」，GraphRAG 归纳「森林」。")
    if os.path.exists(GRAPH_HTML):
        print(f"🌐 配合可视化图谱一起看: data/graph/graph.html")


if __name__ == "__main__":
    main()
