#!/usr/bin/env python3
"""W24 Day 3 — demo.py：RAG-KB 演示脚本

跑 5 个代表性 query，展示 RAG-KB 的核心能力：
  A. 精确术语查询（flat）
  B. 中文口语查询（HyDE 效果）
  C. 跨篇主题查询（hybrid）
  D. 比较类查询（multi-doc）
  E. 严格模式（Self-RAG IsSup）

用法：
  python3 scripts/demo.py            # 跑全部 5 个演示
  python3 scripts/demo.py --quick    # 每个只展示提问+回答，不显示引用细节
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from generate_answer import generate_answer

DEMOS = [
    # (标题, 问题, 模式, 风格, 说明)
    ("A. 精确术语查询 (flat)", "Self-RAG 的反思 token 有哪两类", "flat", "default",
     "展示精确句子召回 + 引用"),
    ("B. 中文口语查询 (HyDE)", "怎么让 AI 在回答前先搜索资料、不要瞎编", "flat", "chat",
     "展示 HyDE 把中文口语翻译成学术英文检索"),
    ("C. 跨篇主题查询 (hybrid)", "RAPTOR 和 GraphRAG 在组织知识上有什么不同", "hybrid", "chat",
     "展示 RAPTOR 摘要融合的多论文召回"),
    ("D. 比较类查询 (multi-doc)", "DPO 和 RLHF 有什么区别", "flat", "chat",
     "展示跨论文比较 + 引用"),
    ("E. 严格模式 (Self-RAG IsSup)", "什么是 embedding 向量化", "flat", "strict",
     "展示每句有依据 + 忠实核对"),
]


def run_demo(item, show_cites):
    title, question, mode, style, note = item
    print("\n" + "=" * 60)
    print(f"🎬 {title}")
    print(f"   问题: {question}   [mode={mode}, style={style}]")
    print(f"   目标: {note}")
    print("=" * 60)
    t0 = time.time()
    res = generate_answer(question, mode=mode, k=5, style=style)
    dt = time.time() - t0
    print(f"\n📝 回答 ({dt:.1f}s):")
    print(res["answer"])
    if show_cites and res["meta"]["citations"]:
        print("\n📚 引用来源:")
        for c in res["meta"]["citations"]:
            if c["chunk_id"] is not None:
                loc = f"第{c['page'] + 1}页" if c.get("page") is not None else f"块{c['chunk_id']}"
            else:
                loc = "(摘要)"
            print(f"  [{c['index']}] {c['display']}  [{loc}]")
    print()


def main():
    quick = "--quick" in sys.argv
    show_cites = not quick
    print("🎥 RAG-KB 演示脚本")
    print("=" * 60)
    print(f"共 {len(DEMOS)} 个演示 | 引用显示={'开' if show_cites else '关'}")
    for d in DEMOS:
        try:
            run_demo(d, show_cites)
        except Exception as e:
            print(f"⚠️ 演示失败: {e}\n")


if __name__ == "__main__":
    main()
