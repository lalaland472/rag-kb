#!/usr/bin/env python3
"""Week 23 Day 4 — test_queries.py：20 条 query 端到端测试

覆盖策略（20 条，跨 5 类）：
  A. 精确术语查询（Flat 主场）      — 4 条
  B. 中文口语/描述型查询（HyDE 主场）— 5 条
  C. 跨篇/主题性查询（Hybrid 主场）  — 4 条
  D. 比较/关系类查询                — 4 条
  E. 边界/容错查询                  — 3 条

用法：
  python3 scripts/test_queries.py               # 跑全部 20 条
  python3 scripts/test_queries.py --quick       # 每条只验证召回，不生成（快）
  python3 scripts/test_queries.py --mode hybrid # 强制用 hybrid 检索引擎
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

TEST_SET = [
    # (编号, 类别, 问题, 期望相关论文关键词)
    # A. 精确术语查询（Flat 主场）
    ("A1", "A", "What is the scaled dot-product attention formula in Transformer?", "Attention"),
    ("A2", "A", "How does FAISS accelerate billion-scale similarity search on GPUs?", "FAISS"),
    ("A3", "A", "What is FlashAttention and how does it reduce memory?", "FlashAttention"),
    ("A4", "A", "Explain the KV cache and paged attention in vLLM serving.", "vLLM"),
    # B. 中文口语/描述型查询（HyDE 主场）
    ("B1", "B", "LoRA 是怎么减少微调参数的？", "LoRA"),
    ("B2", "B", "RAG 检索增强生成是什么意思，怎么工作的？", "RAG"),
    ("B3", "B", "哪些论文解决了大模型说假话、对齐用户意图的问题？", "InstructGPT,DPO"),
    ("B4", "B", "怎么让 AI 在回答前先搜索资料、不要瞎编？", "Self_RAG,RAG,ReAct"),
    ("B5", "B", "树状检索和递归摘要是什么，哪篇论文提的？", "RAPTOR"),
    # C. 跨篇/主题性查询（Hybrid 主场）
    ("C1", "C", "有哪几篇论文是关于大模型高效微调和参数压缩的？", "LoRA,GPTQ,AWQ"),
    ("C2", "C", "关于让多个 AI 智能体协作完成任务的论文", "AutoGen,Generative_Agents"),
    ("C3", "C", "关于模型评估和 LLM 作为裁判的进展", "MT_Bench"),
    ("C4", "C", "检索增强有哪些高级策略，比如假设文档和融合？", "HyDE,RAPTOR"),
    # D. 比较/关系类查询
    ("D1", "D", "RAPTOR 和 GraphRAG 在组织知识上有什么不同？", "RAPTOR,GraphRAG"),
    ("D2", "D", "BERT 和 Transformer 有什么关系？", "BERT,Attention"),
    ("D3", "D", "DPO 和传统的 RLHF 有什么区别？", "DPO,InstructGPT"),
    ("D4", "D", "Self-RAG 和 ReAct 都涉及检索与推理，区别在哪？", "Self_RAG,ReAct"),
    # E. 边界/容错查询
    ("E1", "E", "哪篇论文讲人类记忆的模拟？", "Generative_Agents"),
    ("E2", "E", "思维树在24点游戏上的表现怎么样？", "Tree_of_Thoughts"),
    ("E3", "E", "请解释一下什么是 embedding 向量化", "BERT,FAISS"),
]


def load_env():
    env = {}
    path = os.path.join(BASE, "..", "config", "credentials.env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def run_one(item, mode, use_generation):
    """跑单条 query，返回结果 dict。"""
    qid, cat, question, expect = item
    from generate_answer import generate_answer
    t0 = time.time()
    res = generate_answer(question, mode=mode, k=5)
    latency = time.time() - t0
    answer = res["answer"]
    cites = res["meta"]["citations"]
    cite_docs = [c["doc_id"] for c in cites]
    has_page = any(c.get("page") is not None for c in cites)

    # 召回命中判定：cite_docs 里是否出现期望论文（模糊匹配）
    expect_terms = [e.strip() for e in expect.split(",") if e.strip()]
    hit_terms = [t for t in expect_terms
                 if any(t.lower() in d.lower() for d in cite_docs)]

    # 回答是否有实质内容（排除"无法回答"类）
    empty_words = ("无法完整回答", "根据提供的资料无法", "未检索到", "没有相关", "无法回答")
    has_content = not any(w in answer for w in empty_words)

    return {
        "id": qid, "cat": cat, "question": question,
        "expect": expect, "cite_docs": cite_docs,
        "hit_terms": hit_terms, "hit_ratio": len(hit_terms) / len(expect_terms),
        "has_page": has_page, "has_content": has_content,
        "answer_excerpt": answer[:120].replace("\n", " "),
        "latency": round(latency, 1),
    }


def main():
    quick = "--quick" in sys.argv
    force_mode = None
    if "--mode" in sys.argv:
        force_mode = sys.argv[sys.argv.index("--mode") + 1]

    mode = force_mode or "flat"
    use_gen = not quick
    env = load_env()
    print(f"RAG-KB 端到端测试 | {len(TEST_SET)} 条 | mode={mode} | 生成={'开' if use_gen else '关(仅召回)'}")
    print("=" * 70)

    results = []
    for i, item in enumerate(TEST_SET, 1):
        qid, cat, question, expect = item
        try:
            r = run_one(item, mode, use_gen)
            results.append(r)
            hit = "✓" if r["hit_ratio"] == 1.0 and r["has_content"] else "△" if r["hit_ratio"] > 0 else "✗"
            content = "" if r["has_content"] else " [无内容]"
            print(f"[{i:2d}] {hit} {qid} {cat} | 命中{r['hit_ratio']:.0%} "
                  f"| 页码{'✓' if r['has_page'] else '✗'} | {r['latency']}s")
            if r["hit_ratio"] < 1.0:
                print(f"     期望[{r['expect']}] 实际[{','.join(r['cite_docs'][:4])}]{content}")
        except Exception as e:
            print(f"[{i:2d}] ✗ {qid} {cat} | 异常: {e}")
            results.append({"id": qid, "cat": cat, "question": question, "error": str(e)})

    # 汇总
    print("=" * 70)
    ok = [r for r in results if r.get("hit_ratio") == 1.0 and r.get("has_content")]
    partial = [r for r in results if 0 < r.get("hit_ratio", 0) < 1.0 or (r.get("hit_ratio") == 1.0 and not r.get("has_content"))]
    fail = [r for r in results if r.get("hit_ratio", 0) == 0 and "error" not in r]
    err = [r for r in results if "error" in r]
    avg_lat = sum(r.get("latency", 0) for r in results if "latency" in r) / max(1, len([r for r in results if "latency" in r]))

    print(f"✅ 完全命中+有内容: {len(ok)}/{len(results)}")
    print(f"△  部分/内容缺失: {len(partial)}")
    print(f"✗  未命中: {len(fail)}")
    print(f"⚠  异常: {len(err)}")
    print(f"⏱  平均耗时: {avg_lat:.1f}s")

    # 写报告
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "mode": mode, "total": len(results),
        "full_hit": len(ok), "partial": len(partial), "fail": len(fail), "error": len(err),
        "avg_latency": round(avg_lat, 1),
        "results": results,
    }
    out = os.path.join(BASE, "..", "data", f"w23_test_report_{mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out}")
    return report


if __name__ == "__main__":
    main()
