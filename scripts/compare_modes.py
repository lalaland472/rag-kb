#!/usr/bin/env python3
"""Week 23 Day 5 — compare_modes.py：LLM-only vs RAG vs RAPTOR 三组对比

目标：量化 RAG 的价值。
  - mode=llm_only : 不检索，纯 LLM 知识回答（基线，暴露幻觉/缺失）
  - mode=flat     : RAG 本地叶子检索 + 生成（精确召回）
  - mode=hybrid   : RAPTOR 摘要混合检索 + 生成（主题召回）

评估维度：
  1. 命中率（期望论文是否被检索/回答提到）
  2. 可溯源性（是否给出可核验的来源引用）
  3. 诚实性（回答是否明说"资料不足"而非瞎编——编造=幻觉扣分）

用法：python3 scripts/compare_modes.py [--quick] [--llm-only-only]
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from test_queries import TEST_SET, load_env
from generate_answer import generate_answer

LLM_ONLY_PROMPT = (
    "你是 AI 论文知识问答助手。请基于你自身对人工智能领域论文的知识回答下面的问题。\n"
    "规则：\n"
    "1. 诚实回答：如果『不确定』或『没有把握』，请明说「我不确定」或「我无法确认」，不要编造。\n"
    "2. 如果回答，尽量提到相关的论文名和作者。\n"
    "3. 用中文回答。\n\n用户问题：{q}"
)


def answer_llm_only(api_key, base_url, model, question, max_tokens=600):
    """纯 LLM 回答（不检索）。"""
    from generate_answer import call_llm
    return call_llm(api_key, base_url, model, LLM_ONLY_PROMPT.format(q=question),
                    max_tokens=max_tokens, temperature=0.3)


def _norm_term(s):
    """归一化术语：下划线/连字符/空格统一为空格，转小写，便于模糊匹配。"""
    import re
    s = s.replace("_", " ").replace("-", " ").lower()
    return re.sub(r"\s+", " ", s).strip()


def eval_answer(answer, expect_terms, cite_docs=None):
    """综合评估一条回答。返回 dict。"""
    answer = answer or ""
    ans_norm = _norm_term(answer)
    # 1. 命中：回答文本里是否出现期望论文名（归一化模糊匹配）
    expect_hits = [t for t in expect_terms if _norm_term(t) in ans_norm]
    # 2. 可溯源：是否带 [n] 引用标记 或 提到了规范论文名
    has_citation = bool(cite_docs) or ("[" in answer and "]" in answer)
    # 3. 诚实：是否承认不足/资料缺失（这是低幻觉信号的正面表现，不是扣分）
    admit_words = ("不确定", "无法确认", "无法回答", "没有把握", "资料不足", "未能", "不能确定", "未完整", "无法完整")
    is_honest = any(w in answer for w in admit_words)
    # 4. 幻觉风险：给出具体细节但命中 0 —— 且**没有**承认不足（即编造了具体内容）
    #    若承认不足（is_honest）则不算幻觉风险（那是诚实标注）。
    hallucination_risk = (len(expect_hits) == 0 and not is_honest and len(answer) > 60)
    return {
        "expect_hits": expect_hits,
        "hit_count": len(expect_hits),
        "has_citation": has_citation,
        "is_honest": is_honest,
        "hallucination_risk": hallucination_risk,
        "answer_excerpt": answer[:150].replace("\n", " "),
    }


def run_compare(question, expect, mode, env, max_tokens=600):
    """单条三模式对比，返回结果 dict。"""
    api_key = env.get("LLM_API_KEY", "")
    base = env.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = env.get("LLM_MODEL", "deepseek-v4-flash")
    expect_terms = [e.strip() for e in expect.split(",") if e.strip()]

    evals = {}
    if mode == "llm_only" or mode == "all":
        t0 = time.time()
        ans = answer_llm_only(api_key, base, model, question, max_tokens)
        evals["llm_only"] = eval_answer(ans, expect_terms)
        evals["llm_only"]["latency"] = round(time.time() - t0, 1)
    if mode == "flat" or mode == "all":
        t0 = time.time()
        res = generate_answer(question, mode="flat", k=5, max_tokens=max_tokens)
        evals["flat"] = eval_answer(res["answer"], expect_terms,
                                    [c["doc_id"] for c in res["meta"]["citations"]])
        evals["flat"]["latency"] = round(time.time() - t0, 1)
        evals["flat"]["cite_docs"] = [c["doc_id"] for c in res["meta"]["citations"]]
    if mode == "hybrid" or mode == "all":
        t0 = time.time()
        res = generate_answer(question, mode="hybrid", k=5, max_tokens=max_tokens)
        evals["hybrid"] = eval_answer(res["answer"], expect_terms,
                                      [c["doc_id"] for c in res["meta"]["citations"]])
        evals["hybrid"]["latency"] = round(time.time() - t0, 1)
        evals["hybrid"]["cite_docs"] = [c["doc_id"] for c in res["meta"]["citations"]]

    return {"question": question, "expect": expect, "evals": evals}


def main():
    mode = "all"
    quick = "--quick" in sys.argv
    if "--llm-only-only" in sys.argv:
        mode = "llm_only"
    if quick and mode == "all":
        mode = "hybrid"  # quick 模式只跑混合，看整体

    env = load_env()
    print(f"三组对比: LLM-only vs RAG(flat) vs RAPTOR(hybrid) | "
          f"模式={mode} | {len(TEST_SET)} 条")
    print("=" * 72)

    results = []
    for i, (qid, cat, question, expect) in enumerate(TEST_SET, 1):
        print(f"[{i:2d}] {qid} {question[:28]}...")
        try:
            r = run_compare(question, expect, mode, env)
            results.append(r)
            for m, ev in r["evals"].items():
                risk = " ⚠️幻觉" if ev["hallucination_risk"] else ""
                hon = " 诚实✓" if ev["is_honest"] else ""
                print(f"      {m:8s} 命中{ev['hit_count']}/{len(ev['expect_hits'] + [])}  "
                      f"引用{'✓' if ev['has_citation'] else '✗'}{hon}{risk}  "
                      f"{ev['latency']}s  {ev['answer_excerpt'][:60]}")
        except Exception as e:
            print(f"      ⚠️ 异常: {e}")

    # 汇总
    print("=" * 72)
    summaries = {"llm_only": [], "flat": [], "hybrid": []}
    for r in results:
        for m, ev in r["evals"].items():
            if m in summaries:
                summaries[m].append(ev)

    print(f"\n{'模式':<10} {'平均命中':<10} {'可溯源':<8} {'诚实':<6} {'幻觉风险':<8} {'耗时':<6}")
    for m in ("llm_only", "flat", "hybrid"):
        evs = summaries.get(m, [])
        if not evs:
            continue
        n = len(evs)
        avg_hit = sum(e["hit_count"] for e in evs) / n
        src = sum(e["has_citation"] for e in evs) / n
        hon = sum(e["is_honest"] for e in evs) / n
        hal = sum(e["hallucination_risk"] for e in evs) / n
        lat = sum(e.get("latency", 0) for e in evs) / n
        print(f"{m:<10} {avg_hit:.1f}/条   {src:.0%}     {hon:.0%}    {hal:.0%}   {lat:.1f}s")

    # 存档
    out = os.path.join(BASE, "..", "data", "w23_compare_modes.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
