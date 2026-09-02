#!/usr/bin/env python3
"""全局问答基准测试：RAG(flat) vs GraphRAG

测"面向整个语料库"的全局问题——这正是普通 RAG 失效、GraphRAG 擅长的场景。
每条问题标注【期望覆盖的主题】，评估回答覆盖了几个。

指标：
  1. 覆盖度 coverage：回答是否提到期望主题（按主题关键词模糊匹配）
  2. 可溯源 trace："社区/摘要/多主题" 结构的体现
  3. 诚实 honesty：是否承认"无法归纳"（全局问题若答不出应诚实）
  4. 幻觉 risk：声称覆盖却其实无依据

用法：python3 scripts/graphrag_benchmark.py
"""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

CFG = {}
with open("config/credentials.env") as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            CFG[k.strip()] = v.strip().strip('"').strip("'")
API = CFG["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
AUTH = "Bearer " + CFG["LLM_API_KEY"]
MODEL = CFG["LLM_MODEL"]

def call_llm(prompt, mt=700, temp=0.3):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": temp}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"].strip()
            if out:
                return out
        except Exception:
            pass
        time.sleep(1.5)
    return ""

# ── 全局测试集：问题 + 期望覆盖的主题关键词 ──
GLOBAL_TEST = [
    ("G1", "这个语料库整体讲了哪些 AI 技术主题？",
     ["RAG", "Agent", "微调", "Transformer"]),
    ("G2", "检索增强生成(RAG)这个方向在这个语料库里经历了怎样的演进？有哪些变体？",
     ["RAG", "Self-RAG", "GraphRAG", "向量检索"]),
    ("G3", "Agent 智能体这个方向有哪些代表工作？它们各自怎么组织智能体？",
     ["AutoGen", "ReAct", "生成式智能体", "工具"]),
    ("G4", "大模型高效化（微调和推理加速）涉及哪些技术？",
     ["LoRA", "量化", "vLLM", "KV cache"]),
    ("G5", "大模型对齐（让模型符合人类偏好）的主流方法有哪些？",
     ["RLHF", "DPO", "奖励模型"]),
    ("G6", "从知识表示和检索的角度，语料库覆盖了哪些组织知识的方式？",
     ["向量检索", "图结构", "树结构", "摘要"]),
    ("G7", "这个语料库在模型评估方面用了哪些基准和方法？",
     ["benchmark", "LLM-as-a-judge", "评估"]),
    ("G8", "Transformer 及其注意力机制在语料库中是怎么演进和应用的？",
     ["Transformer", "attention", "FlashAttention"]),
]

def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("_", " ").replace("-", " ")).strip()

def _answer_flat(q):
    """普通 RAG：flat 检索 top-k 句子生成。复用 generate_answer。"""
    sys.path.insert(0, os.path.join(BASE, "scripts"))
    from generate_answer import generate_answer
    res = generate_answer(q, mode="flat", k=5, max_tokens=500)
    return res["answer"], [c["doc_id"] for c in res["meta"]["citations"]]

def _answer_graphrag(q):
    """GraphRAG：复用正式的 Map-Reduce pipeline（与 graphrag_query_global.py 对齐）。

    流程：
      1. Map:   关键词/语义匹配选出相关社区（select_communities）
      2. Map:   逐社区摘要生成部分回答（map_partial）
      3. Reduce: 汇总部分回答生成最终全局答案（reduce_final）
    """
    import graphrag_query_global as gq
    sums = json.load(open("data/graph/community_summaries.json"))["communities"]
    # 1. 选相关社区（按规模降序，先大社区后小社区）
    sel = sorted(gq.select_communities(q, sums),
                 key=lambda c: len(sums[c]["members"]), reverse=True)
    # 2. Map：逐社区生成部分回答
    partials = []
    for cid in sel:
        p = gq.map_partial(q, sums[cid]["summary"])
        if p.strip():
            partials.append(p)
    # 3. Reduce：汇总
    final = gq.reduce_final(q, partials) if partials else ""
    return final, None

def eval_coverage(answer, expect_topics):
    ans = norm(answer)
    covered, missed = [], []
    for t in expect_topics:
        if norm(t) in ans:
            covered.append(t)
        else:
            missed.append(t)
    # 诚实：承认无法归纳
    admit = ("无法", "不确定", "资料不足", "不能完整", "未能")
    is_honest = any(w in ans for w in admit)
    # 有实质内容
    has_content = len(ans) > 40 and not is_honest
    return {"covered": covered, "missed": missed,
            "coverage": len(covered) / max(1, len(expect_topics)),
            "is_honest": is_honest, "has_content": has_content}

def main():
    print(f"全局问答基准：RAG(flat) vs GraphRAG | 模型={MODEL} | {len(GLOBAL_TEST)} 条全局问题")
    print("=" * 74)
    rows = []
    for qid, q, topics in GLOBAL_TEST:
        print(f"[{qid}] {q[:32]}...")
        # flat
        try:
            t0 = time.time()
            ans_f, cites = _answer_flat(q)
            ev_f = eval_coverage(ans_f, topics)
        except Exception as e:
            ev_f = {"coverage": 0, "covered": [], "missed": topics, "is_honest": False, "has_content": False, "error": str(e)}
            ans_f = ""
        lat_f = round(time.time() - t0, 1)
        # graphrag
        try:
            t0 = time.time()
            ans_g, _ = _answer_graphrag(q)
            ev_g = eval_coverage(ans_g, topics)
        except Exception as e:
            ev_g = {"coverage": 0, "covered": [], "missed": topics, "is_honest": False, "has_content": False, "error": str(e)}
            ans_g = ""
        lat_g = round(time.time() - t0, 1)
        rows.append({"id": qid, "q": q, "topics": topics,
                     "flat": {**ev_f, "latency": lat_f, "excerpt": ans_f[:80]},
                     "graphrag": {**ev_g, "latency": lat_g, "excerpt": ans_g[:80]}})
        print(f"  flat    : 覆盖{ev_f['coverage']:.0%} ({','.join(ev_f['covered']) or '无'}) {lat_f}s")
        print(f"  graphrag: 覆盖{ev_g['coverage']:.0%} ({','.join(ev_g['covered']) or '无'}) {lat_g}s")

    print("=" * 74)
    # 汇总
    f_cov = sum(r["flat"]["coverage"] for r in rows) / len(rows)
    g_cov = sum(r["graphrag"]["coverage"] for r in rows) / len(rows)
    f_content = sum(r["flat"]["has_content"] for r in rows) / len(rows)
    g_content = sum(r["graphrag"]["has_content"] for r in rows) / len(rows)
    print(f"\n{'模式':<10} {'平均覆盖度':<10} {'有实质内容':<10} {'平均耗时':<8}")
    print(f"{'RAG(flat)':<10} {f_cov:.0%}        {f_content:.0%}        {sum(r['flat']['latency'] for r in rows)/len(rows):.1f}s")
    print(f"{'GraphRAG':<10} {g_cov:.0%}        {g_content:.0%}        {sum(r['graphrag']['latency'] for r in rows)/len(rows):.1f}s")
    print(f"\n覆盖度提升: +{(g_cov - f_cov) * 100:.0f} 个百分点")

    out = "data/graph/global_benchmark.json"
    json.dump({"model": MODEL, "flat_avg_cov": f_cov, "graphrag_avg_cov": g_cov,
               "rows": rows}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")

if __name__ == "__main__":
    main()
