#!/usr/bin/env python3
"""
GraphRAG 落地 Step ③ —— 社区摘要（Map 阶段）
对每个大社区（>=阈值节点），用 LLM 生成"这个社区是什么主题"的摘要。

- 只处理大社区（小碎片社区无主题意义，忽略）
- 输入：社区成员实体 + 社区内部关系三元组
- 输出：data/graph/community_summaries.json

用法：python3 scripts/graphrag_community_summary.py [--min-size 8]
"""
import json, os, re, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import igraph as ig

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "data", "graph", "entities_normalized.json")
OUT = os.path.join(BASE, "data", "graph", "community_summaries.json")

# ── DeepSeek ──
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

def call_llm(prompt, max_tokens=300, temperature=0.2):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def build_g():
    d = json.load(open(ENT))
    triples = [t for ts in d["documents"].values() for t in ts]
    nodes = sorted({x for t in triples for x in (t[0], t[2])})
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(directed=True); g.add_vertices(len(nodes))
    g.add_edges([(idx[a], idx[b]) for a, b in set((a, b) for a, r, b in triples)])
    part = g.community_leiden(objective_function="modularity")
    return nodes, idx, g, part.membership, triples

def summary_prompt(cid, members, internal_rels):
    mem_str = ", ".join(sorted(members))
    rel_str = "\n".join(f"  {a} --[{r}]--> {b}" for a, r, b in internal_rels[:25])
    return (
        "你是知识图谱社区摘要助手。下面是一个 AI 研究领域知识图谱中的一个『社区』"
        "（一组紧密相关的概念/方法/论文）。\n"
        "请用 2-3 句中文概括：这个社区代表什么技术主题？核心方法/概念是什么？"
        "彼此如何关联？\n"
        "要准确、具体，不要泛泛而谈。\n\n"
        f"社区成员实体: {mem_str}\n\n"
        f"社区内部关系:\n{rel_str}\n\n"
        "社区主题摘要（中文，2-3句）:\n"
    )

def process_community(cid, members, triples):
    member_set = set(members)
    # 社区内部关系 = 两个端点都在社区内的三元组
    internal = [(a, r, b) for a, r, b in triples
                if a in member_set and b in member_set]
    if not internal:
        return cid, None, "无内部关系"
    prompt = summary_prompt(cid, members, internal)
    for attempt in range(3):
        try:
            out = call_llm(prompt)
            if out.strip():
                return cid, out.strip(), None
        except Exception as e:
            last = f"{e}"
        time.sleep(1.5)
    return cid, None, last if 'last' in dir() else "空输出"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=8, help="社区最小节点数才生成摘要")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    nodes, idx, g, mem, triples = build_g()
    n_comm = max(mem) + 1
    comm = defaultdict(list)
    for i, c in enumerate(mem): comm[c].append(nodes[i])

    big = {c: v for c, v in comm.items() if len(v) >= args.min_size}
    print(f"社区 {n_comm} 个 | 大社区(>={args.min_size}) {len(big)} 个将生成摘要")
    print(f"模型: {MODEL}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process_community, c, v, triples): c for c, v in big.items()}
        done = 0
        for fut in as_completed(futs):
            cid, summ, err = fut.result()
            done += 1
            if err or not summ:
                print(f"[{done}/{len(big)}] ❌ 社区{cid}: {err}")
            else:
                results[cid] = {"members": big[cid], "summary": summ}
                print(f"[{done}/{len(big)}] ✅ 社区{cid} ({len(big[cid])}节点): {summ[:45]}...")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"model": MODEL, "min_size": args.min_size,
               "communities": results}, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n✅ 共 {len(results)} 个社区摘要 → {OUT}")

if __name__ == "__main__":
    main()
