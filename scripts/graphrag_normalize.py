#!/usr/bin/env python3
"""
GraphRAG 落地 —— 实体规范化 Step ②a
安全规则层（不误伤）+ LLM 二次聚合生成合并映射表

两步：
1. 安全规则：只去掉 _YYYY 论文尾缀（GPTQ_2023->GPTQ），不盲切复数
2. LLM 聚合：把剩余实体丢给 DeepSeek，让它找出"应合并成同一概念"的组
   （RAG / RAG_Lewis / RAG_Survey -> RAG）输出 {实体: 规范名} 映射表

用法：
    python3 scripts/graphrag_normalize.py [--dry-run] [--out data/graph/entity_map.json]
"""
import argparse, json, os, re, urllib.request, time
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "data", "graph", "entities.json")

# ── DeepSeek 配置 ──
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

def call_llm(prompt, max_tokens=1500, temperature=0.1):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

# ── 安全规则层（只去论文名尾缀，绝不盲切）──
SAFE_SUFFIX_MAP = {}  # 现成的论文名尾缀映射
def safe_normalize(ent):
    e = ent.strip()
    # 只处理明确的 "_论文名/年份" 尾缀模式：RAG_Lewis_2020, GPTQ_2023, LoRA_2021
    e2 = re.sub(r'_\d{4,}$', '', e)   # _2020 _2023
    # "_Lewis" "_ReAct" 这种作者/简称尾缀：保留（LLM 层处理跨词合并）
    return e2

def collect_entities():
    d = json.load(open(ENT))
    docs = d["documents"]
    triples = [t for ts in docs.values() for t in ts]
    ents = set()
    for a, r, b in triples:
        ents.add(safe_normalize(a)); ents.add(safe_normalize(b))
    return ents, triples

# ── LLM 聚合 ──
def llm_cluster(ents):
    ent_list = sorted(ents)
    # 分批，每批 ~120 个，避免超上下文
    BATCH = 120
    merged = {}  # 实体 -> 规范名
    for i in range(0, len(ent_list), BATCH):
        batch = ent_list[i:i+BATCH]
        prompt = (
            "你是知识图谱实体规范化助手。下面是 AI 论文知识库图谱的实体列表。\n"
            "任务：找出【应该合并成同一个概念】的实体组（即同义/同概念的不同叫法，如 'RAG' 与 'RAG paradigm' 与 'RAG_Survey'；"
            "'Transformer' 与 'Attention Is All You Need'）。\n"
            "规则：\n"
            "- 只合并 明确同义的。不要合并层次不同的（如 'RAG' 与 'Advanced RAG' 若你认为它们是子类则应保留）。\n"
            "- 对每一组，选一个最标准、最通用的名字作为『规范名』。\n"
            "- 输出格式：每行一组，用竖线分隔：实体名|规范名\n"
            "  例如：RAG paradigm|RAG\n"
            "- 无法归并到任何组的实体不要输出。只输出需要合并的组。\n\n"
            "实体列表：\n" + ", ".join(batch) + "\n\n合并规则（每行 实体|规范名）:\n"
        )
        print(f"  批次 {i//BATCH+1} ({len(batch)} 实体)...")
        for attempt in range(3):
            try:
                out = call_llm(prompt)
                for line in out.splitlines():
                    line = line.strip()
                    if "|" in line:
                        src, dst = [x.strip() for x in line.split("|", 1)]
                        if src in ents and src != dst:
                            merged[src] = dst
                break
            except Exception as e:
                print(f"    retry {attempt+1}: {str(e)[:60]}")
                time.sleep(2)
        time.sleep(0.5)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只输出分析，不写映射表")
    ap.add_argument("--out", default="data/graph/entity_map.json")
    args = ap.parse_args()

    ents, triples = collect_entities()
    print(f"安全规范化后唯一实体: {len(ents)}")

    print("\n[1/2] 安全规则层：去论文尾缀")
    # 统计尾缀清理的收益
    d = json.load(open(ENT))
    raw_ents = set()
    for ts in d["documents"].values():
        for a, r, b in ts: raw_ents.add(a); raw_ents.add(b)
    cleaned = sum(1 for e in raw_ents if safe_normalize(e) != e)
    print(f"  清理论文尾缀 {cleaned} 个")

    print("\n[2/2] LLM 聚合：生成合并映射表 ...")
    merged = llm_cluster(ents)
    print(f"  LLM 建议合并 {len(merged)} 个实体")

    # 应用映射后的实体数
    final = set()
    for e in ents:
        final.add(merged.get(e, e))
    print(f"  合并后唯一实体: {len(final)} (原 {len(ents)})")

    print("\n=== LLM 合并映射明细 ===")
    from collections import defaultdict as dd
    by_dst = dd(list)
    for src, dst in merged.items(): by_dst[dst].append(src)
    for dst, srcs in sorted(by_dst.items()):
        print(f"  {dst}  ←  {sorted(srcs)}")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"map": merged}, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n✅ 映射表已保存 → {args.out}")
    else:
        print("\n(dry-run，未写入)")

if __name__ == "__main__":
    main()
