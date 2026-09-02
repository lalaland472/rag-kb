#!/usr/bin/env python3
"""
GraphRAG 落地 Step ① —— 实体+关系抽取（DeepSeek API 版）
从 chunks 按论文聚合文本，用 API LLM 抽取 (实体A, 关系, 实体B) 三元组。

策略：
- 按论文聚合，每篇调用一次 API（23 篇 = 23 次调用）
- 输出格式：每行一条  A|关系|B ，regex 兜底解析（对 base 模型更稳）
- 粗粒度实体：只取方法名/论文名/核心概念，避免图谱爆炸
- 复用 config/credentials.env 的 DeepSeek 配置，不硬编码 key

用法：
    python3 scripts/graphrag_entities.py [--max-docs 3] [--limit-chars 6000]
        [--out data/graph/entities.json] [--concurrency 4]
"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE, "data", "chunks")

# ── 读取 DeepSeek 配置（不进 git）──
def load_cfg():
    cfg = {}
    path = os.path.join(BASE, "config", "credentials.env")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg

CFG = load_cfg()
API_URL = CFG.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
API_KEY = CFG.get("LLM_API_KEY", "")
MODEL = CFG.get("LLM_MODEL", "deepseek-chat")


def call_llm(prompt, max_tokens=400, temperature=0.3):
    """OpenAI 兼容 HTTP 调用。返回文本。"""
    import urllib.request
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
        return d["choices"][0]["message"]["content"].strip()


def load_paper_texts(max_docs=None, limit_chars=6000):
    papers = {}
    files = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith(".json"))
    if max_docs: files = files[:max_docs]
    for f in files:
        doc_id = f[:-5]
        try:
            chunks = json.load(open(os.path.join(CHUNK_DIR, f)))
            text = " ".join(c["text"] for c in chunks if isinstance(c, dict))
            papers[doc_id] = text[:limit_chars]
        except Exception as e:
            print(f"  [warn] 跳过 {doc_id}: {e}")
    return papers


def build_prompt(doc_id, text):
    # few-shot 示例：给 base 模型示范三元组格式
    few_shot = (
        "几个示例（仅示范格式，与本文无关）：\n"
        "LoRA | is_a | parameter-efficient fine-tuning\n"
        "DPO | improves_on | RLHF\n"
        "RAG | retrieves | external knowledge\n"
        "attention | is_used_in | Transformer\n"
        "向量数据库 | based_on | FAISS\n\n"
    )
    return (
        "你是知识图谱抽取助手。从下面这篇 AI 论文中，抽取关键的【实体】和它们之间的【关系】。\n"
        "只抽取粗粒度的稳定概念：论文名、方法名（如 LoRA/DPO）、核心概念（如 attention、RAG、vector database）。\n"
        "不要抽 generic 词（如 model、data、paper、system 这类没区分度的词）。\n"
        "输出格式：每行一条三元组，严格用竖线分隔：实体A|关系|实体B。\n"
        "关系用简短英文动词或介词短语（如 uses、improves_on、is_a、based_on）。\n"
        "只输出三元组，不要多余解释、不要编号、不要引号。最多输出 15 条。\n\n"
        + few_shot +
        f"论文: {doc_id}\n\n正文:\n{text}\n\n三元组:\n"
    )

TRIPLE_RE = re.compile(r"^\s*([^|\n]{1,60})\s*\|\s*([^|\n]{1,40})\s*\|\s*([^|\n]{1,60})\s*$")

def parse_triples(text):
    triples = []
    for line in text.splitlines():
        m = TRIPLE_RE.match(line)
        if not m: continue
        a, rel, b = [x.strip().strip('"').strip() for x in m.groups()]
        if a and b and a != b:
            triples.append((a, rel, b))
    return triples


def process_one(doc_id, text, limit_chars, retries=4):
    # flash 对长文本偶尔返回空：0 条也算“失败”，重试直到拿到非空结果
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            out = call_llm(build_prompt(doc_id, text[:limit_chars]))
            triples = parse_triples(out)
            if triples:
                return doc_id, triples, None
            last_err = f"空输出 (attempt {attempt})"
        except Exception as e:
            last_err = f"{e} (attempt {attempt})"
        # 短暂退避再重试
        import time
        time.sleep(attempt)
    return doc_id, [], last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=999)
    ap.add_argument("--limit-chars", type=int, default=6000)
    ap.add_argument("--out", default="data/graph/entities.json")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--only-failed", action="store_true",
                    help="只跑已有输出里失败/空/极少的篇，合并回原文件")
    args = ap.parse_args()

    print(f"DeepSeek API 实体抽取  | model={MODEL} | max_docs={args.max_docs}")
    if not API_KEY: sys.exit("❌ credentials.env 缺 LLM_API_KEY")

    papers = load_paper_texts(args.max_docs, args.limit_chars)

    # 只补跑失败篇：加载已有输出，找出 0 条或极少的
    existing = {}
    if args.only_failed and os.path.exists(args.out):
        prev = json.load(open(args.out))
        existing = prev.get("documents", {})
        weak = {k: v for k, v in existing.items() if len(v) <= 2}
        papers = {k: v for k, v in papers.items() if k in weak}
        print(f"  only-failed: 上次 {len(existing)} 篇已完成，{len(weak)} 篇失败/极少需补跑")
    if not papers:
        print("  ✅ 没有需补跑的篇，直接结束")
        if args.only_failed and existing:
            print(f"  现有结果共 {sum(len(v) for v in existing.values())} 条 → {args.out}")
        return

    print(f"  本次补跑 {len(papers)} 篇: {list(papers.keys())}")

    result, errors = {}, {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(process_one, d, t, args.limit_chars, args.retries): d for d, t in papers.items()}
        done = 0
        for fut in as_completed(futs):
            doc_id, triples, err = fut.result()
            done += 1
            if err or not triples:
                errors[doc_id] = err
                print(f"  [{done}/{len(papers)}] ❌ {doc_id}: {(err or '竟然0条')[:100]}")
            else:
                result[doc_id] = triples
                print(f"  [{done}/{len(papers)}] ✅ {doc_id}: {len(triples)} 条")
                for t in triples[:4]:
                    print(f"       {' | '.join(t)}")

    # 合并回已有结果
    merged = dict(existing)
    merged.update(result)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": MODEL, "documents": merged, "errors": errors},
                  f, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in merged.values())
    print(f"\n完成：合并后共 {total} 条三元组 → {args.out}")
    if errors: print(f"  仍失败 {len(errors)} 篇: {list(errors.keys())}")


if __name__ == "__main__":
    main()
