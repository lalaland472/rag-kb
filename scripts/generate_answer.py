#!/usr/bin/env python3
"""Week 23 Day 1 — generate_answer.py：rag-kb 生成模块

架构：本地检索（BGE + FAISS/RAPTOR）→ 取 chunk 原文 → DeepSeek API 生成带引用回答

设计取舍：
  - 检索在本地（BGE-small-zh + FAISS），保证每一跳快、不依赖 API
  - 生成本机 1GB 内存跑不动 7B，走 DeepSeek API（W22 已验证 deepseek-v4-flash）
  - 上下文只取 top-k 个 chunk 原文，控制 token；引用用统一编号并回填文档名

用法：
  python3 scripts/generate_answer.py "问题" [--mode flat|hybrid] [--k 5] [--max-tokens 600] [--debug]
"""
import os
import json
import sys
import glob
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
DATA = os.path.join(ROOT, "data")
CHUNKS_DIR = os.path.join(DATA, "chunks")

# ── 加载 LLM 凭证（直接解析 credentials.env，避免 source 子 shell 问题）──
def load_env():
    env = {}
    path = os.path.join(ROOT, "config", "credentials.env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    # 环境变量优先
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if os.environ.get(k):
            env[k] = os.environ[k].strip()
    return env


# ── chunk 原文缓存：doc_id -> {chunk_id: text} ──
_CHUNK_CACHE = None


def _load_chunks_cache():
    global _CHUNK_CACHE
    if _CHUNK_CACHE is not None:
        return _CHUNK_CACHE
    cache = {}
    for f in glob.glob(os.path.join(CHUNKS_DIR, "*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if isinstance(data, list):
            for c in data:
                doc = c.get("doc_id")
                ch = c.get("chunk_id")
                if doc is not None and ch is not None:
                    cache.setdefault(doc, {})[ch] = c.get("text", "")
    _CHUNK_CACHE = cache
    return cache


def chunk_text(doc_id, chunk_id):
    """按 doc_id+chunk_id 取 chunk 原文。"""
    return _load_chunks_cache().get(doc_id, {}).get(chunk_id, "")


def doc_display_name(doc_id):
    """doc_id → 可读论文名（去掉年份后缀，下划线转空格）。"""
    name = doc_id
    for suffix in ("_2024", "_2023", "_2022", "_2021", "_2020", "_2019", "_2018", "_2017", "_2016"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ")


# ── 来自 retriever.py 的检索引擎 + citation_pipeline 引用链 ──
from retriever import HybridRetriever
from citation_pipeline import query as cite_query, display_ref as cite_display


# ── HyDE：把口语化/中文 query 翻译成学术英文再检索（W18 实证有效）──
HYDE_SYSTEM = (
    "你是学术论文检索助手。用户会给一个可能是口语化或中文的问题，"
    "请输出一个适合在英文 AI 论文语料库中检索的查询文本：保留问题意图，"
    "用专业学术英文术语改写，并附带 3-5 个关键词。只输出改写后的查询文本，不要解释。"
)


def hyde_query(api_key, base_url, model, query, max_tokens=200):
    """用 LLM 把 query 改写成适合语义检索的学术英文。失败/空结果则原样返回 query。

    ⚠️ deepseek-v4-flash 是 reasoning 模型，max_tokens 需给足（≥200），
    否则预算被 reasoning_tokens 吃光导致 content 为空。
    """
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            r = json.load(resp)
        out = r["choices"][0]["message"].get("content", "").strip()
        # 空/只含关键词行 → 视为失败，退回原 query
        if not out or len(out) < 8:
            return query
        return out
    except Exception:
        return query


def call_llm(api_key, base_url, model, prompt, max_tokens=600, temperature=0.3):
    """调用 DeepSeek API。deepseek-v4-flash 是 reasoning 模型，
    会占用一部分 max_tokens，故生成只读 content 字段。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.load(resp)
    return r["choices"][0]["message"]["content"].strip()


SYSTEM_PROMPT = """你是「RAG-KB」个人论文知识库的问答助手。你只基于给定的论文片段回答，忠实于原文。
规则：
1. 只使用提供的片段，不要编造片段中没有的事实。
2. 回答末尾用 [1]、[2] 等编号标注信息来源，编号对应「片段」前的[序号]。
3. 若片段不足以回答，如实说「根据提供的资料无法完整回答」，并给出最接近的信息。
4. 用中文回答，简洁准确。"""


def build_prompt(query, chunks):
    """组装 user prompt。chunks: [(doc_display, doc_id, chunk_id, text)]"""
    parts = []
    for i, (disp, doc, cid, text) in enumerate(chunks, 1):
        parts.append(f"[{i}]（来源：{disp}）\n{text}\n")
    context = "\n".join(parts)
    return (
        f"请根据下列论文片段回答用户问题。\n\n"
        f"用户问题：{query}\n\n"
        f"===== 论文片段 =====\n{context}\n"
        f"===== 片段结束 =====\n"
        f"请回答问题，并在引用处标注 [序号]。"
    )


def generate_answer(query, mode="flat", k=5, max_tokens=600, debug=False, **kw):
    """主入口：检索 → 取原文 → 生成 → 返回 {answer, meta}"""
    env = load_env()
    api_key = env.get("LLM_API_KEY", "")
    base_url = env.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = env.get("LLM_MODEL", "deepseek-v4-flash")
    if not api_key:
        raise SystemExit("❌ 未找到 LLM_API_KEY（config/credentials.env）")

    retriever = HybridRetriever()
    # HyDE：先把请求 query 翻译成学术英文再检索（解中文口语 query 检索英文 chunk 不相关的问题）
    use_hyde = kw.pop("hyde", True)
    search_query = query
    if use_hyde:
        search_query = hyde_query(api_key, base_url, model, query)
        if debug and search_query != query:
            print(f"\n[HyDE] {query} →\n       {search_query}\n")
    result = retriever.retrieve(search_query, k=k, mode=mode)

    # 取 top chunk 原文作为生成上下文
    #   flat 模式  ：直接按 top-k 叶子 chunk 取原文（精确句子召回）
    #   hybrid 模式：优先用摘要文本（主题概括），再补叶子原文（细节）
    chunks = []
    seen = set()

    if mode == "hybrid":
        # 摘要文本优先（主题命中强）
        for s in result.get("summary_top", [])[:k]:
            doc = s["doc_id"]
            if doc == "__ROOT__" or doc in seen:
                continue
            text = (s.get("text") or "").strip()
            if text:
                seen.add(doc)
                chunks.append((doc_display_name(doc), doc, None, text[:1200]))
        # 未满 k 条时，用 top 叶子原文补细节
        for r in result.get("flat_top", []):
            if len(chunks) >= k:
                break
            doc, cid = r["doc_id"], r["chunk_id"]
            key = (doc, cid)
            if key in seen:
                continue
            text = chunk_text(doc, cid)
            if text:
                seen.add(key)
                chunks.append((doc_display_name(doc), doc, cid, text[:1500]))
    else:
        # flat：直接按 top-k 叶子 chunk 取原文
        for r in result.get("flat_top", [])[:k]:
            doc, cid = r["doc_id"], r["chunk_id"]
            key = (doc, cid)
            if key in seen:
                continue
            text = chunk_text(doc, cid)
            if text:
                seen.add(key)
                chunks.append((doc_display_name(doc), doc, cid, text[:1500]))

    if not chunks:
        return {"answer": "未检索到相关论文片段。", "meta": {"mode": mode, "citations": [], "chunks": []}}

    prompt = build_prompt(query, chunks)
    if debug:
        print(f"\n[Prompt 前 400 字]\n{prompt[:400]}...\n")

    answer = call_llm(api_key, base_url, model, prompt, max_tokens=max_tokens)

    # 引用列表（带页码：chunk → 文件 → 页面）
    citations = []
    for i, (disp, doc, cid, _) in enumerate(chunks, 1):
        page = None
        if cid is not None:
            q = cite_query(doc, cid)
            page = q.get("page")
        citations.append({
            "index": i, "doc_id": doc, "display": disp,
            "chunk_id": cid, "page": page,
            "page_label": (cite_display(doc, cid) if cid is not None else disp),
        })
    return {
        "answer": answer,
        "meta": {
            "query": query, "mode": mode, "k": k, "model": model,
            "citations": citations,
            "chunk_count": len(chunks),
            "fused_docs": result.get("fused_docs", []),
        },
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/generate_answer.py \"问题\" [--mode flat|hybrid] [--k 5] [--max-tokens 600] [--debug]")
        return
    query = sys.argv[1]
    mode, k, mt, debug = "flat", 5, 600, False
    args = sys.argv[2:]
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--k" in args:
        k = int(args[args.index("--k") + 1])
    if "--max-tokens" in args:
        mt = int(args[args.index("--max-tokens") + 1])
    if "--debug" in args:
        debug = True

    print(f"🔍 查询: {query} (mode={mode}, k={k})")
    res = generate_answer(query, mode=mode, k=k, max_tokens=mt, debug=debug)
    print("\n" + "=" * 60)
    print("📝 回答:")
    print("=" * 60)
    print(res["answer"])
    print("\n" + "=" * 60)
    print("📚 引用来源:")
    for c in res["meta"]["citations"]:
        if c["chunk_id"] is not None:
            loc = f"第{c['page'] + 1}页" if c.get("page") is not None else f"块{c['chunk_id']}"
        else:
            loc = "(摘要)"
        print(f"  [{c['index']}] {c['display']}  [{loc}]")
    print(f"\n[mode={res['meta']['mode']}, chunk数={res['meta']['chunk_count']}]")


if __name__ == "__main__":
    main()
