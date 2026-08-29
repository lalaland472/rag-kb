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


def call_llm(api_key, base_url, model, prompt, max_tokens=600, temperature=0.3, system=None):
    """调用 DeepSeek API。deepseek-v4-flash 是 reasoning 模型，
    会占用一部分 max_tokens，故生成只读 content 字段。
    system 可选覆盖默认 SYSTEM_PROMPT。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
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

# ⚠️ Self-RAG 启发（W23 Day 6）：strict 模式 = IsSup（支持性）权重拉满。
# 关键边界：不是禁言模型，而是"每个可核验事实点都要有依据"。
# bridge/总结性表述允许模型自己的语言，但不得编造可核验事实。
STRICT_SYSTEM_PROMPT = """你是「RAG-KB」个人论文知识库的问答助手，运行在【严格模式】。

核心原则（Self-RAG IsSup 权重拉满）：
1. 只使用提供的论文片段回答，忠实于原文。
2. 【每个可核验事实点都必须有引用支撑】——你说出的每个可核验事实，都要能对应到某个 [序号] 片段。
3. 对于片段中找不到依据的事实点：
   - 若该事实是必要的核心结论，明确标注「⚠️ 资料中未找到依据」并说明。
   - 若只是某个细节被我说出但无依据，删除它或在它前面标注「（无资料依据）」。
4. 桥梁性/总结性表述（如"结合以上论文"）允许用自己的语言组织，
   但不得编造"论文里没有的具体事实/数字/结论"。
5. 若片段完全不足以回答，如实说「根据提供的资料无法完整回答」，不要硬编。
6. 回答末尾用 [1]、[2] 编号标注引用来源。
7. 用中文回答，简洁准确。"""

# ⚠️ Self-RAG 启发：chat 模式 = IsUse（有用性）权重拉高，允许适度发散，
# 但核心事实仍以资料为主。
CHAT_SYSTEM_PROMPT = """你是「RAG-KB」个人论文知识库的问答助手，运行在【对话模式】。

核心原则（Self-RAG IsUse 权重拉高）：
1. 优先使用提供的论文片段回答，忠实于原文。
2. 允许结合你自己的知识适度发散、组织更完整的解释，目标是回答"对用户最有用"。
3. 可核验的关键事实尽量标注来源 [序号]；发散/补充部分不需要强行编引用。
4. 用中文回答，简洁准确，语气自然。"""

# ⚠️ Retrieve on-demand 判定不检索时（纯常识/闲聊 query）的默认 prompt：
# 模型自己知道答案，应该放它用自身知识回答，而不是套 RAG 的"只答资料内"规则。
LLM_ONLY_PROMPT = """你是乐于助人的中文问答助手。
直接凭你的知识回答用户问题，简洁准确。
不需要引用来源（这是无需检索的常识/闲聊问题）。"""

# ⚠️ Self-RAG 启发：Retrieve on-demand——先判断"这次需不需要检索"。
# 纯常识/闲聊 query 直接 LLM-only，跳过检索（省耗时+避免无关片段干扰）。
RETRIEVAL_NEED_SYSTEM = """你是问答路由判断器。判断下面这个用户问题是否需要检索外部资料才能回答。
输出 JSON：{"need_retrieval": true|false, "reason": "简短理由"}

判断标准：
- false：纯常识/闲聊/开放创作/模型固有知识即可回答（如"一年有几个季节""写首诗""你好""Python是什么"）。
- true：需要特定资料/论文/数据支撑才能准确回答（如论文细节、对比某个具体研究）。
只输出 JSON，不要其他内容。"""


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


def check_retrieval_needed(api_key, base_url, model, query, max_tokens=150):
    """⚠️ Self-RAG Retrieve on-demand：判断这次 query 需不需要检索。
    返回 True（需要检索）/ False（不需要，走 LLM-only）。失败时保守返回 True。"""
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": RETRIEVAL_NEED_SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            r = json.load(resp)
        out = r["choices"][0]["message"].get("content", "").strip()
        import re
        m = re.search(r'"need_retrieval"\s*:\s*(true|false)', out, re.I)
        if m:
            return m.group(1).lower() == "true"
        # 兼容无引号情况
        if "true" in out.lower():
            return True
        return True
    except Exception:
        return True


FAITHFULNESS_SYSTEM = ("你是资料核验员。检查回答中每个【可核验事实点】是否被提供的论文片段支持。"
                       "若回答里有片段不支持的、看似具体的事实/数字/结论，逐条指出。"
                       "若全部有依据，只输出：全部有依据。")


def check_faithfulness(api_key, base_url, model, query, answer, chunks, max_tokens=300):
    """⚠️ Self-RAG 启发（迷你版 IsSup）：strict 模式下核对回答事实点是否有引用支撑。
    片段无依据/依据不足的事实点 → 返回提醒文案。全有依据 → 返回 None。"""
    try:
        context = "\n\n".join(f"[片段{i}] {t}" for i, (_, _, _, t) in enumerate(chunks, 1))
        prompt = (
            f"用户问题：{query}\n\n"
            f"模型回答：\n{answer}\n\n"
            f"===== 论文片段（判断依据）=====\n{context}\n===== 结束 =====\n\n"
            "请核对：1) 逐条列出回答中『无片段依据的』可核验事实点；"
            "2) 若有，用「⚠️ 无资料依据：...」格式逐条说明；3) 若全部有依据，只输出『全部有依据』。"
        )
        result = call_llm(api_key, base_url, model, prompt, max_tokens=max_tokens,
                          temperature=0.0, system=FAITHFULNESS_SYSTEM)
        result = result.strip()
        if not result or "全部有依据" in result or "全部有支撑" in result:
            return None
        return result
    except Exception:
        return None


def generate_answer(query, mode="flat", k=5, max_tokens=600, debug=False, **kw):
    """主入口：检索 → 取原文 → 生成 → 返回 {answer, meta}

    kw 扩展：
      hyde (bool): 是否先做 HyDE 改写（默认 True）
      style (str): "strict"(IsSup拉满,忠实核对) | "chat"(IsUse拉高,允许发散) | "default"
      retrieval_check (bool): 是否先做 Retrieve on-demand 预判，纯常识 query 走 LLM-only
      llm_only (bool): 强制不检索，直接让模型回答（配合 retrieval_check=False 时用）
    """
    env = load_env()
    api_key = env.get("LLM_API_KEY", "")
    base_url = env.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = env.get("LLM_MODEL", "deepseek-v4-flash")
    if not api_key:
        raise SystemExit("❌ 未找到 LLM_API_KEY（config/credentials.env）")

    style = kw.pop("style", "default")
    retrieval_check = kw.pop("retrieval_check", False)
    llm_only = kw.pop("llm_only", False)

    # ⚠️ Self-RAG Retrieve on-demand：预判这次需不需要检索
    # 纯常识/闲聊 query → 不检索，直接 LLM-only
    if (retrieval_check or llm_only) and not kw.pop("skip_need", False):
        need = check_retrieval_needed(api_key, base_url, model, query)
        if debug:
            print(f"\n[Retrieve-on-demand] need_retrieval={need}\n")
        if need is False:
            # 不需要检索：直接让模型基于自身知识回答
            # ⚠️ 不能用 RAG 的 SYSTEM_PROMPT（它强制"只答资料内"，会误答"无法回答"）
            # 应放模型用自己的知识：chat 风格即 IsUse 拉高，default 也用通用助手 prompt
            system = CHAT_SYSTEM_PROMPT if style == "chat" else (
                LLM_ONLY_PROMPT if style == "default" else STRICT_SYSTEM_PROMPT)
            answer = call_llm(api_key, base_url, model,
                              f"请直接回答下面的问题（不需要检索资料，凭你的知识回答即可）：\n\n用户问题：{query}",
                              max_tokens=max_tokens, system=system)
            return {
                "answer": answer,
                "meta": {
                    "query": query, "mode": mode, "k": k, "model": model,
                    "style": style, "retrieval_needed": False,
                    "citations": [], "chunk_count": 0, "fused_docs": [],
                },
            }

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
        return {"answer": "未检索到相关论文片段。", "meta": {"mode": mode, "citations": [], "chunks": [], "retrieval_needed": True}}

    # 生成：按 style 选 system prompt
    system = SYSTEM_PROMPT
    if style == "strict":
        system = STRICT_SYSTEM_PROMPT
    elif style == "chat":
        system = CHAT_SYSTEM_PROMPT

    prompt = build_prompt(query, chunks)
    if debug:
        print(f"\n[Prompt 前 400 字]\n{prompt[:400]}...\n")

    answer = call_llm(api_key, base_url, model, prompt, max_tokens=max_tokens, system=system)

    # ⚠️ Self-RAG 启发：strict 模式下追加一道忠实性核对（迷你版 IsSup）
    # 核对回答里每个可核验事实点是否有引用支撑；无支撑处标黄/标注提示
    fidelity_note = None
    if style == "strict":
        fidelity_note = check_faithfulness(api_key, base_url, model, query, answer, chunks)
        if fidelity_note and fidelity_note.strip():
            answer = answer + "\n\n【忠实性核对】\n" + fidelity_note.strip()

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
            "style": style, "retrieval_needed": True,
            "fidelity_checked": (style == "strict"),
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
