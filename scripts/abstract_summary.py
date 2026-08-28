#!/usr/bin/env python3
"""W23 Day 1 — abstract_summary.py：用 DeepSeek 生成抽象摘要，替换 extractive 占位

背景：W22 实测 extractive 占位摘要质量不足（内容错配、噪声），限制 RAPTOR 主题检索增益。
      W23 关键一步：用 LLM 生成「抽象摘要」，看检索增益能否兑现。

方法：
  1. 读现有清洗后的 81 个摘要节点（每个节点代表文档内的一个聚簇）
  2. 对每个节点：把它的 extractive 文本送 DeepSeek，提炼成一句「抽象主题摘要」+ 3 个检索关键词
  3. 用抽象摘要重建向量 → 重建摘要索引
  4. 重新跑验证查询，对比清洗版 vs 抽象版

凭证：config/credentials.env（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）

用法：python3 abstract_summary.py [--dry-run 前N个]
"""
import os
import json
import time
import sys

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE, "..", "data", "index")
CRED = os.path.join(BASE, "..", "config", "credentials.env")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512


def load_creds():
    """从 credentials.env 读取 LLM 配置。"""
    env = {}
    if os.path.exists(CRED):
        with open(CRED) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    key = env.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY", "").strip()
    if not key or key.startswith("sk-7d0") and len(key) < 10:
        raise SystemExit("❌ 未找到有效 LLM_API_KEY（config/credentials.env）")
    return env


def call_llm(api_key, base_url, model, prompt, max_tokens=300, retries=3):
    """调用 DeepSeek chat 补全，带重试。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    if "deepseek.com" in base_url and not url.endswith("/chat/completions"):
        url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            print(f"  ⚠️ HTTP {r.status_code}: {r.text[:120]}")
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"  ⚠️ 请求异常: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def build_prompt(node_text, doc_id):
    """构造抽象摘要提示词。"""
    return (
        "你是科研文献检索助手。下面是从一篇 AI 论文中抽取出的一个主题片段的原文节选。\n"
        "请把它提炼成一个【抽象主题摘要】：一段 2-3 句的中文概述，讲清楚这段内容的核心"
        "主题、方法和贡献，方便后续语义检索命中。\n"
        "最后单独一行输出 3 个最能代表这段主题的中文检索关键词，用逗号分隔。\n\n"
        f"[论文] {doc_id}\n[片段原文]\n{node_text[:1500]}\n\n"
        "输出格式：\n摘要：<2-3句中文>\n关键词：<关键词1>,<关键词2>,<关键词3>"
    )


def main():
    dry_run = int(sys.argv[sys.argv.index("--dry-run") + 1]) if "--dry-run" in sys.argv else None

    env = load_creds()
    api_key, base_url, model = env["LLM_API_KEY"], env["LLM_BASE_URL"], env["LLM_MODEL"]
    print(f"LLM: {model} @ {base_url}")

    # 读清洗后的摘要节点
    meta_path = os.path.join(INDEX, "raptor_summary.json")
    meta = json.load(open(meta_path))
    # 过滤根摘要（根摘要由 LLM 之后单独生成）
    nodes = [m for m in meta if m.get("type") != "root_summary"]
    print(f"待生成抽象摘要的节点: {len(nodes)}")

    # 加载 BGE 用于后续编码
    print(f"加载 {MODEL_NAME} ...")
    st = SentenceTransformer(MODEL_NAME)

    # 逐节点生成抽象摘要
    updated = []
    root_source_texts = []
    for i, m in enumerate(nodes):
        if dry_run and i >= dry_run:
            break
        doc_id = m.get("source_doc", m.get("doc_id", "?"))
        prompt = build_prompt(m["text"], doc_id)
        result = call_llm(api_key, base_url, model, prompt)
        if result:
            # 解析 摘要 / 关键词
            abstract = ""
            keywords = ""
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("摘要"):
                    abstract = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
                elif line.startswith("关键词"):
                    keywords = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            if not abstract:
                abstract = result[:200]
            new_text = f"{abstract}" + (f" | 关键词: {keywords}" if keywords else "")
            m = dict(m)
            m["text"] = new_text
            m["abstract"] = abstract
            m["keywords"] = keywords
            updated.append(m)
            root_source_texts.append(abstract)
            print(f"  [{i+1}/{len(nodes)}] {doc_id}: {abstract[:60]}")
        else:
            print(f"  [{i+1}/{len(nodes)}] {doc_id}: ❌ 生成失败，保留原文")
            updated.append(m)
        time.sleep(0.3)  # 温和限速

    if dry_run:
        print(f"\n(dry-run 完成 {len(updated)} 个，未写入)")
        return

    # 生成跨篇根摘要
    print("\n生成跨篇根摘要 ...")
    root_prompt = (
        "下面是我个人知识库中多篇 AI 论文的主题摘要列表。请用一段 2-3 句中文概述整个知识库的"
        "覆盖范围，作为跨篇入口摘要。\n\n" + "\n".join(f"- {t}" for t in root_source_texts[:60])
    )
    root_result = call_llm(api_key, base_url, model, root_prompt, max_tokens=200)
    if not root_result:
        root_result = " ".join(root_source_texts[:10])[:300]
    updated.append({
        "text": root_result, "doc_id": "__ROOT__", "source_doc": "__ROOT__",
        "child_ids": list(range(len(updated))), "type": "root_summary", "abstract": root_result,
    })

    # 重排 summary_id
    for new_id, m in enumerate(updated):
        m["summary_id"] = new_id
        # child_ids 指向旧 id —— 根摘要的 child 指向所有，这里简化：普通节点 child 保留原语义
        if m.get("type") == "root_summary":
            m["child_ids"] = list(range(len(updated)))

    # 重新编码 + 重建索引
    print(f"\n编码 {len(updated)} 个抽象摘要节点 ...")
    texts = [m["text"] for m in updated]
    embs = st.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32).astype("float32")
    index = faiss.IndexHNSWFlat(DIM, 16)
    index.hnsw.efConstruction = 200
    index.add(embs)

    # 备份旧的 extractive 版索引
    old_faiss = os.path.join(INDEX, "raptor_summary.faiss")
    old_json = os.path.join(INDEX, "raptor_summary.json")
    if os.path.exists(old_faiss):
        os.replace(old_faiss, os.path.join(INDEX, "raptor_summary_extractive_backup.faiss"))
        os.replace(old_json, os.path.join(INDEX, "raptor_summary_extractive_backup.json"))

    faiss.write_index(index, old_faiss)
    with open(old_json, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"抽象摘要节点: {len(updated)}")
    print(f"索引: {old_faiss} ({index.ntotal} 条)")
    print(f"extractive 版已备份: raptor_summary_extractive_backup.*")


if __name__ == "__main__":
    main()
