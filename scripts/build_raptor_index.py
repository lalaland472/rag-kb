#!/usr/bin/env python3
"""Week 22 Day 2 — build_raptor_index.py：构建 rag-kb 的 RAPTOR 摘要索引（正式模块）

架构决策（基于 W22 试点结论：「Flat 叶子 + RAPTOR 摘要」互补，Recall@5 从 0.778→1.000）：
  - Flat 路       : 复用现有 data/index/index.faiss（4521 叶子 HNSW），负责细节召回
  - RAPTOR 摘要路 : 新增「摘要索引」，负责跨篇主题入口

摘要层构建策略（避免全量 4521 叶子 UMAP 爆炸，且贴合「跨文档主题入口」价值）：
  按文档维度生成摘要节点 —— 每篇论文用自己的 chunks 聚簇生成若干主题摘要，
  再加一层「跨篇根摘要」。这样摘要层提供的正是试点证明有价值的「主题入口」，
  而不是对 4521 个异质叶子做难以收敛的全局聚类。

产出：
  data/index/raptor_summary.faiss   — 摘要节点向量索引
  data/index/raptor_summary.json    — 摘要节点元数据（text / doc 来源 / children 叶子范围）

运行：python3 build_raptor_index.py
"""
import os
import json
import glob
import re
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 复用 W19 的 GMM 聚类 + extractive 摘要
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "week19-longdoc-rag"))
from raptor_pipeline import cluster_gmm, generate_summary

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512
SUMMARIES_PER_DOC = 4     # 每篇文档生成的摘要节点数上限
ROOT_SUMMARY = True       # 是否生成跨篇根摘要

# ── 摘要清洗：识别无语义的垃圾摘要（致谢/版权/引用/纯参考文献页）──
GARBAGE_PATTERNS = re.compile(
    r'(arXiv\s*[:.]?\s*\d{4}|doi:\s*10\.|\bhttps?://|\bwww\.|acknowledg\w*|was supported by|is supported by|We thank|The authors thank|Fellowship from|gifts from|funding from|Copyright|\bISSN\b|Proceedings of|Conference on Neural Information|ICDT \d{4}|ACM Journal of Experimental|References\s+[A-Z][a-z]+)',
    re.IGNORECASE
)
# 致谢版权占比过高 → 判为垃圾摘要
ACK_HEAVY = re.compile(r"(thank|supported|fund|fellowship|acknowledg|grant|gift)", re.IGNORECASE)


def is_garbage_summary(text: str) -> bool:
    """判断摘要是否是无语义垃圾（致谢/版权/纯引用/arxiv头）。"""
    if not text:
        return True
    # 整段匹配垃圾模式 → 垃圾
    if GARBAGE_PATTERNS.search(text[:400]):
        # 但若开头是论文标题/摘要正文，允许通过（如 GPTQ 那类 ICLR 头）
        # 用 ACK_HEAVY 进一步判断：致谢类词汇密度高才算垃圾
        ack_count = len(ACK_HEAVY.findall(text))
        if ack_count >= 2:
            return True
        # 纯 arXiv 头 / 版权行（很短且无正文）
        if len(text.strip()) < 120 and GARBAGE_PATTERNS.search(text):
            return True
    return False


def clip_summary_text(text: str, max_chars: int = 250) -> str:
    """从摘要里剔除致谢/引用尾巴，保留语义正文。"""
    # 找致谢/references 起始点，截断
    cuts = [m.start() for m in re.finditer(
        r"(acknowledg\w*|references|we thank|is supported by|was supported by|"
        r"arxiv:|\bdoi:\s*10\.|\bresources?\b|\bfunding\b)",
        text, re.IGNORECASE
    )]
    if cuts:
        text = text[:min(cuts)]
    return text.strip()[:max_chars]



def load_all_chunks(chunks_dir: str) -> list[dict]:
    recs = []
    for f in sorted(glob.glob(os.path.join(chunks_dir, "*.json"))):
        with open(f, encoding="utf-8") as fp:
            recs.extend(json.load(fp))
    for i, r in enumerate(recs):
        r["global_id"] = i
    return recs


def group_by_doc(chunks: list[dict]) -> dict[str, list[dict]]:
    from collections import OrderedDict
    groups = OrderedDict()
    for c in chunks:
        groups.setdefault(c["doc_id"], []).append(c)
    return groups


def build_summary_for_doc(model, doc_chunks, doc_id, max_summaries):
    """对单篇文档的 chunks 聚簇生成摘要节点。
    返回 (summaries, cluster_assignments)：
      summaries: list[dict] 每个摘要节点 {text, doc_id, child_ids, type}
      cluster_assignments: 每个 chunk 属于哪个摘要簇（-1 表示无）
    """
    texts = [c["text"] for c in doc_chunks]
    n = len(texts)

    # 文档太小：直接一篇一摘要
    if n <= max_summaries:
        summ_text = generate_summary(texts, max_chars=200)
        return [{"text": summ_text, "doc_id": doc_id,
                 "child_ids": [c["global_id"] for c in doc_chunks], "type": "summary"}]

    embs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32).astype("float32")

    # 聚簇（GMM 自动选 K，限制上限）
    labels = cluster_gmm(embs, max_k=max_summaries, random_state=42)
    unique = sorted(set(labels))

    summaries = []
    cluster_map = {}
    for lab in unique:
        member_idx = [i for i in range(n) if labels[i] == lab]
        member_texts = [texts[i] for i in member_idx]
        summ_text = generate_summary(member_texts, max_chars=200)
        # 清洗：裁剪致谢/引用尾巴
        summ_text = clip_summary_text(summ_text)
        # 垃圾摘要（致谢/版权/纯引用）→ 跳过，不建为摘要节点
        if is_garbage_summary(summ_text):
            continue
        summaries.append({
            "text": summ_text, "doc_id": doc_id,
            "child_ids": [doc_chunks[i]["global_id"] for i in member_idx],
            "type": "summary",
        })
        for i in member_idx:
            cluster_map[doc_chunks[i]["global_id"]] = lab
    return summaries, cluster_map


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "data")
    chunks_dir = os.path.join(data_dir, "chunks")
    index_dir = os.path.join(data_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    print("=" * 64)
    print("W22 Day 2 — 构建 RAPTOR 摘要索引")
    print("=" * 64)

    # 1. 加载全部 chunks
    chunks = load_all_chunks(chunks_dir)
    groups = group_by_doc(chunks)
    print(f"共 {len(chunks)} chunks / {len(groups)} 篇文档")

    # 2. 加载模型
    print(f"加载 {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. 逐篇生成摘要节点
    print("\n=== 逐篇生成摘要节点 ===")
    summaries = []
    for doc_id, doc_chunks in groups.items():
        doc_summaries, _ = build_summary_for_doc(model, doc_chunks, doc_id, SUMMARIES_PER_DOC)
        for s in doc_summaries:
            s["source_doc"] = doc_id
        summaries.extend(doc_summaries)
        print(f"  {doc_id:<28} → {len(doc_summaries)} 摘要节点")

    # 4. 跨篇根摘要（所有摘要的摘要）
    if ROOT_SUMMARY:
        summ_texts = [s["text"] for s in summaries if not is_garbage_summary(s["text"])]
        if summ_texts:
            root_text = generate_summary(summ_texts, max_chars=300)
            root_text = clip_summary_text(root_text)
            summaries.append({
                "text": root_text, "doc_id": "__ROOT__", "source_doc": "__ROOT__",
                "child_ids": [i for i in range(len(summaries))], "type": "root_summary",
            })
            print(f"  跨篇根摘要: 1 节点")

    print(f"\n摘要节点总数: {len(summaries)}")

    # 5. 编码摘要节点 + 建索引
    print("\n=== 编码摘要节点 + 建索引 ===")
    summ_texts = [s["text"] for s in summaries]
    summ_embs = model.encode(summ_texts, normalize_embeddings=True,
                             show_progress_bar=True, batch_size=32).astype("float32")

    index = faiss.IndexHNSWFlat(DIM, 16)
    index.hnsw.efConstruction = 200
    index.add(summ_embs)

    faiss.write_index(index, os.path.join(index_dir, "raptor_summary.faiss"))

    # 6. 持久化摘要元数据
    metadata = []
    for i, s in enumerate(summaries):
        metadata.append({
            "summary_id": i,
            "text": s["text"],
            "doc_id": s.get("doc_id"),
            "source_doc": s.get("source_doc"),
            "child_ids": s.get("child_ids", []),
            "type": s.get("type", "summary"),
        })
    with open(os.path.join(index_dir, "raptor_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"摘要索引: {os.path.join(index_dir, 'raptor_summary.faiss')}")
    print(f"摘要元数据: {os.path.join(index_dir, 'raptor_summary.json')}")
    print(f"摘要节点数: {len(metadata)}")


if __name__ == "__main__":
    main()
