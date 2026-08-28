#!/usr/bin/env python3
"""Week 21 Day 2 — chunk_pipeline() 语义分块管道

策略：Semantic Chunking（基于 embedding 相似度的句子边界切分）
  - 先把每篇论文文本切成句子
  - 用 BGE 对句子编码
  - 相邻句子的相似度低于阈值 → 视为语义边界，切块
  - 保证每块长度在 [min_chars, max_chars] 区间

依赖：BGE-small-zh-v1.5（本地已缓存，无 CUDA 走 CPU）
"""
import os
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

# ---- 可调参数 ----
MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 512 维，中文友好
MIN_CHARS = 300       # 块最小字符数（不足则与相邻块合并）
MAX_CHARS = 800       # 块最大字符数（硬上限，逐句贪心保证不超）
HARD_CHARS = 400      # 触发强制切断的软上限（优先在此切，给语义边界留余地）
TARGET_PERCENTILE = 15  # 用相似度的第 15 百分位作为动态阈值（比之前更敏感）

def split_sentences(text: str) -> list[str]:
    """把文本切分成句子（按句尾标点 + 换行）。"""
    text = re.sub(r"\n+", " ", text)  # 换行合并为空格
    # 中英文句尾标点
    parts = re.split(r"(?<=[。！？!?；;\.])\s+", text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents

def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"  加载 embedding 模型: {model_name} (CPU)")
    model = SentenceTransformer(model_name)
    return model

def semantic_chunk(text: str, model: SentenceTransformer,
                   min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS,
                   percentile: int = TARGET_PERCENTILE,
                   encode_batch: int = 64) -> list[str]:
    """对单篇文本做语义分块（分批编码，控制内存）。"""
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) == 1:
        return [sents[0]]

    # 分批逐句编码（避免小内存机器一次性编码全部句子导致 OOM）
    embeddings = []
    for s in range(0, len(sents), encode_batch):
        batch = sents[s:s + encode_batch]
        emb = model.encode(batch, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=32)
        embeddings.append(emb)
    emb = np.vstack(embeddings)

    # 相邻句余弦相似度
    sims = []
    for i in range(len(emb) - 1):
        s = float(np.dot(emb[i], emb[i + 1]))
        sims.append(s)

    # 动态阈值：低于第 percentile 百分位的相似度 -> 语义边界
    threshold = np.percentile(sims, percentile) if sims else 0.5
    boundary_set = {i + 1 for i, s in enumerate(sims) if s < threshold}

    # 逐句贪心切块：
    #  1. 遇到语义边界，且当前块已 >= min_chars -> 在此切块
    #  2. 当前块达到 HARD_CHARS（软上限） -> 优先在下一句边界切
    #  3. 达到 MAX_CHARS（硬上限） -> 强制切（逐句累计，绝不超限）
    chunks = []
    cur = []
    cur_len = 0
    for idx, sent in enumerate(sents):
        sl = len(sent)
        # 硬上限：当前块再塞这句就超，先结算
        if cur_len + sl > MAX_CHARS and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += sl
        # 软切：达到软上限 或 遇到语义边界，且块已够长
        hit_boundary = idx in boundary_set
        if cur_len >= HARD_CHARS or (hit_boundary and cur_len >= MIN_CHARS):
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))

    # 合并过短的尾部块到前一块
    merged = []
    for c in chunks:
        if merged and len(c) < MIN_CHARS:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)

    # 最终过滤
    merged = [c.strip() for c in merged if c.strip()]
    return merged

def chunk_pipeline(pdf_dir: str, model: SentenceTransformer,
                   out_dir: str, encode_batch: int = 64,
                   start_from: int = 0) -> dict:
    """对 data/papers/ 下所有 PDF 做语义分块。

    内存/容错优化：逐文档处理，每篇独立写一个 JSON 文件，
    支持断点续传（start_from 指定从第几篇开始）。
    """
    import fitz
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    stats = {}
    total = 0
    all_chars = []

    for i, fname in enumerate(pdfs, 1):
        if i < start_from:
            continue
        doc_id = fname[:-4]
        out_path = os.path.join(chunks_dir, f"{doc_id}.json")
        if os.path.exists(out_path):
            print(f"  [{i:2d}/{len(pdfs)}] {fname}: 已存在，跳过", flush=True)
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            n = len(existing)
            stats[fname] = {"pages": "?", "chunks": n}
            total += n
            all_chars.extend(c["chars"] for c in existing)
            continue

        doc = fitz.open(os.path.join(pdf_dir, fname))
        text = "\n".join(page.get_text("text") for page in doc)
        page_count = len(doc)
        doc.close()

        chunks = semantic_chunk(text, model, encode_batch=encode_batch)
        recs = [{"doc_id": doc_id, "file": fname, "chunk_id": ci,
                 "text": c, "chars": len(c)} for ci, c in enumerate(chunks)]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

        stats[fname] = {"pages": page_count, "chunks": len(chunks)}
        total += len(chunks)
        all_chars.extend(len(c) for c in chunks)
        print(f"  [{i:2d}/{len(pdfs)}] {fname}: {page_count}页 -> {len(chunks)} 块", flush=True)

    avg = float(np.mean(all_chars)) if all_chars else 0
    print(f"\n===== 分块完成 =====")
    print(f"总块数: {total}")
    print(f"平均块长: {avg:.0f} 字符")
    print(f"输出目录: {chunks_dir}")
    return {"total_chunks": total, "avg_chars": avg, "stats": stats}

def main():
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base, "..", "data", "papers")
    out_dir = os.path.join(base, "..", "data")

    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    model = load_model()
    chunk_pipeline(pdf_dir, model, out_dir, start_from=start_from)

if __name__ == "__main__":
    main()
