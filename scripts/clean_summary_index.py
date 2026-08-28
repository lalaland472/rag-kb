#!/usr/bin/env python3
"""W22 Day 3 — clean_summary_index.py：轻量清洗已生成的 RAPTOR 摘要索引

背景：全量重建（4521 chunks 逐篇 UMAP）在 1GB 内存机器上不可行（内存耗尽 thrash）。
     改用轻量方案：直接清洗已生成的 91 节点摘要索引——
     1. 用垃圾检测过滤致谢/版权/引用等无意义摘要节点
     2. 裁剪保留摘要的致谢尾巴
     3. 重新编码干净的摘要节点，重建摘要索引

用法：python3 clean_summary_index.py
"""
import os
import json
import re
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE, "..", "data", "index")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512

# 复用 build_raptor_index 的清洗逻辑（import 避免重复定义）
sys.path.insert(0, BASE)
from build_raptor_index import GARBAGE_PATTERNS, ACK_HEAVY, clip_summary_text  # noqa: E402


def is_garbage_summary(text: str) -> bool:
    """判断摘要是否是无语义垃圾（致谢/版权/纯引用/arxiv头）。"""
    if not text:
        return True
    if GARBAGE_PATTERNS.search(text[:400]):
        ack_count = len(ACK_HEAVY.findall(text))
        if ack_count >= 2:
            return True
        if len(text.strip()) < 120 and GARBAGE_PATTERNS.search(text[:400]):
            return True
    return False


def main():
    print("=" * 60)
    print("W22 Day 3 — 清洗 RAPTOR 摘要索引（轻量方案）")
    print("=" * 60)

    meta_path = os.path.join(INDEX, "raptor_summary.json")
    faiss_path = os.path.join(INDEX, "raptor_summary.faiss")

    meta = json.load(open(meta_path))
    print(f"清洗前摘要节点: {len(meta)}")

    # 1. 清洗：过滤垃圾 + 裁剪
    cleaned = []
    dropped = 0
    for m in meta:
        text = clip_summary_text(m["text"])
        if is_garbage_summary(text):
            dropped += 1
            continue
        m = dict(m)
        m["text"] = text
        cleaned.append(m)
    print(f"  过滤垃圾节点: {dropped} 个")
    print(f"  保留干净节点: {len(cleaned)} 个")

    # 重新分配 summary_id（保持 child_ids 指向的索引一致需要小心，这里根摘要已裁剪，普通节点保留来源）
    # 注：child_ids 指向的是 summary 索引内的旧 id，清洗后索引重排，需重建映射
    # 简化：本方法保留所有非垃圾节点的顺序，summary_id 重排
    id_map = {}
    for new_id, m in enumerate(cleaned):
        old_id = m["summary_id"]
        id_map[old_id] = new_id

    # 更新 child_ids 引用（指向旧 id，需映射到新 id）
    for m in cleaned:
        if m.get("child_ids"):
            m["child_ids"] = [id_map.get(c, c) for c in m["child_ids"]]
        m["summary_id"] = id_map.get(m["summary_id"], cleaned.index(m))

    # 2. 重编码干净节点
    print(f"\n加载 {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [m["text"] for m in cleaned]
    print(f"编码 {len(texts)} 个摘要节点 ...")
    embs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=32).astype("float32")

    # 3. 重建索引
    index = faiss.IndexHNSWFlat(DIM, 16)
    index.hnsw.efConstruction = 200
    index.add(embs)

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"清洗后摘要节点: {len(cleaned)}")
    print(f"索引: {faiss_path} ({index.ntotal} 条)")
    print(f"元数据: {meta_path}")

    # 4. 抽样展示清洗结果
    print(f"\n=== 清洗后抽样 ===")
    for m in cleaned[:8]:
        print(f"  [{m.get('source_doc','?')}] {m['text'][:70]}")


if __name__ == "__main__":
    main()
