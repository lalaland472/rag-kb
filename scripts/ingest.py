#!/usr/bin/env python3
"""Week 21 Day 3 — ingest.py：扫描 chunks → BGE 编码 → FAISS HNSW 索引 → 来源映射

流程：
  1. 扫描 data/chunks/*.json 的所有分块
  2. 用 BGE-small-zh-v1.5 逐块编码（512 维，分批控制内存）
  3. 构建 FAISS HNSW 索引（M=16）
  4. 持久化：index.faiss + 元数据 metadata.json（chunk→文件→页面映射）
"""
import os
import json
import glob

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512            # BGE-small 输出维度
HNSW_M = 16          # HNSW 连接数
EF_CONSTRUCTION = 200
EMBED_BATCH = 64     # 编码批次（控制内存）

def load_chunks(chunks_dir: str) -> list[dict]:
    """加载所有分块，返回有序列表，并按 doc_id 关联页码信息。"""
    recs = []
    for f in sorted(glob.glob(os.path.join(chunks_dir, "*.json"))):
        with open(f, encoding="utf-8") as fp:
            recs.extend(json.load(fp))
    # 给每条记录加全局 chunk 序号
    for i, r in enumerate(recs):
        r["global_id"] = i
        # 页面映射占位：当前分块粒度未记录页码，先用 doc_id 作为来源主键
        r["source"] = r["file"]
    return recs

def encode_chunks(model: SentenceTransformer, texts: list[str],
                  batch: int = EMBED_BATCH) -> np.ndarray:
    """分批编码，避免一次把 4521 条塞进内存（虽然 8.8MB 不大，仍稳妥）。"""
    embs = []
    for i in range(0, len(texts), batch):
        vec = model.encode(texts[i:i + batch], normalize_embeddings=True,
                           show_progress_bar=False, batch_size=32)
        embs.append(vec.astype("float32"))
    return np.vstack(embs)

def build_hnsw_index(vectors: np.ndarray, m: int = HNSW_M,
                     ef_construction: int = EF_CONSTRUCTION) -> faiss.Index:
    """构建 L2 归一化 + HNSW 索引（向量已 normalize，L2 等价余弦）。"""
    index = faiss.IndexHNSWFlat(DIM, m)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "data")
    chunks_dir = os.path.join(data_dir, "chunks")
    index_dir = os.path.join(data_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    print(f"=== 1. 加载分块 ===")
    recs = load_chunks(chunks_dir)
    texts = [r["text"] for r in recs]
    print(f"  共 {len(recs)} 块")

    print(f"=== 2. BGE 编码 ({MODEL_NAME}) ===")
    model = SentenceTransformer(MODEL_NAME)
    vectors = encode_chunks(model, texts)
    print(f"  向量矩阵: {vectors.shape} (float32, {vectors.nbytes/1024/1024:.1f} MB)")

    print(f"=== 3. 构建 FAISS HNSW 索引 (M={HNSW_M}) ===")
    index = build_hnsw_index(vectors)
    print(f"  索引规模: {index.ntotal} 条")

    print(f"=== 4. 持久化 ===")
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    metadata = [{
        "global_id": r["global_id"],
        "doc_id": r["doc_id"],
        "file": r["file"],
        "chunk_id": r["chunk_id"],
        "source": r["source"],
        "chars": r["chars"],
    } for r in recs]

    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"块数: {len(recs)}")
    print(f"索引: {os.path.join(index_dir, 'index.faiss')}")
    print(f"元数据: {os.path.join(index_dir, 'metadata.json')}")

    # 快速自检：检索一个 query 验证索引可用
    print(f"\n=== 自检（索引检索） ===")
    q = "What is retrieval augmented generation?"
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k=3)
    for d, i in zip(D[0], I[0]):
        print(f"  top-{i}: 距离={d:.4f} -> {metadata[i]['doc_id']} 块{metadata[i]['chunk_id']}")

if __name__ == "__main__":
    main()
