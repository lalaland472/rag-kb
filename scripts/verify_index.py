#!/usr/bin/env python3
"""Week 21 Day 4 — verify_index.py：FAISS HNSW 索引验证 + efSearch 调优 + 重新持久化

Day 4 任务：
  1. 加载 Day 3 构建的 IndexHNSWFlat，验证索引-元数据对齐
  2. 调优 efSearch（默认 16 偏低，召回受损）→ 128，重新持久化
  3. 检索健康检查：多个代表性 query 跑 top-k，检查结果语义相关性
  4. 量化检索延迟（<50ms 成功标准）

产出：data/index/index.faiss（含调优后 efSearch）+ 验证报告
"""
import os
import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EF_SEARCH = 128   # 调优后的查询阶段连接广度

# 代表性 query（跨多主题论文，中文/英文混合，考察索引覆盖度）
TEST_QUERIES = [
    ("What is retrieval augmented generation?", "RAG_Lewis_2020"),
    ("低秩适配如何微调大语言模型？", "LoRA_2021"),
    ("FlashAttention 如何加速 transformer？", "FlashAttention_2022"),
    ("如何在模型中做人类反馈对齐？", "InstructGPT_2022"),
    ("树形层次检索如何总结长文档？", "RAPTOR_2024"),
    ("知识图谱如何做全局问答？", "GraphRAG_2024"),
]


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "data")
    index_path = os.path.join(data_dir, "index", "index.faiss")
    meta_path = os.path.join(data_dir, "index", "metadata.json")

    print("=" * 60)
    print("=== 1. 加载索引 & 元数据 ===")
    index = faiss.read_index(index_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    assert index.ntotal == len(meta), "索引与元数据条数不一致！"
    print(f"  类型: {type(index).__name__}  维度={index.d}  ntotal={index.ntotal}")
    print(f"  元数据: {len(meta)} 条  ✅ 对齐")

    print("\n=== 2. efSearch 调优 ===")
    old_ef = index.hnsw.efSearch
    index.hnsw.efSearch = EF_SEARCH
    print(f"  efSearch: {old_ef} -> {EF_SEARCH}")

    # 重新持久化（带上调优后的 efSearch）
    faiss.write_index(index, index_path)
    print(f"  已重新持久化 -> {index_path}")

    print("\n=== 3. 加载 embedding 模型 ===")
    model = SentenceTransformer(MODEL_NAME)

    print("\n=== 4. 检索健康检查 ===")
    print(f"{'query':<45} | {'top-1 来源':<22} | 距离")
    print("-" * 90)
    hit = 0
    latencies = []
    for q, expected_doc in TEST_QUERIES:
        t0 = time.perf_counter()
        qv = model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, k=5)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        top_docs = [meta[i]["doc_id"] for i in I[0]]
        ok = any(expected_doc == d for d in top_docs[:5])
        hit += 1 if ok else 0
        top1 = meta[I[0][0]]["doc_id"]
        mark = "✅" if ok else "❌"
        print(f"{q[:44]:<45} | {top1:<22} | {D[0][0]:.4f}  {mark}")

    print("\n=== 5. 结果汇总 ===")
    recall = hit / len(TEST_QUERIES)
    avg_lat = np.mean(latencies)
    print(f"  命中率 (top-5 含预期来源): {hit}/{len(TEST_QUERIES)} = {recall:.0%}")
    print(f"  平均检索延迟: {avg_lat:.1f} ms  {'✅ <50ms' if avg_lat < 50 else '⚠️ 超标'}")
    print(f"  efSearch 最终值: {index.hnsw.efSearch}")
    print("\n=== Day 4 完成 ✅ ===")


if __name__ == "__main__":
    main()
