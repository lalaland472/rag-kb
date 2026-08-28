#!/usr/bin/env python3
"""Week 21 Day 5 — evaluate_recall.py：全资料 Recall@5 正式验证报告

评估方案（程序化构造 ground truth，保证公平可复现）：
  1. 从 23 篇文档中按块数占比抽样 N 个 chunk 作为「查询锚点」
  2. query 构造：取锚点 chunk 的关键句（首句前 60 字符，去掉作者/版权噪音）
  3. 黄金答案：与锚点同文档的相邻 chunk 集合（锚点 ±1，含自身）
  4. Recall@5：检索 top-5，命中黄金集合即视为召回成功
  5. 附加统计：MRR、平均首个命中排名、跨文档混淆率

验收标准（来自 month6-plan-v2）：Recall@5 ≥ 80%
"""
import os
import json
import glob
import random
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
N_SAMPLES = 100       # 评估集规模
TOP_K = 5
EF_SEARCH = 128

def load_all_chunks(chunks_dir: str) -> list[dict]:
    """加载全部 chunk，附 global_id（与 index/metadata 顺序一致）。"""
    recs = []
    for f in sorted(glob.glob(os.path.join(chunks_dir, "*.json"))):
        with open(f, encoding="utf-8") as fp:
            recs.extend(json.load(fp))
    for i, r in enumerate(recs):
        r["global_id"] = i
    return recs

def build_query(chunk_text: str) -> str:
    """从 chunk 文本提取 query：首句前 60 字符，剔除明显噪音行。"""
    # 去掉作者行、版权行、空白
    clean = chunk_text.replace("\n", " ").strip()
    # 取第一个完整句子（到 . 或 。 为止），截断到 60 字符
    sent = clean.split(".")[0]
    if len(sent) < 20:  # 首句太短（可能是标题），往后多取一点
        sent = clean[:60]
    return sent[:60].strip()

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "data")
    chunks_dir = os.path.join(data_dir, "chunks")
    index_path = os.path.join(data_dir, "index", "index.faiss")
    meta_path = os.path.join(data_dir, "index", "metadata.json")

    random.seed(42)
    np.random.seed(42)

    print("=" * 64)
    print("Week 21 Day 5 — Recall@5 正式验证报告")
    print("=" * 64)

    print("\n=== 1. 加载 ===")
    index = faiss.read_index(index_path)
    index.hnsw.efSearch = EF_SEARCH
    meta = json.load(open(meta_path, encoding="utf-8"))
    recs = load_all_chunks(chunks_dir)
    assert index.ntotal == len(recs) == len(meta)
    print(f"  索引 {index.ntotal} 块 | efSearch={EF_SEARCH} | 文档 {len(set(r['doc_id'] for r in recs))} 篇")

    # 构建 doc_id -> 该文档所有 global_id 集合（用于跨块建黄金集）
    doc_to_gids = {}
    for r in recs:
        doc_to_gids.setdefault(r["doc_id"], []).append(r["global_id"])

    print("\n=== 2. 构造评估集（抽样锚点 chunk） ===")
    # 按文档块数占比分层抽样
    sampled = []
    for doc_id, gids in doc_to_gids.items():
        n_doc = max(1, int(round(N_SAMPLES * len(gids) / len(recs))))
        sampled.extend(random.sample(gids, n_doc))
    random.shuffle(sampled)
    sampled = sampled[:N_SAMPLES]
    print(f"  抽样 {len(sampled)} 个锚点 chunk")

    print("\n=== 3. 编码并检索 ===")
    model = SentenceTransformer(MODEL_NAME)
    queries = [build_query(recs[g]["text"]) for g in sampled]

    # 分批编码 query
    qv = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    qv = qv.astype("float32")

    t0 = time.perf_counter()
    D, I = index.search(qv, TOP_K)
    search_time = (time.perf_counter() - t0) * 1000

    print("\n=== 4. 计算 Recall@5 / MRR ===")
    recalls = []
    mrr = 0.0
    first_rank_hits = []
    confusion = {}  # 记录错配到的文档
    detail = []

    for gi, (query, anchor_gid) in enumerate(zip(queries, sampled)):
        anchor_doc = recs[anchor_gid]["doc_id"]
        anchor_chunk = recs[anchor_gid]["chunk_id"]
        # 黄金集：同文档的 锚点 ±1 chunk
        gold = {g for g in doc_to_gids[anchor_doc]
                if abs(recs[g]["chunk_id"] - anchor_chunk) <= 1}
        hits = I[gi]
        hit_set = set(int(h) for h in hits)
        hit = len(hit_set & gold) > 0
        recalls.append(1 if hit else 0)

        # MRR：第一个命中黄金集的位置
        rank = None
        for pos, h in enumerate(hits):
            if int(h) in gold:
                rank = pos + 1
                break
        if rank:
            mrr += 1.0 / rank
            first_rank_hits.append(rank)

        # 记录 top-1 文档（看混淆）
        top1_doc = recs[int(hits[0])]["doc_id"]
        if top1_doc != anchor_doc:
            confusion.setdefault(anchor_doc, set()).add(top1_doc)

        detail.append({
            "anchor_doc": anchor_doc, "chunk": anchor_chunk,
            "query": query[:50], "top1_doc": top1_doc,
            "hit": bool(hit), "rank": rank,
        })

    recall_at5 = np.mean(recalls)
    mrr_val = mrr / len(sampled)
    avg_first_rank = np.mean(first_rank_hits) if first_rank_hits else np.nan

    print("\n" + "=" * 64)
    print("=== 验证报告汇总 ===")
    print("=" * 64)
    print(f"  评估集大小:        {len(sampled)} 条 query")
    print(f"  Recall@5:          {recall_at5:.1%}  {'✅ ≥80%' if recall_at5 >= 0.80 else '❌ <80%'}")
    print(f"  MRR:               {mrr_val:.4f}")
    print(f"  平均首个命中排名:  {avg_first_rank:.2f}  (越小越好)")
    print(f"  检索耗时(批次):    {search_time:.1f} ms / {len(sampled)} 条 = {search_time/len(sampled):.2f} ms/q")
    print(f"  跨文档混淆:        {len(confusion)} 篇文档出现 top-1 错配")

    if confusion:
        print("\n  跨文档混淆明细（anchor -> 被错配到的 top-1 文档）:")
        for doc, wrong in sorted(confusion.items()):
            print(f"    {doc:<28} -> {sorted(wrong)}")

    # 写报告文件
    report = {
        "n_queries": len(sampled),
        "recall_at_5": float(recall_at5),
        "mrr": float(mrr_val),
        "avg_first_rank": float(avg_first_rank),
        "search_time_ms": float(search_time),
        "confusion_docs": {k: sorted(v) for k, v in confusion.items()},
        "detail": detail,
    }
    out = os.path.join(data_dir, "recall_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已写入: {out}")

    print("\n=== Day 5 完成 ===")

if __name__ == "__main__":
    main()
