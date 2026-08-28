#!/usr/bin/env python3
"""Week 22 Day 3 — retriever.py：rag-kb 正式混合检索模块

架构（基于 W22 试点结论：Flat 叶子 + RAPTOR 摘要互补，Recall@5 0.778→1.000）：
  查询  ── Flat 叶子检索（现有 HNSW，4521 chunks）──┐
        ── RAPTOR 摘要检索（91 摘要节点）───────────┼─→ RRF 融合 → top-k → 返回来源
                                                     └→ 摘要命中时展开其 child 叶子补充

用法：
  python3 retriever.py "问题" [--k 5] [--flat-only] [--summary-only] [--debug]
"""
import os
import json
import sys

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
INDEX = os.path.join(DATA, "index")


class HybridRetriever:
    """Flat + RAPTOR 摘要混合检索器。"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

        # Flat 路：叶子 HNSW
        self.leaf_index = faiss.read_index(os.path.join(INDEX, "index.faiss"))
        self.metadata = json.load(open(os.path.join(INDEX, "metadata.json")))

        # RAPTOR 路：摘要索引
        self.summary_index = faiss.read_index(os.path.join(INDEX, "raptor_summary.faiss"))
        self.summary_meta = json.load(open(os.path.join(INDEX, "raptor_summary.json")))

        # 摘要 → 叶子映射（供展开 child）
        self.summary_children = {
            m["summary_id"]: m.get("child_ids", []) for m in self.summary_meta
        }

    def _encode(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].astype("float32")

    def flat_search(self, query, k=5):
        """叶子层检索，返回 [(global_id, score, doc_id, chunk_id, is_summary=False)]。"""
        qv = self._encode(query)
        D, I = self.leaf_index.search(qv.reshape(1, -1), k)
        out = []
        for score, gid in zip(D[0], I[0]):
            if gid < 0 or gid >= len(self.metadata):
                continue
            m = self.metadata[gid]
            out.append({
                "gid": int(gid), "score": float(score),
                "doc_id": m["doc_id"], "chunk_id": m["chunk_id"],
                "source": m.get("source", m["doc_id"]), "is_summary": False,
            })
        return out

    def summary_search(self, query, k=5):
        """摘要层检索，返回 [(summary_id, score, doc_id, is_summary=True)]。"""
        qv = self._encode(query)
        D, I = self.summary_index.search(qv.reshape(1, -1), k)
        out = []
        for score, sid in zip(D[0], I[0]):
            if sid < 0 or sid >= len(self.summary_meta):
                continue
            m = self.summary_meta[sid]
            out.append({
                "sid": int(sid), "score": float(score),
                "doc_id": m["doc_id"], "source": m.get("source_doc", m["doc_id"]),
                "is_summary": True, "text": m["text"],
                "children": self.summary_children.get(int(sid), []),
            })
        return out

    def rrf_fuse(self, lists, k=60, delta=60):
        """Reciprocal Rank Fusion：多路 ranked list 融合。"""
        scores = {}
        for ranked in lists:
            for rank, item in enumerate(ranked):
                key = item["gid"] if not item["is_summary"] else ("SUM_" + str(item["sid"]))
                scores[key] = scores.get(key, 0) + 1.0 / (delta + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def hybrid_retrieve(self, query, k=5, flat_k=8, summary_k=8, summary_weight=3.0):
        """混合检索主入口（纯加权 RRF）。

        核心：RRF 基础上，摘要层命中的 doc 获得额外权重提升——
        因为摘要是对主题的概括，命中比叶子词面相似更有意义。
        纯加权 RRF，不做 Flat 保护（保持主题检索强度）。
        精确句子召回场景请用 mode='flat'。
        """
        flat_res = self.flat_search(query, flat_k)
        summ_res = self.summary_search(query, summary_k)

        doc_score = {}   # doc_id -> 加权 RRF 分数
        doc_flat_best = {}   # doc_id -> 最高叶子分（同分排序用）

        # Flat 路
        for rank, r in enumerate(flat_res):
            doc = r["doc_id"]
            doc_score[doc] = doc_score.get(doc, 0) + 1.0 / (60 + rank + 1)
            doc_flat_best[doc] = max(doc_flat_best.get(doc, 0), r["score"])

        # 摘要路：所有摘要命中统一强提权（不做 Flat 支持判定）
        for rank, s in enumerate(summ_res):
            doc = s["doc_id"]
            if doc == "__ROOT__":
                continue
            doc_score[doc] = doc_score.get(doc, 0) + summary_weight / (60 + rank + 1)

        final_docs = sorted(doc_score.keys(),
                            key=lambda d: (-doc_score[d], -doc_flat_best.get(d, 0)))

        return {
            "query": query,
            "flat_top": flat_res,
            "summary_top": summ_res,
            "summary_hits": [s["doc_id"] for s in summ_res if s["doc_id"] != "__ROOT__"],
            "fused_docs": final_docs[:k],
            "k": k,
        }

    def retrieve(self, query, k=5, mode="flat", **kw):
        """双模式检索引擎入口。

        mode='flat'（默认）: 仅 Flat 叶子检索，返回 chunk 级来源，高精度精确召回。
        mode='hybrid'     : Flat + RAPTOR 摘要加权融合，返回 doc 级来源，强主题检索。
        """
        if mode == "hybrid":
            return self.hybrid_retrieve(query, k=k, **kw)
        # flat 默认：返回叶子 top-k，带来源
        qv = self._encode(query)
        D, I = self.leaf_index.search(qv.reshape(1, -1), k)
        flat = []
        for score, gid in zip(D[0], I[0]):
            if gid < 0 or gid >= len(self.metadata):
                continue
            m = self.metadata[gid]
            flat.append({
                "gid": int(gid), "score": float(score),
                "doc_id": m["doc_id"], "chunk_id": m["chunk_id"],
                "source": m.get("source", m["doc_id"]), "is_summary": False,
            })
        docs, seen = [], set()
        for r in flat:
            if r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                docs.append(r["doc_id"])
        return {"query": query, "flat_top": flat, "summary_top": [],
                "summary_hits": [], "fused_docs": docs[:k], "mode": "flat", "k": k}


def main():
    if len(sys.argv) < 2:
        print("用法: python3 retriever.py \"问题\" [--k 5] [--mode flat|hybrid] [--debug]")
        return

    query = sys.argv[1]
    k = 5
    debug = False
    mode = "flat"
    args = sys.argv[2:]
    if "--k" in args:
        k = int(args[args.index("--k") + 1])
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--debug" in args:
        debug = True

    print("加载检索器 ...")
    retriever = HybridRetriever()
    result = retriever.retrieve(query, k=k, mode=mode)

    print(f"\n{'='*60}")
    print(f"🔍 查询: {query} (top-{k}, mode={mode})")
    print(f"{'='*60}")

    if debug:
        print("\n[Flat 叶子 top-5]")
        for r in result["flat_top"][:5]:
            print(f"  {r['score']:.4f}  {r['doc_id']} 块{r['chunk_id']}")
        print("\n[RAPTOR 摘要 top-5]")
        for r in result["summary_top"][:5]:
            extra = "" if r["doc_id"] != "__ROOT__" else " (跨篇根)"
            print(f"  {r['score']:.4f}  {r['doc_id']}{extra}")

    print("\n[混合融合 → 来源文档]")
    for i, doc in enumerate(result["fused_docs"], 1):
        print(f"  {i}. {doc}")


if __name__ == "__main__":
    main()
