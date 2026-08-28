#!/usr/bin/env python3
"""Week 23 Day 2 — citation_pipeline.py：来源引用链（chunk→文件→页面）

背景：
  chunk_pipeline 切块时用 "\n".join(page.get_text) 拼掉了页面信息（源码136-137行），
  chunk 与页码的映射丢失。本模块用 PyMuPDF 重新提取分页文本，对每个 chunk 的
  文本片段做滑窗匹配，还原「chunk → 文件 → 页面」的引用链。

用法：
  python3 scripts/citation_pipeline.py build          # 扫描全库，生成 chunk_page_map.json
  python3 scripts/citation_pipeline.py query <doc> <chunk_id>   # 查单个 chunk 的页号
  python3 scripts/citation_pipeline.py stats          # 定位成功率统计
"""
import os
import re
import sys
import json
import glob

import fitz  # PyMuPDF

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
DATA = os.path.join(ROOT, "data")
PAPERS = os.path.join(DATA, "papers")
CHUNKS = os.path.join(DATA, "chunks")
MAP_FILE = os.path.join(DATA, "index", "chunk_page_map.json")


def _norm(s):
    """归一化：换行/多空格折叠为单空格，去掉空白差异。"""
    return re.sub(r"\s+", " ", s).strip()


def _extract_pages(pdf_path):
    """返回每页归一化文本列表。"""
    doc = fitz.open(pdf_path)
    pages = [_norm(p.get_text("text")) for p in doc]
    doc.close()
    return pages


def locate_chunk(pages, chunk_text, min_words=18):
    """在分页文本里定位 chunk 的页号。

    策略：先用 chunk 开头的 min_words 个词做精确探针搜索；
    若多页命中，再用更长的词窗逐页比对，取重叠最多的页。
    返回 (page_index, confidence) 或 (None, 0.0)。
    """
    norm_chunk = _norm(chunk_text)
    words = norm_chunk.split()
    if len(words) < 5:
        return None, 0.0

    # 多档探针长度，逐档收紧
    for wlen in (min_words, 30, 50, 80):
        probe = " ".join(words[:wlen])
        if len(probe) < 20:
            continue
        matches = []
        for pi, pt in enumerate(pages):
            if probe in pt:
                matches.append(pi)
        if len(matches) == 1:
            return matches[0], 1.0
        if len(matches) > 1:
            # 多页命中：用最长重合判定
            best_pi, best_ov = None, 0
            for pi in matches:
                ov = _overlap_words(pt, norm_chunk)
                if ov > best_ov:
                    best_ov, best_pi = ov, pi
            if best_pi is not None:
                return best_pi, min(1.0, best_ov / max(1, len(words)))

    # 探针全部失败：退化为逐页整体重叠取最大
    best_pi, best_ov = None, 0
    for pi, pt in enumerate(pages):
        ov = _overlap_words(pt, norm_chunk)
        if ov > best_ov:
            best_ov, best_pi = ov, pi
    if best_pi is not None and best_ov >= max(5, 0.2 * len(words)):
        return best_pi, min(1.0, best_ov / max(1, len(words)))
    return None, 0.0


def _overlap_words(text_a, text_b):
    """两段文本的词级重叠数（用于多页命中判定和退化匹配）。"""
    wa = text_a.split()
    wb = text_b.split()
    if not wa or not wb:
        return 0
    # 用开头窗口匹配（chunk 排在页文本里通常是连续的）
    n = min(len(wa), len(wb))
    score = sum(1 for i in range(n) if wa[i] == wb[i])
    return score


def build():
    """扫描全库 chunk，重建 chunk_page_map.json。"""
    map_data = {}
    stats = {"total": 0, "hit": 0, "miss": 0, "by_doc": {}}
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS, "*.json")))

    for cf in chunk_files:
        doc_id = os.path.basename(cf)[:-5]
        # 找对应 PDF
        pdf_path = os.path.join(PAPERS, doc_id + ".pdf")
        if not os.path.exists(pdf_path):
            stats["by_doc"][doc_id] = {"chunks": 0, "hit": 0}
            continue
        pages = _extract_pages(pdf_path)
        chunks = json.load(open(cf))
        doc_entry = {}
        doc_stat = {"chunks": len(chunks), "hit": 0}
        for c in chunks:
            cid = c["chunk_id"]
            text = c.get("text", "")
            page, conf = locate_chunk(pages, text)
            doc_entry[str(cid)] = {"page": page, "confidence": round(conf, 3)}
            if page is not None:
                doc_stat["hit"] += 1
                stats["hit"] += 1
            else:
                stats["miss"] += 1
        map_data[doc_id] = {
            "file": doc_id + ".pdf", "total_pages": len(pages),
            "chunk_pages": doc_entry,
        }
        stats["by_doc"][doc_id] = doc_stat
        stats["total"] += len(chunks)
        print(f"  {doc_id}: {doc_stat['hit']}/{doc_stat['chunks']} 定位")

    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 引用映射已写入 {MAP_FILE}")
    print(f"总体定位成功率: {stats['hit']}/{stats['total']} = "
          f"{stats['hit'] / max(1, stats['total']) * 100:.1f}%")
    return stats


def load_map():
    if not os.path.exists(MAP_FILE):
        return {}
    return json.load(open(MAP_FILE))


def query(doc_id, chunk_id):
    m = load_map().get(doc_id, {})
    cpage = m.get("chunk_pages", {}).get(str(chunk_id))
    return {
        "doc_id": doc_id,
        "file": m.get("file"),
        "chunk_id": chunk_id,
        "page": cpage["page"] if cpage else None,
        "confidence": cpage["confidence"] if cpage else 0,
        "total_pages": m.get("total_pages"),
    }


def display_ref(doc_id, chunk_id):
    """给 generate_answer 用的显示引用：'论文名 · 第N页'。"""
    q = query(doc_id, chunk_id)
    name = doc_id
    for suf in ("_2024", "_2023", "_2022", "_2021", "_2020", "_2019", "_2018", "_2017"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    name = name.replace("_", " ")
    if q["page"] is not None:
        return f"{name} · 第{q['page'] + 1}页"
    return f"{name}"


def stats():
    if not os.path.exists(MAP_FILE):
        return "映射文件不存在，先运行 build"
    m = load_map()
    total = hit = 0
    for doc, d in m.items():
        cp = d.get("chunk_pages", {})
        total += len(cp)
        hit += sum(1 for v in cp.values() if v["page"] is not None)
    return f"已建映射: {len(m)} 篇 | chunk 定位 {hit}/{total} = {hit / max(1, total) * 100:.1f}%"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "query":
        q = query(sys.argv[2], int(sys.argv[3]))
        print(json.dumps(q, ensure_ascii=False, indent=2))
        print("display:", display_ref(sys.argv[2], int(sys.argv[3])))
    elif cmd == "stats":
        print(stats())


if __name__ == "__main__":
    main()
