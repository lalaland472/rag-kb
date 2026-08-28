#!/usr/bin/env python3
"""Week 21 Day 1 — load_pdfs() 批量解析 data/papers/ 下的 PDF 为文本"""
import os
import json

import fitz  # PyMuPDF

def load_pdfs(pdf_dir: str) -> list[dict]:
    """扫描目录，解析所有 PDF，返回 [{file, title, pages, chars, text}]"""
    results = []
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    for i, fname in enumerate(pdfs, 1):
        path = os.path.join(pdf_dir, fname)
        try:
            doc = fitz.open(path)
            page_count = len(doc)
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            text = "\n\n".join(pages)
            results.append({
                "file": fname,
                "title": fname[:-4],
                "pages": page_count,
                "chars": len(text),
                "text": text,
            })
            doc.close()
            print(f"  [{i:2d}/{len(pdfs)}] {fname}: {page_count} 页, {len(text):,} 字符")
        except Exception as e:
            print(f"  [FAIL] {fname}: {e}")
            results.append({
                "file": fname, "title": fname[:-4],
                "pages": 0, "chars": 0, "text": "", "error": str(e),
            })
    return results

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base, "..", "data", "papers")
    out_dir = os.path.join(base, "..", "data")

    docs = load_pdfs(pdf_dir)
    total_chars = sum(d["chars"] for d in docs)
    total_pages = sum(d["pages"] for d in docs)

    # 保存解析结果（暂存文本，供后续 ingest.py 使用）
    out_json = os.path.join(out_dir, "parsed_docs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\n===== 解析完成 =====")
    print(f"论文数: {len(docs)} 篇")
    print(f"总页数: {total_pages} 页")
    print(f"总字符: {total_chars:,} 字符")
    print(f"输出: {out_json}")

if __name__ == "__main__":
    main()
