#!/usr/bin/env python3
"""Week 21 Day 1 — 批量下载 arxiv 论文 PDF 到 data/papers/"""
import os
import time
import urllib.request

# arxiv ID -> 论文标题（用于文件命名）
PAPERS = {
    # Month 6 要精读的 4 篇
    "2312.10997": "RAG_Survey_2024",
    "2307.03172": "Lost_in_the_Middle_2023",
    "2310.11511": "Self_RAG_2023",
    "2404.16130": "GraphRAG_2024",
    # RAG 基础 / 检索
    "2005.11401": "RAG_Lewis_2020",
    "1702.08734": "FAISS_2019",
    "2212.10496": "HyDE_2022",
    "2401.18059": "RAPTOR_2024",
    "1810.04805": "BERT_2018",
    "1706.03762": "Attention_Is_All_You_Need_2017",
    # LLM 基础 / 训练 / 对齐
    "2106.09685": "LoRA_2021",
    "2203.02155": "InstructGPT_2022",
    "2305.18290": "DPO_2023",
    "2005.14165": "GPT3_2020",
    # 推理 / 部署 / 评估
    "2205.14135": "FlashAttention_2022",
    "2210.17323": "GPTQ_2023",
    "2306.01337": "AWQ_2023",
    "2309.06180": "vLLM_2023",
    "2306.05685": "MT_Bench_2023",
    # Agent 方向
    "2210.03629": "ReAct_2022",
    "2308.08155": "AutoGen_2023",
    "2305.10601": "Tree_of_Thoughts_2023",
    "2304.03442": "Generative_Agents_2023",
    # 长上下文 / 检索增强前沿
    "2305.14327": "RAPTOR_alternative_not_used",
}

PAPERS.pop("2305.14327", None)  # 移除占位（非真实论文 ID）

PDF_URL = "https://arxiv.org/pdf/{}.pdf"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
DATA_DIR = os.path.abspath(DATA_DIR)

def download(arxiv_id: str, name: str) -> str:
    fname = f"{name}.pdf"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"  ⏭ 已存在，跳过: {fname}")
        return path
    url = PDF_URL.format(arxiv_id)
    print(f"  ⬇ 下载 {name} <- {arxiv_id}")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            size = os.path.getsize(path)
            print(f"    ✅ {fname} ({size//1024} KB)")
            return path
        except Exception as e:
            print(f"    ⚠️ 第{attempt+1}次失败: {e}")
            time.sleep(3)
    return ""

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"目标目录: {DATA_DIR}\n共 {len(PAPERS)} 篇论文\n")
    ok = fail = 0
    for aid, name in PAPERS.items():
        p = download(aid, name)
        if p:
            ok += 1
        else:
            fail += 1
        time.sleep(1)  # 礼貌间隔，避免被封
    print(f"\n===== 完成：成功 {ok} 篇 / 失败 {fail} 篇 =====")

if __name__ == "__main__":
    main()
