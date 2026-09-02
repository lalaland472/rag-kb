#!/usr/bin/env python3
"""GraphRAG 全局问答 —— 最终验收（写入文件版，带重试）"""
import os, json, urllib.request, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

CFG = {}
with open("config/credentials.env") as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            CFG[k.strip()] = v.strip().strip('"').strip("'")
API = CFG["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
AUTH = "Bearer " + CFG["LLM_API_KEY"]
MODEL = CFG["LLM_MODEL"]

def call(p, mt=700):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": mt, "temperature": 0.2}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"].strip()
            if out:
                return out
        except Exception:
            pass
        time.sleep(1.5)
    return ""

d = json.load(open("data/graph/community_summaries.json"))
sums = d["communities"]
query = "这个语料库整体讲了哪些 AI 技术主题？这些主题之间什么关系？RAG 和 Agent 各自处于什么位置？"
sel = ['3', '19', '4', '22', '18', '2']  # RAG + GraphRAG + Agent(AutoGen/MathChat/GenAgents) + LoRA

partials = []
for cid in sel:
    summ = sums[cid]["summary"]
    p = ("基于下面社区摘要，从本社区角度回答全局问题，1-2句：\n"
         "问题:" + query + "\n社区摘要:" + summ + "\n部分回答:\n")
    partials.append(call(p, 250))

parts = "\n\n".join("[主题%d] %s" % (i + 1, p) for i, p in enumerate(partials) if p.strip())
final_prompt = ("你是总结助手。整合以下对同一全局问题的多个部分回答，成一份连贯有条理的最终答案，"
    "覆盖主要主题并说明关系，中文分点。\n问题:" + query + "\n各部分:\n" + parts + "\n最终答案:\n")
final = call(final_prompt, 900)

print("=" * 55)
print("GraphRAG 全局问答 —— 最终验收答案")
print("=" * 55)
print(final if final else "(Reduce 返回空，fallback 部分回答)\n" + parts)
