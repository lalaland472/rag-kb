#!/usr/bin/env python3
"""全局问答 head-to-head 小评测：RAG(flat) vs GraphRAG —— 方法二

方法一（graphrag_benchmark.py）用【关键词覆盖度】量化回答提到了几个期望主题；
本脚本是【方法二】：LLM-as-Judge 盲评（head-to-head），按论文评测惯例用三标准给分。

三标准（论文用）：
  1. 忠实性 Faithfulness  —— 是否严格基于语料/材料，无幻觉、无凭空捏造   (0-10)
  2. 完整性 Completeness  —— 是否覆盖全局问题的全部关键维度，不片面     (0-10)
  3. 可溯源 Traceability  —— 归纳是否清晰、结构化、有依据可查（社区/层次） (0-10)

流程：
  1. 对每条全局问题分别生成 flat 与 graphrag 回答
  2. 打乱顺序、匿名（blind）交给 judge 按三标准逐条打分
  3. 汇总每标准均值 + 每局胜负（总分对比）→ 输出报告

用法：python3 scripts/graphrag_head2head.py
"""
import os, sys, json, time, random, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

# 复用 graphrag_benchmark 的全局测试集 + 回答生成器
from graphrag_benchmark import GLOBAL_TEST, _answer_flat, _answer_graphrag

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
# judge 用更强的模型（可 --judge 覆盖），回答生成沿用配置默认模型
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", CFG.get("JUDGE_MODEL", "deepseek-v4-pro"))

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--judge", default=JUDGE_MODEL, help="judge 模型，默认用更强模型 deepseek-v4-pro")
_p.add_argument("--all", action="store_true", help="对 judge 解析失败的题重试")
ARGS = _p.parse_args()
JUDGE_MODEL = ARGS.judge

def call_llm(prompt, mt=900, temp=0.2, model=None, disable_thinking=False):
    mdl = model or MODEL
    body = {"model": mdl, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": temp}
    # 推理型模型（deepseek-v4-pro）：关闭 thinking，否则 reasoning_content 吃光预算、content 为空
    if disable_thinking:
        body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": AUTH})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"].strip()
            if out:
                return out
        except Exception:
            pass
        time.sleep(1.5)
    return ""

JUDGE_SYSTEM = """你是一位严谨的 RAG 系统评测专家。下面有两个匿名系统（系统A、系统B）对一个"面向整个语料库的全局问题"的回答。
请按三个标准分别给两个回答打分（0-10，可带小数）：
1. 忠实性 Faithfulness：回答是否严格基于语料材料，无幻觉、无凭空捏造
2. 完整性 Completeness：是否覆盖问题的全部关键维度，不片面、不遗漏
3. 可溯源 Traceability：归纳是否清晰、结构化、层次分明、有依据可查

只输出 JSON，格式：
{"A": {"faithfulness": x, "completeness": y, "traceability": z},
 "B": {"faithfulness": a, "completeness": b, "traceability": c},
 "winner": "A"|"B"|"tie",
 "reason": "一句话说明胜负原因"}"""

def judge(q, ans_a, ans_b):
    prompt = f"【全局问题】\n{q}\n\n【系统A回答】\n{ans_a or '(空)'}\n\n【系统B回答】\n{ans_b or '(空)'}"
    # 推理型 judge（如 deepseek-v4-pro）：需关闭 thinking，否则 reasoning 吃光 token、content 为空
    disable_thinking = "pro" in JUDGE_MODEL
    judge_mt = 1500
    out = call_llm(f"{JUDGE_SYSTEM}\n\n{prompt}", mt=judge_mt, temp=0.2, model=JUDGE_MODEL, disable_thinking=disable_thinking)
    if not out:
        return None
    import re
    # 去掉可能的 markdown 代码围栏
    out = re.sub(r"```(?:json)?", "", out).strip()
    try:
        # 提取 JSON 块
        s = out[out.find("{"):out.rfind("}") + 1]
        return json.loads(s)
    except Exception:
        # 宽松回退：逐行尝试找到完整 JSON 对象
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except Exception:
                    continue
        return None

def main():
    print(f"全局问答 head-to-head 盲评：RAG(flat) vs GraphRAG | judge={JUDGE_MODEL}")
    print(f"三标准：忠实性/完整性/可溯源 (0-10) | {len(GLOBAL_TEST)} 条全局问题")
    print("=" * 76)
    rows = []
    agg = {"flat": [0.0, 0.0, 0.0], "graphrag": [0.0, 0.0, 0.0]}
    win = {"flat": 0, "graphrag": 0, "tie": 0}
    for qid, q, topics in GLOBAL_TEST:
        # 生成回答
        ans_f, _ = _answer_flat(q)
        ans_g, _ = _answer_graphrag(q)
        # 盲评：随机打乱匿名身份
        labeled = [("flat", ans_f), ("graphrag", ans_g)]
        random.shuffle(labeled)
        name_a, ans_a = labeled[0]
        name_b, ans_b = labeled[1]
        res = judge(q, ans_a, ans_b)
        if not res:
            print(f"[{qid}] judge 解析失败，跳过")
            continue
        # 映射回真实身份
        def scores(name):
            k = "A" if name == name_a else "B"
            s = res[k]
            return [float(s.get("faithfulness", 0)), float(s.get("completeness", 0)), float(s.get("traceability", 0))]
        fs = scores("flat")
        gs = scores("graphrag")
        for i in range(3):
            agg["flat"][i] += fs[i]
            agg["graphrag"][i] += gs[i]
        winner = {"A": name_a, "B": name_b, "tie": "tie"}.get(res.get("winner"), "tie")
        if winner == "flat":
            win["flat"] += 1
        elif winner == "graphrag":
            win["graphrag"] += 1
        else:
            win["tie"] += 1
        rows.append({"id": qid, "q": q,
                     "flat": {"scores": fs, "excerpt": ans_f[:70]},
                     "graphrag": {"scores": gs, "excerpt": ans_g[:70]},
                     "winner": winner, "reason": res.get("reason", "")})
        print(f"[{qid}] flat={fs[0]:.0f}/{fs[1]:.0f}/{fs[2]:.0f} "
              f"graphrag={gs[0]:.0f}/{gs[1]:.0f}/{gs[2]:.0f} → {'flat' if winner=='flat' else 'graphrag' if winner=='graphrag' else '平'}")

    print("=" * 76)
    n = len(rows) or 1
    print(f"\n{'标准':<8}{'RAG(flat)':<12}{'GraphRAG':<12}{'差值'}")
    names = ["忠实性", "完整性", "可溯源"]
    diff_total = 0
    for i, nm in enumerate(names):
        f = agg["flat"][i] / n
        g = agg["graphrag"][i] / n
        diff_total += (g - f)
        print(f"{nm:<8}{f:<12.1f}{g:<12.1f}{g - f:+.1f}")
    print(f"\n总分均值  flat={sum(agg['flat'])/n:.1f}  graphrag={sum(agg['graphrag'])/n:.1f}  ({'+' if diff_total>=0 else ''}{diff_total:.1f})")
    print(f"胜负     flat {win['flat']} 胜 / graphrag {win['graphrag']} 胜 / 平 {win['tie']}")
    print(f"\n评价小结：GraphRAG 在三标准下{'全面占优' if (win['graphrag']>win['flat'] and diff_total>0) else '与 flat 互有胜负' if win['graphrag']==win['flat'] else '略处下风'}")

    out = "data/graph/head2head.json"
    json.dump({"model": MODEL, "criteria": names, "n": len(rows),
               "avg": {"flat": [round(agg['flat'][i]/n,2) for i in range(3)],
                       "graphrag": [round(agg['graphrag'][i]/n,2) for i in range(3)]},
               "total": {"flat": round(sum(agg['flat'])/n,2),
                         "graphrag": round(sum(agg['graphrag'])/n,2)},
               "wins": win, "rows": rows},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")

if __name__ == "__main__":
    main()
