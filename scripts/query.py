#!/usr/bin/env python3
"""Week 23 Day 3 — query.py：rag-kb 问答 CLI（交互式）

把 generate_answer 封装成正式问答界面：
  - 单次查询：python3 scripts/query.py "问题" [--mode flat|hybrid] [--k 5] [--style strict|chat|default]
  - 交互会话：python3 scripts/query.py [--interactive]
  - 输出：回答 + 引用来源（论文名 · 第N页）

⚠️ Self-RAG 启发（W23 Day 6）：
  --style strict    → IsSup 拉满，每句可核验事实都要有依据，末尾附忠实性核对
  --style chat      → IsUse 拉高，允许适度发散
  --retrieval-check → Retrieve on-demand 预判：纯常识/闲聊 query 直接 LLM-only 跳过检索
  --llm-only        → 强制不检索（验证对比用）

退出交互会话：输入 q / quit / exit / 退出。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from generate_answer import generate_answer

BANNER = """
╔══════════════════════════════════════════════════╗
║   RAG-KB · 个人 AI 论文知识库问答                ║
║   20+ 篇论文 · 本地检索 · 生成带来源引用          ║
╚══════════════════════════════════════════════════╝"""

HELP = """可用命令：
  输入问题开始问答
  /mode flat|hybrid   切换检索模式（当前: {mode}）
  /k <数字>           设置检索条数（当前: {k}）
  /clear              清屏
  q | quit | exit     退出
  help                显示本帮助"""


def format_answer(res):
    """把 generate_answer 结果格式化成可读文本。"""
    answer = res["answer"]
    meta = res["meta"]
    cites = meta["citations"]

    lines = []
    lines.append("\n" + "─" * 56)
    lines.append("📝 回答")
    lines.append("─" * 56)
    lines.append(answer)
    if cites:
        lines.append("")
        lines.append("📚 引用来源")
        for c in cites:
            if c["chunk_id"] is not None:
                loc = f"第{c['page'] + 1}页" if c.get("page") is not None else f"块{c['chunk_id']}"
            else:
                loc = "(摘要)"
            lines.append(f"  [{c['index']}] {c['display']}  [{loc}]")
    lines.append("")
    extra = f" · style={meta.get('style')}" if meta.get("style") else ""
    if not meta.get("retrieval_needed", True):
        extra += " · ⚡跳过检索(LLM-only)"
    if meta.get("fidelity_checked"):
        extra += " · ⚠️已忠实核对"
    lines.append(f"  ⏱ {meta.get('latency', 0):.1f}s · mode={meta['mode']}{extra} · "
                 f"chunk={meta['chunk_count']}")
    return "\n".join(lines)


def ask_once(question, mode, k, interactive=False, style="default", retrieval_check=False, llm_only=False, no_rerank=False):
    """单次问答，带耗时统计。"""
    t0 = time.time()
    res = generate_answer(question, mode=mode, k=k, style=style,
                          retrieval_check=retrieval_check, llm_only=llm_only,
                          rerank=not no_rerank)
    res["meta"]["latency"] = time.time() - t0
    if interactive:
        print(format_answer(res))
    return res


def interactive_loop(default_mode, default_k, style="default", retrieval_check=False, llm_only=False):
    """交互式问答会话。"""
    mode, k = default_mode, default_k
    print(BANNER)
    print(HELP.format(mode=mode, k=k))
    print()
    while True:
        try:
            q = input("\n❓ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit", "退出"):
            print("再见！")
            break
        if q.lower() in ("help", "帮助"):
            print(HELP.format(mode=mode, k=k))
            continue
        if q.lower() == "/clear":
            os.system("clear" if os.name == "posix" else "cls")
            print(BANNER)
            continue
        if q.startswith("/mode"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in ("flat", "hybrid"):
                mode = parts[1]
                print(f"✅ 检索模式切换到: {mode}")
            else:
                print("用法: /mode flat|hybrid")
            continue
        if q.startswith("/k"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = max(1, min(20, int(parts[1])))
                print(f"✅ 检索条数设置为: {k}")
            else:
                print("用法: /k <1-20>")
            continue
        if q.startswith("/style") :
            parts = q.split()
            if len(parts) == 2 and parts[1] in ("strict", "chat", "default"):
                style = parts[1]
                print(f"✅ 生成风格切换到: {style}")
            else:
                print("用法: /style strict|chat|default")
            continue
        if q.startswith("/rc"):
            retrieval_check = not retrieval_check
            print(f"✅ 按需检索预判(Retrieve on-demand): {'开' if retrieval_check else '关'}")
            continue
        try:
            ask_once(q, mode, k, interactive=True, style=style,
                     retrieval_check=retrieval_check, llm_only=llm_only)
        except Exception as e:
            print(f"⚠️ 出错: {e}")


def main():
    args = sys.argv[1:]
    # 支持 --mode / --k / --style / --retrieval-check / --llm-only / --interactive
    mode, k = "flat", 5
    style, retrieval_check, llm_only, no_rerank = "default", False, False, False
    interactive = False
    positionals = []

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif a == "--k" and i + 1 < len(args):
            k = int(args[i + 1])
            i += 2
        elif a == "--style" and i + 1 < len(args):
            style = args[i + 1]
            i += 2
        elif a == "--retrieval-check":
            retrieval_check = True
            i += 1
        elif a == "--llm-only":
            llm_only = True
            i += 1
        elif a == "--no-rerank":
            no_rerank = True
            i += 1
        elif a in ("--interactive", "-i"):
            interactive = True
            i += 1
        else:
            positionals.append(a)
            i += 1

    if interactive or not positionals:
        # 交互模式（无位置参数默认交互）
        interactive_loop(mode, k, style=style, retrieval_check=retrieval_check, llm_only=llm_only)
        return

    question = " ".join(positionals)
    print(f"🔍 查询: {question} (mode={mode}, k={k}, style={style}, rc={retrieval_check}, lo={llm_only}, noRerank={no_rerank})")
    res = ask_once(question, mode, k, style=style,
                   retrieval_check=retrieval_check, llm_only=llm_only, no_rerank=no_rerank)
    print(format_answer(res))


if __name__ == "__main__":
    main()
