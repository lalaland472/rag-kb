#!/usr/bin/env python3
"""Week 23 Day 3 — query.py：rag-kb 问答 CLI（交互式）

把 generate_answer 封装成正式问答界面：
  - 单次查询：python3 scripts/query.py "问题" [--mode flat|hybrid] [--k 5]
  - 交互会话：python3 scripts/query.py [--interactive]
  - 输出：回答 + 引用来源（论文名 · 第N页）

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
    lines.append("")
    lines.append("📚 引用来源")
    for c in cites:
        if c["chunk_id"] is not None:
            loc = f"第{c['page'] + 1}页" if c.get("page") is not None else f"块{c['chunk_id']}"
        else:
            loc = "(摘要)"
        lines.append(f"  [{c['index']}] {c['display']}  [{loc}]")
    lines.append("")
    lines.append(f"  ⏱ {meta.get('latency', 0):.1f}s · mode={meta['mode']} · "
                 f"chunk={meta['chunk_count']}")
    return "\n".join(lines)


def ask_once(question, mode, k, interactive=False):
    """单次问答，带耗时统计。"""
    t0 = time.time()
    res = generate_answer(question, mode=mode, k=k)
    res["meta"]["latency"] = time.time() - t0
    if interactive:
        print(format_answer(res))
    return res


def interactive_loop(default_mode, default_k):
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
        try:
            ask_once(q, mode, k, interactive=True)
        except Exception as e:
            print(f"⚠️ 出错: {e}")


def main():
    args = sys.argv[1:]
    # 支持 --mode / --k / --interactive
    mode, k = "flat", 5
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
        elif a in ("--interactive", "-i"):
            interactive = True
            i += 1
        else:
            positionals.append(a)
            i += 1

    if interactive or not positionals:
        # 交互模式（无位置参数默认交互）
        interactive_loop(mode, k)
        return

    question = " ".join(positionals)
    print(f"🔍 查询: {question} (mode={mode}, k={k})")
    res = ask_once(question, mode, k)
    print(format_answer(res))


if __name__ == "__main__":
    main()
