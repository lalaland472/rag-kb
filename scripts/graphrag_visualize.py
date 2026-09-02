#!/usr/bin/env python3
"""
GraphRAG 落地 —— 图谱可视化 + Leiden 社区检测
把 entities_normalized.json 的三元组建图，用 Leiden 算法检测社区，
按社区配色渲染成 PNG。

用法：python3 scripts/graphrag_visualize.py --out data/graph/graph.png
"""
import json, os, sys
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "data", "graph", "entities_normalized.json")

def main():
    d = json.load(open(ENT))
    docs = d["documents"]
    triples = [t for ts in docs.values() for t in ts]

    # 建节点集合 + 边（去重）
    nodes = set()
    edges = set()  # (a,b) 去重
    for a, r, b in triples:
        nodes.add(a); nodes.add(b)
        edges.add((a, b))

    nodes = sorted(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes
    g.add_edges([(idx[a], idx[b]) for a, b in edges])

    # 度（入+出）
    deg = [g.degree(i) for i in range(g.vcount())]

    # Leiden 社区检测
    partition = g.community_leiden(objective_function="modularity", weights=None)
    membership = partition.membership
    n_comm = max(membership) + 1
    print(f"节点: {len(nodes)} | 边: {len(edges)} | 社区数: {n_comm}")

    # 每个社区的成员和规模
    comm_members = defaultdict(list)
    for i, c in enumerate(membership):
        comm_members[c].append(nodes[i])
    comm_sizes = sorted([(c, len(m)) for c, m in comm_members.items()], key=lambda x: -x[1])
    print("\n=== 社区规模 top 10 ===")
    for c, sz in comm_sizes[:10]:
        print(f"  社区{c}: {sz} 节点")

    # 力导向布局：改用 networkx spring_layout，拉开间距减少重叠
    import networkx as _nx
    Gnx = _nx.DiGraph()
    Gnx.add_nodes_from(nodes)
    Gnx.add_edges_from(edges)
    pos = _nx.spring_layout(Gnx, seed=42, k=1.2, iterations=400)
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]

    # 颜色（按社区，HSL 循环）
    cmap = plt.get_cmap("tab20", max(n_comm, 20))
    colors = [cmap(membership[i] % 20) for i in range(g.vcount())]

    fig, ax = plt.subplots(figsize=(22, 22))

    # 画边（浅色）
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#dddddd", alpha=0.35, linewidth=0.6, zorder=1)

    # 节点（大小按度）
    sizes = [6 + 45 * (deg[i] / max(deg)) ** 1.6 for i in range(g.vcount())]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.9, edgecolors="white",
               linewidths=1.0, zorder=2)

    # 标注：只标度 >= 2 的节点；度越高字越大越清晰
    for i in range(g.vcount()):
        if deg[i] >= 6:
            # 枢纽节点：大标题 + 白底框
            ax.annotate(nodes[i], (xs[i], ys[i]), fontsize=15, fontweight="bold",
                        ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9, ec="#333333", lw=1.0))
        elif deg[i] >= 3:
            # 次要节点：中等字号，浅框
            ax.annotate(nodes[i], (xs[i], ys[i]), fontsize=10, fontweight="normal",
                        ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, ec="none"))
        # deg < 3 的叶子节点不标标签，只画点（减少视觉噪声）

    # 社区图例
    legend_handles = []
    for c, sz in comm_sizes[:10]:
        legend_handles.append(mpatches.Patch(color=cmap(c % 20), label=f"社区{c} ({sz})"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, frameon=True)

    ax.set_title(f"RAG-KB 知识图谱 ({len(nodes)} 实体, {len(edges)} 关系, {n_comm} 社区)",
                 fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    out = os.path.join(BASE, "data", "graph", "graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✅ 已保存 → {out}")

if __name__ == "__main__":
    main()
