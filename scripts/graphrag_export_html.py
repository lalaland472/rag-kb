#!/usr/bin/env python3
"""
GraphRAG 落地 —— 交互式图谱 HTML 导出
用 pyvis 生成可交互的图谱：缩放、拖动、点节点看关系、社区配色。

用法：python3 scripts/graphrag_export_html.py --out data/graph/graph.html
"""
import json, os
import igraph as ig
from pyvis.network import Network

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "data", "graph", "entities_normalized.json")

def main():
    out = os.path.join(BASE, "data", "graph", "graph.html")
    d = json.load(open(ENT))
    triples = [t for ts in d["documents"].values() for t in ts]

    nodes = sorted({x for t in triples for x in (t[0], t[2])})
    idx = {n: i for i, n in enumerate(nodes)}

    # 建 igraph 跑 Leiden 社区
    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    edges = list({(a, b) for a, r, b in triples})
    g.add_edges([(idx[a], idx[b]) for a, b in edges])
    part = g.community_leiden(objective_function="modularity")
    membership = part.membership
    n_comm = max(membership) + 1
    print(f"节点 {len(nodes)} | 边 {len(edges)} | 社区 {n_comm}")

    # 度
    deg = [g.degree(i) for i in range(g.vcount())]
    deg_by_name = {nodes[i]: deg[i] for i in range(g.vcount())}

    # pyvis 网络（物理引擎 + 交互）
    net = Network(
        height="900px", width="100%",
        directed=True, notebook=False,
        bgcolor="#ffffff", font_color="#222222",
    )
    net.set_options("""
    var options = {
      "physics": {"barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3,
                  "springLength": 95, "springConstant": 0.04, "damping": 0.09},
        "stabilization": {"iterations": 200}},
      "interaction": {"hover": true, "tooltipDelay": 50},
      "nodes": {"font": {"size": 14, "face": "Tahoma"}},
      "edges": {"color": {"color": "#b0b0b0"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}}
    }
    """)

    # 配色
    palette = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
               "#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#800000",
               "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#e6beff"]

    # 关系合并展示（用于 tooltip）
    rels_by_edge = {}
    for a, r, b in triples:
        rels_by_edge.setdefault((a, b), []).append(r)

    for i, n in enumerate(nodes):
        cid = membership[i]
        size = 8 + 18 * (deg[i] / max(deg)) ** 1.5
        color = palette[cid % len(palette)]
        net.add_node(i, label=n, color=color, size=size,
                     title=f"{n}<br>度:{deg[i]} | 社区:{cid}")

    for a, b in edges:
        rels = ", ".join(rels_by_edge[(a, b)])
        net.add_edge(idx[a], idx[b], title=rels, arrows="to")

    net.save_graph(out)
    print(f"✅ 已导出 → {out} ({os.path.getsize(out)//1024} KB)")

if __name__ == "__main__":
    main()
