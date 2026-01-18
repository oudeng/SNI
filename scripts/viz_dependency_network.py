#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dependency-matrix", type=str, required=True, help="dependency_matrix.csv from a run.")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    D = pd.read_csv(args.dependency_matrix, index_col=0)

    # node sizes by column sum (attention received)
    sigma = D.sum(axis=0)

    G = nx.DiGraph()
    for node in D.columns:
        G.add_node(node, sigma=float(sigma[node]))

    for target in D.index:
        for src in D.columns:
            if target == src:
                continue
            w = float(D.loc[target, src])
            if w > args.tau:
                G.add_edge(target, src, weight=w)

    # layout
    pos = nx.spring_layout(G, seed=1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    node_sizes = [300 + 1200 * G.nodes[n]["sigma"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    weights = [G[u][v]["weight"] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=[2.0 * w for w in weights], alpha=0.7)

    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outdir / "dependency_network.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(outdir / "dependency_network.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] saved to {outdir}")


if __name__ == "__main__":
    main()
