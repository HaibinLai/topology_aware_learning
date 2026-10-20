#!/usr/bin/env python3
"""Draw the BA-33 communication graph and highlight backdoor clients."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "src/create_topo/topology/topo_ba33.txt"
OUTPUT = ROOT / "artifacts/ba33_backdoor_nodes.png"
BACKDOOR_NODES = {14, 15, 24, 31}


def main() -> None:
    adjacency = np.loadtxt(TOPOLOGY)
    graph = nx.from_numpy_array(adjacency)
    position = nx.spring_layout(graph, seed=7, iterations=200)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 9), dpi=180)
    normal_nodes = sorted(set(graph) - BACKDOOR_NODES)
    nx.draw_networkx_edges(graph, position, ax=axis, edge_color="#9aa4b2", width=1.2, alpha=0.75)
    nx.draw_networkx_nodes(
        graph, position, nodelist=normal_nodes, ax=axis,
        node_color="#78a6d8", node_size=520, edgecolors="white", linewidths=0.8,
    )
    nx.draw_networkx_nodes(
        graph, position, nodelist=sorted(BACKDOOR_NODES), ax=axis,
        node_color="#d62728", node_size=700, edgecolors="#7f0000", linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, position, ax=axis, font_size=8, font_color="black")
    axis.set_title("BA-33 topology with backdoor clients highlighted", fontsize=15, pad=14)
    axis.text(
        0.01, 0.01,
        "Red nodes: backdoor clients 14, 15, 24, 31 (degree 1)",
        transform=axis.transAxes, fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
