import numpy as np
import networkx as nx
from numpy import random
import os

np.random.seed(0)
random.seed(0)
"""
In this file we create a few sample network topologies for testing

"""
os.makedirs("topology", exist_ok=True)

topology = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
np.savetxt("topology/topo_1.txt", topology, fmt="%d")

topology = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
np.savetxt("topology/topo_2.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ]
)
np.savetxt("topology/topo_3.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ]
)
np.savetxt("topology/topo_4.txt", topology, fmt="%d")


G = nx.powerlaw_cluster_graph(15, 2, p=0.01, seed=0)
# randomly assign edge weights (network connection probabilities)
for u, v, w in G.edges(data=True):
    # weight with chance of sustaining network connection
    w["weight"] = random.choice([0.7, 0.8, 0.9, 1], p=[0.03, 0.07, 0.3, 0.6])


topology = nx.to_numpy_array(G)
np.savetxt("topology/topo_5.txt", topology, fmt="%d")
print(topology)

G = nx.complete_graph(3)
topology = nx.to_numpy_array(G)
np.savetxt("topology/topo_6.txt", topology, fmt="%d")

G = nx.complete_graph(10)
topology = nx.to_numpy_array(G)
np.savetxt("topology/topo_7.txt", topology, fmt="%d")

G = nx.barabasi_albert_graph(n=16, m=2, seed=0)
topology = nx.to_numpy_array(G)
np.savetxt("topology/topo_8.txt", topology, fmt="%d")

seed = 0

sizes = [11, 11, 11]
self_conect = 0.5
community_connect = 0.009
probs = [
        [self_conect, community_connect, community_connect],
        [community_connect, self_conect, community_connect],
        [
            community_connect,
            community_connect,
            self_conect,
        ],
]
g = nx.stochastic_block_model(sizes, probs, seed=seed)
topology = nx.to_numpy_array(g)
np.savetxt("topology/topo_sb_low_connect.txt", topology, fmt="%d")


g = nx.barabasi_albert_graph(n=33, m=1, seed=seed)
topology = nx.to_numpy_array(g)
np.savetxt("topology/topo_ba33.txt", topology, fmt="%d")
