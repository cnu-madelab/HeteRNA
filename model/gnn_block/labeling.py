
# model/gnn_blocks/labeling.py
import networkx as nx
import torch
import dgl


def apply_drnl_node_labeling(subgraph, src, dst):
    """
    DRNL (Double Radius Node Labeling) for subgraph nodes
    """
    G = dgl.to_networkx(subgraph.cpu()).to_undirected()
    src, dst = int(src), int(dst)

    d_src = nx.single_source_shortest_path_length(G, src, cutoff=3)
    d_dst = nx.single_source_shortest_path_length(G, dst, cutoff=3)
    labels = []

    for node in G.nodes():
        ds = d_src.get(node, 1e6)
        dd = d_dst.get(node, 1e6)
        if ds == 0 and dd == 0:
            labels.append(1)
        elif ds + dd == 0:
            labels.append(0)
        else:
            labels.append(1 + min(ds, dd) + (ds + dd) * (ds + dd + 1) // 2)

    labels = torch.tensor(labels, dtype=torch.long)
    return labels