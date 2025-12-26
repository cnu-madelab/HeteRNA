import torch

def extract_metapath_indices(g, metapath):
    """
    edge_type 정보를 기반으로 주어진 메타패스 순서에 해당하는 edge 인덱스를 리스트로 반환

    Args:
        g (dgl.DGLGraph): DGL 그래프 (g.edata['type']가 있어야 함)
        metapath (list[int]): edge_type 순서 예: [0, 1, 2]

    Returns:
        list[Tensor]: 각 단계에 해당하는 edge index tensor 리스트
    """
    edge_types = g.edata['type']
    edge_indices_per_hop = []

    for edge_type in metapath:
        indices = torch.nonzero(edge_types == edge_type, as_tuple=False).squeeze()
        edge_indices_per_hop.append(indices)

    return edge_indices_per_hop
