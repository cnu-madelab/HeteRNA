import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.gnn_block.graphsage import GraphSAGEBlock
from model.gnn_block.magnn import MAGNNLayer
from model.gnn_block.labeling import apply_drnl_node_labeling


class MAGNNLinkPrediction(nn.Module):
    def __init__(self, params, edge_index, edge_type, edge_metapath_indices_list):
        super(MAGNNLinkPrediction, self).__init__()

        self.p = params
        self.hidden_dim = params.gcn_dim
        self.input_dim = params.init_dim

        self.edge_metapath_indices_list = edge_metapath_indices_list
        self.label_embed = nn.Embedding(100, 32)
        self.graphsage = GraphSAGEBlock(self.input_dim + 32, self.hidden_dim)
        self.magna = MAGNNLayer(self.hidden_dim, heads=4)

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, full_graph, src_idx, dst_idx, node_features):

        """
        MAGNN forward 함수: 서브그래프 추출, 라벨링, 임베딩 학습, 링크 예측까지 포함

        Args:
            full_graph (dgl.DGLGraph): 전체 그래프
            src_idx (int): source 노드 인덱스
            dst_idx (int): target 노드 인덱스
            node_features (Tensor): 노드 임베딩 초기값

        Returns:
            Tensor: 링크 존재 확률
        """
        # (1) 3-hop 서브그래프 추출
        subgraph, sub_nodes = dgl.khop_in_subgraph(full_graph, [src_idx, dst_idx], k=3)

        # # (2) DRNL 라벨링 적용 (source, target 기준)
        # labels = apply_drnl_node_labeling(subgraph, src_idx, dst_idx)
        # label_embeds = self.label_embed(labels)
        #
        #
        #
        #
        # # (3) 노드 feature + 라벨 feature 결합
        # feats = node_features[sub_nodes]  # subgraph에 포함된 노드의 feature 추출
        # h = torch.cat([feats, label_embeds], dim=1)
        #
        # # (4) GraphSAGE -> MAGNA
        # h_sage = self.graphsage(subgraph, h)
        # h_magna = self.magna(subgraph, h_sage, self.edge_metapath_indices_list)
        #
        # # (5) src, dst 노드의 서브그래프 내 로컬 index 가져오기
        # local_src = (sub_nodes == src_idx).nonzero(as_tuple=False).item()
        # local_dst = (sub_nodes == dst_idx).nonzero(as_tuple=False).item()
        #
        # labels = apply_drnl_node_labeling(subgraph, local_src, local_dst)
        #
        # # (6) 링크 예측 (element-wise product -> FC layer)
        # src_h = h_magna[local_src]
        # dst_h = h_magna[local_dst]
        # link_rep = src_h * dst_h
        #
        # out = self.predictor(link_rep)
        # return out.squeeze()

        matches_src = (sub_nodes == src_idx).nonzero(as_tuple=False)
        matches_dst = (sub_nodes == dst_idx).nonzero(as_tuple=False)

        # 예외 처리: src 또는 dst가 subgraph에 없다면 0.5 반환 (중립 확률)
        if matches_src.numel() == 0 or matches_dst.numel() == 0:
            return torch.tensor(0.5, device=node_features.device)

        local_src = matches_src.item()
        local_dst = matches_dst.item()

        # (1) local index 계산 (subgraph 내에서 src/dst 위치 찾기)
        # local_src = (sub_nodes == src_idx).nonzero(as_tuple=False).item()
        # local_dst = (sub_nodes == dst_idx).nonzero(as_tuple=False).item()

        # (2) DRNL 라벨 생성 (local index 기준)
        labels = apply_drnl_node_labeling(subgraph, local_src, local_dst)
        label_embeds = self.label_embed(labels)

        # (3) 노드 feature 추출 (sub_nodes를 indexing)
        feats = node_features[sub_nodes]
        h = torch.cat([feats, label_embeds], dim=1)

        # (4) GraphSAGE + MAGNA
        h_sage = self.graphsage(subgraph, h)
        h_magna = self.magna(subgraph, h_sage, self.edge_metapath_indices_list)

        # (5) src, dst 노드의 임베딩 추출
        src_h = h_magna[local_src]
        dst_h = h_magna[local_dst]
        link_rep = src_h * dst_h

        out = self.predictor(link_rep)
        return out.squeeze()