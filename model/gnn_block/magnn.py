# model/gnn_blocks/magna.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAGNNLayer(nn.Module):
    def __init__(self, in_dim, heads=4):
        super(MAGNNLayer, self).__init__()
        self.heads = heads
        self.attn_layers = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(heads)
        ])
        self.out_proj = nn.Linear(in_dim * heads, in_dim)

    def forward(self, g, features, edge_metapath_indices_list):
        # 단순한 attention: 각 메타패스 별로 독립적 처리 후 concat
        outputs = []
        for i, edge_indices in enumerate(edge_metapath_indices_list):
            x = features
            for edge_idx in edge_indices:
                x = x.clone()  # dummy propagate (실제 구현은 attention diffusion 사용 가능)
            score = self.attn_layers[i](x)
            outputs.append(x * score)
        h_out = torch.cat(outputs, dim=1)
        return self.out_proj(h_out)