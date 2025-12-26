import torch
import torch.nn as nn
import dgl.function as fn

class GraphSAGEBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphSAGEBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_total = torch.cat([h, h_neigh], dim=1)
            return self.linear(h_total)