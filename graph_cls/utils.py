import torch
from torch import nn

class GINConv(nn.Module):
    def __init__(self, nn, eps=0):
        super(GINConv, self).__init__()
        self.eps = eps
        self.nn = nn
        
    def forward(self, x, adj):
        size = adj.size(0)
        out = adj@x
        out += (self.eps+1)*x
        return self.nn(out)
    
def edge2adj(x, edge_index, mode="sp"):
    """Convert edge index to adjacency matrix"""
    assert mode in ["sp", "dense"]
    num_nodes = x.shape[0]
    edge_weight = torch.ones(edge_index.size(1), dtype=None,
                                     device=edge_index.device)
    sp_tensor =  torch.sparse.FloatTensor(edge_index, edge_weight,torch.Size((num_nodes, num_nodes)))
    if mode == "sp":
        return sp_tensor
    else:
        return sp_tensor.to_dense()