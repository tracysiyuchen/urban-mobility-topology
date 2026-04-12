import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x  


class ODDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        od_pred = torch.mm(z, z.t()) + self.bias
        return F.softplus(od_pred)  


class GCNAutoencoder(nn.Module):
    """
    - encoder: 2-layer GCN on geographic adjacency graph
    - decoder: inner product to reconstruct OD matrix
    """
    def __init__(self, in_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = ODDecoder(embed_dim)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        od_pred = self.decoder(z)
        return od_pred, z

    def encode(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)