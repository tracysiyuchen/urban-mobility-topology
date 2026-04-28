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


class TemporalAggregator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=False  
        )

    def forward(self, z_seq):
        out, _ = self.gru(z_seq)   
        return out[-1]             


class ODDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        od_pred = torch.mm(z, z.t()) + self.bias
        return F.softplus(od_pred)


class GCNAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.encoder  = GCNEncoder(in_dim, hidden_dim, embed_dim)
        self.temporal = TemporalAggregator(embed_dim)
        self.decoder  = ODDecoder(embed_dim)

    def forward(self, x_seq, edge_index, edge_weight=None):
        z_seq = torch.stack(
            [self.encoder(x, edge_index, edge_weight) for x in x_seq],
            dim=0
        )  

        z = self.temporal(z_seq)  

        od_pred = self.decoder(z)
        return od_pred, z

    def encode(self, x_seq, edge_index, edge_weight=None):
        z_seq = torch.stack(
            [self.encoder(x, edge_index, edge_weight) for x in x_seq],
            dim=0
        )
        return self.temporal(z_seq)