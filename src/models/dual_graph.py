"""
Dual-Graph Adaptive Spatial Model (Proposed Method).

Architecture:
  1. Two parallel GCN branches:
     - Geographic branch: operates on static A_geo (inverse-distance weighted)
     - Flow branch: operates on dynamic A_flow (OD flow weighted, per snapshot)
  2. Multi-head cross-attention fusion:
     - Learns per-node weighting between geographic and flow representations
  3. OD decoder:
     - Bilinear inner product + bias → softplus → predicted OD matrix
  4. Loss: MSE with emphasis on non-zero flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBranch(nn.Module):
    """Two-layer GCN encoder used for each graph branch."""

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x  # [N, embed_dim]


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention that fuses geographic and flow embeddings.

    For each node, learns how much weight to assign to geographic proximity
    versus mobility-driven connectivity. The two branch outputs serve as
    queries/keys/values for each other via cross-attention, then combined.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        # Projections for geo branch
        self.W_q_geo = nn.Linear(embed_dim, embed_dim)
        self.W_k_geo = nn.Linear(embed_dim, embed_dim)
        self.W_v_geo = nn.Linear(embed_dim, embed_dim)

        # Projections for flow branch
        self.W_q_flow = nn.Linear(embed_dim, embed_dim)
        self.W_k_flow = nn.Linear(embed_dim, embed_dim)
        self.W_v_flow = nn.Linear(embed_dim, embed_dim)

        # Output projection after concatenating both attended outputs
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def _multi_head_attention(self, Q, K, V):
        """Standard scaled dot-product multi-head attention.
        Q, K, V: [N, embed_dim] → returns [N, embed_dim]
        """
        N = Q.size(0)
        # Reshape to [N, n_heads, head_dim] then transpose to [n_heads, N, head_dim]
        Q = Q.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.n_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention: [n_heads, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum: [n_heads, N, head_dim]
        out = torch.matmul(attn, V)
        # Concatenate heads: [N, embed_dim]
        out = out.transpose(0, 1).contiguous().view(N, self.embed_dim)
        return out

    def forward(self, z_geo, z_flow):
        """
        Args:
            z_geo:  [N, embed_dim] from geographic GCN branch
            z_flow: [N, embed_dim] from flow GCN branch
        Returns:
            z_fused: [N, embed_dim] fused node embeddings
        """
        # Cross-attention: geo queries flow context
        Q_geo = self.W_q_geo(z_geo)
        K_flow = self.W_k_flow(z_flow)
        V_flow = self.W_v_flow(z_flow)
        attn_geo = self._multi_head_attention(Q_geo, K_flow, V_flow)

        # Cross-attention: flow queries geo context
        Q_flow = self.W_q_flow(z_flow)
        K_geo = self.W_k_geo(z_geo)
        V_geo = self.W_v_geo(z_geo)
        attn_flow = self._multi_head_attention(Q_flow, K_geo, V_geo)

        # Concatenate and project
        z_cat = torch.cat([attn_geo, attn_flow], dim=-1)  # [N, 2*embed_dim]
        z_fused = self.out_proj(z_cat)                      # [N, embed_dim]
        z_fused = self.layer_norm(z_fused)

        return z_fused


class ODDecoder(nn.Module):
    """Reconstruct OD matrix from node embeddings via bilinear inner product."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        od_pred = torch.mm(z, z.t()) + self.bias
        return F.softplus(od_pred)


class DualGraphModel(nn.Module):
    """
    Full Dual-Graph Adaptive Spatial Model.

    Forward pass:
      1. GCN on geographic graph  →  z_geo   [N, embed_dim]
      2. GCN on flow graph        →  z_flow  [N, embed_dim]
      3. Cross-attention fusion   →  z_fused [N, embed_dim]
      4. OD reconstruction        →  od_pred [N, N]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, embed_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.geo_branch = GCNBranch(in_dim, hidden_dim, embed_dim, dropout)
        self.flow_branch = GCNBranch(in_dim, hidden_dim, embed_dim, dropout)
        self.fusion = CrossAttentionFusion(embed_dim, n_heads)
        self.decoder = ODDecoder(embed_dim)

    def forward(self, x, geo_edge_index, geo_edge_weight,
                flow_edge_index, flow_edge_weight):
        z_geo = self.geo_branch(x, geo_edge_index, geo_edge_weight)
        z_flow = self.flow_branch(x, flow_edge_index, flow_edge_weight)
        z_fused = self.fusion(z_geo, z_flow)
        od_pred = self.decoder(z_fused)
        return od_pred, z_fused

    def encode(self, x, geo_edge_index, geo_edge_weight,
               flow_edge_index, flow_edge_weight):
        z_geo = self.geo_branch(x, geo_edge_index, geo_edge_weight)
        z_flow = self.flow_branch(x, flow_edge_index, flow_edge_weight)
        z_fused = self.fusion(z_geo, z_flow)
        return z_fused
