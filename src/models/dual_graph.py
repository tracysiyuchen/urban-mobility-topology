"""
Dual-Graph Adaptive Spatial Model (Proposed Method).

Architecture:
  1. Two parallel GCN branches:
     - Geographic branch: operates on static A_geo (inverse-distance weighted)
     - Flow branch: operates on dynamic A_flow (OD flow weighted, per snapshot)
  2. Multi-head cross-attention fusion:
     - Learns per-node weighting between geographic and flow representations
  3. Temporal aggregation (optional, per-day sequence of snapshots):
     - "lstm":              Bi-LSTM over the daily snapshot sequence
     - "temporal_attention": Multi-head self-attention over the daily sequence
     - "none":              No temporal modeling (independent snapshots, mean pooling)
  4. OD decoder:
     - Bilinear inner product + bias -> softplus -> predicted OD matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ── GCN Branch ──────────────────────────────────────────────────────────────

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


# ── Cross-Attention Fusion ──────────────────────────────────────────────────

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
        Q, K, V: [N, embed_dim] -> returns [N, embed_dim]
        """
        N = Q.size(0)
        Q = Q.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
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


# ── Temporal Modules ────────────────────────────────────────────────────────

class TemporalLSTM(nn.Module):
    """
    Bi-LSTM that aggregates a daily sequence of per-snapshot fused embeddings.

    Input:  [T, N, embed_dim]  — T snapshots in one day, N nodes
    Output: [N, embed_dim]     — temporally-informed node embeddings
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # Bi-LSTM: each direction outputs embed_dim // 2
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=1,
            batch_first=False,       # input: [T, batch, embed_dim]
            bidirectional=True,
            dropout=0.0,
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_seq):
        """
        Args:
            z_seq: [T, N, embed_dim] — sequence of fused embeddings for one day
        Returns:
            z_temporal: [N, embed_dim]
        """
        T, N, D = z_seq.shape
        # Reshape to [T, N, D] — treat each node as a batch element
        # LSTM expects [seq_len, batch, input_size]
        out, _ = self.lstm(z_seq)           # [T, N, embed_dim]
        # Take the mean across time steps as the aggregated representation
        z_agg = out.mean(dim=0)             # [N, embed_dim]
        z_temporal = self.layer_norm(self.proj(self.dropout(z_agg)))
        return z_temporal


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the daily snapshot sequence.

    Input:  [T, N, embed_dim]  — T snapshots in one day, N nodes
    Output: [N, embed_dim]     — temporally-informed node embeddings

    For each node independently, attends across T time steps to learn
    which time periods are most informative for its representation.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Learnable positional encoding for time bin positions
        # Max 6 time bins per day (5 configured + 1 buffer)
        self.pos_embedding = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_seq):
        """
        Args:
            z_seq: [T, N, embed_dim] — sequence of fused embeddings for one day
        Returns:
            z_temporal: [N, embed_dim]
        """
        T, N, D = z_seq.shape

        # Add positional encoding for time steps
        z_seq = z_seq + self.pos_embedding[:T].unsqueeze(1)  # [T, N, D]

        # Reshape: treat each node independently → [N, T, D]
        z = z_seq.permute(1, 0, 2)  # [N, T, D]

        Q = self.W_q(z)  # [N, T, D]
        K = self.W_k(z)
        V = self.W_v(z)

        # Multi-head: [N, T, D] → [N, n_heads, T, head_dim]
        Q = Q.view(N, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: [N, n_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum: [N, n_heads, T, head_dim]
        out = torch.matmul(attn, V)
        # Concatenate heads: [N, T, embed_dim]
        out = out.transpose(1, 2).contiguous().view(N, T, self.embed_dim)

        # Pool across time: weighted mean via learned attention, or simple mean
        z_agg = out.mean(dim=1)  # [N, D]

        z_temporal = self.layer_norm(self.out_proj(self.dropout(z_agg)))
        return z_temporal


# ── OD Decoder ──────────────────────────────────────────────────────────────

class ODDecoder(nn.Module):
    """Reconstruct OD matrix from node embeddings via bilinear inner product."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        od_pred = torch.mm(z, z.t()) + self.bias
        return F.softplus(od_pred)


# ── Full Model ──────────────────────────────────────────────────────────────

class DualGraphModel(nn.Module):
    """
    Full Dual-Graph Adaptive Spatial Model.

    Supports three temporal modes (set via `temporal_mode`):
      - "none":               process each snapshot independently (original behavior)
      - "lstm":               Bi-LSTM aggregation over daily snapshot sequence
      - "temporal_attention":  Multi-head self-attention over daily snapshot sequence

    When temporal_mode is "none":
      forward(x, geo_ei, geo_ew, flow_ei, flow_ew) → (od_pred, z_fused)

    When temporal_mode is "lstm" or "temporal_attention":
      forward_sequence(day_data, geo_ei, geo_ew) → (od_preds, z_temporal)
        where day_data = [(feat_t, flow_ei_t, flow_ew_t, od_true_t), ...]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, embed_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.3,
                 temporal_mode: str = "none"):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.embed_dim = embed_dim

        self.geo_branch = GCNBranch(in_dim, hidden_dim, embed_dim, dropout)
        self.flow_branch = GCNBranch(in_dim, hidden_dim, embed_dim, dropout)
        self.fusion = CrossAttentionFusion(embed_dim, n_heads)
        self.decoder = ODDecoder(embed_dim)

        # Temporal module (only created when needed)
        if temporal_mode == "lstm":
            self.temporal = TemporalLSTM(embed_dim, dropout=0.1)
        elif temporal_mode == "temporal_attention":
            self.temporal = TemporalSelfAttention(embed_dim, n_heads=n_heads, dropout=0.1)
        elif temporal_mode == "none":
            self.temporal = None
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode!r}. "
                             f"Choose from: 'none', 'lstm', 'temporal_attention'")

    def _encode_single(self, x, geo_edge_index, geo_edge_weight,
                        flow_edge_index, flow_edge_weight):
        """Encode a single snapshot → z_fused [N, embed_dim]."""
        z_geo = self.geo_branch(x, geo_edge_index, geo_edge_weight)
        z_flow = self.flow_branch(x, flow_edge_index, flow_edge_weight)
        z_fused = self.fusion(z_geo, z_flow)
        return z_fused

    # ── Mode: none (original, per-snapshot) ─────────────────────────────────

    def forward(self, x, geo_edge_index, geo_edge_weight,
                flow_edge_index, flow_edge_weight):
        """Per-snapshot forward (temporal_mode='none')."""
        z_fused = self._encode_single(x, geo_edge_index, geo_edge_weight,
                                       flow_edge_index, flow_edge_weight)
        od_pred = self.decoder(z_fused)
        return od_pred, z_fused

    def encode(self, x, geo_edge_index, geo_edge_weight,
               flow_edge_index, flow_edge_weight):
        """Per-snapshot encode (temporal_mode='none')."""
        return self._encode_single(x, geo_edge_index, geo_edge_weight,
                                    flow_edge_index, flow_edge_weight)

    # ── Mode: lstm / temporal_attention (per-day sequence) ──────────────────

    def forward_sequence(self, day_data, geo_edge_index, geo_edge_weight):
        """
        Process a full day's snapshot sequence through dual-graph + temporal module.

        Args:
            day_data: list of (feat, flow_edge_index, flow_edge_weight, od_true)
                      ordered by time bin within one day. Length T (up to 5).
            geo_edge_index:  [2, E_geo]  static
            geo_edge_weight: [E_geo]     static

        Returns:
            od_preds: list of T predicted OD matrices [N, N] (one per snapshot,
                      all decoded from the same z_temporal for consistency)
            z_temporal: [N, embed_dim] temporally-informed node embedding
        """
        assert self.temporal is not None, \
            "forward_sequence requires temporal_mode='lstm' or 'temporal_attention'"

        # 1. Encode each snapshot independently via dual-graph + fusion
        z_list = []
        for feat, flow_ei, flow_ew, _ in day_data:
            z_fused = self._encode_single(feat, geo_edge_index, geo_edge_weight,
                                           flow_ei, flow_ew)
            z_list.append(z_fused)

        # 2. Stack into sequence [T, N, embed_dim]
        z_seq = torch.stack(z_list, dim=0)

        # 3. Temporal aggregation → z_temporal [N, embed_dim]
        z_temporal = self.temporal(z_seq)

        # 4. Decode OD from temporal embedding (same for all time steps)
        od_pred = self.decoder(z_temporal)
        od_preds = [od_pred] * len(day_data)

        return od_preds, z_temporal

    def encode_sequence(self, day_data, geo_edge_index, geo_edge_weight):
        """
        Encode a day's sequence → z_temporal [N, embed_dim].
        Used for embedding extraction.
        """
        assert self.temporal is not None

        z_list = []
        for feat, flow_ei, flow_ew in day_data:
            z_fused = self._encode_single(feat, geo_edge_index, geo_edge_weight,
                                           flow_ei, flow_ew)
            z_list.append(z_fused)

        z_seq = torch.stack(z_list, dim=0)
        z_temporal = self.temporal(z_seq)
        return z_temporal
