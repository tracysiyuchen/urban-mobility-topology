"""
Training script for the Dual-Graph Adaptive Spatial Model.

Usage:
    conda activate urban-mobility
    python src/train_dual_graph.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm

from src.models.dual_graph import DualGraphModel


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_geo_graph(processed_dir: str):
    """Load static geographic adjacency → PyG edge_index + edge_weight."""
    A = sp.load_npz(os.path.join(processed_dir, "A_geo.npz"))
    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    return edge_index, edge_weight.float()


def build_flow_graph(od_matrix: sp.spmatrix, top_k: int):
    """
    Build a dynamic flow adjacency graph from an OD snapshot.

    Symmetrize OD (undirected flow), keep top-k neighbors per node,
    row-normalize weights, then convert to PyG edge_index + edge_weight.
    """
    N = od_matrix.shape[0]

    # Symmetrize: treat flow as undirected
    od_sym = od_matrix + od_matrix.T
    od_dense = od_sym.toarray().astype(np.float32) if sp.issparse(od_sym) else od_sym.astype(np.float32)
    np.fill_diagonal(od_dense, 0.0)

    rows, cols, vals = [], [], []
    for i in range(N):
        row = od_dense[i]
        if row.sum() == 0:
            continue
        # Top-k neighbors by flow volume
        top_idx = np.argsort(row)[::-1][:top_k]
        for j in top_idx:
            if row[j] > 0:
                rows.append(i)
                cols.append(j)
                vals.append(row[j])

    if len(rows) == 0:
        # Fallback: empty graph (self-loops only to avoid errors)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        return edge_index, edge_weight

    A_flow = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    # Row-normalize
    row_sums = np.array(A_flow.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_flow = sp.diags(1.0 / row_sums) @ A_flow

    edge_index, edge_weight = from_scipy_sparse_matrix(A_flow)
    return edge_index, edge_weight.float()


def load_snapshot(processed_dir: str, key: str):
    """Load node features and OD target for one snapshot."""
    feat = np.load(os.path.join(processed_dir, "node_features", f"{key}.npz"))["feat"]
    od = sp.load_npz(os.path.join(processed_dir, "od_matrices", f"{key}_log.npz"))
    od_raw = sp.load_npz(os.path.join(processed_dir, "od_matrices", f"{key}_raw.npz"))
    feat_t = torch.tensor(feat, dtype=torch.float32)
    od_dense = torch.tensor(od.toarray(), dtype=torch.float32)
    return feat_t, od_dense, od_raw


def od_loss(od_pred: torch.Tensor, od_true: torch.Tensor) -> torch.Tensor:
    """MSE loss, weighted to emphasise non-zero flows."""
    mask_nonzero = (od_true > 0).float()
    mask_zero = 1.0 - mask_nonzero
    loss = (mask_nonzero * (od_pred - od_true) ** 2).mean() \
         + 0.1 * (mask_zero * od_pred ** 2).mean()
    return loss


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = cfg["data"]["processed_dir"]
    dg_cfg = cfg["dual_graph"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load static geographic graph ──
    geo_edge_index, geo_edge_weight = load_geo_graph(processed)
    geo_edge_index = geo_edge_index.to(device)
    geo_edge_weight = geo_edge_weight.to(device)

    # ── Load snapshot manifest & split ──
    manifest = pd.read_csv(os.path.join(processed, "snapshot_manifest.csv"))
    train_keys = manifest[manifest["date"].apply(
        lambda d: int(d.split("-")[1])).isin(cfg["data"]["train_months"])]["key"].tolist()
    val_keys = manifest[manifest["date"].apply(
        lambda d: int(d.split("-")[1])).isin(cfg["data"]["val_months"])]["key"].tolist()
    print(f"Train snapshots: {len(train_keys)}  |  Val snapshots: {len(val_keys)}")

    # ── Model ──
    model = DualGraphModel(
        in_dim=2,
        hidden_dim=dg_cfg["hidden_dim"],
        embed_dim=dg_cfg["embedding_dim"],
        n_heads=dg_cfg["n_heads"],
        dropout=dg_cfg["dropout"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=dg_cfg["lr"], weight_decay=dg_cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=dg_cfg["scheduler_step"], gamma=dg_cfg["scheduler_gamma"])

    best_val = float("inf")
    epochs = dg_cfg["epochs"]
    flow_top_k = dg_cfg["flow_adj_top_k"]
    model_dir = os.path.join(processed, "models", "dual_graph")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "dual_graph_best.pt")

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for key in tqdm(train_keys, desc=f"Epoch {epoch:02d} train", leave=False):
            feat, od_true, od_raw = load_snapshot(processed, key)

            # Build dynamic flow adjacency from raw OD counts
            flow_edge_index, flow_edge_weight = build_flow_graph(od_raw, flow_top_k)

            feat = feat.to(device)
            od_true = od_true.to(device)
            flow_edge_index = flow_edge_index.to(device)
            flow_edge_weight = flow_edge_weight.to(device)

            optimizer.zero_grad()
            od_pred, _ = model(feat, geo_edge_index, geo_edge_weight,
                               flow_edge_index, flow_edge_weight)
            loss = od_loss(od_pred, od_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_keys)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for key in val_keys:
                feat, od_true, od_raw = load_snapshot(processed, key)
                flow_edge_index, flow_edge_weight = build_flow_graph(od_raw, flow_top_k)

                feat = feat.to(device)
                od_true = od_true.to(device)
                flow_edge_index = flow_edge_index.to(device)
                flow_edge_weight = flow_edge_weight.to(device)

                od_pred, _ = model(feat, geo_edge_index, geo_edge_weight,
                                   flow_edge_index, flow_edge_weight)
                val_loss += od_loss(od_pred, od_true).item()

        val_loss /= len(val_keys)
        scheduler.step()

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (val_loss={best_val:.4f})")

    # ── Extract embeddings ──
    print("\nExtracting embeddings (averaging over training snapshots) ...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for key in tqdm(train_keys, desc="Extracting", leave=False):
            feat, _, od_raw = load_snapshot(processed, key)
            flow_edge_index, flow_edge_weight = build_flow_graph(od_raw, flow_top_k)

            feat = feat.to(device)
            flow_edge_index = flow_edge_index.to(device)
            flow_edge_weight = flow_edge_weight.to(device)

            z = model.encode(feat, geo_edge_index, geo_edge_weight,
                             flow_edge_index, flow_edge_weight)
            all_embeddings.append(z.cpu().numpy())

    embeddings = np.mean(all_embeddings, axis=0)  # [N, embed_dim]
    emb_path = os.path.join(model_dir, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Embeddings saved: {embeddings.shape} -> {emb_path}")

    # ── Summary ──
    summary = {
        "model": "dual_graph",
        "best_val_loss": best_val,
        "epochs": epochs,
        "hidden_dim": dg_cfg["hidden_dim"],
        "embed_dim": dg_cfg["embedding_dim"],
        "n_heads": dg_cfg["n_heads"],
        "flow_adj_top_k": flow_top_k,
    }
    summary_path = os.path.join(model_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
