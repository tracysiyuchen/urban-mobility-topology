"""
Training script for the Dual-Graph Adaptive Spatial Model.

Supports three temporal modes:
  --temporal none                (default) independent snapshots, original behavior
  --temporal lstm                Bi-LSTM over daily snapshot sequences
  --temporal temporal_attention  Multi-head self-attention over daily sequences

Usage:
    conda activate urban-mobility
    python src/train_dual_graph.py --config configs/config.yaml --temporal none
    python src/train_dual_graph.py --config configs/config.yaml --temporal lstm
    python src/train_dual_graph.py --config configs/config.yaml --temporal temporal_attention
"""

import argparse
import json
import os
import sys
from collections import defaultdict

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
    """Load static geographic adjacency -> PyG edge_index + edge_weight."""
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
        top_idx = np.argsort(row)[::-1][:top_k]
        for j in top_idx:
            if row[j] > 0:
                rows.append(i)
                cols.append(j)
                vals.append(row[j])

    if len(rows) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        return edge_index, edge_weight

    A_flow = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
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


# ── Time-bin ordering (consistent sequence within a day) ────────────────────

# Canonical order matching config.yaml time_bins
TIME_BIN_ORDER = ["Morning_Peak", "Midday", "Evening_Peak", "Night_early", "Night_late"]


def group_keys_by_day(keys: list) -> dict:
    """
    Group snapshot keys by date, sorted by time-bin order within each day.

    Keys have format: "2016-01-05_Morning_Peak"
    Returns: {date_str: [key_1, key_2, ...]} ordered by TIME_BIN_ORDER
    """
    day_map = defaultdict(list)
    for key in keys:
        # Key format: "2016-01-05_Morning_Peak"
        # Date uses dashes, so split on FIRST underscore only
        date_str, time_bin = key.split("_", 1)   # "2016-01-05", "Morning_Peak"
        day_map[date_str].append((time_bin, key))

    # Sort each day's snapshots by canonical time-bin order
    order_map = {tb: i for i, tb in enumerate(TIME_BIN_ORDER)}
    result = {}
    for date_str, items in day_map.items():
        items.sort(key=lambda x: order_map.get(x[0], 99))
        result[date_str] = [key for _, key in items]

    return result


# ── Training: temporal_mode = "none" ────────────────────────────────────────

def train_no_temporal(cfg, model, processed, device,
                      geo_edge_index, geo_edge_weight,
                      train_keys, val_keys):
    """Original per-snapshot training (no temporal modeling)."""
    dg_cfg = cfg["dual_graph"]
    optimizer = Adam(model.parameters(), lr=dg_cfg["lr"], weight_decay=dg_cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=dg_cfg["scheduler_step"], gamma=dg_cfg["scheduler_gamma"])

    best_val = float("inf")
    epochs = dg_cfg["epochs"]
    flow_top_k = dg_cfg["flow_adj_top_k"]
    model_dir = os.path.join(processed, "models", "dual_graph")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "dual_graph_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for key in tqdm(train_keys, desc=f"Epoch {epoch:02d} train", leave=False):
            feat, od_true, od_raw = load_snapshot(processed, key)
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

    # Extract embeddings: average across all training snapshots
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

    embeddings = np.mean(all_embeddings, axis=0)
    return embeddings, best_val, best_model_path


# ── Training: temporal_mode = "lstm" or "temporal_attention" ────────────────

def _load_day_data(processed, day_keys, flow_top_k, device):
    """Load and prepare all snapshots for one day, moved to device."""
    day_data = []
    for key in day_keys:
        feat, od_true, od_raw = load_snapshot(processed, key)
        flow_ei, flow_ew = build_flow_graph(od_raw, flow_top_k)
        day_data.append((
            feat.to(device),
            flow_ei.to(device),
            flow_ew.to(device),
            od_true.to(device),
        ))
    return day_data


def train_temporal(cfg, model, processed, device,
                   geo_edge_index, geo_edge_weight,
                   train_keys, val_keys):
    """Per-day sequence training (LSTM or temporal attention)."""
    dg_cfg = cfg["dual_graph"]
    optimizer = Adam(model.parameters(), lr=dg_cfg["lr"], weight_decay=dg_cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=dg_cfg["scheduler_step"], gamma=dg_cfg["scheduler_gamma"])

    best_val = float("inf")
    epochs = dg_cfg["epochs"]
    flow_top_k = dg_cfg["flow_adj_top_k"]
    model_dir = os.path.join(processed, "models", "dual_graph")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "dual_graph_best.pt")

    # Group snapshots into daily sequences
    train_days = group_keys_by_day(train_keys)
    val_days = group_keys_by_day(val_keys)
    print(f"Train days: {len(train_days)}  |  Val days: {len(val_days)}")

    train_day_list = list(train_days.items())
    val_day_list = list(val_days.items())

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_snapshots = 0

        for date_str, day_keys in tqdm(train_day_list, desc=f"Epoch {epoch:02d} train", leave=False):
            day_data = _load_day_data(processed, day_keys, flow_top_k, device)

            optimizer.zero_grad()
            od_preds, _ = model.forward_sequence(day_data, geo_edge_index, geo_edge_weight)

            # Average loss across all snapshots in the day
            loss = torch.tensor(0.0, device=device)
            for od_pred, (_, _, _, od_true) in zip(od_preds, day_data):
                loss = loss + od_loss(od_pred, od_true)
            loss = loss / len(day_data)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(day_data)
            n_snapshots += len(day_data)

        train_loss /= n_snapshots

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_snapshots = 0
        with torch.no_grad():
            for date_str, day_keys in val_day_list:
                day_data = _load_day_data(processed, day_keys, flow_top_k, device)
                od_preds, _ = model.forward_sequence(day_data, geo_edge_index, geo_edge_weight)

                for od_pred, (_, _, _, od_true) in zip(od_preds, day_data):
                    val_loss += od_loss(od_pred, od_true).item()
                    n_val_snapshots += 1

        val_loss /= n_val_snapshots
        scheduler.step()

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (val_loss={best_val:.4f})")

    # Extract embeddings: average z_temporal across all training days
    print("\nExtracting embeddings (averaging over training days) ...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for date_str, day_keys in tqdm(train_day_list, desc="Extracting", leave=False):
            # Prepare data without od_true for encode_sequence
            encode_data = []
            for key in day_keys:
                feat, _, od_raw = load_snapshot(processed, key)
                flow_ei, flow_ew = build_flow_graph(od_raw, flow_top_k)
                encode_data.append((
                    feat.to(device),
                    flow_ei.to(device),
                    flow_ew.to(device),
                ))

            z = model.encode_sequence(encode_data, geo_edge_index, geo_edge_weight)
            all_embeddings.append(z.cpu().numpy())

    embeddings = np.mean(all_embeddings, axis=0)  # [N, embed_dim]
    return embeddings, best_val, best_model_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--temporal", default="none",
                        choices=["none", "lstm", "temporal_attention"],
                        help="Temporal aggregation mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = cfg["data"]["processed_dir"]
    dg_cfg = cfg["dual_graph"]
    temporal_mode = args.temporal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Temporal mode: {temporal_mode}")

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
        temporal_mode=temporal_mode,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ── Train ──
    if temporal_mode == "none":
        embeddings, best_val, best_model_path = train_no_temporal(
            cfg, model, processed, device,
            geo_edge_index, geo_edge_weight,
            train_keys, val_keys,
        )
    else:
        embeddings, best_val, best_model_path = train_temporal(
            cfg, model, processed, device,
            geo_edge_index, geo_edge_weight,
            train_keys, val_keys,
        )

    # ── Save embeddings ──
    model_dir = os.path.join(processed, "models", "dual_graph")
    suffix = "" if temporal_mode == "none" else f"_{temporal_mode}"
    emb_path = os.path.join(model_dir, f"embeddings{suffix}.npy")
    np.save(emb_path, embeddings)
    print(f"Embeddings saved: {embeddings.shape} -> {emb_path}")

    # ── Summary ──
    summary = {
        "model": "dual_graph",
        "temporal_mode": temporal_mode,
        "best_val_loss": best_val,
        "epochs": dg_cfg["epochs"],
        "hidden_dim": dg_cfg["hidden_dim"],
        "embed_dim": dg_cfg["embedding_dim"],
        "n_heads": dg_cfg["n_heads"],
        "flow_adj_top_k": dg_cfg["flow_adj_top_k"],
        "param_count": param_count,
    }
    summary_path = os.path.join(model_dir, f"train_summary{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
