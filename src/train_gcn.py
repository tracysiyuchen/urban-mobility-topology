import argparse
import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gcn_autoencoder import GCNAutoencoder


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_graph(processed_dir):
    A = sp.load_npz(os.path.join(processed_dir, "A_geo.npz"))
    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    return edge_index, edge_weight.float()


def load_snapshot(processed_dir, key):
    feat = np.load(os.path.join(processed_dir, "node_features", f"{key}.npz"))["feat"]
    od   = sp.load_npz(os.path.join(processed_dir, "od_matrices", f"{key}_log.npz"))
    return (
        torch.tensor(feat, dtype=torch.float32),
        torch.tensor(od.toarray(), dtype=torch.float32)
    )


def group_by_day(manifest, months):
    """Group snapshot keys by date, only keeping dates in given months."""
    subset = manifest[manifest["date"].apply(
        lambda d: int(d.split("-")[1])).isin(months)]
    groups = defaultdict(list)
    for _, row in subset.iterrows():
        groups[row["date"]].append(row["key"])
    for date in groups:
        groups[date] = sorted(groups[date])
    return groups


def od_loss(od_pred, od_true):
    mask_nonzero = (od_true > 0).float()
    mask_zero    = 1.0 - mask_nonzero
    loss = (mask_nonzero * (od_pred - od_true) ** 2).mean() \
         + 0.1 * (mask_zero * od_pred ** 2).mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    processed = cfg["data"]["processed_dir"]
    gcn_cfg   = cfg["gcn"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    edge_index, edge_weight = load_graph(processed)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    manifest   = pd.read_csv(os.path.join(processed, "snapshot_manifest.csv"))
    train_days = group_by_day(manifest, cfg["data"]["train_months"])
    val_days   = group_by_day(manifest, cfg["data"]["val_months"])

    print(f"Train days: {len(train_days)}  |  Val days: {len(val_days)}")

    model     = GCNAutoencoder(in_dim=2,
                               hidden_dim=gcn_cfg["hidden_dim"],
                               embed_dim=gcn_cfg["embedding_dim"]).to(device)
    optimizer = Adam(model.parameters(),
                     lr=gcn_cfg["lr"],
                     weight_decay=gcn_cfg["weight_decay"])
    scheduler = StepLR(optimizer,
                       step_size=gcn_cfg["scheduler_step"],
                       gamma=gcn_cfg["scheduler_gamma"])

    best_val = float("inf")
    epochs   = gcn_cfg["epochs"]

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0.0
        for date, keys in tqdm(train_days.items(), desc=f"Epoch {epoch:02d} train", leave=False):
            feats   = []
            od_true = None
            for key in keys:
                feat, od = load_snapshot(processed, key)
                feats.append(feat.to(device))
                if od_true is None:
                    od_true = od.to(device)
                else:
                    od_true = od_true + od.to(device)  

            optimizer.zero_grad()
            od_pred, _ = model(feats, edge_index, edge_weight)
            loss = od_loss(od_pred, od_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_days)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for date, keys in val_days.items():
                feats   = []
                od_true = None
                for key in keys:
                    feat, od = load_snapshot(processed, key)
                    feats.append(feat.to(device))
                    if od_true is None:
                        od_true = od.to(device)
                    else:
                        od_true = od_true + od.to(device)

                od_pred, _ = model(feats, edge_index, edge_weight)
                val_loss  += od_loss(od_pred, od_true).item()

        val_loss /= len(val_days)
        scheduler.step()
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model_dir = os.path.join(processed, "models", "gcn")
            os.makedirs(model_dir, exist_ok=True)
            best_path = os.path.join(model_dir, "gcn_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (val_loss={best_val:.4f})")

    print("\nExtracting embeddings ...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for date, keys in train_days.items():
            feats = [load_snapshot(processed, key)[0].to(device) for key in keys]
            z = model.encode(feats, edge_index, edge_weight)
            all_embeddings.append(z.cpu().numpy())

    embeddings = np.mean(all_embeddings, axis=0)
    emb_dir  = os.path.join(processed, "models", "gcn")
    os.makedirs(emb_dir, exist_ok=True)
    out_path = os.path.join(emb_dir, "embeddings.npy")
    np.save(out_path, embeddings)
    print(f"Embeddings saved: {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()