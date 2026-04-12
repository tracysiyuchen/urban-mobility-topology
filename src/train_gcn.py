import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm

from src.models.gcn_autoencoder import GCNAutoencoder

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_graph(processed_dir):
    """Load geographic adjacency matrix as PyG edge_index + edge_weight."""
    A = sp.load_npz(os.path.join(processed_dir, "A_geo.npz"))
    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    return edge_index, edge_weight.float()


def load_snapshot(processed_dir, key):
    """Load node features and OD target for one time snapshot."""
    feat = np.load(os.path.join(processed_dir, "node_features", f"{key}.npz"))["feat"]  # [N, 2]
    od   = sp.load_npz(os.path.join(processed_dir, "od_matrices", f"{key}_log.npz"))
    od_dense = torch.tensor(od.toarray(), dtype=torch.float32)
    feat = torch.tensor(feat, dtype=torch.float32)
    return feat, od_dense


def od_loss(od_pred, od_true):
    """MSE loss, weighted to emphasise non-zero flows."""
    mask_nonzero = (od_true > 0).float()
    mask_zero    = 1.0 - mask_nonzero
    loss = (mask_nonzero * (od_pred - od_true) ** 2).mean() \
         + 0.1 * (mask_zero * od_pred ** 2).mean()
    return loss


def main():
    cfg          = load_config()
    processed    = cfg["data"]["processed_dir"]
    embed_dim    = cfg["trip2vec"]["embedding_dim"]   # reuse same dim = 64
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #load graph structure
    edge_index, edge_weight = load_graph(processed)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    manifest = pd.read_csv(os.path.join(processed, "snapshot_manifest.csv"))
    train_keys = manifest[manifest["date"].apply(
        lambda d: int(d.split("-")[1])).isin(cfg["data"]["train_months"])]["key"].tolist()
    val_keys   = manifest[manifest["date"].apply(
        lambda d: int(d.split("-")[1])).isin(cfg["data"]["val_months"])]["key"].tolist()

    print(f"Train snapshots: {len(train_keys)}  |  Val snapshots: {len(val_keys)}")

    #modeling
    model = GCNAutoencoder(in_dim=2, hidden_dim=128, embed_dim=embed_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val = float("inf")
    epochs   = 20

    for epoch in range(1, epochs + 1):
        #train
        model.train()
        train_loss = 0.0
        for key in tqdm(train_keys, desc=f"Epoch {epoch:02d} train", leave=False):
            feat, od_true = load_snapshot(processed, key)
            feat    = feat.to(device)
            od_true = od_true.to(device)

            optimizer.zero_grad()
            od_pred, _ = model(feat, edge_index, edge_weight)
            loss = od_loss(od_pred, od_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_keys)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for key in val_keys:
                feat, od_true = load_snapshot(processed, key)
                feat    = feat.to(device)
                od_true = od_true.to(device)
                od_pred, _ = model(feat, edge_index, edge_weight)
                val_loss += od_loss(od_pred, od_true).item()
        val_loss /= len(val_keys)

        scheduler.step()
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs("data/processed/models", exist_ok=True)
            torch.save(model.state_dict(), "data/processed/models/gcn_autoencoder_best.pt")
            print(f"  ✓ Best model saved (val_loss={best_val:.4f})")

    print("\nExtracting embeddings …")
    model.load_state_dict(torch.load("data/processed/models/gcn_autoencoder_best.pt"))
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for key in train_keys:
            feat, _ = load_snapshot(processed, key)
            feat = feat.to(device)
            z = model.encode(feat, edge_index, edge_weight)
            all_embeddings.append(z.cpu().numpy())

    embeddings = np.mean(all_embeddings, axis=0)  # [N, embed_dim]
    out_path = "data/processed/analysis/gcn/embeddings.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Embeddings saved: {embeddings.shape} → {out_path}")


if __name__ == "__main__":
    main()