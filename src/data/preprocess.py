"""
Data preprocessing pipeline for NYC Taxi Trip Duration dataset.

Steps:
  1. Load & filter trips (remove sub-60s trips)
  2. Map GPS coordinates to H3 cells (resolution 8, ~460m edge)
  3. Filter to active cells (≥ min_trips pickup OR dropoff across full dataset)
  4. Assign each trip to a configured time bin per day
  5. Build per-snapshot OD matrices (log1p-transformed) + node features [outflow, inflow]
  6. Build static geographic adjacency matrix A_geo (inverse-distance, top-k neighbors)
  7. Save mapped trip records for downstream models
  8. Save everything to data/processed/

Usage:
    conda activate urban-mobility
    python src/data/preprocess.py --config configs/config.yaml
"""

import argparse
import json
import os

import h3
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from tqdm import tqdm

# filter
def load_and_filter(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["raw_path"], parse_dates=["pickup_datetime"])
    print(f"  Total trips loaded: {len(df):,}")

    # Drop obvious data errors: trips shorter than min_dur
    min_dur = cfg["data"]["trip_duration_min"]
    df = df[df["trip_duration"] >= min_dur].copy()
    print(f"  After removing trips < {min_dur}s: {len(df):,} trips")
    return df


# Time-bin assignment
def assign_time_bin(hour: int, time_bins: dict) -> str:
    for label, (start, end) in time_bins.items():
        if start <= end:
            if start <= hour < end:
                return label
        else:
            # Support bins that wrap past midnight, e.g. [20, 6].
            if hour >= start or hour < end:
                return label

    raise ValueError(f"Hour {hour} does not match any configured time bin")

def add_temporal_fields(df: pd.DataFrame, time_bins: dict) -> pd.DataFrame:
    df["date"]     = df["pickup_datetime"].dt.date
    df["hour"]     = df["pickup_datetime"].dt.hour
    df["time_bin"] = df["hour"].apply(lambda h: assign_time_bin(h, time_bins))
    return df



# H3 mapping
def map_to_h3(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    print(f"Mapping GPS coordinates to H3 resolution {resolution} …")
    df["pickup_h3"] = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(df["pickup_latitude"], df["pickup_longitude"])]
    df["dropoff_h3"] = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(df["dropoff_latitude"], df["dropoff_longitude"])]
    print(f"  Unique pickup cells:  {df['pickup_h3'].nunique()}")
    print(f"  Unique dropoff cells: {df['dropoff_h3'].nunique()}")
    return df


# baseline: min total trips (pickup OR dropoff) to keep a cell
def get_active_cells(df: pd.DataFrame, min_trips: int) -> list:
    pickup_counts  = df["pickup_h3"].value_counts()
    dropoff_counts = df["dropoff_h3"].value_counts()
    active = sorted(
        set(pickup_counts[pickup_counts   >= min_trips].index) |
        set(dropoff_counts[dropoff_counts >= min_trips].index)
    )
    print(f"  Active cells (≥{min_trips} trips as pickup or dropoff): {len(active)}")
    return active

# OD matrices + node features per snapshot

def build_snapshots(df: pd.DataFrame, active_cells: list, out_dir: str) -> list:
    # For each (date, time_bin) snapshot build:
    #   - od_matrix  [N, N] — log1p trip count between each pair of active cells
    #   - node_feat  [N, 2] — log1p [outflow_sum, inflow_sum] per active cell
    # Snapshots are saved as individual compressed .npz files.
    
    cell_to_idx = {c: i for i, c in enumerate(active_cells)}
    N = len(active_cells)
    active_set = set(active_cells)

    mask      = df["pickup_h3"].isin(active_set) & df["dropoff_h3"].isin(active_set)
    df_active = df[mask].copy()
    print(f"  Trips with both endpoints in active cells: {len(df_active):,}")

    df_active["pu_idx"] = df_active["pickup_h3"].map(cell_to_idx)
    df_active["do_idx"] = df_active["dropoff_h3"].map(cell_to_idx)

    od_dir   = os.path.join(out_dir, "od_matrices")
    feat_dir = os.path.join(out_dir, "node_features")
    os.makedirs(od_dir,   exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    groups    = list(df_active.groupby(["date", "time_bin"]))
    snapshots = []
    print(f"  Building {len(groups)} snapshots …")

    for (date, tbin), grp in tqdm(groups, desc="snapshots"):
        rows   = grp["pu_idx"].values
        cols   = grp["do_idx"].values
        od_raw = sp.coo_matrix(
            (np.ones(len(grp), dtype=np.float32), (rows, cols)), shape=(N, N)
        ).toarray()

        # od_raw  — raw integer trip counts, used for empirical OD analysis
        # od_log  — log1p(od_raw), used as training target (normalises scale)
        od_log    = np.log1p(od_raw).astype(np.float32)
        outflow   = np.log1p(od_raw.sum(axis=1)).astype(np.float32)  # [N]
        inflow    = np.log1p(od_raw.sum(axis=0)).astype(np.float32)  # [N]
        node_feat = np.stack([outflow, inflow], axis=1)               # [N, 2]

        key = f"{date}_{tbin.replace(' ', '_')}"
        # Save both: log1p version for training, raw counts for analysis
        np.savez_compressed(os.path.join(od_dir,   f"{key}.npz"),
                            od=od_log, od_raw=od_raw.astype(np.int32))
        np.savez_compressed(os.path.join(feat_dir, f"{key}.npz"), feat=node_feat)
        snapshots.append({"date": str(date), "time_bin": tbin, "key": key, "n_trips": len(grp)})

    return snapshots


# Geographic adjacency matrix

def build_geo_adj(active_cells: list, top_k: int, out_dir: str) -> sp.csr_matrix:
    # Build a row-normalised geographic adjacency matrix A_geo where each cell
    # connects to its top_k nearest neighbours weighted by inverse Haversine distance.
    # Saved once — this matrix is static across all snapshots and models.
    print(f"Building geographic adjacency matrix (top-k={top_k}) …")
    N         = len(active_cells)
    centroids = np.array([h3.cell_to_latlng(c) for c in active_cells])  # [N, 2] lat/lon

    lat = np.radians(centroids[:, 0])
    lon = np.radians(centroids[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a    = (np.sin(dlat / 2) ** 2
            + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
    dist_km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    np.fill_diagonal(dist_km, np.inf)
    inv_dist = 1.0 / (dist_km + 1e-6)

    rows, cols, vals = [], [], []
    for i in range(N):
        top_idx = np.argsort(inv_dist[i])[::-1][:top_k]
        for j in top_idx:
            rows.append(i)
            cols.append(j)
            vals.append(float(inv_dist[i, j]))

    A_geo  = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    row_sums = np.array(A_geo.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_geo  = sp.diags(1.0 / row_sums) @ A_geo

    sp.save_npz(os.path.join(out_dir, "A_geo.npz"), A_geo.astype(np.float32))
    print(f"  A_geo saved: shape={A_geo.shape}, nnz={A_geo.nnz}")
    return A_geo


def save_processed_trips(df: pd.DataFrame, out_dir: str):
    """
    Save the H3-mapped trip table once so downstream models can build
    task-specific inputs without redoing preprocessing.
    """
    trip_cols = ["pickup_datetime", "pickup_h3", "dropoff_h3", "date", "hour", "time_bin"]
    trip_path = os.path.join(out_dir, "processed_trips.csv")
    df[trip_cols].to_csv(trip_path, index=False)
    print(f"  Processed trips saved: {len(df):,} rows → {trip_path}")


def build_empirical_od(manifest: pd.DataFrame, active_cells: list,
                       out_dir: str, train_months: list):
    """
    Aggregate raw trip counts across all training snapshots into a single
    empirical OD matrix [N, N] of true flow magnitudes.

    This is the ground-truth used by analyze.py for:
      - Intra/Inter Flow Ratio  (must use raw counts, not log-summed values)
      - Spearman correlation    (must use raw counts)
    Saved once here so analyze.py never has to loop through 728 snapshot files.
    """
    print("Building aggregated empirical OD matrix (train months only) …")
    N           = len(active_cells)
    od_empirical = np.zeros((N, N), dtype=np.int64)
    od_dir      = os.path.join(out_dir, "od_matrices")

    train_keys = manifest[
        pd.to_datetime(manifest["date"]).dt.month.isin(train_months)
    ]["key"].tolist()

    for key in tqdm(train_keys, desc="aggregating OD"):
        od_raw = np.load(os.path.join(od_dir, f"{key}.npz"))["od_raw"]
        od_empirical += od_raw.astype(np.int64)

    path = os.path.join(out_dir, "od_empirical_train.npz")
    np.savez_compressed(path, od=od_empirical.astype(np.float32))
    print(f"  Empirical OD saved: total flow = {od_empirical.sum():,} trips → {path}")


# Main

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df = load_and_filter(cfg)
    df = map_to_h3(df, cfg["data"]["h3_resolution"])

    active_cells = get_active_cells(df, cfg["data"]["active_cell_min_trips"])

    with open(os.path.join(out_dir, "cell_index.json"), "w") as f:
        json.dump({"cells": active_cells,
                   "cell_to_idx": {c: i for i, c in enumerate(active_cells)}}, f)
    print(f"  Cell index saved ({len(active_cells)} cells)")

    df = add_temporal_fields(df, cfg["data"]["time_bins"])

    snapshots = build_snapshots(df, active_cells, out_dir)

    manifest = pd.DataFrame(snapshots)
    manifest.to_csv(os.path.join(out_dir, "snapshot_manifest.csv"), index=False)
    print(f"  Manifest saved: {len(manifest)} snapshots")
    print(f"  Snapshots per time_bin:\n{manifest['time_bin'].value_counts().sort_index()}")

    build_geo_adj(active_cells, cfg["data"]["geo_adj_top_k"], out_dir)
    save_processed_trips(df, out_dir)
    build_empirical_od(manifest, active_cells, out_dir, cfg["data"]["train_months"])

    print("\nPreprocessing complete.")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
