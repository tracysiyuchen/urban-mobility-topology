"""
Baseline 1 — Trip2Vec

Word2Vec-style skip-gram model that treats each taxi trip as a two-token sentence:
    [pickup_h3_cell, dropoff_h3_cell]

Learns static cell embeddings purely from co-occurrence patterns, with no graph
structure or temporal modeling. 
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


# Corpus reader 
class TripCorpus:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
    def __iter__(self):
        with open(self.corpus_path) as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 2:
                    yield tokens

    def __len__(self):
        count = 0
        with open(self.corpus_path) as f:
            for _ in f:
                count += 1
        return count


# tqdm progress callback for gensim Word2Vec
class TqdmCallback(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.bar = tqdm(total=total_epochs, desc="Trip2Vec training", unit="epoch")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.bar.set_postfix(loss=f"{loss:.1f}")
        self.bar.update(1)

    def on_train_end(self, _model):
        self.bar.close()


def build_trip_corpus(cfg: dict, model_dir: str) -> str:
    processed_dir = cfg["data"]["processed_dir"]
    trips_path = os.path.join(processed_dir, "processed_trips.csv")
    cell_idx_path = os.path.join(processed_dir, "cell_index.json")

    print("Building Trip2Vec corpus …")
    trips = pd.read_csv(trips_path, parse_dates=["pickup_datetime"])
    with open(cell_idx_path) as f:
        cell_data = json.load(f)

    active_set = set(cell_data["cells"])
    train_months = cfg["data"]["train_months"]
    month_mask = trips["pickup_datetime"].dt.month.isin(train_months)
    active_mask = trips["pickup_h3"].isin(active_set) & trips["dropoff_h3"].isin(active_set)
    corpus_df = trips[month_mask & active_mask][["pickup_h3", "dropoff_h3"]]

    corpus_path = os.path.join(model_dir, "trip_corpus.txt")
    lines = corpus_df["pickup_h3"] + " " + corpus_df["dropoff_h3"]
    lines.to_csv(corpus_path, index=False, header=False)
    print(f"  Corpus saved: {len(lines):,} trips from train months {sorted(train_months)} → {corpus_path}")
    return corpus_path


# Train
def train(cfg: dict) -> np.ndarray:
    processed_dir = cfg["data"]["processed_dir"]
    r2v_cfg = cfg["trip2vec"]

    cell_idx_path = os.path.join(processed_dir, "cell_index.json")
    model_dir = os.path.join(processed_dir, "models", "trip2vec")
    os.makedirs(model_dir, exist_ok=True)

    print("Loading cell index …")
    with open(cell_idx_path) as f:
        cell_data = json.load(f)
    cells = cell_data["cells"]          # ordered list of H3 cell IDs
    N     = len(cells)
    print(f"  {N} active cells")

    corpus_path = build_trip_corpus(cfg, model_dir)

    print("Training Word2Vec (Trip2Vec) …")
    corpus = TripCorpus(corpus_path)
    epochs = r2v_cfg["epochs"]
    model  = Word2Vec(
        sentences   = corpus,
        vector_size = r2v_cfg["embedding_dim"],
        window      = r2v_cfg["window"],
        min_count   = r2v_cfg["min_count"],
        workers     = r2v_cfg["workers"],
        epochs      = epochs,
        sg          = r2v_cfg["sg"],
        negative    = r2v_cfg["negative"],
        compute_loss= True,
        callbacks   = [TqdmCallback(epochs)],
    )

    # Build embedding matrix aligned to cell_index ordering
    dim = r2v_cfg["embedding_dim"]
    embeddings = np.zeros((N, dim), dtype=np.float32)
    missing = 0
    for idx, cell in enumerate(cells):
        if cell in model.wv:
            embeddings[idx] = model.wv[cell]
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing} cells not in W2V vocabulary (zero-padded)")

    # Save
    model.save(os.path.join(model_dir, "trip2vec.model"))
    emb_path = os.path.join(model_dir, "embeddings.npy")
    np.save(emb_path, embeddings)

    print(f"  Embeddings saved: {embeddings.shape} → {emb_path}")
    return embeddings


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    embeddings = train(cfg)

    cell_idx_path = os.path.join(cfg["data"]["processed_dir"], "cell_index.json")
