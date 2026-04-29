# Urban Mobility Topology

**CPSC 4520/5520 Deep Learning Theory and Applications — Final Project**

*Tracy Chen, Lucas Liu, Grace Yin*

Can deep learning models discover the **functional topology** of a city purely from human mobility patterns? This project learns meaningful representations of urban functional relationships by integrating physical distance with mobility flow patterns into a learned graph structure, using the NYC Taxi Trip Duration dataset.

## Architecture

We develop three progressive models:

| Model | Branch | Description |
|-------|--------|-------------|
| **Baseline 1: Trip2Vec** | `trip2vec` | Word2Vec skip-gram on (pickup, dropoff) co-occurrence pairs. No graph structure or temporal modeling. |
| **Baseline 2: GCN Autoencoder** | `gcn-clean-2` | 2-layer GCN encoder on static geographic adjacency graph + inner-product OD decoder. |
| **Proposed: Dual-Graph Adaptive Spatial Model** | `new_method` | Two parallel GCN branches (geographic + dynamic flow adjacency) fused via multi-head cross-attention, with OD reconstruction decoder. |

### Dual-Graph Model (Proposed Method)

```
Day = [Morning_Peak, Midday, Evening_Peak, Night_early, Night_late]

For each snapshot t:
    Node Features [N, 2]
           |
           ├──► GCN Branch (A_geo)   ──► z_geo_t   [N, 64]
           |         static geographic graph
           |
           ├──► GCN Branch (A_flow_t) ──► z_flow_t  [N, 64]
           |         dynamic per-snapshot flow graph
           |
           └──► Multi-Head Cross-Attention Fusion
                        |
                    z_fused_t [N, 64]

Stack: [z_fused_1, ..., z_fused_T]  →  [T, N, 64]
                    |
        Temporal Aggregation Module
        (Bi-LSTM / Temporal Self-Attention / None)
                    |
              z_temporal [N, 64]
                    |
           OD Decoder (bilinear + softplus)
                    |
              od_pred [N, N]
```

**Three temporal modes** (select via `--temporal`):

| Mode | Flag | Description |
|------|------|-------------|
| No temporal | `--temporal none` | Independent snapshots, mean pooling (original) |
| Bi-LSTM | `--temporal lstm` | Bidirectional LSTM over daily sequence |
| Temporal Attention | `--temporal temporal_attention` | Multi-head self-attention with positional encoding |

## Results

| Metric | Trip2Vec | **Dual-Graph** | Improvement |
|--------|----------|----------------|-------------|
| Silhouette Score | 0.188 | **0.781** | +315% |
| Davies-Bouldin Index | 1.776 | **0.429** | -76% (lower is better) |
| Intra/Inter Flow Ratio | 4.473 | **5.994** | +34% |

## Dataset

- **NYC Taxi Trip Duration** (Kaggle, 2017): ~1.46M taxi trips, Jan–Jun 2016
- Spatial discretization: **Uber H3** hexagonal cells (resolution 8, ~460m edge)
- Temporal binning: 5 time bins per day (Morning Peak, Midday, Evening Peak, Night Early, Night Late)
- Active cells after filtering: **642**
- Total snapshots: **909** (across 6 months)

## Setup

```bash
# Create conda environment
conda create -n urban-mobility python=3.11 -y
conda activate urban-mobility
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm \
    h3 torch torch-geometric gensim pyarrow folium contextily pyyaml
```

## Usage

### 1. Preprocessing
This project studies urban mobility patterns from the NYC Taxi Trip Duration dataset using H3-based spatial discretization, trip embeddings, and downstream clustering/visual analysis.

Current pipeline:
- preprocess raw taxi trips into H3 cells and OD snapshots
- train a `Trip2Vec` baseline embedding model
- analyze embeddings with clustering, t-SNE, and geographic maps

## Run

Activate the environment first:

```bash
conda activate urban-mobility
```

Run preprocessing:

```bash
python src/data/preprocess.py --config configs/config.yaml
```

Outputs to `data/processed/`:
- `od_matrices/` — sparse OD matrices per snapshot (raw + log1p)
- `node_features/` — node features [outflow, inflow] per snapshot
- `A_geo.npz` — static geographic adjacency matrix (top-k=8 inverse-distance)
- `cell_index.json` — H3 cell ID to index mapping
- `snapshot_manifest.csv` — snapshot metadata
- `od_empirical_train.npz` — aggregated ground-truth OD matrix

### 2. Train Dual-Graph Model

```bash
# No temporal modeling (independent snapshots)
python src/train_dual_graph.py --config configs/config.yaml --temporal none

# With Bi-LSTM temporal aggregation
python src/train_dual_graph.py --config configs/config.yaml --temporal lstm

# With Temporal Self-Attention aggregation
python src/train_dual_graph.py --config configs/config.yaml --temporal temporal_attention
```

### 3. Analyze Embeddings

```bash
# Analyze no-temporal embeddings
python src/analyze.py \
  --embeddings data/processed/models/dual_graph/embeddings.npy \
  --model_name dual_graph \
  --config configs/config.yaml

# Analyze LSTM embeddings
python src/analyze.py \
  --embeddings data/processed/models/dual_graph/embeddings_lstm.npy \
  --model_name dual_graph_lstm \
  --config configs/config.yaml

# Analyze temporal-attention embeddings
python src/analyze.py \
  --embeddings data/processed/models/dual_graph/embeddings_temporal_attention.npy \
  --model_name dual_graph_temporal_attention \
  --config configs/config.yaml
```

Outputs to `data/processed/analysis/dual_graph/`:
- `clustering_metrics.csv` — Silhouette, DBI, Intra/Inter ratio for each k
- `tsne_dual_graph_k3.png` — t-SNE 2D projection
- `geo_map_dual_graph_k3.png` — geographic cluster map
- `summary.json` — summary metrics

## Project Structure

```
├── configs/config.yaml            # All hyperparameters and data settings
├── environment.yml                # Conda environment spec
├── src/
│   ├── data/preprocess.py         # H3 mapping, OD snapshots, adjacency matrix
│   ├── models/
│   │   └── dual_graph.py          # Dual-Graph Adaptive Spatial Model
│   ├── train_dual_graph.py        # Training script
│   └── analyze.py                 # Clustering, t-SNE, geo maps, Spearman correlation
└── data/
    ├── nyc-taxi-trip-duration/     # Raw CSV (not tracked)
    └── processed/                 # Pipeline outputs
        └── analysis/              # Evaluation results (tracked)
```

## Evaluation Metrics

1. **K-means Clustering** — Silhouette Score & Davies-Bouldin Index
2. **Intra/Inter Flow Ratio** — measures if clusters concentrate real taxi flow within communities
3. **t-SNE Visualization** — 2D embedding projection colored by cluster and flow volume
4. **Geographic Maps** — cluster assignments overlaid on NYC basemap
5. **Spearman Rank Correlation** — cosine similarity of embeddings vs. empirical OD flow
Train the Trip2Vec baseline:

```bash
python src/models/trip2vec.py --config configs/config.yaml
```

Run embedding analysis:

```bash
python src/analyze.py \
  --embeddings data/processed/models/trip2vec/embeddings.npy \
  --model_name trip2vec \
  --config configs/config.yaml
```

Main outputs are written to `data/processed/`, including processed snapshots, trained embeddings, and analysis figures/results.
