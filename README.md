# Urban Mobility Topology

**CPSC 4520/5520 Deep Learning Theory and Applications — Final Project**

*Tracy Chen, Lucas Liu, Grace Yin*

How well do different mobility embedding strategies capture the **functional community structure** of a city? We ask a concrete question: when urban regions are clustered by their learned embeddings, do those clusters align with real OD flow communities — i.e., does the majority of empirical taxi demand stay *within* discovered zones rather than crossing between them?

We systematically compare three model families — a co-occurrence baseline, a geographic graph autoencoder, and a proposed dual-graph fusion model — across two real-world taxi datasets. Our primary evaluation metric is the **Intra/Inter Flow Ratio** (Region2Vec, 2022), which directly measures OD flow community coherence. We also report Spearman rank correlation between embedding similarity and OD-profile similarity as a complementary pairwise alignment measure, and discuss the trade-off between the two.

---

## Models

| # | Model | Description |
|---|-------|-------------|
| Baseline 1 | **Trip2Vec** | Word2Vec skip-gram on (pickup, dropoff) cell co-occurrence pairs. No graph structure or temporal modeling. |
| Baseline 2 | **GCN Autoencoder** | 2-layer GCN encoder on static geographic adjacency + GRU temporal pooling + inner-product OD decoder. |
| Proposed | **Dual-Graph Adaptive Spatial Model** | Two parallel GCN branches (geographic graph + dynamic flow graph) fused via multi-head cross-attention, with optional temporal aggregation and OD reconstruction decoder. |

### Dual-Graph Architecture

```
For each time snapshot t:
    Node Features [N, 2]
           |
           ├──► GCN (A_geo, static)    ──► z_geo_t   [N, 64]
           └──► GCN (A_flow_t, dynamic) ──► z_flow_t  [N, 64]
                        |
              Multi-Head Cross-Attention Fusion
                        |
                   z_fused_t [N, 64]

Stack snapshots: [z_fused_1, ..., z_fused_T]
                        |
          Temporal Aggregation (one of three modes)
                        |
                  z [N, 64]  ──►  OD Decoder  ──►  od_pred [N, N]
```

**Three temporal modes:**

| Flag | Mode | Description |
|------|------|-------------|
| `--temporal none` | Mean pooling | Independent snapshots, averaged |
| `--temporal lstm` | Bi-LSTM | Bidirectional LSTM over daily snapshot sequence |
| `--temporal temporal_attention` | Temporal Self-Attention | Multi-head self-attention with sinusoidal positional encoding |

---

## Datasets

| Dataset | Source | Period | Trips | Active H3 Cells |
|---------|--------|--------|-------|-----------------|
| **NYC Taxi** | Kaggle (Jan–Jun 2016) | 6 months | ~1.46M | 642 |
| **Porto Taxi** | UCI (Jul 2013–Jun 2014) | 12 months | ~1.67M | 508 |

- Spatial discretization: **Uber H3** hexagonal grid (resolution 8, ~460m edge length)
- Temporal binning: 5 bins per day — Morning Peak (6–10), Midday (10–16), Evening Peak (16–20), Night Early (20–24), Night Late (0–6)

---

## Results

### NYC Taxi (642 cells, train Jan–Apr, val May, test Jun)

| Model | best k | Intra/Inter ↑ | Silhouette ↑ | DBI ↓ | Spearman ρ ↑ |
|-------|--------|--------------|--------------|-------|--------------|
| Trip2Vec | 6 | 6.53 | 0.133 | 2.02 | **0.413** |
| GCN Autoencoder | 3 | 1.28 | 0.804 | 0.887 | 0.156 |
| Dual-Graph | 3 | **11.15** | 0.710 | 0.503 | 0.167 |
| Dual-Graph + LSTM | 3 | 7.44 | 0.776 | **0.434** | 0.039 |
| Dual-Graph + Temporal Attn | 3 | 2.65 | **0.814** | 0.527 | 0.064 |

### Porto Taxi (508 cells, train Jul–Oct, val Nov, test Dec)

| Model | best k | Intra/Inter ↑ | Silhouette ↑ | DBI ↓ | Spearman ρ ↑ |
|-------|--------|--------------|--------------|-------|--------------|
| Trip2Vec | 4 | 3.90 | 0.100 | 2.67 | **0.125** |
| GCN Autoencoder | 3 | 1.26 | 0.766 | 0.509 | -0.027 |
| Dual-Graph | 3 | **5.97** | 0.756 | 0.533 | -0.141 |
| Dual-Graph + LSTM | 3 | 3.87 | 0.762 | 0.569 | -0.303 |
| Dual-Graph + Temporal Attn | 3 | 1.18 | **0.879** | **0.460** | -0.311 |

`best k` is selected by highest Intra/Inter Flow Ratio across k ∈ {3,4,5,6,7,8,10}.


---

## Setup

```bash
conda env create -f environment.yml
conda activate urban-mobility
```

---

## Reproducing Results

All commands assume `conda activate urban-mobility` and are run from the project root.

### Step 1 — Preprocess

```bash
# NYC
python src/data/preprocess.py --config configs/config_nyc.yaml

# Porto
python src/data/preprocess.py --config configs/config_porto.yaml
```

Outputs (written to `data/processed/{nyc,porto}/`):
- `od_matrices/` — sparse OD matrices per snapshot
- `node_features/` — node feature arrays per snapshot
- `A_geo.npz` — static geographic adjacency (top-8 inverse-distance)
- `od_empirical_train.npz` — aggregated ground-truth OD matrix
- `cell_index.json`, `snapshot_manifest.csv`

---

### Step 2 — Train

Replace `CONFIG` with `configs/config_nyc.yaml` or `configs/config_porto.yaml`.

```bash
# Trip2Vec
python src/train_trip2vec.py --config CONFIG

# GCN Autoencoder
python src/train_gcn.py --config CONFIG

# Dual-Graph (three temporal modes)
python src/train_dual_graph.py --config CONFIG --temporal none
python src/train_dual_graph.py --config CONFIG --temporal lstm
python src/train_dual_graph.py --config CONFIG --temporal temporal_attention
```

Embeddings are saved under `data/processed/{nyc,porto}/models/`.

---

### Step 3 — Analyze

```bash
PROC=data/processed/nyc   # or data/processed/porto
CFG=configs/config_nyc.yaml  # or config_porto.yaml

python src/analyze.py --config $CFG --embeddings $PROC/models/trip2vec/embeddings.npy        --model_name trip2vec
python src/analyze.py --config $CFG --embeddings $PROC/models/gcn/embeddings.npy             --model_name gcn
python src/analyze.py --config $CFG --embeddings $PROC/models/dual_graph/embeddings.npy      --model_name dual_graph
python src/analyze.py --config $CFG --embeddings $PROC/models/dual_graph/embeddings_lstm.npy --model_name dual_graph_lstm
python src/analyze.py --config $CFG --embeddings $PROC/models/dual_graph/embeddings_temporal_attention.npy --model_name dual_graph_temporal_attention
```

Outputs written to `data/processed/{nyc,porto}/analysis/{model_name}/`:
- `clustering_metrics.csv` — Silhouette, DBI, Intra/Inter for each k
- `summary.json` — best-k metrics + Spearman ρ
- `tsne_{model}_k{k}.png` — t-SNE projection
- `geo_map_{model}_k{k}.png` — geographic cluster overlay

---

## Evaluation Metrics

| Metric | Role | Description | Best |
|--------|------|-------------|------|
| **Intra/Inter Flow Ratio** | **Primary** | Ratio of empirical OD flow within clusters to flow across clusters (Region2Vec Eq. 3). Directly measures whether discovered zones form coherent mobility communities. | Higher |
| **Spearman ρ** | Complementary | Rank correlation between pairwise embedding cosine similarity and pairwise OD-profile cosine similarity (filtered to active cells above the 25th-percentile flow threshold). Measures fine-grained functional alignment. | Higher |
| **Silhouette Score** | Reference | Geometric compactness and separation of embedding clusters. High values can reflect spatial smoothness rather than functional structure. | Higher |
| **Davies-Bouldin Index** | Reference | Ratio of intra-cluster scatter to inter-cluster separation in embedding space. | Lower |

`best k` is selected by highest Intra/Inter Flow Ratio across k ∈ {3,4,5,6,7,8,10}. Silhouette and DBI are reported for reference but are not used for model selection, as high geometric compactness does not imply meaningful OD flow community structure.

---

## Project Structure

```
urban-mobility-topology/
├── configs/
│   ├── config_nyc.yaml          # NYC dataset hyperparameters
│   └── config_porto.yaml        # Porto dataset hyperparameters
├── environment.yml
├── src/
│   ├── data/
│   │   └── preprocess.py        # H3 mapping, OD snapshots, adjacency matrix
│   ├── models/
│   │   ├── trip2vec.py          # TripCorpus + Word2Vec wrapper
│   │   ├── gcn_autoencoder.py   # GCN encoder + OD decoder
│   │   └── dual_graph.py        # Dual-Graph Adaptive Spatial Model
│   ├── train_trip2vec.py
│   ├── train_gcn.py
│   ├── train_dual_graph.py
│   └── analyze.py               # Clustering, t-SNE, geo maps, Spearman
└── data/
    ├── nyc-taxi-trip-duration/  # Raw CSV (not tracked, download from Kaggle)
    ├── porto-trajectory/        # Raw CSV (not tracked, download from UCI)
    └── processed/
        ├── nyc/analysis/        # NYC evaluation outputs (tracked)
        └── porto/analysis/      # Porto evaluation outputs (tracked)
```

---

## References

- Yan et al., *Region2Vec: Community Detection on Spatial Networks Using Graph Embedding* (2022)
- Yao et al., *Representing Urban Functions through Zone Embedding with Human Mobility Patterns* (2022)
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017)
- Uber H3: [https://h3geo.org](https://h3geo.org)
