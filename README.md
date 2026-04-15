# Urban Mobility Topology

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
