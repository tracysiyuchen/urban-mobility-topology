"""
Embedding Analysis — works for all three models (Region2Vec, ST-GCN AE, Dual-Graph).

Metrics (per the project proposal):
  1. k-means clustering → Silhouette Score + Davies-Bouldin Index
  2. Intra/Inter Flow Ratio (from Region2Vec paper, Eq. 3)
  3. t-SNE 2D projection (colored by borough / flow volume quartile)
  4. Spearman rank correlation: embedding similarity vs. empirical OD-profile similarity
"""

import argparse
import json
import os

import contextily as ctx
import folium
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize


# Load data
def load_resources(cfg: dict):
    processed_dir = cfg["data"]["processed_dir"]

    with open(os.path.join(processed_dir, "cell_index.json")) as f:
        cell_data = json.load(f)
    cells = cell_data["cells"]

    A_geo = sp.load_npz(os.path.join(processed_dir, "A_geo.npz"))
    od_path = os.path.join(processed_dir, "od_empirical_train.npz")
    od_agg  = np.load(od_path)["od"]

    return cells, od_agg


# 1. k-means + Silhouette + DBI
def run_kmeans(embeddings: np.ndarray, k_range: list, seed: int) -> pd.DataFrame:
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels)
        dbi = davies_bouldin_score(embeddings, labels)
        results.append({"k": k, "silhouette": sil, "dbi": dbi, "labels": labels})
        print(f"  k={k:2d}  Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return results


# 2. Intra/Inter Flow Ratio  (Region2Vec paper, Eq. 3)
def intra_inter_flow_ratio(od_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio = Σ_{ci=cj} s_ij  /  Σ_{ci≠cj} s_ij
    where s_ij = empirical OD flow between cells i and j.
    Higher = clusters concentrate flow within communities.
    """
    N = len(labels)
    same_cluster = labels[:, None] == labels[None, :]   # [N, N] bool
    intra = od_matrix[same_cluster].sum()
    inter = od_matrix[~same_cluster].sum()
    if inter == 0:
        return float("inf")
    return float(intra / inter)


# 3. Geographic map plots
# 10 visually distinct colours for up to 10 clusters
CLUSTER_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]

def _cell_centroids(cells: list) -> np.ndarray:
    """Return [N, 2] array of (lat, lon) centroids for each H3 cell."""
    return np.array([h3.cell_to_latlng(c) for c in cells])


def _to_web_mercator(lons: np.ndarray, lats: np.ndarray):
    R = 6378137.0
    xs = np.radians(lons) * R
    ys = np.log(np.tan(np.pi / 4 + np.radians(lats) / 2)) * R
    return xs, ys


def plot_geo_static(cells: list, labels: np.ndarray, od_matrix: np.ndarray,
                    model_name: str, k: int, out_dir: str):
    centroids = _cell_centroids(cells)   # [N, 2] (lat, lon)
    lons = centroids[:, 1]
    lats = centroids[:, 0]
    xs, ys = _to_web_mercator(lons, lats)

    node_flow = od_matrix.sum(axis=1) + od_matrix.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Geographic cluster map — {model_name} (k={k})", fontsize=13)

    # Left: clusters
    for cluster_id in range(k):
        mask = labels == cluster_id
        axes[0].scatter(xs[mask], ys[mask],
                        c=CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)],
                        s=45, alpha=0.85, label=f"Cluster {cluster_id}",
                        zorder=3)
    ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron, zoom=11)
    axes[0].set_title("k-means clusters")
    axes[0].set_axis_off()
    axes[0].legend(markerscale=1.5, fontsize=8, loc="lower right")

    # Right: flow volume
    sc = axes[1].scatter(xs, ys, c=np.log1p(node_flow),
                         cmap="YlOrRd", s=45, alpha=0.85, zorder=3)
    ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron, zoom=11)
    plt.colorbar(sc, ax=axes[1], label="log(total flow)", shrink=0.7)
    axes[1].set_title("Total flow volume")
    axes[1].set_axis_off()

    plt.tight_layout()
    path = os.path.join(out_dir, f"geo_map_{model_name}_k{k}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Static geo map saved → {path}")



# 4. t-SNE
def run_tsne(embeddings: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000)
    return tsne.fit_transform(embeddings)


def plot_tsne(xy: np.ndarray, labels: np.ndarray, od_matrix: np.ndarray,
              model_name: str, k: int, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"t-SNE — {model_name} (k={k})", fontsize=13)

    # Left: colored by cluster
    scatter = axes[0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=30, alpha=0.8)
    axes[0].set_title("Colored by k-means cluster")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    # Right: colored by total flow volume (log-scale)
    node_flow = od_matrix.sum(axis=1) + od_matrix.sum(axis=0)
    axes[1].scatter(xy[:, 0], xy[:, 1], c=np.log1p(node_flow), cmap="YlOrRd", s=30, alpha=0.8)
    axes[1].set_title("Colored by log(total flow volume)")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tsne_{model_name}_k{k}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE plot saved → {path}")


# 4. Spearman: embedding cosine similarity vs OD-profile cosine similarity
def spearman_sim_vs_od_profile(embeddings: np.ndarray, od_matrix: np.ndarray) -> tuple:
    # Compare node-pair similarity in embedding space against similarity in
    # empirical mobility profiles. Each OD row is treated as one region's
    # aggregate travel pattern to all other regions.
    #
    # Only pairs where BOTH cells have meaningful OD flow are included.
    # Zero-flow cells contribute near-zero od_cos for all their pairs, which
    # dominates the rank correlation and suppresses the true signal.
    emb_norm = normalize(embeddings, norm="l2")
    emb_cos = emb_norm @ emb_norm.T                      # [N, N]

    od_sym = od_matrix + od_matrix.T                     # combine in + out flow
    od_norm = normalize(od_sym, norm="l2")
    od_cos = od_norm @ od_norm.T                         # [N, N]

    # Filter to cells above the 25th percentile of non-zero total flow
    cell_flow = od_sym.sum(axis=1)
    nonzero_flows = cell_flow[cell_flow > 0]
    threshold = float(np.percentile(nonzero_flows, 25)) if len(nonzero_flows) else 0.0
    active_idx = np.where(cell_flow > threshold)[0]
    n_active = len(active_idx)

    emb_sub = emb_cos[np.ix_(active_idx, active_idx)]   # [M, M]
    od_sub  = od_cos[np.ix_(active_idx, active_idx)]    # [M, M]

    idx = np.triu_indices(n_active, k=1)
    rho, pval = spearmanr(emb_sub[idx], od_sub[idx])
    print(f"  (active cells: {n_active}/{len(embeddings)}, pairs: {len(idx[0])})")
    return float(rho), float(pval)


# Main
def main(embeddings_path: str, model_name: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    analysis_cfg = cfg["analysis"]
    seed = analysis_cfg["random_seed"]
    k_range = analysis_cfg["kmeans_k_range"]

    embeddings = np.load(embeddings_path)
    cells, od_agg = load_resources(cfg)
    N = len(cells)
    
    out_dir = os.path.join(cfg["data"]["processed_dir"], "analysis", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. k-means ──────────────────────────────────────────────────────────
    print("\n[1] k-means clustering")
    kmeans_results = run_kmeans(embeddings, k_range, seed)

    # ── 2. Intra/Inter Flow Ratio ────────────────────────────────────────────
    print("\n[2] Intra/Inter Flow Ratio")
    best_k_result = None
    all_rows = []
    for r in kmeans_results:
        ratio = intra_inter_flow_ratio(od_agg, r["labels"])
        r["intra_inter_ratio"] = ratio
        print(f"  k={r['k']:2d}  Intra/Inter={ratio:.4f}")
        all_rows.append({
            "k":                r["k"],
            "silhouette":       r["silhouette"],
            "dbi":              r["dbi"],
            "intra_inter_ratio":ratio,
        })
        # Pick k with best intra/inter flow ratio — directly measures whether clusters
        # correspond to OD flow communities, aligned with the functional topology goal.
        if best_k_result is None or r["intra_inter_ratio"] > best_k_result["intra_inter_ratio"]:
            best_k_result = r

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = os.path.join(out_dir, "clustering_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Best k by Intra/Inter Flow Ratio: k={best_k_result['k']}")
    print(f"  Metrics table saved → {metrics_path}")

    # ── 3. t-SNE ────────────────────────────────────────────────────────────
    print("\n[3] t-SNE")
    xy = run_tsne(embeddings, analysis_cfg["tsne_perplexity"], seed)
    plot_tsne(xy, best_k_result["labels"], od_agg, model_name, best_k_result["k"], out_dir)

    # ── 3b. Geographic maps ──────────────────────────────────────────────────
    print("\n[3b] Geographic maps")
    plot_geo_static(cells, best_k_result["labels"], od_agg,model_name, best_k_result["k"], out_dir)

    # ── 4. Spearman correlation ──────────────────────────────────────────────
    print("\n[4] Spearman rank correlation (embedding sim vs OD-profile sim)")
    rho_profile, pval_profile = spearman_sim_vs_od_profile(embeddings, od_agg)
    print(f"  ρ = {rho_profile:.4f}  (p = {pval_profile:.2e})")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "model":                  model_name,
        "n_cells":                N,
        "best_k":                 best_k_result["k"],
        "best_intra_inter":       best_k_result["intra_inter_ratio"],
        "silhouette_at_best_k":   best_k_result["silhouette"],
        "dbi_at_best_k":          best_k_result["dbi"],
        "spearman_profile_rho":   rho_profile,
        "spearman_profile_pval":  pval_profile,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        import json as _json
        _json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SUMMARY — {model_name}")
    print(f"{'='*50}")
    for k, v in summary.items():
        if k != "model":
            print(f"  {k:<25} {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help="Path to .npy embeddings file")
    parser.add_argument("--model_name", required=True,
                        help="Model name for labeling outputs (e.g. region2vec)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    main(args.embeddings, args.model_name, args.config)
