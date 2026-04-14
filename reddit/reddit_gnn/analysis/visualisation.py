"""
Visualization — t-SNE + UMAP embedding projection and plotting.
Produces 4 plot types × 6 models = 24 plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import RESULTS_ROOT


def stratified_sample(y_test, test_indices, n_per_class=500, num_classes=41):
    """
    Stratified class sampling: n_per_class nodes from test set per class.
    Ensures all 41 subreddits are equally visible.
    """
    indices_per_class = []
    for c in range(num_classes):
        class_mask = y_test == c
        class_indices = test_indices[class_mask]
        n_sample = min(n_per_class, len(class_indices))
        sampled = class_indices[torch.randperm(len(class_indices))[:n_sample]]
        indices_per_class.append(sampled)
    vis_indices = torch.cat(indices_per_class)
    return vis_indices


def compute_tsne(embeddings, perplexity=30, n_iter=1000, seed=42):
    """t-SNE projection with recommended settings."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        n_iter=n_iter,
        random_state=seed,
    )
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, seed=42):
    """UMAP projection with recommended settings."""
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def compute_embedding_quality_metrics(embeddings_2d, labels):
    """Quantitative embedding quality metrics."""
    metrics = {
        "silhouette": silhouette_score(embeddings_2d, labels, sample_size=5000),
        "davies_bouldin": davies_bouldin_score(embeddings_2d, labels),
        "calinski_harabasz": calinski_harabasz_score(embeddings_2d, labels),
    }
    return metrics


def plot_type1_ground_truth(embeddings_2d, labels, model_name, test_acc,
                           save_path=None, num_classes=41):
    """Plot Type 1 — Ground truth class coloring with 41-color palette."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 41-color palette from tab20 + tab20b
    colors = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
             list(plt.cm.tab20b(np.linspace(0, 1, 20))) + \
             list(plt.cm.Set3(np.linspace(0, 1, 1)))

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[colors[c]], s=3, alpha=0.6, label=f"Class {c}",
            )

    sil = silhouette_score(embeddings_2d, labels, sample_size=5000)
    ax.set_title(f"{model_name} — Ground Truth (Acc: {test_acc:.4f}, Silhouette: {sil:.4f})",
                fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_type2_error_overlay(embeddings_2d, labels, preds, model_name,
                            save_path=None, num_classes=41):
    """Plot Type 2 — Predicted vs true label error overlay."""
    fig, ax = plt.subplots(figsize=(12, 10))
    correct = preds == labels
    incorrect = ~correct

    # 41-color palette
    colors = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
             list(plt.cm.tab20b(np.linspace(0, 1, 20))) + \
             list(plt.cm.Set3(np.linspace(0, 1, 1)))

    # Correctly classified (small dots, by true class)
    for c in range(num_classes):
        mask = correct & (labels == c)
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[colors[c]], s=2, alpha=0.3,
            )

    # Misclassified (large RED x markers)
    if incorrect.sum() > 0:
        ax.scatter(
            embeddings_2d[incorrect, 0], embeddings_2d[incorrect, 1],
            c="red", marker="x", s=30, alpha=0.8, label=f"Errors ({incorrect.sum()})",
        )

    error_rate = incorrect.sum() / len(labels)
    ax.set_title(f"{model_name} — Error Overlay (Error rate: {error_rate:.4f})", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_type3_degree_overlay(embeddings_2d, degrees, model_name, save_path=None):
    """Plot Type 3 — Node degree overlay (viridis colormap)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    log_deg = np.log(degrees + 1)
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=log_deg, cmap="viridis", s=3, alpha=0.6,
    )
    plt.colorbar(scatter, ax=ax, label="log(degree + 1)")
    ax.set_title(f"{model_name} — Degree Overlay", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_type4_cross_model_grid(model_embeddings, model_labels, model_names,
                                model_accs, save_path=None):
    """Plot Type 4 — 2×3 cross-model comparison grid."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    colors = list(plt.cm.tab20(np.linspace(0, 1, 20))) + \
             list(plt.cm.tab20b(np.linspace(0, 1, 20))) + \
             list(plt.cm.Set3(np.linspace(0, 1, 1)))

    for idx, (ax, name, emb, labels, acc) in enumerate(
        zip(axes.flat, model_names, model_embeddings, model_labels, model_accs)
    ):
        num_classes = len(np.unique(labels))
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]], s=2, alpha=0.5)
        ax.set_title(f"{name} (Acc: {acc:.4f})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Cross-Model Embedding Comparison", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
