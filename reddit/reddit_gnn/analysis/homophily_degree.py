"""
Homophily × Degree Analysis — Yan et al. (ICDM'22) 3×3 grid.
Used for A4 (structure vs features) presentation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import PREPROCESSED, RESULTS_ROOT


def plot_homophily_degree_heatmap(grid, model_name, save_path=None):
    """
    Plot the 3×4 heatmap of accuracy by homophily regime × degree bin.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    deg_labels = ["1-10", "11-50", "51-200", ">200"]
    hv_labels = ["Low h_v\n(<0.3)", "Mid h_v\n(0.3-0.7)", "High h_v\n(>0.7)"]

    mask = np.isnan(grid)
    sns.heatmap(
        grid, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=deg_labels, yticklabels=hv_labels,
        mask=mask, vmin=0.5, vmax=1.0, ax=ax,
    )
    ax.set_xlabel("Degree Bin", fontsize=12)
    ax.set_ylabel("Homophily Regime", fontsize=12)
    ax.set_title(f"{model_name} — Accuracy by Homophily × Degree", fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def classify_nodes_by_regime(data, h_v=None):
    """
    Classify nodes into Yan et al. regimes for visualization flagging.
    """
    from torch_geometric.utils import degree

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)

    if h_v is None:
        h_v = torch.load(os.path.join(PREPROCESSED, "node_homophily.pt"),
                        weights_only=False)

    regimes = {
        "regime1_low_hom": torch.where((h_v < 0.3))[0],
        "regime2_high_hom_low_deg": torch.where((h_v > 0.7) & (deg < 10))[0],
        "high_degree": torch.topk(deg, 200).indices,
        "low_degree": torch.where(deg == 1)[0][:200],
    }

    for name, indices in regimes.items():
        print(f"  {name}: {len(indices)} nodes")

    return regimes
