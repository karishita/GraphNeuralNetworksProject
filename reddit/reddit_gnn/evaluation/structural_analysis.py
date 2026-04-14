"""
Structural Error Analysis — Degree × Homophily grid analysis.
Links prediction errors to graph structural properties.
Based on Yan et al. (ICDM'22) heterophily × degree analysis from Lecture 11.
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import PREPROCESSED


def structural_error_analysis(preds, labels, data, h_v=None):
    """
    Compute accuracy breakdown by degree bins and homophily bins.
    Returns 3×4 heatmap grid for the A4/E1 presentation.

    Args:
        preds: numpy array of predictions (test nodes only)
        labels: numpy array of true labels (test nodes only)
        data: PyG Data object
        h_v: Per-node homophily tensor (loaded from disk if None)
    """
    from torch_geometric.utils import degree

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).numpy()

    if h_v is None:
        h_v_path = os.path.join(PREPROCESSED, "node_homophily.pt")
        h_v = torch.load(h_v_path, weights_only=False).numpy()
    elif isinstance(h_v, torch.Tensor):
        h_v = h_v.numpy()

    test_mask = data.test_mask.numpy()
    test_deg = deg[test_mask]
    test_hv = h_v[test_mask]
    correct = preds == labels

    # 1. Accuracy by degree bin
    deg_bins = [0, 10, 50, 200, float("inf")]
    deg_labels = ["1-10", "11-50", "51-200", ">200"]

    print("\n  Accuracy by degree bin:")
    deg_accs = {}
    for lo, hi, name in zip(deg_bins, deg_bins[1:], deg_labels):
        mask = (test_deg >= lo) & (test_deg < hi)
        if mask.sum() > 0:
            acc = correct[mask].mean()
            deg_accs[name] = acc
            print(f"    Deg {name:>6s}: acc={acc:.4f} n={mask.sum()}")

    # 2. Accuracy by homophily bin (Yan et al. 3 regimes)
    hv_bins = [
        (0, 0.3, "Low-hom (Regime 1)"),
        (0.3, 0.7, "Mid-hom"),
        (0.7, 1.01, "High-hom (Regime 2/3)"),
    ]

    print("\n  Accuracy by homophily regime:")
    hv_accs = {}
    for lo, hi, name in hv_bins:
        mask = (test_hv >= lo) & (test_hv < hi)
        if mask.sum() > 0:
            acc = correct[mask].mean()
            hv_accs[name] = acc
            print(f"    {name:>25s}: acc={acc:.4f} n={mask.sum()}")

    # 3. 3×4 heatmap: homophily-bin × degree-bin
    grid = np.full((3, 4), float("nan"))
    for i, (hlo, hhi, _) in enumerate(hv_bins):
        for j, (dlo, dhi) in enumerate(zip(deg_bins, deg_bins[1:])):
            m = (test_hv >= hlo) & (test_hv < hhi) & (test_deg >= dlo) & (test_deg < dhi)
            if m.sum() > 0:
                grid[i, j] = correct[m].mean()

    print("\n  3×4 heatmap (homophily × degree):")
    print(f"    {'':>25s} | {'1-10':>6s} | {'11-50':>6s} | {'51-200':>6s} | {'>200':>6s}")
    for i, (_, _, name) in enumerate(hv_bins):
        row = " | ".join(f"{grid[i,j]:.4f}" if not np.isnan(grid[i,j]) else "   n/a" for j in range(4))
        print(f"    {name:>25s} | {row}")

    return grid, deg_accs, hv_accs


def identify_boundary_nodes(data, partition_id):
    """
    Identify boundary vs interior nodes for F2 ablation analysis.
    Boundary nodes have at least one neighbor in a different partition.
    """
    row, col = data.edge_index
    cross_partition = partition_id[row] != partition_id[col]

    # Nodes that have any cross-partition edge
    boundary_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    boundary_nodes = row[cross_partition].unique()
    boundary_mask[boundary_nodes] = True

    interior_mask = ~boundary_mask

    n_boundary = boundary_mask.sum().item()
    n_interior = interior_mask.sum().item()
    print(f"  Boundary nodes: {n_boundary:,} ({n_boundary/data.num_nodes*100:.1f}%)")
    print(f"  Interior nodes: {n_interior:,} ({n_interior/data.num_nodes*100:.1f}%)")

    return boundary_mask, interior_mask
