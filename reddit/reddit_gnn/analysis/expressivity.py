"""
Expressivity Analysis — Aggregator collision detection for A1 ablation.
Tests whether different aggregators can distinguish structurally similar nodes.
"""

import torch
import torch.nn.functional as F
import numpy as np


def detect_expressivity_collapse(model, data, device, n_pairs=1000):
    """
    For Mean and Max aggregators, test whether expressivity collapse occurs.

    Sample node pairs (u, v) where:
    (a) true labels differ
    (b) their 1-hop neighborhood label distributions are identical

    If h_u = h_v after aggregation → aggregator has collapsed.

    Returns collapse_rate: fraction of pairs where embeddings are identical.
    """
    model.eval()

    with torch.no_grad():
        # Get embeddings
        data_dev = data.to(device)
        h = model.encode(data_dev.x, data_dev.edge_index)
        h = h.cpu()

    y = data.y
    row, col = data.edge_index

    # Build neighborhood label distributions
    print("  Computing neighborhood label distributions...")
    n_classes = y.max().item() + 1
    label_dist = torch.zeros(data.num_nodes, n_classes)

    for node in range(data.num_nodes):
        nbr_mask = row == node
        nbr_labels = y[col[nbr_mask]]
        if len(nbr_labels) > 0:
            counts = nbr_labels.bincount(minlength=n_classes)
            label_dist[node] = counts.float() / counts.sum()

    # Find pairs with different labels but identical label distributions
    print("  Searching for structurally indistinguishable pairs...")
    pairs_found = 0
    collapsed = 0

    test_indices = torch.where(data.test_mask)[0]
    perm = torch.randperm(len(test_indices))

    for i in range(min(len(perm) - 1, n_pairs * 10)):
        for j in range(i + 1, min(i + 10, len(perm))):
            u, v = test_indices[perm[i]].item(), test_indices[perm[j]].item()

            if y[u] != y[v]:
                # Check if label distributions are similar
                dist_diff = (label_dist[u] - label_dist[v]).abs().sum().item()
                if dist_diff < 0.01:  # Nearly identical distributions
                    pairs_found += 1

                    # Check if embeddings collapsed
                    cos_sim = F.cosine_similarity(
                        h[u].unsqueeze(0), h[v].unsqueeze(0)
                    ).item()
                    if cos_sim > 0.99:
                        collapsed += 1

                    if pairs_found >= n_pairs:
                        break
        if pairs_found >= n_pairs:
            break

    collapse_rate = collapsed / max(pairs_found, 1)
    print(f"  Pairs found: {pairs_found}, Collapsed: {collapsed}")
    print(f"  Collapse rate: {collapse_rate:.4f}")

    return collapse_rate, pairs_found, collapsed
