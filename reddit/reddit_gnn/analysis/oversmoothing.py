"""
Oversmoothing Analysis — Embedding variance tracking per layer.
Used for ablations A2 (SAGE depth), D3 (GAT depth), E3 (GATv2 depth).
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_embedding_variance_per_layer(model, data, device, model_type="default"):
    """
    For models supporting layer-wise embedding extraction,
    compute per-layer embedding variance to track oversmoothing.

    Returns:
        dict with 'variance', 'intra_class_sim', 'inter_class_sim' lists
    """
    model.eval()
    variances = []
    intra_sim = []
    inter_sim = []

    with torch.no_grad():
        h = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        for layer_idx, conv in enumerate(model.convs):
            if hasattr(conv, '__call__'):
                h = conv(h, edge_index)
            if layer_idx < len(model.convs) - 1:
                h = h.relu()

            # Compute embedding variance
            var = h.var(dim=0).mean().item()
            variances.append(var)

            # Compute same/diff class cosine similarity
            h_norm = F.normalize(h, dim=1)
            s_intra, s_inter = _sample_cosine_sim(h_norm, y, n_pairs=2000)
            intra_sim.append(s_intra)
            inter_sim.append(s_inter)

    return {
        "variance": variances,
        "intra_class_sim": intra_sim,
        "inter_class_sim": inter_sim,
    }


def _sample_cosine_sim(h_norm, y, n_pairs=2000):
    """
    Sample-based cosine similarity for same-class and different-class pairs.
    """
    N = h_norm.shape[0]
    idx = torch.randperm(N, device=h_norm.device)[:n_pairs * 2]
    a_idx = idx[:n_pairs]
    b_idx = idx[n_pairs:]

    a = h_norm[a_idx]
    b = h_norm[b_idx]
    sim = (a * b).sum(dim=1)

    same_mask = y[a_idx] == y[b_idx]
    diff_mask = ~same_mask

    intra = sim[same_mask].mean().item() if same_mask.sum() > 0 else 0.0
    inter = sim[diff_mask].mean().item() if diff_mask.sum() > 0 else 0.0

    return intra, inter


def oversmoothing_summary(stats, model_name=""):
    """Print oversmoothing analysis summary."""
    print(f"\n  Oversmoothing Analysis: {model_name}")
    print(f"  {'Layer':>6s} | {'Variance':>10s} | {'Intra-sim':>10s} | {'Inter-sim':>10s} | {'Gap':>8s}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

    for i, (var, intra, inter) in enumerate(zip(
        stats["variance"], stats["intra_class_sim"], stats["inter_class_sim"]
    )):
        gap = intra - inter
        print(f"  {i+1:>6d} | {var:>10.4f} | {intra:>10.4f} | {inter:>10.4f} | {gap:>8.4f}")
