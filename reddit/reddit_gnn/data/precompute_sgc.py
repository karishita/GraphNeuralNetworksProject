"""
Stage 2A — SGC Offline Feature Smoothing.
Precomputes X_K = A_hat^K * X for K=1..5.
Uses sparse matrix multiply to avoid OOM.
"""

import torch
import torch.nn.functional as F
import os
import sys
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import SGC_DIR


def compute_normalized_adjacency(edge_index, num_nodes):
    """
    Compute A_hat = D^{-1/2} (A + I) D^{-1/2} (Kipf symmetric normalization).
    Returns edge_index with self-loops and normalization edge weights.
    """
    from torch_geometric.utils import add_self_loops, degree

    # Add self-loops: A_hat = A + I
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Compute degree of each node in A + I
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

    # D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0  # isolated node guard

    # Edge weights: d_i^{-1/2} * d_j^{-1/2}
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, norm


def pairwise_cos_sim_sample(X, n_pairs=5000):
    """Sample-based pairwise cosine similarity to track oversmoothing."""
    idx = torch.randperm(X.shape[0])[: n_pairs * 2]
    a, b = X[idx[:n_pairs]], X[idx[n_pairs:]]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1).mean().item()


def precompute_sgc_features(data, max_K=5, save_dir=None, norm_type="symmetric"):
    """
    Precompute X_K = A_hat^K * X for K=1..max_K.
    Uses torch_sparse SparseTensor for efficient sparse-dense multiply.

    Args:
        data: PyG Data object (normalized features expected)
        max_K: Maximum propagation hops (saves intermediate K too)
        save_dir: Directory to save X_K tensors
        norm_type: 'symmetric' | 'row' | 'no_selfloop' (for C2 ablation)
    """
    if save_dir is None:
        save_dir = SGC_DIR

    os.makedirs(save_dir, exist_ok=True)

    print(f"[2A] SGC Feature Precomputation (K=1..{max_K}, norm={norm_type})")

    # Build normalized adjacency
    if norm_type == "symmetric":
        edge_index, norm_weights = compute_normalized_adjacency(
            data.edge_index, data.num_nodes
        )
    elif norm_type == "row":
        edge_index, norm_weights = _compute_row_normalized(
            data.edge_index, data.num_nodes
        )
    elif norm_type == "no_selfloop":
        edge_index, norm_weights = _compute_no_selfloop(
            data.edge_index, data.num_nodes
        )
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    # Build SparseTensor for efficient multiplication
    from torch_sparse import SparseTensor

    N = data.num_nodes
    row, col = edge_index
    adj = SparseTensor(
        row=row, col=col, value=norm_weights, sparse_sizes=(N, N)
    )

    X = data.x.clone()  # [N, 602]
    cos_sim_log = []

    total_start = time.time()

    for k in range(1, max_K + 1):
        t0 = time.time()
        X = adj @ X  # Sparse-dense multiply
        elapsed = time.time() - t0

        cos_sim = pairwise_cos_sim_sample(X)
        cos_sim_log.append({"K": k, "cos_sim": cos_sim})

        save_path = os.path.join(save_dir, f"reddit_sgc_K{k}.pt")
        torch.save(X, save_path)
        size_mb = os.path.getsize(save_path) / (1024**2)

        print(
            f"  K={k}: shape={list(X.shape)}, "
            f"cos_sim={cos_sim:.4f}, "
            f"time={elapsed:.1f}s, "
            f"size={size_mb:.0f}MB → {save_path}"
        )

    total_time = time.time() - total_start
    print(f"  Total precomputation time: {total_time:.1f}s")

    # Save oversmoothing log
    log_path = os.path.join(save_dir, "sgc_oversmoothing_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "cos_sim"])
        writer.writeheader()
        writer.writerows(cos_sim_log)
    print(f"  Oversmoothing log: {log_path}")

    return X


def _compute_row_normalized(edge_index, num_nodes):
    """Row-normalized adjacency: D^{-1}(A+I)."""
    from torch_geometric.utils import add_self_loops, degree

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float("inf")] = 0
    norm = deg_inv[row]
    return edge_index, norm


def _compute_no_selfloop(edge_index, num_nodes):
    """Symmetric normalization WITHOUT self-loops: D^{-1/2}AD^{-1/2}."""
    from torch_geometric.utils import degree

    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm


if __name__ == "__main__":
    from reddit_gnn.data.normalize import load_normalized_data

    data, _, _ = load_normalized_data()
    precompute_sgc_features(data, max_K=5)
