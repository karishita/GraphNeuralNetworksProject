"""
Attention Analysis — GAT/GATv2 attention weight inspection.
D4: Static attention collapse and hub concentration.
E1: Dynamic vs static attention ranking comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau


@torch.no_grad()
def extract_attention_weights(model, data, device, sample_node_ids, layer=0):
    """
    Extract attention weights for sampled nodes.

    Args:
        model: GAT or GATv2 model
        data: PyG Data object
        sample_node_ids: list of node indices to analyze
        layer: Which layer's attention to extract (0 = first)

    Returns:
        dict: {node_id: {neighbor_id: attention_weight}}
    """
    from torch_geometric.loader import NeighborLoader

    sample_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    sample_ids = torch.tensor(sample_node_ids)
    sample_mask[sample_ids] = True

    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=128,
        input_nodes=sample_mask,
    )

    attn_dict = {}
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        _, attn_weights_list = model(
            batch.x, batch.edge_index, return_attention_weights=True
        )

        edge_idx, attn = attn_weights_list[layer]

        for i in range(batch.batch_size):
            global_id = batch.n_id[i].item()
            if global_id in sample_node_ids:
                # Find edges where target == node i
                mask = edge_idx[1] == i
                nbrs = batch.n_id[edge_idx[0][mask]].tolist()
                weights = attn[mask].mean(dim=-1).tolist()  # avg across heads
                attn_dict[global_id] = dict(zip(nbrs, weights))

    return attn_dict


def compute_attention_entropy(attn_dict):
    """
    Compute attention entropy per node.
    H(alpha) = -sum(alpha_ij * log(alpha_ij))
    Low entropy = selective attention.
    High entropy = near-uniform (GCN-like).
    """
    entropies = {}
    for node_id, nbr_weights in attn_dict.items():
        weights = np.array(list(nbr_weights.values()))
        weights = weights / (weights.sum() + 1e-10)  # re-normalize
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights) + 1e-10)
        entropies[node_id] = {
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": entropy / (max_entropy + 1e-10),
            "num_neighbors": len(weights),
        }
    return entropies


def hub_concentration_test(attn_dict, top_k=10):
    """
    D4: Hub Concentration Test.
    Compute what fraction of total attention goes to globally top-attended nodes.
    If GAT's static attention is collapsed, a few global hubs will capture
    disproportionate attention from ALL query nodes.
    """
    # Aggregate total attention received by each neighbor across all query nodes
    total_attn_received = {}
    for node_id, nbr_weights in attn_dict.items():
        for nbr, w in nbr_weights.items():
            total_attn_received[nbr] = total_attn_received.get(nbr, 0) + w

    # Sort by total attention
    sorted_nbrs = sorted(total_attn_received.items(), key=lambda x: x[1], reverse=True)
    top_hubs = sorted_nbrs[:top_k]
    total = sum(v for _, v in sorted_nbrs)

    hub_fraction = sum(v for _, v in top_hubs) / (total + 1e-10)

    print(f"\n  Hub Concentration Test (top-{top_k}):")
    print(f"    Total attention distributed: {total:.2f}")
    print(f"    Top-{top_k} hubs receive: {hub_fraction*100:.1f}% of total attention")
    for nbr, w in top_hubs[:5]:
        print(f"      Node {nbr}: received {w:.4f} total attention")

    return hub_fraction, top_hubs


def compare_attention_rankings(attn_dict_gat, attn_dict_gatv2, n_pairs=50):
    """
    E1: Compare neighbor attention rankings between GAT (static) and GATv2 (dynamic).
    For each pair of query nodes, compute Kendall's tau of neighbor rankings.

    GAT (static): tau ≈ 1.0 (identical rankings — confirms static)
    GATv2 (dynamic): tau < 1.0 (varying rankings — confirms dynamic)
    """
    results = {"gat_taus": [], "gatv2_taus": []}

    for model_name, attn_dict in [("gat", attn_dict_gat), ("gatv2", attn_dict_gatv2)]:
        node_ids = list(attn_dict.keys())
        taus = []

        for i in range(min(n_pairs, len(node_ids) - 1)):
            for j in range(i + 1, min(i + 2, len(node_ids))):
                n1, n2 = node_ids[i], node_ids[j]
                # Find common neighbors
                common = set(attn_dict[n1].keys()) & set(attn_dict[n2].keys())
                if len(common) >= 3:
                    common = sorted(common)
                    rank1 = [attn_dict[n1][c] for c in common]
                    rank2 = [attn_dict[n2][c] for c in common]
                    tau, _ = kendalltau(rank1, rank2)
                    if not np.isnan(tau):
                        taus.append(tau)

        results[f"{model_name}_taus"] = taus
        mean_tau = np.mean(taus) if taus else float("nan")
        print(f"  {model_name.upper()} ranking consistency (Kendall's tau): "
              f"mean={mean_tau:.4f} ± {np.std(taus):.4f} (n={len(taus)} pairs)")

    return results


def homophily_aware_attention(attn_dict, data):
    """
    Compute mean attention to same-class vs different-class neighbors.
    Ratio > 1 = homophily-aware attention learning.
    """
    same_class_weights = []
    diff_class_weights = []

    for node_id, nbr_weights in attn_dict.items():
        node_label = data.y[node_id].item()
        for nbr, w in nbr_weights.items():
            if nbr < data.num_nodes:
                nbr_label = data.y[nbr].item()
                if nbr_label == node_label:
                    same_class_weights.append(w)
                else:
                    diff_class_weights.append(w)

    mean_same = np.mean(same_class_weights) if same_class_weights else 0
    mean_diff = np.mean(diff_class_weights) if diff_class_weights else 0
    ratio = mean_same / (mean_diff + 1e-10)

    print(f"\n  Homophily-aware attention:")
    print(f"    Mean attn to same-class neighbors: {mean_same:.6f}")
    print(f"    Mean attn to diff-class neighbors:  {mean_diff:.6f}")
    print(f"    Ratio (>1 = homophily-aware): {ratio:.4f}")

    return ratio, mean_same, mean_diff
