#!/usr/bin/env python3
"""
GAT Ablation Studies — D1 through D4.
Run: python -m reddit_gnn.ablations.run_gat_ablations [--ablation D1] [--seeds 0 1 2]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.models.gat import GAT
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_train_loader, get_val_loader
from reddit_gnn.training.train_neighbor import train_neighbor_sampled
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import get_test_predictions, compute_all_metrics, aggregate_seeds
from reddit_gnn.evaluation.serialize import save_run_results

ABLATION_CONFIGS = {
    # D1 — Number of Attention Heads (constant total hidden dim = 256)
    "D1": {
        "name": "Number of Attention Heads",
        "variants": {
            "D1-1h":  {"heads": 1,  "hidden_per_head": 256},
            "D1-2h":  {"heads": 2,  "hidden_per_head": 128},
            "D1-4h":  {"heads": 4,  "hidden_per_head": 64},
            "D1-8h":  {"heads": 8,  "hidden_per_head": 32},
            "D1-16h": {"heads": 16, "hidden_per_head": 16},
        },
    },
    # D2 — Attention Dropout Rate
    "D2": {
        "name": "Attention Dropout Rate",
        "variants": {
            "D2-0.0": {"attn_dropout": 0.0},
            "D2-0.1": {"attn_dropout": 0.1},
            "D2-0.2": {"attn_dropout": 0.2},
            "D2-0.3": {"attn_dropout": 0.3},
            "D2-0.6": {"attn_dropout": 0.6},
        },
    },
    # D3 — Layer Depth + GAT-Specific Oversmoothing
    "D3": {
        "name": "Layer Depth (Oversmoothing)",
        "variants": {
            "D3-1L": {"layers": 1, "num_neighbors": [25]},
            "D3-2L": {"layers": 2, "num_neighbors": [25, 10]},
            "D3-3L": {"layers": 3, "num_neighbors": [25, 10, 5]},
            "D3-4L": {"layers": 4, "num_neighbors": [25, 10, 5, 5]},
        },
    },
    # D4 — Static Attention Analysis (post-training analysis, not a new training run)
    "D4": {
        "name": "Static Attention Analysis",
        "variants": {
            "D4-analysis": {"analysis_only": True},
        },
    },
}


def run_single_variant(data, ablation_id, variant_name, variant_overrides, seeds):
    hp = DEFAULT_HPARAMS["gat"].copy()
    all_metrics = []

    # D4 is analysis-only — uses trained baseline model
    if variant_overrides.get("analysis_only"):
        return run_d4_analysis(data, seeds)

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'─'*60}")
        print(f"  {ablation_id} | {variant_name} | seed={seed}")
        print(f"{'─'*60}")

        params = {**hp, **variant_overrides}
        model = GAT(
            in_channels=NUM_FEATURES,
            out_channels=NUM_CLASSES,
            hidden_per_head=params["hidden_per_head"],
            num_heads=params["heads"],
            num_layers=params.get("layers", 2),
            attn_dropout=params["attn_dropout"],
            feat_dropout=params["feat_dropout"],
        ).to(DEVICE)

        print(f"  Params: {count_parameters(model):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        num_neighbors = params.get("num_neighbors", [25, 10])
        train_loader = get_train_loader(data, num_neighbors, params["batch_size"])
        val_loader = get_val_loader(data, num_layers=params.get("layers", 2))

        history = train_neighbor_sampled(
            model, train_loader, val_loader, optimizer, DEVICE,
            max_epochs=params["max_epochs"], patience=params["patience"],
            model_name=f"GAT-{variant_name}",
        )

        preds, labels = get_test_predictions(model, data, DEVICE)
        metrics = compute_all_metrics(preds, labels, "gat", f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)

        save_run_results(metrics, history, model_name="gat",
                        ablation_id=ablation_id, variant=variant_name, seed=seed)
        print(f"  → acc={metrics['test_acc']:.4f}")

    agg = aggregate_seeds(all_metrics)
    print(f"\n  {variant_name} AGGREGATE: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return agg


def run_d4_analysis(data, seeds):
    """D4: Static attention analysis on trained GAT baseline."""
    from reddit_gnn.analysis.attention_analysis import (
        extract_attention_weights, compute_attention_entropy,
        hub_concentration_test, homophily_aware_attention,
    )
    from reddit_gnn.config import CHECKPOINTS_DIR

    print(f"\n{'='*60}")
    print("D4: Static Attention Analysis (post-training)")
    print(f"{'='*60}")

    # Load best baseline checkpoint
    hp = DEFAULT_HPARAMS["gat"]
    model = GAT(
        in_channels=NUM_FEATURES, out_channels=NUM_CLASSES,
        hidden_per_head=hp["hidden_per_head"], num_heads=hp["heads"],
    ).to(DEVICE)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "gat", "baseline", "default", "seed0", "best_model.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  ⚠️  No baseline checkpoint found at {ckpt_path}")
        print("  Run baseline training first: python -m reddit_gnn.scripts.run_all_baselines")
        return {}

    # Sample 200 nodes: mix of high/low degree, different classes
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    test_idx = torch.where(data.test_mask)[0]
    top_deg = test_idx[deg[test_idx].argsort(descending=True)[:100]].tolist()
    low_deg = test_idx[deg[test_idx].argsort()[:100]].tolist()
    sample_ids = top_deg + low_deg

    # Extract attention weights
    attn_dict = extract_attention_weights(model, data, DEVICE, sample_ids)
    entropies = compute_attention_entropy(attn_dict)
    hub_concentration_test(attn_dict, top_k=10)
    homophily_aware_attention(attn_dict, data)

    # Report entropy stats
    import numpy as np
    norm_ents = [v["normalized_entropy"] for v in entropies.values()]
    print(f"\n  Attention entropy (normalized): mean={np.mean(norm_ents):.4f}, std={np.std(norm_ents):.4f}")
    print(f"  High entropy (>0.9): {sum(1 for e in norm_ents if e > 0.9)} / {len(norm_ents)} nodes (near-uniform = GCN-like)")

    return {"entropy_mean": float(np.mean(norm_ents)), "entropy_std": float(np.std(norm_ents))}


def run_ablation(ablation_id, data, seeds):
    config = ABLATION_CONFIGS[ablation_id]
    print(f"\n{'='*60}")
    print(f"ABLATION {ablation_id}: {config['name']}")
    print(f"{'='*60}")

    results = {}
    for vname, overrides in config["variants"].items():
        agg = run_single_variant(data, ablation_id, vname, overrides, seeds)
        results[vname] = agg
    return results


def main():
    parser = argparse.ArgumentParser(description="GAT Ablation Studies (D1-D4)")
    parser.add_argument("--ablation", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    data, _, _ = load_normalized_data()
    for abl in args.ablation:
        if abl in ABLATION_CONFIGS:
            run_ablation(abl, data, args.seeds)


if __name__ == "__main__":
    main()
