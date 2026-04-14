#!/usr/bin/env python3
"""
GATv2 Ablation Studies — E1 through E3.
Run: python -m reddit_gnn.ablations.run_gatv2_ablations [--ablation E1] [--seeds 0 1 2]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.models.gatv2 import GATv2
from reddit_gnn.models.gat import GAT
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_train_loader, get_val_loader
from reddit_gnn.training.train_neighbor import train_neighbor_sampled
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import get_test_predictions, compute_all_metrics, aggregate_seeds
from reddit_gnn.evaluation.serialize import save_run_results

ABLATION_CONFIGS = {
    # E1 — Dynamic vs Static Attention
    "E1": {
        "name": "Dynamic vs Static Attention",
        "variants": {
            "E1-GATv2": {"model_class": "gatv2"},
            "E1-GAT":   {"model_class": "gat"},
        },
    },
    # E2 — Shared vs Separate Weight Matrices
    "E2": {
        "name": "Shared vs Separate Weights",
        "variants": {
            "E2-Shared":   {"share_weights": True},
            "E2-Separate": {"share_weights": False},
        },
    },
    # E3 — GATv2 Depth vs GAT Depth (Oversmoothing tolerance)
    "E3": {
        "name": "Depth: GATv2 vs GAT",
        "variants": {
            "E3-GATv2-2L": {"model_class": "gatv2", "layers": 2, "num_neighbors": [25, 10]},
            "E3-GATv2-3L": {"model_class": "gatv2", "layers": 3, "num_neighbors": [25, 10, 5]},
            "E3-GATv2-4L": {"model_class": "gatv2", "layers": 4, "num_neighbors": [25, 10, 5, 5]},
            "E3-GAT-2L":   {"model_class": "gat",   "layers": 2, "num_neighbors": [25, 10]},
            "E3-GAT-3L":   {"model_class": "gat",   "layers": 3, "num_neighbors": [25, 10, 5]},
            "E3-GAT-4L":   {"model_class": "gat",   "layers": 4, "num_neighbors": [25, 10, 5, 5]},
        },
    },
}


def build_model(variant_overrides, hp):
    model_class = variant_overrides.get("model_class", "gatv2")
    layers = variant_overrides.get("layers", hp["layers"])
    share_weights = variant_overrides.get("share_weights", hp.get("share_weights", True))

    if model_class == "gatv2":
        model = GATv2(
            in_channels=NUM_FEATURES, out_channels=NUM_CLASSES,
            hidden_per_head=hp["hidden_per_head"], num_heads=hp["heads"],
            num_layers=layers, attn_dropout=hp["attn_dropout"],
            feat_dropout=hp["feat_dropout"], share_weights=share_weights,
        )
    else:
        model = GAT(
            in_channels=NUM_FEATURES, out_channels=NUM_CLASSES,
            hidden_per_head=hp["hidden_per_head"], num_heads=hp["heads"],
            num_layers=layers, attn_dropout=hp["attn_dropout"],
            feat_dropout=hp["feat_dropout"],
        )
    return model.to(DEVICE)


def run_single_variant(data, ablation_id, variant_name, variant_overrides, seeds):
    hp = DEFAULT_HPARAMS["gatv2"].copy()
    all_metrics = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'─'*60}")
        print(f"  {ablation_id} | {variant_name} | seed={seed}")
        print(f"{'─'*60}")

        model = build_model(variant_overrides, hp)
        print(f"  Params: {count_parameters(model):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        num_neighbors = variant_overrides.get("num_neighbors", hp["num_neighbors"])
        num_layers = variant_overrides.get("layers", hp["layers"])
        train_loader = get_train_loader(data, num_neighbors, hp["batch_size"])
        val_loader = get_val_loader(data, num_layers=num_layers)

        history = train_neighbor_sampled(
            model, train_loader, val_loader, optimizer, DEVICE,
            max_epochs=hp["max_epochs"], patience=hp["patience"],
            model_name=f"GATv2-{variant_name}",
        )

        preds, labels = get_test_predictions(model, data, DEVICE)
        metrics = compute_all_metrics(preds, labels, "gatv2", f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)

        save_run_results(metrics, history, model_name="gatv2",
                        ablation_id=ablation_id, variant=variant_name, seed=seed)

        # For E1/E3: also run attention analysis
        if ablation_id in ("E1", "E3"):
            from reddit_gnn.analysis.oversmoothing import compute_embedding_variance_per_layer, oversmoothing_summary
            try:
                stats = compute_embedding_variance_per_layer(model, data, DEVICE)
                oversmoothing_summary(stats, variant_name)
            except Exception as e:
                print(f"  Oversmoothing analysis skipped: {e}")

        print(f"  → acc={metrics['test_acc']:.4f}")

    agg = aggregate_seeds(all_metrics)
    print(f"\n  {variant_name} AGGREGATE: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return agg


def run_ablation(ablation_id, data, seeds):
    config = ABLATION_CONFIGS[ablation_id]
    print(f"\n{'='*60}")
    print(f"ABLATION {ablation_id}: {config['name']}")
    print(f"{'='*60}")

    results = {}
    for vname, overrides in config["variants"].items():
        agg = run_single_variant(data, ablation_id, vname, overrides, seeds)
        results[vname] = agg

    print(f"\nSUMMARY — {ablation_id}:")
    for vname, agg in results.items():
        print(f"  {vname:20s}: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="GATv2 Ablation Studies (E1-E3)")
    parser.add_argument("--ablation", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    data, _, _ = load_normalized_data()
    for abl in args.ablation:
        if abl in ABLATION_CONFIGS:
            run_ablation(abl, data, args.seeds)


if __name__ == "__main__":
    main()
