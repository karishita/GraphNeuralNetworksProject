#!/usr/bin/env python3
"""
GraphSAGE Ablation Studies — A1 through A6.
Run: python -m reddit_gnn.ablations.run_sage_ablations [--ablation A1] [--seeds 0 1 2]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.models.graphsage import GraphSAGE
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_train_loader, get_val_loader
from reddit_gnn.training.train_neighbor import train_neighbor_sampled
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import get_test_predictions, compute_all_metrics, aggregate_seeds
from reddit_gnn.evaluation.serialize import save_run_results

# ─────────────────────────────────────────────────────────────────────────────
# Ablation Configs
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    # A1 — Aggregator Function (Xu et al. ICLR'19 expressivity hierarchy)
    "A1": {
        "name": "Aggregator Function",
        "param": "aggregator",
        "variants": {
            "A1-Mean": {"aggregator": "mean"},
            "A1-Max":  {"aggregator": "max"},
            "A1-LSTM": {"aggregator": "lstm"},
            "A1-Sum":  {"aggregator": "sum"},  # New — captures multiset cardinality
        },
    },
    # A2 — Number of Layers (Depth + Oversmoothing)
    "A2": {
        "name": "Number of Layers",
        "param": "layers",
        "variants": {
            "A2-1L": {"layers": 1, "num_neighbors": [25]},
            "A2-2L": {"layers": 2, "num_neighbors": [25, 10]},
            "A2-3L": {"layers": 3, "num_neighbors": [25, 10, 5]},
            "A2-4L": {"layers": 4, "num_neighbors": [25, 10, 5, 5]},
            "A2-5L": {"layers": 5, "num_neighbors": [25, 10, 5, 5, 5]},
        },
    },
    # A3 — Neighbor Sample Size
    "A3": {
        "name": "Neighbor Sample Size",
        "param": "num_neighbors",
        "variants": {
            "A3-[5,5]":   {"num_neighbors": [5, 5]},
            "A3-[10,5]":  {"num_neighbors": [10, 5]},
            "A3-[15,10]": {"num_neighbors": [15, 10]},
            "A3-[25,10]": {"num_neighbors": [25, 10]},
            "A3-[50,25]": {"num_neighbors": [50, 25]},
        },
    },
    # A4 — Graph Structure vs Features Only (Core hypothesis)
    "A4": {
        "name": "Graph Structure vs Features",
        "param": "structure",
        "variants": {
            "A4-Full":          {"mode": "full"},
            "A4-FeaturesOnly":  {"mode": "features_only"},
            "A4-StructureOnly": {"mode": "structure_only"},
        },
    },
    # A5 — Skip (Residual) Connections
    "A5": {
        "name": "Skip Connections",
        "param": "skip",
        "variants": {
            "A5-NoSkip":    {"skip": False},
            "A5-ResAdd":    {"skip": True, "skip_type": "add"},
            "A5-ResConcat": {"skip": True, "skip_type": "concat"},
        },
    },
    # A6 — Normalization Type
    "A6": {
        "name": "Normalization Type",
        "param": "norm",
        "variants": {
            "A6-BatchNorm": {"norm": "batchnorm"},
            "A6-LayerNorm": {"norm": "layernorm"},
            "A6-NoNorm":    {"norm": None},
        },
    },
}


def build_model(variant_overrides, hp):
    """Build a GraphSAGE model with ablation-specific overrides."""
    params = {**hp, **variant_overrides}
    model = GraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=params.get("hidden", 256),
        out_channels=NUM_CLASSES,
        num_layers=params.get("layers", 2),
        dropout=params.get("dropout", 0.5),
        aggregator=params.get("aggregator", "mean"),
        skip=params.get("skip", False),
        skip_type=params.get("skip_type", "add"),
        norm=params.get("norm", "batchnorm"),
    ).to(DEVICE)
    return model, params


def run_single_variant(data, ablation_id, variant_name, variant_overrides, seeds):
    """Train one variant across all seeds."""
    hp = DEFAULT_HPARAMS["graphsage"].copy()
    all_metrics = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'─'*60}")
        print(f"  {ablation_id} | {variant_name} | seed={seed}")
        print(f"{'─'*60}")

        model, params = build_model(variant_overrides, hp)
        print(f"  Params: {count_parameters(model):,}")

        # Handle A4 special modes
        train_data = data
        if variant_overrides.get("mode") == "features_only":
            # Remove edges → MLP equivalent
            import copy
            train_data = copy.copy(data)
            train_data.edge_index = torch.zeros(2, 0, dtype=torch.long)
        elif variant_overrides.get("mode") == "structure_only":
            # Replace features with random
            import copy
            train_data = copy.copy(data)
            train_data.x = torch.randn_like(data.x)

        num_neighbors = params.get("num_neighbors", [25, 10])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        train_loader = get_train_loader(train_data, num_neighbors, params["batch_size"])
        val_loader = get_val_loader(train_data, num_layers=params.get("layers", 2))

        history = train_neighbor_sampled(
            model, train_loader, val_loader, optimizer, DEVICE,
            max_epochs=params["max_epochs"], patience=params["patience"],
            model_name=f"SAGE-{variant_name}",
        )

        # Evaluate on ORIGINAL data (not modified)
        preds, labels = get_test_predictions(model, data, DEVICE)
        metrics = compute_all_metrics(preds, labels, "graphsage", f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)

        save_run_results(metrics, history, model_name="graphsage",
                        ablation_id=ablation_id, variant=variant_name, seed=seed)
        save_checkpoint(model, "graphsage", ablation_id=ablation_id,
                       variant=variant_name, seed=seed)

        print(f"  → acc={metrics['test_acc']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    agg = aggregate_seeds(all_metrics)
    print(f"\n  {variant_name} AGGREGATE: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return agg


def run_ablation(ablation_id, data, seeds):
    """Run all variants of one ablation study."""
    config = ABLATION_CONFIGS[ablation_id]
    print(f"\n{'='*60}")
    print(f"ABLATION {ablation_id}: {config['name']}")
    print(f"{'='*60}")

    results = {}
    for variant_name, overrides in config["variants"].items():
        agg = run_single_variant(data, ablation_id, variant_name, overrides, seeds)
        results[variant_name] = agg

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {ablation_id}: {config['name']}")
    print(f"{'='*60}")
    for vname, agg in results.items():
        print(f"  {vname:20s}: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE Ablation Studies (A1-A6)")
    parser.add_argument("--ablation", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                       help="Which ablations to run (e.g., A1 A2 A4)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                       help="Random seeds to use")
    args = parser.parse_args()

    print("Loading preprocessed data...")
    data, _, _ = load_normalized_data()

    for abl in args.ablation:
        if abl in ABLATION_CONFIGS:
            run_ablation(abl, data, args.seeds)
        else:
            print(f"Unknown ablation: {abl}. Available: {list(ABLATION_CONFIGS.keys())}")


if __name__ == "__main__":
    main()
