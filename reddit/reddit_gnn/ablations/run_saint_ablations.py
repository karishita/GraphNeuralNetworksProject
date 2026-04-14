#!/usr/bin/env python3
"""
GraphSAINT Ablation Studies — B1 through B4.
Run: python -m reddit_gnn.ablations.run_saint_ablations [--ablation B1] [--seeds 0 1 2]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, set_seed
from reddit_gnn.models.graphsaint import GraphSAINTNet
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.loaders import get_saint_loader
from reddit_gnn.training.train_saint import train_saint
from reddit_gnn.training.utils import save_checkpoint, count_parameters
from reddit_gnn.evaluation.metrics import get_test_predictions, compute_all_metrics, aggregate_seeds
from reddit_gnn.evaluation.serialize import save_run_results

ABLATION_CONFIGS = {
    # B1 — Sampler Type
    "B1": {
        "name": "Sampler Type",
        "variants": {
            "B1-Node": {"sampler": "node"},
            "B1-Edge": {"sampler": "edge"},
            "B1-RW":   {"sampler": "rw"},
        },
    },
    # B2 — Normalization Correction
    "B2": {
        "name": "Normalization Correction",
        "variants": {
            "B2-Norm":   {"use_norm": True},
            "B2-NoNorm": {"use_norm": False},
        },
    },
    # B3 — Subgraph Budget
    "B3": {
        "name": "Subgraph Budget",
        "variants": {
            "B3-2000":  {"budget": 2000},
            "B3-4000":  {"budget": 4000},
            "B3-6000":  {"budget": 6000},
            "B3-10000": {"budget": 10000},
            "B3-15000": {"budget": 15000},
        },
    },
    # B4 — Random Walk Parameters
    "B4": {
        "name": "Random Walk Parameters",
        "variants": {
            "B4-short":  {"walk_length": 2, "num_steps": 200},
            "B4-medium": {"walk_length": 4, "num_steps": 100},
            "B4-long":   {"walk_length": 8, "num_steps": 50},
        },
    },
}


def run_single_variant(data, ablation_id, variant_name, variant_overrides, seeds):
    hp = DEFAULT_HPARAMS["graphsaint"].copy()
    all_metrics = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'─'*60}")
        print(f"  {ablation_id} | {variant_name} | seed={seed}")
        print(f"{'─'*60}")

        model = GraphSAINTNet(
            in_channels=NUM_FEATURES,
            hidden_channels=hp["hidden"],
            out_channels=NUM_CLASSES,
            num_layers=hp["layers"],
            dropout=hp["dropout"],
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        # Build loader with overridden params
        sampler_type = variant_overrides.get("sampler", hp["sampler"])
        budget = variant_overrides.get("budget", hp["budget"])
        walk_length = variant_overrides.get("walk_length", hp["walk_length"])
        num_steps = variant_overrides.get("num_steps", hp["num_steps"])

        saint_loader = get_saint_loader(
            data, sampler_type=sampler_type, budget=budget,
            walk_length=walk_length, num_steps=num_steps,
            sample_coverage=hp["sample_coverage"],
        )

        use_norm = variant_overrides.get("use_norm", True)

        history = train_saint(
            model, saint_loader, data, optimizer, DEVICE,
            max_epochs=hp["max_epochs"], patience=hp["patience"],
            use_norm=use_norm,
            model_name=f"SAINT-{variant_name}",
        )

        preds, labels = get_test_predictions(model, data, DEVICE)
        metrics = compute_all_metrics(preds, labels, "graphsaint", f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)

        save_run_results(metrics, history, model_name="graphsaint",
                        ablation_id=ablation_id, variant=variant_name, seed=seed)
        print(f"  → acc={metrics['test_acc']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

    agg = aggregate_seeds(all_metrics)
    print(f"\n  {variant_name} AGGREGATE: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return agg


def run_ablation(ablation_id, data, seeds):
    config = ABLATION_CONFIGS[ablation_id]
    print(f"\n{'='*60}")
    print(f"ABLATION {ablation_id}: {config['name']}")
    print(f"{'='*60}")

    results = {}
    for variant_name, overrides in config["variants"].items():
        agg = run_single_variant(data, ablation_id, variant_name, overrides, seeds)
        results[variant_name] = agg

    print(f"\n{'='*60}")
    print(f"SUMMARY — {ablation_id}: {config['name']}")
    print(f"{'='*60}")
    for vname, agg in results.items():
        print(f"  {vname:20s}: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="GraphSAINT Ablation Studies (B1-B4)")
    parser.add_argument("--ablation", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    data, _, _ = load_normalized_data()
    for abl in args.ablation:
        if abl in ABLATION_CONFIGS:
            run_ablation(abl, data, args.seeds)


if __name__ == "__main__":
    main()
