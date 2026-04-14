#!/usr/bin/env python3
"""
SGC Ablation Studies — C1 through C3.
Run: python -m reddit_gnn.ablations.run_sgc_ablations [--ablation C1] [--seeds 0 1 2]
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import DEVICE, SEEDS, NUM_CLASSES, NUM_FEATURES, DEFAULT_HPARAMS, SGC_DIR, set_seed
from reddit_gnn.data.normalize import load_normalized_data
from reddit_gnn.data.precompute_sgc import precompute_sgc_features
from reddit_gnn.training.train_sgc import train_sgc
from reddit_gnn.training.utils import count_parameters
from reddit_gnn.evaluation.metrics import compute_all_metrics, aggregate_seeds
from reddit_gnn.evaluation.serialize import save_run_results

ABLATION_CONFIGS = {
    # C1 — Propagation Hops K (Oversmoothing proof)
    "C1": {
        "name": "Propagation Hops K",
        "variants": {
            "C1-K1": {"K": 1},
            "C1-K2": {"K": 2},
            "C1-K3": {"K": 3},
            "C1-K4": {"K": 4},
            "C1-K5": {"K": 5},
        },
    },
    # C2 — Normalization Scheme
    "C2": {
        "name": "Normalization Scheme",
        "variants": {
            "C2-Symmetric":  {"norm_type": "symmetric"},
            "C2-RowNorm":    {"norm_type": "row"},
            "C2-NoSelfLoop": {"norm_type": "no_selfloop"},
        },
    },
    # C3 — SGC vs MLP vs GCN (Core structure hypothesis)
    "C3": {
        "name": "SGC vs MLP vs GCN",
        "variants": {
            "C3-SGC": {"model_type": "sgc"},
            "C3-MLP": {"model_type": "mlp"},
            "C3-GCN": {"model_type": "gcn"},
        },
    },
}


class MLP(nn.Module):
    """Simple 2-layer MLP baseline — no graph access."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def run_c2_variant(data, variant_name, norm_type, seeds):
    """C2: Recompute features with different normalization, then train SGC."""
    save_dir = os.path.join(SGC_DIR, f"norm_{norm_type}")
    precompute_sgc_features(data, max_K=2, save_dir=save_dir, norm_type=norm_type)

    all_metrics = []
    for seed in seeds:
        set_seed(seed)
        model, history = train_sgc(K=2, data=data, device=DEVICE, sgc_dir=save_dir)

        X_K = torch.load(os.path.join(save_dir, "reddit_sgc_K2.pt"), weights_only=False).to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(X_K[data.test_mask.to(DEVICE)])
            preds = out.argmax(1).cpu().numpy()
            labels = data.y[data.test_mask].numpy()

        metrics = compute_all_metrics(preds, labels, "sgc", f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)
        save_run_results(metrics, history, model_name="sgc", ablation_id="C2", variant=variant_name, seed=seed)
        print(f"  {variant_name} seed={seed}: acc={metrics['test_acc']:.4f}")

    return aggregate_seeds(all_metrics)


def run_c3_variant(data, variant_name, model_type, seeds):
    """C3: Compare SGC, MLP, and GCN on the same task."""
    all_metrics = []
    hp = DEFAULT_HPARAMS["sgc"]

    for seed in seeds:
        set_seed(seed)
        print(f"  {variant_name} seed={seed}...")

        if model_type == "sgc":
            model, history = train_sgc(K=2, data=data, device=DEVICE)
            X_K = torch.load(os.path.join(SGC_DIR, "reddit_sgc_K2.pt"), weights_only=False).to(DEVICE)
            model.eval()
            with torch.no_grad():
                preds = model(X_K[data.test_mask.to(DEVICE)]).argmax(1).cpu().numpy()

        elif model_type == "mlp":
            model = MLP(NUM_FEATURES, 256, NUM_CLASSES).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            x, y = data.x.to(DEVICE), data.y.to(DEVICE)
            history = []
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[data.train_mask]), y[data.train_mask])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(x[data.test_mask]).argmax(1).cpu().numpy()

        elif model_type == "gcn":
            from torch_geometric.nn import GCNConv
            class GCN2L(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(NUM_FEATURES, 256)
                    self.conv2 = GCNConv(256, NUM_CLASSES)
                def forward(self, x, edge_index):
                    x = F.relu(self.conv1(x, edge_index))
                    x = F.dropout(x, p=0.5, training=self.training)
                    return self.conv2(x, edge_index)

            model = GCN2L().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            data_dev = data.to(DEVICE)
            history = []
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                out = model(data_dev.x, data_dev.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data_dev.y[data.train_mask])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(data_dev.x, data_dev.edge_index)[data.test_mask].argmax(1).cpu().numpy()

        labels = data.y[data.test_mask].numpy()
        metrics = compute_all_metrics(preds, labels, model_type, f"{variant_name}_seed{seed}")
        all_metrics.append(metrics)
        save_run_results(metrics, history, model_name="sgc", ablation_id="C3", variant=variant_name, seed=seed)
        print(f"    → acc={metrics['test_acc']:.4f}")

    return aggregate_seeds(all_metrics)


def run_ablation(ablation_id, data, seeds):
    config = ABLATION_CONFIGS[ablation_id]
    print(f"\n{'='*60}")
    print(f"ABLATION {ablation_id}: {config['name']}")
    print(f"{'='*60}")

    results = {}
    for variant_name, overrides in config["variants"].items():
        if ablation_id == "C1":
            all_metrics = []
            for seed in seeds:
                set_seed(seed)
                K = overrides["K"]
                model, history = train_sgc(K=K, data=data, device=DEVICE)
                X_K = torch.load(os.path.join(SGC_DIR, f"reddit_sgc_K{K}.pt"), weights_only=False).to(DEVICE)
                model.eval()
                with torch.no_grad():
                    preds = model(X_K[data.test_mask.to(DEVICE)]).argmax(1).cpu().numpy()
                labels = data.y[data.test_mask].numpy()
                metrics = compute_all_metrics(preds, labels, "sgc", f"{variant_name}_seed{seed}")
                all_metrics.append(metrics)
                save_run_results(metrics, history, model_name="sgc", ablation_id="C1", variant=variant_name, seed=seed)
                print(f"  {variant_name} seed={seed}: acc={metrics['test_acc']:.4f}")
            agg = aggregate_seeds(all_metrics)
        elif ablation_id == "C2":
            agg = run_c2_variant(data, variant_name, overrides["norm_type"], seeds)
        elif ablation_id == "C3":
            agg = run_c3_variant(data, variant_name, overrides["model_type"], seeds)
        else:
            continue

        results[variant_name] = agg
        print(f"  {variant_name} AGGREGATE: acc={agg.get('test_acc_mean',0):.4f}")

    print(f"\nSUMMARY — {ablation_id}:")
    for vname, agg in results.items():
        print(f"  {vname:20s}: acc={agg.get('test_acc_mean',0):.4f} ± {agg.get('test_acc_std',0):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="SGC Ablation Studies (C1-C3)")
    parser.add_argument("--ablation", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    data, _, _ = load_normalized_data()
    for abl in args.ablation:
        if abl in ABLATION_CONFIGS:
            run_ablation(abl, data, args.seeds)


if __name__ == "__main__":
    main()
