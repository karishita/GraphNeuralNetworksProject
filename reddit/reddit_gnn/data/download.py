"""
Step 1A — Reddit Dataset Download and Verification.
Downloads via PyG, verifies expected statistics.
"""

import torch
from torch_geometric.datasets import Reddit
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import REDDIT_RAW, EXPECTED_NODES, EXPECTED_EDGES, NUM_FEATURES, NUM_CLASSES, EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST


def download_reddit(root: str = REDDIT_RAW) -> "torch_geometric.data.Data":
    """Download Reddit dataset via PyG and verify statistics."""
    print(f"[1A] Downloading Reddit dataset to {root}...")
    dataset = Reddit(root=root)
    data = dataset[0]

    # ── Verify dataset properties ──
    checks = [
        ("Nodes", data.num_nodes, EXPECTED_NODES),
        ("Edges", data.num_edges, EXPECTED_EDGES),
        ("Node features", data.num_node_features, NUM_FEATURES),
        ("Classes", dataset.num_classes, NUM_CLASSES),
        ("Train nodes", data.train_mask.sum().item(), EXPECTED_TRAIN),
        ("Val nodes", data.val_mask.sum().item(), EXPECTED_VAL),
        ("Test nodes", data.test_mask.sum().item(), EXPECTED_TEST),
    ]

    all_pass = True
    for name, actual, expected in checks:
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} {name}: {actual:,} (expected {expected:,})")

    if not all_pass:
        print("\n⚠️  WARNING: Some dataset statistics don't match expected values!")
    else:
        print("\n✓ All dataset statistics verified successfully.")

    return data, dataset


if __name__ == "__main__":
    data, dataset = download_reddit()
    print(f"\nDataset loaded: {data}")
