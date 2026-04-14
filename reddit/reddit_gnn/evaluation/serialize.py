"""
Result Serialization — Save all experiment outputs in a structured format.
"""

import json
import csv
import os
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import RESULTS_ROOT


def save_run_results(
    metrics,
    history,
    embeddings=None,
    structural_grid=None,
    oversmoothing_stats=None,
    model_name="model",
    ablation_id="baseline",
    variant="default",
    seed=0,
    save_root=None,
):
    """
    Save all outputs for a single training run.

    Directory structure:
        results/{model_name}/{ablation_id}/{variant}/seed{seed}/
            ├── metrics.json
            ├── history.csv
            ├── embeddings.npy (optional)
            ├── structural_grid.npy (optional)
            └── oversmoothing.json (optional)
    """
    if save_root is None:
        save_root = RESULTS_ROOT

    run_dir = os.path.join(
        save_root, model_name, ablation_id, variant, f"seed{seed}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Metrics JSON
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Training history CSV
    if history:
        history_path = os.path.join(run_dir, "history.csv")
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # Embeddings (numpy)
    if embeddings is not None:
        emb_path = os.path.join(run_dir, "embeddings.npy")
        np.save(emb_path, embeddings)

    # Structural grid (3×4 heatmap)
    if structural_grid is not None:
        grid_path = os.path.join(run_dir, "structural_grid.npy")
        np.save(grid_path, structural_grid)

    # Oversmoothing stats
    if oversmoothing_stats is not None:
        os_path = os.path.join(run_dir, "oversmoothing.json")
        with open(os_path, "w") as f:
            json.dump(oversmoothing_stats, f, indent=2)

    print(f"  Results saved: {run_dir}")
    return run_dir


def load_run_results(model_name, ablation_id="baseline", variant="default",
                     seed=0, save_root=None):
    """Load results for a single training run."""
    if save_root is None:
        save_root = RESULTS_ROOT

    run_dir = os.path.join(save_root, model_name, ablation_id, variant, f"seed{seed}")

    results = {}

    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)

    history_path = os.path.join(run_dir, "history.csv")
    if os.path.exists(history_path):
        with open(history_path) as f:
            reader = csv.DictReader(f)
            results["history"] = list(reader)

    emb_path = os.path.join(run_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        results["embeddings"] = np.load(emb_path)

    grid_path = os.path.join(run_dir, "structural_grid.npy")
    if os.path.exists(grid_path):
        results["structural_grid"] = np.load(grid_path)

    os_path = os.path.join(run_dir, "oversmoothing.json")
    if os.path.exists(os_path):
        with open(os_path) as f:
            results["oversmoothing"] = json.load(f)

    return results
