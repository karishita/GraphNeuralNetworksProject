#!/usr/bin/env python3
"""
generate_plots.py — Generate all analysis plots from saved results.
Runs all 4 notebooks as scripts sequentially.

Run from parent of reddit_gnn/:
    python -m reddit_gnn.scripts.generate_plots
    python -m reddit_gnn.scripts.generate_plots --notebooks 01 03
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import RESULTS_ROOT

NOTEBOOKS = {
    "01": "Baseline Results — accuracy table, training curves, per-class F1",
    "02": "Ablation Analysis — all 24 studies, oversmoothing comparison",
    "03": "Embedding Visualization — t-SNE/UMAP (4 plot types × 6 models)",
    "04": "Efficiency Report — latency, VRAM, accuracy vs params",
}

NOTEBOOK_MODULES = {
    "01": "reddit_gnn.notebooks.01_baseline_results",
    "02": "reddit_gnn.notebooks.02_ablation_analysis",
    "03": "reddit_gnn.notebooks.03_visualisation",
    "04": "reddit_gnn.notebooks.04_efficiency_report",
}


def run_notebook_as_script(nb_id):
    """Execute a notebook module (the .py version embedded script logic)."""
    module = NOTEBOOK_MODULES[nb_id]
    print(f"\n{'='*60}")
    print(f"  Notebook {nb_id}: {NOTEBOOKS[nb_id]}")
    print(f"{'='*60}")

    # Use nbconvert if .ipynb exists, else fall back to direct import
    nb_path = os.path.join(
        os.path.dirname(__file__), "..", "notebooks", f"{nb_id}_{_nb_name(nb_id)}.ipynb"
    )
    if os.path.exists(nb_path):
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
               "--execute", "--inplace", nb_path]
        print(f"  Executing notebook: {nb_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  nbconvert failed:\n{result.stderr}")
            print("  Falling back to direct execution...")
            _run_direct(nb_id)
        else:
            print(f"  ✓ Notebook executed successfully")
    else:
        print(f"  Notebook .ipynb not found, running direct execution...")
        _run_direct(nb_id)


def _nb_name(nb_id):
    names = {
        "01": "baseline_results",
        "02": "ablation_analysis",
        "03": "visualisation",
        "04": "efficiency_report",
    }
    return names[nb_id]


def _run_direct(nb_id):
    """Execute embedded script logic directly (no Jupyter required)."""
    if nb_id == "01":
        from reddit_gnn.notebooks import _01_baseline_main as fn
        fn()
    elif nb_id == "02":
        from reddit_gnn.notebooks import _02_ablation_main as fn
        fn()
    elif nb_id == "03":
        from reddit_gnn.notebooks import _03_vis_main as fn
        fn()
    elif nb_id == "04":
        from reddit_gnn.notebooks import _04_efficiency_main as fn
        fn()


def main():
    parser = argparse.ArgumentParser(description="Generate all analysis plots")
    parser.add_argument("--notebooks", nargs="+", default=list(NOTEBOOKS.keys()),
                       choices=list(NOTEBOOKS.keys()),
                       help="Which notebooks to run (01 02 03 04)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Reddit GNN — Plot Generation")
    print(f"  Notebooks: {args.notebooks}")
    print(f"  Output: {RESULTS_ROOT}/figures/")
    print("=" * 60)

    for nb_id in args.notebooks:
        run_notebook_as_script(nb_id)

    print(f"\n{'='*60}")
    print("PLOT GENERATION COMPLETE")
    print(f"Figures saved to: {RESULTS_ROOT}/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
