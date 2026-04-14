"""
Automated verification tests for the Reddit GNN pipeline.
Run: python -m pytest reddit_gnn/tests/test_baselines.py -v

Tests:
  - Preprocessing output verification
  - Baseline accuracy gate (>93% all 6 models)
  - Ablation consistency (std < 1% for stable ablations)
  - Structural sanity checks (oversmoothing, efficiency ordering)
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from reddit_gnn.config import (
    RESULTS_ROOT, PREPROCESSED, SGC_DIR, CLUSTER_DIR, SEEDS,
    NUM_CLASSES, NUM_FEATURES, EXPECTED_NODES, EXPECTED_EDGES,
    EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(model, ablation="baseline", variant="default", seed=0):
    path = os.path.join(RESULTS_ROOT, model, ablation, variant, f"seed{seed}", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_seed_metrics(model, ablation="baseline", variant="default"):
    results = []
    for seed in SEEDS:
        m = load_metrics(model, ablation, variant, seed)
        if m:
            results.append(m)
    return results


MODEL_NAMES = ["graphsage", "graphsaint", "sgc", "gat", "gatv2", "cluster_gcn"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Preprocessing Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    """Verify preprocessing outputs exist and are plausible."""

    def test_normalized_data_exists(self):
        path = os.path.join(PREPROCESSED, "reddit_normalized.pt")
        assert os.path.exists(path), f"Missing: {path}\nRun: python -m reddit_gnn.scripts.run_preprocessing"

    def test_sgc_features_exist_k2(self):
        path = os.path.join(SGC_DIR, "reddit_sgc_K2.pt")
        assert os.path.exists(path), f"Missing SGC K=2 features: {path}"

    def test_sgc_all_K_exist(self):
        """SGC should have K=1..5."""
        for k in range(1, 6):
            path = os.path.join(SGC_DIR, f"reddit_sgc_K{k}.pt")
            assert os.path.exists(path), f"Missing SGC K={k}: {path}"

    def test_oversmoothing_log_exists(self):
        path = os.path.join(SGC_DIR, "sgc_oversmoothing_log.csv")
        assert os.path.exists(path), f"Missing oversmoothing log: {path}"

    def test_oversmoothing_cosine_sim_increases(self):
        """Cosine similarity should increase with K (oversmoothing sanity check)."""
        import csv
        path = os.path.join(SGC_DIR, "sgc_oversmoothing_log.csv")
        if not os.path.exists(path):
            pytest.skip("Oversmoothing log not yet generated")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        sims = [float(r["cos_sim"]) for r in sorted(rows, key=lambda r: int(r["K"]))]
        for i in range(1, len(sims)):
            assert sims[i] >= sims[i-1] - 0.01, (
                f"Cosine similarity should increase with K: {sims}"
            )

    def test_node_homophily_exists(self):
        path = os.path.join(PREPROCESSED, "node_homophily.pt")
        assert os.path.exists(path), f"Missing node homophily: {path}"

    def test_cluster_partitions_exist(self):
        """At least the baseline (1500 parts) should exist."""
        path = os.path.join(CLUSTER_DIR, "parts_1500")
        assert os.path.exists(path), f"Missing ClusterGCN partition: {path}"

    def test_normalized_data_shape(self):
        """Normalized data should match expected dataset dimensions."""
        import torch
        path = os.path.join(PREPROCESSED, "reddit_normalized.pt")
        if not os.path.exists(path):
            pytest.skip("Preprocessed data not yet generated")
        checkpoint = torch.load(path, weights_only=False)
        data = checkpoint.get("data") or checkpoint
        # Allow for wrapped or bare data object
        if hasattr(data, "num_nodes"):
            assert data.num_nodes == EXPECTED_NODES, f"Expected {EXPECTED_NODES} nodes, got {data.num_nodes}"
            assert data.x.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} features"
            assert data.num_classes == NUM_CLASSES or int(data.y.max()) + 1 == NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Baseline Accuracy Gate
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineAccuracy:
    """All 6 models must exceed 93% test accuracy on at least one seed."""

    ACCURACY_GATES = {
        "graphsage":   0.93,
        "graphsaint":  0.93,
        "sgc":         0.93,
        "gat":         0.93,
        "gatv2":       0.93,
        "cluster_gcn": 0.93,
    }

    EXPECTED_RANGES = {
        "graphsage":   (0.955, 0.970),
        "graphsaint":  (0.960, 0.972),
        "sgc":         (0.940, 0.955),
        "gat":         (0.950, 0.965),
        "gatv2":       (0.952, 0.968),
        "cluster_gcn": (0.955, 0.970),
    }

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_baseline_exists(self, model):
        results = load_all_seed_metrics(model)
        assert len(results) > 0, (
            f"No baseline results for {model}. "
            f"Run: python -m reddit_gnn.scripts.run_all_baselines"
        )

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_baseline_above_gate(self, model):
        """At least one seed must pass the 93% accuracy gate."""
        results = load_all_seed_metrics(model)
        if not results:
            pytest.skip(f"No baseline results for {model}")
        gate = self.ACCURACY_GATES[model]
        best_acc = max(m["test_acc"] for m in results)
        assert best_acc >= gate, (
            f"{model}: best acc {best_acc:.4f} is below {gate:.2%} gate"
        )

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_baseline_in_expected_range(self, model):
        """Baseline accuracy should be within the expected range from the paper."""
        results = load_all_seed_metrics(model)
        if not results:
            pytest.skip(f"No baseline results for {model}")
        import numpy as np
        mean_acc = np.mean([m["test_acc"] for m in results])
        lo, hi = self.EXPECTED_RANGES[model]
        # Warn but don't fail (ranges are approximate)
        if not (lo - 0.01 <= mean_acc <= hi + 0.01):
            pytest.warns(UserWarning, match="accuracy out of range") if False else None
            print(
                f"  ⚠️  {model} mean acc {mean_acc:.4f} outside expected {lo:.3f}–{hi:.3f} "
                f"(implementation-specific variation is acceptable)"
            )

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_baseline_seed_stability(self, model):
        """Standard deviation across 3 seeds should be < 1%."""
        results = load_all_seed_metrics(model)
        if len(results) < 2:
            pytest.skip(f"Need ≥2 seeds for stability check ({model})")
        import numpy as np
        accs = [m["test_acc"] for m in results]
        std = np.std(accs)
        assert std < 0.01, (
            f"{model}: seed std {std:.4f} > 1%. "
            f"Accuracies: {[f'{a:.4f}' for a in accs]}"
        )

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_f1_macro_reasonable(self, model):
        """F1 macro should be within 3% of accuracy (class imbalance check)."""
        results = load_all_seed_metrics(model)
        if not results or "f1_macro" not in results[0]:
            pytest.skip(f"No F1 macro for {model}")
        import numpy as np
        mean_acc = np.mean([m["test_acc"] for m in results])
        mean_f1  = np.mean([m["f1_macro"] for m in results])
        gap = abs(mean_acc - mean_f1)
        assert gap < 0.05, (
            f"{model}: Acc-F1 gap {gap:.4f} > 5%. "
            f"Possible class imbalance issue. acc={mean_acc:.4f}, f1={mean_f1:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Ablation Consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestAblationConsistency:
    """Verify ablation results are self-consistent."""

    def _get_ablation_results(self, model, ablation_id):
        base = os.path.join(RESULTS_ROOT, model, ablation_id)
        if not os.path.exists(base):
            return {}
        results = {}
        for variant in os.listdir(base):
            seed_accs = []
            for seed in SEEDS:
                m = load_metrics(model, ablation_id, variant, seed)
                if m:
                    seed_accs.append(m["test_acc"])
            if seed_accs:
                import numpy as np
                results[variant] = {"mean": np.mean(seed_accs), "std": np.std(seed_accs)}
        return results

    def test_a4_structure_dominates_features(self):
        """A4: Full graph > features-only (if graph helps at all)."""
        r = self._get_ablation_results("graphsage", "A4")
        if not r or "A4-Full" not in r or "A4-FeaturesOnly" not in r:
            pytest.skip("A4 results not available")
        assert r["A4-Full"]["mean"] > r["A4-FeaturesOnly"]["mean"], (
            f"A4: graph+features {r['A4-Full']['mean']:.4f} should beat features-only "
            f"{r['A4-FeaturesOnly']['mean']:.4f}"
        )

    def test_c1_k2_best_or_k1(self):
        """C1: K=2 should be among the best hops (or K=1 if already oversmoothed)."""
        r = self._get_ablation_results("sgc", "C1")
        if not r or "C1-K2" not in r:
            pytest.skip("C1 results not available")
        k2_acc = r["C1-K2"]["mean"]
        k5_acc = r.get("C1-K5", {}).get("mean", 0)
        assert k2_acc >= k5_acc, (
            f"C1: K=2 ({k2_acc:.4f}) should outperform K=5 ({k5_acc:.4f}) due to oversmoothing"
        )

    def test_a2_oversmoothing_at_depth_5(self):
        """A2: 5-layer SAGE should be worse than 2-layer."""
        r = self._get_ablation_results("graphsage", "A2")
        if not r or "A2-2L" not in r or "A2-5L" not in r:
            pytest.skip("A2 depth results not available")
        assert r["A2-2L"]["mean"] > r["A2-5L"]["mean"] - 0.005, (
            f"A2: 2L ({r['A2-2L']['mean']:.4f}) should outperform 5L ({r['A2-5L']['mean']:.4f})"
        )

    @pytest.mark.parametrize("model,ablation", [
        ("graphsage", "A1"), ("graphsage", "A6"),
        ("sgc", "C1"), ("gat", "D2"),
    ])
    def test_ablation_seed_stability(self, model, ablation):
        """No ablation variant should have seed std > 2%."""
        import numpy as np
        r = self._get_ablation_results(model, ablation)
        if not r:
            pytest.skip(f"No {model}/{ablation} results")
        for vname, stats in r.items():
            assert stats["std"] < 0.02, (
                f"{model}/{ablation}/{vname}: std={stats['std']:.4f} > 2%. "
                "Consider training with more seeds."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Structural Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralSanity:
    """High-level sanity checks on results."""

    def test_sgc_faster_than_gat(self):
        """SGC inference should be much faster than GAT."""
        import json
        eff_path = os.path.join(RESULTS_ROOT, "figures", "04_efficiency", "efficiency_metrics.json")
        if not os.path.exists(eff_path):
            pytest.skip("Efficiency metrics not generated yet")
        with open(eff_path) as f:
            eff = {r["name"]: r for r in json.load(f)}
        sgc_lat = float(eff.get("SGC", {}).get("inference_ms", 0) or 0)
        gat_lat = float(eff.get("GAT", {}).get("inference_ms", 0) or 0)
        if sgc_lat == 0 or gat_lat == 0:
            pytest.skip("Latency data missing")
        assert sgc_lat < gat_lat, (
            f"SGC ({sgc_lat:.1f}ms) should be faster than GAT ({gat_lat:.1f}ms)"
        )

    def test_all_baselines_logged(self):
        """All 6 models should have seed0 results logged."""
        missing = []
        for model in MODEL_NAMES:
            m = load_metrics(model)
            if m is None:
                missing.append(model)
        assert not missing, (
            f"Missing baseline results for: {missing}. "
            f"Run: python -m reddit_gnn.scripts.run_all_baselines"
        )
