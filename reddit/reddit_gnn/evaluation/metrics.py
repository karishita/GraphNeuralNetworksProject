"""
Evaluation Metrics — Accuracy, F1, classification report, aggregation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def get_test_predictions(model, data, device, model_type="default"):
    """Get predictions on the full test set."""
    model.eval()

    if model_type == "sgc":
        # SGC: direct forward on precomputed features
        out = model(data.x.to(device))
    else:
        data_dev = data.to(device)
        out = model(data_dev.x, data_dev.edge_index)

    test_mask = data.test_mask
    preds = out[test_mask].argmax(dim=1).cpu().numpy()
    labels = data.y[test_mask].cpu().numpy()
    return preds, labels


def compute_all_metrics(preds, labels, model_name="", run_id=""):
    """Compute comprehensive evaluation metrics."""
    metrics = {
        "model": model_name,
        "run_id": run_id,
        # Primary metric
        "test_acc": accuracy_score(labels, preds),
        # Class-averaged metrics
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_micro": f1_score(labels, preds, average="micro"),
        # Per-class F1 (41 values)
        "f1_per_class": f1_score(labels, preds, average=None).tolist(),
    }
    return metrics


def print_classification_report(preds, labels, model_name=""):
    """Print sklearn classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report: {model_name}")
    print("=" * 60)
    print(classification_report(labels, preds, digits=4))


def compute_confusion_matrix(preds, labels):
    """Compute confusion matrix."""
    return confusion_matrix(labels, preds)


def aggregate_seeds(metrics_list):
    """
    Aggregate metrics across multiple seeds.
    Returns mean ± std for each numeric metric.
    """
    if not metrics_list:
        return {}

    aggregated = {}
    numeric_keys = ["test_acc", "f1_macro", "f1_weighted", "f1_micro"]

    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    # Per-class F1 aggregation
    if "f1_per_class" in metrics_list[0]:
        per_class = np.array([m["f1_per_class"] for m in metrics_list])
        aggregated["f1_per_class_mean"] = per_class.mean(axis=0).tolist()
        aggregated["f1_per_class_std"] = per_class.std(axis=0).tolist()

    return aggregated
