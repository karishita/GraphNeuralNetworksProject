"""
Efficiency Metrics — Timing, VRAM, throughput, parameter count.
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.training.utils import count_parameters, reset_gpu_memory


def measure_epoch_time(model, loader, optimizer, loss_fn, device, n_runs=5,
                       model_type="neighbor"):
    """
    Measure wall-clock time for one full training epoch.
    Returns mean and std across n_runs.
    """
    import torch.nn.functional as F

    times = []
    for run in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if model_type == "sgc":
                out = model(batch.x)
                loss = F.cross_entropy(out, batch.y)
            else:
                out = model(batch.x, batch.edge_index)
                if hasattr(batch, "batch_size"):
                    loss = F.cross_entropy(out[:batch.batch_size],
                                          batch.y[:batch.batch_size])
                elif hasattr(batch, "train_mask"):
                    loss = F.cross_entropy(out[batch.train_mask],
                                          batch.y[batch.train_mask])
                else:
                    loss = F.cross_entropy(out, batch.y)

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def measure_inference_latency(model, data, device, model_type="default", n_runs=5):
    """Measure full-graph inference latency in milliseconds."""
    model.eval()
    latencies = []

    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            if model_type == "sgc":
                out = model(data.x.to(device))
                preds = out[data.test_mask].argmax(dim=1)
            else:
                out = model(data.x.to(device), data.edge_index.to(device))
                preds = out[data.test_mask].argmax(dim=1)

            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    return np.mean(latencies) * 1000, np.std(latencies) * 1000  # ms


def measure_gpu_memory(model, data, device, model_type="default"):
    """Measure peak GPU VRAM during a forward+backward pass."""
    reset_gpu_memory(device)

    model.train()
    data_dev = data.to(device)

    import torch.nn.functional as F

    if model_type == "sgc":
        out = model(data_dev.x)
        loss = F.cross_entropy(out[data.train_mask], data_dev.y[data.train_mask])
    else:
        out = model(data_dev.x, data_dev.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data_dev.y[data.train_mask])

    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return peak_mb


def measure_throughput(model, loader, device, n_epochs=3, model_type="neighbor"):
    """Measure training throughput in nodes per second."""
    import torch.nn.functional as F

    total_nodes = 0
    total_time = 0

    model.train()
    for epoch in range(n_epochs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for batch in loader:
            batch = batch.to(device)
            if model_type == "sgc":
                out = model(batch.x)
            else:
                out = model(batch.x, batch.edge_index)

            if hasattr(batch, "batch_size"):
                total_nodes += batch.batch_size
            elif hasattr(batch, "train_mask"):
                total_nodes += batch.train_mask.sum().item()
            else:
                total_nodes += batch.num_nodes

        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

    return total_nodes / total_time if total_time > 0 else 0


def efficiency_dashboard(models_info):
    """
    Print efficiency comparison dashboard.
    models_info: list of dicts with model efficiency data.
    """
    print("\n" + "=" * 90)
    print("EFFICIENCY DASHBOARD")
    print("=" * 90)
    header = f"{'Model':<15s} | {'Acc':>6s} | {'Params':>10s} | {'Epoch(s)':>9s} | {'Infer(ms)':>10s} | {'VRAM(MB)':>9s} | {'Throughput':>12s}"
    print(header)
    print("-" * 90)

    for m in models_info:
        print(
            f"{m.get('name',''):15s} | "
            f"{m.get('acc',0):6.4f} | "
            f"{m.get('params',0):10,d} | "
            f"{m.get('epoch_time',0):9.2f} | "
            f"{m.get('inference_ms',0):10.2f} | "
            f"{m.get('vram_mb',0):9.0f} | "
            f"{m.get('throughput',0):12,.0f}"
        )
    print("=" * 90)
