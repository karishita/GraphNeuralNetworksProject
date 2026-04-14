"""
ClusterGCN Training Loop.
Trains on METIS-partitioned subgraphs with optional diagonal enhancement.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.data.partition_cluster import apply_diagonal_enhancement
from reddit_gnn.training.utils import (
    EarlyStopping,
    get_scheduler,
    clip_gradients,
    log_epoch,
    measure_gpu_memory,
    reset_gpu_memory,
)


@torch.no_grad()
def evaluate_cluster(model, data, device):
    """Full-graph evaluation for ClusterGCN."""
    model.eval()
    data_dev = data.to(device)
    out = model(data_dev.x, data_dev.edge_index)

    val_out = out[data.val_mask]
    val_y = data_dev.y[data.val_mask]
    val_loss = F.cross_entropy(val_out, val_y).item()
    val_acc = (val_out.argmax(dim=1) == val_y).float().mean().item()

    return val_acc, val_loss


def train_cluster_gcn(
    model,
    cluster_data,
    data,
    optimizer,
    device,
    clusters_per_batch=20,
    lambda_val=0.1,
    max_epochs=50,
    patience=10,
    model_name="cluster_gcn",
    scheduler=None,
    verbose=True,
):
    """
    ClusterGCN training loop.

    Args:
        model: ClusterGCN model
        cluster_data: PyG ClusterData object (from METIS)
        data: Full graph data (for validation)
        clusters_per_batch: Number of clusters sampled per mini-batch
        lambda_val: Diagonal enhancement weight (0 = disabled)
    """
    from torch_geometric.loader import ClusterLoader

    train_loader = ClusterLoader(
        cluster_data,
        batch_size=clusters_per_batch,
        shuffle=True,
        num_workers=4,
    )

    early_stop = EarlyStopping(patience=patience)
    if scheduler is None:
        scheduler = get_scheduler(optimizer)

    history = []
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        n_batches = 0
        reset_gpu_memory(device)

        t0 = time.time()

        for batch in train_loader:
            # Apply diagonal enhancement per mini-batch
            if lambda_val > 0:
                batch = apply_diagonal_enhancement(batch, lambda_val)

            batch = batch.to(device)
            optimizer.zero_grad()

            edge_weight = batch.edge_weight if hasattr(batch, "edge_weight") else None
            out = model(batch.x, batch.edge_index, edge_weight)

            # Only train on training nodes within this cluster batch
            if batch.train_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                out[batch.train_mask], batch.y[batch.train_mask]
            )
            loss.backward()
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (
                (out[batch.train_mask].argmax(1) == batch.y[batch.train_mask]).sum().item()
            )
            total_nodes += batch.train_mask.sum().item()
            n_batches += 1

        epoch_time = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_correct / max(total_nodes, 1)
        gpu_mem = measure_gpu_memory(device)

        # Validation (full graph)
        val_acc, val_loss = evaluate_cluster(model, data, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        entry = log_epoch(
            epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem, current_lr,
            extra={"train_acc": round(train_acc, 6)},
        )
        history.append(entry)

        if verbose:
            print(
                f"  [{model_name}] Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s | VRAM: {gpu_mem:.0f}MB"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if early_stop.step(val_loss, model):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    early_stop.restore_best(model)
    if verbose:
        print(f"  Best val acc: {best_val_acc:.4f}")

    return history
