"""
Training loop for NeighborLoader-based models: GraphSAGE, GAT, GATv2.
Shared structure — only the model class and loader config differ.

CRITICAL: Loss is computed on SEED nodes only (out[:batch.batch_size]).
NeighborLoader places seed nodes at the front of each batch.
Using the full tensor would compute loss on neighbor nodes that may
include val/test nodes, causing data leakage.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.training.utils import (
    EarlyStopping,
    get_scheduler,
    clip_gradients,
    log_epoch,
    measure_gpu_memory,
    reset_gpu_memory,
)


@torch.no_grad()
def evaluate_neighbor(model, loader, device):
    """
    Evaluate model using NeighborLoader.
    Only evaluates on seed nodes (first batch_size nodes in each batch).
    """
    model.eval()
    total_correct = 0
    total_nodes = 0
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)

        # Only evaluate seed nodes
        bs = batch.batch_size
        out_eval = out[:bs]
        y_eval = batch.y[:bs]

        total_loss += F.cross_entropy(out_eval, y_eval).item()
        total_correct += (out_eval.argmax(dim=1) == y_eval).sum().item()
        total_nodes += bs
        n_batches += 1

    acc = total_correct / max(total_nodes, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return acc, avg_loss


def train_neighbor_sampled(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    max_epochs=50,
    patience=10,
    model_name="model",
    scheduler=None,
    verbose=True,
):
    """
    Standard training loop for NeighborLoader-based models.

    Args:
        model: GraphSAGE, GAT, or GATv2 model
        train_loader: NeighborLoader with bounded sampling
        val_loader: NeighborLoader with full neighborhood
        optimizer: torch.optim optimizer
        device: torch.device
        max_epochs: Maximum epochs
        patience: Early stopping patience
        model_name: For logging
        scheduler: Optional LR scheduler
    """
    early_stop = EarlyStopping(patience=patience)
    if scheduler is None:
        scheduler = get_scheduler(optimizer)

    history = []
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        # ── Training phase ──
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        reset_gpu_memory(device)

        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            # CRITICAL: Only compute loss on SEED nodes
            bs = batch.batch_size
            loss = F.cross_entropy(out[:bs], batch.y[:bs])
            loss.backward()
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (out[:bs].argmax(dim=1) == batch.y[:bs]).sum().item()
            total_nodes += bs

        epoch_time = time.time() - t0
        train_loss = total_loss / max(total_nodes / batch.batch_size, 1)
        train_acc = total_correct / max(total_nodes, 1)
        gpu_mem = measure_gpu_memory(device)

        # ── Validation phase ──
        val_acc, val_loss = evaluate_neighbor(model, val_loader, device)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
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
                f"Time: {epoch_time:.1f}s | VRAM: {gpu_mem:.0f}MB | "
                f"LR: {current_lr:.6f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Early stopping
        if early_stop.step(val_loss, model):
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    early_stop.restore_best(model)

    if verbose:
        print(f"  Best val acc: {best_val_acc:.4f}")

    return history
