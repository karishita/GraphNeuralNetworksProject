"""
GraphSAINT Training Loop.
Key difference: normalization-corrected loss for unbiased gradients.
Loss = mean(per_node_loss * node_norm[train_mask])
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
from reddit_gnn.training.train_neighbor import evaluate_neighbor


@torch.no_grad()
def evaluate_saint(model, data, device):
    """Full-graph evaluation for GraphSAINT (no sampling at eval time)."""
    model.eval()
    data_dev = data.to(device)
    out = model(data_dev.x, data_dev.edge_index)

    # Validation
    val_out = out[data.val_mask]
    val_y = data_dev.y[data.val_mask]
    val_loss = F.cross_entropy(val_out, val_y).item()
    val_acc = (val_out.argmax(dim=1) == val_y).float().mean().item()

    return val_acc, val_loss


def train_saint(
    model,
    saint_loader,
    data,
    optimizer,
    device,
    max_epochs=30,
    patience=10,
    use_norm=True,
    model_name="graphsaint",
    scheduler=None,
    verbose=True,
):
    """
    GraphSAINT training loop.

    Args:
        model: GraphSAINTNet model
        saint_loader: GraphSAINT sampler loader
        data: Full graph data (for validation)
        optimizer: Optimizer
        device: Device
        use_norm: If True, apply normalization correction (B2 ablation)
        model_name: For logging
    """
    early_stop = EarlyStopping(patience=patience)
    if scheduler is None:
        scheduler = get_scheduler(optimizer)

    history = []
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        reset_gpu_memory(device)

        t0 = time.time()

        for batch in saint_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            # Compute per-node loss
            loss_per_node = F.cross_entropy(
                out[batch.train_mask],
                batch.y[batch.train_mask],
                reduction="none",
            )

            if use_norm and hasattr(batch, "node_norm"):
                # Weight each node's loss by normalization correction
                norm_weights = batch.node_norm[batch.train_mask]
                loss = (loss_per_node * norm_weights).mean()
            else:
                # B2 ablation: no normalization correction
                loss = loss_per_node.mean()

            loss.backward()
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)
        gpu_mem = measure_gpu_memory(device)

        # Validation (full graph)
        val_acc, val_loss = evaluate_saint(model, data, device)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        entry = log_epoch(epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem, current_lr)
        history.append(entry)

        if verbose:
            print(
                f"  [{model_name}] Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} | "
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
