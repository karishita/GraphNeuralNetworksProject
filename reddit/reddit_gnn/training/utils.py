"""
Training Utilities — EarlyStopping, LR scheduler, gradient clipping, seed management.
Shared across all model training loops.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reddit_gnn.config import LOGS_DIR, CHECKPOINTS_DIR


class EarlyStopping:
    """Early stopping based on validation loss with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves best model state internally.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_scheduler(optimizer, factor=0.5, patience=5, min_lr=1e-6):
    """ReduceLROnPlateau scheduler."""
    return ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )


def clip_gradients(model, max_norm=1.0):
    """Clip gradient norms."""
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_gpu_memory(device=None):
    """Get current peak GPU memory in MB."""
    if device is None:
        device = torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return 0.0


def reset_gpu_memory(device=None):
    """Reset GPU memory tracking."""
    if device is None:
        device = torch.device("cuda:0")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def save_history(history, model_name, ablation_id="baseline", variant="default",
                 seed=0, save_dir=None):
    """Save training history to CSV."""
    if save_dir is None:
        save_dir = os.path.join(LOGS_DIR, model_name, ablation_id, variant, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "history.csv")

    if history:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    print(f"  History saved: {path}")
    return path


def save_checkpoint(model, model_name, ablation_id="baseline", variant="default",
                    seed=0, save_dir=None):
    """Save model checkpoint."""
    if save_dir is None:
        save_dir = os.path.join(CHECKPOINTS_DIR, model_name, ablation_id, variant, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "best_model.pt")
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved: {path}")
    return path


def log_epoch(epoch, train_loss, val_loss, val_acc, epoch_time, gpu_mem_mb=0,
              lr=None, extra=None):
    """Create a history entry dict for one epoch."""
    entry = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "val_acc": round(val_acc, 6),
        "epoch_time_s": round(epoch_time, 2),
        "gpu_mem_mb": round(gpu_mem_mb, 1),
    }
    if lr is not None:
        entry["lr"] = lr
    if extra:
        entry.update(extra)
    return entry
