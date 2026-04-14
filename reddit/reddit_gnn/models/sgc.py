"""
SGC — Simple Graph Convolution, Wu et al., 2019.
Collapses multi-layer GCN into a single linear transform on pre-smoothed features.
Model: Y = softmax(A^K * X * theta)

During training, NO graph access is needed — features are precomputed.
This makes SGC ~25x faster than GCN (Lecture 6).
"""

import torch
import torch.nn as nn


class SGC(nn.Module):
    """
    Linear classifier on precomputed X_K = A^K * X.
    Training is standard logistic regression (no graph in the loop).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.classifier = nn.Linear(in_channels, out_channels)
        self.hidden_dim = in_channels  # X_K itself is the "embedding"

    def forward(self, x):
        """
        Forward pass — simple linear transformation.
        x: Precomputed X_K features [N, F] or [batch, F]
        """
        return self.classifier(x)

    def encode(self, x):
        """
        For SGC, the pre-smoothed X_K IS the embedding.
        No nonlinear layers to extract intermediate representations from.
        """
        return x
