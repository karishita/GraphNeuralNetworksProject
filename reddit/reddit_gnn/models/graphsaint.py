"""
GraphSAINT Model — Zeng et al., 2020.
Architecture-agnostic subgraph sampling with normalization correction.
Uses a standard GCN backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphSAINTNet(nn.Module):
    """
    GCN backbone for use with GraphSAINT sampling.
    The sampling and normalization correction happen in the data loader
    and training loop, NOT in the model.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))

        # BatchNorm for all but last layer
        for _ in range(num_layers - 1):
            self.norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def encode(self, x, edge_index, edge_weight=None):
        """Extract penultimate layer embeddings."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
        return x
