"""
GraphSAGE Model — Hamilton et al., 2017.
Inductive neighborhood sampling with configurable aggregator.
Separates self from neighbors via concatenation.

Supports: mean/max/lstm/sum aggregators, skip connections, BatchNorm/LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = "mean",
        skip: bool = False,
        skip_type: str = "add",  # 'add' or 'concat'
        norm: str = "batchnorm",  # 'batchnorm', 'layernorm', None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip = skip
        self.skip_type = skip_type
        self.hidden_dim = hidden_channels

        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
            )

        # Final layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

        # Normalization layers (for all but the last conv)
        for _ in range(num_layers - 1):
            if norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif norm == "layernorm":
                self.norms.append(nn.LayerNorm(hidden_channels))
            else:
                self.norms.append(nn.Identity())

        # Skip connection projection (if input/hidden dims differ)
        if skip and skip_type == "add" and in_channels != hidden_channels:
            self.skip_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.skip_proj = None

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x_prev = x
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Skip connection
            if self.skip:
                if self.skip_type == "add":
                    if i == 0 and self.skip_proj is not None:
                        x = x + self.skip_proj(x_prev)
                    elif x_prev.shape == x.shape:
                        x = x + x_prev

        # Final layer (no activation, no norm, no skip)
        x = self.convs[-1](x, edge_index)
        return x

    def encode(self, x, edge_index):
        """Extract penultimate layer embeddings (before final linear)."""
        for i, conv in enumerate(self.convs[:-1]):
            x_prev = x
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)

            if self.skip:
                if self.skip_type == "add":
                    if i == 0 and self.skip_proj is not None:
                        x = x + self.skip_proj(x_prev)
                    elif x_prev.shape == x.shape:
                        x = x + x_prev
        return x
