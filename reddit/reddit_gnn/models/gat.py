"""
GAT — Graph Attention Network, Veličković et al., 2018.
STATIC attention: W is applied BEFORE concatenation.
alpha_{ij} = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))

The ranking of neighbors is FIXED across all query nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_per_head: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        attn_dropout: float = 0.3,
        feat_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.feat_dropout = feat_dropout
        self.hidden_dim = hidden_per_head * num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: multi-head, concat output
        self.convs.append(
            GATConv(
                in_channels,
                hidden_per_head,
                heads=num_heads,
                dropout=attn_dropout,
                concat=True,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_per_head * num_heads,
                    hidden_per_head,
                    heads=num_heads,
                    dropout=attn_dropout,
                    concat=True,
                )
            )

        # Final layer: single head (or average heads), no concat
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_per_head * num_heads,
                    out_channels,
                    heads=1,
                    dropout=attn_dropout,
                    concat=False,
                )
            )

        # Normalization for all but last layer
        for _ in range(num_layers - 1):
            self.norms.append(nn.BatchNorm1d(hidden_per_head * num_heads))

    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Args:
            return_attention_weights: If True, return (output, attn_weights_list).
                                     Used for D4 ablation analysis.
        """
        attn_weights_list = []

        for i, conv in enumerate(self.convs[:-1]):
            if return_attention_weights:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attn_weights_list.append(attn)
            else:
                x = conv(x, edge_index)

            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.feat_dropout, training=self.training)

        # Final layer
        if return_attention_weights:
            x, attn = self.convs[-1](x, edge_index, return_attention_weights=True)
            attn_weights_list.append(attn)
            return x, attn_weights_list
        else:
            x = self.convs[-1](x, edge_index)
            return x

    def encode(self, x, edge_index):
        """Extract penultimate layer embeddings."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
        return x
