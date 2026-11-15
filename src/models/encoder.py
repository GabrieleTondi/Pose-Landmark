from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2Encoder(nn.Module):
    """
    Graph encoder based on two edge-conditioned GATv2 layers, per-graph LayerNorm,
    global pooling, and MLP projection into the metric latent space.
    """

    def __init__(
        self,
        in_channels: int = 3,
        edge_dim: int = 5,
        hidden_dim: int = 32,
        embed_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.20,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.normalize_embeddings = bool(normalize_embeddings)

        self.conv1 = GATv2Conv(
            in_channels=int(in_channels),
            out_channels=int(hidden_dim),
            heads=int(heads),
            concat=True,
            dropout=self.dropout,
            edge_dim=int(edge_dim),
            add_self_loops=True,
            share_weights=False,
        )
        self.norm1 = nn.LayerNorm(int(hidden_dim) * int(heads))

        self.conv2 = GATv2Conv(
            in_channels=int(hidden_dim) * int(heads),
            out_channels=int(hidden_dim),
            heads=int(heads),
            concat=False,
            dropout=self.dropout,
            edge_dim=int(edge_dim),
            add_self_loops=True,
            share_weights=False,
        )
        self.norm2 = nn.LayerNorm(int(hidden_dim))

        self.proj = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        assert x.ndim == 2, f"Expected 2D x (N_tot, in_channels), got shape {tuple(x.shape)}."
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
            f"Expected edge_index (2, E_tot), got shape {tuple(edge_index.shape)}."
        )
        assert edge_attr.ndim == 2 and edge_attr.shape[0] == edge_index.shape[1], (
            f"Edge count mismatch between edge_index ({edge_index.shape[1]}) and edge_attr ({edge_attr.shape[0]})."
        )

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if return_attention_weights:
            h1, (e_idx1, alpha1) = self.conv1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            h1 = F.elu(self.norm1(h1))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)

            h2, (e_idx2, alpha2) = self.conv2(
                h1, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            h2 = F.elu(self.norm2(h2))
        else:
            h1 = self.conv1(x, edge_index, edge_attr=edge_attr)
            h1 = F.elu(self.norm1(h1))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)

            h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)
            h2 = F.elu(self.norm2(h2))

        pooled = global_mean_pool(h2, batch)
        z = self.proj(pooled)
        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)

        if return_attention_weights:
            return z, {"layer1": (e_idx1, alpha1), "layer2": (e_idx2, alpha2)}
        return z

