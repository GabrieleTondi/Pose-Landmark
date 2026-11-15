from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models.encoder import GATv2Encoder


class SiameseGATv2(nn.Module):
    """
    Wrapper Siamese a pesi strettamente condivisi attorno a una singola istanza di GATv2Encoder.
    """

    def __init__(self, encoder: GATv2Encoder) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_config(cls, train_cfg: Dict[str, Any]) -> "SiameseGATv2":
        m_cfg = train_cfg["model"]
        enc = GATv2Encoder(
            in_channels=int(m_cfg["in_channels"]),
            edge_dim=int(m_cfg["edge_dim"]),
            hidden_dim=int(m_cfg["hidden_dim"]),
            embed_dim=int(m_cfg["embed_dim"]),
            heads=int(m_cfg["heads"]),
            dropout=float(m_cfg["dropout"]),
            normalize_embeddings=bool(m_cfg.get("normalize_embeddings", False)),
        )
        return cls(enc)

    def encode(
        self,
        data: Union[Data, Batch],
        return_attention_weights: bool = False,
    ):
        batch_vec = getattr(data, "batch", None)
        return self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch=batch_vec,
            return_attention_weights=return_attention_weights,
        )

    def forward_triplet(
        self,
        batch_A: Union[Data, Batch],
        batch_P: Union[Data, Batch],
        batch_N: Union[Data, Batch],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_A = self.encode(batch_A, return_attention_weights=False)
        z_P = self.encode(batch_P, return_attention_weights=False)
        z_N = self.encode(batch_N, return_attention_weights=False)
        return z_A, z_P, z_N

    def compute_distance(
        self,
        data_1: Union[Data, Batch],
        data_2: Union[Data, Batch],
    ) -> torch.Tensor:
        z_1 = self.encode(data_1, return_attention_weights=False)
        z_2 = self.encode(data_2, return_attention_weights=False)
        return torch.sqrt(torch.sum((z_1 - z_2) ** 2, dim=-1) + 1e-12)

