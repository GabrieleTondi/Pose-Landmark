# 06 — Sector Sub-Specification: Siamese GATv2 Architecture and Training

**Sector:** 06 — Deep Learning Model: `GATv2Encoder`, `SiameseGATv2`, Regularization, and Training Loop  
**Parent Specification References:** §7.1–§7.5, §16.4  
**Input Dependencies:** Sector 04 (PyG Graphs), Sector 05 (`TripletDataset`, `AdaptiveTripletMarginLoss`), `configs/training.yaml`  
**Enabled Outputs:** Sector 07 (Evaluation, Prototypes, and Explainability) and Sector 08 (3D UI and What-If Lab)

---

## 1. Sector Objective

Implement the neural model `GATv2Encoder` (`src/models/encoder.py`), the shared-weight wrapper `SiameseGATv2` (`src/models/siamese_gat.py`), and the training pipeline with deterministic validation, anti-collapse monitoring, and best checkpoint saving (`src/training/train.py`).

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 6.1 — Implementation of `GATv2Encoder` (`src/models/encoder.py`)
- **File to create:** `src/models/encoder.py`
- **Exact class contract:**
  ```python
  from typing import Optional, Tuple, Dict, Union
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch_geometric.nn import GATv2Conv, global_mean_pool

  class GATv2Encoder(nn.Module):
      def __init__(
          self,
          in_channels: int = 3,
          edge_dim: int = 5,
          hidden_dim: int = 32,
          embed_dim: int = 32,
          heads: int = 4,
          dropout: float = 0.20,
          normalize_embeddings: bool = False
      ) -> None: ...

      def forward(
          self,
          x: torch.Tensor,
          edge_index: torch.Tensor,
          edge_attr: torch.Tensor,
          batch: Optional[torch.Tensor] = None,
          return_attention_weights: bool = False
      ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]: ...
  ```
- **HOW to implement `__init__` and `forward` of `GATv2Encoder` (Exact code):**
  ```python
  class GATv2Encoder(nn.Module):
      def __init__(
          self,
          in_channels: int = 3,
          edge_dim: int = 5,
          hidden_dim: int = 32,
          embed_dim: int = 32,
          heads: int = 4,
          dropout: float = 0.20,
          normalize_embeddings: bool = False
      ) -> None:
          super().__init__()
          self.dropout = float(dropout)
          self.normalize_embeddings = bool(normalize_embeddings)

          self.conv1 = GATv2Conv(
              in_channels=in_channels,
              out_channels=hidden_dim,
              heads=heads,
              concat=True,
              dropout=self.dropout,
              edge_dim=edge_dim,
              add_self_loops=True,
              share_weights=False
          )
          self.norm1 = nn.LayerNorm(hidden_dim * heads)

          self.conv2 = GATv2Conv(
              in_channels=hidden_dim * heads,
              out_channels=hidden_dim,
              heads=heads,
              concat=False,
              dropout=self.dropout,
              edge_dim=edge_dim,
              add_self_loops=True,
              share_weights=False
          )
          self.norm2 = nn.LayerNorm(hidden_dim)

          self.proj = nn.Sequential(
              nn.Linear(hidden_dim, hidden_dim),
              nn.ELU(),
              nn.Dropout(self.dropout),
              nn.Linear(hidden_dim, embed_dim)
          )

      def forward(
          self,
          x: torch.Tensor,
          edge_index: torch.Tensor,
          edge_attr: torch.Tensor,
          batch: Optional[torch.Tensor] = None,
          return_attention_weights: bool = False
      ):
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

          g = global_mean_pool(h2, batch)  # shape (B, hidden_dim)
          z = self.proj(g)                 # shape (B, embed_dim)
          if self.normalize_embeddings:
              z = F.normalize(z, p=2, dim=-1)

          if return_attention_weights:
              return z, {"layer1": (e_idx1, alpha1), "layer2": (e_idx2, alpha2)}
          return z
  ```
- **CONSTRAINT:** Never use `nn.BatchNorm1d` (use `nn.LayerNorm`) so that inference on a single graph (`B=1`, 17 nodes) works without depending on running mean/var statistics.

---

### Task 6.2 — Implementation of `SiameseGATv2` (`src/models/siamese_gat.py`)
- **File to create:** `src/models/siamese_gat.py`
- **Exact class contract:**
  ```python
  from typing import Tuple, Dict, Any, Union
  import torch
  import torch.nn as nn
  from torch_geometric.data import Data, Batch
  from src.models.encoder import GATv2Encoder

  class SiameseGATv2(nn.Module):
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
              normalize_embeddings=bool(m_cfg.get("normalize_embeddings", False))
          )
          return cls(enc)

      def encode(
          self,
          data: Union[Data, Batch],
          return_attention_weights: bool = False
      ):
          batch_vec = getattr(data, "batch", None)
          return self.encoder(
              data.x,
              data.edge_index,
              data.edge_attr,
              batch=batch_vec,
              return_attention_weights=return_attention_weights
          )

      def forward_triplet(
          self,
          batch_A: Union[Data, Batch],
          batch_P: Union[Data, Batch],
          batch_N: Union[Data, Batch]
      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          z_A = self.encode(batch_A, return_attention_weights=False)
          z_P = self.encode(batch_P, return_attention_weights=False)
          z_N = self.encode(batch_N, return_attention_weights=False)
          return z_A, z_P, z_N

      def compute_distance(
          self,
          data_1: Union[Data, Batch],
          data_2: Union[Data, Batch]
      ) -> torch.Tensor:
          z_1 = self.encode(data_1, return_attention_weights=False)
          z_2 = self.encode(data_2, return_attention_weights=False)
          return torch.sqrt(torch.sum((z_1 - z_2) ** 2, dim=-1) + 1e-12)
  ```
- **CONSTRAINT:** There must be **a single instance** `self.encoder`. Never clone or instantiate three separate encoders.

---

### Task 6.3 — Training and Validation Pipeline (`src/training/train.py`)
- **File to create:** `src/training/train.py`
- **Exact function contracts:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List
  import torch
  from torch_geometric.data import Data
  from torch_geometric.loader import DataLoader
  from src.models.siamese_gat import SiameseGATv2
  from src.training.losses import AdaptiveTripletMarginLoss

  def train_one_epoch(
      model: SiameseGATv2,
      loader: DataLoader,
      criterion: AdaptiveTripletMarginLoss,
      optimizer: torch.optim.Optimizer,
      device: torch.device,
      max_grad_norm: float = 1.0
  ) -> Dict[str, float]: ...

  @torch.no_grad()
  def evaluate_triplets(
      model: SiameseGATv2,
      loader: DataLoader,
      criterion: AdaptiveTripletMarginLoss,
      device: torch.device
  ) -> Dict[str, float]: ...

  def run_training_pipeline(
      train_graphs: List[Data],
      val_graphs: List[Data],
      data_cfg: Dict[str, Any],
      graph_cfg: Dict[str, Any],
      train_cfg: Dict[str, Any]
  ) -> Dict[str, Any]: ...
  ```
- **HOW to implement `train_one_epoch` and `evaluate_triplets`:**
  1. In `train_one_epoch`:
     - Set `model.train()`.
     - For each `batch_A, batch_P, batch_N, batch_alpha` in `loader`:
       - Move to device: `batch_A = batch_A.to(device)`, `batch_P = batch_P.to(device)`, `batch_N = batch_N.to(device)`, `batch_alpha = batch_alpha.to(device)`.
       - `optimizer.zero_grad()`
       - `z_A, z_P, z_N = model.forward_triplet(batch_A, batch_P, batch_N)`
       - `loss, stats = criterion(z_A, z_P, z_N, batch_alpha)`
       - `loss.backward()`
       - `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)`
       - `optimizer.step()`
     - Compute the weighted average over the number of triplets of `loss`, `triplet_acc`, `margin_satisfied_acc`, `mean_d_ap`, `mean_d_an`.
  2. In `evaluate_triplets`:
     - Set `model.eval()`.
     - Iterate over the validation `loader` while also collecting all embeddings `z_A` to compute the mean standard deviation of the latent coordinates:
       ```python
       all_z = torch.cat(collected_z_A, dim=0)
       embedding_std = float(torch.std(all_z, dim=0).mean().item())
       ```
     - Return the dictionary with `val_loss`, `val_triplet_acc`, `val_margin_satisfied_acc`, `val_mean_d_ap`, `val_mean_d_an`, `embedding_std`.
  3. In `run_training_pipeline`:
     - Generate synthetic negatives on `train_graphs` only via `generate_synthetic_negatives`.
     - Build `train_pools, thresholds = partition_by_goe_tiers(train_graphs + synth_graphs, train_cfg, fit_thresholds=None)`.
     - Build `val_pools, _ = partition_by_goe_tiers(val_graphs, train_cfg, fit_thresholds=thresholds)`.
     - Create `val_ds = TripletDataset(val_pools, train_cfg, num_triplets=num_val_triplets, split="val", seed=seed)` once outside the epoch loop.
     - At each epoch $e$, recreate `train_ds = TripletDataset(train_pools, train_cfg, num_triplets=num_train_triplets, split="train", seed=seed + e)` to sample new combinations of training triplets.
     - Save `checkpoints/best_model.pt` whenever `val_triplet_acc` improves (or on a tie in `val_triplet_acc`, when `val_loss` decreases):
       ```python
       torch.save({
           "epoch": epoch,
           "model_state_dict": model.state_dict(),
           "val_loss": val_metrics["val_loss"],
           "val_triplet_acc": val_metrics["val_triplet_acc"],
           "goe_thresholds": thresholds,
           "train_cfg": train_cfg,
           "graph_cfg": graph_cfg,
           "data_cfg": data_cfg
       }, ckpt_path)
       ```
     - Save the list of per-epoch metrics to `reports/training_history.json`.

---

## 3. Unit Tests to Implement (in `tests/test_triplets.py`)

Add the following 3 tests on the model and training to `tests/test_triplets.py`:
1. `test_siamese_shared_weights_and_edge_attr_sensitivity()`: verifies that `len(list(model.parameters())) == len(list(model.encoder.parameters()))` and that changing `data.edge_attr` while keeping `data.x` fixed changes the embedding returned by `model.encode(data)` (`not torch.allclose(z1, z2)`).
2. `test_attention_weights_extraction_shape()`: verifies that for a single graph ($B=1$, 17 nodes, 32 edges) `z, attn = model.encode(data, return_attention_weights=True)` returns `attn["layer2"][1].shape == (49, 4)`.
3. `test_single_batch_overfit()`: takes 16 fixed triplets, runs 35 steps of `train_one_epoch` with `lr=5e-3`, and verifies that the final loss is $< 0.15$ and `triplet_acc >= 0.90`.

---

## 4. Sector 06 Acceptance Criteria (Definition of Done)

- [ ] The files `src/models/encoder.py`, `src/models/siamese_gat.py`, and `src/training/train.py` have been created.
- [ ] All tests in `pytest tests/test_triplets.py` pass 100%.
