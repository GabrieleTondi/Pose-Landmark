# 05 — Sector Sub-Specification: Triplet Sampling, Adaptive Loss, and Data Augmentation

**Sector:** 05 — Metric Learning Pipeline: Rodrigues Rotation, Intra-Class Triplet Sampler, and Adaptive Loss  
**Parent Specification References:** §8.1–§8.6  
**Input Dependencies:** Sector 01 (`configs/training.yaml`, `configs/graph.yaml`), Sector 03 (`canonicalize_frame_3d`), Sector 04 (`build_pyg_graph`)  
**Enabled Outputs:** Sector 06 (`SiameseGATv2` Training), Sector 08 (*What-If Lab* Sliders)

---

## 1. Sector Objective

Implement:
1. Rigid 3D Rodrigues rotation on kinematic subtrees (`src/training/augmentation.py`) to generate synthetic negatives in the training set only (and to animate the interactive sliders in the *What-If Lab* of Sector 08);
2. GOE tier partitioning and the PyG triplet dataset `TripletDataset` (`src/training/triplet_sampler.py`);
3. The `AdaptiveTripletMarginLoss` cost function (`src/training/losses.py`) with protection against `NaN` gradients.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 5.1 — Rigid Rodrigues Rotation and Augmentation (`src/training/augmentation.py`)
- **File to create:** `src/training/augmentation.py`
- **Exact function contracts:**
  ```python
  from typing import Dict, Any, List
  import numpy as np
  from torch_geometric.data import Data
  from src.data.preprocessing import canonicalize_frame_3d
  from src.graph.builder import build_pyg_graph

  def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
      """
      Computes the 3x3 rotation matrix in SO(3) around the 'axis' vector by an angle 'angle_rad'
      using Rodrigues' formula: R = I + sin(theta)*K + (1 - cos(theta))*(K @ K).
      Returns a float32 np.ndarray of shape (3, 3).
      """

  def rotate_kinematic_subtree(
      P_bar: np.ndarray,
      joint_idx: int,
      child_idx: int,
      axis: np.ndarray,
      angle_deg: float,
      graph_cfg: Dict[str, Any]
  ) -> np.ndarray:
      """
      Rigidly rotates around the pivot point P_bar[joint_idx] all nodes belonging to the
      subtree kinematic_subtrees[f"{joint_idx}->{child_idx}"] by 'angle_deg' degrees.
      Returns the new matrix P_prime of shape (17, 3) float32 with unchanged bone lengths.
      """

  def generate_synthetic_negatives(
      train_graphs: List[Data],
      data_cfg: Dict[str, Any],
      graph_cfg: Dict[str, Any],
      train_cfg: Dict[str, Any]
  ) -> List[Data]:
      """
      For each real positive graph in the training set ONLY (split == 'train' and not is_augmented),
      generates 3 synthetic negative graphs (one for each severity_tier 1, 2, 3)
      and returns the list of new synthetic Data objects.
      """
  ```
- **HOW to implement `rodrigues_rotation_matrix` and `rotate_kinematic_subtree` (Exact code):**
  ```python
  def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
      u = np.asarray(axis, dtype=np.float64)
      norm = np.linalg.norm(u)
      if norm < 1e-8:
          return np.eye(3, dtype=np.float32)
      u = u / norm
      ux, uy, uz = u[0], u[1], u[2]
      K = np.array([
          [0.0, -uz,  uy],
          [ uz, 0.0, -ux],
          [-uy,  ux, 0.0]
      ], dtype=np.float64)
      I = np.eye(3, dtype=np.float64)
      R = I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)
      return R.astype(np.float32)

  def rotate_kinematic_subtree(
      P_bar: np.ndarray,
      joint_idx: int,
      child_idx: int,
      axis: np.ndarray,
      angle_deg: float,
      graph_cfg: Dict[str, Any]
  ) -> np.ndarray:
      key = f"{joint_idx}->{child_idx}"
      subtree_nodes = graph_cfg["kinematic_subtrees"].get(key, [child_idx])
      R = rodrigues_rotation_matrix(axis, np.deg2rad(float(angle_deg)))
      P_prime = P_bar.copy().astype(np.float32)
      pivot = P_bar[joint_idx]
      for m in subtree_nodes:
          P_prime[m] = pivot + R @ (P_bar[m] - pivot)
      return P_prime
  ```
- **HOW to implement `generate_synthetic_negatives`:**
  1. Filter the input graphs keeping only those with `g.split == "train"` and `not g.is_augmented`.
  2. Read `tier_ranges = train_cfg["augmentation"]["tier_angle_ranges_deg"]` and `noise_std = float(train_cfg["augmentation"]["noise_std"])`.
  3. For each positive graph in the training set and for each `tier` in `[1, 2, 3]`:
     - Choose a key from `list(graph_cfg["kinematic_subtrees"].keys())` (prioritizing lower-limb joints `["11->13", "13->15", "12->14", "14->16"]` and upper-limb joints `["5->7", "6->8"]`);
     - Sample `angle_deg = rng.uniform(tier_ranges[tier][0], tier_ranges[tier][1]) * rng.choice([-1.0, 1.0])`;
     - Choose a unit axis orthogonal to the bone (e.g., `[0.0, 1.0, 0.0]` or `[1.0, 0.0, 0.0]`);
     - Apply `P_pert = rotate_kinematic_subtree(g.pos.numpy(), joint_idx, child_idx, axis, angle_deg, graph_cfg)`;
     - If `noise_std > 0`: add `rng.normal(0.0, noise_std, size=P_pert.shape).astype(np.float32)`;
     - Re-canonicalize with `P_pert_canon, _ = canonicalize_frame_3d(P_pert, data_cfg)`;
     - Build the new graph with `build_pyg_graph(P_pert_canon, meta, graph_cfg)` where `meta` has:
       - `sample_id = f"{g.sample_id}_aug_t{tier}"`
       - `performance_id = g.performance_id`
       - `athlete_id = g.athlete_id`
       - `element_label = g.element_label`
       - `goe = float(g.goe) - float(tier)`
       - `split = "train"`
       - `is_augmented = True`
       - `parent_sample_id = g.sample_id`
       - `severity_tier = int(tier)`
       - `perturbation_joint = int(joint_idx)`
       - `perturbation_angle_deg = float(angle_deg)`
- **CONSTRAINT:** Raise `ValueError` if a graph with `g.split != "train"` is passed to `generate_synthetic_negatives`.

---

### Task 5.2 — GOE Partitioning and `TripletDataset` (`src/training/triplet_sampler.py`)
- **File to create:** `src/training/triplet_sampler.py`
- **Exact function and class contracts:**
  ```python
  from typing import Dict, Any, List, Optional, Tuple
  import numpy as np
  import torch
  from torch_geometric.data import Data

  def partition_by_goe_tiers(
      graphs: List[Data],
      train_cfg: Dict[str, Any],
      fit_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
  ) -> Tuple[Dict[str, Dict[str, List[Data]]], Dict[str, Tuple[float, float]]]:
      """
      Groups graphs by element_label and partitions them into:
      pools[label] = {'pos': [...], 'mid': [...], 'neg_real': [...], 'neg_synth': [...]}
      If fit_thresholds is None (i.e., on the training set), computes (tau_neg, tau_pos) for each class
      using the absolute thresholds from train_cfg (or quantiles if either the 'pos' or 'neg_real' pool has < 2 performances).
      If fit_thresholds is provided (on val and test), uses the already computed thresholds.
      """

  class TripletDataset(torch.utils.data.Dataset):
      """
      Dataset that returns at index idx a 4-element tuple compatible with
      torch_geometric.loader.DataLoader:
      (graph_A, graph_P, graph_N, margin_alpha_tensor)
      where margin_alpha_tensor is torch.tensor(alpha, dtype=torch.float32).
      """
      def __init__(
          self,
          pools: Dict[str, Dict[str, List[Data]]],
          train_cfg: Dict[str, Any],
          num_triplets: int,
          split: str = "train",
          seed: int = 42
      ) -> None: ...
  ```
- **HOW to construct triplets in `TripletDataset` (Exact logic):**
  1. In the `__init__` constructor, generate a deterministic list `self.triplets: List[Tuple[Data, Data, Data, torch.Tensor]]` of length `num_triplets` using `rng = np.random.default_rng(seed)`.
  2. For each triplet $k \in \{0, \dots, \text{num\_triplets}-1\}$:
     - Choose a class `c` among those available in `pools`;
     - Draw `A` from `pools[c]["pos"]` (all graphs in `"pos"` have `is_augmented == False`);
     - Find candidate `P` in `pools[c]["pos"]` such that `P.performance_id != A.performance_id` (and `P.athlete_id != A.athlete_id` if at least 2 distinct athletes exist in the pool; if only 1 exists in the mini-split, require `P.sample_id != A.sample_id`);
     - **Selection of `N`:**
       - If `split in ("val", "test")` or `len(pools[c]["neg_synth"]) == 0` or `rng.random() < real_negative_prob`:
         Find `N` in `pools[c]["neg_real"]` with `N.performance_id != A.performance_id`.
         Compute the severity of the real negative based on $\Delta \text{GOE} = A.\text{goe} - N.\text{goe}$:
         - $\Delta \text{GOE} < 2.0 \implies \text{tier} = 1$
         - $2.0 \le \Delta \text{GOE} < 3.5 \implies \text{tier} = 2$
         - $\Delta \text{GOE} \ge 3.5 \implies \text{tier} = 3$
       - Otherwise (only in `split == "train"`):
         With probability $0.5$ choose a *self-structural negative* of `A` (`N` in `pools[c]["neg_synth"]` with `N.parent_sample_id == A.sample_id`), otherwise any `N` in `pools[c]["neg_synth"]`. The `tier` is `N.severity_tier` ($\in \{1, 2, 3\}$).
     - **Computation of `alpha`:**
       - If `train_cfg["loss"]["margin_mode"] == "fixed"`: `alpha = float(train_cfg["loss"]["fixed_margin"])`
       - Otherwise: `alpha = float(train_cfg["loss"]["adaptive_margins"][tier])`
     - Store `(A, P, N, torch.tensor(alpha, dtype=torch.float32))`.
- **CRITICAL DATALOADER CONSTRAINT (PyTorch vs PyG Pitfall):**
  When iterating over `TripletDataset` in train/val/test, **always and only** use `from torch_geometric.loader import DataLoader` (and **NEVER** standard `torch.utils.data.DataLoader`!). The PyG `DataLoader` automatically recognizes a `(Data, Data, Data, Tensor)` tuple and returns at each iteration `(batch_A, batch_P, batch_N, batch_alpha)` where `batch_A, batch_P, batch_N` are PyG `Batch` objects and `batch_alpha` is a `(B,)` tensor!

---

### Task 5.3 — `AdaptiveTripletMarginLoss` Cost Function (`src/training/losses.py`)
- **File to create:** `src/training/losses.py`
- **Exact class contract:**
  ```python
  from typing import Tuple, Dict
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  class AdaptiveTripletMarginLoss(nn.Module):
      def __init__(self, eps: float = 1e-12) -> None:
          super().__init__()
          self.eps = eps

      def forward(
          self,
          z_A: torch.Tensor,
          z_P: torch.Tensor,
          z_N: torch.Tensor,
          margin_alpha: torch.Tensor
      ) -> Tuple[torch.Tensor, Dict[str, float]]:
          """
          Computes the Triplet Margin Loss with per-sample margin margin_alpha of shape (B,).
          Returns:
          - loss: differentiable scalar Tensor
          - stats: {'loss': float, 'triplet_acc': float, 'margin_satisfied_acc': float,
                    'mean_d_ap': float, 'mean_d_an': float}
          """
  ```
- **HOW to implement `forward` (Exact code):**
  ```python
  d_ap = torch.sqrt(torch.sum((z_A - z_P) ** 2, dim=-1) + self.eps)  # shape (B,)
  d_an = torch.sqrt(torch.sum((z_A - z_N) ** 2, dim=-1) + self.eps)  # shape (B,)
  margins = margin_alpha.to(z_A.device).view(-1)                     # shape (B,)
  losses = F.relu(d_ap - d_an + margins)                             # shape (B,)
  loss = torch.mean(losses)

  with torch.no_grad():
      triplet_acc = float(torch.mean((d_ap < d_an).float()).item())
      margin_acc = float(torch.mean((d_ap + margins < d_an).float()).item())
      stats = {
          "loss": float(loss.item()),
          "triplet_acc": triplet_acc,
          "margin_satisfied_acc": margin_acc,
          "mean_d_ap": float(d_ap.mean().item()),
          "mean_d_an": float(d_an.mean().item())
      }
  return loss, stats
  ```

---

## 3. Unit Tests to Implement (`tests/test_triplets.py`)

The file `tests/test_triplets.py` must implement the following 4 automated tests:
1. `test_rodrigues_subtree_preserves_bone_lengths()`: applies a $+30^\circ$ rotation to subtree `"11->13"` (left hip $\to$ left knee and left ankle), rebuilds the graph with `build_pyg_graph`, and verifies that all 32 bone lengths `edge_attr[:, 3]` remain unchanged within $10^{-5}$, while the angle cosine in `edge_attr[:, 4]` at the left hip has changed.
2. `test_triplet_dataset_intra_class_and_no_val_augmentation()`: verifies over 100 triplets of `train` and `val` that `A.element_label == P.element_label == N.element_label`, that `A.performance_id != P.performance_id`, and that in `val` no graph has `is_augmented == True`.
3. `test_adaptive_triplet_loss_backward_and_zero_distance_safety()`: verifies that even when passing `z_A == z_P` (exact zero distance), `loss.backward()` produces finite gradients (`torch.isfinite(z_A.grad).all()`).
4. `test_pyg_dataloader_on_triplet_dataset()`: verifies that `torch_geometric.loader.DataLoader(triplet_ds, batch_size=8)` returns `(batch_A, batch_P, batch_N, batch_alpha)` with `batch_alpha.shape == (8,)`.

---

## 4. Sector 05 Acceptance Criteria (Definition of Done)

- [ ] The files `src/training/augmentation.py`, `src/training/triplet_sampler.py`, `src/training/losses.py`, and `tests/test_triplets.py` have been created.
- [ ] The command `pytest tests/test_triplets.py` passes 100% (4/4 green tests).
