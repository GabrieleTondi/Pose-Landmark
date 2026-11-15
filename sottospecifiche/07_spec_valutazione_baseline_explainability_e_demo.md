# 07 — Sector Sub-Specification: Evaluation, Baselines, Explainability, and Demo

**Sector:** 07 — Experimental Evaluation, Non-Neural Geometric Baselines, Index $Q$ Calibration, Explainability (XAI), and Demo  
**Parent Specification References:** §9, §10, §11, §12, §16.5  
**Input Dependencies:** Sector 04 (PyG Graphs), Sector 06 (`SiameseGATv2`, `checkpoints/best_model.pt`)  
**Enabled Outputs:** Sector 08 (3D Streamlit Dashboard and Vector Report Figures)

---

## 1. Sector Objective

Implement:
1. The **3 Non-Neural Geometric Baselines** (`src/evaluation/baselines.py`);
2. The computation of **Per-Class Reference Prototypes** (Latent Centroid $z_R^{(c)}$ + Medoid Graph $G_R^{(c)}$), calibration of parameter $\beta$ for the index $Q = 100\exp(-\beta d)$ on the validation set, and symmetrized extraction of attention weights across the 16 bones (`src/evaluation/metrics.py`);
3. The complete comparative evaluation script on the test set (`src/evaluation/evaluate.py`);
4. The single-sample inference script (`src/inference/predict.py`).

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 7.1 — The 3 Non-Neural Geometric Baselines (`src/evaluation/baselines.py`)
- **File to create:** `src/evaluation/baselines.py`
- **Exact function contracts:**
  ```python
  import torch
  import torch.nn.functional as F
  from torch_geometric.data import Data

  def coord_euclidean_distance(g1: Data, g2: Data) -> float:
      """Baseline 1: Frobenius Euclidean distance between canonicalized 3D coordinates g1.pos and g2.pos (17, 3)."""
      diff = g1.pos.cpu().float() - g2.pos.cpu().float()
      return float(torch.norm(diff, p="fro").item())

  def angle_euclidean_distance(g1: Data, g2: Data) -> float:
      """Baseline 2: L2 Euclidean distance between joint angle cosines (5th column of edge_attr, index 4)."""
      c1 = g1.edge_attr[:, 4].cpu().float()
      c2 = g2.edge_attr[:, 4].cpu().float()
      return float(torch.norm(c1 - c2, p=2).item())

  def feature_cosine_distance(g1: Data, g2: Data) -> float:
      """Baseline 3: Cosine distance 1 - cos(f1, f2) on the concatenated vector [vec(pos) || vec(edge_attr)]."""
      f1 = torch.cat([g1.pos.cpu().float().flatten(), g1.edge_attr.cpu().float().flatten()], dim=0)
      f2 = torch.cat([g2.pos.cpu().float().flatten(), g2.edge_attr.cpu().float().flatten()], dim=0)
      cos_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=-1, eps=1e-8).item()
      return float(1.0 - cos_sim)
  ```

---

### Task 7.2 — Prototypes $z_R^{(c)}$, Medoids $G_R^{(c)}$, $\beta$ Calibration, and Attention Extraction (`src/evaluation/metrics.py`)
- **File to create:** `src/evaluation/metrics.py`
- **Exact function contracts:**
  ```python
  from typing import Dict, Any, List, Union
  import numpy as np
  from scipy.stats import spearmanr
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  @torch.no_grad()
  def compute_class_references(
      train_graphs: List[Data],
      model: SiameseGATv2,
      goe_thresholds: Dict[str, tuple],
      device: torch.device
  ) -> Dict[str, Dict[str, Any]]:
      """
      For each technical element c present in train_graphs, filters real high-quality samples
      (not g.is_augmented and g.goe >= goe_thresholds[c][1]) and computes:
      - 'centroid_z': Tensor of shape (embed_dim,) on CPU (mean of z embeddings)
      - 'medoid_graph': real Data object in R_c^train whose embedding z is closest to 'centroid_z'
      - 'mean_pos': Tensor (17, 3) mean of pos coordinates
      - 'mean_edge_attr': Tensor (32, 5) mean of edge_attr
      - 'num_ref_samples': int
      """

  def calibrate_similarity_beta(val_distances: np.ndarray) -> float:
      """
      Calibrates beta > 0 exclusively on prototype distances from the validation set,
      enforcing that the median validation distance corresponds to Q = 50.0:
      beta = ln(2.0) / (median(val_distances) + 1e-8).
      """

  def distance_to_similarity_index(
      d: Union[float, np.ndarray],
      beta: float
  ) -> Union[float, np.ndarray]:
      """
      Returns the Internal Similarity Index Q = 100.0 * exp(-beta * d) in [0, 100].
      """

  @torch.no_grad()
  def extract_bone_attention_weights(
      model: SiameseGATv2,
      graph: Data,
      layer_name: str = "layer2",
      device: torch.device = torch.device("cpu")
  ) -> Dict[str, np.ndarray]:
      """
      Runs model.encode(graph, return_attention_weights=True) for a single graph (17 nodes, 32 edges).
      Returns:
      - 'bone_weights_norm': np.ndarray of shape (16,) normalized to sum to 1.0 across the 16 undirected bones
      - 'bone_weights_raw': np.ndarray of shape (16,) mean of the 4 heads on each pair {2b, 2b+1}
      - 'head_matrix': np.ndarray of shape (16, 4) with weights per bone and per individual head
      """
  ```
- **HOW to implement `extract_bone_attention_weights` (Bug-free index code):**
  ```python
  model.eval()
  g = graph.clone().to(device)
  _, attn_dict = model.encode(g, return_attention_weights=True)
  edge_idx_attn, alpha = attn_dict[layer_name]
  # In PyG GATv2Conv(add_self_loops=True), the first 32 edges are exactly the 32 edges passed in g.edge_index
  # and the last 17 are self-loops (i -> i). We verify or filter with a mask:
  non_self_mask = edge_idx_attn[0] != edge_idx_attn[1]
  alpha_directed = alpha[non_self_mask].detach().cpu().numpy()  # shape (32, H) where H=4
  assert alpha_directed.shape[0] == 32, f"Expected 32 directed edges, found {alpha_directed.shape[0]}"

  # Since in Sector 04 bone b (b=0..15) generates contiguous edges 2*b and 2*b+1:
  # Reshape to (16, 2, H) and average over the 2 opposite directions (i->j) and (j->i):
  head_matrix = alpha_directed.reshape(16, 2, -1).mean(axis=1)  # shape (16, 4)
  bone_weights_raw = head_matrix.mean(axis=-1)                  # shape (16,)
  bone_weights_norm = bone_weights_raw / (np.sum(bone_weights_raw) + 1e-8)
  return {
      "bone_weights_norm": bone_weights_norm.astype(np.float32),
      "bone_weights_raw": bone_weights_raw.astype(np.float32),
      "head_matrix": head_matrix.astype(np.float32)
  }
  ```

---

### Task 7.3 — Complete Evaluation vs Baselines (`src/evaluation/evaluate.py`)
- **File to create:** `src/evaluation/evaluate.py`
- **Main function contract:**
  ```python
  from typing import Dict, Any, List
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  def evaluate_all_methods(
      model: SiameseGATv2,
      train_graphs: List[Data],
      val_graphs: List[Data],
      test_graphs: List[Data],
      goe_thresholds: Dict[str, tuple],
      train_cfg: Dict[str, Any],
      device: torch.device
  ) -> Dict[str, Any]:
      """
      Computes and returns a structured dictionary saved to reports/evaluation_results.json with:
      1. 'triplet_accuracy': {'siamese_gatv2': ..., 'baseline_coord': ..., 'baseline_angle': ..., 'baseline_cosine': ...}
      2. 'spearman_overall': {'siamese_gatv2': {'rho': ..., 'p_val': ...}, 'baseline_coord': ..., ...}
      3. 'spearman_by_element': {element_label: {...}}
      4. 'beta_calibrated': float
      5. 'test_samples_records': list of dicts for each test sample with
         {'sample_id', 'element_label', 'goe', 'd_gatv2', 'Q_gatv2', 'd_coord', 'd_angle', 'd_cos'}
      """
  ```
- **HOW to compute Spearman $\rho$ correctly:**
  - For each test sample $G_i$ of class $c$, the distance from the reference for Siamese GATv2 is $d_i = \| f_\theta(G_i) - z_R^{(c)} \|_2$, whereas for the 3 baselines it is the distance relative to the reference medoid/centroid graph of class $c$.
  - Since a small distance must correspond to a high GOE, call `rho, p_val = spearmanr(-distances_array, goe_array)`. If `np.isnan(rho)` (degenerate case with constant values), set `rho = 0.0, p_val = 1.0`.

---

### Task 7.4 — Single-Sample Inference (`src/inference/predict.py`)
- **File to create:** `src/inference/predict.py`
- **Function contract:**
  ```python
  from typing import Dict, Any
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  @torch.no_grad()
  def predict_single_sample(
      graph: Data,
      model: SiameseGATv2,
      class_references: Dict[str, Dict[str, Any]],
      beta: float,
      device: torch.device = torch.device("cpu")
  ) -> Dict[str, Any]:
      """
      Given an athlete's PyG graph, computes:
      - 'sample_id': graph.sample_id
      - 'element_label': graph.element_label
      - 'true_goe': float(graph.goe)
      - 'distance_to_ref': float d(A, R)
      - 'similarity_index_Q': float Q in [0, 100]
      - 'reference_sample_id': medoid_graph.sample_id
      - 'athlete_attention': dict returned by extract_bone_attention_weights(model, graph)
      - 'reference_attention': dict returned by extract_bone_attention_weights(model, medoid_graph)
      - 'bone_angle_diff_deg': np.ndarray of shape (16,) with the angular difference in degrees across the 16 bones
      """
  ```

---

## 3. Sector 07 Acceptance Criteria (Definition of Done)

- [ ] The files `src/evaluation/baselines.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`, and `src/inference/predict.py` have been created.
- [ ] `extract_bone_attention_weights` returns finite `(16,)` vectors with `bone_weights_norm` summing to $1.0 \pm 10^{-5}$.
- [ ] `evaluate_all_methods` generates `reports/evaluation_results.json` containing the comparison between `siamese_gatv2` and the 3 geometric baselines.
