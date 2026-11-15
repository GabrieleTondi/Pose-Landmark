# 04 — Sector Sub-Specification: Graph Construction and Feature Engineering

**Sector:** 04 — Graph Representation: 17-Node Canonical Topology, `x`, `pos`, `edge_attr`, and PyG Serialization  
**Parent Spec References:** §6.1–§6.4, §16.3  
**Input Dependencies:** Sector 01 (`configs/graph.yaml`), Sector 03 ($\bar{P} \in \mathbb{R}^{17 \times 3}$)  
**Enabled Outputs:** Sector 05, Sector 06, Sector 07, Sector 08

---

## 1. Sector Objective

Convert each canonicalized pose $\bar{P} \in \mathbb{R}^{17 \times 3}$ into a `torch_geometric.data.Data` object with:
- `num_nodes = 17`;
- `pos`: `(17, 3)` `float32` tensor always containing the canonicalized 3D coordinates $\bar{P}$;
- `x`: `(17, 3)` tensor in `"baseline"` mode (or `(17, 5)` in `"extended"` mode);
- `edge_index`: `(2, 32)` `long` tensor built according to the contiguous paired indexing rule $\{2b, 2b+1\}$;
- `edge_attr`: `(32, 5)` `float32` tensor where each row $m \in \{0, \dots, 31\}$ contains `[u_x, u_y, u_z, bone_length, cos_angle]`.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 4.1 — Canonical Topology and Contiguous Indexing `{2b, 2b+1}` (`src/graph/topology.py`)
- **File to create:** `src/graph/topology.py`
- **Exact function contracts:**
  ```python
  from typing import Dict, Any, List, Tuple
  import torch

  def get_directed_edge_list(graph_cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
      """
      Reads the 16 undirected bones from graph_cfg['undirected_bones'].
      For each bone b = (i, j) with b in 0..15, generates in order:
      - index 2*b:   (i, j)
      - index 2*b+1: (j, i)
      Returns a list of 32 (src, dst) tuples.
      """

  def build_canonical_edge_index(graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Builds the edge_index tensor of shape (2, 32) and dtype torch.long
      from get_directed_edge_list(graph_cfg).
      """
  ```
- **HOW to implement `get_directed_edge_list` and `build_canonical_edge_index`:**
  ```python
  bones = graph_cfg["undirected_bones"]
  assert len(bones) == 16, f"Expected 16 bones, found {len(bones)}"
  directed_edges: List[Tuple[int, int]] = []
  for i, j in bones:
      directed_edges.append((int(i), int(j)))  # index 2*b
      directed_edges.append((int(j), int(i)))  # index 2*b + 1
  # In build_canonical_edge_index:
  src = [e[0] for e in directed_edges]
  dst = [e[1] for e in directed_edges]
  return torch.tensor([src, dst], dtype=torch.long)
  ```
- **CONSTRAINT:** Never change the edge order and never add self-loops in `build_canonical_edge_index` (self-loops are added internally by the `GATv2Conv(..., add_self_loops=True)` layer).

---

### Task 4.2 — Computation of `x` and `edge_attr` (`src/graph/features.py`)
- **File to create:** `src/graph/features.py`
- **Exact function contracts:**
  ```python
  from typing import Dict, Any
  import numpy as np
  import torch
  from src.graph.topology import get_directed_edge_list

  def compute_node_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Computes node features from P_bar of shape (17, 3).
      - If graph_cfg['node_feature_mode'] == 'baseline': returns Tensor (17, 3) float32 [x, y, z].
      - If graph_cfg['node_feature_mode'] == 'extended': returns Tensor (17, 5) float32 [x, y, z, r_i, sym_i].
      """

  def compute_edge_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Computes features for the 32 directed edges, returning a Tensor of shape (32, 5) float32:
      columns 0..2: unit direction vector u_ij = (p_j - p_i) / (||p_j - p_i||_2 + eps)
      column 3:     bone length l_ij = ||p_j - p_i||_2
      column 4:     cosine of the joint angle cos(theta_abc) associated with edge (i->j)
      """
  ```
- **HOW to implement `compute_edge_features(P_bar, graph_cfg)` (Exact algorithm):**
  1. Verify `assert P_bar.shape == (17, 3) and np.isfinite(P_bar).all()`.
  2. Read `eps = float(graph_cfg.get("epsilon", 1e-8))`, `directed_edges = get_directed_edge_list(graph_cfg)`, and `triplets_map = graph_cfg["directed_edge_angle_triplets"]`.
  3. Allocate `edge_attr = np.zeros((32, 5), dtype=np.float32)`.
  4. For `m, (i, j)` in `enumerate(directed_edges)`:
     - Compute the bone vector and its length:
       ```python
       diff = P_bar[j] - P_bar[i]
       length = float(np.linalg.norm(diff))
       u_ij = diff / (length + eps)
       ```
     - Retrieve the angle triplet `a, b, c = triplets_map[f"{i}->{j}"]` (where `b` is the angle vertex):
       ```python
       v1 = P_bar[a] - P_bar[b]
       v2 = P_bar[c] - P_bar[b]
       denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
       cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
       ```
     - Write row `m`:
       ```python
       edge_attr[m, 0:3] = u_ij
       edge_attr[m, 3] = length
       edge_attr[m, 4] = cos_theta
       ```
  5. Convert to `torch.from_numpy(edge_attr)` and verify `assert torch.isfinite(tensor).all()`.

---

### Task 4.3 — PyTorch Geometric `Data` Builder and Serializer (`src/graph/builder.py`)
- **File to create:** `src/graph/builder.py`
- **Exact function contracts:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List
  import numpy as np
  import torch
  from torch_geometric.data import Data
  from src.data.harmonization import UnifiedSample
  from src.graph.topology import build_canonical_edge_index
  from src.graph.features import compute_node_features, compute_edge_features

  def build_pyg_graph(
      P_bar: np.ndarray,
      sample_meta: Dict[str, Any],
      graph_cfg: Dict[str, Any]
  ) -> Data:
      """
      Builds and validates a complete torch_geometric.data.Data object.
      """

  def process_and_save_dataset_graphs(
      samples: List[UnifiedSample],
      split_dict: Dict[str, List[str]],
      data_cfg: Dict[str, Any],
      graph_cfg: Dict[str, Any]
  ) -> Dict[str, List[Data]]:
      """
      Runs extract_stable_window -> canonicalize_window_3d -> build_pyg_graph
      for all samples, saves train_graphs.pt, val_graphs.pt, test_graphs.pt
      in data/processed/ and returns {'train': [...], 'val': [...], 'test': [...]}.
      """
  ```
- **HOW to implement `build_pyg_graph(P_bar, sample_meta, graph_cfg)`:**
  Build the `Data` object with the following **exact** attributes:
  ```python
  x = compute_node_features(P_bar, graph_cfg)          # (17, 3) or (17, 5) float32
  pos = torch.tensor(P_bar, dtype=torch.float32)       # ALWAYS (17, 3) float32
  edge_index = build_canonical_edge_index(graph_cfg)   # (2, 32) long
  edge_attr = compute_edge_features(P_bar, graph_cfg)  # (32, 5) float32
  goe_val = float(sample_meta.get("goe", 0.0))
  y = torch.tensor([goe_val], dtype=torch.float32)     # (1,) float32

  data = Data(
      x=x,
      pos=pos,
      edge_index=edge_index,
      edge_attr=edge_attr,
      y=y,
      sample_id=str(sample_meta["sample_id"]),
      dataset_id=str(sample_meta.get("dataset_id", "finefs")),
      performance_id=str(sample_meta["performance_id"]),
      athlete_id=str(sample_meta.get("athlete_id") or "unknown"),
      element_label=str(sample_meta["element_label"]),
      goe=goe_val,
      split=str(sample_meta.get("split", "train")),
      is_augmented=bool(sample_meta.get("is_augmented", False)),
      parent_sample_id=str(sample_meta.get("parent_sample_id") or sample_meta["sample_id"]),
      severity_tier=int(sample_meta.get("severity_tier", 0)),
      perturbation_joint=int(sample_meta.get("perturbation_joint", -1)),
      perturbation_angle_deg=float(sample_meta.get("perturbation_angle_deg", 0.0))
  )
  return data
  ```
- **CONSTRAINT (Important note for PyTorch 2.6+):**
  When reloading `.pt` files containing `torch_geometric.data.Data` objects via `torch.load()`, always pass `weights_only=False` (`torch.load(path, map_location="cpu", weights_only=False)`) to avoid the `UnpicklingError` introduced in recent PyTorch versions on custom PyG classes.

---

## 3. Unit Tests to Implement (`tests/test_graph.py`)

The file `tests/test_graph.py` must implement the following 4 automated tests:
1. `test_topology_paired_ordering()`: verifies that `edge_index` has shape `(2, 32)` and that for every $b \in \{0, \dots, 15\}$, `edge_index[0, 2*b] == edge_index[1, 2*b+1]` and `edge_index[1, 2*b] == edge_index[0, 2*b+1]` hold.
2. `test_edge_features_geometry()`: verifies that on a graph built from a canonical pose, `edge_attr` has shape `(32, 5)`, `torch.norm(edge_attr[:, 0:3], dim=-1)` is `1.0` ($\pm 10^{-4}$), `edge_attr[2*b, 0:3] == -edge_attr[2*b+1, 0:3]`, `edge_attr[:, 3] > 0`, and `edge_attr[:, 4]` is in `[-1.0, 1.0]`.
3. `test_pyg_data_and_batching()`: builds 8 graphs with `build_pyg_graph`, passes them to `torch_geometric.loader.DataLoader(graphs, batch_size=4)`, and verifies that the first batch has `batch.x.shape == (68, 3)`, `batch.pos.shape == (68, 3)`, `batch.edge_index.shape == (2, 128)`, and `batch.edge_attr.shape == (128, 5)`.
4. `test_extended_node_features_preserves_pos()`: sets `node_feature_mode = "extended"` and verifies that `data.x.shape == (17, 5)` while `data.pos.shape == (17, 3)` remains unchanged.

---

## 4. Sector 04 Acceptance Criteria (Definition of Done)

- [ ] The files `src/graph/topology.py`, `src/graph/features.py`, `src/graph/builder.py`, and `tests/test_graph.py` have been created.
- [ ] Running `pytest tests/test_graph.py` passes 100% (4/4 green tests).
