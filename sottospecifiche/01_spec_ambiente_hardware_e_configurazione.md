# 01 — Sector Sub-Specification: Environment, Hardware Profiling, and Exact YAML Schemas

**Sector:** 01 — Infrastructure, Compute Environment, and Declarative Configuration  
**Parent Spec References:** §2, §13, §14, §15  
**Input Dependencies:** None  
**Enabled Outputs:** All subsequent sectors (`02`–`08`)

---

## 1. Sector Objective

Set up the project directory tree, the core utility modules (`seed.py`, `hardware.py`, `config.py`, `env_check.py`), and the **three complete YAML configuration files** (`configs/data.yaml`, `configs/graph.yaml`, `configs/training.yaml`). The model implementing this sector must write files that strictly conform to the literal schemas provided below.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 1.1 — Determinism Module (`src/utils/seed.py`)
- **File to create:** `src/utils/seed.py`
- **Function contract:**
  ```python
  import os
  import random
  import numpy as np
  import torch

  def seed_everything(seed: int = 42) -> None:
      """Sets the global seed for Python, NumPy, and PyTorch (CPU and CUDA)."""
  ```
- **HOW (Exact steps to implement in the function):**
  1. Set `os.environ["PYTHONHASHSEED"] = str(seed)`.
  2. Call `random.seed(seed)` and `np.random.seed(seed)`.
  3. Call `torch.manual_seed(seed)`.
  4. If `torch.cuda.is_available()` is `True`, call `torch.cuda.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`.
  5. Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- **CONSTRAINT:** Do not use `torch.use_deterministic_algorithms(True)` without `warn_only=True`, because some scatter operations in PyTorch Geometric on CUDA do not have an atomic deterministic implementation and would raise a blocking `RuntimeError`.

---

### Task 1.2 — Hardware Profiling Module (`src/utils/hardware.py`)
- **File to create:** `src/utils/hardware.py`
- **Function contracts:**
  ```python
  from typing import Dict, Any
  import torch

  def get_compute_device(prefer_gpu: bool = True) -> torch.device:
      """Returns torch.device('cuda') if available and working, otherwise torch.device('cpu')."""

  def get_hardware_profile() -> Dict[str, Any]:
      """Returns a JSON-serializable dictionary with CPU/GPU/VRAM/CUDA details."""

  def get_vram_usage_mb() -> Dict[str, float]:
      """Returns {'allocated_mb': ..., 'reserved_mb': ..., 'max_allocated_mb': ...} (0.0 if on CPU)."""
  ```
- **HOW (Exact logic to implement):**
  1. In `get_compute_device(prefer_gpu=True)`:
     - If `prefer_gpu` and `torch.cuda.is_available()`: run a micro-test inside a `try...except Exception` block: create `t = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")`. If it succeeds, return `torch.device("cuda")`; if it fails (e.g., CUDA kernel incompatibility), print a warning and return `torch.device("cpu")`.
     - Otherwise return `torch.device("cpu")`.
  2. In `get_hardware_profile()`:
     - Return a dictionary with the exact keys:
       - `"pytorch_version": str(torch.__version__)`
       - `"cuda_available": bool(torch.cuda.is_available())`
       - `"cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else None`
       - `"device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"`
       - `"total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2) if torch.cuda.is_available() else 0.0`
- **CONSTRAINT:**
  - No other module in the project must ever call `.cuda()`; all modules must use `.to(device)` with the `device` returned by `get_compute_device()`.
  - During training, `max_allocated_mb` must remain $< 1024.0\text{ MB}$.

---

### Task 1.3 — YAML Configuration Loader and Validator (`src/utils/config.py`)
- **Files to create:** `src/utils/config.py`, `configs/data.yaml`, `configs/graph.yaml`, `configs/training.yaml`
- **Function contracts:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, Union
  import yaml

  def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
      """Loads a single YAML file and verifies that it is not empty."""

  def load_all_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, Dict[str, Any]]:
      """
      Loads and validates data.yaml, graph.yaml, and training.yaml.
      Returns {"data": data_cfg, "graph": graph_cfg, "training": train_cfg}.
      """
  ```
- **HOW (EXACT content of the 3 YAML files to write in `configs/`):**

#### A. Exact content of `configs/data.yaml`:
```yaml
paths:
  raw_finefs_dir: "data/raw/finefs"
  raw_mmfs_dir: "data/raw/mmfs"
  intermediate_dir: "data/intermediate"
  processed_dir: "data/processed"
  splits_dir: "data/splits"
  reports_dir: "reports"
  checkpoints_dir: "checkpoints"

dataset:
  primary_dataset: "finefs"
  enable_mmfs: false
  skeleton_convention: "coco17"  # Allowed values: "coco17" or "h36m17" (verified in Task 2.2)
  target_elements:
    - "Camel Spin"
    - "Sit Spin"
    - "Layback Spin"

windowing:
  window_size: 9
  min_valid_frames: 5
  trim_ratio_start: 0.20
  trim_ratio_end: 0.20
  min_keypoint_confidence: 0.20

canonicalization:
  horizontal_axes: [0, 1]       # Horizontal axis indices: 0=X, 1=Y (if Z is vertical)
  vertical_axis: 2              # Vertical axis index: 2=Z (set to 1 if Y is vertical)
  scale_epsilon: 1.0e-4
  yaw_epsilon: 1.0e-4
  # Indices for COCO-17 standard (if h36m17: pelvis=[4, 1], shoulders=[11, 14]):
  pelvis_indices: [11, 12]      # [hip_left, hip_right]
  shoulder_indices: [5, 6]      # [shoulder_left, shoulder_right]

splits:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

#### B. Exact content of `configs/graph.yaml`:
```yaml
num_nodes: 17
node_feature_mode: "baseline"   # "baseline" -> shape (17, 3) | "extended" -> shape (17, 5)
node_feature_dim: 3
edge_feature_dim: 5
epsilon: 1.0e-8

# Standard COCO-17 names:
joint_names:
  0: "nose"
  1: "left_eye"
  2: "right_eye"
  3: "left_ear"
  4: "right_ear"
  5: "left_shoulder"
  6: "right_shoulder"
  7: "left_elbow"
  8: "right_elbow"
  9: "left_wrist"
  10: "right_wrist"
  11: "left_hip"
  12: "right_hip"
  13: "left_knee"
  14: "right_knee"
  15: "left_ankle"
  16: "right_ankle"

# Symmetry indicator for extended mode (-1: left, 0: center, +1: right):
joint_symmetry: [0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]

# 16 undirected anatomical connections (bones b = 0..15).
# Each bone b=[i, j] generates directed edge 2*b=(i->j) and directed edge 2*b+1=(j->i), for a total of 32 directed edges.
undirected_bones:
  - [0, 1]    # Bone 0: nose - left_eye
  - [0, 2]    # Bone 1: nose - right_eye
  - [1, 3]    # Bone 2: left_eye - left_ear
  - [2, 4]    # Bone 3: right_eye - right_ear
  - [0, 5]    # Bone 4: nose - left_shoulder
  - [0, 6]    # Bone 5: nose - right_shoulder
  - [5, 6]    # Bone 6: left_shoulder - right_shoulder
  - [5, 7]    # Bone 7: left_shoulder - left_elbow
  - [7, 9]    # Bone 8: left_elbow - left_wrist
  - [6, 8]    # Bone 9: right_shoulder - right_elbow
  - [8, 10]   # Bone 10: right_elbow - right_wrist
  - [5, 11]   # Bone 11: left_shoulder - left_hip
  - [6, 12]   # Bone 12: right_shoulder - right_hip
  - [11, 12]  # Bone 13: left_hip - right_hip
  - [11, 13]  # Bone 14: left_hip - left_knee
  - [13, 15]  # Bone 15: left_knee - left_ankle
  # Note: if you prefer to include the right leg while keeping 16 bones on a 17-node tree,
  # the standard connected-tree list (17 nodes, 16 edges) without cycles on eyes/shoulders is:
  # [[0,1], [0,2], [1,3], [2,4], [0,5], [0,6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,13], [13,15], [12,14], [14,16]]
```
*(WARNING for the coding model: in `configs/graph.yaml`, use as the default `undirected_bones` list the exact 16-bone connected tree covering all 17 joints including right knee `14` and right ankle `16`:)*
```yaml
undirected_bones:
  - [0, 1]    # Bone 0: nose - left_eye
  - [0, 2]    # Bone 1: nose - right_eye
  - [1, 3]    # Bone 2: left_eye - left_ear
  - [2, 4]    # Bone 3: right_eye - right_ear
  - [0, 5]    # Bone 4: nose - left_shoulder
  - [0, 6]    # Bone 5: nose - right_shoulder
  - [5, 7]    # Bone 6: left_shoulder - left_elbow
  - [7, 9]    # Bone 7: left_elbow - left_wrist
  - [6, 8]    # Bone 8: right_shoulder - right_elbow
  - [8, 10]   # Bone 9: right_elbow - right_wrist
  - [5, 11]   # Bone 10: left_shoulder - left_hip
  - [6, 12]   # Bone 11: right_shoulder - right_hip
  - [11, 13]  # Bone 12: left_hip - left_knee
  - [13, 15]  # Bone 13: left_knee - left_ankle
  - [12, 14]  # Bone 14: right_hip - right_knee
  - [14, 16]  # Bone 15: right_knee - right_ankle

# Deterministic map of the 32 angle triplets [a, b, c] (angle computed at the central vertex b)
# for each directed edge "i->j":
directed_edge_angle_triplets:
  "0->1": [2, 0, 1]
  "1->0": [3, 1, 0]
  "0->2": [1, 0, 2]
  "2->0": [4, 2, 0]
  "1->3": [0, 1, 3]
  "3->1": [0, 1, 3]
  "2->4": [0, 2, 4]
  "4->2": [0, 2, 4]
  "0->5": [6, 0, 5]
  "5->0": [7, 5, 0]
  "0->6": [5, 0, 6]
  "6->0": [8, 6, 0]
  "5->7": [11, 5, 7]
  "7->5": [9, 7, 5]
  "7->9": [5, 7, 9]
  "9->7": [5, 7, 9]
  "6->8": [12, 6, 8]
  "8->6": [10, 8, 6]
  "8->10": [6, 8, 10]
  "10->8": [6, 8, 10]
  "5->11": [7, 5, 11]
  "11->5": [13, 11, 5]
  "6->12": [8, 6, 12]
  "12->6": [14, 12, 6]
  "11->13": [5, 11, 13]
  "13->11": [15, 13, 11]
  "13->15": [11, 13, 15]
  "15->13": [11, 13, 15]
  "12->14": [6, 12, 14]
  "14->12": [16, 14, 12]
  "14->16": [12, 14, 16]
  "16->14": [12, 14, 16]

# Map of descendant kinematic subtrees for Rodrigues rotation (Sector 05 and Sector 08):
# Key "parent->child": list of all nodes in the subtree that rotate around "parent".
kinematic_subtrees:
  "5->7": [7, 9]          # Left Shoulder -> rotates left elbow and left wrist
  "7->9": [9]             # Left Elbow -> rotates left wrist
  "6->8": [8, 10]         # Right Shoulder -> rotates right elbow and right wrist
  "8->10": [10]           # Right Elbow -> rotates right wrist
  "11->13": [13, 15]      # Left Hip -> rotates left knee and left ankle
  "13->15": [15]          # Left Knee -> rotates left ankle
  "12->14": [14, 16]      # Right Hip -> rotates right knee and right ankle
  "14->16": [16]          # Right Knee -> rotates right ankle
```

#### C. Exact content of `configs/training.yaml`:
```yaml
model:
  in_channels: 3
  edge_dim: 5
  hidden_dim: 32
  embed_dim: 32
  heads: 4
  dropout: 0.20
  normalize_embeddings: false

training:
  seed: 42
  batch_size: 32
  lr: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs: 80
  early_stopping_patience: 15
  grad_clip_max_norm: 1.0
  num_train_triplets_per_epoch: 600
  num_val_triplets: 200
  num_test_triplets: 200

sampling:
  tier_rule: "threshold"          # "threshold" (with automatic fallback to "quantile" if pool < 5)
  goe_pos_threshold: 1.5
  goe_neg_threshold: 0.0
  goe_pos_quantile: 0.70
  goe_neg_quantile: 0.35
  max_pos_goe_diff: 1.5
  min_neg_goe_gap: 1.5
  real_negative_prob: 0.50

augmentation:
  enabled: true
  num_synthetic_per_positive: 3   # 1 for each tier (1=mild, 2=medium, 3=severe)
  noise_std: 0.005
  tier_angle_ranges_deg:
    1: [8.0, 15.0]
    2: [16.0, 28.0]
    3: [29.0, 42.0]

loss:
  margin_mode: "adaptive"         # "adaptive" or "fixed"
  fixed_margin: 0.50
  adaptive_margins:
    1: 0.20
    2: 0.60
    3: 1.00
```

- **Validation CONSTRAINT in `load_all_configs()`:**
  - Verify `len(graph_cfg["undirected_bones"]) == 16`.
  - Verify `len(graph_cfg["directed_edge_angle_triplets"]) == 32`.
  - Verify `abs(data_cfg["splits"]["train_ratio"] + data_cfg["splits"]["val_ratio"] + data_cfg["splits"]["test_ratio"] - 1.0) < 1e-6`.
  - If any of these assertions fail, raise `ValueError` with a descriptive message.

---

### Task 1.4 — Environment Smoke Test Script (`src/utils/env_check.py`)
- **File to create:** `src/utils/env_check.py`
- **HOW (Exact steps to execute in the `if __name__ == "__main__":` block):**
  1. Call `seed_everything(42)`.
  2. Call `cfgs = load_all_configs("configs")`.
  3. Call `device = get_compute_device()` and `profile = get_hardware_profile()`.
  4. Save `profile` to `reports/hardware_profile.json` (creating `Path("reports").mkdir(parents=True, exist_ok=True)`).
  5. Create a dummy PyG graph:
     - `x = torch.randn((17, 3), dtype=torch.float32, device=device, requires_grad=True)`
     - `edge_index = torch.randint(0, 17, (2, 32), dtype=torch.long, device=device)`
     - `edge_attr = torch.randn((32, 5), dtype=torch.float32, device=device)`
  6. Instantiate `conv = GATv2Conv(in_channels=3, out_channels=32, heads=4, edge_dim=5, add_self_loops=True).to(device)`.
  7. Run `out, (edge_idx_attn, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)`.
  8. Verify with `assert`:
     - `out.shape == (17, 128)`
     - `edge_idx_attn.shape == (2, 49)` *(32 edges + 17 self-loops)*
     - `alpha.shape == (49, 4)`
  9. Run `out.sum().backward()` and verify `assert x.grad is not None`.
  10. Print `"ENV_CHECK_OK"` and exit with code `0`.

---

## 3. Sector 01 Acceptance Criteria (Definition of Done)

- [ ] All repository folders and `__init__.py` files have been created (`src/`, `src/utils/`, `src/data/`, `src/graph/`, `src/training/`, `src/models/`, `src/evaluation/`, `src/inference/`, `src/visualization/`, `tests/`, `configs/`, `data/raw/`, `data/intermediate/`, `data/processed/`, `data/splits/`, `checkpoints/`, `reports/figures/`).
- [ ] Running `python -m src.utils.env_check` prints `"ENV_CHECK_OK"`, exits with code `0`, and creates the file `reports/hardware_profile.json`.
