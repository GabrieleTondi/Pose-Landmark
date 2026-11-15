# Project Operational Diary — Siamese GATv2 for Action Quality Assessment (Figure Skating)

> **Purpose of this document:** To describe in a detailed, accessible, and technically faithful way **everything that has been designed, implemented, fixed, and verified** from the beginning of the project up to the current state.  
> This file is maintained and progressively updated at the end of each Phase planned in [specs/09_piano_implementativo_operativo.md](specs/09_piano_implementativo_operativo.md).

---

## Work Progress Status (Roadmap)

| Phase | Main Content | Status | Verification Gate |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Infrastructure, YAML Configs, Segmentation, 3D Canonicalization, and PyG Graph Construction | ✅ Completed | `python -m src.utils.env_check` → `ENV_CHECK_OK`; `test_config_and_utils.py`, `test_normalization.py`, `test_graph.py` tests passed |
| **Phase 2** | `UnifiedSample` Schema, Real FineFS Audit & Loader (`H36M-17` + ISU GOE), Anti-Leakage Splits, and Graph Materialization | ✅ Completed | `1583` real spins loaded and transformed into PyG graphs (`0` discarded); `pytest tests/ -v` → **`22 passed`** |
| **Phase 3 (Run v1 — Baseline)** | Rodrigues Rotation Augmentation, Triplet Sampler, Adaptive Margin Loss, `GATv2` Encoder, and Baseline Training | ✅ Archived | `pytest tests/test_triplets.py -v` (`8 passed`); real CUDA training on FineFS (`62` epochs, best epoch `47`, `val_triplet_acc = 0.6500`, `val_margin_acc = 0.3750`, `embedding_std = 0.289`) + `pytest tests/ -v` → **`30 passed`** |
| **Spec 10 Revision & Run v2** | Configurable Bilateral Mirroring, Sliding Window Modes, `temporal_extended` Features, Ablations on FineFS, and Calibrated Training v2 | ✅ Completed | [specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md](specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md); `pytest tests/ -v` → **`31 passed`**; Training Run v2 (`80` epochs, best epoch `62`, **`val_triplet_acc = 0.6600`**, **`val_margin_acc = 0.4550`**, **`embedding_std = 0.3434`**) |
| **Phase 4** | Non-Neural Geometric Baselines, AQA Metrics, $Q$ Index Calibration, Evaluation, and Inference | ⏳ Next | Evaluation on `test_graphs.pt` → `reports/evaluation_results.json` |
| **Phase 5** | 3D/2D Visualization, Report Figure Export, and Interactive Streamlit Dashboard | ⏳ To do | `pytest tests/ -v` + `python -m src.visualization.export_report_figures` |

---

## 0. Core Project Idea (Why We Perform These Steps)

The goal of the system is to evaluate the **execution quality (Action Quality Assessment, AQA)** of spins in artistic figure skating (specifically **Camel Spin**, **Sit Spin**, and **Layback Spin**) starting from 3D human skeleton poses (`17` joints in space).

Instead of limiting ourselves to a "black box" that outputs a numerical score on a single raw video:
1. We want to represent each spin as a **clean, spatially aligned average 3D pose** (independent of where the skater is on the rink, their height, and which direction they are facing while spinning in that frame).
2. We transform that pose into an **anatomical graph** (`17` nodes = joints, `32` directed edges = bones in both directions) enriched with 3D directions, bone lengths, and joint angles.
3. In the second half of the project (Phases 3–5), a **Siamese GATv2** neural network will learn to compare a spin performed by an athlete against excellent reference spins and gradual biomechanical errors, producing both a calibrated quality index and a visual explanation of *which limbs and joints* caused any penalties via edge attention weights.

**Important:** At the end of **Phase 2**, **no trained neural network exists yet**. We have completed the entire pipeline that takes raw dataset files all the way to PyTorch Geometric graph files ready to enter the model.

---

## 1. Phase 1 — Infrastructure, 3D Geometric Cleaning, and Graph Construction

### 1.1 Python Environment, CUDA GPU, Dependencies, and Repository Management
We configured the environment on the system Python 3.13.3 interpreter, installing the libraries pinned in [requirements.txt](requirements.txt):
- **Numerical & data core:** `numpy==2.1.3`, `scipy==1.16.2`, `scikit-learn==1.7.2`, `pandas==3.1.1`, `PyYAML==6.0.2`.
- **Deep Learning & Graphs:** `torch==2.8.0+cu126`, `torch-geometric==2.8.1` (with CUDA 12.6 support).
- **Testing & Visualization:** `pytest==9.2.0`, `matplotlib==3.11.2`, `plotly==7.1.0`, `streamlit==1.67.0`.

We also configured:
- [pyproject.toml](pyproject.toml): sets the `figure-skating-aqa-gatv2` package metadata, adds the root to `pythonpath` for `pytest`, and uses `addopts = "-p no:cacheprovider"` (so tests can run cleanly even in Windows environments with restrictive cache write permissions).
- [.gitignore](.gitignore): excludes heavy and binary files from version control (`.venv/`, `__pycache__/`, `*.pt`, `*.pth`, `data/raw/`, `data/intermediate/`, `data/processed/`, `checkpoints/*.pt`) while keeping `.gitkeep` placeholder files.

#### Utility Modules (`src/utils/`)
1. **[src/utils/seed.py](src/utils/seed.py) — `seed_everything(seed=42)`**  
   Simultaneously sets `PYTHONHASHSEED`, `random`, `numpy.random`, `torch.manual_seed` (CPU and CUDA) and configures `torch.backends.cudnn.deterministic = True` (with `benchmark = False`) to make splits, weights, and sampling reproducible.
2. **[src/utils/hardware.py](src/utils/hardware.py) — `get_compute_device`, `get_hardware_profile`, `get_vram_usage_mb`**  
   Detects the GPU and **before trusting it** runs an actual micro-computation on the GPU (`2x2 @ 2x2` on `cuda`), automatically falling back to CPU with a warning only if CUDA genuinely fails.
3. **[src/utils/config.py](src/utils/config.py) — `load_yaml_config`, `load_all_configs`**  
   Loads the three files in `configs/` (`data.yaml`, `graph.yaml`, `training.yaml`) and immediately validates critical invariants:
   - `graph.yaml` must contain exactly `16` entries in `undirected_bones` and `32` entries in `directed_edge_angle_triplets`;
   - `data.yaml` must have `train_ratio + val_ratio + test_ratio = 1.0` (tolerance `1e-6`).
4. **[src/utils/env_check.py](src/utils/env_check.py)**  
   Runs a complete *smoke test*: sets the seed, loads configs, selects the device, saves [reports/hardware_profile.json](reports/hardware_profile.json), and tests a real `GATv2Conv(in_channels=3, out_channels=32, heads=4, edge_dim=5, add_self_loops=True)` layer, verifying that:
   - the output on the `17` nodes has shape `(17, 128)` (`32 * 4` heads);
   - the attention indices and weights have shapes `(2, 49)` and `(49, 4)` (`32` real edges + `17` self-loops added internally by `GATv2Conv`);
   - backpropagation (`out.sum().backward()`) produces non-zero gradients with respect to the input `x`.
   - **Observed result on the local machine** ([reports/hardware_profile.json](reports/hardware_profile.json)):
     - PyTorch `2.8.0+cu126`, CUDA `12.6` available (`true`)
     - Detected GPU: **`NVIDIA GeForce RTX 3050 Laptop GPU`** (`4095.5 MB` of VRAM)
     - `selected_compute_device`: `"cuda"`, output `ENV_CHECK_OK`.

---

### 1.2 The Three Centralized Configuration Files (`configs/`)
All pipeline math is not hardcoded inside scripts, but read from three YAML files:
- **[configs/data.yaml](configs/data.yaml):** defines paths (`data/raw/finefs`, `data/intermediate`, `data/processed`, `data/splits`, `reports`, `checkpoints`), skeleton convention (`h36m17`), target classes (`Camel Spin`, `Sit Spin`, `Layback Spin`), temporal window parameters (`window_size: 9`, `min_valid_frames: 5`, start/end trim `0.20`, minimum confidence `0.20`), indices and axes for 3D canonicalization, and split ratios (`0.70 / 0.15 / 0.15`, `seed: 42`).
- **[configs/graph.yaml](configs/graph.yaml):** defines the `17` nodes, node feature mode (`node_feature_mode: "baseline"` → `3` features `[x, y, z]`; also supports `"extended"` → `5` features), `edge_feature_dim: 5`, names and anatomical symmetry of each joint, the `16` undirected bones, the `32` triplets `[a, b, c]` for joint angles, and the `9` kinematic subtrees for future Rodrigues rotations.
- **[configs/training.yaml](configs/training.yaml):** contains parameters ready for Phase 3 (`GATv2` `hidden_dim: 32`, `embed_dim: 32`, `heads: 4`, `dropout: 0.20`, `batch_size: 32`, `lr: 1e-3`, `max_epochs: 80`, GOE thresholds `>= 1.5` positives and `<= 0.0` negatives, adaptive margins `0.20 / 0.60 / 1.00`, etc.).

---

### 1.3 Temporal Slicing of the Stable Window (`src/data/segmentation.py`)
An annotated competition spin can last 10–25 seconds (hundreds of frames at 25 fps). In the first few seconds the skater is still entering the spin, and in the last few seconds they are already exiting the position.

The **`extract_stable_window(keypoints, confidence, data_cfg)`** function in [src/data/segmentation.py](src/data/segmentation.py) does the following:
1. Checks that the input sequence has shape `(T_seg, 17, 3)` and at least `min_valid_frames` (`5`) total frames; otherwise discards the segment (`segment_too_short`).
2. Trims off the first `20%` (`trim_ratio_start: 0.20`) and the last `20%` (`trim_ratio_end: 0.20`), focusing on the central `60%`. *(If the segment is so short that fewer than 5 frames would remain after trimming, it uses the entire `[0, T_seg)` interval as a fallback).*
3. Filters frame by frame, keeping only valid frames where:
   - all coordinates are finite (no `NaN`/`Inf`);
   - the frame standard deviation is `> 1e-5` (discards skeletons collapsed to a single point);
   - if the `confidence` array is present, the average confidence across the 17 joints is `>= 0.20` (`min_keypoint_confidence`).
4. If at least `5` valid frames remain, it takes the **central** sub-window of maximum length `window_size = 9` (`T_win <= 9`, corresponding to about 0.36 seconds at 25 fps: short enough to capture a stable pose, long enough to smooth out frame-by-frame tracking noise).

---

### 1.4 3D Geometric Canonicalization (`src/data/preprocessing.py`)
Raw athlete coordinates cannot enter the graph directly: they depend on the position on the rink, the person's height (or camera/reconstruction scale), and the rotation angle around the vertical axis during the spin.

To eliminate these nuisance factors, **`canonicalize_frame_3d(joints, data_cfg)`** applies three exact mathematical transformations to each individual frame `(17, 3)`:
1. **Root-Centering (Translation Invariance):**  
   Computes the pelvis center `pelvis = 0.5 * (pts[hip_l] + pts[hip_r])` (in H36M-17: joints `4` and `1`) and subtracts it from all 17 joints. The pelvis becomes the origin `(0, 0, 0)`.
2. **Scale Normalization (Height / Scale Invariance):**  
   Computes the shoulder midpoint `shoulders_mid = 0.5 * (centered[sh_l] + centered[sh_r])` (in H36M-17: joints `11` and `14`) and measures the 3D torso length `torso_length = ||shoulders_mid||_2`. If it is below `scale_epsilon` (`1e-4`), raises `DegenerateSkeletonError`; otherwise divides all coordinates by `torso_length`. From this point on, all distances are expressed in **"torso units"**.
3. **Yaw Alignment (Invariance to Horizontal Rotation on Ice):**  
   Computes the horizontal hip vector in the `(X, Y)` plane (`hip_vec = scaled[hip_r] - scaled[hip_l]`; if near zero, i.e., `<= yaw_epsilon`, falls back to the shoulder vector `sh_vec`; if that is also zero, raises `DegenerateSkeletonError`). Computes the angle `angle = arctan2(vy, vx)` and applies a planar rotation of `-angle` on axes `[0, 1]` (keeping the vertical axis `2 = Z` intact). Thus, every frame of the spin is rotated to the same local coordinate system with the hips aligned along the `+X` axis.

Next, **`canonicalize_window_3d(window_joints, data_cfg)`**:
- runs `canonicalize_frame_3d` on each of the `T_win` frames in the window (tolerating a few degenerate frames as long as at least `min_valid_frames` valid frames remain);
- computes the **average canonical pose** `P_bar = mean(aligned_frames, axis=0)` of shape `(17, 3)` in `float32`, along with diagnostic statistics (`num_valid_frames`, `mean_torso_length`, `window_jitter` = RMS dispersion between aligned frames and `P_bar`).

---

### 1.5 PyTorch Geometric Graph Construction (`src/graph/`)
Once `P_bar` `(17, 3)` is obtained, the `src/graph/` package transforms it into a `torch_geometric.data.Data` object:

1. **Pairwise Ordered Topology — [src/graph/topology.py](src/graph/topology.py)**  
   - `get_directed_edge_list()` reads the `16` bones in `configs/graph.yaml` and for each bone `b = (i, j)` creates in deterministic order the edge `2*b: (i -> j)` followed immediately by the reverse edge `2*b + 1: (j -> i)`, obtaining `32` directed edges.
   - `build_canonical_edge_index()` builds the `edge_index` tensor of shape `(2, 32)` (`torch.long`). It does not insert manual self-loops because they are handled internally by `GATv2Conv(..., add_self_loops=True)`.
   - This pairwise ordering `2*b` and `2*b+1` will be used in Phases 4–5 to easily aggregate forward and backward attention on each of the 16 anatomical segments.

2. **Node and Edge Features — [src/graph/features.py](src/graph/features.py)**  
   - `compute_node_features(P_bar, graph_cfg)`:
     - in `"baseline"` mode (currently active): returns `(17, 3)` `float32` with canonical `[x, y, z]`;
     - in `"extended"` mode: returns `(17, 5)` adding the radius from the pelvis `||p_i||_2` and the symmetry indicator `joint_symmetry` (`-1` left, `0` center, `+1` right).
   - `compute_edge_features(P_bar, graph_cfg)`: returns a `(32, 5)` `float32` tensor. For each edge `(i -> j)`:
     - columns `0..2`: unit directional vector `u_ij = (p_j - p_i) / (||p_j - p_i||_2 + eps)`;
     - column `3`: bone length `l_ij = ||p_j - p_i||_2` (in torso units);
     - column `4`: cosine of the joint angle $\cos(\theta_{abc})$ computed on the triplet `[a, b, c]` defined for that edge in `directed_edge_angle_triplets` (using `v1 = p_a - p_b`, `v2 = p_c - p_b`, clamped with `np.clip(..., -1.0, 1.0)`).

3. **Assembly and Serialization — [src/graph/builder.py](src/graph/builder.py)**  
   - `build_pyg_graph(P_bar, sample_meta, graph_cfg)`: builds the `Data(x, pos, edge_index, edge_attr, y, ...)` object, always preserving `data.pos` `(17, 3)` (even if `x` is in `extended` `(17, 5)` mode), the target `y = [goe]`, and all metadata (`sample_id`, `dataset_id`, `performance_id`, `athlete_id`, `element_label`, `goe`, `split`, `is_augmented`, `parent_sample_id`, `severity_tier`, `perturbation_joint`, `perturbation_angle_deg`).
   - `process_and_save_dataset_graphs(samples, split_dict, data_cfg, graph_cfg)`: runs the entire chain `extract_stable_window → canonicalize_window_3d → build_pyg_graph` on all samples split into `train`, `val`, `test` and saves `train_graphs.pt`, `val_graphs.pt`, and `test_graphs.pt` in `data/processed/`.

---

## 2. Phase 2 — Harmonization, Real FineFS Audit, Loader, and Anti-Leakage Splits

### 2.1 Unified Data Schema (`src/data/harmonization.py`)
In [src/data/harmonization.py](src/data/harmonization.py) we defined the **`UnifiedSample`** dataclass, which represents a single extracted technical element in a source-independent way (real FineFS or synthetic mock generator):
- identifiers: `sample_id`, `dataset_id`, `performance_id`, `athlete_id` (`Optional[str]`), `element_label`;
- window in the source clip: `start_frame`, `end_frame` (with `end_frame >= start_frame`);
- 3D data: `keypoints` `(T_seg, 17, 3)` `float32`, `keypoint_confidence` `(T_seg, 17)` `float32` or `None`;
- quality label and traceability: `goe` (`float` finite), `source_metadata` (`dict`).

The `validate()` method automatically checks all shapes, types, and finiteness before saving (`save_unified_samples`) or after reloading (`load_unified_samples`) the pickle file.

### 2.2 Acquisition and Inspection of the Real FineFS Dataset (`data/raw/finefs/`)
For the project, only the **annotation** and **3D skeleton** files were needed (avoiding downloading tens of gigabytes of unnecessary raw RGB videos):
1. The annotation archives (`1167` `.json` files) were downloaded and placed in `data/raw/finefs/annotation/` (`0.json` … `1166.json`).
2. The skeleton archive (`skeleton.zip`, ~828 MB) had an anti-bot/CAPTCHA block on Google Drive; in compliance with security rules, we did not bypass it, but stopped the automated download and the user manually downloaded and extracted the `1167` `.npz` files into `data/raw/finefs/skeleton/` (`0.npz` … `1166.npz`).

#### The 4 Critical Findings from the Real Audit on the 1167 Files
When we actually inspected the extracted files via [src/data/loaders.py](src/data/loaders.py) (`inspect_npz_file`, `audit_finefs_directory`), we discovered four essential aspects that guided the implementation:

1. **Perfect integrity of `.npz` files:**  
   All `1167` `.npz` files in `skeleton/` exist, open without errors, contain the array under the `"reconstruction"` key with shape `(T, 17, 3)` in `float32`, **zero `NaN`/`Inf`**, and non-trivial real 3D coordinates (`num_3d_nonzero_npz = 1167`). The only quirk to handle during reading was the `"reconstruction"` key name and possible shape variations (`extract_3d_keypoints_from_array` automatically handles `(T,17,3)`, `(T,17,4)`, `(M,T,17,3/4)` by taking the main person, and `(T,51)`).

2. **Finding 1 — FineFS's actual anatomical convention is `h36m17` (not `coco17`):**  
   The preliminary specifications provisionally used `coco17` indexing while awaiting inspection of the real data. By geometrically verifying bone lengths, symmetries, and alignment between the hip vector and shoulder vector in the real `.npz` files, it emerged that FineFS uses the **Human3.6M 17-joint standard (`h36m17`)** with **vertical axis `Z` (index `2`)**.  
   Without this correction, `canonicalize_frame_3d` would have swapped ears/shoulders/hips and graph edges would have connected non-existent joints.  
   We therefore updated [configs/data.yaml](configs/data.yaml), [configs/graph.yaml](configs/graph.yaml), the mock generator in [src/data/loaders.py](src/data/loaders.py), and the tests related to the real `h36m17` map:

   | Index | Joint (`h36m17`) | Index | Joint (`h36m17`) |
   | :---: | :--- | :---: | :--- |
   | `0` | `pelvis` (root) | `9` | `neck` |
   | `1` | `right_hip` | `10` | `head` |
   | `2` | `right_knee` | `11` | `left_shoulder` |
   | `3` | `right_ankle` | `12` | `left_elbow` |
   | `4` | `left_hip` | `13` | `left_wrist` |
   | `5` | `left_knee` | `14` | `right_shoulder` |
   | `6` | `left_ankle` | `15` | `right_elbow` |
   | `7` | `spine` | `16` | `right_wrist` |
   | `8` | `thorax` | | |

   Real example: in `canonicalization`, the pelvis is defined by `pelvis_indices: [4, 1]` (`left_hip`, `right_hip`), the shoulders by `shoulder_indices: [11, 14]` (`left_shoulder`, `right_shoulder`), and the horizontal plane on which yaw rotates is `horizontal_axes: [0, 1]` (`X, Y`).

3. **Finding 2 — The `"goe"` field in `.json` files is weighted by Base Value, while `"judge_score"` is on the ISU `[-5, +5]` scale:**  
   Looking for example at [data/raw/finefs/annotation/0.json](data/raw/finefs/annotation/0.json), each element in `executed_element` contains:
   - `"bv"`: Base Value (e.g., `2.8` for an `FCSp3`);
   - `"judge_score"`: the 9 raw judge marks on the ISU scale from `-5` to `+5` (e.g., `[1, 2, 2, 2, 1, 2, 1, 2, 2]`);
   - `"goe"`: the score already multiplied by `0.1 * bv` (e.g., `0.48`), which therefore depends on the element's difficulty level and is compressed around small values (`<= 1.5`).  
   Since [configs/training.yaml](configs/training.yaml) adopts thresholds on the pure ISU scale (`goe_pos_threshold: 1.5`, `goe_neg_threshold: 0.0`), in `load_finefs_samples()` we implemented `_compute_normalized_judge_goe()`:
   - discards elements with `bv <= 0` or with all sentinel marks `-6`;
   - filters valid marks in `[-5.0, 5.0]` (discarding rare OCR outliers present in the `.json` files);
   - computes the **ISU trimmed mean** (`np.mean(valid_marks[1:-1])` removing minimum and maximum when there are at least 5 valid marks, otherwise simple mean or fallback `raw_goe / (0.1 * bv)`).

4. **Finding 3 — Selection of pure spins and interval parsing (`map_finefs_spin_label`, `_parse_finefs_time_range`):**  
   Each `.json` file describes an entire competition program with jumps (`jump`), sequences (`sequence`), and spins (`spin`).  
   - `map_finefs_spin_label(element_code, include_change_foot=False)` filters only `"spin"` elements of the 3 target classes without change of foot:
     - `FCSP*` / `CSP*` → **`Camel Spin`**
     - `FSSP*` / `SSP*` → **`Sit Spin`**
     - `LSP*` → **`Layback Spin`**
     *(Codes with change of foot `CCSP*`/`CSSP*` are excluded by default because in the central temporal window they would mix two different poses and two different skating legs).*
   - `_parse_finefs_time_range(time_str, fps=25)` converts the `"m-s,m-s"` string (e.g., `"0-52,1-07"`) into `[start_frame, end_frame)` frame indices within the corresponding `.npz`.

5. **Finding 4 — Absence of `athlete_id` in `.json` files and Anti-Leakage Split by `performance_id` (`create_grouped_splits`):**  
   Public FineFS `.json` files anonymize the skater (`athlete_id = None`, flag `athlete_id_available: false` in the report).  
   In `create_grouped_splits()` of [src/data/loaders.py](src/data/loaders.py):
   - when `athlete_id` is available (e.g., in synthetic mock tests) it groups by athlete (`ath_{athlete_id}`);
   - on the real FineFS dataset it groups by **entire performance** (`perf_{performance_id}`, e.g., `finefs_0`), preventing two spins from the same program from going one into `train` and one into `val`/`test`;
   - evaluates up to `50` random permutations guided by `seed` (`42`), choosing the one with the best coverage above/below the GOE median across all `(split, class)` pairs;
   - finally runs strict disjointness assertions (`isdisjoint`) on `sample_id`, `performance_id`, and `athlete_id`.

---

### 2.3 Exact Results of Materialization on the Real FineFS Dataset
Running the entire Phase 1 + Phase 2 pipeline on the real data in `data/raw/finefs/`, the following artifacts were generated and verified on disk:

#### A. Inventory and Loading
- **[reports/finefs_inventory.json](reports/finefs_inventory.json)** (`1,209,740` bytes, `status: "ready"`):
  - `.npz` files found: `1167` (`1167` valid, `0` invalid, `1167` non-zero 3D).
  - `.json` annotation files scanned: `1167`.
  - `UnifiedSample` samples loaded by `load_finefs_samples`: **`1583`** total (`1` single segment discarded due to `segment_out_of_bounds_or_short`):
    - **`Camel Spin`**: `696`
    - **`Layback Spin`**: `470`
    - **`Sit Spin`**: `417`
- **[data/intermediate/unified_samples.pkl](data/intermediate/unified_samples.pkl)** (`104,969,494` bytes, ~100.1 MB): contains all `1583` validated spins in `UnifiedSample` format.

#### B. Anti-Leakage Splits ([data/splits/grouped_splits.json](data/splits/grouped_splits.json), `44,606` bytes)
| Split | # Samples | # Distinct Performances | Camel Spin | Layback Spin | Sit Spin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **1102** | `776` | `481` | `328` | `293` |
| **`val`** | **235** | `166` | `115` | `65` | `55` |
| **`test`** | **246** | `166` | `100` | `77` | `69` |
| **Total** | **1583** | `1108` *(with at least 1 target spin)* | `696` | `470` | `417` |

#### C. Serialized PyG Graphs in `data/processed/`
All `1583` spins passed both `extract_stable_window` and `canonicalize_window_3d` with **zero discards** (`discarded_in_preprocessing: 0` in every split):

| Serialized File | Size | Graphs Saved | Discarded | `GOE >= 1.5` (Camel / Layback / Sit) | `GOE <= 0.0` (Camel / Layback / Sit) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`data/processed/train_graphs.pt`** | `3,392,589` B (~3.24 MB) | **1102** | `0` | `318` / `247` / `156` (**721** tot.) | `21` / `13` / `32` (**66** tot.) |
| **`data/processed/val_graphs.pt`** | `719,959` B (~703 KB) | **235** | `0` | `71` / `52` / `33` (**156** tot.) | `4` / `2` / `2` (**8** tot.) |
| **`data/processed/test_graphs.pt`** | `755,289` B (~738 KB) | **246** | `0` | `67` / `63` / `35` (**165** tot.) | `4` / `2` / `2` (**8** tot.) |

> **Key observation for the upcoming Phase 3:**  
> In the real training set there are **721** high-quality spins (`GOE >= 1.5`) but only **66** low/negative-quality spins (`GOE <= 0.0`), because in international competitions included in FineFS severe errors are relatively rare.  
> This confirms exactly why the architecture includes **biomechanical data augmentation with Rodrigues rotations** (Phase 3): generating controlled synthetic negatives of severity 1/2/3 from correct executions is essential to give the Siamese network abundant and graded negative examples.

---

## 3. Phase 3 — Rodrigues Geometric Augmentation, Triplet Sampling, Adaptive Loss, and Siamese GATv2 Training

In Phase 3 we built the neural core of the system: controlled generation of biomechanical errors on real 3D poses, deterministic construction of `(Anchor, Positive, Negative, margin_alpha)` triplets, training with the **Siamese GATv2** network, and systematic prevention of latent space collapse.

### 3.1 Architectural Choices and Empirical Diagnostics on Real FineFS
During experimental tuning on [configs/training.yaml](configs/training.yaml) and on the real graphs (`train_graphs.pt` = `1102`, `val_graphs.pt` = `235`), validation highlighted two fundamental phenomena that guided the final configuration:

1. **Impact of `dropout` on compact skeletal graphs (17 nodes) with Siamese Triplet Loss (`dropout: 0.0`):**  
   In `GATv2Encoder`, each graph has only `17` nodes and `32` directed edges plus `17` self-loops, normalized with `LayerNorm` before `ELU`. Applying random independent dropout masks to the `(A, P, N)` branches produced stochastic noise between masks larger than the initial separation `(d_an - d_ap)`, preventing overfitting on a fixed batch of `16` triplets (`train_acc ~ 0.50` even when `eval_acc == 1.000`) and inducing AdamW to collapse `z` toward a constant vector.  
   Validating `dropout: 0.0` (as planned by the parent specification for validation tuning), the overfit test on `16` triplets converges stably without collapse.

2. **Preventing Latent Collapse on Real FineFS (`Spin Variants` & Clockwise/Counter-Clockwise Rotations):**  
   On the real FineFS dataset, the same spin class (e.g., *Camel Spin* or *Sit Spin*) includes legitimate stylistic variants and skaters rotating both clockwise and counter-clockwise. With naive sampling:
   - a positive `P` drawn from a different variant has a large raw Euclidean distance from anchor `A`, whereas a synthetic negative `self_synth` obtained by slightly perturbing `A` itself resembles it much more globally;
   - a purely geometric baseline on aligned coordinates (`Raw Coord Fro`) achieves only `0.325` accuracy on validation triplets (`Raw Angle L2` = `0.565`), and an unregularized Siamese training collapsed rapidly (`embedding_std -> 0.004`, `val_triplet_acc = 0.405`).
   
   To structurally solve the problem without touching evaluation on `val`/`test` in any way, three synergistic measures were introduced in [configs/training.yaml](configs/training.yaml), [src/training/triplet_sampler.py](src/training/triplet_sampler.py), and [src/training/losses.py](src/training/losses.py):
   - **`positive_top_k: 8` (active ONLY on `split == "train"`):** `_select_angle_nearest_positives()` filters positive candidates `P` (always with distinct `performance_id` and athlete when available), restricting sampling to the `8` closest to `A` in the joint angle space `edge_attr[:, 4]` (`cos(theta)`), avoiding conflicts between incompatible sub-variants during training; in `val` and `test`, sampling remains completely uniform and unbiased.
   - **Negative balancing (`real_negative_prob: 0.70`, `self_synth_prob: 0.20`):** prioritizes real low-quality negatives (`70%`) and, among synthetic negatives (`30%`), reduces excess perturbations of the same anchor (`20%` `self_synth`, `80%` `cross_synth` from other positive examples).
   - **Anti-collapse latent variance hinge (`var_reg_weight: 0.35`, `var_reg_target: 1.0`, active ONLY on `train_criterion`):** penalizes drops in per-dimension standard deviation below `1.0` on `[z_A, z_P, z_N]`, leaving `eval_criterion` unchanged and unregularized so as to measure pure `triplet_loss` in validation.

---

### 3.2 Modules Implemented in Phase 3

1. **[src/training/augmentation.py](src/training/augmentation.py) — Biomechanical Errors via Rodrigues SO(3) Rotations:**
   - `rodrigues_rotation_matrix(axis, angle_rad)`: computes the `3x3` orthogonal matrix $R = I + \sin(\theta)K + (1 - \cos(\theta))K^2$.
   - `rotate_kinematic_subtree(P_bar, joint_idx, child_idx, axis, angle_deg, graph_cfg)`: reads `H36M-17` kinematic chains from `graph_cfg["kinematic_subtrees"]` (e.g., `"4->5": [5, 6]`) and rigidly rotates the subtree around pivot `P_bar[joint_idx]`, preserving all bone lengths exactly.
   - `_select_weighted_subtree_keys()` and `_orthogonal_unit_axis()`: dynamically prioritize from `joint_names` lower-limb joints (`hip`/`knee`, weight `2.5`) and upper-limb joints (`shoulder`/`elbow`, weight `1.2`) over the trunk (`0.3`), generating random axes orthogonal to the bone.
   - `generate_synthetic_negatives(train_graphs, data_cfg, graph_cfg, train_cfg)`: blocks with `ValueError` any graph with `split != "train"` (anti-leakage barrier); for each of the `721` real positive graphs in `train` only, generates `3` synthetic negatives (`tier` `1`, `2`, `3` with perturbations `[8, 15]°`, `[16, 28]°`, `[29, 42]°` plus Gaussian noise `noise_std = 0.005`), re-canonicalizes them with `canonicalize_frame_3d()` (with a noise-free fallback in case of `DegenerateSkeletonError`), and rebuilds the PyG graphs (`2163` total synthetic graphs on FineFS).

2. **[src/training/triplet_sampler.py](src/training/triplet_sampler.py) — GOE-Tier Partitioning and Deterministic Dataset:**
   - `resolve_class_thresholds(real_class_graphs, train_cfg)`: computes `(tau_neg, tau_pos)` for each class using thresholds `goe_neg_threshold = 0.0` and `goe_pos_threshold = 1.5` (with automatic fallback to `0.35` / `0.70` quantiles if the pool is too small).
   - `partition_by_goe_tiers(graphs, train_cfg, fit_thresholds=None)`: separates graphs per class into `pos`, `mid`, `neg_real`, `neg_synth`; `val` and `test` always receive the thresholds estimated on the training set only.
   - `TripletDataset`: generates a deterministic set of `num_triplets` tuples `(A, P, N, margin_alpha)` within the same class (`element_label`):
     - explicitly verifies the absence of augmented graphs in the real `val` and `test` pools;
     - chooses `P` from distinct `performance_id` and `athlete_id` relative to `A`, preferring `|GOE(A) - GOE(P)| <= max_pos_goe_diff` (`1.5`) and, only on `split == "train"`, applying `_select_angle_nearest_positives(..., positive_top_k=8)`;
     - chooses `N` from `neg_real` with probability `0.70` (mapping `delta_goe` to tiers `1`, `2`, `3` for adaptive margin `0.20 / 0.60 / 1.00`) or from `neg_synth` with probability `0.30` (of which `self_synth_prob = 0.20` from the same `parent_sample_id` and `0.80` `cross_synth` with fallback to `neg_synth_pool`).

3. **[src/training/losses.py](src/training/losses.py) — Numerically Safe Adaptive Loss:**
   - `AdaptiveTripletMarginLoss(eps=1e-12, var_reg_weight=0.0, var_reg_target=1.0)`: computes $d_{AP} = \sqrt{\|z_A - z_P\|_2^2 + \varepsilon}$ and $d_{AN} = \sqrt{\|z_A - z_N\|_2^2 + \varepsilon}$ (avoiding `NaN`/`inf` derivatives at zero distance), applies `F.relu(d_ap - d_an + margins)`, and optionally adds the variance regularization `var_reg_weight * mean(F.relu(var_reg_target - std_per_dim))`. Returns `(loss, stats)` where `stats["loss"]` always records the pure `triplet_loss`.

4. **[src/models/encoder.py](src/models/encoder.py) and [src/models/siamese_gat.py](src/models/siamese_gat.py) — Siamese GATv2 Network:**
   - `GATv2Encoder`: multi-head `GATv2Conv` (`in_channels=3`, `hidden_dim=32`, `heads=4`, `edge_dim=5`, `concat=True`, `add_self_loops=True`, `share_weights=False`) → `LayerNorm(128)` → `ELU` → `GATv2Conv` (`concat=False`, `heads=4`) → `LayerNorm(32)` → `ELU` → `global_mean_pool` → MLP `proj` (`32 -> 32 -> ELU -> 32`). Supports `return_attention_weights=True` returning `{"layer1": (edge_idx1, alpha1), "layer2": (edge_idx2, alpha2)}` for explainability in Phases 4 and 5.
   - `SiameseGATv2`: encapsulates **a single shared instance** `self.encoder` used by `encode`, `forward_triplet`, and `compute_distance`.

5. **[src/training/train.py](src/training/train.py) — Training Loop, Validation, and Checkpoint:**
   - `train_one_epoch`: Siamese forward pass, `train_criterion` calculation (with variance regularization), `loss.backward()`, gradient clipping (`max_norm = 1.0`), and `AdamW` step (`lr = 1e-3`, `weight_decay = 1e-4`).
   - `evaluate_triplets`: evaluation in `@torch.no_grad()` with pure `eval_criterion` (`var_reg_weight = 0.0`) and explicit monitoring of `embedding_std` across all anchor embeddings `z_A`.
   - `run_training_pipeline` & `main()`: generates augmentations once on train only, rebuilds `TripletDataset(..., seed=seed + epoch)` at each epoch, saves `checkpoints/best_model.pt` only when `val_triplet_acc` improves (or `val_loss` improves on a tie) and `embedding_std > 1e-6`, applies early stopping (`patience = 15`), and saves the full history in [reports/training_history.json](reports/training_history.json).

---

### 3.3 Historical Record of Training Run v1 (Baseline 3-Channel Run on FineFS `cuda`)

To preserve a permanent, reproducible baseline record before regenerating graphs with the Spec 10 improvements, all metrics and hyperparameters of **Training Run v1** (`python -m src.training.train` on **NVIDIA GeForce RTX 3050 Laptop GPU**, `device = "cuda"`) are archived below.

#### 3.3.1 Run v1 Configuration Snapshot
- **Preprocessing & Graph Schema (v1):**
  - Window selection: fixed midpoint (`mid = len(valid_indices) // 2` on `[20%, 80%]`), `window_size = 9`
  - Bilateral supporting-leg canonicalization: **disabled** (`canonicalize_supporting_leg = False`)
  - Node features: `node_feature_mode = "baseline"` (`in_channels = 3`: `[x, y, z]`), `edge_dim = 5`
- **Dataset Pools & GOE Thresholds (v1):**
  - `train_graphs.pt`: `1102` real graphs (`721` positives with `GOE >= 1.5`, `66` real negatives with `GOE <= 0.0`, `315` mid) + **`2163` synthetic negative graphs** (`721 × 3` tiers)
  - `val_graphs.pt`: `235` real graphs (`156` positives with `GOE >= 1.5`, **`8` real negatives** with `GOE <= 0.0` [`4` Camel, `2` Layback, `2` Sit], `71` mid), `0` synthetic
  - `test_graphs.pt`: `246` real graphs (`165` positives with `GOE >= 1.5`, **`8` real negatives** with `GOE <= 0.0`), untouched
- **Training & Loss Hyperparameters (v1):**
  - `hidden_dim = 32`, `embed_dim = 32`, `heads = 4`, `dropout = 0.0`, `normalize_embeddings = False`
  - `batch_size = 32`, `lr = 1e-3`, `weight_decay = 1e-4`, `max_epochs = 80`, `early_stopping_patience = 15`, `grad_clip_max_norm = 1.0`
  - `num_train_triplets_per_epoch = 600`, `num_val_triplets = 200`
  - `tier_rule = "threshold"` (`goe_pos_threshold = 1.5`, `goe_neg_threshold = 0.0`), `positive_top_k = 8`, `real_negative_prob = 0.70`, `self_synth_prob = 0.20`
  - `adaptive_margins = {1: 0.20, 2: 0.60, 3: 1.00}`, `var_reg_weight = 0.35`, `var_reg_target = 1.0`

#### 3.3.2 Run v1 Epoch-by-Epoch Milestone Table (`62` Epochs Total)

| Milestone / Epoch | `train_loss` | `train_triplet_acc` | `train_margin_acc` | `train_d_ap` / `train_d_an` | `val_loss` | `val_triplet_acc` | `val_margin_acc` | `val_d_ap` / `val_d_an` | `embedding_std` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** *(Start)* | `0.6637` | `0.5583` | `0.0450` | `0.2891` / `0.3314` | `0.7853` | `0.4650` | `0.0500` | `0.6647` / `0.6075` | `0.1028` |
| **Epoch 11** *(First >50% val)* | `0.6155` | `0.5850` | `0.1967` | `0.7292` / `0.8775` | `0.7516` | `0.5350` | `0.1400` | `0.9456` / `0.9492` | `0.1509` |
| **Epoch 22** *(Min `val_loss`)* | `0.5818` | `0.6217` | `0.2767` | `0.9557` / `1.2348` | **`0.6768`** | `0.5700` | `0.1850` | `1.1595` / `1.2431` | `0.1813` |
| **Epoch 35** *(Mid convergence)* | `0.5142` | `0.6833` | `0.3883` | `1.1328` / `1.6749` | `0.7184` | `0.6250` | `0.3100` | `1.6214` / `1.8590` | `0.2491` |
| **Epoch 47** *(**Best Checkpoint**)* | `0.4849` | `0.7117` | `0.4533` | `1.2028` / `1.8667` | **`0.7290`** | **`0.6500`** | **`0.3750`** | `1.8857` / `2.2023` | **`0.2889`** |
| **Epoch 62** *(Early Stop)* | `0.4298` | `0.7250` | `0.4983` | `1.1267` / `1.8743` | `0.7982` | `0.6050` | `0.3950` | `2.0159` / `2.3398` | `0.3157` |

- **Non-learned geometric baselines on the exact same validation triplets (v1):**
  - `Raw Coord Fro` (Frobenius norm on canonicalized 3D coordinates): **`0.3250`**
  - `Raw Angle L2` (Euclidean norm on the 32 bone-angle cosines): **`0.5650`**
  - `SiameseGATv2 (Best Epoch 47)`: **`0.6500`** (`+32.5` pts over `Raw Coord Fro`, `+8.5` pts over `Raw Angle L2`).

#### 3.3.3 Post-Mortem Diagnostic Analysis of Run v1
1. **What went great:**
   - **Complete elimination of latent collapse:** whereas the unregularized baseline collapsed to `embedding_std = 0.004` (`val_triplet_acc = 0.405`), Run v1 expanded `embedding_std` smoothly from `0.1028` to `0.2889` at epoch 47 (`0.3157` at epoch 62).
   - **Metric inversion on validation:** at epoch 1, positive pairs were farther apart than negative pairs on validation (`val_d_ap = 0.6647 > val_d_an = 0.6075`); by epoch 47, the network established a clear positive separation (`val_d_ap = 1.8857 < val_d_an = 2.2023`, gap `+0.3166`), raising `val_margin_satisfied_acc` from `5.0%` to `37.5%`.
2. **What matched expectations:**
   - **Physiological late-epoch overfitting:** between epoch 47 and epoch 62, training metrics continued improving (`train_loss: 0.4849 -> 0.4298`, `train_acc: 0.7117 -> 0.7250`) while validation degraded (`val_loss: 0.7290 -> 0.7982`, `val_acc: 0.6500 -> 0.6050`), validating the `patience = 15` early-stopping gate.
   - **Scale drift between `val_loss` and `val_triplet_acc`:** with `normalize_embeddings: false`, minimum `val_loss` (`0.6768`) occurred at epoch 22 when embeddings were still compact (`embedding_std = 0.1813`), whereas ranking accuracy peaked at epoch 47 (`0.6500`) as embedding norms grew toward `var_reg_target = 1.0`.
3. **What disappointed & root causes discovered in Phases 1–3:**
   - **B1 — Bilateral supporting-leg asymmetry (`51.9%` vs `48.1%`):** direct inspection of `train_graphs.pt` revealed that `572 / 1102` spins (`51.9%`) have `left_ankle` (`joint 6`) as the supporting leg on the ice, while `530 / 1102` (`48.1%`) have `right_ankle` (`joint 3`) lower. Without bilateral reflection, left-leg and right-leg spins of the exact same class have swapped node indices (`[1,2,3]` vs `[4,5,6]`), inflating `val_mean_d_ap` to `1.8857` (`+57%` above `train_mean_d_ap = 1.2028`).
   - **B2 — Misaligned GOE thresholds (`0.0` / `1.5`) vs. real FineFS quantiles (`Q25 = 1.29`, `Q35 = 1.57`, `Q50 = 2.00`, `Q70 = 2.43`):** setting `goe_pos_threshold = 1.5` labeled `66%` of the dataset as `pos`, while `goe_neg_threshold = 0.0` left only **`8` real negatives** in validation (`4` Camel, `2` Layback, `2` Sit), making validation noisy and bottlenecked by negative sample scarcity.
   - **B3 & B4 — Blind midpoint windowing and discarded `window_jitter`:** taking the fixed midpoint of `[20%, 80%]` often captured position transitions, and discarding `window_jitter` in `builder.py` threw away spin centering/stability information.

---

### 3.4 Spec 10 Revision, Empirical Ablation Study, and Training Run v2 ([specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md](specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md))

#### 3.4.1 Configurable Upgrades Implemented in Code
To systematically investigate bottlenecks **B1–B5**, we implemented the full modular revision specified in [specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md](specs/10_spec_revisione_fasi_1_3_e_ricalibrazione.md):
1. **Bilateral Chiral Canonicalization (`src/data/preprocessing.py` & `configs/data.yaml`):** added `mirror_skeleton_left_right()` and configurable `canonicalize_supporting_leg` (performed *after* Yaw Alignment so the sagittal front-back axis $Y$ is never inverted) along with window-consistent `force_mirror` and metadata tracking (`mirrored_to_canonical_leg`).
2. **Sliding-Window Selection Modes (`src/data/segmentation.py`):** added configurable `selection_mode` supporting both `"center"` (midpoint of `[20%, 80%]`) and `"min_jitter"` (rotation-invariant posture instability minimization).
3. **Per-Joint Temporal Stability & `"temporal_extended"` Mode (`src/graph/features.py` & `src/graph/builder.py`):** preserved `window_jitter` and per-joint 3D standard deviation `joint_jitter` ($\sigma_i \in \mathbb{R}^{17}$) on every PyG `Data` object, added 6-channel `"temporal_extended"` node feature support `[x, y, z, r_i, sym_i, sigma_i]`, and added the standalone CLI entrypoint `python -m src.graph.builder`.

#### 3.4.2 Empirical Ablation Study on Real FineFS Validation (`train = 1102`, `val = 235`)
Running controlled ablations on the regenerated FineFS graphs revealed four crucial domain findings and one key optimization bottleneck:
- **Why `tier_rule: "quantile"` (`0.30 / 0.70`) degraded validation (`val_triplet_acc = 0.5000` vs. `0.6500`):** in FineFS, international competition spins are heavily skewed toward positive execution (`Q30 = +1.43` on Camel, `+1.67` on Layback). Under ISU judging rules, `GOE = +1.5` is a **clean spin without biomechanical errors**. Labeling `GOE <= 1.67` as "Negative" forced the Siamese network to push clean spins away from clean spins, dropping even the non-learned `Raw Angle L2` baseline from `0.5650` to `0.4500`. Keeping `tier_rule: "threshold"` (`0.0 / 1.5`) preserves true biomechanical error semantics.
- **Why `min_jitter` windowing degraded validation (`val_triplet_acc = 0.4900`):** monocular 3D pose estimation exhibits higher frame-to-frame reconstruction jitter during high-RPM rotation at the center (`50%`) of the spin than during the slower upright entry/exit glide (`20%` / `80%`). Consequently, minimizing raw 3D jitter selected entry/exit transitions rather than the canonical spin hold.
- **Why `canonicalize_supporting_leg: true` degraded validation (`val_triplet_acc = 0.5250`):** in low or crossed spins (`Sit Spin`, `Layback Spin`), the free ankle and supporting ankle have similar vertical $Z$ coordinates, causing noisy bilateral flips (`46.1%`), and `directed_edge_angle_triplets["0->7"] = [1, 0, 7]` in `configs/graph.yaml` is asymmetric between hips `1` and `4`.
- **Why 6-channel `temporal_extended` degraded validation (`val_triplet_acc = 0.4800`):** constant `sym_i` (`mean abs = 0.706`) diluted the 3D coordinates `[x, y, z]`, and `joint_jitter` created a train-only shortcut between synthetic and real negatives.
- **Root Optimization Bottleneck Identified — Premature Early Stopping During `var_reg` Warmup:** during the first `~25` epochs, `AdaptiveTripletMarginLoss` primarily expands `embedding_std` from `0.10` toward `0.25+` under the anti-collapse variance hinge (`var_reg_weight = 0.35`, `var_reg_target = 1.0`). With `early_stopping_patience = 15`, any random early-epoch accuracy spike (e.g., `0.550` at Epoch 11 caused by CUDA `scatter_add_` atomicAdd non-determinism) triggered early stopping at Epoch 26, right before the Epoch 30–65 metric convergence phase. Recalibrating `early_stopping_patience: 30` resolved this bottleneck completely.

#### 3.4.3 Training Run v2 Results (`checkpoints/best_model.pt` & `reports/training_history.json`)
With clean regenerated graphs (`1102` train, `235` val, `246` test, enriched with `window_jitter`, `joint_jitter`, and `mirrored_to_canonical_leg` attributes) and `early_stopping_patience: 30`, **Training Run v2** completed all `80` epochs on `cuda`, saving the new best checkpoint at **Epoch 62**:

| Milestone / Epoch (Run v2) | `train_loss` | `train_triplet_acc` | `train_margin_acc` | `train_d_ap` / `train_d_an` | `val_loss` | `val_triplet_acc` | `val_margin_acc` | `val_d_ap` / `val_d_an` | `embedding_std` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** *(Start)* | `0.6637` | `0.5583` | `0.0450` | `0.2891` / `0.3314` | `0.7853` | `0.4650` | `0.0500` | `0.6647` / `0.6075` | `0.1028` |
| **Epoch 11** *(Warmup peak)* | `0.6100` | `0.5883` | `0.1967` | `0.7194` / `0.8683` | `0.7390` | `0.5500` | `0.1400` | `0.9603` / `0.9764` | `0.1550` |
| **Epoch 34** *(Mid expansion)* | `0.5404` | `0.6750` | `0.3800` | `1.1057` / `1.6281` | `0.8740` | `0.5850` | `0.3300` | `1.8659` / `2.0287` | `0.3005` |
| **Epoch 47** *(Matches Run v1)* | `0.4981` | `0.7100` | `0.4267` | `1.2019` / `1.8060` | **`0.7316`** | `0.6500` | `0.3800` | `1.8829` / `2.1827` | `0.2876` |
| **Epoch 57** *(First 66% val)* | `0.5128` | `0.7000` | `0.4567` | `1.3097` / `1.9615` | `0.7416` | **`0.6600`** | `0.3850` | `1.9985` / `2.3173` | `0.3084` |
| **Epoch 62** *(**Best Checkpoint v2**)* | **`0.4508`** | **`0.7283`** | **`0.5033`** | `1.2340` / `2.0727` | **`0.7410`** | **`0.6600`** | **`0.4550`** | `2.1504` / `2.7053` | **`0.3434`** |
| **Epoch 74** *(Max margin acc)* | `0.4681` | `0.7400` | `0.5417` | `1.6323` / `2.6116` | `0.7889` | **`0.6600`** | **`0.4900`** | `2.6792` / `3.2221` | `0.3967` |

#### 3.4.4 Direct Comparison: Run v1 vs. Run v2 (Best Checkpoint)
| Metric | Non-Neural `Raw Coord Fro` | Non-Neural `Raw Angle L2` | **SiameseGATv2 Run v1 (Ep. 47)** | **SiameseGATv2 Run v2 (Ep. 62)** | **Improvement (v2 vs. v1)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`val_triplet_acc`** | `0.3250` | `0.5650` | `0.6500` | **`0.6600`** | **`+1.00%`** (`+9.5` pts over Angle L2) |
| **`val_margin_satisfied_acc`** | — | — | `0.3750` | **`0.4550`** *(up to `0.4900` at Ep. 74)* | **`+8.00` percentage points** (`+21.3%` rel.) |
| **Validation Gap $\Delta d_{\text{val}} = d_{AN} - d_{AP}$** | — | — | `+0.3166` | **`+0.5549`** | **`+75.3%` wider metric separation** |
| **Latent Dispersion (`embedding_std`)** | — | — | `0.2889` | **`0.3434`** | **`+18.9%`** |
| **`train_triplet_acc` / `train_margin_acc`** | — | — | `0.7117` / `0.4533` | **`0.7283` / `0.5033`** | **`+1.66%` / `+5.00%`** |

---

## 4. Verified Automated Test Suite (`pytest tests/ -v`)

After implementing Spec 10, the full automated test suite passes with **`31 passed` (`0` errors, `0` skipped)**:

- **[tests/test_config_and_utils.py](tests/test_config_and_utils.py)** (`5` tests)
- **[tests/test_dataset.py](tests/test_dataset.py)** (`7` tests, including direct verification on real FineFS files)
- **[tests/test_graph.py](tests/test_graph.py)** (`4` tests, verifying `"baseline"` 3-ch, `"extended"` 5-ch, and `"temporal_extended"` 6-ch modes)
- **[tests/test_normalization.py](tests/test_normalization.py)** (`7` tests, including `test_bilateral_supporting_leg_mirror_invariance`)
- **[tests/test_triplets.py](tests/test_triplets.py)** (`8` tests — Gate 3)

*(Windows environment note: during development, the sandbox blocks direct write operations in workspace paths with `PermissionError` / `RuntimeError`; therefore, steps that write generated files in `data/`, `checkpoints/`, and `reports/` are run with write permissions enabled, while `pytest` runs cleanly thanks to `-p no:cacheprovider` in `pyproject.toml`.)*

---

## 5. Upcoming Updates (Sections Reserved for Phases 4 and 5)

### Phase 4 — Geometric Baselines, AQA Metrics, $Q$ Calibration, and Inference
*(To be filled in upon completion of Phase 4)*

### Phase 5 — 3D Visualization, Automated Figures, and Streamlit Dashboard
*(To be filled in upon completion of Phase 5)*
