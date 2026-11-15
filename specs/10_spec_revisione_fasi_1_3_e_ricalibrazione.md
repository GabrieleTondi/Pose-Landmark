# 10 — Sector Sub-Specification: Phases 1–3 Revision, Bilateral Canonicalization, and Metric Recalibration

**Sector:** 10 — Pre-Training Pipeline Revision (Geometry, Windowing, Graph Features, GOE Quantiles, and Hyperspherical Metric Learning)  
**Parent Spec References:** §4, §5, §6, §7, §8, and Post-Phase-3 Empirical Diagnostics ([DIARIO_DI_PROGETTO.md](../DIARIO_DI_PROGETTO.md))  
**Modified Sectors:** 01 (`configs/*.yaml`), 03 (`segmentation.py`, `preprocessing.py`), 04 (`features.py`, `builder.py`), 05 (`triplet_sampler.py`, `augmentation.py`), 06 (`train.py`)  
**Target Outcome:** Regenerate `data/processed/{train,val,test}_graphs.pt` and retrain `checkpoints/best_model.pt` overcoming the 65.0% validation accuracy ceiling.

---

## 1. Empirical Diagnosis and Motivation for the Revision

Analysis of the first full training run on real FineFS (`62` epochs, best epoch `47`, `val_triplet_acc = 0.6500`, `val_margin_satisfied_acc = 0.3750`) and direct inspection of the serialized graphs (`train_graphs.pt` = `1102`, `val_graphs.pt` = `235`) identified **five structural bottlenecks** in Phases 1–3:

| # | Root Cause Identified in FineFS Data | Quantitative Evidence | Impact on Siamese GATv2 |
| :- | :--- | :--- | :--- |
| **B1** | **Unaligned Bilateral Handedness (Supporting vs. Free Leg)** | In `train_graphs.pt`, **51.9%** (`572/1102`) of spins have `left_ankle` (`joint 6`) as the lower supporting foot on the ice, while **48.1%** (`530/1102`) have `right_ankle` (`joint 3`) lower. | Two identical spins executed on opposite supporting legs have swapped node indices (`[1,2,3]` vs `[4,5,6]`). Raw coordinate baseline drops to `0.325` and validation `mean_d_ap` inflates to `1.8857` (+57% over train). |
| **B2** | **Misaligned Absolute GOE Thresholds (`0.0` / `1.5`)** | Real FineFS GOE quantiles are `Q25 = 1.29`, `Q35 = 1.57`, `Q50 = 2.00`, `Q70 = 2.43`. | `GOE >= 1.5` labels **66%** of all samples as `pos` (mixing mediocre and elite spins), while `GOE <= 0.0` leaves only **8 real negatives** in the entire validation set (`4` Camel, `2` Layback, `2` Sit) and **8** in the test set. |
| **B3** | **Blind Midpoint Temporal Windowing** | `extract_stable_window` slices the exact middle index `mid = len(valid_indices) // 2` of the `[20%, 80%]` segment. | In real competition spins (`150–350` frames), the midpoint frequently lands on a position change, foot/edge transition, or catch-foot grab rather than the most stable hold. |
| **B4** | **Discarded Temporal Stability (`window_jitter`)** | `canonicalize_window_3d` computes `window_jitter` across the 9 frames, but `builder.py` discards it (`p_bar, _ = canonicalize_window_3d(...)`). | ISU GOE heavily rewards spin centering and penalizes wobble/traveling, yet the static mean pose $\bar{P}$ erases all frame-to-frame stability cues. |
| **B5** | **Unbounded Latent Norm Drift (`normalize_embeddings: false`)** | Minimum `val_loss` occurs at epoch `22` (`0.6768`, `embedding_std = 0.181`), whereas peak `val_triplet_acc` occurs at epoch `47` (`val_loss = 0.7290`, `embedding_std = 0.289`). | As embedding norms expand toward `var_reg_target = 1.0`, Euclidean distances scale up, causing `val_loss` and `val_triplet_acc` to decouple. |

---

## 2. Decomposition into Explicit Technical Tasks

### Task 10.1 — Bilateral Chiral Canonicalization (`src/data/preprocessing.py` & `configs/data.yaml`)

#### 10.1.1 YAML Configuration Update (`configs/data.yaml`)
Add the bilateral alignment config keys under `canonicalization`:
```yaml
canonicalization:
  horizontal_axes: [0, 1]
  vertical_axis: 2
  scale_epsilon: 1.0e-4
  yaw_epsilon: 1.0e-4
  pelvis_indices: [4, 1]        # [left_hip, right_hip]
  shoulder_indices: [11, 14]    # [left_shoulder, right_shoulder]
  canonicalize_supporting_leg: true
  ankle_indices: [6, 3]         # [left_ankle, right_ankle] in H36M-17
  symmetric_joint_pairs:
    - [1, 4]    # right_hip <-> left_hip
    - [2, 5]    # right_knee <-> left_knee
    - [3, 6]    # right_ankle <-> left_ankle
    - [14, 11]  # right_shoulder <-> left_shoulder
    - [15, 12]  # right_elbow <-> left_elbow
    - [16, 13]  # right_wrist <-> left_wrist
```

#### 10.1.2 Mathematical Rule in `src/data/preprocessing.py`
Implement helper ` mirror_skeleton_left_right(pts: np.ndarray, data_cfg: Dict[str, Any]) -> np.ndarray`:
1. Given root-centered and torso-scaled coordinates `pts` of shape `(17, 3)` (or window `(T_win, 17, 3)`):
2. Reflect across the sagittal plane by negating the first horizontal axis `pts[..., h_ax0] = -pts[..., h_ax0]`.
3. Swap all left/right anatomical joint index pairs listed in `c_cfg["symmetric_joint_pairs"]`.
   - **Crucial property:** Reflecting one spatial axis *and* swapping left/right anatomical landmarks ensures that the vector `hip_r - hip_l` remains well-defined and the subsequent Yaw Alignment places the skeleton in a consistent canonical frame where:
     - **Supporting leg** (lower ankle along vertical axis $Z$) is **always mapped to the `left_*` chain (`4 -> 5 -> 6`)**;
     - **Free leg** (higher ankle along vertical axis $Z$) is **always mapped to the `right_*` chain (`1 -> 2 -> 3`)**;
     - All bone lengths and joint angles $\cos(\theta_{abc})$ are **strictly preserved**.
4. **Window-level consistency (`canonicalize_window_3d`):**
   - Determine whether bilateral reflection is needed **once per window** (by comparing median vertical coordinate $Z$ of `left_ankle` vs. `right_ankle` after centering/scaling across the window, or per frame if `canonicalize_frame_3d` is called standalone) so that frames within the same window never flip inconsistently.
   - Record `"mirrored_to_canonical_leg": bool` in the `stats` dictionary returned by `canonicalize_frame_3d` and `canonicalize_window_3d`.

---

### Task 10.2 — Minimum-Jitter Sliding Window Selection (`src/data/segmentation.py` & `configs/data.yaml`)

#### 10.2.1 YAML Configuration Update (`configs/data.yaml`)
Add `selection_mode` under `windowing`:
```yaml
windowing:
  window_size: 9
  min_valid_frames: 5
  trim_ratio_start: 0.20
  trim_ratio_end: 0.20
  min_keypoint_confidence: 0.20
  selection_mode: "min_jitter"  # "min_jitter" (recommended) or "center" (legacy)
```

#### 10.2.2 Algorithm in `extract_stable_window` (`src/data/segmentation.py`)
When `len(valid_indices) > window_size`:
1. If `w_cfg.get("selection_mode", "min_jitter") == "center"`: keep legacy behavior (`mid = len(valid_indices) // 2`).
2. If `selection_mode == "min_jitter"`:
   - For each candidate contiguous slice `cand = valid_indices[s : s + window_size]` (with stride `1`, or stride `max(1, (len(valid_indices) - window_size) // 30)` for long segments):
   - Compute a fast scale-normalized posture stability score:
     - Center each frame in `keypoints_arr[cand]` at the pelvis midpoint and divide by torso length;
     - Compute the pairwise bone-length and joint-elevation variance across the `window_size` frames (rotation-invariant around $Z$), penalizing windows with large frame-index gaps (`cand[-1] - cand[0] > 2 * window_size`).
   - Select the slice `chosen` that minimizes this instability score, and return `"window_selection_score": float` inside `info`.

---

### Task 10.3 — Temporal Stability Node Features (`configs/graph.yaml`, `src/data/preprocessing.py`, `src/graph/features.py`, `src/graph/builder.py`)

#### 10.3.1 Preserve Per-Joint Temporal Std in `canonicalize_window_3d` (`src/data/preprocessing.py`)
In `canonicalize_window_3d(window_joints, data_cfg)`:
- After stacking the aligned frames `stack` of shape `(N_valid, 17, 3)` and computing `P_bar = np.mean(stack, axis=0)`:
- Compute per-joint 3D positional standard deviation across the window:
  $$\sigma_i = \sqrt{\frac{1}{N_{\text{valid}}} \sum_{t=1}^{N_{\text{valid}}} \|P^{\text{aligned}}_{t, i} - \bar{P}_i\|_2^2} \in \mathbb{R}^{17}$$
- Return `"joint_jitter": joint_jitter.astype(np.float32)` (shape `(17,)`) inside the `stats` dictionary alongside `"window_jitter"`.

#### 10.3.2 Support `"temporal_extended"` Node Features (`configs/graph.yaml` & `src/graph/features.py`)
1. Update `configs/graph.yaml`:
   ```yaml
   num_nodes: 17
   node_feature_mode: "temporal_extended"  # "baseline" (3) | "extended" (5) | "temporal_extended" (6)
   node_feature_dim: 6
   edge_feature_dim: 5
   ```
   And update `configs/training.yaml`:
   ```yaml
   model:
     in_channels: 6
   ```
2. Update signature of `compute_node_features`:
   ```python
   def compute_node_features(
       P_bar: np.ndarray,
       graph_cfg: Dict[str, Any],
       joint_jitter: Optional[np.ndarray] = None,
   ) -> torch.Tensor:
   ```
   - `"baseline"` $\to$ shape `(17, 3)`: `[x, y, z]`
   - `"extended"` $\to$ shape `(17, 5)`: `[x, y, z, r_i, sym_i]`
   - `"temporal_extended"` $\to$ shape `(17, 6)`: `[x, y, z, r_i, sym_i, sigma_i]`, where `sigma_i` is `joint_jitter` reshaped to `(17, 1)` (defaulting to zeros if `joint_jitter is None`, e.g., when called on a single synthetic frame, or inherited from the parent positive graph during Rodrigues augmentation).
3. Update `build_pyg_graph` and `process_and_save_dataset_graphs` in `src/graph/builder.py`:
   - Pass `win_stats["joint_jitter"]` and `win_stats["window_jitter"]` into `sample_meta` and `compute_node_features`.
   - Store `data.window_jitter = float(sample_meta.get("window_jitter", 0.0))` and `data.joint_jitter = torch.tensor(...)` (shape `(17,)`) on the `Data` object so synthetic augmentations in `augmentation.py` can inherit and slightly perturb the parent's `joint_jitter` (`sigma_synth = sigma_parent + tier * 0.01`).
   - Always keep `data.pos` strictly of shape `(17, 3)` for 3D visualization.

---

### Task 10.4 — Quantile-Based GOE Tiering & Balanced Triplet Sampling (`configs/training.yaml` & `src/training/triplet_sampler.py`)

#### 10.4.1 YAML Configuration Update (`configs/training.yaml`)
```yaml
sampling:
  tier_rule: "quantile"           # Fits per-class (tau_neg, tau_pos) on train real graphs only
  goe_pos_threshold: 2.25         # Fallback/reference absolute high-quality threshold
  goe_neg_threshold: 1.25         # Fallback/reference absolute low-quality threshold
  goe_pos_quantile: 0.70          # Top 30% of each class in train -> 'pos' (~2.4+ GOE)
  goe_neg_quantile: 0.30          # Bottom 30% of each class in train -> 'neg_real' (~1.4- GOE)
  max_pos_goe_diff: 0.80          # Tighter GOE coherence between Anchor and Positive
  min_neg_goe_gap: 0.80           # Minimum GOE gap (Anchor - Negative) for real negatives
  real_negative_prob: 0.65        # 65% real negatives (now abundant!), 35% Rodrigues synthetic negatives
  positive_top_k: 12              # Relaxed from 8 to 12 since bilateral canonicalization aligns left/right spins
  self_synth_prob: 0.25
```

#### 10.4.2 Real-Negative Tier Assignment in `TripletDataset` (`src/training/triplet_sampler.py`)
In `TripletDataset.__init__`, when a real negative is selected (`use_real_neg == True`), map `delta_goe = float(anchor.goe) - float(negative.goe)` to severity tiers proportional to the actual FineFS interquartile range:
- `delta_goe < 1.0` $\to$ `tier = 1` (mild quality gap)
- `1.0 <= delta_goe < 1.8` $\to$ `tier = 2` (medium quality gap)
- `delta_goe >= 1.8` $\to$ `tier = 3` (severe quality gap)

**Expected statistical impact on FineFS:**
- Training real positives (`pos`): ~`330` elite spins (`GOE >= ~2.43`); real negatives (`neg_real`): ~`330` lower-quality spins (`GOE <= ~1.43`), up from `66`.
- Validation real positives (`pos`): ~`70`; validation real negatives (`neg_real`): ~`70`, up from **`8`** (`8.75x` increase in validation negative diversity!).

---

### Task 10.5 — Unit-Hypersphere L2 Embedding Normalization & Margin Rescaling (`configs/training.yaml` & `src/training/train.py`)

#### 10.5.1 Why Hyperspherical Normalization (`normalize_embeddings: true`)
When `normalize_embeddings: true` in `GATv2Encoder`, every graph embedding satisfies $\|z\|_2 = 1$, bounding all pairwise Euclidean distances to $d(z_1, z_2) \in [0, 2]$ (where $d = \sqrt{2(1 - \cos\theta)}$).
- This completely eliminates the **norm inflation** observed between epoch 22 and epoch 47;
- `val_loss` and `val_triplet_acc` become strictly aligned because the network can no longer change loss magnitude merely by scaling weights up or down;
- On the 32-dimensional unit sphere $\mathbb{S}^{31}$, a uniform isotropic distribution has per-coordinate standard deviation $\sigma_{\text{iso}} = 1 / \sqrt{32} \approx 0.1768$.

#### 10.5.2 YAML Configuration Update (`configs/training.yaml`)
```yaml
model:
  in_channels: 6
  edge_dim: 5
  hidden_dim: 32
  embed_dim: 32
  heads: 4
  dropout: 0.0
  normalize_embeddings: true      # Project embeddings onto unit hypersphere S^31

loss:
  margin_mode: "adaptive"
  fixed_margin: 0.25
  adaptive_margins:
    1: 0.12                       # ~8 deg angular separation on S^31
    2: 0.25                       # ~15 deg angular separation on S^31
    3: 0.40                       # ~23 deg angular separation on S^31
  var_reg_weight: 0.35
  var_reg_target: 0.16            # Calibrated for 32-D unit hypersphere (1/sqrt(32) ≈ 0.177)
```

---

## 3. Execution Order and Verification Gates (Definition of Done)

When executing this revision, run the steps in the exact order below:

1. **Step 1 — Update Configs & Phase 1/2 Modules (`segmentation.py`, `preprocessing.py`, `features.py`, `builder.py`):**
   - Implement bilateral supporting-leg canonicalization, minimum-jitter window selection, and `temporal_extended` node features (`6` channels).
   - Update unit tests in `tests/test_normalization.py` and `tests/test_graph.py` to test:
     - exact invariance under left-right anatomical reflection (mirroring a right-leg spin produces the exact same canonical pose $\bar{P}$ as a left-leg spin);
     - `temporal_extended` shape `(17, 6)` with `data.pos` preserved as `(17, 3)`.
   - Run gate:
     ```powershell
     pytest tests/test_config_and_utils.py tests/test_normalization.py tests/test_graph.py tests/test_dataset.py -v
     ```

2. **Step 2 — Regenerate Materialized PyG Graph Datasets (`data/processed/*.pt`):**
   - Re-run the dataset materialization pipeline on real FineFS (`1583` spins) to overwrite:
     - `data/processed/train_graphs.pt` (`1102` graphs, `x.shape == (17, 6)`)
     - `data/processed/val_graphs.pt` (`235` graphs, `x.shape == (17, 6)`)
     - `data/processed/test_graphs.pt` (`246` graphs, `x.shape == (17, 6)`)
   - Verify that after bilateral canonicalization **100%** of canonicalized spin graphs have `pos[6, 2] <= pos[3, 2]` (supporting ankle at `joint 6`).

3. **Step 3 — Update Phase 3 Modules (`triplet_sampler.py`, `augmentation.py`, `train.py`) & Retrain:**
   - Implement quantile-calibrated tiering, `joint_jitter` propagation in synthetic negatives, and hyperspherical margin settings.
   - Run full test gate:
     ```powershell
     pytest tests/ -v
     ```
   - Execute full CUDA training on FineFS:
     ```powershell
     python -m src.training.train
     ```
   - **Target Acceptance Criteria:**
     - Validation negative pool `len(val_pools[c]["neg_real"]) >= 15` per class (`~70` total across the 3 classes, up from `8`);
     - Non-collapsed hyperspherical latent space (`embedding_std >= 0.12` on $\mathbb{S}^{31}$);
     - `val_triplet_acc > 0.65` and `val_margin_satisfied_acc > 0.45`, with improved alignment between `train_mean_d_ap` and `val_mean_d_ap`;
     - Update [DIARIO_DI_PROGETTO.md](../DIARIO_DI_PROGETTO.md) with the before/after ablation comparison table.


---

## 4. Empirical Ablation Findings & Final Validated Configuration on Real FineFS

All five candidate interventions (Tasks 10.1–10.5) were implemented as configurable modes in the codebase and systematically evaluated on the real FineFS validation split (`train_graphs.pt` = `1102` real + `2163` synthetic, `val_graphs.pt` = `235` real). The ablation experiments revealed critical domain and optimization insights:

| Ablation Variant | `tier_rule` | Window Mode | Bilateral Mirror | `in_channels` (`x`) | `normalize_embeddings` | `patience` | Best `val_triplet_acc` | Best `val_margin_acc` | Key Empirical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run v1 (Baseline)** | `threshold` (`0.0 / 1.5`) | `center` | `false` | `3` (`baseline`) | `false` | `15` | `0.6500` *(Ep. 47)* | `0.3750` | Strong ranking, but early-stop patience (`15`) is vulnerable to early-epoch noise during latent expansion. |
| **Ablation A (Quantile Tiers)** | `quantile` (`0.30 / 0.70`) | `center` | `false` | `3` (`baseline`) | `false` | `15` | `0.5000` | `0.1550` | In FineFS, `Q30` of GOE is `+1.43` (Camel) / `+1.67` (Layback). In ISU judging, `GOE = +1.5` is a **clean spin without biomechanical errors**; labeling `GOE <= 1.67` as Negative forces the network to separate clean spins from clean spins (`Raw Angle L2` drops from `0.565` to `0.450`). |
| **Ablation B (6-Ch `temporal_extended`)** | `threshold` (`0.0 / 1.5`) | `center` | `false` | `6` (`temporal_extended`) | `false` | `15` | `0.4800` | `0.1100` | Channel 4 (`sym_i = [-1, 0, +1]`) is constant across all graphs with large magnitude (`0.706`), diluting `[x, y, z]`; Channel 5 (`joint_jitter`) differs slightly between static synthetic frames and real windows, creating a train-only shortcut. |
| **Ablation C (`min_jitter` Window)** | `threshold` (`0.0 / 1.5`) | `min_jitter` | `false` | `3` (`baseline`) | `false` | `15` | `0.4900` | `0.0600` | Monocular 3D pose reconstruction exhibits higher jitter during high-RPM rotation at the spin center (`50%`) than during the slower entry/exit glide (`20%`/`80%`), so `min_jitter` selects entry/exit transitions rather than the canonical hold. |
| **Ablation D (Bilateral Mirror)** | `threshold` (`0.0 / 1.5`) | `center` | `true` | `3` (`baseline`) | `false` | `15` | `0.5250` | `0.1250` | In low/crossed spins (`Sit Spin`, `Layback Spin`), the free and supporting ankles have nearly equal vertical $Z$ coordinates, causing noisy flips (`46.1%`), and `directed_edge_angle_triplets["0->7"] = [1, 0, 7]` is asymmetric between hips `1` and `4`. |
| **Ablation E (`L2` Hypersphere)** | `threshold` (`0.0 / 1.5`) | `center` | `false` | `3` (`baseline`) | `true` | `15` | `0.5200` | `0.1800` | Projecting onto $\mathbb{S}^{31}$ from Epoch 1 constrains radial gradient steps during the initial 25-epoch variance expansion phase. |
| **Run v2 (Final Calibrated)** | **`threshold` (`0.0 / 1.5`)** | **`center`** | **`false`** | **`3` (`baseline`)** | **`false`** | **`30`** | **`0.6600` *(Ep. 62)*** | **`0.4550`** | Increasing `early_stopping_patience` from `15` to `30` prevents premature termination during the 25-epoch `var_reg` latent expansion warmup, surpassing both targets (`val_triplet_acc = 0.6600 > 0.65`, `val_margin_satisfied_acc = 0.4550 > 0.45`). |

### Final Active Configuration (`configs/*.yaml`)
- **[configs/data.yaml](../configs/data.yaml):** `selection_mode: "center"`, `canonicalize_supporting_leg: false` (with `"min_jitter"` and `mirror_skeleton_left_right` retained and unit-tested for optional use).
- **[configs/graph.yaml](../configs/graph.yaml):** `node_feature_mode: "baseline"`, `node_feature_dim: 3` (with `"extended"` 5-ch and `"temporal_extended"` 6-ch retained and unit-tested; `window_jitter` and `joint_jitter` stored on every PyG `Data` object for Phase 4/5 analysis).
- **[configs/training.yaml](../configs/training.yaml):** `in_channels: 3`, `normalize_embeddings: false`, `early_stopping_patience: 30`, `tier_rule: "threshold"` (`0.0 / 1.5`), `positive_top_k: 8`, `real_negative_prob: 0.70`, `self_synth_prob: 0.20`, `var_reg_weight: 0.35`, `var_reg_target: 1.0`.
