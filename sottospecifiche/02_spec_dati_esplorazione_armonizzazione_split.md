# 02 — Sector Sub-Specification: Data, Exploration, Harmonization, and Splits

**Sector:** 02 — Data Engineering: FineFS, MMFS Verification, `UnifiedSample`, and Anti-Leakage Group Split  
**Parent Spec References:** §3.1–§3.5, §16.1, §18  
**Input Dependencies:** Sector 01 (`configs/data.yaml`, `src/utils/seed.py`, `src/utils/config.py`)  
**Enabled Outputs:** Sector 03 (Windowing and 3D Canonicalization)

---

## 1. Sector Objective

Provide the data structures and functions to:
1. Inspect FineFS `.npz` files and metadata without assuming key names a priori;
2. Encapsulate each valid skeletal segment in the standardized `UnifiedSample` dataclass;
3. Generate a `train / val / test` split grouped by athlete/performance (`GroupSplit`) with mathematical guarantee of zero data leakage;
4. Also provide a synthetic dataset generator (`generate_mock_finefs_samples`) to allow running `pytest` tests and testing the entire pipeline even before or in the absence of the full FineFS `.npz` files.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 2.1 — `UnifiedSample` Data Structure and Serialization (`src/data/harmonization.py`)
- **File to create:** `src/data/harmonization.py`
- **Exact contract of the class and functions:**
  ```python
  from dataclasses import dataclass, field
  from pathlib import Path
  from typing import Optional, Dict, Any, List
  import pickle
  import numpy as np

  @dataclass
  class UnifiedSample:
      sample_id: str
      dataset_id: str                           # e.g. "finefs" or "synthetic_mock"
      performance_id: str                       # Performance ID (used for Group Split)
      athlete_id: Optional[str]                 # Athlete ID (if available, otherwise None)
      element_label: str                        # e.g. "Camel Spin", "Sit Spin", "Layback Spin"
      start_frame: int
      end_frame: int
      keypoints: np.ndarray                     # Shape (T_seg, 17, 3), dtype float32
      keypoint_confidence: Optional[np.ndarray] # Shape (T_seg, 17), dtype float32 or None
      goe: float                                # Real GOE score of the individual element
      source_metadata: Dict[str, Any] = field(default_factory=dict)

      def validate(self) -> None:
          """Raises ValueError if fields or shapes violate the unified schema constraints."""

  def save_unified_samples(samples: List[UnifiedSample], output_path: Path) -> None:
      """Validates each sample and saves the list with pickle to output_path."""

  def load_unified_samples(input_path: Path) -> List[UnifiedSample]:
      """Loads the list of UnifiedSample from input_path and re-runs validate()."""
  ```
- **HOW (Exact rules to implement inside `UnifiedSample.validate()`):**
  1. Verify `isinstance(self.sample_id, str) and len(self.sample_id) > 0`.
  2. Verify `isinstance(self.performance_id, str) and len(self.performance_id) > 0`.
  3. Verify `isinstance(self.keypoints, np.ndarray)` and convert to `np.float32` if necessary.
  4. Verify `self.keypoints.ndim == 3 and self.keypoints.shape[1] == 17 and self.keypoints.shape[2] == 3`.
  5. Verify `self.keypoints.shape[0] >= 1` and `self.end_frame >= self.start_frame`.
  6. If `self.keypoint_confidence is not None`: verify `self.keypoint_confidence.shape == (self.keypoints.shape[0], 17)`.
  7. Verify `np.isfinite(self.goe)` (must not be `NaN` or `None`).
- **CONSTRAINT:** Never remove `source_metadata`; it must contain at least `{"raw_file": ..., "raw_label": ...}`.

---

### Task 2.2 — Audit and Loader for FineFS + Mock Test Generator (`src/data/loaders.py`)
- **File to create:** `src/data/loaders.py`
- **Function contracts:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List, Optional, Tuple
  import numpy as np
  from src.data.harmonization import UnifiedSample

  def inspect_npz_file(npz_path: Path) -> Dict[str, Any]:
      """
      Opens a .npz file and returns:
      - 'valid': bool
      - 'keys': List[str]
      - 'shapes': Dict[str, Tuple[int, ...]]
      - 'dtypes': Dict[str, str]
      - 'has_nan': bool
      - 'is_3d_nonzero': bool (True if Z axis has variance > 1e-6)
      - 'error': Optional[str]
      """

  def audit_finefs_directory(raw_dir: Path) -> Dict[str, Any]:
      """
      Scans raw_dir, inspects .npz files and annotation files (.json/.csv/.pkl),
      and returns a complete report savable to reports/finefs_inventory.json.
      """

  def extract_3d_keypoints_from_array(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      """
      Normalizes the different possible shapes of HRNet/FineFS arrays into:
      - keypoints_3d: np.ndarray of shape (T, 17, 3) in float32
      - confidence: Optional[np.ndarray] of shape (T, 17) in float32
      Handles the cases:
      - shape (T, 17, 3) -> keypoints=arr, conf=None
      - shape (T, 17, 4) -> keypoints=arr[:, :, :3], conf=arr[:, :, 3]
      - shape (1, T, 17, 3) or (M, T, 17, 3) -> takes the primary person arr[0]
      - shape (T, 51) -> reshapes to (T, 17, 3)
      Raises ValueError for any other shape that cannot be mapped to 17 3D joints.
      """

  def generate_mock_finefs_samples(
      num_performances: int = 30,
      elements: Tuple[str, ...] = ("Camel Spin", "Sit Spin", "Layback Spin"),
      seed: int = 42
  ) -> List[UnifiedSample]:
      """
      Generates a collection of biomechanically plausible 17-joint UnifiedSamples (COCO-17 standard)
      with different athletes, frame-by-frame spin rotations, translations, scales, and GOE scores in [-3.0, +3.0]
      correlated with joint angle accuracy.
      Used for automated pytest tests and to verify the end-to-end pipeline.
      """
  ```
- **HOW to implement `generate_mock_finefs_samples`:**
  1. Define for each of the 3 elements (`"Camel Spin"`, `"Sit Spin"`, `"Layback Spin"`) a 17-joint 3D base template $\hat{P}_c \in \mathbb{R}^{17 \times 3}$ in COCO-17 coordinates with torso length $\approx 0.5\text{ m}$.
  2. For each performance $p \in \{0, \dots, \text{num\_performances}-1\}$ (associated with an `athlete_id = f"ath_{p // 2:03d}"` and `performance_id = f"perf_{p:04d}"`):
     - Generate one element per class $c$ with duration $T_{\text{seg}} = 25$ frames;
     - Draw a `goe` score from $[-3.0, +3.0]$;
     - If `goe < 0`, apply an angular deviation proportional to $|\text{goe}|$ (e.g., lowering the free leg or bending the knee by $5^\circ \cdot |\text{goe}|$) so that in the mock data there is a real physical relationship between skeleton geometry and GOE!
     - For each frame $t \in \{0, \dots, 24\}$, apply a spin rotation around the $Z$ axis by angle $\omega t$, a scale factor $\lambda_p \in [0.8, 1.5]$, and a global translation $\mathbf{t}_p \in [-2, 2]^3$, adding a transient in the first 4 and last 4 frames.
- **CONSTRAINT:**
  - When the `data/raw/finefs` folder contains the real FineFS files, the loader loads the real files; in unit tests (`tests/test_dataset.py`), or when running a smoke test without downloaded `.npz` files, tests use `generate_mock_finefs_samples` to deterministically verify all contracts.

---

### Task 2.3 — Anti-Leakage Partitioning by Athlete/Performance (`create_grouped_splits`)
- **Involved file:** `src/data/loaders.py`
- **Function contract:**
  ```python
  def create_grouped_splits(
      samples: List[UnifiedSample],
      data_cfg: Dict[str, Any]
  ) -> Dict[str, List[str]]:
      """
      Splits sample_ids into {'train': [...], 'val': [...], 'test': [...]}
      guaranteeing that all samples with the same athlete_id and performance_id
      fall into the exact same split.
      """
  ```
- **HOW (Unambiguous step-by-step algorithm):**
  1. For each `s` in `samples`, determine the group identifier:
     ```python
     group_id = f"ath_{s.athlete_id}" if s.athlete_id else f"perf_{s.performance_id}"
     ```
     *(Furthermore, collect all `performance_id`s associated with each `group_id` to verify that no `performance_id` is ever shared across different groups).*
  2. Group the samples by `group_id`. Sort the list of unique `group_id`s and shuffle it deterministically using `rng = np.random.default_rng(data_cfg["splits"]["seed"])`.
  3. Assign groups to `train`, `val`, and `test` respecting the proportions `train_ratio` (`0.70`), `val_ratio` (`0.15`), `test_ratio` (`0.15`), verifying that in each of the 3 splits there is at least one sample with $\text{GOE} \ge \text{median}$ and one with $\text{GOE} < \text{median}$ for every `element_label` class (repeating the shuffle up to 50 attempts if a small split turns out to lack positives or negatives for a class).
  4. Execute the **3 mandatory anti-leakage assertions** before `return`:
     ```python
     train_perfs = {s.performance_id for s in samples if s.sample_id in split_dict["train"]}
     val_perfs   = {s.performance_id for s in samples if s.sample_id in split_dict["val"]}
     test_perfs  = {s.performance_id for s in samples if s.sample_id in split_dict["test"]}
     assert train_perfs.isdisjoint(val_perfs), "LEAKAGE DETECTED: train and val share performance_id!"
     assert train_perfs.isdisjoint(test_perfs), "LEAKAGE DETECTED: train and test share performance_id!"
     assert val_perfs.isdisjoint(test_perfs), "LEAKAGE DETECTED: val and test share performance_id!"
     ```
     *(Repeat the exact same assertion on `athlete_id`s other than `None`).*
- **CONSTRAINT:** Never use `sklearn.model_selection.train_test_split` directly on the `samples` list without grouping by `group_id`.

---

## 3. Unit Tests to Implement (`tests/test_dataset.py`)

The file `tests/test_dataset.py` must implement the following 4 automated tests:
1. `test_unified_sample_validation()`: verifies that a valid `UnifiedSample` passes `.validate()`, whereas one with shape `(10, 16, 3)` or `goe = np.nan` raises `ValueError`.
2. `test_extract_3d_keypoints_shapes()`: verifies that arrays of shape `(20, 17, 3)`, `(20, 17, 4)`, `(2, 20, 17, 3)`, and `(20, 51)` are all converted to `(20, 17, 3)` `float32`.
3. `test_grouped_splits_zero_leakage()`: generates 40 performances with `generate_mock_finefs_samples()`, calls `create_grouped_splits()`, and verifies that the intersection of `performance_id` and `athlete_id` between `train`, `val`, and `test` is `set()`.
4. `test_save_and_load_unified_samples(tmp_path)`: verifies round-trip saving and reloading without metadata loss.

---

## 4. Sector 02 Acceptance Criteria (Definition of Done)

- [ ] The files `src/data/harmonization.py`, `src/data/loaders.py`, and `tests/test_dataset.py` have been created.
- [ ] Running `pytest tests/test_dataset.py` passes 100% (4/4 green tests).
