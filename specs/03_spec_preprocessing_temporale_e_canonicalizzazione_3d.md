# 03 — Sector Sub-Specification: Temporal Preprocessing and 3D Canonicalization

**Sector:** 03 — 3D Geometry: Temporal Windowing, Quality Control, and Skeletal Canonicalization  
**Parent Spec References:** §4, §5.1–§5.5, §16.2  
**Input Dependencies:** Sector 01 (`configs/data.yaml`), Sector 02 (`UnifiedSample`)  
**Enabled Outputs:** Sector 04 (PyG Graph Construction)

---

## 1. Sector Objective

Implement stable temporal window extraction (`src/data/segmentation.py`) and 3D geometric canonicalization (`src/data/preprocessing.py`).
**FUNDAMENTAL MATHEMATICAL RULE:** Every single frame $t$ of the selected window must undergo in order:
1. **Root-Centering** (translation of the origin to the midpoint of the hips);
2. **Scale Normalization** (division by the 3D torso length $s = \|p_{\text{shoulders}} - p_{\text{pelvis}}\|_2$);
3. **Yaw Alignment** (rotation around the vertical axis to align the left hip $\to$ right hip vector along the positive semi-axis of the first horizontal axis);
**BEFORE** computing the temporal arithmetic mean $\bar{P} = \frac{1}{T}\sum_{t=1}^T P'''_t$.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 3.1 — Stable Temporal Window Isolation (`src/data/segmentation.py`)
- **File to create:** `src/data/segmentation.py`
- **Exact function contract:**
  ```python
  from typing import Optional, Dict, Any, Tuple
  import numpy as np

  def extract_stable_window(
      keypoints: np.ndarray,
      confidence: Optional[np.ndarray],
      data_cfg: Dict[str, Any]
  ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
      """
      Extracts the central window of valid frames from the keypoints segment of shape (T_seg, 17, 3).
      Returns:
      - window_joints: np.ndarray of shape (T_win, 17, 3) in float32, or None if the segment is invalid
      - info: dict with {'status': 'ok'|'discarded', 'reason': ..., 'selected_indices': [...]}
      """
  ```
- **HOW (Step-by-step algorithm to implement):**
  1. Read from `w_cfg = data_cfg["windowing"]`:
     - `window_size = int(w_cfg["window_size"])` (default `9`)
     - `min_valid_frames = int(w_cfg["min_valid_frames"])` (default `5`)
     - `trim_start = float(w_cfg["trim_ratio_start"])` (default `0.20`)
     - `trim_end = float(w_cfg["trim_ratio_end"])` (default `0.20`)
     - `min_conf = float(w_cfg["min_keypoint_confidence"])` (default `0.20`)
  2. Let $L = \text{keypoints.shape}[0]$. If $L < \text{min\_valid\_frames}$, return `(None, {"status": "discarded", "reason": "segment_too_short", "selected_indices": []})`.
  3. Compute the indices of the central core excluding entry/exit transients:
     ```python
     start_idx = int(np.floor(L * trim_start))
     end_idx = int(np.ceil(L * (1.0 - trim_end)))
     if end_idx - start_idx < min_valid_frames:
         start_idx, end_idx = 0, L
     ```
  4. For each frame index $t \in [\text{start\_idx}, \text{end\_idx})$, verify that frame $P_t = \text{keypoints}[t]$ is valid:
     - `np.isfinite(P_t).all()` is `True`;
     - `np.std(P_t) > 1e-5` (not a skeleton collapsed to zero);
     - If `confidence is not None`: `np.mean(confidence[t]) >= min_conf`.
     Collect the indices that pass the check into the list `valid_indices`.
  5. If `len(valid_indices) < min_valid_frames`, return `(None, {"status": "discarded", "reason": "insufficient_valid_frames", "selected_indices": []})`.
  6. If `len(valid_indices) > window_size`, take the central `window_size` indices from the `valid_indices` list:
     ```python
     mid = len(valid_indices) // 2
     half = window_size // 2
     chosen = valid_indices[max(0, mid - half) : max(0, mid - half) + window_size]
     ```
     otherwise use all `valid_indices`.
  7. Return `(keypoints[chosen].astype(np.float32), {"status": "ok", "reason": None, "selected_indices": chosen})`.

---

### Task 3.2 — Frame-by-Frame 3D Geometric Canonicalization (`src/data/preprocessing.py`)
- **File to create:** `src/data/preprocessing.py`
- **Exact contracts of the exception and functions:**
  ```python
  from typing import Dict, Any, Tuple
  import numpy as np

  class DegenerateSkeletonError(ValueError):
      """Raised when a skeleton has zero/degenerate torso length or zero hip/shoulder vector."""
      pass

  def canonicalize_frame_3d(
      joints: np.ndarray,
      data_cfg: Dict[str, Any]
  ) -> Tuple[np.ndarray, Dict[str, float]]:
      """
      Applies Root-Centering, Scale Normalization, and Yaw Alignment to a single frame of shape (17, 3).
      Returns:
      - aligned: np.ndarray of shape (17, 3), dtype float32
      - stats: {'torso_length': float, 'yaw_angle_rad': float}
      Raises DegenerateSkeletonError if the frame contains NaN/Inf or denominators <= epsilon.
      """

  def canonicalize_window_3d(
      window_joints: np.ndarray,
      data_cfg: Dict[str, Any]
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
      """
      Applies canonicalize_frame_3d to each valid frame of the window (T_win, 17, 3)
      and then computes the mean pose P_bar = mean(P_aligned, axis=0) of shape (17, 3).
      Raises DegenerateSkeletonError if the number of successfully canonicalized frames is < min_valid_frames.
      """
  ```
- **HOW to implement `canonicalize_frame_3d(joints, data_cfg)` (Exact code/logic):**
  1. **Preliminary check:**
     ```python
     if joints.shape != (17, 3) or not np.isfinite(joints).all():
         raise DegenerateSkeletonError("Invalid frame shape or presence of NaN/Inf.")
     c_cfg = data_cfg["canonicalization"]
     hip_l, hip_r = c_cfg["pelvis_indices"]         # default [11, 12]
     sh_l, sh_r = c_cfg["shoulder_indices"]         # default [5, 6]
     h_ax0, h_ax1 = c_cfg["horizontal_axes"]        # default [0, 1] (X, Y)
     scale_eps = float(c_cfg["scale_epsilon"])      # default 1e-4
     yaw_eps = float(c_cfg["yaw_epsilon"])          # default 1e-4
     ```
  2. **Step 1 — Root-Centering:**
     ```python
     pelvis = 0.5 * (joints[hip_l] + joints[hip_r])
     centered = joints - pelvis  # shape (17, 3)
     ```
  3. **Step 2 — Scale Normalization:**
     ```python
     shoulders_mid = 0.5 * (centered[sh_l] + centered[sh_r])
     torso_length = float(np.linalg.norm(shoulders_mid))  # since pelvis is already at (0,0,0)
     if torso_length <= scale_eps:
         raise DegenerateSkeletonError(f"Degenerate torso length: {torso_length}")
     scaled = centered / torso_length
     ```
  4. **Step 3 — Yaw Alignment on the horizontal plane `(h_ax0, h_ax1)`:**
     ```python
     hip_vec = scaled[hip_r] - scaled[hip_l]
     vx = float(hip_vec[h_ax0])
     vy = float(hip_vec[h_ax1])
     horiz_norm = float(np.hypot(vx, vy))
     if horiz_norm <= yaw_eps:
         # Fallback on shoulder vector if hips are vertically aligned
         sh_vec = scaled[sh_r] - scaled[sh_l]
         vx = float(sh_vec[h_ax0])
         vy = float(sh_vec[h_ax1])
         horiz_norm = float(np.hypot(vx, vy))
         if horiz_norm <= yaw_eps:
             raise DegenerateSkeletonError("Unable to determine yaw angle: hips and shoulders vertically aligned.")
     angle = float(np.arctan2(vy, vx))
     c = float(np.cos(-angle))
     s = float(np.sin(-angle))
     aligned = scaled.copy()
     # Rotation on the (h_ax0, h_ax1) plane leaving the vertical axis unchanged:
     x_old = scaled[:, h_ax0]
     y_old = scaled[:, h_ax1]
     aligned[:, h_ax0] = c * x_old - s * y_old
     aligned[:, h_ax1] = s * x_old + c * y_old
     return aligned.astype(np.float32), {"torso_length": torso_length, "yaw_angle_rad": angle}
     ```
- **HOW to implement `canonicalize_window_3d(window_joints, data_cfg)`:**
  1. Iterate over frames `t` of `window_joints`, calling `canonicalize_frame_3d(window_joints[t], data_cfg)` inside a `try...except DegenerateSkeletonError` block.
  2. Collect the valid aligned frames into `aligned_frames`.
  3. If `len(aligned_frames) < data_cfg["windowing"].get("min_valid_frames", 3)`, raise `DegenerateSkeletonError("Too many degenerate frames in the window")`.
  4. Stack `stack = np.stack(aligned_frames, axis=0)` of shape `(T_valid, 17, 3)` and compute:
     - `P_bar = np.mean(stack, axis=0).astype(np.float32)` (shape `(17, 3)`)
     - `window_jitter = float(np.sqrt(np.mean(np.sum((stack - P_bar[None, :, :])**2, axis=-1))))`
  5. Return `(P_bar, {"num_valid_frames": len(aligned_frames), "mean_torso_length": ..., "window_jitter": window_jitter})`.

- **CONSTRAINT (ANTI-PATTERN TO ABSOLUTELY AVOID):**
  - **NEVER** do `else: scaled = centered` when `torso_length <= scale_eps`: it must always raise `DegenerateSkeletonError`.
  - **NEVER** average `window_joints` before calling `canonicalize_frame_3d` on each frame.

---

## 3. Unit Tests to Implement (`tests/test_normalization.py`)

The file `tests/test_normalization.py` must implement the following 5 automated tests:
1. `test_root_centering_at_origin()`: verifies that on a canonicalized skeleton the midpoint `0.5 * (P[11] + P[12])` has norm $< 10^{-6}$.
2. `test_unit_torso_scale()`: verifies that the distance between `0.5 * (P[5] + P[6])` and `0.5 * (P[11] + P[12])` after `canonicalize_frame_3d` is $1.0 \pm 10^{-5}$.
3. `test_yaw_alignment_along_positive_x()`: verifies that the vector `P[12] - P[11]` after `canonicalize_frame_3d` has component `Y == 0.0` ($\pm 10^{-5}$) and component `X > 0`.
4. `test_exact_invariance_to_translation_scale_and_z_rotation()`: given an arbitrary 3D skeleton $P$, applies 10 random transformations $P_{\text{trans}} = \lambda (P \cdot R_z(\phi)^\top) + \mathbf{t}_0$ with $\lambda \in [0.2, 50.0]$, $\phi \in [-\pi, \pi]$, $\mathbf{t}_0 \in [-100, 100]^3$ and verifies `np.allclose(canonicalize_frame_3d(P)[0], canonicalize_frame_3d(P_trans)[0], atol=1e-5)`.
5. `test_rotating_spin_window_preservation()`: simulates a 9-frame window where a fixed pose rotates by $40^\circ$ at each frame (a full $360^\circ$ revolution during a spin!): verifies that `canonicalize_window_3d` reconstructs the original pose exactly without vector cancellation (`window_jitter < 1e-5`), and verifies that an all-zero skeleton raises `DegenerateSkeletonError`.

---

## 4. Sector 03 Acceptance Criteria (Definition of Done)

- [ ] The files `src/data/segmentation.py`, `src/data/preprocessing.py`, and `tests/test_normalization.py` have been created.
- [ ] Running `pytest tests/test_normalization.py` passes 100% (5/5 green tests).
