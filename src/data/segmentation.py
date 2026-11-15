from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _window_instability_score(
    win_frames: np.ndarray,
    frame_indices: List[int],
    data_cfg: Dict[str, Any],
) -> float:
    """
    Compute a fast rotation-invariant posture instability score across a candidate window (W, 17, 3).
    Measures variance of torso-normalized joint radial distances and vertical elevations,
    plus a penalty for non-contiguous frame gaps.
    """
    c_cfg = data_cfg.get("canonicalization", {})
    pelvis_idx = c_cfg.get("pelvis_indices", [4, 1])
    shoulder_idx = c_cfg.get("shoulder_indices", [11, 14])
    v_ax = int(c_cfg.get("vertical_axis", 2))
    scale_eps = float(c_cfg.get("scale_epsilon", 1e-4))

    hip_l, hip_r = int(pelvis_idx[0]), int(pelvis_idx[1])
    sh_l, sh_r = int(shoulder_idx[0]), int(shoulder_idx[1])

    pts = np.asarray(win_frames, dtype=np.float64)
    pelvis = 0.5 * (pts[:, hip_l : hip_l + 1, :] + pts[:, hip_r : hip_r + 1, :])
    centered = pts - pelvis
    shoulders_mid = 0.5 * (centered[:, sh_l, :] + centered[:, sh_r, :])
    torso = np.linalg.norm(shoulders_mid, axis=-1, keepdims=True)
    if np.any(torso <= scale_eps):
        return float("inf")

    scaled = centered / torso[:, :, None]
    radii = np.linalg.norm(scaled, axis=-1)      # (W, 17), invariant to Z-rotation
    vert = scaled[:, :, v_ax]                    # (W, 17), invariant to Z-rotation
    posture_std = float(np.sqrt(np.mean(np.var(radii, axis=0) + np.var(vert, axis=0))))

    span = int(frame_indices[-1] - frame_indices[0]) + 1
    w_len = len(frame_indices)
    gap_penalty = 0.01 * float(max(0, span - w_len))
    return posture_std + gap_penalty


def extract_stable_window(
    keypoints: np.ndarray,
    confidence: Optional[np.ndarray],
    data_cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Extract the most stable (or central) window of valid frames from the keypoints segment of shape (T_seg, 17, 3).
    Returns:
    - window_joints: np.ndarray of shape (T_win, 17, 3) in float32, or None if the segment is invalid
    - info: dict with {'status': 'ok'|'discarded', 'reason': ..., 'selected_indices': [...], 'window_selection_score': ...}
    """
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)
    assert keypoints_arr.ndim == 3 and keypoints_arr.shape[1:] == (17, 3), (
        f"Expected keypoints of shape (T_seg, 17, 3), got {keypoints_arr.shape}"
    )

    conf_arr: Optional[np.ndarray] = None
    if confidence is not None:
        conf_arr = np.asarray(confidence, dtype=np.float32)
        assert conf_arr.shape == (keypoints_arr.shape[0], 17), (
            f"Expected confidence of shape {(keypoints_arr.shape[0], 17)}, got {conf_arr.shape}"
        )

    w_cfg = data_cfg["windowing"]
    window_size = int(w_cfg.get("window_size", 9))
    min_valid_frames = int(w_cfg.get("min_valid_frames", 5))
    trim_start = float(w_cfg.get("trim_ratio_start", 0.20))
    trim_end = float(w_cfg.get("trim_ratio_end", 0.20))
    min_conf = float(w_cfg.get("min_keypoint_confidence", 0.20))
    selection_mode = str(w_cfg.get("selection_mode", "min_jitter")).lower()

    total_frames = int(keypoints_arr.shape[0])
    if total_frames < min_valid_frames:
        return None, {
            "status": "discarded",
            "reason": "segment_too_short",
            "selected_indices": [],
        }

    start_idx = int(np.floor(total_frames * trim_start))
    end_idx = int(np.ceil(total_frames * (1.0 - trim_end)))
    if end_idx - start_idx < min_valid_frames:
        start_idx, end_idx = 0, total_frames

    valid_indices: List[int] = []
    for t in range(start_idx, end_idx):
        frame = keypoints_arr[t]
        if not np.isfinite(frame).all():
            continue
        if float(np.std(frame)) <= 1e-5:
            continue
        if conf_arr is not None:
            frame_conf = conf_arr[t]
            if not np.isfinite(frame_conf).all() or float(np.mean(frame_conf)) < min_conf:
                continue
        valid_indices.append(t)

    if len(valid_indices) < min_valid_frames:
        return None, {
            "status": "discarded",
            "reason": "insufficient_valid_frames",
            "selected_indices": [],
        }

    best_score = 0.0
    if len(valid_indices) > window_size:
        if selection_mode == "center":
            mid = len(valid_indices) // 2
            half = window_size // 2
            start_slice = max(0, mid - half)
            chosen = valid_indices[start_slice : start_slice + window_size]
            best_score = _window_instability_score(keypoints_arr[chosen], chosen, data_cfg)
        else:
            num_cands = len(valid_indices) - window_size + 1
            stride = max(1, num_cands // 40)
            best_score = float("inf")
            chosen = valid_indices[:window_size]
            for s in range(0, num_cands, stride):
                cand = valid_indices[s : s + window_size]
                score = _window_instability_score(keypoints_arr[cand], cand, data_cfg)
                if score < best_score:
                    best_score = score
                    chosen = cand
    else:
        chosen = valid_indices
        best_score = _window_instability_score(keypoints_arr[chosen], chosen, data_cfg)

    window_joints = keypoints_arr[chosen].astype(np.float32)
    assert window_joints.shape == (len(chosen), 17, 3)
    return window_joints, {
        "status": "ok",
        "reason": None,
        "selected_indices": chosen,
        "window_selection_score": float(best_score),
    }

