from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_stable_window(
    keypoints: np.ndarray,
    confidence: Optional[np.ndarray],
    data_cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Estrae la finestra centrale di frame validi dal segmento keypoints di shape (T_seg, 17, 3).
    Restituisce:
    - window_joints: np.ndarray di shape (T_win, 17, 3) in float32, oppure None se il segmento è invalido
    - info: dict con {'status': 'ok'|'discarded', 'reason': ..., 'selected_indices': [...]}
    """
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)
    assert keypoints_arr.ndim == 3 and keypoints_arr.shape[1:] == (17, 3), (
        f"Atteso keypoints di shape (T_seg, 17, 3), trovata {keypoints_arr.shape}"
    )

    conf_arr: Optional[np.ndarray] = None
    if confidence is not None:
        conf_arr = np.asarray(confidence, dtype=np.float32)
        assert conf_arr.shape == (keypoints_arr.shape[0], 17), (
            f"Attesa confidence di shape {(keypoints_arr.shape[0], 17)}, trovata {conf_arr.shape}"
        )

    w_cfg = data_cfg["windowing"]
    window_size = int(w_cfg.get("window_size", 9))
    min_valid_frames = int(w_cfg.get("min_valid_frames", 5))
    trim_start = float(w_cfg.get("trim_ratio_start", 0.20))
    trim_end = float(w_cfg.get("trim_ratio_end", 0.20))
    min_conf = float(w_cfg.get("min_keypoint_confidence", 0.20))

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

    if len(valid_indices) > window_size:
        mid = len(valid_indices) // 2
        half = window_size // 2
        start_slice = max(0, mid - half)
        chosen = valid_indices[start_slice : start_slice + window_size]
    else:
        chosen = valid_indices

    window_joints = keypoints_arr[chosen].astype(np.float32)
    assert window_joints.shape == (len(chosen), 17, 3)
    return window_joints, {
        "status": "ok",
        "reason": None,
        "selected_indices": chosen,
    }

