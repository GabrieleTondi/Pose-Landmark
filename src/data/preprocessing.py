from typing import Any, Dict, List, Tuple

import numpy as np


class DegenerateSkeletonError(ValueError):
    """Raised when a skeleton has zero/degenerate torso length or zero hip/shoulder vector."""


def canonicalize_frame_3d(
    joints: np.ndarray,
    data_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply Root-Centering, Scale Normalization, and Yaw Alignment to a single frame of shape (17, 3).
    Returns:
    - aligned: np.ndarray of shape (17, 3), dtype float32
    - stats: {'torso_length': float, 'yaw_angle_rad': float}
    Raises DegenerateSkeletonError if the frame contains NaN/Inf or denominators <= epsilon.
    """
    pts = np.asarray(joints, dtype=np.float64)
    if pts.shape != (17, 3) or not np.isfinite(pts).all():
        raise DegenerateSkeletonError("Invalid frame shape or presence of NaN/Inf.")

    c_cfg = data_cfg["canonicalization"]
    hip_l, hip_r = int(c_cfg["pelvis_indices"][0]), int(c_cfg["pelvis_indices"][1])
    sh_l, sh_r = int(c_cfg["shoulder_indices"][0]), int(c_cfg["shoulder_indices"][1])
    h_ax0, h_ax1 = int(c_cfg["horizontal_axes"][0]), int(c_cfg["horizontal_axes"][1])
    scale_eps = float(c_cfg["scale_epsilon"])
    yaw_eps = float(c_cfg["yaw_epsilon"])

    # 1. Root-Centering: move the midpoint of the hips to the origin (0, 0, 0)
    pelvis = 0.5 * (pts[hip_l] + pts[hip_r])
    centered = pts - pelvis

    # 2. Scale Normalization: divide by 3D torso length (pelvis -> shoulders_mid)
    shoulders_mid = 0.5 * (centered[sh_l] + centered[sh_r])
    torso_length = float(np.linalg.norm(shoulders_mid))
    if torso_length <= scale_eps:
        raise DegenerateSkeletonError(f"Degenerate torso length: {torso_length}")
    scaled = centered / torso_length

    # 3. Yaw Alignment on the horizontal plane (h_ax0, h_ax1)
    hip_vec = scaled[hip_r] - scaled[hip_l]
    vx = float(hip_vec[h_ax0])
    vy = float(hip_vec[h_ax1])
    horiz_norm = float(np.hypot(vx, vy))
    if horiz_norm <= yaw_eps:
        sh_vec = scaled[sh_r] - scaled[sh_l]
        vx = float(sh_vec[h_ax0])
        vy = float(sh_vec[h_ax1])
        horiz_norm = float(np.hypot(vx, vy))
        if horiz_norm <= yaw_eps:
            raise DegenerateSkeletonError(
                "Unable to determine yaw angle: hips and shoulders are vertically aligned."
            )

    angle = float(np.arctan2(vy, vx))
    cos_a = float(np.cos(-angle))
    sin_a = float(np.sin(-angle))

    aligned = scaled.copy()
    x_old = scaled[:, h_ax0]
    y_old = scaled[:, h_ax1]
    aligned[:, h_ax0] = cos_a * x_old - sin_a * y_old
    aligned[:, h_ax1] = sin_a * x_old + cos_a * y_old

    result = aligned.astype(np.float32)
    assert result.shape == (17, 3) and np.isfinite(result).all()
    return result, {"torso_length": torso_length, "yaw_angle_rad": angle}


def canonicalize_window_3d(
    window_joints: np.ndarray,
    data_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply canonicalize_frame_3d to each valid frame in the window (T_win, 17, 3)
    and then compute the mean pose P_bar = mean(P_aligned, axis=0) of shape (17, 3).
    Raises DegenerateSkeletonError if the number of successfully canonicalized frames is < min_valid_frames.
    """
    win = np.asarray(window_joints, dtype=np.float32)
    if win.ndim != 3 or win.shape[1:] != (17, 3):
        raise DegenerateSkeletonError(
            f"Expected window_joints of shape (T_win, 17, 3), got {win.shape}"
        )

    aligned_frames: List[np.ndarray] = []
    torso_lengths: List[float] = []
    for t in range(win.shape[0]):
        try:
            aligned_t, stats_t = canonicalize_frame_3d(win[t], data_cfg)
            aligned_frames.append(aligned_t)
            torso_lengths.append(stats_t["torso_length"])
        except DegenerateSkeletonError:
            continue

    min_valid_frames = int(data_cfg.get("windowing", {}).get("min_valid_frames", 3))
    if len(aligned_frames) < min_valid_frames:
        raise DegenerateSkeletonError(
            f"Too many degenerate frames in window: {len(aligned_frames)} valid < {min_valid_frames}"
        )

    stack = np.stack(aligned_frames, axis=0)
    P_bar = np.mean(stack, axis=0).astype(np.float32)
    assert P_bar.shape == (17, 3) and np.isfinite(P_bar).all()

    window_jitter = float(
        np.sqrt(np.mean(np.sum((stack - P_bar[None, :, :]) ** 2, axis=-1)))
    )
    mean_torso_length = float(np.mean(torso_lengths))

    return P_bar, {
        "num_valid_frames": len(aligned_frames),
        "mean_torso_length": mean_torso_length,
        "window_jitter": window_jitter,
    }

