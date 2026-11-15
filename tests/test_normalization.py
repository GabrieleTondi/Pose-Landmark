import numpy as np
import pytest

from src.data.preprocessing import (
    DegenerateSkeletonError,
    canonicalize_frame_3d,
    canonicalize_window_3d,
    mirror_skeleton_left_right,
)
from src.data.segmentation import extract_stable_window
from src.utils.config import load_all_configs


def _make_realistic_h36m17_pose() -> np.ndarray:
    """Generate an anatomically plausible, non-degenerate 3D H36M-17 pose."""
    pts = np.array(
        [
            [0.01, 0.00, 0.02],   # 0: pelvis
            [0.14, 0.04, 0.00],   # 1: right_hip
            [0.25, -0.35, -0.08], # 2: right_knee (free leg camel)
            [0.35, -0.78, -0.05], # 3: right_ankle
            [-0.12, -0.02, 0.02], # 4: left_hip
            [-0.15, 0.05, -0.40], # 5: left_knee
            [-0.14, 0.02, -0.82], # 6: left_ankle
            [0.02, 0.02, 0.28],   # 7: spine
            [0.01, 0.03, 0.52],   # 8: thorax
            [0.02, 0.06, 0.64],   # 9: neck
            [0.01, 0.08, 0.76],   # 10: head
            [-0.18, 0.03, 0.52],  # 11: left_shoulder
            [-0.36, 0.14, 0.44],  # 12: left_elbow
            [-0.50, 0.25, 0.55],  # 13: left_wrist
            [0.20, 0.01, 0.53],   # 14: right_shoulder
            [0.42, -0.05, 0.46],  # 15: right_elbow
            [0.58, -0.10, 0.58],  # 16: right_wrist
        ],
        dtype=np.float32,
    )
    assert pts.shape == (17, 3)
    return pts


def _rotation_matrix_z(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_root_centering_at_origin() -> None:
    cfg = load_all_configs("configs")["data"]
    hip_l, hip_r = cfg["canonicalization"]["pelvis_indices"]
    pose = _make_realistic_h36m17_pose() + np.array([12.5, -7.3, 4.1], dtype=np.float32)
    aligned, _ = canonicalize_frame_3d(pose, cfg)
    pelvis_mid = 0.5 * (aligned[hip_l] + aligned[hip_r])
    assert float(np.linalg.norm(pelvis_mid)) < 1e-6


def test_unit_torso_scale() -> None:
    cfg = load_all_configs("configs")["data"]
    hip_l, hip_r = cfg["canonicalization"]["pelvis_indices"]
    sh_l, sh_r = cfg["canonicalization"]["shoulder_indices"]
    pose = _make_realistic_h36m17_pose() * 3.7
    aligned, _ = canonicalize_frame_3d(pose, cfg)
    pelvis_mid = 0.5 * (aligned[hip_l] + aligned[hip_r])
    shoulders_mid = 0.5 * (aligned[sh_l] + aligned[sh_r])
    torso_dist = float(np.linalg.norm(shoulders_mid - pelvis_mid))
    assert abs(torso_dist - 1.0) < 1e-5


def test_yaw_alignment_along_positive_x() -> None:
    cfg = load_all_configs("configs")["data"]
    hip_l, hip_r = cfg["canonicalization"]["pelvis_indices"]
    pose = (_make_realistic_h36m17_pose().astype(np.float64) @ _rotation_matrix_z(1.23).T).astype(
        np.float32
    )
    aligned, _ = canonicalize_frame_3d(pose, cfg)
    hip_vec = aligned[hip_r] - aligned[hip_l]
    assert abs(float(hip_vec[1])) < 1e-5
    assert float(hip_vec[0]) > 0.0


def test_exact_invariance_to_translation_scale_and_z_rotation() -> None:
    cfg = load_all_configs("configs")["data"]
    rng = np.random.default_rng(42)
    base_pose = _make_realistic_h36m17_pose()
    ref_aligned, _ = canonicalize_frame_3d(base_pose, cfg)

    for _ in range(10):
        lam = float(rng.uniform(0.2, 50.0))
        phi = float(rng.uniform(-np.pi, np.pi))
        t0 = rng.uniform(-100.0, 100.0, size=(1, 3))
        rot = _rotation_matrix_z(phi)
        transformed = (lam * (base_pose.astype(np.float64) @ rot.T) + t0).astype(np.float32)
        trans_aligned, _ = canonicalize_frame_3d(transformed, cfg)
        assert np.allclose(ref_aligned, trans_aligned, atol=1e-5)


def test_rotating_spin_window_preservation() -> None:
    cfg = load_all_configs("configs")["data"]
    base_pose = _make_realistic_h36m17_pose()
    expected_canonical, _ = canonicalize_frame_3d(base_pose, cfg)

    angles_deg = np.arange(0, 360, 40, dtype=np.float64)  # 9 frames -> full 360 degrees
    assert len(angles_deg) == 9
    frames = []
    for deg in angles_deg:
        rot = _rotation_matrix_z(np.deg2rad(deg))
        frames.append((base_pose.astype(np.float64) @ rot.T).astype(np.float32))
    window_joints = np.stack(frames, axis=0)

    p_bar, stats = canonicalize_window_3d(window_joints, cfg)
    assert stats["num_valid_frames"] == 9
    assert stats["window_jitter"] < 1e-5
    assert stats["joint_jitter"].shape == (17,)
    assert float(np.max(stats["joint_jitter"])) < 1e-5
    assert np.allclose(p_bar, expected_canonical, atol=1e-5)

    zero_skeleton = np.zeros((17, 3), dtype=np.float32)
    with pytest.raises(DegenerateSkeletonError):
        canonicalize_frame_3d(zero_skeleton, cfg)


def test_bilateral_supporting_leg_mirror_invariance() -> None:
    cfg = load_all_configs("configs")["data"]
    cfg["canonicalization"]["canonicalize_supporting_leg"] = True
    left_leg_pose = _make_realistic_h36m17_pose()
    right_leg_pose = mirror_skeleton_left_right(left_leg_pose, cfg)
    rot = _rotation_matrix_z(0.85)
    right_leg_rotated = (2.3 * (right_leg_pose.astype(np.float64) @ rot.T) + np.array([3.0, -4.0, 1.5])).astype(
        np.float32
    )

    canon_left, stats_left = canonicalize_frame_3d(left_leg_pose, cfg)
    canon_right, stats_right = canonicalize_frame_3d(right_leg_rotated, cfg)

    assert stats_left["mirrored_to_canonical_leg"] is False
    assert stats_right["mirrored_to_canonical_leg"] is True
    assert np.allclose(canon_left, canon_right, atol=1e-5)
    assert float(canon_right[6, 2]) < float(canon_right[3, 2])


def test_extract_stable_window_filtering() -> None:
    cfg = load_all_configs("configs")["data"]
    base_pose = _make_realistic_h36m17_pose()
    seq = np.repeat(base_pose[None, :, :], 20, axis=0)
    conf = np.ones((20, 17), dtype=np.float32) * 0.9

    seq[10, 0, 0] = np.nan
    win, info = extract_stable_window(seq, conf, cfg)
    assert win is not None
    assert info["status"] == "ok"
    assert win.shape[0] == 9
    assert 10 not in info["selected_indices"]

    short_win, short_info = extract_stable_window(seq[:3], conf[:3], cfg)
    assert short_win is None
    assert short_info["status"] == "discarded"
    assert short_info["reason"] == "segment_too_short"
