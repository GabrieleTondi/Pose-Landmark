from typing import Any, Dict, List, Tuple

import numpy as np
from torch_geometric.data import Data

from src.data.preprocessing import DegenerateSkeletonError, canonicalize_frame_3d
from src.graph.builder import build_pyg_graph
from src.training.triplet_sampler import resolve_class_thresholds


def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix in SO(3) around vector 'axis' by angle 'angle_rad'
    using Rodrigues' rotation formula: R = I + sin(theta)*K + (1 - cos(theta))*(K @ K).
    Returns an np.ndarray of shape (3, 3) float32.
    """
    u = np.asarray(axis, dtype=np.float64)
    assert u.shape == (3,), f"Expected axis of shape (3,), got {u.shape}."
    norm = float(np.linalg.norm(u))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    u = u / norm
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    k_mat = np.array(
        [
            [0.0, -uz, uy],
            [uz, 0.0, -ux],
            [-uy, ux, 0.0],
        ],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    theta = float(angle_rad)
    rot = eye + np.sin(theta) * k_mat + (1.0 - np.cos(theta)) * (k_mat @ k_mat)
    out = rot.astype(np.float32)
    assert out.shape == (3, 3) and np.isfinite(out).all()
    return out


def rotate_kinematic_subtree(
    P_bar: np.ndarray,
    joint_idx: int,
    child_idx: int,
    axis: np.ndarray,
    angle_deg: float,
    graph_cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Rigidly rotate all nodes belonging to the subtree
    kinematic_subtrees[f"{joint_idx}->{child_idx}"] around pivot P_bar[joint_idx] by 'angle_deg' degrees.
    Returns the new matrix P_prime of shape (17, 3) float32 with unchanged bone lengths.
    """
    pose = np.asarray(P_bar, dtype=np.float32)
    assert pose.shape == (17, 3) and np.isfinite(pose).all(), (
        f"Expected finite P_bar of shape (17, 3), got {pose.shape}."
    )

    p_idx = int(joint_idx)
    c_idx = int(child_idx)
    key = f"{p_idx}->{c_idx}"
    subtree_map = graph_cfg.get("kinematic_subtrees", {})
    subtree_nodes = [int(idx) for idx in subtree_map.get(key, [c_idx])]

    rot = rodrigues_rotation_matrix(axis, float(np.deg2rad(float(angle_deg))))
    p_prime = pose.copy()
    pivot = pose[p_idx]
    for node_idx in subtree_nodes:
        p_prime[node_idx] = pivot + rot @ (pose[node_idx] - pivot)

    assert p_prime.shape == (17, 3) and np.isfinite(p_prime).all()
    return p_prime


def _select_weighted_subtree_keys(graph_cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    """
    Dynamically prioritize lower- and upper-limb joints over the trunk,
    reading joint names from graph_cfg without hardcoding coco17/h36m17 indices.
    """
    subtrees = graph_cfg["kinematic_subtrees"]
    joint_names = graph_cfg.get("joint_names", {})
    keys = list(subtrees.keys())
    if len(keys) == 0:
        raise ValueError("graph_cfg['kinematic_subtrees'] cannot be empty.")

    weights: List[float] = []
    for key in keys:
        p_str = key.split("->")[0]
        p_name = str(joint_names.get(int(p_str), joint_names.get(p_str, ""))).lower()
        subtree_len = len(subtrees[key])
        if any(tok in p_name for tok in ("hip", "knee")):
            weights.append(2.5)
        elif any(tok in p_name for tok in ("shoulder", "elbow")):
            weights.append(1.2)
        elif subtree_len <= 2:
            weights.append(1.5)
        else:
            weights.append(0.3)

    probs = np.asarray(weights, dtype=np.float64)
    probs /= float(np.sum(probs))
    return keys, probs


def _orthogonal_unit_axis(
    pose: np.ndarray,
    joint_idx: int,
    child_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a unit axis orthogonal to the bone pose[child_idx] - pose[joint_idx]."""
    bone_vec = (pose[child_idx] - pose[joint_idx]).astype(np.float64)
    norm = float(np.linalg.norm(bone_vec))
    if norm < 1e-6:
        vec = rng.normal(0.0, 1.0, size=3)
        return (vec / (float(np.linalg.norm(vec)) + 1e-12)).astype(np.float32)

    bone_dir = bone_vec / norm
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(bone_dir, ref))) > 0.90:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    e1 = np.cross(bone_dir, ref)
    e1 /= float(np.linalg.norm(e1)) + 1e-12
    e2 = np.cross(bone_dir, e1)
    e2 /= float(np.linalg.norm(e2)) + 1e-12

    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    axis = np.cos(phi) * e1 + np.sin(phi) * e2
    axis /= float(np.linalg.norm(axis)) + 1e-12
    return axis.astype(np.float32)


def _select_positive_sources_for_augmentation(
    real_train_graphs: List[Data],
    train_cfg: Dict[str, Any],
) -> List[Data]:
    """Select real positive graphs from the training set to generate synthetic negatives."""
    abs_pos = float(train_cfg["sampling"].get("goe_pos_threshold", 1.5))
    abs_neg = float(train_cfg["sampling"].get("goe_neg_threshold", 0.0))

    by_class: Dict[str, List[Data]] = {}
    for g in real_train_graphs:
        by_class.setdefault(str(g.element_label), []).append(g)

    positives: List[Data] = []
    for cls_graphs in by_class.values():
        pos_abs = [g for g in cls_graphs if float(g.goe) >= abs_pos]
        neg_abs = [g for g in cls_graphs if float(g.goe) <= abs_neg]
        if len(neg_abs) == 0 and len(pos_abs) == len(cls_graphs):
            positives.extend(pos_abs)
            continue

        _, tau_pos = resolve_class_thresholds(cls_graphs, train_cfg)
        positives.extend(g for g in cls_graphs if float(g.goe) >= tau_pos)

    return positives


def generate_synthetic_negatives(
    train_graphs: List[Data],
    data_cfg: Dict[str, Any],
    graph_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> List[Data]:
    """
    For each real positive graph in the training set ONLY (split == 'train' and not is_augmented),
    generate synthetic negative graphs (default 3, one per severity_tier 1, 2, 3)
    and return the list of new synthetic Data objects.
    """
    for g in train_graphs:
        split_val = str(getattr(g, "split", ""))
        if split_val != "train":
            raise ValueError(
                f"Data leakage blocked: generate_synthetic_negatives only accepts graphs "
                f"with split == 'train', found split='{split_val}' ({getattr(g, 'sample_id', 'unknown')})."
            )

    aug_cfg = train_cfg.get("augmentation", {})
    if not bool(aug_cfg.get("enabled", True)):
        return []

    real_train = [g for g in train_graphs if not bool(getattr(g, "is_augmented", False))]
    if len(real_train) == 0:
        return []

    source_positives = _select_positive_sources_for_augmentation(real_train, train_cfg)
    if len(source_positives) == 0:
        return []

    tier_ranges = aug_cfg["tier_angle_ranges_deg"]
    noise_std = float(aug_cfg.get("noise_std", 0.005))
    num_per_pos = int(aug_cfg.get("num_synthetic_per_positive", 3))
    seed = int(train_cfg.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    keys, key_probs = _select_weighted_subtree_keys(graph_cfg)
    tiers_cycle = [1, 2, 3]
    synthetic_graphs: List[Data] = []

    for g in source_positives:
        base_pos = g.pos.detach().cpu().numpy().astype(np.float32)

        for rep_idx in range(num_per_pos):
            tier = tiers_cycle[rep_idx % 3]
            angle_bounds = tier_ranges.get(tier, tier_ranges.get(str(tier)))
            low_deg, high_deg = float(angle_bounds[0]), float(angle_bounds[1])

            chosen_key = str(rng.choice(keys, p=key_probs))
            joint_idx, child_idx = (int(tok) for tok in chosen_key.split("->"))
            sign = float(rng.choice([-1.0, 1.0]))
            angle_deg = float(rng.uniform(low_deg, high_deg)) * sign
            axis = _orthogonal_unit_axis(base_pos, joint_idx, child_idx, rng)

            p_pert = rotate_kinematic_subtree(
                base_pos, joint_idx, child_idx, axis, angle_deg, graph_cfg
            )
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=p_pert.shape).astype(np.float32)
                p_pert = p_pert + noise

            try:
                p_pert_canon, _ = canonicalize_frame_3d(p_pert, data_cfg)
            except DegenerateSkeletonError:
                p_pert_no_noise = rotate_kinematic_subtree(
                    base_pos, joint_idx, child_idx, axis, angle_deg, graph_cfg
                )
                p_pert_canon, _ = canonicalize_frame_3d(p_pert_no_noise, data_cfg)

            suffix = f"t{tier}" if num_per_pos <= 3 else f"t{tier}_{rep_idx}"
            meta = {
                "sample_id": f"{g.sample_id}_aug_{suffix}",
                "dataset_id": str(getattr(g, "dataset_id", "finefs")),
                "performance_id": str(g.performance_id),
                "athlete_id": str(getattr(g, "athlete_id", "unknown")),
                "element_label": str(g.element_label),
                "goe": float(g.goe) - float(tier),
                "split": "train",
                "is_augmented": True,
                "parent_sample_id": str(g.sample_id),
                "severity_tier": int(tier),
                "perturbation_joint": int(joint_idx),
                "perturbation_angle_deg": float(angle_deg),
            }
            synth_graph = build_pyg_graph(p_pert_canon, meta, graph_cfg)
            synthetic_graphs.append(synth_graph)

    return synthetic_graphs

