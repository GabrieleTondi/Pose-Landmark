from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.preprocessing import DegenerateSkeletonError, canonicalize_window_3d
from src.data.segmentation import extract_stable_window
from src.graph.features import compute_edge_features, compute_node_features
from src.graph.topology import build_canonical_edge_index

if TYPE_CHECKING:
    from src.data.harmonization import UnifiedSample


def build_pyg_graph(
    P_bar: np.ndarray,
    sample_meta: Dict[str, Any],
    graph_cfg: Dict[str, Any],
) -> Data:
    """
    Build and validate a complete torch_geometric.data.Data object.
    Always preserves 3D canonical coordinates in data.pos (17, 3), even in extended mode.
    """
    coords = np.asarray(P_bar, dtype=np.float32)
    assert coords.shape == (17, 3) and np.isfinite(coords).all(), (
        f"Invalid P_bar for build_pyg_graph: shape={coords.shape}"
    )

    raw_jitter = sample_meta.get("joint_jitter", None)
    if raw_jitter is None:
        joint_jitter_arr = np.zeros((17,), dtype=np.float32)
    else:
        joint_jitter_arr = np.asarray(raw_jitter, dtype=np.float32).reshape(17)

    x = compute_node_features(coords, graph_cfg, joint_jitter=joint_jitter_arr)
    pos = torch.tensor(coords, dtype=torch.float32)
    edge_index = build_canonical_edge_index(graph_cfg)
    edge_attr = compute_edge_features(coords, graph_cfg)
    goe_val = float(sample_meta.get("goe", 0.0))
    y = torch.tensor([goe_val], dtype=torch.float32)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        sample_id=str(sample_meta["sample_id"]),
        dataset_id=str(sample_meta.get("dataset_id", "finefs")),
        performance_id=str(sample_meta["performance_id"]),
        athlete_id=str(sample_meta.get("athlete_id") or "unknown"),
        element_label=str(sample_meta["element_label"]),
        goe=goe_val,
        split=str(sample_meta.get("split", "train")),
        is_augmented=bool(sample_meta.get("is_augmented", False)),
        parent_sample_id=str(sample_meta.get("parent_sample_id") or sample_meta["sample_id"]),
        severity_tier=int(sample_meta.get("severity_tier", 0)),
        perturbation_joint=int(sample_meta.get("perturbation_joint", -1)),
        perturbation_angle_deg=float(sample_meta.get("perturbation_angle_deg", 0.0)),
        window_jitter=float(sample_meta.get("window_jitter", 0.0)),
        joint_jitter=torch.from_numpy(joint_jitter_arr.copy()).to(torch.float32),
        mirrored_to_canonical_leg=bool(sample_meta.get("mirrored_to_canonical_leg", False)),
    )

    assert data.num_nodes == 17
    assert data.pos.shape == (17, 3)
    assert data.edge_index.shape == (2, 32)
    assert data.edge_attr.shape == (32, 5)
    return data


def process_and_save_dataset_graphs(
    samples: List[UnifiedSample],
    split_dict: Dict[str, List[str]],
    data_cfg: Dict[str, Any],
    graph_cfg: Dict[str, Any],
) -> Dict[str, List[Data]]:
    """
    Run extract_stable_window -> canonicalize_window_3d -> build_pyg_graph
    for all samples, save train_graphs.pt, val_graphs.pt, test_graphs.pt
    into data/processed/, and return {'train': [...], 'val': [...], 'test': [...]}.
    """
    sample_to_split: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for sid in split_dict.get(split_name, []):
            sample_to_split[str(sid)] = split_name

    grouped_graphs: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}

    for sample in samples:
        split_name = sample_to_split.get(sample.sample_id)
        if split_name is None:
            continue

        window_joints, win_info = extract_stable_window(
            sample.keypoints, sample.keypoint_confidence, data_cfg
        )
        if window_joints is None or win_info.get("status") != "ok":
            continue

        try:
            p_bar, win_stats = canonicalize_window_3d(window_joints, data_cfg)
        except DegenerateSkeletonError:
            continue

        sample_meta = {
            "sample_id": sample.sample_id,
            "dataset_id": sample.dataset_id,
            "performance_id": sample.performance_id,
            "athlete_id": sample.athlete_id or "unknown",
            "element_label": sample.element_label,
            "goe": float(sample.goe),
            "split": split_name,
            "is_augmented": False,
            "parent_sample_id": sample.sample_id,
            "severity_tier": 0,
            "perturbation_joint": -1,
            "perturbation_angle_deg": 0.0,
            "window_jitter": float(win_stats.get("window_jitter", 0.0)),
            "joint_jitter": win_stats.get("joint_jitter"),
            "mirrored_to_canonical_leg": bool(win_stats.get("mirrored_to_canonical_leg", False)),
        }
        graph = build_pyg_graph(p_bar, sample_meta, graph_cfg)
        grouped_graphs[split_name].append(graph)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        out_path = processed_dir / f"{split_name}_graphs.pt"
        torch.save(grouped_graphs[split_name], out_path)

    return grouped_graphs


def main() -> None:
    """CLI entrypoint: rebuild train/val/test PyG graphs from FineFS (or intermediate pickle)."""
    import json
    from src.data.harmonization import load_unified_samples, save_unified_samples
    from src.data.loaders import create_grouped_splits, generate_mock_finefs_samples, load_finefs_samples
    from src.utils.config import load_all_configs

    cfgs = load_all_configs("configs")
    data_cfg = cfgs["data"]
    graph_cfg = cfgs["graph"]

    inter_pkl = Path(data_cfg["paths"]["intermediate_dir"]) / "unified_samples.pkl"
    raw_dir = Path(data_cfg["paths"]["raw_finefs_dir"])
    splits_json = Path(data_cfg["paths"]["splits_dir"]) / "grouped_splits.json"

    if inter_pkl.is_file():
        samples = load_unified_samples(inter_pkl)
    elif (raw_dir / "skeleton").is_dir() and (raw_dir / "annotation").is_dir():
        samples, _ = load_finefs_samples(raw_dir, data_cfg)
        save_unified_samples(samples, inter_pkl)
    else:
        samples = generate_mock_finefs_samples(num_performances=30, seed=42)

    if splits_json.is_file():
        split_dict = json.loads(splits_json.read_text(encoding="utf-8"))
    else:
        split_dict = create_grouped_splits(samples, data_cfg)
        splits_json.parent.mkdir(parents=True, exist_ok=True)
        splits_json.write_text(json.dumps(split_dict, indent=2), encoding="utf-8")

    grouped = process_and_save_dataset_graphs(samples, split_dict, data_cfg, graph_cfg)
    print(
        json.dumps(
            {k: len(v) for k, v in grouped.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

