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
    Costruisce e valida un oggetto torch_geometric.data.Data completo.
    Preserva sempre le coordinate canoniche 3D in data.pos (17, 3) anche in modalità extended.
    """
    coords = np.asarray(P_bar, dtype=np.float32)
    assert coords.shape == (17, 3) and np.isfinite(coords).all(), (
        f"P_bar non valido per build_pyg_graph: shape={coords.shape}"
    )

    x = compute_node_features(coords, graph_cfg)
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
    Esegue extract_stable_window -> canonicalize_window_3d -> build_pyg_graph
    per tutti i campioni, salva train_graphs.pt, val_graphs.pt, test_graphs.pt
    in data/processed/ e restituisce {'train': [...], 'val': [...], 'test': [...]}.
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
            p_bar, _ = canonicalize_window_3d(window_joints, data_cfg)
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
        }
        graph = build_pyg_graph(p_bar, sample_meta, graph_cfg)
        grouped_graphs[split_name].append(graph)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        out_path = processed_dir / f"{split_name}_graphs.pt"
        torch.save(grouped_graphs[split_name], out_path)

    return grouped_graphs

