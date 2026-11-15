import copy
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.preprocessing import canonicalize_frame_3d
from src.graph.builder import build_pyg_graph
from src.graph.features import compute_edge_features
from src.graph.topology import build_canonical_edge_index
from src.utils.config import load_all_configs


def _sample_canonical_pose() -> np.ndarray:
    cfg = load_all_configs("configs")["data"]
    pts = np.array(
        [
            [0.01, 0.00, 0.02],   # 0: pelvis
            [0.14, 0.04, 0.00],   # 1: right_hip
            [0.25, -0.35, -0.08], # 2: right_knee
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
    aligned, _ = canonicalize_frame_3d(pts, cfg)
    return aligned


def test_topology_paired_ordering() -> None:
    graph_cfg = load_all_configs("configs")["graph"]
    edge_index = build_canonical_edge_index(graph_cfg)
    assert edge_index.shape == (2, 32)
    assert edge_index.dtype == torch.long

    for b in range(16):
        assert int(edge_index[0, 2 * b]) == int(edge_index[1, 2 * b + 1])
        assert int(edge_index[1, 2 * b]) == int(edge_index[0, 2 * b + 1])


def test_edge_features_geometry() -> None:
    graph_cfg = load_all_configs("configs")["graph"]
    p_bar = _sample_canonical_pose()
    edge_attr = compute_edge_features(p_bar, graph_cfg)

    assert edge_attr.shape == (32, 5)
    assert edge_attr.dtype == torch.float32
    assert torch.isfinite(edge_attr).all()

    norms = torch.norm(edge_attr[:, 0:3], dim=-1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-4)

    for b in range(16):
        assert torch.allclose(edge_attr[2 * b, 0:3], -edge_attr[2 * b + 1, 0:3], atol=1e-5)

    assert bool((edge_attr[:, 3] > 0.0).all())
    assert bool((edge_attr[:, 4] >= -1.0).all() and (edge_attr[:, 4] <= 1.0).all())


def test_pyg_data_and_batching() -> None:
    graph_cfg = load_all_configs("configs")["graph"]
    p_bar = _sample_canonical_pose()
    graphs = []
    for idx in range(8):
        meta = {
            "sample_id": f"mock_{idx:03d}",
            "dataset_id": "synthetic_mock",
            "performance_id": f"perf_{idx // 2:03d}",
            "athlete_id": f"ath_{idx // 4:02d}",
            "element_label": "Camel Spin",
            "goe": float(idx - 3) * 0.5,
            "split": "train",
        }
        graphs.append(build_pyg_graph(p_bar, meta, graph_cfg))

    loader = DataLoader(graphs, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch.x.shape == (68, 3)
    assert batch.pos.shape == (68, 3)
    assert batch.edge_index.shape == (2, 128)
    assert batch.edge_attr.shape == (128, 5)


def test_extended_node_features_preserves_pos() -> None:
    graph_cfg = copy.deepcopy(load_all_configs("configs")["graph"])
    graph_cfg["node_feature_mode"] = "extended"
    p_bar = _sample_canonical_pose()

    meta = {
        "sample_id": "mock_ext_001",
        "performance_id": "perf_001",
        "element_label": "Sit Spin",
        "goe": 1.5,
    }
    data = build_pyg_graph(p_bar, meta, graph_cfg)

    assert data.x.shape == (17, 5)
    assert data.pos.shape == (17, 3)
    assert torch.allclose(data.x[:, 0:3], data.pos, atol=1e-6)
