import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data.loaders import create_grouped_splits, generate_mock_finefs_samples
from src.data.preprocessing import canonicalize_window_3d
from src.data.segmentation import extract_stable_window
from src.graph.builder import build_pyg_graph, process_and_save_dataset_graphs
from src.graph.topology import get_directed_edge_list
from src.models.siamese_gat import SiameseGATv2
from src.training.augmentation import generate_synthetic_negatives, rotate_kinematic_subtree
from src.training.losses import AdaptiveTripletMarginLoss
from src.training.train import evaluate_triplets, run_training_pipeline, train_one_epoch
from src.training.triplet_sampler import TripletDataset, partition_by_goe_tiers
from src.utils.config import load_all_configs
from src.utils.seed import seed_everything


def _prepare_mock_graphs_by_split(tmp_path: Path, num_performances: int = 60, seed: int = 42):
    cfgs = load_all_configs("configs")
    data_cfg = copy.deepcopy(cfgs["data"])
    data_cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    data_cfg["paths"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    data_cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    graph_cfg = cfgs["graph"]
    train_cfg = copy.deepcopy(cfgs["training"])

    samples = generate_mock_finefs_samples(num_performances=num_performances, seed=seed)
    split_dict = create_grouped_splits(samples, data_cfg)
    grouped = process_and_save_dataset_graphs(samples, split_dict, data_cfg, graph_cfg)
    return grouped, data_cfg, graph_cfg, train_cfg


def test_rodrigues_subtree_preserves_bone_lengths() -> None:
    cfgs = load_all_configs("configs")
    data_cfg = cfgs["data"]
    graph_cfg = cfgs["graph"]

    sample = generate_mock_finefs_samples(num_performances=1, elements=("Sit Spin",), seed=7)[0]
    win, info = extract_stable_window(sample.keypoints, sample.keypoint_confidence, data_cfg)
    assert win is not None and info["status"] == "ok"
    p_bar, _ = canonicalize_window_3d(win, data_cfg)

    meta = {
        "sample_id": sample.sample_id,
        "performance_id": sample.performance_id,
        "athlete_id": sample.athlete_id,
        "element_label": sample.element_label,
        "goe": sample.goe,
        "split": "train",
    }
    g_orig = build_pyg_graph(p_bar, meta, graph_cfg)

    # Sotto-albero H36M-17 anca SX -> ginocchio SX e caviglia SX ("4->5": [5, 6])
    thigh_vec = (p_bar[5] - p_bar[4]).astype(np.float64)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(thigh_vec / (np.linalg.norm(thigh_vec) + 1e-8), ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ortho_axis = np.cross(thigh_vec, ref)
    ortho_axis = (ortho_axis / (np.linalg.norm(ortho_axis) + 1e-8)).astype(np.float32)

    p_rot = rotate_kinematic_subtree(
        p_bar, joint_idx=4, child_idx=5, axis=ortho_axis, angle_deg=30.0, graph_cfg=graph_cfg
    )
    g_rot = build_pyg_graph(p_rot, meta, graph_cfg)

    lengths_diff = torch.abs(g_orig.edge_attr[:, 3] - g_rot.edge_attr[:, 3]).max().item()
    assert lengths_diff < 1e-5, f"Bone lengths altered after Rodrigues rotation: max_diff={lengths_diff}"

    directed_edges = get_directed_edge_list(graph_cfg)
    left_hip_edge_idx = directed_edges.index((4, 5))
    cos_before = float(g_orig.edge_attr[left_hip_edge_idx, 4].item())
    cos_after = float(g_rot.edge_attr[left_hip_edge_idx, 4].item())
    assert abs(cos_before - cos_after) > 1e-2, (
        f"Expected significant angle change at left hip: before={cos_before}, after={cos_after}"
    )


def test_triplet_dataset_intra_class_and_no_val_augmentation(tmp_path: Path) -> None:
    grouped, data_cfg, graph_cfg, train_cfg = _prepare_mock_graphs_by_split(tmp_path)
    train_graphs = grouped["train"]
    val_graphs = grouped["val"]

    with pytest.raises(ValueError):
        generate_synthetic_negatives(val_graphs, data_cfg, graph_cfg, train_cfg)

    synth_graphs = generate_synthetic_negatives(train_graphs, data_cfg, graph_cfg, train_cfg)
    assert len(synth_graphs) > 0
    assert all(bool(g.is_augmented) and g.split == "train" for g in synth_graphs)

    train_pools, thresholds = partition_by_goe_tiers(
        train_graphs + synth_graphs, train_cfg, fit_thresholds=None
    )
    val_pools, _ = partition_by_goe_tiers(val_graphs, train_cfg, fit_thresholds=thresholds)

    train_ds = TripletDataset(train_pools, train_cfg, num_triplets=100, split="train", seed=42)
    val_ds = TripletDataset(val_pools, train_cfg, num_triplets=100, split="val", seed=42)

    assert len(train_ds) == 100 and len(val_ds) == 100
    for anchor, pos, neg, alpha in train_ds:
        assert anchor.element_label == pos.element_label == neg.element_label
        assert anchor.performance_id != pos.performance_id
        assert float(alpha.item()) > 0.0

    for anchor, pos, neg, alpha in val_ds:
        assert anchor.element_label == pos.element_label == neg.element_label
        assert anchor.performance_id != pos.performance_id
        assert not bool(anchor.is_augmented)
        assert not bool(pos.is_augmented)
        assert not bool(neg.is_augmented)
        assert float(alpha.item()) > 0.0


def test_adaptive_triplet_loss_backward_and_zero_distance_safety() -> None:
    criterion = AdaptiveTripletMarginLoss()
    z_a = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    z_p = z_a.detach().clone()  # distanza esattamente zero!
    z_n = torch.tensor([[1.1, -0.5, 0.25], [0.2, -0.1, 0.0]], dtype=torch.float32)
    margins = torch.tensor([0.20, 1.00], dtype=torch.float32)

    loss, stats = criterion(z_a, z_p, z_n, margins)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_a.grad is not None and torch.isfinite(z_a.grad).all()
    assert set(stats.keys()) == {
        "loss",
        "triplet_acc",
        "margin_satisfied_acc",
        "mean_d_ap",
        "mean_d_an",
    }


def test_pyg_dataloader_on_triplet_dataset(tmp_path: Path) -> None:
    grouped, data_cfg, graph_cfg, train_cfg = _prepare_mock_graphs_by_split(tmp_path)
    synth = generate_synthetic_negatives(grouped["train"], data_cfg, graph_cfg, train_cfg)
    train_pools, _ = partition_by_goe_tiers(grouped["train"] + synth, train_cfg, fit_thresholds=None)
    triplet_ds = TripletDataset(train_pools, train_cfg, num_triplets=16, split="train", seed=42)

    loader = DataLoader(triplet_ds, batch_size=8, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2

    batch_a, batch_p, batch_n, batch_alpha = batches[0]
    assert batch_a.num_graphs == 8
    assert batch_p.num_graphs == 8
    assert batch_n.num_graphs == 8
    assert batch_a.x.shape == (8 * 17, int(graph_cfg["node_feature_dim"]))
    assert batch_alpha.shape == (8,) and batch_alpha.dtype == torch.float32


def test_siamese_shared_weights_and_edge_attr_sensitivity(tmp_path: Path) -> None:
    seed_everything(42)
    grouped, _, _, train_cfg = _prepare_mock_graphs_by_split(tmp_path, num_performances=6)
    model = SiameseGATv2.from_config(train_cfg)
    model.eval()

    assert len(list(model.parameters())) == len(list(model.encoder.parameters()))

    g1 = grouped["train"][0].clone()
    g2 = grouped["train"][0].clone()
    g2.edge_attr = g2.edge_attr.clone() + 0.5

    with torch.no_grad():
        z1 = model.encode(g1)
        z2 = model.encode(g2)

    assert z1.shape == (1, int(train_cfg["model"]["embed_dim"]))
    assert not torch.allclose(z1, z2, atol=1e-5)


def test_attention_weights_extraction_shape(tmp_path: Path) -> None:
    seed_everything(42)
    grouped, _, _, train_cfg = _prepare_mock_graphs_by_split(tmp_path, num_performances=6)
    model = SiameseGATv2.from_config(train_cfg)
    model.eval()

    single_graph = grouped["train"][0].clone()
    with torch.no_grad():
        z, attn = model.encode(single_graph, return_attention_weights=True)

    heads = int(train_cfg["model"]["heads"])
    assert z.shape == (1, int(train_cfg["model"]["embed_dim"]))
    assert set(attn.keys()) == {"layer1", "layer2"}
    assert attn["layer1"][0].shape == (2, 49)
    assert attn["layer1"][1].shape == (49, heads)
    assert attn["layer2"][0].shape == (2, 49)
    assert attn["layer2"][1].shape == (49, heads)


def test_single_batch_overfit(tmp_path: Path) -> None:
    seed_everything(42)
    grouped, data_cfg, graph_cfg, train_cfg = _prepare_mock_graphs_by_split(tmp_path)
    synth = generate_synthetic_negatives(grouped["train"], data_cfg, graph_cfg, train_cfg)
    train_pools, _ = partition_by_goe_tiers(grouped["train"] + synth, train_cfg, fit_thresholds=None)

    fixed_ds = TripletDataset(train_pools, train_cfg, num_triplets=16, split="train", seed=42)
    fixed_loader = DataLoader(fixed_ds, batch_size=16, shuffle=False)

    device = torch.device("cpu")
    model = SiameseGATv2.from_config(train_cfg).to(device)
    criterion = AdaptiveTripletMarginLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    final_stats = {}
    for _ in range(35):
        final_stats = train_one_epoch(
            model, fixed_loader, criterion, optimizer, device, max_grad_norm=1.0
        )

    eval_stats = evaluate_triplets(model, fixed_loader, criterion, device)
    assert final_stats["loss"] < 0.15, (
        f"Overfit su 16 triplette non riuscito: train_loss={final_stats['loss']:.4f}"
    )
    assert eval_stats["val_loss"] < 0.15, (
        f"Overfit eval su 16 triplette non riuscito: eval_loss={eval_stats['val_loss']:.4f}"
    )
    assert final_stats["triplet_acc"] >= 0.90, (
        f"Train accuracy attesa >= 0.90, ottenuta {final_stats['triplet_acc']:.4f}"
    )
    assert eval_stats["val_triplet_acc"] >= 0.90, (
        f"Eval accuracy attesa >= 0.90, ottenuta {eval_stats['val_triplet_acc']:.4f}"
    )


def test_run_training_pipeline_end_to_end_mock(tmp_path: Path) -> None:
    grouped, data_cfg, graph_cfg, train_cfg = _prepare_mock_graphs_by_split(
        tmp_path, num_performances=60, seed=42
    )
    train_cfg["training"]["max_epochs"] = 3
    train_cfg["training"]["num_train_triplets_per_epoch"] = 64
    train_cfg["training"]["num_val_triplets"] = 32

    result = run_training_pipeline(
        grouped["train"], grouped["val"], data_cfg, graph_cfg, train_cfg
    )

    ckpt_path = Path(data_cfg["paths"]["checkpoints_dir"]) / "best_model.pt"
    hist_path = Path(data_cfg["paths"]["reports_dir"]) / "training_history.json"
    assert ckpt_path.is_file()
    assert hist_path.is_file()
    assert result["epochs_completed"] == 3
    assert result["best_embedding_std"] > 0.0

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in (
        "epoch",
        "model_state_dict",
        "val_loss",
        "val_triplet_acc",
        "goe_thresholds",
        "train_cfg",
        "graph_cfg",
        "data_cfg",
    ):
        assert key in ckpt

    history = json.loads(hist_path.read_text(encoding="utf-8"))
    assert len(history) == 3

