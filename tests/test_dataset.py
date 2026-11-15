import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.harmonization import (
    UnifiedSample,
    load_unified_samples,
    save_unified_samples,
)
from src.data.loaders import (
    audit_finefs_directory,
    create_grouped_splits,
    extract_3d_keypoints_from_array,
    generate_mock_finefs_samples,
    inspect_npz_file,
    load_finefs_samples,
)
from src.graph.builder import process_and_save_dataset_graphs
from src.utils.config import load_all_configs


def test_unified_sample_validation() -> None:
    valid = UnifiedSample(
        sample_id="s_001",
        dataset_id="finefs",
        performance_id="perf_001",
        athlete_id="ath_001",
        element_label="Camel Spin",
        start_frame=10,
        end_frame=19,
        keypoints=np.zeros((10, 17, 3), dtype=np.float32),
        keypoint_confidence=np.ones((10, 17), dtype=np.float32),
        goe=1.5,
        source_metadata={"raw_file": "clip_01.npz", "raw_label": "Camel Spin"},
    )
    valid.validate()

    bad_shape = copy.deepcopy(valid)
    bad_shape.keypoints = np.zeros((10, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        bad_shape.validate()

    bad_goe = copy.deepcopy(valid)
    bad_goe.goe = float(np.nan)
    with pytest.raises(ValueError):
        bad_goe.validate()


def test_extract_3d_keypoints_shapes() -> None:
    arr_3 = np.random.randn(20, 17, 3).astype(np.float64)
    kpts_3, conf_3 = extract_3d_keypoints_from_array(arr_3)
    assert kpts_3.shape == (20, 17, 3) and kpts_3.dtype == np.float32
    assert conf_3 is None

    arr_4 = np.random.randn(20, 17, 4).astype(np.float64)
    kpts_4, conf_4 = extract_3d_keypoints_from_array(arr_4)
    assert kpts_4.shape == (20, 17, 3) and kpts_4.dtype == np.float32
    assert conf_4 is not None and conf_4.shape == (20, 17) and conf_4.dtype == np.float32

    arr_m = np.random.randn(2, 20, 17, 3).astype(np.float32)
    kpts_m, conf_m = extract_3d_keypoints_from_array(arr_m)
    assert kpts_m.shape == (20, 17, 3) and kpts_m.dtype == np.float32
    assert conf_m is None
    assert np.allclose(kpts_m, arr_m[0])

    arr_51 = np.random.randn(20, 51).astype(np.float64)
    kpts_51, conf_51 = extract_3d_keypoints_from_array(arr_51)
    assert kpts_51.shape == (20, 17, 3) and kpts_51.dtype == np.float32
    assert conf_51 is None

    with pytest.raises(ValueError):
        extract_3d_keypoints_from_array(np.zeros((20, 16, 3), dtype=np.float32))


def test_grouped_splits_zero_leakage() -> None:
    data_cfg = load_all_configs("configs")["data"]
    samples = generate_mock_finefs_samples(num_performances=40, seed=42)
    split_dict = create_grouped_splits(samples, data_cfg)

    assert set(split_dict.keys()) == {"train", "val", "test"}
    total_assigned = (
        len(split_dict["train"]) + len(split_dict["val"]) + len(split_dict["test"])
    )
    assert total_assigned == len(samples)

    by_id = {s.sample_id: s for s in samples}
    perfs = {
        split: {by_id[sid].performance_id for sid in sids}
        for split, sids in split_dict.items()
    }
    aths = {
        split: {by_id[sid].athlete_id for sid in sids if by_id[sid].athlete_id is not None}
        for split, sids in split_dict.items()
    }

    assert perfs["train"] & perfs["val"] == set()
    assert perfs["train"] & perfs["test"] == set()
    assert perfs["val"] & perfs["test"] == set()

    assert aths["train"] & aths["val"] == set()
    assert aths["train"] & aths["test"] == set()
    assert aths["val"] & aths["test"] == set()


def test_save_and_load_unified_samples(tmp_path: Path) -> None:
    samples = generate_mock_finefs_samples(num_performances=4, seed=7)
    out_file = tmp_path / "unified_samples.pkl"
    save_unified_samples(samples, out_file)
    reloaded = load_unified_samples(out_file)

    assert len(reloaded) == len(samples)
    for orig, loaded in zip(samples, reloaded):
        assert loaded.sample_id == orig.sample_id
        assert loaded.performance_id == orig.performance_id
        assert loaded.athlete_id == orig.athlete_id
        assert loaded.element_label == orig.element_label
        assert loaded.goe == orig.goe
        assert loaded.source_metadata == orig.source_metadata
        assert np.array_equal(loaded.keypoints, orig.keypoints)
        assert orig.keypoint_confidence is not None and loaded.keypoint_confidence is not None
        assert np.array_equal(loaded.keypoint_confidence, orig.keypoint_confidence)


def test_inspect_npz_and_audit_directory(tmp_path: Path) -> None:
    kpts = np.random.randn(15, 17, 3).astype(np.float32)
    npz_path = tmp_path / "clip_valid.npz"
    np.savez(npz_path, keypoints=kpts)
    (tmp_path / "annotations.json").write_text(
        json.dumps({"clips": ["clip_valid"]}), encoding="utf-8"
    )
    bad_npz = tmp_path / "corrupt.npz"
    bad_npz.write_bytes(b"not_a_zip_file")

    info = inspect_npz_file(npz_path)
    assert info["valid"] is True
    assert info["keys"] == ["keypoints"]
    assert info["has_nan"] is False
    assert info["is_3d_nonzero"] is True

    report = audit_finefs_directory(tmp_path)
    assert report["directory_exists"] is True
    assert report["status"] == "ready"
    assert report["num_npz_files"] == 2
    assert report["num_valid_npz"] == 1
    assert report["num_invalid_npz"] == 1
    assert report["num_annotation_files"] == 1


def test_end_to_end_mock_to_pyg_graphs(tmp_path: Path) -> None:
    cfgs = load_all_configs("configs")
    data_cfg = copy.deepcopy(cfgs["data"])
    data_cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    graph_cfg = cfgs["graph"]

    samples = generate_mock_finefs_samples(num_performances=12, seed=42)
    split_dict = create_grouped_splits(samples, data_cfg)
    graphs_by_split = process_and_save_dataset_graphs(samples, split_dict, data_cfg, graph_cfg)

    assert len(graphs_by_split["train"]) > 0
    assert len(graphs_by_split["val"]) > 0
    assert len(graphs_by_split["test"]) > 0

    train_pt = Path(data_cfg["paths"]["processed_dir"]) / "train_graphs.pt"
    assert train_pt.is_file()
    reloaded = torch.load(train_pt, map_location="cpu", weights_only=False)
    assert len(reloaded) == len(graphs_by_split["train"])
    assert reloaded[0].num_nodes == 17
    assert reloaded[0].edge_index.shape == (2, 32)
    assert reloaded[0].edge_attr.shape == (32, 5)


def test_load_real_finefs_samples_and_splits() -> None:
    data_cfg = load_all_configs("configs")["data"]
    raw_dir = Path(data_cfg["paths"]["raw_finefs_dir"])
    if not (raw_dir / "skeleton" / "0.npz").is_file():
        pytest.skip("Real FineFS dataset not present in data/raw/finefs.")

    samples, report = load_finefs_samples(raw_dir, data_cfg)
    assert report["num_samples_loaded"] == len(samples)
    assert len(samples) > 1000
    assert set(report["kept_per_class"].keys()) == set(data_cfg["dataset"]["target_elements"])

    split_dict = create_grouped_splits(samples, data_cfg)
    assert len(split_dict["train"]) > 0
    assert len(split_dict["val"]) > 0
    assert len(split_dict["test"]) > 0
