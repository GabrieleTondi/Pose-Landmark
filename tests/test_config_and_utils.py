import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from src.utils import env_check
from src.utils.config import load_all_configs, load_yaml_config
from src.utils.hardware import get_compute_device, get_hardware_profile, get_vram_usage_mb
from src.utils.seed import seed_everything


def test_seed_everything_reproducibility() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_load_all_configs_valid() -> None:
    cfgs = load_all_configs("configs")
    assert set(cfgs.keys()) == {"data", "graph", "training"}
    assert len(cfgs["graph"]["undirected_bones"]) == 16
    assert len(cfgs["graph"]["directed_edge_angle_triplets"]) == 32


def test_load_all_configs_validation_errors(tmp_path: Path) -> None:
    cfgs = load_all_configs("configs")
    for name in ("data", "graph", "training"):
        with (tmp_path / f"{name}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfgs[name], handle)

    # Corrupt split ratio sum
    bad_data = load_yaml_config(tmp_path / "data.yaml")
    bad_data["splits"]["train_ratio"] = 0.99
    with (tmp_path / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(bad_data, handle)

    with pytest.raises(ValueError, match="splits"):
        load_all_configs(tmp_path)


def test_hardware_helpers_contract() -> None:
    device = get_compute_device(prefer_gpu=True)
    assert device.type in {"cuda", "cpu"}
    profile = get_hardware_profile()
    for key in (
        "pytorch_version",
        "cuda_available",
        "cuda_version",
        "device_name",
        "total_vram_mb",
    ):
        assert key in profile
    vram = get_vram_usage_mb()
    assert set(vram.keys()) == {"allocated_mb", "reserved_mb", "max_allocated_mb"}


def test_env_check_main_writes_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(Path("configs"), tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    env_check.main()
    profile_path = tmp_path / "reports" / "hardware_profile.json"
    assert profile_path.is_file()
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    assert "pytorch_version" in data
