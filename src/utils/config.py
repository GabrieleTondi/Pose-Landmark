from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a single YAML file and verify that it is non-empty."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"YAML configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(
            f"YAML file {cfg_path} is empty or does not contain a valid key-value mapping."
        )
    return data


def load_all_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Load and validate data.yaml, graph.yaml, and training.yaml.
    Returns {"data": data_cfg, "graph": graph_cfg, "training": train_cfg}.
    """
    base_dir = Path(config_dir)
    data_cfg = load_yaml_config(base_dir / "data.yaml")
    graph_cfg = load_yaml_config(base_dir / "graph.yaml")
    train_cfg = load_yaml_config(base_dir / "training.yaml")

    bones = graph_cfg.get("undirected_bones", [])
    if len(bones) != 16:
        raise ValueError(
            f"Invalid graph.yaml configuration: expected 16 'undirected_bones', found {len(bones)}."
        )

    triplets = graph_cfg.get("directed_edge_angle_triplets", {})
    if len(triplets) != 32:
        raise ValueError(
            "Invalid graph.yaml configuration: expected 32 entries in "
            f"'directed_edge_angle_triplets', found {len(triplets)}."
        )

    splits_cfg = data_cfg.get("splits", {})
    if not all(k in splits_cfg for k in ("train_ratio", "val_ratio", "test_ratio")):
        raise ValueError(
            "Invalid data.yaml configuration: the 'splits' section must contain "
            "'train_ratio', 'val_ratio', and 'test_ratio'."
        )

    ratio_sum = (
        float(splits_cfg["train_ratio"])
        + float(splits_cfg["val_ratio"])
        + float(splits_cfg["test_ratio"])
    )
    if abs(ratio_sum - 1.0) >= 1e-6:
        raise ValueError(
            "Invalid data.yaml configuration: the sum of ratios in 'splits' "
            f"(train+val+test) must be 1.0, found {ratio_sum:.8f}."
        )

    return {
        "data": data_cfg,
        "graph": graph_cfg,
        "training": train_cfg,
    }

