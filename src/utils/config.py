from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Carica un singolo file YAML e verifica che non sia vuoto."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"File di configurazione YAML non trovato: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(
            f"Il file YAML {cfg_path} è vuoto o non contiene una mappatura chiave-valore valida."
        )
    return data


def load_all_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Carica e valida data.yaml, graph.yaml e training.yaml.
    Restituisce {"data": data_cfg, "graph": graph_cfg, "training": train_cfg}.
    """
    base_dir = Path(config_dir)
    data_cfg = load_yaml_config(base_dir / "data.yaml")
    graph_cfg = load_yaml_config(base_dir / "graph.yaml")
    train_cfg = load_yaml_config(base_dir / "training.yaml")

    bones = graph_cfg.get("undirected_bones", [])
    if len(bones) != 16:
        raise ValueError(
            f"Configurazione graph.yaml non valida: attese 16 'undirected_bones', trovate {len(bones)}."
        )

    triplets = graph_cfg.get("directed_edge_angle_triplets", {})
    if len(triplets) != 32:
        raise ValueError(
            "Configurazione graph.yaml non valida: attese 32 voci in "
            f"'directed_edge_angle_triplets', trovate {len(triplets)}."
        )

    splits_cfg = data_cfg.get("splits", {})
    if not all(k in splits_cfg for k in ("train_ratio", "val_ratio", "test_ratio")):
        raise ValueError(
            "Configurazione data.yaml non valida: la sezione 'splits' deve contenere "
            "'train_ratio', 'val_ratio' e 'test_ratio'."
        )

    ratio_sum = (
        float(splits_cfg["train_ratio"])
        + float(splits_cfg["val_ratio"])
        + float(splits_cfg["test_ratio"])
    )
    if abs(ratio_sum - 1.0) >= 1e-6:
        raise ValueError(
            "Configurazione data.yaml non valida: la somma dei rapporti in 'splits' "
            f"(train+val+test) deve essere 1.0, trovata {ratio_sum:.8f}."
        )

    return {
        "data": data_cfg,
        "graph": graph_cfg,
        "training": train_cfg,
    }

