from typing import Any, Dict

import numpy as np
import torch

from src.graph.topology import get_directed_edge_list


def compute_node_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Calcola le feature dei nodi a partire da P_bar di shape (17, 3).
    - Se graph_cfg['node_feature_mode'] == 'baseline': restituisce Tensor (17, 3) float32 [x, y, z].
    - Se graph_cfg['node_feature_mode'] == 'extended': restituisce Tensor (17, 5) float32 [x, y, z, r_i, sym_i].
    """
    coords = np.asarray(P_bar, dtype=np.float32)
    assert coords.shape == (17, 3) and np.isfinite(coords).all(), (
        f"P_bar non valido: shape={coords.shape}, finite={np.isfinite(coords).all()}"
    )

    mode = str(graph_cfg.get("node_feature_mode", "baseline"))
    if mode == "baseline":
        tensor = torch.from_numpy(coords.copy()).to(torch.float32)
        assert tensor.shape == (17, 3) and torch.isfinite(tensor).all()
        return tensor

    if mode == "extended":
        radii = np.linalg.norm(coords, axis=-1, keepdims=True).astype(np.float32)
        sym = np.asarray(
            graph_cfg["joint_symmetry"],
            dtype=np.float32,
        ).reshape(17, 1)
        extended = np.concatenate([coords, radii, sym], axis=-1).astype(np.float32)
        tensor = torch.from_numpy(extended).to(torch.float32)
        assert tensor.shape == (17, 5) and torch.isfinite(tensor).all()
        return tensor

    raise ValueError(
        f"node_feature_mode='{mode}' non supportato. Valori ammessi: 'baseline', 'extended'."
    )


def compute_edge_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Calcola le feature dei 32 archi diretti restituendo un Tensor di shape (32, 5) float32:
    colonne 0..2: versore direzionale unitario u_ij = (p_j - p_i) / (||p_j - p_i||_2 + eps)
    colonna 3:    lunghezza ossea l_ij = ||p_j - p_i||_2
    colonna 4:    coseno dell'angolo articolare cos(theta_abc) associato all'arco (i->j)
    """
    coords = np.asarray(P_bar, dtype=np.float64)
    assert coords.shape == (17, 3) and np.isfinite(coords).all(), (
        f"P_bar non valido: shape={coords.shape}, finite={np.isfinite(coords).all()}"
    )

    eps = float(graph_cfg.get("epsilon", 1e-8))
    directed_edges = get_directed_edge_list(graph_cfg)
    triplets_map = graph_cfg["directed_edge_angle_triplets"]

    edge_attr = np.zeros((32, 5), dtype=np.float32)
    for m, (i, j) in enumerate(directed_edges):
        diff = coords[j] - coords[i]
        length = float(np.linalg.norm(diff))
        u_ij = diff / (length + eps)

        edge_key = f"{i}->{j}"
        if edge_key not in triplets_map:
            raise KeyError(
                f"Chiave '{edge_key}' mancante in graph_cfg['directed_edge_angle_triplets']."
            )
        a, b, c = (int(idx) for idx in triplets_map[edge_key])
        v1 = coords[a] - coords[b]
        v2 = coords[c] - coords[b]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
        cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))

        edge_attr[m, 0:3] = u_ij.astype(np.float32)
        edge_attr[m, 3] = np.float32(length)
        edge_attr[m, 4] = np.float32(cos_theta)

    tensor = torch.from_numpy(edge_attr).to(torch.float32)
    assert tensor.shape == (32, 5) and torch.isfinite(tensor).all()
    return tensor

