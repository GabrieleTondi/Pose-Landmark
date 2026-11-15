from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def _is_augmented_graph(graph: Data) -> bool:
    return bool(getattr(graph, "is_augmented", False))


def _known_athlete_id(graph: Data) -> Optional[str]:
    ath = getattr(graph, "athlete_id", None)
    if ath is None:
        return None
    ath_str = str(ath).strip()
    if ath_str == "" or ath_str.lower() in {"unknown", "none", "nan"}:
        return None
    return ath_str


def _select_angle_nearest_positives(
    anchor: Data,
    candidates: List[Data],
    top_k: int,
) -> List[Data]:
    """Restrict positive candidates to the top-K nearest in bone-angle space (edge_attr[:, 4])."""
    if top_k <= 0 or len(candidates) <= top_k:
        return candidates
    a_edge = getattr(anchor, "edge_attr", None)
    if a_edge is None or a_edge.dim() != 2 or a_edge.size(1) < 5:
        return candidates
    a_cos = a_edge[:, 4]
    valid: List[Tuple[float, int, Data]] = []
    for idx, cand in enumerate(candidates):
        c_edge = getattr(cand, "edge_attr", None)
        if c_edge is not None and c_edge.shape == a_edge.shape:
            dist = float(torch.linalg.norm(c_edge[:, 4] - a_cos).item())
            valid.append((dist, idx, cand))
    if len(valid) <= top_k:
        return [item[2] for item in valid] if len(valid) > 0 else candidates
    valid.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in valid[:top_k]]


def resolve_class_thresholds(
    real_class_graphs: List[Data],
    train_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Compute (tau_neg, tau_pos) for a single class from real graphs only.
    Uses absolute thresholds from train_cfg['sampling'] when pools are sufficient,
    otherwise falls back to configured quantiles while ensuring at least 2 positive performances if available.
    """
    if len(real_class_graphs) == 0:
        raise ValueError("Cannot compute GOE thresholds on a class with no real graphs.")

    s_cfg = train_cfg["sampling"]
    tier_rule = str(s_cfg.get("tier_rule", "threshold")).lower()
    abs_pos = float(s_cfg.get("goe_pos_threshold", 1.5))
    abs_neg = float(s_cfg.get("goe_neg_threshold", 0.0))
    q_pos = float(s_cfg.get("goe_pos_quantile", 0.70))
    q_neg = float(s_cfg.get("goe_neg_quantile", 0.35))

    goes = np.array([float(g.goe) for g in real_class_graphs], dtype=np.float64)

    if tier_rule == "threshold":
        pos_cands = [g for g in real_class_graphs if float(g.goe) >= abs_pos]
        neg_cands = [g for g in real_class_graphs if float(g.goe) <= abs_neg]
        pos_perfs = {str(g.performance_id) for g in pos_cands}
        neg_perfs = {str(g.performance_id) for g in neg_cands}

        small_pool = len(real_class_graphs) >= 15 and (len(pos_cands) < 5 or len(neg_cands) < 5)
        if len(pos_perfs) >= 2 and len(neg_perfs) >= 2 and not small_pool:
            return abs_neg, abs_pos

    tau_neg = float(np.quantile(goes, q_neg))
    tau_pos = float(np.quantile(goes, q_pos))

    if tau_neg >= tau_pos:
        med = float(np.median(goes))
        tau_neg = med - 1e-6
        tau_pos = med

    pos_q_perfs = {
        str(g.performance_id) for g in real_class_graphs if float(g.goe) >= tau_pos
    }
    if len(pos_q_perfs) < 2 and len({str(g.performance_id) for g in real_class_graphs}) >= 2:
        sorted_goes = np.sort(goes)[::-1]
        if len(sorted_goes) >= 2:
            tau_pos = float(sorted_goes[1])
            tau_neg = min(tau_neg, float(sorted_goes[-1]))
            if tau_neg >= tau_pos:
                tau_neg = tau_pos - 1e-6

    return tau_neg, tau_pos


def partition_by_goe_tiers(
    graphs: List[Data],
    train_cfg: Dict[str, Any],
    fit_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, Dict[str, List[Data]]], Dict[str, Tuple[float, float]]]:
    """
    Group graphs by element_label and partition them into:
    pools[label] = {'pos': [...], 'mid': [...], 'neg_real': [...], 'neg_synth': [...]}
    If fit_thresholds is None (training set), compute (tau_neg, tau_pos) per class
    using absolute thresholds from train_cfg or quantile fallback on real graphs only.
    If fit_thresholds is provided (val/test), reuse the thresholds computed on the training set.
    """
    by_class: Dict[str, List[Data]] = {}
    for g in graphs:
        label = str(g.element_label)
        by_class.setdefault(label, []).append(g)

    pools: Dict[str, Dict[str, List[Data]]] = {}
    thresholds: Dict[str, Tuple[float, float]] = {}

    for label in sorted(by_class.keys()):
        cls_graphs = by_class[label]
        real_graphs = [g for g in cls_graphs if not _is_augmented_graph(g)]
        synth_graphs = [g for g in cls_graphs if _is_augmented_graph(g)]

        if fit_thresholds is not None and label in fit_thresholds:
            tau_neg, tau_pos = float(fit_thresholds[label][0]), float(fit_thresholds[label][1])
        else:
            tau_neg, tau_pos = resolve_class_thresholds(real_graphs, train_cfg)

        thresholds[label] = (tau_neg, tau_pos)
        cls_pool: Dict[str, List[Data]] = {
            "pos": [],
            "mid": [],
            "neg_real": [],
            "neg_synth": list(synth_graphs),
        }
        for g in real_graphs:
            goe_val = float(g.goe)
            if goe_val >= tau_pos:
                cls_pool["pos"].append(g)
            elif goe_val <= tau_neg:
                cls_pool["neg_real"].append(g)
            else:
                cls_pool["mid"].append(g)

        pools[label] = cls_pool

    return pools, thresholds


class TripletDataset(torch.utils.data.Dataset):
    """
    Deterministic dataset returning a 4-element tuple at index idx compatible
    with torch_geometric.loader.DataLoader:
    (graph_A, graph_P, graph_N, margin_alpha_tensor)
    where margin_alpha_tensor is torch.tensor(alpha, dtype=torch.float32).
    """

    def __init__(
        self,
        pools: Dict[str, Dict[str, List[Data]]],
        train_cfg: Dict[str, Any],
        num_triplets: int,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_triplets <= 0:
            raise ValueError(f"num_triplets must be > 0, got {num_triplets}.")

        self.split = str(split)
        self.num_triplets = int(num_triplets)

        if self.split in ("val", "test"):
            for cls_name, cls_pools in pools.items():
                for pool_key in ("pos", "mid", "neg_real"):
                    for g in cls_pools.get(pool_key, []):
                        if _is_augmented_graph(g):
                            raise ValueError(
                                f"Augmented graph detected in pool '{pool_key}' of split='{self.split}' "
                                f"for class '{cls_name}' ({g.sample_id})."
                            )

        s_cfg = train_cfg["sampling"]
        l_cfg = train_cfg["loss"]
        max_pos_goe_diff = float(s_cfg.get("max_pos_goe_diff", 1.5))
        min_neg_goe_gap = float(s_cfg.get("min_neg_goe_gap", 1.5))
        real_neg_prob = float(s_cfg.get("real_negative_prob", 0.50))
        positive_top_k = int(s_cfg.get("positive_top_k", 0))
        self_synth_prob = float(s_cfg.get("self_synth_prob", 0.50))
        margin_mode = str(l_cfg.get("margin_mode", "adaptive"))
        fixed_margin = float(l_cfg.get("fixed_margin", 0.50))
        adaptive_margins = l_cfg.get("adaptive_margins", {1: 0.20, 2: 0.60, 3: 1.00})

        eligible_classes: List[str] = []
        for cls_name in sorted(pools.keys()):
            cls_p = pools[cls_name]
            pos_pool = cls_p.get("pos", [])
            has_neg_real = len(cls_p.get("neg_real", [])) > 0
            has_neg_synth = self.split == "train" and len(cls_p.get("neg_synth", [])) > 0
            has_mid_fallback = len(cls_p.get("mid", [])) > 0
            if len(pos_pool) >= 2 and (has_neg_real or has_neg_synth or has_mid_fallback):
                eligible_classes.append(cls_name)

        if len(eligible_classes) == 0:
            raise ValueError(
                f"No class in split='{self.split}' has enough samples to build triplets."
            )

        rng = np.random.default_rng(int(seed))
        self.triplets: List[Tuple[Data, Data, Data, torch.Tensor]] = []

        for _ in range(self.num_triplets):
            c = eligible_classes[int(rng.integers(0, len(eligible_classes)))]
            cls_p = pools[c]
            pos_pool = cls_p["pos"]
            neg_real_pool = cls_p["neg_real"]
            mid_pool = cls_p["mid"]
            neg_synth_pool = [] if self.split in ("val", "test") else cls_p["neg_synth"]

            anchor = pos_pool[int(rng.integers(0, len(pos_pool)))]

            distinct_athletes = {
                aid for aid in (_known_athlete_id(g) for g in pos_pool) if aid is not None
            }
            anchor_ath = _known_athlete_id(anchor)

            p_cands = [g for g in pos_pool if str(g.performance_id) != str(anchor.performance_id)]
            if len(distinct_athletes) >= 2 and anchor_ath is not None:
                ath_filtered = [g for g in p_cands if _known_athlete_id(g) != anchor_ath]
                if len(ath_filtered) > 0:
                    p_cands = ath_filtered
            if len(p_cands) == 0:
                p_cands = [g for g in pos_pool if str(g.sample_id) != str(anchor.sample_id)]
            if len(p_cands) == 0:
                raise ValueError(
                    f"Unable to select a Positive distinct from Anchor '{anchor.sample_id}' "
                    f"in class '{c}'."
                )

            p_close = [
                g for g in p_cands if abs(float(g.goe) - float(anchor.goe)) <= max_pos_goe_diff
            ]
            p_final = p_close if len(p_close) > 0 else p_cands
            if self.split == "train" and positive_top_k > 0:
                p_final = _select_angle_nearest_positives(anchor, p_final, positive_top_k)
            positive = p_final[int(rng.integers(0, len(p_final)))]

            use_real_neg = (
                self.split in ("val", "test")
                or len(neg_synth_pool) == 0
                or (len(neg_real_pool) > 0 and float(rng.random()) < real_neg_prob)
            )

            if use_real_neg:
                base_real_pool = neg_real_pool if len(neg_real_pool) > 0 else mid_pool
                n_diff_perf = [
                    g for g in base_real_pool if str(g.performance_id) != str(anchor.performance_id)
                ]
                n_cands = n_diff_perf if len(n_diff_perf) > 0 else base_real_pool
                n_gap = [
                    g for g in n_cands if float(anchor.goe) - float(g.goe) >= min_neg_goe_gap
                ]
                if len(n_gap) == 0:
                    n_gap = [g for g in n_cands if float(g.goe) < float(anchor.goe)]
                n_final = n_gap if len(n_gap) > 0 else n_cands
                negative = n_final[int(rng.integers(0, len(n_final)))]

                delta_goe = float(anchor.goe) - float(negative.goe)
                if delta_goe < 2.0:
                    tier = 1
                elif delta_goe < 3.5:
                    tier = 2
                else:
                    tier = 3
            else:
                self_synth = [
                    g
                    for g in neg_synth_pool
                    if str(getattr(g, "parent_sample_id", "")) == str(anchor.sample_id)
                ]
                cross_synth = [
                    g
                    for g in neg_synth_pool
                    if str(getattr(g, "parent_sample_id", "")) != str(anchor.sample_id)
                ]
                if len(self_synth) > 0 and float(rng.random()) < self_synth_prob:
                    negative = self_synth[int(rng.integers(0, len(self_synth)))]
                elif len(cross_synth) > 0:
                    negative = cross_synth[int(rng.integers(0, len(cross_synth)))]
                else:
                    negative = neg_synth_pool[int(rng.integers(0, len(neg_synth_pool)))]

                raw_tier = int(getattr(negative, "severity_tier", 1))
                tier = max(1, min(3, raw_tier))

            assert str(anchor.element_label) == str(positive.element_label) == str(negative.element_label)
            if self.split in ("val", "test"):
                assert not _is_augmented_graph(anchor)
                assert not _is_augmented_graph(positive)
                assert not _is_augmented_graph(negative)

            if margin_mode == "fixed":
                alpha = fixed_margin
            else:
                alpha = float(adaptive_margins.get(tier, adaptive_margins.get(str(tier), 0.50)))

            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            self.triplets.append((anchor, positive, negative, alpha_tensor))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, Data, torch.Tensor]:
        return self.triplets[idx]

