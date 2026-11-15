from typing import Any, Dict, List, Tuple

import torch


def get_directed_edge_list(graph_cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Read the 16 undirected bones from graph_cfg['undirected_bones'].
    For each bone b = (i, j) with b in 0..15, generate in order:
    - index 2*b:   (i, j)
    - index 2*b+1: (j, i)
    Return a list of 32 (src, dst) tuples.
    """
    bones = graph_cfg["undirected_bones"]
    assert len(bones) == 16, f"Expected 16 bones, found {len(bones)}"

    directed_edges: List[Tuple[int, int]] = []
    for bone in bones:
        assert len(bone) == 2, f"Each bone must have 2 endpoints, found {bone}"
        i, j = int(bone[0]), int(bone[1])
        directed_edges.append((i, j))  # index 2*b
        directed_edges.append((j, i))  # index 2*b + 1

    assert len(directed_edges) == 32
    return directed_edges


def build_canonical_edge_index(graph_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Build the edge_index tensor of shape (2, 32) and dtype torch.long
    from get_directed_edge_list(graph_cfg).
    Does not add self-loops (handled internally by GATv2Conv with add_self_loops=True).
    """
    directed_edges = get_directed_edge_list(graph_cfg)
    src = [edge[0] for edge in directed_edges]
    dst = [edge[1] for edge in directed_edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    assert edge_index.shape == (2, 32)
    return edge_index

