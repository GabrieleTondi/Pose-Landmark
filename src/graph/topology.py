from typing import Any, Dict, List, Tuple

import torch


def get_directed_edge_list(graph_cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Legge le 16 ossa non orientate da graph_cfg['undirected_bones'].
    Per ogni osso b = (i, j) con b in 0..15, genera in ordine:
    - indice 2*b:   (i, j)
    - indice 2*b+1: (j, i)
    Restituisce una lista di 32 tuple (src, dst).
    """
    bones = graph_cfg["undirected_bones"]
    assert len(bones) == 16, f"Attese 16 ossa, trovate {len(bones)}"

    directed_edges: List[Tuple[int, int]] = []
    for bone in bones:
        assert len(bone) == 2, f"Ogni osso deve avere 2 estremi, trovato {bone}"
        i, j = int(bone[0]), int(bone[1])
        directed_edges.append((i, j))  # indice 2*b
        directed_edges.append((j, i))  # indice 2*b + 1

    assert len(directed_edges) == 32
    return directed_edges


def build_canonical_edge_index(graph_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Costruisce il tensore edge_index di shape (2, 32) e dtype torch.long
    a partire da get_directed_edge_list(graph_cfg).
    Non aggiunge self-loop (gestiti internamente da GATv2Conv con add_self_loops=True).
    """
    directed_edges = get_directed_edge_list(graph_cfg)
    src = [edge[0] for edge in directed_edges]
    dst = [edge[1] for edge in directed_edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    assert edge_index.shape == (2, 32)
    return edge_index

