# 04 — Sottospecifica di Settore: Costruzione del Grafo e Feature Engineering

**Settore:** 04 — Graph Representation: Topologia Canonica a 17 Nodi, `x`, `pos`, `edge_attr` e Serializzazione PyG  
**Riferimenti Specifica Madre:** §6.1–§6.4, §16.3  
**Dipendenze in ingresso:** Settore 01 (`configs/graph.yaml`), Settore 03 ($\bar{P} \in \mathbb{R}^{17 \times 3}$)  
**Output abilitati:** Settore 05, Settore 06, Settore 07, Settore 08

---

## 1. Obiettivo del Settore

Convertire ogni posa canonicalizzata $\bar{P} \in \mathbb{R}^{17 \times 3}$ in un oggetto `torch_geometric.data.Data` con:
- `num_nodes = 17`;
- `pos`: tensore `(17, 3)` `float32` contenente sempre le coordinate 3D canonicalizzate $\bar{P}$;
- `x`: tensore `(17, 3)` in modalità `"baseline"` (oppure `(17, 5)` in modalità `"extended"`);
- `edge_index`: tensore `(2, 32)` `long` costruito secondo la regola di indicizzazione contigua a coppie $\{2b, 2b+1\}$;
- `edge_attr`: tensore `(32, 5)` `float32` dove ogni riga $m \in \{0, \dots, 31\}$ contiene `[u_x, u_y, u_z, bone_length, cos_angle]`.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 4.1 — Topologia Canonica e Indicizzazione Contigua `{2b, 2b+1}` (`src/graph/topology.py`)
- **File da creare:** `src/graph/topology.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any, List, Tuple
  import torch

  def get_directed_edge_list(graph_cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
      """
      Legge le 16 ossa non orientate da graph_cfg['undirected_bones'].
      Per ogni osso b = (i, j) con b in 0..15, genera in ordine:
      - indice 2*b:   (i, j)
      - indice 2*b+1: (j, i)
      Restituisce una lista di 32 tuple (src, dst).
      """

  def build_canonical_edge_index(graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Costruisce il tensore edge_index di shape (2, 32) e dtype torch.long
      a partire da get_directed_edge_list(graph_cfg).
      """
  ```
- **COME implementare `get_directed_edge_list` e `build_canonical_edge_index`:**
  ```python
  bones = graph_cfg["undirected_bones"]
  assert len(bones) == 16, f"Attese 16 ossa, trovate {len(bones)}"
  directed_edges: List[Tuple[int, int]] = []
  for i, j in bones:
      directed_edges.append((int(i), int(j)))  # indice 2*b
      directed_edges.append((int(j), int(i)))  # indice 2*b + 1
  # In build_canonical_edge_index:
  src = [e[0] for e in directed_edges]
  dst = [e[1] for e in directed_edges]
  return torch.tensor([src, dst], dtype=torch.long)
  ```
- **CONSTRAINT:** Non cambiare mai l'ordine degli archi e non aggiungere i self-loop in `build_canonical_edge_index` (i self-loop vengono aggiunti internamente dal layer `GATv2Conv(..., add_self_loops=True)`).

---

### Task 4.2 — Calcolo di `x` e `edge_attr` (`src/graph/features.py`)
- **File da creare:** `src/graph/features.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any
  import numpy as np
  import torch
  from src.graph.topology import get_directed_edge_list

  def compute_node_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Calcola le feature dei nodi a partire da P_bar di shape (17, 3).
      - Se graph_cfg['node_feature_mode'] == 'baseline': restituisce Tensor (17, 3) float32 [x, y, z].
      - Se graph_cfg['node_feature_mode'] == 'extended': restituisce Tensor (17, 5) float32 [x, y, z, r_i, sym_i].
      """

  def compute_edge_features(P_bar: np.ndarray, graph_cfg: Dict[str, Any]) -> torch.Tensor:
      """
      Calcola le feature dei 32 archi diretti restituendo un Tensor di shape (32, 5) float32:
      colonne 0..2: versore direzionale unitario u_ij = (p_j - p_i) / (||p_j - p_i||_2 + eps)
      colonna 3:    lunghezza ossea l_ij = ||p_j - p_i||_2
      colonna 4:    coseno dell'angolo articolare cos(theta_abc) associato all'arco (i->j)
      """
  ```
- **COME implementare `compute_edge_features(P_bar, graph_cfg)` (Algoritmo esatto):**
  1. Verificare `assert P_bar.shape == (17, 3) and np.isfinite(P_bar).all()`.
  2. Leggere `eps = float(graph_cfg.get("epsilon", 1e-8))`, `directed_edges = get_directed_edge_list(graph_cfg)` e `triplets_map = graph_cfg["directed_edge_angle_triplets"]`.
  3. Allocare `edge_attr = np.zeros((32, 5), dtype=np.float32)`.
  4. Per `m, (i, j)` in `enumerate(directed_edges)`:
     - Calcolare il vettore osseo e la sua lunghezza:
       ```python
       diff = P_bar[j] - P_bar[i]
       length = float(np.linalg.norm(diff))
       u_ij = diff / (length + eps)
       ```
     - Recuperare la terna angolare `a, b, c = triplets_map[f"{i}->{j}"]` (dove `b` è il vertice dell'angolo):
       ```python
       v1 = P_bar[a] - P_bar[b]
       v2 = P_bar[c] - P_bar[b]
       denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
       cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
       ```
     - Scrivere la riga `m`:
       ```python
       edge_attr[m, 0:3] = u_ij
       edge_attr[m, 3] = length
       edge_attr[m, 4] = cos_theta
       ```
  5. Convertire in `torch.from_numpy(edge_attr)` e verificare `assert torch.isfinite(tensor).all()`.

---

### Task 4.3 — Costruttore e Serializzatore `Data` di PyTorch Geometric (`src/graph/builder.py`)
- **File da creare:** `src/graph/builder.py`
- **Contratti esatti delle funzioni:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List
  import numpy as np
  import torch
  from torch_geometric.data import Data
  from src.data.harmonization import UnifiedSample
  from src.graph.topology import build_canonical_edge_index
  from src.graph.features import compute_node_features, compute_edge_features

  def build_pyg_graph(
      P_bar: np.ndarray,
      sample_meta: Dict[str, Any],
      graph_cfg: Dict[str, Any]
  ) -> Data:
      """
      Costruisce e valida un oggetto torch_geometric.data.Data completo.
      """

  def process_and_save_dataset_graphs(
      samples: List[UnifiedSample],
      split_dict: Dict[str, List[str]],
      data_cfg: Dict[str, Any],
      graph_cfg: Dict[str, Any]
  ) -> Dict[str, List[Data]]:
      """
      Esegue extract_stable_window -> canonicalize_window_3d -> build_pyg_graph
      per tutti i campioni, salva train_graphs.pt, val_graphs.pt, test_graphs.pt
      in data/processed/ e restituisce {'train': [...], 'val': [...], 'test': [...]}.
      """
  ```
- **COME implementare `build_pyg_graph(P_bar, sample_meta, graph_cfg)`:**
  Costruire l'oggetto `Data` con i seguenti attributi **esatti**:
  ```python
  x = compute_node_features(P_bar, graph_cfg)          # (17, 3) o (17, 5) float32
  pos = torch.tensor(P_bar, dtype=torch.float32)       # SEMPRE (17, 3) float32
  edge_index = build_canonical_edge_index(graph_cfg)   # (2, 32) long
  edge_attr = compute_edge_features(P_bar, graph_cfg)  # (32, 5) float32
  goe_val = float(sample_meta.get("goe", 0.0))
  y = torch.tensor([goe_val], dtype=torch.float32)     # (1,) float32

  data = Data(
      x=x,
      pos=pos,
      edge_index=edge_index,
      edge_attr=edge_attr,
      y=y,
      sample_id=str(sample_meta["sample_id"]),
      dataset_id=str(sample_meta.get("dataset_id", "finefs")),
      performance_id=str(sample_meta["performance_id"]),
      athlete_id=str(sample_meta.get("athlete_id") or "unknown"),
      element_label=str(sample_meta["element_label"]),
      goe=goe_val,
      split=str(sample_meta.get("split", "train")),
      is_augmented=bool(sample_meta.get("is_augmented", False)),
      parent_sample_id=str(sample_meta.get("parent_sample_id") or sample_meta["sample_id"]),
      severity_tier=int(sample_meta.get("severity_tier", 0)),
      perturbation_joint=int(sample_meta.get("perturbation_joint", -1)),
      perturbation_angle_deg=float(sample_meta.get("perturbation_angle_deg", 0.0))
  )
  return data
  ```
- **CONSTRAINT (Nota importante per PyTorch 2.6+):**
  Quando si ricaricano i file `.pt` contenenti oggetti `torch_geometric.data.Data` tramite `torch.load()`, passare sempre `weights_only=False` (`torch.load(path, map_location="cpu", weights_only=False)`) per evitare l'`UnpicklingError` introdotto nelle versioni recenti di PyTorch sulle classi custom di PyG.

---

## 3. Unit Test da Implementare (`tests/test_graph.py`)

Il file `tests/test_graph.py` deve implementare i seguenti 4 test automatici:
1. `test_topology_paired_ordering()`: verifica che `edge_index` abbia shape `(2, 32)` e che per ogni $b \in \{0, \dots, 15\}$ valga `edge_index[0, 2*b] == edge_index[1, 2*b+1]` e `edge_index[1, 2*b] == edge_index[0, 2*b+1]`.
2. `test_edge_features_geometry()`: verifica che su un grafo costruito da una posa canonica `edge_attr` abbia shape `(32, 5)`, che `torch.norm(edge_attr[:, 0:3], dim=-1)` sia `1.0` ($\pm 10^{-4}$), che `edge_attr[2*b, 0:3] == -edge_attr[2*b+1, 0:3]`, che `edge_attr[:, 3] > 0` e che `edge_attr[:, 4]` sia in `[-1.0, 1.0]`.
3. `test_pyg_data_and_batching()`: costruisce 8 grafi con `build_pyg_graph`, li passa a `torch_geometric.loader.DataLoader(graphs, batch_size=4)` e verifica che il primo batch abbia `batch.x.shape == (68, 3)`, `batch.pos.shape == (68, 3)`, `batch.edge_index.shape == (2, 128)` e `batch.edge_attr.shape == (128, 5)`.
4. `test_extended_node_features_preserves_pos()`: imposta `node_feature_mode = "extended"` e verifica che `data.x.shape == (17, 5)` mentre `data.pos.shape == (17, 3)` rimane invariato.

---

## 4. Criteri di Accettazione del Settore 04 (Definition of Done)

- [ ] I file `src/graph/topology.py`, `src/graph/features.py`, `src/graph/builder.py` e `tests/test_graph.py` sono stati creati.
- [ ] Il comando `pytest tests/test_graph.py` passa al 100% (4/4 test verdi).
