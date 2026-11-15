# 05 — Sottospecifica di Settore: Triplet Sampling, Loss Adattiva e Data Augmentation

**Settore:** 05 — Metric Learning Pipeline: Rotazione di Rodrigues, Triplet Sampler Intra-Classe e Loss Adattiva  
**Riferimenti Specifica Madre:** §8.1–§8.6  
**Dipendenze in ingresso:** Settore 01 (`configs/training.yaml`, `configs/graph.yaml`), Settore 03 (`canonicalize_frame_3d`), Settore 04 (`build_pyg_graph`)  
**Output abilitati:** Settore 06 (Training `SiameseGATv2`), Settore 08 (Slider del *What-If Lab*)

---

## 1. Obiettivo del Settore

Implementare:
1. La rotazione rigida 3D di Rodrigues su sotto-alberi cinematici (`src/training/augmentation.py`) per generare negativi sintetici nel solo training set (e per animare gli slider interattivi nel *What-If Lab* del Settore 08);
2. Il partizionamento per fasce di GOE e il dataset di triplette PyG `TripletDataset` (`src/training/triplet_sampler.py`);
3. La funzione di costo `AdaptiveTripletMarginLoss` (`src/training/losses.py`) con protezione contro gradienti `NaN`.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 5.1 — Rotazione Rigida di Rodrigues e Augmentation (`src/training/augmentation.py`)
- **File da creare:** `src/training/augmentation.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any, List
  import numpy as np
  from torch_geometric.data import Data
  from src.data.preprocessing import canonicalize_frame_3d
  from src.graph.builder import build_pyg_graph

  def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
      """
      Calcola la matrice di rotazione 3x3 in SO(3) attorno al vettore 'axis' di un angolo 'angle_rad'
      usando la formula di Rodrigues: R = I + sin(theta)*K + (1 - cos(theta))*(K @ K).
      Restituisce un np.ndarray di shape (3, 3) float32.
      """

  def rotate_kinematic_subtree(
      P_bar: np.ndarray,
      joint_idx: int,
      child_idx: int,
      axis: np.ndarray,
      angle_deg: float,
      graph_cfg: Dict[str, Any]
  ) -> np.ndarray:
      """
      Ruota rigidamente attorno al punto perno P_bar[joint_idx] tutti i nodi appartenenti al
      sotto-albero kinematic_subtrees[f"{joint_idx}->{child_idx}"] di 'angle_deg' gradi.
      Restituisce la nuova matrice P_prime di shape (17, 3) float32 con lunghezze ossee invariate.
      """

  def generate_synthetic_negatives(
      train_graphs: List[Data],
      data_cfg: Dict[str, Any],
      graph_cfg: Dict[str, Any],
      train_cfg: Dict[str, Any]
  ) -> List[Data]:
      """
      Per ogni grafo reale positivo del SOLO training set (split == 'train' e not is_augmented),
      genera 3 grafi negativi sintetici (uno per ciascun severity_tier 1, 2, 3)
      e restituisce la lista dei nuovi oggetti Data sintetici.
      """
  ```
- **COME implementare `rodrigues_rotation_matrix` e `rotate_kinematic_subtree` (Codice esatto):**
  ```python
  def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
      u = np.asarray(axis, dtype=np.float64)
      norm = np.linalg.norm(u)
      if norm < 1e-8:
          return np.eye(3, dtype=np.float32)
      u = u / norm
      ux, uy, uz = u[0], u[1], u[2]
      K = np.array([
          [0.0, -uz,  uy],
          [ uz, 0.0, -ux],
          [-uy,  ux, 0.0]
      ], dtype=np.float64)
      I = np.eye(3, dtype=np.float64)
      R = I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)
      return R.astype(np.float32)

  def rotate_kinematic_subtree(
      P_bar: np.ndarray,
      joint_idx: int,
      child_idx: int,
      axis: np.ndarray,
      angle_deg: float,
      graph_cfg: Dict[str, Any]
  ) -> np.ndarray:
      key = f"{joint_idx}->{child_idx}"
      subtree_nodes = graph_cfg["kinematic_subtrees"].get(key, [child_idx])
      R = rodrigues_rotation_matrix(axis, np.deg2rad(float(angle_deg)))
      P_prime = P_bar.copy().astype(np.float32)
      pivot = P_bar[joint_idx]
      for m in subtree_nodes:
          P_prime[m] = pivot + R @ (P_bar[m] - pivot)
      return P_prime
  ```
- **COME implementare `generate_synthetic_negatives`:**
  1. Filtrare i grafi di input mantenendo solo quelli con `g.split == "train"` e `not g.is_augmented`.
  2. Leggere `tier_ranges = train_cfg["augmentation"]["tier_angle_ranges_deg"]` e `noise_std = float(train_cfg["augmentation"]["noise_std"])`.
  3. Per ogni grafo positivo del training set e per ogni `tier` in `[1, 2, 3]`:
     - Scegliere una chiave da `list(graph_cfg["kinematic_subtrees"].keys())` (privilegiando le articolazioni degli arti inferiori `["11->13", "13->15", "12->14", "14->16"]` e superiori `["5->7", "6->8"]`);
     - Estrarre `angle_deg = rng.uniform(tier_ranges[tier][0], tier_ranges[tier][1]) * rng.choice([-1.0, 1.0])`;
     - Scegliere un asse unitario ortogonale all'osso (es. `[0.0, 1.0, 0.0]` o `[1.0, 0.0, 0.0]`);
     - Applicare `P_pert = rotate_kinematic_subtree(g.pos.numpy(), joint_idx, child_idx, axis, angle_deg, graph_cfg)`;
     - Se `noise_std > 0`: aggiungere `rng.normal(0.0, noise_std, size=P_pert.shape).astype(np.float32)`;
     - Ri-canonicalizzare con `P_pert_canon, _ = canonicalize_frame_3d(P_pert, data_cfg)`;
     - Costruire il nuovo grafo con `build_pyg_graph(P_pert_canon, meta, graph_cfg)` dove `meta` ha:
       - `sample_id = f"{g.sample_id}_aug_t{tier}"`
       - `performance_id = g.performance_id`
       - `athlete_id = g.athlete_id`
       - `element_label = g.element_label`
       - `goe = float(g.goe) - float(tier)`
       - `split = "train"`
       - `is_augmented = True`
       - `parent_sample_id = g.sample_id`
       - `severity_tier = int(tier)`
       - `perturbation_joint = int(joint_idx)`
       - `perturbation_angle_deg = float(angle_deg)`
- **CONSTRAINT:** Sollevare `ValueError` se in `generate_synthetic_negatives` viene passato un grafo con `g.split != "train"`.

---

### Task 5.2 — Partizionamento GOE e `TripletDataset` (`src/training/triplet_sampler.py`)
- **File da creare:** `src/training/triplet_sampler.py`
- **Contratti esatti della funzione e della classe:**
  ```python
  from typing import Dict, Any, List, Optional, Tuple
  import numpy as np
  import torch
  from torch_geometric.data import Data

  def partition_by_goe_tiers(
      graphs: List[Data],
      train_cfg: Dict[str, Any],
      fit_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
  ) -> Tuple[Dict[str, Dict[str, List[Data]]], Dict[str, Tuple[float, float]]]:
      """
      Raggruppa i grafi per element_label e li divide in:
      pools[label] = {'pos': [...], 'mid': [...], 'neg_real': [...], 'neg_synth': [...]}
      Se fit_thresholds è None (cioè siamo sul training set), calcola per ogni classe (tau_neg, tau_pos)
      usando le soglie assolute da train_cfg (o i quantili se uno dei pool 'pos' o 'neg_real' ha < 2 esibizioni).
      Se fit_thresholds è fornito (su val e test), usa le soglie già calcolate.
      """

  class TripletDataset(torch.utils.data.Dataset):
      """
      Dataset che restituisce all'indice idx una tupla di 4 elementi compatibile con
      torch_geometric.loader.DataLoader:
      (graph_A, graph_P, graph_N, margin_alpha_tensor)
      dove margin_alpha_tensor è torch.tensor(alpha, dtype=torch.float32).
      """
      def __init__(
          self,
          pools: Dict[str, Dict[str, List[Data]]],
          train_cfg: Dict[str, Any],
          num_triplets: int,
          split: str = "train",
          seed: int = 42
      ) -> None: ...
  ```
- **COME costruire le triplette in `TripletDataset` (Logica esatta):**
  1. Nel costruttore `__init__`, generare una lista deterministica `self.triplets: List[Tuple[Data, Data, Data, torch.Tensor]]` di lunghezza `num_triplets` usando `rng = np.random.default_rng(seed)`.
  2. Per ciascuna tripla $k \in \{0, \dots, \text{num\_triplets}-1\}$:
     - Scegliere una classe `c` tra quelle disponibili in `pools`;
     - Estrarre `A` da `pools[c]["pos"]` (tutti i grafi in `"pos"` hanno `is_augmented == False`);
     - Cercare i candidati `P` in `pools[c]["pos"]` tali che `P.performance_id != A.performance_id` (e `P.athlete_id != A.athlete_id` se esistono almeno 2 atleti distinti nel pool; se nel mini-split ce n'è 1 solo, richiedere `P.sample_id != A.sample_id`);
     - **Selezione di `N`:**
       - Se `split in ("val", "test")` oppure `len(pools[c]["neg_synth"]) == 0` oppure `rng.random() < real_negative_prob`:
         Cercare `N` in `pools[c]["neg_real"]` con `N.performance_id != A.performance_id`.
         Calcolare la severità del negativo reale in base a $\Delta \text{GOE} = A.\text{goe} - N.\text{goe}$:
         - $\Delta \text{GOE} < 2.0 \implies \text{tier} = 1$
         - $2.0 \le \Delta \text{GOE} < 3.5 \implies \text{tier} = 2$
         - $\Delta \text{GOE} \ge 3.5 \implies \text{tier} = 3$
       - Altrimenti (solo in `split == "train"`):
         Con probabilità $0.5$ scegliere un *self-structural negative* di `A` (`N` in `pools[c]["neg_synth"]` con `N.parent_sample_id == A.sample_id`), altrimenti un qualsiasi `N` in `pools[c]["neg_synth"]`. Il `tier` è `N.severity_tier` ($\in \{1, 2, 3\}$).
     - **Calcolo di `alpha`:**
       - Se `train_cfg["loss"]["margin_mode"] == "fixed"`: `alpha = float(train_cfg["loss"]["fixed_margin"])`
       - Altrimenti: `alpha = float(train_cfg["loss"]["adaptive_margins"][tier])`
     - Salvare `(A, P, N, torch.tensor(alpha, dtype=torch.float32))`.
- **CONSTRAINT CRITICO SUL DATALOADER (Trappola PyTorch vs PyG):**
  Quando si itera su `TripletDataset` in training/val/test, usare **sempre e solo** `from torch_geometric.loader import DataLoader` (e **MAI** `torch.utils.data.DataLoader` standard!). Il `DataLoader` di PyG riconosce automaticamente una tupla `(Data, Data, Data, Tensor)` e restituisce ad ogni iterazione `(batch_A, batch_P, batch_N, batch_alpha)` dove `batch_A, batch_P, batch_N` sono oggetti `Batch` di PyG e `batch_alpha` è un tensore `(B,)`!

---

### Task 5.3 — Funzione di Costo `AdaptiveTripletMarginLoss` (`src/training/losses.py`)
- **File da creare:** `src/training/losses.py`
- **Contratto esatto della classe:**
  ```python
  from typing import Tuple, Dict
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  class AdaptiveTripletMarginLoss(nn.Module):
      def __init__(self, eps: float = 1e-12) -> None:
          super().__init__()
          self.eps = eps

      def forward(
          self,
          z_A: torch.Tensor,
          z_P: torch.Tensor,
          z_N: torch.Tensor,
          margin_alpha: torch.Tensor
      ) -> Tuple[torch.Tensor, Dict[str, float]]:
          """
          Calcola la Triplet Margin Loss con margine per-sample margin_alpha di shape (B,).
          Restituisce:
          - loss: Tensor scalare differenziabile
          - stats: {'loss': float, 'triplet_acc': float, 'margin_satisfied_acc': float,
                    'mean_d_ap': float, 'mean_d_an': float}
          """
  ```
- **COME implementare `forward` (Codice esatto):**
  ```python
  d_ap = torch.sqrt(torch.sum((z_A - z_P) ** 2, dim=-1) + self.eps)  # shape (B,)
  d_an = torch.sqrt(torch.sum((z_A - z_N) ** 2, dim=-1) + self.eps)  # shape (B,)
  margins = margin_alpha.to(z_A.device).view(-1)                     # shape (B,)
  losses = F.relu(d_ap - d_an + margins)                             # shape (B,)
  loss = torch.mean(losses)

  with torch.no_grad():
      triplet_acc = float(torch.mean((d_ap < d_an).float()).item())
      margin_acc = float(torch.mean((d_ap + margins < d_an).float()).item())
      stats = {
          "loss": float(loss.item()),
          "triplet_acc": triplet_acc,
          "margin_satisfied_acc": margin_acc,
          "mean_d_ap": float(d_ap.mean().item()),
          "mean_d_an": float(d_an.mean().item())
      }
  return loss, stats
  ```

---

## 3. Unit Test da Implementare (`tests/test_triplets.py`)

Il file `tests/test_triplets.py` deve implementare i seguenti 4 test automatici:
1. `test_rodrigues_subtree_preserves_bone_lengths()`: applica una rotazione di $+30^\circ$ al sotto-albero `"11->13"` (anca SX $\to$ ginocchio SX e caviglia SX), ricostruisce il grafo con `build_pyg_graph` e verifica che tutte le 32 lunghezze ossee `edge_attr[:, 3]` siano invariate entro $10^{-5}$, mentre il coseno dell'angolo in `edge_attr[:, 4]` sull'anca SX sia cambiato.
2. `test_triplet_dataset_intra_class_and_no_val_augmentation()`: verifica su 100 triplette di `train` e `val` che `A.element_label == P.element_label == N.element_label`, che `A.performance_id != P.performance_id` e che in `val` nessun grafo abbia `is_augmented == True`.
3. `test_adaptive_triplet_loss_backward_and_zero_distance_safety()`: verifica che anche passando `z_A == z_P` (distanza esattamente zero), `loss.backward()` produca gradienti finiti (`torch.isfinite(z_A.grad).all()`).
4. `test_pyg_dataloader_on_triplet_dataset()`: verifica che `torch_geometric.loader.DataLoader(triplet_ds, batch_size=8)` restituisca `(batch_A, batch_P, batch_N, batch_alpha)` con `batch_alpha.shape == (8,)`.

---

## 4. Criteri di Accettazione del Settore 05 (Definition of Done)

- [ ] I file `src/training/augmentation.py`, `src/training/triplet_sampler.py`, `src/training/losses.py` e `tests/test_triplets.py` sono stati creati.
- [ ] Il comando `pytest tests/test_triplets.py` passa al 100% (4/4 test verdi).
