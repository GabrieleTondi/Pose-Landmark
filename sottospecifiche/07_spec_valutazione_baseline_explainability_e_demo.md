# 07 — Sottospecifica di Settore: Valutazione, Baseline, Explainability e Demo

**Settore:** 07 — Valutazione Sperimentale, Baseline Geometriche Non-Neurali, Calibrazione Indice $Q$, Explainability (XAI) e Demo  
**Riferimenti Specifica Madre:** §9, §10, §11, §12, §16.5  
**Dipendenze in ingresso:** Settore 04 (Grafi PyG), Settore 06 (`SiameseGATv2`, `checkpoints/best_model.pt`)  
**Output abilitati:** Settore 08 (Dashboard 3D Streamlit e Figure Vettoriali Report)

---

## 1. Obiettivo del Settore

Implementare:
1. Le **3 Baseline Geometriche Non-Neurali** (`src/evaluation/baselines.py`);
2. Il calcolo dei **Prototipi di Riferimento per Classe** (Centroide Latente $z_R^{(c)}$ + Grafo Medoide $G_R^{(c)}$), la calibrazione del parametro $\beta$ per l'indice $Q = 100\exp(-\beta d)$ sul validation set e l'estrazione simmetrizzata dei pesi di attenzione sulle 16 ossa (`src/evaluation/metrics.py`);
3. Lo script di valutazione comparativa completa sul test set (`src/evaluation/evaluate.py`);
4. Lo script di inferenza su singolo campione (`src/inference/predict.py`).

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 7.1 — Le 3 Baseline Geometriche Non-Neurali (`src/evaluation/baselines.py`)
- **File da creare:** `src/evaluation/baselines.py`
- **Contratti esatti delle funzioni:**
  ```python
  import torch
  import torch.nn.functional as F
  from torch_geometric.data import Data

  def coord_euclidean_distance(g1: Data, g2: Data) -> float:
      """Baseline 1: Distanza Euclidea di Frobenius tra le coordinate 3D canonicalizzate g1.pos e g2.pos (17, 3)."""
      diff = g1.pos.cpu().float() - g2.pos.cpu().float()
      return float(torch.norm(diff, p="fro").item())

  def angle_euclidean_distance(g1: Data, g2: Data) -> float:
      """Baseline 2: Distanza Euclidea L2 tra i coseni degli angoli articolari (5a colonna di edge_attr, indice 4)."""
      c1 = g1.edge_attr[:, 4].cpu().float()
      c2 = g2.edge_attr[:, 4].cpu().float()
      return float(torch.norm(c1 - c2, p=2).item())

  def feature_cosine_distance(g1: Data, g2: Data) -> float:
      """Baseline 3: Cosine distance 1 - cos(f1, f2) sul vettore concatenato [vec(pos) || vec(edge_attr)]."""
      f1 = torch.cat([g1.pos.cpu().float().flatten(), g1.edge_attr.cpu().float().flatten()], dim=0)
      f2 = torch.cat([g2.pos.cpu().float().flatten(), g2.edge_attr.cpu().float().flatten()], dim=0)
      cos_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=-1, eps=1e-8).item()
      return float(1.0 - cos_sim)
  ```

---

### Task 7.2 — Prototipi $z_R^{(c)}$, Medoidi $G_R^{(c)}$, Calibrazione $\beta$ ed Estrazione Attenzione (`src/evaluation/metrics.py`)
- **File da creare:** `src/evaluation/metrics.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any, List, Union
  import numpy as np
  from scipy.stats import spearmanr
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  @torch.no_grad()
  def compute_class_references(
      train_graphs: List[Data],
      model: SiameseGATv2,
      goe_thresholds: Dict[str, tuple],
      device: torch.device
  ) -> Dict[str, Dict[str, Any]]:
      """
      Per ogni elemento tecnico c presente in train_graphs, filtra i campioni reali di alta qualità
      (not g.is_augmented and g.goe >= goe_thresholds[c][1]) e calcola:
      - 'centroid_z': Tensor di shape (embed_dim,) su CPU (media degli embedding z)
      - 'medoid_graph': oggetto Data reale in R_c^train il cui embedding z è più vicino a 'centroid_z'
      - 'mean_pos': Tensor (17, 3) media delle coordinate pos
      - 'mean_edge_attr': Tensor (32, 5) media di edge_attr
      - 'num_ref_samples': int
      """

  def calibrate_similarity_beta(val_distances: np.ndarray) -> float:
      """
      Calibra beta > 0 esclusivamente sulle distanze dal prototipo del validation set,
      imponendo che la distanza mediana di validation corrisponda a Q = 50.0:
      beta = ln(2.0) / (median(val_distances) + 1e-8).
      """

  def distance_to_similarity_index(
      d: Union[float, np.ndarray],
      beta: float
  ) -> Union[float, np.ndarray]:
      """
      Restituisce l'Indice Interno di Similarità Q = 100.0 * exp(-beta * d) in [0, 100].
      """

  @torch.no_grad()
  def extract_bone_attention_weights(
      model: SiameseGATv2,
      graph: Data,
      layer_name: str = "layer2",
      device: torch.device = torch.device("cpu")
  ) -> Dict[str, np.ndarray]:
      """
      Esegue model.encode(graph, return_attention_weights=True) per un singolo grafo (17 nodi, 32 archi).
      Restituisce:
      - 'bone_weights_norm': np.ndarray di shape (16,) normalizzato a somma 1.0 sulle 16 ossa non orientate
      - 'bone_weights_raw': np.ndarray di shape (16,) media delle 4 head su ciascuna coppia {2b, 2b+1}
      - 'head_matrix': np.ndarray di shape (16, 4) con i pesi per osso e per singola head
      """
  ```
- **COME implementare `extract_bone_attention_weights` (Codice esatto privo di bug sugli indici):**
  ```python
  model.eval()
  g = graph.clone().to(device)
  _, attn_dict = model.encode(g, return_attention_weights=True)
  edge_idx_attn, alpha = attn_dict[layer_name]
  # In PyG GATv2Conv(add_self_loops=True), i primi 32 archi sono esattamente i 32 archi passati in g.edge_index
  # e gli ultimi 17 sono i self-loops (i -> i). Verifichiamo o filtriamo con mask:
  non_self_mask = edge_idx_attn[0] != edge_idx_attn[1]
  alpha_directed = alpha[non_self_mask].detach().cpu().numpy()  # shape (32, H) dove H=4
  assert alpha_directed.shape[0] == 32, f"Attesi 32 archi diretti, trovati {alpha_directed.shape[0]}"

  # Poiché nel Settore 04 l'osso b (b=0..15) genera gli archi contigui 2*b e 2*b+1:
  # Reshape a (16, 2, H) e media sulle 2 direzioni opposte (i->j) e (j->i):
  head_matrix = alpha_directed.reshape(16, 2, -1).mean(axis=1)  # shape (16, 4)
  bone_weights_raw = head_matrix.mean(axis=-1)                  # shape (16,)
  bone_weights_norm = bone_weights_raw / (np.sum(bone_weights_raw) + 1e-8)
  return {
      "bone_weights_norm": bone_weights_norm.astype(np.float32),
      "bone_weights_raw": bone_weights_raw.astype(np.float32),
      "head_matrix": head_matrix.astype(np.float32)
  }
  ```

---

### Task 7.3 — Valutazione Completa vs Baseline (`src/evaluation/evaluate.py`)
- **File da creare:** `src/evaluation/evaluate.py`
- **Contratto della funzione principale:**
  ```python
  from typing import Dict, Any, List
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  def evaluate_all_methods(
      model: SiameseGATv2,
      train_graphs: List[Data],
      val_graphs: List[Data],
      test_graphs: List[Data],
      goe_thresholds: Dict[str, tuple],
      train_cfg: Dict[str, Any],
      device: torch.device
  ) -> Dict[str, Any]:
      """
      Calcola e restituisce un dizionario strutturato salvato in reports/evaluation_results.json con:
      1. 'triplet_accuracy': {'siamese_gatv2': ..., 'baseline_coord': ..., 'baseline_angle': ..., 'baseline_cosine': ...}
      2. 'spearman_overall': {'siamese_gatv2': {'rho': ..., 'p_val': ...}, 'baseline_coord': ..., ...}
      3. 'spearman_by_element': {element_label: {...}}
      4. 'beta_calibrated': float
      5. 'test_samples_records': lista di dict per ogni campione di test con
         {'sample_id', 'element_label', 'goe', 'd_gatv2', 'Q_gatv2', 'd_coord', 'd_angle', 'd_cos'}
      """
  ```
- **COME calcolare Spearman $\rho$ correttamente:**
  - Per ogni campione di test $G_i$ di classe $c$, la distanza dal riferimento per la Siamese GATv2 è $d_i = \| f_\theta(G_i) - z_R^{(c)} \|_2$, mentre per le 3 baseline è la distanza rispetto al grafo medoide/centroide di riferimento della classe $c$.
  - Poiché una distanza piccola deve corrispondere a un GOE alto, chiamare `rho, p_val = spearmanr(-distances_array, goe_array)`. Se `np.isnan(rho)` (caso degenere con valori costanti), impostare `rho = 0.0, p_val = 1.0`.

---

### Task 7.4 — Inferenza su Singolo Campione (`src/inference/predict.py`)
- **File da creare:** `src/inference/predict.py`
- **Contratto della funzione:**
  ```python
  from typing import Dict, Any
  import torch
  from torch_geometric.data import Data
  from src.models.siamese_gat import SiameseGATv2

  @torch.no_grad()
  def predict_single_sample(
      graph: Data,
      model: SiameseGATv2,
      class_references: Dict[str, Dict[str, Any]],
      beta: float,
      device: torch.device = torch.device("cpu")
  ) -> Dict[str, Any]:
      """
      Dato un grafo PyG di un atleta, calcola:
      - 'sample_id': graph.sample_id
      - 'element_label': graph.element_label
      - 'true_goe': float(graph.goe)
      - 'distance_to_ref': float d(A, R)
      - 'similarity_index_Q': float Q in [0, 100]
      - 'reference_sample_id': medoid_graph.sample_id
      - 'athlete_attention': dict restituito da extract_bone_attention_weights(model, graph)
      - 'reference_attention': dict restituito da extract_bone_attention_weights(model, medoid_graph)
      - 'bone_angle_diff_deg': np.ndarray di shape (16,) con la differenza angolare in gradi sulle 16 ossa
      """
  ```

---

## 3. Criteri di Accettazione del Settore 07 (Definition of Done)

- [ ] I file `src/evaluation/baselines.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluate.py` e `src/inference/predict.py` sono stati creati.
- [ ] `extract_bone_attention_weights` restituisce vettori di shape `(16,)` finiti e con somma di `bone_weights_norm` pari a $1.0 \pm 10^{-5}$.
- [ ] `evaluate_all_methods` genera `reports/evaluation_results.json` contenente il confronto tra `siamese_gatv2` e le 3 baseline geometriche.
