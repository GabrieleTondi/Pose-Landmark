# 01 — Sottospecifica di Settore: Ambiente, Hardware Profiling e Schemi YAML Esatti

**Settore:** 01 — Infrastruttura, Ambiente di Calcolo e Configurazione Dichiarativa  
**Riferimenti Specifica Madre:** §2, §13, §14, §15  
**Dipendenze in ingresso:** Nessuna  
**Output abilitati:** Tutti i settori successivi (`02`–`08`)

---

## 1. Obiettivo del Settore

Predisporre l'albero delle directory del progetto, i moduli di utilità di base (`seed.py`, `hardware.py`, `config.py`, `env_check.py`) e i **tre file di configurazione YAML completi** (`configs/data.yaml`, `configs/graph.yaml`, `configs/training.yaml`). Il modello che implementa questo settore deve scrivere file esattamente conformi agli schemi letterali riportati qui sotto.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 1.1 — Modulo Determinismo (`src/utils/seed.py`)
- **File da creare:** `src/utils/seed.py`
- **Contratto della funzione:**
  ```python
  import os
  import random
  import numpy as np
  import torch

  def seed_everything(seed: int = 42) -> None:
      """Fissa il seed globale per Python, NumPy e PyTorch (CPU e CUDA)."""
  ```
- **COME (Passi esatti da implementare nella funzione):**
  1. Impostare `os.environ["PYTHONHASHSEED"] = str(seed)`.
  2. Chiamare `random.seed(seed)` e `np.random.seed(seed)`.
  3. Chiamare `torch.manual_seed(seed)`.
  4. Se `torch.cuda.is_available()` è `True`, chiamare `torch.cuda.manual_seed(seed)` e `torch.cuda.manual_seed_all(seed)`.
  5. Impostare `torch.backends.cudnn.deterministic = True` e `torch.backends.cudnn.benchmark = False`.
- **CONSTRAINT:** Non usare `torch.use_deterministic_algorithms(True)` senza `warn_only=True`, perché alcune operazioni di scatter in PyTorch Geometric su CUDA non hanno implementazione deterministica atomica e solleverebbero un `RuntimeError` bloccante.

---

### Task 1.2 — Modulo Profiling Hardware (`src/utils/hardware.py`)
- **File da creare:** `src/utils/hardware.py`
- **Contratti delle funzioni:**
  ```python
  from typing import Dict, Any
  import torch

  def get_compute_device(prefer_gpu: bool = True) -> torch.device:
      """Restituisce torch.device('cuda') se disponibile e funzionante, altrimenti torch.device('cpu')."""

  def get_hardware_profile() -> Dict[str, Any]:
      """Restituisce un dizionario serializzabile JSON con i dettagli CPU/GPU/VRAM/CUDA."""

  def get_vram_usage_mb() -> Dict[str, float]:
      """Restituisce {'allocated_mb': ..., 'reserved_mb': ..., 'max_allocated_mb': ...} (0.0 se su CPU)."""
  ```
- **COME (Logica esatta da implementare):**
  1. In `get_compute_device(prefer_gpu=True)`:
     - Se `prefer_gpu` e `torch.cuda.is_available()`: eseguire un micro-test dentro un blocco `try...except Exception`: creare `t = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")`. Se riesce, restituire `torch.device("cuda")`; se fallisce (es. incompatibilità kernel CUDA), stampare un warning e restituire `torch.device("cpu")`.
     - Altrimenti restituire `torch.device("cpu")`.
  2. In `get_hardware_profile()`:
     - Restituire un dizionario con le chiavi esatte:
       - `"pytorch_version": str(torch.__version__)`
       - `"cuda_available": bool(torch.cuda.is_available())`
       - `"cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else None`
       - `"device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"`
       - `"total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2) if torch.cuda.is_available() else 0.0`
- **CONSTRAINT:**
  - Nessun altro modulo del progetto deve mai chiamare `.cuda()`; tutti i moduli devono usare `.to(device)` con il `device` restituito da `get_compute_device()`.
  - Durante il training, `max_allocated_mb` deve restare $< 1024.0\text{ MB}$.

---

### Task 1.3 — Caricatore e Validatore Configurazioni YAML (`src/utils/config.py`)
- **File da creare:** `src/utils/config.py`, `configs/data.yaml`, `configs/graph.yaml`, `configs/training.yaml`
- **Contratti delle funzioni:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, Union
  import yaml

  def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
      """Carica un singolo file YAML e verifica che non sia vuoto."""

  def load_all_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, Dict[str, Any]]:
      """
      Carica e valida data.yaml, graph.yaml e training.yaml.
      Restituisce {"data": data_cfg, "graph": graph_cfg, "training": train_cfg}.
      """
  ```
- **COME (Contenuto ESATTO dei 3 file YAML da scrivere in `configs/`):**

#### A. Contenuto esatto di `configs/data.yaml`:
```yaml
paths:
  raw_finefs_dir: "data/raw/finefs"
  raw_mmfs_dir: "data/raw/mmfs"
  intermediate_dir: "data/intermediate"
  processed_dir: "data/processed"
  splits_dir: "data/splits"
  reports_dir: "reports"
  checkpoints_dir: "checkpoints"

dataset:
  primary_dataset: "finefs"
  enable_mmfs: false
  skeleton_convention: "coco17"  # Valori ammessi: "coco17" oppure "h36m17" (verificato in Task 2.2)
  target_elements:
    - "Camel Spin"
    - "Sit Spin"
    - "Layback Spin"

windowing:
  window_size: 9
  min_valid_frames: 5
  trim_ratio_start: 0.20
  trim_ratio_end: 0.20
  min_keypoint_confidence: 0.20

canonicalization:
  horizontal_axes: [0, 1]       # Indici degli assi orizzontali: 0=X, 1=Y (se Z è verticale)
  vertical_axis: 2              # Indice dell'asse verticale: 2=Z (impostare a 1 se Y è verticale)
  scale_epsilon: 1.0e-4
  yaw_epsilon: 1.0e-4
  # Indici per standard COCO-17 (se h36m17: pelvis=[4, 1], shoulders=[11, 14]):
  pelvis_indices: [11, 12]      # [hip_left, hip_right]
  shoulder_indices: [5, 6]      # [shoulder_left, shoulder_right]

splits:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

#### B. Contenuto esatto di `configs/graph.yaml`:
```yaml
num_nodes: 17
node_feature_mode: "baseline"   # "baseline" -> shape (17, 3) | "extended" -> shape (17, 5)
node_feature_dim: 3
edge_feature_dim: 5
epsilon: 1.0e-8

# Nomi standard COCO-17:
joint_names:
  0: "nose"
  1: "left_eye"
  2: "right_eye"
  3: "left_ear"
  4: "right_ear"
  5: "left_shoulder"
  6: "right_shoulder"
  7: "left_elbow"
  8: "right_elbow"
  9: "left_wrist"
  10: "right_wrist"
  11: "left_hip"
  12: "right_hip"
  13: "left_knee"
  14: "right_knee"
  15: "left_ankle"
  16: "right_ankle"

# Indicatore di simmetria per modalità extended (-1: sx, 0: centro, +1: dx):
joint_symmetry: [0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]

# 16 connessioni anatomiche non orientate (ossa b = 0..15).
# Ogni osso b=[i, j] genera l'arco diretto 2*b=(i->j) e l'arco diretto 2*b+1=(j->i), per un totale di 32 archi diretti.
undirected_bones:
  - [0, 1]    # Bone 0: nose - left_eye
  - [0, 2]    # Bone 1: nose - right_eye
  - [1, 3]    # Bone 2: left_eye - left_ear
  - [2, 4]    # Bone 3: right_eye - right_ear
  - [0, 5]    # Bone 4: nose - left_shoulder
  - [0, 6]    # Bone 5: nose - right_shoulder
  - [5, 6]    # Bone 6: left_shoulder - right_shoulder
  - [5, 7]    # Bone 7: left_shoulder - left_elbow
  - [7, 9]    # Bone 8: left_elbow - left_wrist
  - [6, 8]    # Bone 9: right_shoulder - right_elbow
  - [8, 10]   # Bone 10: right_elbow - right_wrist
  - [5, 11]   # Bone 11: left_shoulder - left_hip
  - [6, 12]   # Bone 12: right_shoulder - right_hip
  - [11, 12]  # Bone 13: left_hip - right_hip
  - [11, 13]  # Bone 14: left_hip - left_knee
  - [13, 15]  # Bone 15: left_knee - left_ankle
  # Nota: se si preferisce includere la gamba destra mantenendo 16 ossa su albero a 17 nodi,
  # la lista standard ad albero connesso (17 nodi, 16 archi) senza cicli su occhi/spalle è:
  # [[0,1], [0,2], [1,3], [2,4], [0,5], [0,6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,13], [13,15], [12,14], [14,16]]
```
*(ATTENZIONE per il coding model: in `configs/graph.yaml`, utilizzare come lista predefinita di `undirected_bones` esattamente l'albero connesso a 16 ossa che copre tutti i 17 joint inclusi ginocchio destro `14` e caviglia destra `16`:)*
```yaml
undirected_bones:
  - [0, 1]    # Bone 0: nose - left_eye
  - [0, 2]    # Bone 1: nose - right_eye
  - [1, 3]    # Bone 2: left_eye - left_ear
  - [2, 4]    # Bone 3: right_eye - right_ear
  - [0, 5]    # Bone 4: nose - left_shoulder
  - [0, 6]    # Bone 5: nose - right_shoulder
  - [5, 7]    # Bone 6: left_shoulder - left_elbow
  - [7, 9]    # Bone 7: left_elbow - left_wrist
  - [6, 8]    # Bone 8: right_shoulder - right_elbow
  - [8, 10]   # Bone 9: right_elbow - right_wrist
  - [5, 11]   # Bone 10: left_shoulder - left_hip
  - [6, 12]   # Bone 11: right_shoulder - right_hip
  - [11, 13]  # Bone 12: left_hip - left_knee
  - [13, 15]  # Bone 13: left_knee - left_ankle
  - [12, 14]  # Bone 14: right_hip - right_knee
  - [14, 16]  # Bone 15: right_knee - right_ankle

# Mappa deterministica delle 32 terne angolari [a, b, c] (angolo calcolato sul vertice centrale b)
# per ciascun arco diretto "i->j":
directed_edge_angle_triplets:
  "0->1": [2, 0, 1]
  "1->0": [3, 1, 0]
  "0->2": [1, 0, 2]
  "2->0": [4, 2, 0]
  "1->3": [0, 1, 3]
  "3->1": [0, 1, 3]
  "2->4": [0, 2, 4]
  "4->2": [0, 2, 4]
  "0->5": [6, 0, 5]
  "5->0": [7, 5, 0]
  "0->6": [5, 0, 6]
  "6->0": [8, 6, 0]
  "5->7": [11, 5, 7]
  "7->5": [9, 7, 5]
  "7->9": [5, 7, 9]
  "9->7": [5, 7, 9]
  "6->8": [12, 6, 8]
  "8->6": [10, 8, 6]
  "8->10": [6, 8, 10]
  "10->8": [6, 8, 10]
  "5->11": [7, 5, 11]
  "11->5": [13, 11, 5]
  "6->12": [8, 6, 12]
  "12->6": [14, 12, 6]
  "11->13": [5, 11, 13]
  "13->11": [15, 13, 11]
  "13->15": [11, 13, 15]
  "15->13": [11, 13, 15]
  "12->14": [6, 12, 14]
  "14->12": [16, 14, 12]
  "14->16": [12, 14, 16]
  "16->14": [12, 14, 16]

# Mappa dei sotto-alberi cinematici discendenti per la rotazione di Rodrigues (Settore 05 e Settore 08):
# Chiave "parent->child": lista di tutti i nodi nel sotto-albero che ruotano attorno a "parent".
kinematic_subtrees:
  "5->7": [7, 9]          # Spalla SX -> ruota gomito SX e polso SX
  "7->9": [9]             # Gomito SX -> ruota polso SX
  "6->8": [8, 10]         # Spalla DX -> ruota gomito DX e polso DX
  "8->10": [10]           # Gomito DX -> ruota polso DX
  "11->13": [13, 15]      # Anca SX -> ruota ginocchio SX e caviglia SX
  "13->15": [15]          # Ginocchio SX -> ruota caviglia SX
  "12->14": [14, 16]      # Anca DX -> ruota ginocchio DX e caviglia DX
  "14->16": [16]          # Ginocchio DX -> ruota caviglia DX
```

#### C. Contenuto esatto di `configs/training.yaml`:
```yaml
model:
  in_channels: 3
  edge_dim: 5
  hidden_dim: 32
  embed_dim: 32
  heads: 4
  dropout: 0.20
  normalize_embeddings: false

training:
  seed: 42
  batch_size: 32
  lr: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs: 80
  early_stopping_patience: 15
  grad_clip_max_norm: 1.0
  num_train_triplets_per_epoch: 600
  num_val_triplets: 200
  num_test_triplets: 200

sampling:
  tier_rule: "threshold"          # "threshold" (con fallback automatico a "quantile" se pool < 5)
  goe_pos_threshold: 1.5
  goe_neg_threshold: 0.0
  goe_pos_quantile: 0.70
  goe_neg_quantile: 0.35
  max_pos_goe_diff: 1.5
  min_neg_goe_gap: 1.5
  real_negative_prob: 0.50

augmentation:
  enabled: true
  num_synthetic_per_positive: 3   # 1 per ciascun tier (1=lieve, 2=medio, 3=grave)
  noise_std: 0.005
  tier_angle_ranges_deg:
    1: [8.0, 15.0]
    2: [16.0, 28.0]
    3: [29.0, 42.0]

loss:
  margin_mode: "adaptive"         # "adaptive" oppure "fixed"
  fixed_margin: 0.50
  adaptive_margins:
    1: 0.20
    2: 0.60
    3: 1.00
```

- **CONSTRAINT di Validazione in `load_all_configs()`:**
  - Verificare `len(graph_cfg["undirected_bones"]) == 16`.
  - Verificare `len(graph_cfg["directed_edge_angle_triplets"]) == 32`.
  - Verificare `abs(data_cfg["splits"]["train_ratio"] + data_cfg["splits"]["val_ratio"] + data_cfg["splits"]["test_ratio"] - 1.0) < 1e-6`.
  - Se una qualsiasi di queste asserzioni fallisce, sollevare `ValueError` con messaggio descrittivo.

---

### Task 1.4 — Script Smoke Test Ambiente (`src/utils/env_check.py`)
- **File da creare:** `src/utils/env_check.py`
- **COME (Passi esatti da eseguire nel blocco `if __name__ == "__main__":`):**
  1. Chiamare `seed_everything(42)`.
  2. Chiamare `cfgs = load_all_configs("configs")`.
  3. Chiamare `device = get_compute_device()` e `profile = get_hardware_profile()`.
  4. Salvare `profile` in `reports/hardware_profile.json` (creando `Path("reports").mkdir(parents=True, exist_ok=True)`).
  5. Creare un grafo PyG dummy:
     - `x = torch.randn((17, 3), dtype=torch.float32, device=device, requires_grad=True)`
     - `edge_index = torch.randint(0, 17, (2, 32), dtype=torch.long, device=device)`
     - `edge_attr = torch.randn((32, 5), dtype=torch.float32, device=device)`
  6. Istanziare `conv = GATv2Conv(in_channels=3, out_channels=32, heads=4, edge_dim=5, add_self_loops=True).to(device)`.
  7. Eseguire `out, (edge_idx_attn, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)`.
  8. Verificare con `assert`:
     - `out.shape == (17, 128)`
     - `edge_idx_attn.shape == (2, 49)` *(32 archi + 17 self-loop)*
     - `alpha.shape == (49, 4)`
  9. Eseguire `out.sum().backward()` e verificare `assert x.grad is not None`.
  10. Stampare `"ENV_CHECK_OK"` e terminare con codice `0`.

---

## 3. Criteri di Accettazione del Settore 01 (Definition of Done)

- [ ] Tutte le cartelle del repository e i file `__init__.py` sono stati creati (`src/`, `src/utils/`, `src/data/`, `src/graph/`, `src/training/`, `src/models/`, `src/evaluation/`, `src/inference/`, `src/visualization/`, `tests/`, `configs/`, `data/raw/`, `data/intermediate/`, `data/processed/`, `data/splits/`, `checkpoints/`, `reports/figures/`).
- [ ] L'esecuzione di `python -m src.utils.env_check` stampa `"ENV_CHECK_OK"`, esce con codice `0` e crea il file `reports/hardware_profile.json`.
