# 02 — Sottospecifica di Settore: Dati, Esplorazione, Armonizzazione e Split

**Settore:** 02 — Data Engineering: FineFS, Verifica MMFS, `UnifiedSample` e Group Split Anti-Leakage  
**Riferimenti Specifica Madre:** §3.1–§3.5, §16.1, §18  
**Dipendenze in ingresso:** Settore 01 (`configs/data.yaml`, `src/utils/seed.py`, `src/utils/config.py`)  
**Output abilitati:** Settore 03 (Windowing e Canonicalizzazione 3D)

---

## 1. Obiettivo del Settore

Fornire le strutture dati e le funzioni per:
1. Ispezionare i file `.npz` e i metadati di FineFS senza assumere a priori i nomi delle chiavi;
2. Incapsulare ogni segmento scheletrico valido nella dataclass standardizzata `UnifiedSample`;
3. Generare uno split `train / val / test` raggruppato per atleta/esibizione (`GroupSplit`) con garanzia matematica di assenza di data leakage;
4. Fornire anche un generatore di dataset sintetico (`generate_mock_finefs_samples`) per consentire l'esecuzione dei test `pytest` e il collaudo dell'intera pipeline anche prima o in assenza dei file `.npz` completi di FineFS.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 2.1 — Struttura Dati `UnifiedSample` e Serializzazione (`src/data/harmonization.py`)
- **File da creare:** `src/data/harmonization.py`
- **Contratto esatto della classe e delle funzioni:**
  ```python
  from dataclasses import dataclass, field
  from pathlib import Path
  from typing import Optional, Dict, Any, List
  import pickle
  import numpy as np

  @dataclass
  class UnifiedSample:
      sample_id: str
      dataset_id: str                           # es. "finefs" o "synthetic_mock"
      performance_id: str                       # ID esibizione (usato per Group Split)
      athlete_id: Optional[str]                 # ID atleta (se disponibile, altrimenti None)
      element_label: str                        # es. "Camel Spin", "Sit Spin", "Layback Spin"
      start_frame: int
      end_frame: int
      keypoints: np.ndarray                     # Shape (T_seg, 17, 3), dtype float32
      keypoint_confidence: Optional[np.ndarray] # Shape (T_seg, 17), dtype float32 oppure None
      goe: float                                # Punteggio GOE reale del singolo elemento
      source_metadata: Dict[str, Any] = field(default_factory=dict)

      def validate(self) -> None:
          """Solleva ValueError se i campi o le shape violano i vincoli dello schema unificato."""

  def save_unified_samples(samples: List[UnifiedSample], output_path: Path) -> None:
      """Valida ogni campione e salva la lista con pickle in output_path."""

  def load_unified_samples(input_path: Path) -> List[UnifiedSample]:
      """Carica la lista di UnifiedSample da input_path e riesegue validate()."""
  ```
- **COME (Regole esatte da implementare dentro `UnifiedSample.validate()`):**
  1. Verificare `isinstance(self.sample_id, str) and len(self.sample_id) > 0`.
  2. Verificare `isinstance(self.performance_id, str) and len(self.performance_id) > 0`.
  3. Verificare `isinstance(self.keypoints, np.ndarray)` e convertire se necessario in `np.float32`.
  4. Verificare `self.keypoints.ndim == 3 and self.keypoints.shape[1] == 17 and self.keypoints.shape[2] == 3`.
  5. Verificare `self.keypoints.shape[0] >= 1` e `self.end_frame >= self.start_frame`.
  6. Se `self.keypoint_confidence is not None`: verificare `self.keypoint_confidence.shape == (self.keypoints.shape[0], 17)`.
  7. Verificare `np.isfinite(self.goe)` (non deve essere `NaN` o `None`).
- **CONSTRAINT:** Non eliminare mai `source_metadata`; deve contenere almeno `{"raw_file": ..., "raw_label": ...}`.

---

### Task 2.2 — Audit e Loader per FineFS + Generatore Mock di Collaudo (`src/data/loaders.py`)
- **File da creare:** `src/data/loaders.py`
- **Contratti delle funzioni:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List, Optional, Tuple
  import numpy as np
  from src.data.harmonization import UnifiedSample

  def inspect_npz_file(npz_path: Path) -> Dict[str, Any]:
      """
      Apre un file .npz e restituisce:
      - 'valid': bool
      - 'keys': List[str]
      - 'shapes': Dict[str, Tuple[int, ...]]
      - 'dtypes': Dict[str, str]
      - 'has_nan': bool
      - 'is_3d_nonzero': bool (True se l'asse Z ha varianza > 1e-6)
      - 'error': Optional[str]
      """

  def audit_finefs_directory(raw_dir: Path) -> Dict[str, Any]:
      """
      Scansiona raw_dir, ispeziona i file .npz e i file di annotazione (.json/.csv/.pkl),
      e restituisce un report completo salvabile in reports/finefs_inventory.json.
      """

  def extract_3d_keypoints_from_array(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      """
      Normalizza le diverse shape possibili degli array HRNet/FineFS in:
      - keypoints_3d: np.ndarray di shape (T, 17, 3) in float32
      - confidence: Optional[np.ndarray] di shape (T, 17) in float32
      Gestisce i casi:
      - shape (T, 17, 3) -> keypoints=arr, conf=None
      - shape (T, 17, 4) -> keypoints=arr[:, :, :3], conf=arr[:, :, 3]
      - shape (1, T, 17, 3) o (M, T, 17, 3) -> prende la persona principale arr[0]
      - shape (T, 51) -> reshape a (T, 17, 3)
      Solleva ValueError per qualsiasi altra shape non riconducibile a 17 joint 3D.
      """

  def generate_mock_finefs_samples(
      num_performances: int = 30,
      elements: Tuple[str, ...] = ("Camel Spin", "Sit Spin", "Layback Spin"),
      seed: int = 42
  ) -> List[UnifiedSample]:
      """
      Genera una collezione di UnifiedSample biomeccanicamente plausibili a 17 joint (standard COCO-17)
      con diversi atleti, rotazioni di piroetta frame-by-frame, traslazioni, scale e punteggi GOE in [-3.0, +3.0]
      correlati alla precisione degli angoli articolari.
      Usato per i test automatici pytest e per verificare la pipeline end-to-end.
      """
  ```
- **COME implementare `generate_mock_finefs_samples`:**
  1. Definire per ciascuno dei 3 elementi (`"Camel Spin"`, `"Sit Spin"`, `"Layback Spin"`) un template base 3D a 17 joint $\hat{P}_c \in \mathbb{R}^{17 \times 3}$ in coordinate COCO-17 con lunghezza del busto $\approx 0.5\text{ m}$.
  2. Per ogni esibizione $p \in \{0, \dots, \text{num\_performances}-1\}$ (associata a un `athlete_id = f"ath_{p // 2:03d}"` e `performance_id = f"perf_{p:04d}"`):
     - Generare un elemento per ciascuna classe $c$ con durata $T_{\text{seg}} = 25$ frame;
     - Estrarre un punteggio `goe` da $[-3.0, +3.0]$;
     - Se `goe < 0`, applicare una deviazione angolare proporzionale a $|\text{goe}|$ (es. abbassare la gamba libera o piegare il ginocchio di $5^\circ \cdot |\text{goe}|$) così che nei dati mock esista una vera relazione fisica tra geometria dello scheletro e GOE!
     - Per ogni frame $t \in \{0, \dots, 24\}$, applicare una rotazione di piroetta attorno all'asse $Z$ di angolo $\omega t$, un fattore di scala $\lambda_p \in [0.8, 1.5]$ e una traslazione globale $\mathbf{t}_p \in [-2, 2]^3$, aggiungendo un transitorio nei primi 4 e ultimi 4 frame.
- **CONSTRAINT:**
  - Quando la cartella `data/raw/finefs` contiene i file reali di FineFS, il loader carica i file reali; nei test unitari (`tests/test_dataset.py`), o quando si esegue uno smoke test senza i file `.npz` scaricati, i test usano `generate_mock_finefs_samples` per verificare deterministicamente tutti i contratti.

---

### Task 2.3 — Partizionamento Anti-Leakage per Atleta/Esibizione (`create_grouped_splits`)
- **File coinvolti:** `src/data/loaders.py`
- **Contratto della funzione:**
  ```python
  def create_grouped_splits(
      samples: List[UnifiedSample],
      data_cfg: Dict[str, Any]
  ) -> Dict[str, List[str]]:
      """
      Suddivide i sample_id in {'train': [...], 'val': [...], 'test': [...]}
      garantendo che tutti i campioni con lo stesso athlete_id e performance_id
      cadano nello stesso identico split.
      """
  ```
- **COME (Algoritmo passo-passo privo di ambiguità):**
  1. Per ogni `s` in `samples`, determinare l'identificativo di gruppo:
     ```python
     group_id = f"ath_{s.athlete_id}" if s.athlete_id else f"perf_{s.performance_id}"
     ```
     *(Inoltre, raccogliere tutti i `performance_id` associati a ciascun `group_id` per verificare che nessun `performance_id` sia mai condiviso tra gruppi diversi).*
  2. Raggruppare i campioni per `group_id`. Ordinare la lista dei `group_id` unici e mescolarla in modo deterministico con `rng = np.random.default_rng(data_cfg["splits"]["seed"])`.
  3. Assegnare i gruppi a `train`, `val` e `test` rispettando le proporzioni `train_ratio` (`0.70`), `val_ratio` (`0.15`), `test_ratio` (`0.15`), verificando che in ciascuno dei 3 split compaia almeno un campione con $\text{GOE} \ge \text{mediana}$ e uno con $\text{GOE} < \text{mediana}$ per ogni classe `element_label` (ripetendo lo shuffle fino a 50 tentativi se uno split piccolo risultasse privo di positivi o negativi per una classe).
  4. Eseguire le **3 asserzioni obbligatorie anti-leakage** prima del `return`:
     ```python
     train_perfs = {s.performance_id for s in samples if s.sample_id in split_dict["train"]}
     val_perfs   = {s.performance_id for s in samples if s.sample_id in split_dict["val"]}
     test_perfs  = {s.performance_id for s in samples if s.sample_id in split_dict["test"]}
     assert train_perfs.isdisjoint(val_perfs), "LEAKAGE RILEVATO: train e val condividono performance_id!"
     assert train_perfs.isdisjoint(test_perfs), "LEAKAGE RILEVATO: train e test condividono performance_id!"
     assert val_perfs.isdisjoint(test_perfs), "LEAKAGE RILEVATO: val e test condividono performance_id!"
     ```
     *(Ripetere la stessa identica asserzione anche sugli `athlete_id` diversi da `None`).*
- **CONSTRAINT:** Non usare mai `sklearn.model_selection.train_test_split` direttamente sulla lista `samples` senza raggruppamento per `group_id`.

---

## 3. Unit Test da Implementare (`tests/test_dataset.py`)

Il file `tests/test_dataset.py` deve implementare i seguenti 4 test automatici:
1. `test_unified_sample_validation()`: verifica che un `UnifiedSample` valido passi `.validate()`, mentre uno con shape `(10, 16, 3)` o `goe = np.nan` sollevi `ValueError`.
2. `test_extract_3d_keypoints_shapes()`: verifica che array di shape `(20, 17, 3)`, `(20, 17, 4)`, `(2, 20, 17, 3)` e `(20, 51)` vengano tutti convertiti in `(20, 17, 3)` `float32`.
3. `test_grouped_splits_zero_leakage()`: genera 40 esibizioni con `generate_mock_finefs_samples()`, chiama `create_grouped_splits()` e verifica che l'intersezione di `performance_id` e `athlete_id` tra `train`, `val` e `test` sia `set()`.
4. `test_save_and_load_unified_samples(tmp_path)`: verifica il salvataggio e il ricaricamento round-trip senza perdita di metadati.

---

## 4. Criteri di Accettazione del Settore 02 (Definition of Done)

- [ ] I file `src/data/harmonization.py`, `src/data/loaders.py` e `tests/test_dataset.py` sono stati creati.
- [ ] Il comando `pytest tests/test_dataset.py` passa al 100% (4/4 test verdi).
