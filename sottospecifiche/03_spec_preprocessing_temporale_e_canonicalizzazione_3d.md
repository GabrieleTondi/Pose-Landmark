# 03 — Sottospecifica di Settore: Preprocessing Temporale e Canonicalizzazione 3D

**Settore:** 03 — Geometria 3D: Windowing Temporale, Controllo Qualità e Canonicalizzazione Scheletrica  
**Riferimenti Specifica Madre:** §4, §5.1–§5.5, §16.2  
**Dipendenze in ingresso:** Settore 01 (`configs/data.yaml`), Settore 02 (`UnifiedSample`)  
**Output abilitati:** Settore 04 (Costruzione del Grafo PyG)

---

## 1. Obiettivo del Settore

Implementare l'estrazione della finestra temporale stabile (`src/data/segmentation.py`) e la canonicalizzazione geometrica 3D (`src/data/preprocessing.py`).
**REGOLA MATEMATICA FONDAMENTALE:** Ogni singolo frame $t$ della finestra selezionata deve subire in ordine:
1. **Root-Centering** (traslazione dell'origine nel punto medio delle anche);
2. **Scale Normalization** (divisione per la lunghezza 3D del busto $s = \|p_{\text{shoulders}} - p_{\text{pelvis}}\|_2$);
3. **Yaw Alignment** (rotazione attorno all'asse verticale per allineare il vettore anca sinistra $\to$ anca destra lungo il semiasse positivo del primo asse orizzontale);
**PRIMA** di calcolare la media aritmetica temporale $\bar{P} = \frac{1}{T}\sum_{t=1}^T P'''_t$.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 3.1 — Isolamento della Finestra Temporale Stabile (`src/data/segmentation.py`)
- **File da creare:** `src/data/segmentation.py`
- **Contratto esatto della funzione:**
  ```python
  from typing import Optional, Dict, Any, Tuple
  import numpy as np

  def extract_stable_window(
      keypoints: np.ndarray,
      confidence: Optional[np.ndarray],
      data_cfg: Dict[str, Any]
  ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
      """
      Estrae la finestra centrale di frame validi dal segmento keypoints di shape (T_seg, 17, 3).
      Restituisce:
      - window_joints: np.ndarray di shape (T_win, 17, 3) in float32, oppure None se il segmento è invalido
      - info: dict con {'status': 'ok'|'discarded', 'reason': ..., 'selected_indices': [...]}
      """
  ```
- **COME (Algoritmo passo-passo da implementare):**
  1. Leggere da `w_cfg = data_cfg["windowing"]`:
     - `window_size = int(w_cfg["window_size"])` (default `9`)
     - `min_valid_frames = int(w_cfg["min_valid_frames"])` (default `5`)
     - `trim_start = float(w_cfg["trim_ratio_start"])` (default `0.20`)
     - `trim_end = float(w_cfg["trim_ratio_end"])` (default `0.20`)
     - `min_conf = float(w_cfg["min_keypoint_confidence"])` (default `0.20`)
  2. Sia $L = \text{keypoints.shape}[0]$. Se $L < \text{min\_valid\_frames}$, restituire `(None, {"status": "discarded", "reason": "segment_too_short", "selected_indices": []})`.
  3. Calcolare gli indici del nucleo centrale escludendo i transitori di ingresso/uscita:
     ```python
     start_idx = int(np.floor(L * trim_start))
     end_idx = int(np.ceil(L * (1.0 - trim_end)))
     if end_idx - start_idx < min_valid_frames:
         start_idx, end_idx = 0, L
     ```
  4. Per ogni indice di frame $t \in [\text{start\_idx}, \text{end\_idx})$ verificare che il frame $P_t = \text{keypoints}[t]$ sia valido:
     - `np.isfinite(P_t).all()` è `True`;
     - `np.std(P_t) > 1e-5` (non è uno scheletro collassato a zero);
     - Se `confidence is not None`: `np.mean(confidence[t]) >= min_conf`.
     Raccogliere gli indici che superano il controllo nella lista `valid_indices`.
  5. Se `len(valid_indices) < min_valid_frames`, restituire `(None, {"status": "discarded", "reason": "insufficient_valid_frames", "selected_indices": []})`.
  6. Se `len(valid_indices) > window_size`, prendere i `window_size` indici centrali della lista `valid_indices`:
     ```python
     mid = len(valid_indices) // 2
     half = window_size // 2
     chosen = valid_indices[max(0, mid - half) : max(0, mid - half) + window_size]
     ```
     altrimenti usare tutti i `valid_indices`.
  7. Restituire `(keypoints[chosen].astype(np.float32), {"status": "ok", "reason": None, "selected_indices": chosen})`.

---

### Task 3.2 — Canonicalizzazione Geometrica 3D Frame-by-Frame (`src/data/preprocessing.py`)
- **File da creare:** `src/data/preprocessing.py`
- **Contratti esatti dell'eccezione e delle funzioni:**
  ```python
  from typing import Dict, Any, Tuple
  import numpy as np

  class DegenerateSkeletonError(ValueError):
      """Sollevata quando uno scheletro ha lunghezza del busto nulla/degenere o vettore anca/spalla nullo."""
      pass

  def canonicalize_frame_3d(
      joints: np.ndarray,
      data_cfg: Dict[str, Any]
  ) -> Tuple[np.ndarray, Dict[str, float]]:
      """
      Applica Root-Centering, Scale Normalization e Yaw Alignment a un singolo frame di shape (17, 3).
      Restituisce:
      - aligned: np.ndarray di shape (17, 3), dtype float32
      - stats: {'torso_length': float, 'yaw_angle_rad': float}
      Solleva DegenerateSkeletonError se il frame contiene NaN/Inf o denominatori <= epsilon.
      """

  def canonicalize_window_3d(
      window_joints: np.ndarray,
      data_cfg: Dict[str, Any]
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
      """
      Applica canonicalize_frame_3d a ciascun frame valido della finestra (T_win, 17, 3)
      e poi calcola la posa media P_bar = mean(P_aligned, axis=0) di shape (17, 3).
      Solleva DegenerateSkeletonError se il numero di frame canonicalizzati con successo è < min_valid_frames.
      """
  ```
- **COME implementare `canonicalize_frame_3d(joints, data_cfg)` (Codice/Logica esatta):**
  1. **Controllo preliminare:**
     ```python
     if joints.shape != (17, 3) or not np.isfinite(joints).all():
         raise DegenerateSkeletonError("Frame shape errata o presenza di NaN/Inf.")
     c_cfg = data_cfg["canonicalization"]
     hip_l, hip_r = c_cfg["pelvis_indices"]         # default [11, 12]
     sh_l, sh_r = c_cfg["shoulder_indices"]         # default [5, 6]
     h_ax0, h_ax1 = c_cfg["horizontal_axes"]        # default [0, 1] (X, Y)
     scale_eps = float(c_cfg["scale_epsilon"])      # default 1e-4
     yaw_eps = float(c_cfg["yaw_epsilon"])          # default 1e-4
     ```
  2. **Passo 1 — Root-Centering:**
     ```python
     pelvis = 0.5 * (joints[hip_l] + joints[hip_r])
     centered = joints - pelvis  # shape (17, 3)
     ```
  3. **Passo 2 — Scale Normalization:**
     ```python
     shoulders_mid = 0.5 * (centered[sh_l] + centered[sh_r])
     torso_length = float(np.linalg.norm(shoulders_mid))  # poiché pelvis è già in (0,0,0)
     if torso_length <= scale_eps:
         raise DegenerateSkeletonError(f"Torso length degenere: {torso_length}")
     scaled = centered / torso_length
     ```
  4. **Passo 3 — Yaw Alignment sul piano orizzontale `(h_ax0, h_ax1)`:**
     ```python
     hip_vec = scaled[hip_r] - scaled[hip_l]
     vx = float(hip_vec[h_ax0])
     vy = float(hip_vec[h_ax1])
     horiz_norm = float(np.hypot(vx, vy))
     if horiz_norm <= yaw_eps:
         # Fallback sul vettore delle spalle se le anche sono verticalmente sovrapposte
         sh_vec = scaled[sh_r] - scaled[sh_l]
         vx = float(sh_vec[h_ax0])
         vy = float(sh_vec[h_ax1])
         horiz_norm = float(np.hypot(vx, vy))
         if horiz_norm <= yaw_eps:
             raise DegenerateSkeletonError("Impossibile determinare l'angolo di yaw: anche e spalle allineate verticalmente.")
     angle = float(np.arctan2(vy, vx))
     c = float(np.cos(-angle))
     s = float(np.sin(-angle))
     aligned = scaled.copy()
     # Rotazione sul piano (h_ax0, h_ax1) che lascia invariato l'asse verticale:
     x_old = scaled[:, h_ax0]
     y_old = scaled[:, h_ax1]
     aligned[:, h_ax0] = c * x_old - s * y_old
     aligned[:, h_ax1] = s * x_old + c * y_old
     return aligned.astype(np.float32), {"torso_length": torso_length, "yaw_angle_rad": angle}
     ```
- **COME implementare `canonicalize_window_3d(window_joints, data_cfg)`:**
  1. Iterare sui frame `t` di `window_joints` chiamando `canonicalize_frame_3d(window_joints[t], data_cfg)` dentro un blocco `try...except DegenerateSkeletonError`.
  2. Raccogliere i frame allineati validi in `aligned_frames`.
  3. Se `len(aligned_frames) < data_cfg["windowing"].get("min_valid_frames", 3)`, sollevare `DegenerateSkeletonError("Troppi frame degeneri nella finestra")`.
  4. Impilare `stack = np.stack(aligned_frames, axis=0)` di shape `(T_valid, 17, 3)` e calcolare:
     - `P_bar = np.mean(stack, axis=0).astype(np.float32)` (shape `(17, 3)`)
     - `window_jitter = float(np.sqrt(np.mean(np.sum((stack - P_bar[None, :, :])**2, axis=-1))))`
  5. Restituire `(P_bar, {"num_valid_frames": len(aligned_frames), "mean_torso_length": ..., "window_jitter": window_jitter})`.

- **CONSTRAINT (ANTI-PATTERN DA EVITARE ASSOLUTAMENTE):**
  - **NON** fare mai `else: scaled = centered` quando `torso_length <= scale_eps`: deve sempre sollevare `DegenerateSkeletonError`.
  - **NON** mediare mai `window_joints` prima di aver chiamato `canonicalize_frame_3d` su ciascun frame.

---

## 3. Unit Test da Implementare (`tests/test_normalization.py`)

Il file `tests/test_normalization.py` deve implementare i seguenti 5 test automatici:
1. `test_root_centering_at_origin()`: verifica che su uno scheletro canonicalizzato il punto medio `0.5 * (P[11] + P[12])` abbia norma $< 10^{-6}$.
2. `test_unit_torso_scale()`: verifica che la distanza tra `0.5 * (P[5] + P[6])` e `0.5 * (P[11] + P[12])` dopo `canonicalize_frame_3d` sia $1.0 \pm 10^{-5}$.
3. `test_yaw_alignment_along_positive_x()`: verifica che il vettore `P[12] - P[11]` dopo `canonicalize_frame_3d` abbia componente `Y == 0.0` ($\pm 10^{-5}$) e componente `X > 0`.
4. `test_exact_invariance_to_translation_scale_and_z_rotation()`: dato uno scheletro 3D arbitrario $P$, applica 10 trasformazioni casuali $P_{\text{trans}} = \lambda (P \cdot R_z(\phi)^\top) + \mathbf{t}_0$ con $\lambda \in [0.2, 50.0]$, $\phi \in [-\pi, \pi]$, $\mathbf{t}_0 \in [-100, 100]^3$ e verifica `np.allclose(canonicalize_frame_3d(P)[0], canonicalize_frame_3d(P_trans)[0], atol=1e-5)`.
5. `test_rotating_spin_window_preservation()`: simula una finestra di 9 frame in cui una posa fissa ruota di $40^\circ$ ad ogni frame (un giro completo di $360^\circ$ durante una piroetta!): verifica che `canonicalize_window_3d` ricostruisca esattamente la posa originale senza cancellazione vettoriale (`window_jitter < 1e-5`), e verifica che uno scheletro tutto a zero sollevi `DegenerateSkeletonError`.

---

## 4. Criteri di Accettazione del Settore 03 (Definition of Done)

- [ ] I file `src/data/segmentation.py`, `src/data/preprocessing.py` e `tests/test_normalization.py` sono stati creati.
- [ ] Il comando `pytest tests/test_normalization.py` passa al 100% (5/5 test verdi).
