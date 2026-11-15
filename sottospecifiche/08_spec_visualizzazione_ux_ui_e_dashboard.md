# 08 — Sottospecifica di Settore: Visualizzazione Finale, UX/UI e Dashboard Interattiva

**Settore:** 08 — Visualizzazione Finale: Plotting 2D/3D, Esportazione Figure Report e Dashboard Interattiva Streamlit  
**Riferimenti Specifica Madre:** §9, §10, §11, §12, §14  
**Dipendenze in ingresso:** Settore 04 (`Data`), Settore 05 (`rotate_kinematic_subtree`), Settore 06 (`SiameseGATv2`), Settore 07 (`predict_single_sample`, `evaluate_all_methods`)  
**Output abilitati:** Figure statiche in `reports/figures/` e applicazione interattiva `src/visualization/dashboard_app.py`

---

## 1. Obiettivo del Settore

Implementare:
1. I moduli di plotting 3D interattivo (Plotly) e 2D statico (Matplotlib) per gli scheletri e per le mappe di attenzione (`src/visualization/skeleton_plot.py`, `src/visualization/attention_plot.py`);
2. Lo script automatico di esportazione delle 6 figure vettoriali/raster per la relazione (`src/visualization/export_report_figures.py`);
3. La dashboard interattiva **Streamlit** a 4 tab (`src/visualization/dashboard_app.py`) con *Ghost Reference Overlay 3D* e *What-If Biomechanical Lab*.

---

## 2. Decomposizione in Sotto-Task Espliciti

### Task 8.1 — Plotting Scheletro 3D Interattivo e *Ghost Reference Overlay* (`src/visualization/skeleton_plot.py`)
- **File da creare:** `src/visualization/skeleton_plot.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any, Optional
  import numpy as np
  import plotly.graph_objects as go
  import matplotlib.pyplot as plt

  def create_interactive_3d_skeleton_figure(
      athlete_pos: np.ndarray,
      ref_pos: Optional[np.ndarray],
      bone_weights: np.ndarray,
      graph_cfg: Dict[str, Any],
      title: str = "3D Skeleton Inspection",
      show_ghost_reference: bool = True,
      cmin: float = 0.0,
      cmax: float = 0.15
  ) -> go.Figure:
      """
      Costruisce una figura Plotly 3D interattiva con:
      - Se show_ghost_reference e ref_pos is not None: disegna lo scheletro di riferimento in ciano (#00B4D8),
        opacità 0.35, linea tratteggiata ('dash').
      - Disegna le 16 ossa di athlete_pos colorandole secondo bone_weights (shape (16,)) sulla scala [cmin, cmax]
        usando una colormap sequenziale ('YlOrRd') e spessore di linea proporzionale al peso.
      - Imposta OBBLIGATORIAMENTE scene.aspectmode = 'cube' e range [-1.6, 1.6] su X, Y, Z.
      """

  def plot_static_canonicalization_stages(
      raw_frame: np.ndarray,
      centered_frame: np.ndarray,
      scaled_frame: np.ndarray,
      aligned_frame: np.ndarray,
      graph_cfg: Dict[str, Any]
  ) -> plt.Figure:
      """
      Genera una figura Matplotlib a 4 pannelli affiancati (Raw -> Root-Centered -> Scale-Normalized -> Yaw-Aligned)
      proiettati sul piano X-Z e X-Y.
      """
  ```
- **CONSTRAINT INDEROGABILE PER PLOTLY 3D:**
  In `create_interactive_3d_skeleton_figure`, configurare sempre il layout con:
  ```python
  fig.update_layout(
      title=title,
      scene=dict(
          xaxis=dict(range=[-1.6, 1.6], title="X (Canonical)"),
          yaxis=dict(range=[-1.6, 1.6], title="Y (Canonical)"),
          zaxis=dict(range=[-1.6, 1.6], title="Z (Vertical)"),
          aspectmode="cube"
      ),
      margin=dict(l=10, r=10, b=10, t=45)
  )
  ```

---

### Task 8.2 — Plotting Explainability (XAI): Attenzione vs Discrepanza Angolare (`src/visualization/attention_plot.py`)
- **File da creare:** `src/visualization/attention_plot.py`
- **Contratti esatti delle funzioni:**
  ```python
  from typing import Dict, Any
  import numpy as np
  import matplotlib.pyplot as plt

  def plot_attention_vs_angle_bars(
      bone_weights_norm: np.ndarray,
      bone_angle_diff_deg: np.ndarray,
      graph_cfg: Dict[str, Any],
      title: str = "GATv2 Attention vs Biomechanical Angular Deviation"
  ) -> plt.Figure:
      """
      Costruisce una figura Matplotlib a 2 subplot affiancati con barre orizzontali ordinate per
      peso di attenzione decrescente sulle 16 ossa:
      - Pannello SX: Peso di attenzione normalizzato w_b (in %)
      - Pannello DX: Differenza angolare assoluta |Delta theta| in gradi rispetto al medoide di riferimento.
      """

  def plot_multihead_attention_heatmap(
      head_matrix: np.ndarray,
      graph_cfg: Dict[str, Any],
      title: str = "Multi-Head GATv2 Attention per Bone (16 Bones x 4 Heads)"
  ) -> plt.Figure:
      """
      Disegna una heatmap 16x4 dei pesi di attenzione delle 4 head sulle 16 connessioni ossee.
      """

  def plot_skeleton_attention_comparison_2d(
      ref_pos: np.ndarray,
      good_pos: np.ndarray,
      bad_pos: np.ndarray,
      good_weights: np.ndarray,
      bad_weights: np.ndarray,
      graph_cfg: Dict[str, Any],
      meta_info: Dict[str, Any]
  ) -> plt.Figure:
      """
      Tavola a 3 pannelli (1: Medoide Riferimento, 2: Esecuzione High-GOE, 3: Esecuzione Low-GOE)
      con gli archi dello scheletro colorati sulla stessa scala cromatica [vmin, vmax].
      """
  ```

---

### Task 8.3 — Esportazione Automatica delle 6 Figure Vettoriali per la Relazione (`src/visualization/export_report_figures.py`)
- **File da creare:** `src/visualization/export_report_figures.py`
- **Contratto della funzione:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List

  def export_all_report_figures(
      output_dir: Path = Path("reports/figures"),
      dpi: int = 300
  ) -> List[Path]:
      """
      Genera e salva in output_dir sia in formato .png (300 dpi) sia .pdf le 6 figure ufficiali:
      1. fig01_canonicalization_stages.{png,pdf}
      2. fig02_training_curves.{png,pdf}
      3. fig03_distance_vs_goe_correlation.{png,pdf}
      4. fig04_baseline_comparison.{png,pdf}
      5. fig05_xai_attention_comparison.{png,pdf}
      6. fig06_whatif_sensitivity_curve.{png,pdf}
      Restituisce la lista dei Path dei file creati.
      """
  ```

---

### Task 8.4 — Dashboard Interattiva Streamlit (`src/visualization/dashboard_app.py`)
- **File da creare:** `src/visualization/dashboard_app.py`
- **COME strutturare `dashboard_app.py`:**
  1. Configurare la pagina con `st.set_page_config(page_title="Figure Skating AQA — Siamese GATv2", layout="wide")`.
  2. Caricare configurazioni, checkpoint, grafi e riferimenti tramite una funzione decorata con `@st.cache_resource` (se i file `.pt` non esistono ancora su disco, inizializzare automaticamente una sessione demo con `generate_mock_finefs_samples` così la dashboard non va mai in crash all'avvio!).
  3. **Sidebar laterale:**
     - Selettore `element_label` (`"Camel Spin"`, `"Sit Spin"`, `"Layback Spin"`);
     - Selettore `sample_id` tra i campioni del `test` set della classe selezionata;
     - Toggle `show_ghost_reference = st.sidebar.checkbox("Mostra Ghost Reference Overlay 3D", value=True)`;
     - Selettore layer attenzione `layer_name = st.sidebar.radio("Layer Attenzione GATv2", ["layer2", "layer1"])`.
  4. **Banner Superiore a 4 Colonne (`st.columns(4)`):**
     - `st.metric("Elemento / Split", f"{element_label} [TEST]")`
     - `st.metric("Distanza Latente d(A, R)", f"{pred['distance_to_ref']:.4f}")`
     - `st.metric("Indice Similarità Q (0-100)", f"{pred['similarity_index_Q']:.1f} / 100")`
     - `st.metric("GOE Giudice ISU (Ground Truth)", f"{pred['true_goe']:+.2f}")`
  5. **4 Tab Principali (`st.tabs([...])`):**
     - **Tab 1 (`"1. 3D Execution Inspector"`):** `st.plotly_chart(fig_3d, use_container_width=True)` + expander con `plot_static_canonicalization_stages`.
     - **Tab 2 (`"2. Explainability (XAI)"`):** Box informativo di cautela metodologica (`st.info(...)`) + `plot_attention_vs_angle_bars` + `plot_multihead_attention_heatmap`.
     - **Tab 3 (`"3. What-If Biomechanical Lab"`):** 3 slider angolari da `-45.0` a `+45.0` gradi (es. `"Anca SX (11->13)"`, `"Ginocchio SX (13->15)"`, `"Spalla SX (5->7)"`) che invocano al volo `rotate_kinematic_subtree` sul medoide di riferimento, costruiscono il grafo con `build_pyg_graph`, chiamano `predict_single_sample` e mostrano la variazione live di $d$, $Q$ e della mappa di attenzione 3D.
     - **Tab 4 (`"4. Benchmark & Latent Space"`):** Bar chart `Siamese GATv2` vs `3 Baseline Geometriche` + scatter plot Distanza vs GOE con Spearman $\rho$ + scatter 2D PCA degli embedding del test set.

---

## 3. Unit Test da Implementare (`tests/test_visualization.py`)

Il file `tests/test_visualization.py` deve implementare i seguenti 3 test automatici (usando `matplotlib.use("Agg")` per non aprire finestre GUI durante i test):
1. `test_interactive_3d_skeleton_figure_aspectmode_cube()`: chiama `create_interactive_3d_skeleton_figure` e verifica che `fig.layout.scene.aspectmode == "cube"` e che `list(fig.layout.scene.zaxis.range) == [-1.6, 1.6]`.
2. `test_xai_matplotlib_figures_creation()`: verifica che `plot_attention_vs_angle_bars` e `plot_multihead_attention_heatmap` restituiscano oggetti `matplotlib.figure.Figure` validi senza sollevare eccezioni, chiudendoli poi con `plt.close(fig)`.
3. `test_export_all_report_figures(tmp_path)`: esegue `export_all_report_figures(output_dir=tmp_path, dpi=100)` e verifica che vengano creati tutti i 12 file attesi (`6 .png` + `6 .pdf`) con dimensione su disco $> 0$ byte.

---

## 4. Criteri di Accettazione del Settore 08 (Definition of Done)

- [ ] I file `src/visualization/skeleton_plot.py`, `src/visualization/attention_plot.py`, `src/visualization/export_report_figures.py`, `src/visualization/dashboard_app.py` e `tests/test_visualization.py` sono stati creati.
- [ ] Il comando `pytest tests/test_visualization.py` passa al 100% (3/3 test verdi).
