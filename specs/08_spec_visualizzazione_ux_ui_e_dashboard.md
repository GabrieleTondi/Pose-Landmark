# 08 — Sector Sub-Specification: Final Visualization, UX/UI, and Interactive Dashboard

**Sector:** 08 — Final Visualization: 2D/3D Plotting, Report Figure Export, and Interactive Streamlit Dashboard  
**Parent Specification References:** §9, §10, §11, §12, §14  
**Input Dependencies:** Sector 04 (`Data`), Sector 05 (`rotate_kinematic_subtree`), Sector 06 (`SiameseGATv2`), Sector 07 (`predict_single_sample`, `evaluate_all_methods`)  
**Enabled Outputs:** Static figures in `reports/figures/` and interactive application `src/visualization/dashboard_app.py`

---

## 1. Sector Objective

Implement:
1. The interactive 3D (Plotly) and static 2D (Matplotlib) plotting modules for skeletons and attention maps (`src/visualization/skeleton_plot.py`, `src/visualization/attention_plot.py`);
2. The automated export script for the 6 vector/raster report figures (`src/visualization/export_report_figures.py`);
3. The 4-tab interactive **Streamlit** dashboard (`src/visualization/dashboard_app.py`) with *3D Ghost Reference Overlay* and *What-If Biomechanical Lab*.

---

## 2. Decomposition into Explicit Sub-Tasks

### Task 8.1 — Interactive 3D Skeleton Plotting and *Ghost Reference Overlay* (`src/visualization/skeleton_plot.py`)
- **File to create:** `src/visualization/skeleton_plot.py`
- **Exact function contracts:**
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
      Builds an interactive Plotly 3D figure with:
      - If show_ghost_reference and ref_pos is not None: draws the reference skeleton in cyan (#00B4D8),
        opacity 0.35, dashed line ('dash').
      - Draws the 16 bones of athlete_pos colored according to bone_weights (shape (16,)) on the [cmin, cmax] scale
        using a sequential colormap ('YlOrRd') and line width proportional to the weight.
      - MANDATORILY sets scene.aspectmode = 'cube' and range [-1.6, 1.6] on X, Y, Z.
      """

  def plot_static_canonicalization_stages(
      raw_frame: np.ndarray,
      centered_frame: np.ndarray,
      scaled_frame: np.ndarray,
      aligned_frame: np.ndarray,
      graph_cfg: Dict[str, Any]
  ) -> plt.Figure:
      """
      Generates a 4-panel side-by-side Matplotlib figure (Raw -> Root-Centered -> Scale-Normalized -> Yaw-Aligned)
      projected onto the X-Z and X-Y planes.
      """
  ```
- **MANDATORY CONSTRAINT FOR PLOTLY 3D:**
  In `create_interactive_3d_skeleton_figure`, always configure the layout with:
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

### Task 8.2 — Explainability (XAI) Plotting: Attention vs Angular Discrepancy (`src/visualization/attention_plot.py`)
- **File to create:** `src/visualization/attention_plot.py`
- **Exact function contracts:**
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
      Builds a 2-subplot side-by-side Matplotlib figure with horizontal bars sorted by
      descending attention weight across the 16 bones:
      - Left panel: Normalized attention weight w_b (in %)
      - Right panel: Absolute angular difference |Delta theta| in degrees relative to the reference medoid.
      """

  def plot_multihead_attention_heatmap(
      head_matrix: np.ndarray,
      graph_cfg: Dict[str, Any],
      title: str = "Multi-Head GATv2 Attention per Bone (16 Bones x 4 Heads)"
  ) -> plt.Figure:
      """
      Draws a 16x4 heatmap of the attention weights of the 4 heads across the 16 bone connections.
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
      3-panel plate (1: Reference Medoid, 2: High-GOE Execution, 3: Low-GOE Execution)
      with skeleton edges colored on the same [vmin, vmax] color scale.
      """
  ```

---

### Task 8.3 — Automated Export of the 6 Vector Figures for the Report (`src/visualization/export_report_figures.py`)
- **File to create:** `src/visualization/export_report_figures.py`
- **Function contract:**
  ```python
  from pathlib import Path
  from typing import Dict, Any, List

  def export_all_report_figures(
      output_dir: Path = Path("reports/figures"),
      dpi: int = 300
  ) -> List[Path]:
      """
      Generates and saves in output_dir in both .png (300 dpi) and .pdf formats the 6 official figures:
      1. fig01_canonicalization_stages.{png,pdf}
      2. fig02_training_curves.{png,pdf}
      3. fig03_distance_vs_goe_correlation.{png,pdf}
      4. fig04_baseline_comparison.{png,pdf}
      5. fig05_xai_attention_comparison.{png,pdf}
      6. fig06_whatif_sensitivity_curve.{png,pdf}
      Returns the list of Paths of the created files.
      """
  ```

---

### Task 8.4 — Interactive Streamlit Dashboard (`src/visualization/dashboard_app.py`)
- **File to create:** `src/visualization/dashboard_app.py`
- **HOW to structure `dashboard_app.py`:**
  1. Configure the page with `st.set_page_config(page_title="Figure Skating AQA — Siamese GATv2", layout="wide")`.
  2. Load configurations, checkpoint, graphs, and references via a function decorated with `@st.cache_resource` (if the `.pt` files do not yet exist on disk, automatically initialize a demo session with `generate_mock_finefs_samples` so the dashboard never crashes on startup!).
  3. **Sidebar:**
     - `element_label` selector (`"Camel Spin"`, `"Sit Spin"`, `"Layback Spin"`);
     - `sample_id` selector among the `test` set samples of the selected class;
     - Toggle `show_ghost_reference = st.sidebar.checkbox("Show 3D Ghost Reference Overlay", value=True)`;
     - Attention layer selector `layer_name = st.sidebar.radio("GATv2 Attention Layer", ["layer2", "layer1"])`.
  4. **4-Column Top Banner (`st.columns(4)`):**
     - `st.metric("Element / Split", f"{element_label} [TEST]")`
     - `st.metric("Latent Distance d(A, R)", f"{pred['distance_to_ref']:.4f}")`
     - `st.metric("Similarity Index Q (0-100)", f"{pred['similarity_index_Q']:.1f} / 100")`
     - `st.metric("ISU Judge GOE (Ground Truth)", f"{pred['true_goe']:+.2f}")`
  5. **4 Main Tabs (`st.tabs([...])`):**
     - **Tab 1 (`"1. 3D Execution Inspector"`):** `st.plotly_chart(fig_3d, use_container_width=True)` + expander with `plot_static_canonicalization_stages`.
     - **Tab 2 (`"2. Explainability (XAI)"`):** Methodological caution info box (`st.info(...)`) + `plot_attention_vs_angle_bars` + `plot_multihead_attention_heatmap`.
     - **Tab 3 (`"3. What-If Biomechanical Lab"`):** 3 angular sliders from `-45.0` to `+45.0` degrees (e.g., `"Left Hip (11->13)"`, `"Left Knee (13->15)"`, `"Left Shoulder (5->7)"`) that invoke `rotate_kinematic_subtree` on the fly on the reference medoid, build the graph with `build_pyg_graph`, call `predict_single_sample`, and display the live variation of $d$, $Q$, and the 3D attention map.
     - **Tab 4 (`"4. Benchmark & Latent Space"`):** Bar chart `Siamese GATv2` vs `3 Geometric Baselines` + Distance vs GOE scatter plot with Spearman $\rho$ + 2D PCA scatter plot of test set embeddings.

---

## 3. Unit Tests to Implement (`tests/test_visualization.py`)

The file `tests/test_visualization.py` must implement the following 3 automated tests (using `matplotlib.use("Agg")` to avoid opening GUI windows during tests):
1. `test_interactive_3d_skeleton_figure_aspectmode_cube()`: calls `create_interactive_3d_skeleton_figure` and verifies that `fig.layout.scene.aspectmode == "cube"` and that `list(fig.layout.scene.zaxis.range) == [-1.6, 1.6]`.
2. `test_xai_matplotlib_figures_creation()`: verifies that `plot_attention_vs_angle_bars` and `plot_multihead_attention_heatmap` return valid `matplotlib.figure.Figure` objects without raising exceptions, then closes them with `plt.close(fig)`.
3. `test_export_all_report_figures(tmp_path)`: runs `export_all_report_figures(output_dir=tmp_path, dpi=100)` and verifies that all 12 expected files (`6 .png` + `6 .pdf`) are created with size on disk $> 0$ bytes.

---

## 4. Sector 08 Acceptance Criteria (Definition of Done)

- [ ] The files `src/visualization/skeleton_plot.py`, `src/visualization/attention_plot.py`, `src/visualization/export_report_figures.py`, `src/visualization/dashboard_app.py`, and `tests/test_visualization.py` have been created.
- [ ] The command `pytest tests/test_visualization.py` passes 100% (3/3 green tests).
