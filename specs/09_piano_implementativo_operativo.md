# 09 — Operational Implementation Plan (Step-by-Step Guide for the Coding Model)

**Project:** Siamese GATv2 for Action Quality Assessment (AQA) in Figure Skating  
**Index and API Contracts Reference:** [00_indice_architettura_e_dipendenze.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/00_indice_architettura_e_dipendenze.md)

---

## 1. Execution Rules for the Coding Model

1. **Proceed strictly in Phase order (1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5):** Do not write files for Phase $k+1$ until the `pytest` command for Phase $k$ returns `0 errors, 100% passed`.
2. **Do not modify function signatures or file names:** Every module must strictly adhere to the *Cross-Module Symbol Table* defined in [00_indice_architettura_e_dipendenze.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/00_indice_architettura_e_dipendenze.md#L33-L58).
3. **Guaranteed Operation with both Real FineFS and Mock/Test Mode:** If the `.npz` files downloaded by the user are not yet present in `data/raw/finefs/`, the pipeline and training/evaluation/dashboard scripts must be able to run automatically using `generate_mock_finefs_samples()` from Sector 02, ensuring that every component of the repository is immediately executable and verifiable.

---

## 2. Implementation Sequence and Verification Commands

### PHASE 1 — Infrastructure, YAML Configs, 3D Canonicalization, and PyG Graph Construction
**Documents to follow:** [Sector 01](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/01_spec_ambiente_hardware_e_configurazione.md), [Sector 03](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/03_spec_preprocessing_temporale_e_canonicalizzazione_3d.md), [Sector 04](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/04_spec_costruzione_grafo_e_feature_engineering.md)

1. Create the directories and `__init__.py` files as specified in [Sector 01](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/01_spec_ambiente_hardware_e_configurazione.md).
2. Write `configs/data.yaml`, `configs/graph.yaml`, and `configs/training.yaml` verbatim.
3. Implement `src/utils/seed.py`, `src/utils/hardware.py`, `src/utils/config.py`, and `src/utils/env_check.py`.
4. Implement `src/data/segmentation.py` (`extract_stable_window`) and `src/data/preprocessing.py` (`DegenerateSkeletonError`, `canonicalize_frame_3d`, `canonicalize_window_3d`).
5. Implement `src/graph/topology.py`, `src/graph/features.py`, and `src/graph/builder.py`.
6. Write the tests `tests/test_normalization.py` and `tests/test_graph.py`.
- **Verification command (Gate 1):**
  ```powershell
  python -m src.utils.env_check
  pytest tests/test_normalization.py tests/test_graph.py -v
  ```

---

### PHASE 2 — Unified Data Schema, FineFS/Mock Loader, and Anti-Leakage Group Split
**Document to follow:** [Sector 02](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/02_spec_dati_esplorazione_armonizzazione_split.md)

1. Implement `src/data/harmonization.py` (`UnifiedSample`, `save_unified_samples`, `load_unified_samples`).
2. Implement `src/data/loaders.py` (`inspect_npz_file`, `audit_finefs_directory`, `extract_3d_keypoints_from_array`, `generate_mock_finefs_samples`, `create_grouped_splits`).
3. Write the tests `tests/test_dataset.py`.
- **Verification command (Gate 2):**
  ```powershell
  pytest tests/test_dataset.py -v
  ```

---

### PHASE 3 — Rodrigues Rotation, Triplet Sampler, Adaptive Loss, and Siamese GATv2 Model
**Documents to follow:** [Sector 05](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/05_spec_triplet_sampling_loss_e_augmentation.md), [Sector 06](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/06_spec_architettura_siamese_gatv2_e_training.md)

1. Implement `src/training/augmentation.py` (`rodrigues_rotation_matrix`, `rotate_kinematic_subtree`, `generate_synthetic_negatives`).
2. Implement `src/training/triplet_sampler.py` (`partition_by_goe_tiers`, `TripletDataset`).
3. Implement `src/training/losses.py` (`AdaptiveTripletMarginLoss`).
4. Implement `src/models/encoder.py` (`GATv2Encoder`) and `src/models/siamese_gat.py` (`SiameseGATv2`).
5. Implement `src/training/train.py` (`train_one_epoch`, `evaluate_triplets`, `run_training_pipeline`).
6. Write the tests `tests/test_triplets.py` (including single-batch overfit and weight-sharing tests).
- **Verification command (Gate 3):**
  ```powershell
  pytest tests/test_triplets.py -v
  ```

---

### PHASE 4 — Non-Neural Geometric Baselines, AQA Metrics, $Q$ Calibration, and Inference
**Document to follow:** [Sector 07](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/07_spec_valutazione_baseline_explainability_e_demo.md)

1. Implement `src/evaluation/baselines.py` (`coord_euclidean_distance`, `angle_euclidean_distance`, `feature_cosine_distance`).
2. Implement `src/evaluation/metrics.py` (`compute_class_references`, `calibrate_similarity_beta`, `distance_to_similarity_index`, `extract_bone_attention_weights`).
3. Implement `src/evaluation/evaluate.py` (`evaluate_all_methods`).
4. Implement `src/inference/predict.py` (`predict_single_sample`).
- **Verification command (Gate 4):**
  Run a complete training + evaluation run (on FineFS if present in `data/raw/finefs`, or on the mock dataset), verifying the creation of `checkpoints/best_model.pt` and `reports/evaluation_results.json`.

---

### PHASE 5 — 3D Visualization, Automated Report Figures, and Streamlit Dashboard
**Document to follow:** [Sector 08](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/08_spec_visualizzazione_ux_ui_e_dashboard.md)

1. Implement `src/visualization/skeleton_plot.py` (`create_interactive_3d_skeleton_figure`, `plot_static_canonicalization_stages`).
2. Implement `src/visualization/attention_plot.py` (`plot_attention_vs_angle_bars`, `plot_multihead_attention_heatmap`, `plot_skeleton_attention_comparison_2d`).
3. Implement `src/visualization/export_report_figures.py` (`export_all_report_figures`).
4. Implement `src/visualization/dashboard_app.py` (4-Tab Streamlit Dashboard with *3D Ghost Reference Overlay* and *What-If Lab*).
5. Write the tests `tests/test_visualization.py`.
- **Final verification command (Gate 5 — MVP Completion):**
  ```powershell
  pytest tests/ -v
  python -m src.visualization.export_report_figures
  ```
