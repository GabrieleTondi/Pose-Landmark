1# 09 — Piano Implementativo Operativo (Guida Step-by-Step per il Coding Model)

**Progetto:** Siamese GATv2 per Action Quality Assessment (AQA) nel Pattinaggio di Figura  
**Riferimento Indice e Contratti API:** [00_indice_architettura_e_dipendenze.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/00_indice_architettura_e_dipendenze.md)

---

## 1. Regole Esecutive per il Modello di Coding

1. **Procedere rigorosamente in ordine di Fase (1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5):** Non scrivere i file della Fase $k+1$ finché il comando `pytest` della Fase $k$ non restituisce `0 errors, 100% passed`.
2. **Non modificare le firme delle funzioni o i nomi dei file:** Ogni modulo deve rispettare alla lettera la *Cross-Module Symbol Table* definita in [00_indice_architettura_e_dipendenze.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/00_indice_architettura_e_dipendenze.md#L33-L58).
3. **Funzionamento Garantito sia con FineFS Reale sia in Modalità Mock/Test:** Se nella cartella `data/raw/finefs/` non sono ancora presenti i file `.npz` scaricati dall'utente, la pipeline e gli script di training/valutazione/dashboard devono poter girare automaticamente usando `generate_mock_finefs_samples()` del Settore 02, così da garantire che ogni componente del repository sia immediatamente eseguibile e verificabile.

---

## 2. Sequenza di Implementazione e Comandi di Verifica

### FASE 1 — Infrastruttura, Config YAML, Canonicalizzazione 3D e Costruzione Grafo PyG
**Documenti da seguire:** [Settore 01](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/01_spec_ambiente_hardware_e_configurazione.md), [Settore 03](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/03_spec_preprocessing_temporale_e_canonicalizzazione_3d.md), [Settore 04](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/04_spec_costruzione_grafo_e_feature_engineering.md)

1. Creare le directory e i file `__init__.py` come specificato in [Settore 01](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/01_spec_ambiente_hardware_e_configurazione.md).
2. Scrivere letteralmente `configs/data.yaml`, `configs/graph.yaml` e `configs/training.yaml`.
3. Implementare `src/utils/seed.py`, `src/utils/hardware.py`, `src/utils/config.py` e `src/utils/env_check.py`.
4. Implementare `src/data/segmentation.py` (`extract_stable_window`) e `src/data/preprocessing.py` (`DegenerateSkeletonError`, `canonicalize_frame_3d`, `canonicalize_window_3d`).
5. Implementare `src/graph/topology.py`, `src/graph/features.py` e `src/graph/builder.py`.
6. Scrivere i test `tests/test_normalization.py` e `tests/test_graph.py`.
- **Comando di verifica (Gate 1):**
  ```powershell
  python -m src.utils.env_check
  pytest tests/test_normalization.py tests/test_graph.py -v
  ```

---

### FASE 2 — Schema Dati Unificato, Loader FineFS/Mock e Group Split Anti-Leakage
**Documento da seguire:** [Settore 02](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/02_spec_dati_esplorazione_armonizzazione_split.md)

1. Implementare `src/data/harmonization.py` (`UnifiedSample`, `save_unified_samples`, `load_unified_samples`).
2. Implementare `src/data/loaders.py` (`inspect_npz_file`, `audit_finefs_directory`, `extract_3d_keypoints_from_array`, `generate_mock_finefs_samples`, `create_grouped_splits`).
3. Scrivere i test `tests/test_dataset.py`.
- **Comando di verifica (Gate 2):**
  ```powershell
  pytest tests/test_dataset.py -v
  ```

---

### FASE 3 — Rotazione di Rodrigues, Triplet Sampler, Loss Adattiva e Modello Siamese GATv2
**Documenti da seguire:** [Settore 05](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/05_spec_triplet_sampling_loss_e_augmentation.md), [Settore 06](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/06_spec_architettura_siamese_gatv2_e_training.md)

1. Implementare `src/training/augmentation.py` (`rodrigues_rotation_matrix`, `rotate_kinematic_subtree`, `generate_synthetic_negatives`).
2. Implementare `src/training/triplet_sampler.py` (`partition_by_goe_tiers`, `TripletDataset`).
3. Implementare `src/training/losses.py` (`AdaptiveTripletMarginLoss`).
4. Implementare `src/models/encoder.py` (`GATv2Encoder`) e `src/models/siamese_gat.py` (`SiameseGATv2`).
5. Implementare `src/training/train.py` (`train_one_epoch`, `evaluate_triplets`, `run_training_pipeline`).
6. Scrivere i test `tests/test_triplets.py` (inclusi i test di overfit su singolo batch e condivisione pesi).
- **Comando di verifica (Gate 3):**
  ```powershell
  pytest tests/test_triplets.py -v
  ```

---

### FASE 4 — Baseline Geometriche Non-Neurali, Metriche AQA, Calibrazione $Q$ e Inferenza
**Documento da seguire:** [Settore 07](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/07_spec_valutazione_baseline_explainability_e_demo.md)

1. Implementare `src/evaluation/baselines.py` (`coord_euclidean_distance`, `angle_euclidean_distance`, `feature_cosine_distance`).
2. Implementare `src/evaluation/metrics.py` (`compute_class_references`, `calibrate_similarity_beta`, `distance_to_similarity_index`, `extract_bone_attention_weights`).
3. Implementare `src/evaluation/evaluate.py` (`evaluate_all_methods`).
4. Implementare `src/inference/predict.py` (`predict_single_sample`).
- **Comando di verifica (Gate 4):**
  Eseguire un run completo di training + valutazione (su FineFS se presente in `data/raw/finefs`, o sul dataset mock) verificando la creazione di `checkpoints/best_model.pt` e `reports/evaluation_results.json`.

---

### FASE 5 — Visualizzazione 3D, Figure Automatiche per la Relazione e Dashboard Streamlit
**Documento da seguire:** [Settore 08](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/08_spec_visualizzazione_ux_ui_e_dashboard.md)

1. Implementare `src/visualization/skeleton_plot.py` (`create_interactive_3d_skeleton_figure`, `plot_static_canonicalization_stages`).
2. Implementare `src/visualization/attention_plot.py` (`plot_attention_vs_angle_bars`, `plot_multihead_attention_heatmap`, `plot_skeleton_attention_comparison_2d`).
3. Implementare `src/visualization/export_report_figures.py` (`export_all_report_figures`).
4. Implementare `src/visualization/dashboard_app.py` (Dashboard Streamlit a 4 Tab con *Ghost Reference Overlay 3D* e *What-If Lab*).
5. Scrivere i test `tests/test_visualization.py`.
- **Comando di verifica finale (Gate 5 — Completamento MVP):**
  ```powershell
  pytest tests/ -v
  python -m src.visualization.export_report_figures
  ```
