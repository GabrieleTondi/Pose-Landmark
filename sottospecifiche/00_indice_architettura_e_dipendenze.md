# 00 — Indice Master delle Sottospecifiche, Contratti API e Dipendenze

**Progetto:** Siamese GATv2 per la valutazione della qualità degli elementi nel pattinaggio di figura (AQA)  
**Specifica Madre:** [specifica_implementativa_gnn_pattinaggio.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/specifica_implementativa_gnn_pattinaggio.md)  
**Revisione Critica:** [Revisione critica progetto GNN sportivo.pdf](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/Revisione%20critica%20progetto%20GNN%20sportivo.pdf)

---

## 1. Istruzioni per l'Agente / Modello Esecutore (Coding Guidelines)

Questa suite di specifiche è progettata per guidare passo-passo l'implementazione senza ambiguità. Il modello incaricato di scrivere il codice deve attenersi alle seguenti **regole di ingegneria del software**:

1. **Rispetto Tassativo delle Firme (API Contracts):** Non cambiare i nomi dei file, delle classi, delle funzioni, degli argomenti o delle chiavi YAML definiti in queste sottospecifiche. I moduli successivi importano esattamente i simboli specificati nella Tabella 3 qui sotto.
2. **Validazione Immediata tramite Test:** Al termine di ogni singolo settore, eseguire il comando `pytest` indicato nella sezione *Definition of Done* del settore stesso. Non passare mai al settore successivo se un test fallisce.
3. **Gestione dei Percorsi su Windows/Linux:** Utilizzare sempre `pathlib.Path` per costruire i percorsi dei file, evitando concatenazioni manuali di stringhe con `/` o `\`.
4. **Tipizzazione e Docstring:** Includere sempre i type hint Python (`from typing import Optional, Dict, List, Tuple, Any`) e verificare le shape dei tensori con istruzioni `assert` esplicite ai confini di ogni funzione.

---

## 2. Catalogo delle Sottospecifiche per Settore

| ID | File di Sottospecifica | Settore Tecnico | Sezioni Specifica Madre | Giorni Roadmap |
| :- | :--- | :--- | :--- | :-: |
| **01** | [01_spec_ambiente_hardware_e_configurazione.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/01_spec_ambiente_hardware_e_configurazione.md) | **Infrastruttura, Hardware Profiling & Schemi YAML Esatti** | §2, §13.1, §14, §15 | Giorno 1 |
| **02** | [02_spec_dati_esplorazione_armonizzazione_split.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/02_spec_dati_esplorazione_armonizzazione_split.md) | **Data Engineering: FineFS, MMFS, `UnifiedSample` & Group Split** | §3.1–§3.5, §16.1 | Giorni 1–3 |
| **03** | [03_spec_preprocessing_temporale_e_canonicalizzazione_3d.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/03_spec_preprocessing_temporale_e_canonicalizzazione_3d.md) | **Geometria 3D: Windowing Temporale & Canonicalizzazione** | §4, §5.1–§5.5, §16.2 | Giorno 4 |
| **04** | [04_spec_costruzione_grafo_e_feature_engineering.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/04_spec_costruzione_grafo_e_feature_engineering.md) | **Graph Representation: Topologia 17 Nodi, `x`, `edge_attr` & PyG** | §6.1–§6.4, §16.3 | Giorno 5 |
| **05** | [05_spec_triplet_sampling_loss_e_augmentation.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/05_spec_triplet_sampling_loss_e_augmentation.md) | **Metric Learning Pipeline: Triplet Sampler, Loss Adattiva & Rodrigues** | §8.1–§8.6 | Giorno 6 |
| **06** | [06_spec_architettura_siamese_gatv2_e_training.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/06_spec_architettura_siamese_gatv2_e_training.md) | **Deep Learning: Architettura `SiameseGATv2`, Regolarizzazione & Training** | §7.1–§7.5, §16.4 | Giorni 7–10 |
| **07** | [07_spec_valutazione_baseline_explainability_e_demo.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/07_spec_valutazione_baseline_explainability_e_demo.md) | **Valutazione, Baseline Geometriche, Calibrazione $Q$, Explainability & Demo** | §9, §10, §11, §12, §16.5 | Giorni 11–13 |
| **08** | [08_spec_visualizzazione_ux_ui_e_dashboard.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/08_spec_visualizzazione_ux_ui_e_dashboard.md) | **Visualizzazione Finale, UX/UI, Dashboard 3D, What-If Lab & Report Grafico** | §9, §10, §11, §12, §14 | Giorni 12–14 |
| **09** | [09_piano_implementativo_operativo.md](file:///c:/MAGISTRALE/DeepLearningBigData/Pose%20Landmark/sottospecifiche/09_piano_implementativo_operativo.md) | **Piano Implementativo Operativo ("Vertical Slice & Fail-Fast")** | §17, §18 | Giorni 1–14 |

---

## 3. Registro Centrale dei Contratti API tra Moduli (Cross-Module Symbol Table)

Ogni modulo deve esportare esattamente i seguenti simboli pubblici affinché gli import tra settori diversi funzionino senza errori:

| Modulo Python | Simboli Pubblici Esportati (Classi / Funzioni) | Settore |
| :--- | :--- | :-: |
| `src/utils/seed.py` | `seed_everything(seed: int = 42) -> None` | 01 |
| `src/utils/hardware.py` | `get_compute_device(prefer_gpu: bool = True) -> torch.device`<br>`get_hardware_profile() -> Dict[str, Any]` | 01 |
| `src/utils/config.py` | `load_yaml_config(path: str \| Path) -> Dict[str, Any]`<br>`load_all_configs(config_dir: str \| Path = "configs") -> Dict[str, Dict[str, Any]]` | 01 |
| `src/data/harmonization.py` | `UnifiedSample` (dataclass)<br>`save_unified_samples(samples: List[UnifiedSample], path: Path) -> None`<br>`load_unified_samples(path: Path) -> List[UnifiedSample]` | 02 |
| `src/data/loaders.py` | `audit_finefs_directory(raw_dir: Path) -> Dict[str, Any]`<br>`create_grouped_splits(samples: List[UnifiedSample], data_cfg: Dict) -> Dict[str, List[str]]` | 02 |
| `src/data/segmentation.py` | `extract_stable_window(keypoints: np.ndarray, confidence: Optional[np.ndarray], data_cfg: Dict) -> Tuple[Optional[np.ndarray], Dict[str, Any]]` | 03 |
| `src/data/preprocessing.py` | `DegenerateSkeletonError` (Exception)<br>`canonicalize_frame_3d(joints: np.ndarray, data_cfg: Dict) -> Tuple[np.ndarray, Dict[str, float]]`<br>`canonicalize_window_3d(window_joints: np.ndarray, data_cfg: Dict) -> Tuple[np.ndarray, Dict[str, Any]]` | 03 |
| `src/graph/topology.py` | `build_canonical_edge_index(graph_cfg: Dict) -> torch.Tensor` *(shape `(2, 32)`, `torch.long`)*<br>`get_directed_edge_list(graph_cfg: Dict) -> List[Tuple[int, int]]` | 04 |
| `src/graph/features.py` | `compute_node_features(P_bar: np.ndarray, graph_cfg: Dict) -> torch.Tensor` *(shape `(17, 3)` o `(17, 5)`)*<br>`compute_edge_features(P_bar: np.ndarray, graph_cfg: Dict) -> torch.Tensor` *(shape `(32, 5)`)* | 04 |
| `src/graph/builder.py` | `build_pyg_graph(P_bar: np.ndarray, sample_meta: Dict[str, Any], graph_cfg: Dict) -> Data` | 04 |
| `src/training/augmentation.py` | `rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray`<br>`rotate_kinematic_subtree(P_bar: np.ndarray, joint_idx: int, child_idx: int, axis: np.ndarray, angle_deg: float, graph_cfg: Dict) -> np.ndarray`<br>`generate_synthetic_negatives(train_graphs: List[Data], data_cfg: Dict, graph_cfg: Dict, train_cfg: Dict) -> List[Data]` | 05 |
| `src/training/triplet_sampler.py` | `partition_by_goe_tiers(graphs: List[Data], train_cfg: Dict, fit_thresholds: Optional[Dict] = None) -> Tuple[Dict, Dict]`<br>`TripletDataset(torch.utils.data.Dataset)` | 05 |
| `src/training/losses.py` | `AdaptiveTripletMarginLoss(nn.Module)` | 05 |
| `src/models/encoder.py` | `GATv2Encoder(nn.Module)` | 06 |
| `src/models/siamese_gat.py` | `SiameseGATv2(nn.Module)` | 06 |
| `src/training/train.py` | `train_one_epoch(...)`, `evaluate_triplets(...)`, `run_training_pipeline(...)` | 06 |
| `src/evaluation/baselines.py` | `coord_euclidean_distance(g1: Data, g2: Data) -> float`<br>`angle_euclidean_distance(g1: Data, g2: Data) -> float`<br>`feature_cosine_distance(g1: Data, g2: Data) -> float` | 07 |
| `src/evaluation/metrics.py` | `compute_class_references(train_graphs: List[Data], model: SiameseGATv2, train_cfg: Dict, device: torch.device) -> Dict[str, Dict[str, Any]]`<br>`calibrate_similarity_beta(val_distances: np.ndarray) -> float`<br>`distance_to_similarity_index(d: float \| np.ndarray, beta: float) -> float \| np.ndarray`<br>`extract_bone_attention_weights(model: SiameseGATv2, graph: Data, layer_name: str, device: torch.device) -> np.ndarray` *(shape `(16,)`)* | 07 |

---

## 4. Grafo delle Dipendenze e Gating Checkpoints

```mermaid
flowchart TD
    S01["Settore 01: Infrastruttura, GPU Profiling & Config YAML"]
    S02["Settore 02: Data Audit FineFS/MMFS, Schema Unificato & Group Split"]
    S03["Settore 03: Windowing Temporale & Canonicalizzazione 3D"]
    S04["Settore 04: Costruzione Grafo 17 Nodi, edge_attr & Serializzazione PyG"]
    S05["Settore 05: Triplet Sampler Intra-Classe, Loss & Augmentation"]
    S06["Settore 06: Modello Siamese GATv2 & Training Loop"]
    S07["Settore 07: Baseline Non-Neurali, Metriche, Calibrazione Q, XAI & Demo"]
    S08["Settore 08: UX/UI Dashboard Interattiva 3D, What-If Lab & Figure Report"]

    S01 -->|"Gate 1: PyG funzionante + Seed e YAML validati"| S02
    S02 -->|"Gate 2: Assi 3D/Joint verificati + Zero Leakage negli Split"| S03
    S03 -->|"Gate 3: Test invarianza roto-traslazione/scala superati"| S04
    S04 -->|"Gate 4: Grafi PyG (17 nodi, 32 archi, edge_attr finiti) su disco"| S05
    S04 -->|"Grafi serializzati pronti"| S07
    S05 -->|"Gate 5: Triplette intra-classe e augmentation solo su train"| S06
    S06 -->|"Gate 6: Best Checkpoint salvato senza collasso latente"| S07
    S07 -->|"Gate 7: Metriche, Prototipi z_R, Medoidi G_R, Beta e Attenzione pronti"| S08
```
