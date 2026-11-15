# Diario Operativo di Progetto — Siamese GATv2 per Action Quality Assessment (Pattinaggio di Figura)

> **Scopo di questo documento:** Raccontare in modo dettagliato, accessibile e tecnicamente fedele **tutto ciò che è stato progettato, implementato, corretto e verificato** dall'inizio del progetto fino allo stato attuale.  
> Questo file viene mantenuto e aggiornato progressivamente al termine di ogni Fase prevista in [sottospecifiche/09_piano_implementativo_operativo.md](sottospecifiche/09_piano_implementativo_operativo.md).

---

## Stato Avanzamento Lavori (Roadmap)

| Fase | Contenuto Principale | Stato | Gate di Verifica |
| :--- | :--- | :---: | :--- |
| **Fase 1** | Infrastruttura, Config YAML, Segmentazione, Canonicalizzazione 3D e Costruzione Grafo PyG | ✅ Completata | `python -m src.utils.env_check` → `ENV_CHECK_OK`; test `test_config_and_utils.py`, `test_normalization.py`, `test_graph.py` passati |
| **Fase 2** | Schema `UnifiedSample`, Audit & Loader FineFS Reale (`H36M-17` + GOE ISU), Split Anti-Leakage e Materializzazione Grafi | ✅ Completata | `1583` trottole reali caricate e trasformate in grafi PyG (`0` scartate); `pytest tests/ -v` → **`22 passed`** |
| **Fase 3** | Augmentation con Rotazioni di Rodrigues, Triplet Sampler, Adaptive Margin Loss, Encoder `GATv2` e Training | ✅ Completata | `pytest tests/test_triplets.py -v` (`8 passed`); training CUDA reale su FineFS (`62` epoche, best epoch `47`, `val_triplet_acc = 0.65`, `embedding_std = 0.289`) + `pytest tests/ -v` → **`30 passed`** |
| **Fase 4** | Baseline Geometriche Non-Neurali, Metriche AQA, Calibrazione Indice $Q$, Valutazione e Inferenza | ⏳ Prossima | Valutazione su `test_graphs.pt` (con `checkpoints/best_model.pt` già generato) → `reports/evaluation_results.json` |
| **Fase 5** | Visualizzazione 3D/2D, Export Figure per Relazione e Dashboard Interattiva Streamlit | ⏳ Da fare | `pytest tests/ -v` + `python -m src.visualization.export_report_figures` |

---

## 0. Idea di Base del Progetto (Perché facciamo questi passaggi)

L'obiettivo del sistema è valutare la **qualità esecutiva (Action Quality Assessment, AQA)** delle trottole nel pattinaggio di figura artistico (in particolare **Camel Spin**, **Sit Spin** e **Layback Spin**) a partire dalle pose 3D dello scheletro umano (`17` articolazioni nello spazio).

Invece di limitarci a una "scatola nera" che spara un voto numerico su un singolo video grezzo:
1. Vogliamo rappresentare ogni trottola come una **posa 3D media pulita e allineata nello spazio** (indipendente da dove si trova il pattinatore sulla pista, dalla sua statura e da verso quale lato sta girando in quel fotogramma).
2. Trasformiamo quella posa in un **grafo anatomico** (`17` nodi = articolazioni, `32` archi diretti = ossa nei due versi) arricchito da direzioni 3D, lunghezze ossee e angoli articolari.
3. Nella seconda metà del progetto (Fasi 3–5), una rete neurale **Siamese GATv2** imparerà a confrontare una trottola eseguita da un atleta con trottole di riferimento eccellenti e con errori biomeccanici graduali, producendo sia un indice di qualità calibrato sia una spiegazione visiva di *quali arti e giunti* hanno causato eventuali penalità tramite i pesi di attenzione degli archi.

**Importante:** Al termine della **Fase 2**, **non esiste ancora nessuna rete neurale addestrata**. Abbiamo completato l'intera catena che porta dai file grezzi del dataset fino ai file di grafi PyTorch Geometric pronti per entrare nel modello.

---

## 1. Fase 1 — Infrastruttura, Pulizia Geometrica 3D e Costruzione del Grafo

### 1.1 Ambiente Python, GPU CUDA, Dipendenze e Gestione Repository
Abbiamo configurato l'ambiente sull'interprete Python 3.13.3 del sistema, installando le librerie fissate in [requirements.txt](requirements.txt):
- **Core numerico e dati:** `numpy==2.1.3`, `scipy==1.16.2`, `scikit-learn==1.7.2`, `pandas==3.1.1`, `PyYAML==6.0.2`.
- **Deep Learning & Grafi:** `torch==2.8.0+cu126`, `torch-geometric==2.8.1` (con supporto CUDA 12.6).
- **Test e Visualizzazione:** `pytest==9.2.0`, `matplotlib==3.11.2`, `plotly==7.1.0`, `streamlit==1.67.0`.

Abbiamo inoltre configurato:
- [pyproject.toml](pyproject.toml): imposta i metadati del pacchetto `figure-skating-aqa-gatv2`, aggiunge la root al `pythonpath` per `pytest` e usa `addopts = "-p no:cacheprovider"` (così i test possono girare puliti anche in ambienti Windows con permessi restrittivi sulla scrittura della cache).
- [.gitignore](.gitignore): esclude dal versioning file pesanti e binari (`.venv/`, `__pycache__/`, `*.pt`, `*.pth`, `data/raw/`, `data/intermediate/`, `data/processed/`, `checkpoints/*.pt`) mantenendo i file segnaposto `.gitkeep`.

#### Moduli Utility (`src/utils/`)
1. **[src/utils/seed.py](src/utils/seed.py) — `seed_everything(seed=42)`**  
   Fissa contemporaneamente `PYTHONHASHSEED`, `random`, `numpy.random`, `torch.manual_seed` (CPU e CUDA) e imposta `torch.backends.cudnn.deterministic = True` (con `benchmark = False`) per rendere riproducibili split, pesi e campionamenti.
2. **[src/utils/hardware.py](src/utils/hardware.py) — `get_compute_device`, `get_hardware_profile`, `get_vram_usage_mb`**  
   Rileva la GPU e **prima di fidarsi** esegue un micro-calcolo reale su GPU (`2x2 @ 2x2` su `cuda`), ricadendo automaticamente su CPU con warning solo se CUDA fallisce davvero.
3. **[src/utils/config.py](src/utils/config.py) — `load_yaml_config`, `load_all_configs`**  
   Carica i tre file in `configs/` (`data.yaml`, `graph.yaml`, `training.yaml`) e valida subito alcune invarianti critiche:
   - `graph.yaml` deve contenere esattamente `16` voci in `undirected_bones` e `32` voci in `directed_edge_angle_triplets`;
   - `data.yaml` deve avere `train_ratio + val_ratio + test_ratio = 1.0` (tolleranza `1e-6`).
4. **[src/utils/env_check.py](src/utils/env_check.py)**  
   Esegue uno *smoke test* completo: fissa il seed, carica i config, seleziona il device, salva [reports/hardware_profile.json](reports/hardware_profile.json) e testa un layer reale `GATv2Conv(in_channels=3, out_channels=32, heads=4, edge_dim=5, add_self_loops=True)` verificando che:
   - l'output sui `17` nodi abbia shape `(17, 128)` (`32 * 4` teste);
   - gli indici e i pesi di attenzione abbiano shape `(2, 49)` e `(49, 4)` (`32` archi reali + `17` self-loop aggiunti internamente da `GATv2Conv`);
   - il backpropagation (`out.sum().backward()`) produca gradienti non nulli rispetto all'input `x`.
   - **Risultato osservato sulla macchina locale** ([reports/hardware_profile.json](reports/hardware_profile.json)):
     - PyTorch `2.8.0+cu126`, CUDA `12.6` disponibile (`true`)
     - GPU rilevata: **`NVIDIA GeForce RTX 3050 Laptop GPU`** (`4095.5 MB` di VRAM)
     - `selected_compute_device`: `"cuda"`, output `ENV_CHECK_OK`.

---

### 1.2 I Tre File di Configurazione Centralizzati (`configs/`)
Tutta la matematica della pipeline non è "hardcodata" dentro gli script, ma letta da tre YAML:
- **[configs/data.yaml](configs/data.yaml):** definisce percorsi (`data/raw/finefs`, `data/intermediate`, `data/processed`, `data/splits`, `reports`, `checkpoints`), convenzione scheletro (`h36m17`), classi target (`Camel Spin`, `Sit Spin`, `Layback Spin`), parametri della finestra temporale (`window_size: 9`, `min_valid_frames: 5`, taglio iniziale/finale `0.20`, confidenza minima `0.20`), indici e assi per la canonicalizzazione 3D, e rapporti di split (`0.70 / 0.15 / 0.15`, `seed: 42`).
- **[configs/graph.yaml](configs/graph.yaml):** definisce i `17` nodi, la modalità delle feature dei nodi (`node_feature_mode: "baseline"` → `3` feature `[x, y, z]`; supporta anche `"extended"` → `5` feature), `edge_feature_dim: 5`, i nomi e la simmetria anatomica di ciascun giunto, le `16` ossa non orientate, le `32` terne `[a, b, c]` per gli angoli articolari e i `9` sotto-alberi cinematici per le future rotazioni di Rodrigues.
- **[configs/training.yaml](configs/training.yaml):** contiene i parametri già pronti per la Fase 3 (`GATv2` `hidden_dim: 32`, `embed_dim: 32`, `heads: 4`, `dropout: 0.20`, `batch_size: 32`, `lr: 1e-3`, `max_epochs: 80`, soglie GOE `>= 1.5` positivi e `<= 0.0` negativi, margini adattivi `0.20 / 0.60 / 1.00`, ecc.).

---

### 1.3 Taglio Temporale della Finestra Stabile (`src/data/segmentation.py`)
Una trottola annotata in gara può durare 10–25 secondi (centinaia di frame a 25 fps). Nei primi secondi il pattinatore è ancora in fase di ingresso e negli ultimi secondi sta già sciogliendo la posizione per uscire.

La funzione **`extract_stable_window(keypoints, confidence, data_cfg)`** in [src/data/segmentation.py](src/data/segmentation.py) fa questo:
1. Controlla che la sequenza in ingresso abbia shape `(T_seg, 17, 3)` e almeno `min_valid_frames` (`5`) frame totali; altrimenti scarta il segmento (`segment_too_short`).
2. Taglia via il primo `20%` (`trim_ratio_start: 0.20`) e l'ultimo `20%` (`trim_ratio_end: 0.20`) concentrandosi sul `60%` centrale. *(Se il segmento è così corto che dopo il taglio resterebbero meno di 5 frame, usa l'intero intervallo `[0, T_seg)` come fallback).*
3. Filtra frame per frame tenendo solo i frame validi dove:
   - tutte le coordinate sono finite (no `NaN`/`Inf`);
   - la deviazione standard del frame è `> 1e-5` (scarta scheletri collassati in un singolo punto);
   - se è presente l'array `confidence`, la confidenza media sui 17 giunti è `>= 0.20` (`min_keypoint_confidence`).
4. Se restano almeno `5` frame validi, prende la sotto-finestra **centrale** di lunghezza massima `window_size = 9` (`T_win <= 9`, corrispondente a circa 0.36 secondi a 25 fps: abbastanza breve da catturare una posa stabile, abbastanza lunga da smorzare il rumore di tracking frame-per-frame).

---

### 1.4 Canonicalizzazione Geometrica 3D (`src/data/preprocessing.py`)
Le coordinate grezze di un atleta non possono entrare direttamente nel grafo: dipendono dalla posizione in pista, dall'altezza della persona (o dalla scala della camera/ricostruzione) e dall'angolo di rotazione attorno all'asse verticale durante la trottola.

Per eliminare questi fattori di disturbo, **`canonicalize_frame_3d(joints, data_cfg)`** applica tre trasformazioni matematiche esatte a ogni singolo frame `(17, 3)`:
1. **Root-Centering (Invarianza alla traslazione):**  
   Calcola il centro del bacino `pelvis = 0.5 * (pts[hip_l] + pts[hip_r])` (in H36M-17: giunti `4` e `1`) e lo sottrae a tutti i 17 giunti. Il bacino diventa l'origine `(0, 0, 0)`.
2. **Scale Normalization (Invarianza alla statura / scala):**  
   Calcola il punto medio delle spalle `shoulders_mid = 0.5 * (centered[sh_l] + centered[sh_r])` (in H36M-17: giunti `11` e `14`) e misura la lunghezza 3D del busto `torso_length = ||shoulders_mid||_2`. Se è inferiore a `scale_epsilon` (`1e-4`), solleva `DegenerateSkeletonError`; altrimenti divide tutte le coordinate per `torso_length`. Da questo momento tutte le distanze sono espresse in **"unità di busto"**.
3. **Yaw Alignment (Invarianza alla rotazione orizzontale sul ghiaccio):**  
   Calcola il vettore orizzontale delle anche sul piano `(X, Y)` (`hip_vec = scaled[hip_r] - scaled[hip_l]`; se quasi nullo, cioè `<= yaw_epsilon`, usa come ripiego il vettore delle spalle `sh_vec`; se anche quello è nullo, solleva `DegenerateSkeletonError`). Calcola l'angolo `angle = arctan2(vy, vx)` e applica una rotazione piana di `-angle` sugli assi `[0, 1]` (mantenendo intatto l'asse verticale `2 = Z`). Così ogni frame della trottola viene ruotato sullo stesso sistema di riferimento locale con le anche allineate all'asse `+X`.

Successivamente **`canonicalize_window_3d(window_joints, data_cfg)`**:
- esegue `canonicalize_frame_3d` su ciascuno dei `T_win` frame della finestra (tollerando qualche singolo frame degenere purché restino almeno `min_valid_frames` frame validi);
- calcola la **posa canonica media** `P_bar = mean(aligned_frames, axis=0)` di shape `(17, 3)` in `float32`, insieme a statistiche diagnostiche (`num_valid_frames`, `mean_torso_length`, `window_jitter` = dispersione RMS tra i frame allineati e `P_bar`).

---

### 1.5 Costruzione del Grafo PyTorch Geometric (`src/graph/`)
Una volta ottenuta `P_bar` `(17, 3)`, il pacchetto `src/graph/` la trasforma in un oggetto `torch_geometric.data.Data`:

1. **Topologia ordinata a coppie — [src/graph/topology.py](src/graph/topology.py)**  
   - `get_directed_edge_list()` legge le `16` ossa in `configs/graph.yaml` e per ogni osso `b = (i, j)` crea in ordine deterministico l'arco `2*b: (i -> j)` e subito dopo l'arco inverso `2*b + 1: (j -> i)`, ottenendo `32` archi diretti.
   - `build_canonical_edge_index()` costruisce il tensore `edge_index` di shape `(2, 32)` (`torch.long`). Non inserisce self-loop manuali perché vengono gestiti internamente da `GATv2Conv(..., add_self_loops=True)`.
   - Questo ordinamento a coppie `2*b` e `2*b+1` servirà in Fase 4–5 per aggregare facilmente l'attenzione di andata e ritorno su ciascuno dei 16 segmenti anatomici.

2. **Feature dei Nodi e degli Archi — [src/graph/features.py](src/graph/features.py)**  
   - `compute_node_features(P_bar, graph_cfg)`:
     - in modalità `"baseline"` (attiva ora): restituisce `(17, 3)` `float32` con `[x, y, z]` canonici;
     - in modalità `"extended"`: restituisce `(17, 5)` aggiungendo il raggio dal bacino `||p_i||_2` e l'indicatore di simmetria `joint_symmetry` (`-1` sinistra, `0` centro, `+1` destra).
   - `compute_edge_features(P_bar, graph_cfg)`: restituisce un tensore `(32, 5)` `float32`. Per ogni arco `(i -> j)`:
     - colonne `0..2`: versore direzionale unitario `u_ij = (p_j - p_i) / (||p_j - p_i||_2 + eps)`;
     - colonna `3`: lunghezza dell'osso `l_ij = ||p_j - p_i||_2` (in unità di busto);
     - colonna `4`: coseno dell'angolo articolare $\cos(\theta_{abc})$ calcolato sulla terna `[a, b, c]` definita per quell'arco in `directed_edge_angle_triplets` (usando `v1 = p_a - p_b`, `v2 = p_c - p_b`, bloccato con `np.clip(..., -1.0, 1.0)`).

3. **Assemblaggio e Serializzazione — [src/graph/builder.py](src/graph/builder.py)**  
   - `build_pyg_graph(P_bar, sample_meta, graph_cfg)`: costruisce l'oggetto `Data(x, pos, edge_index, edge_attr, y, ...)` preservando sempre `data.pos` `(17, 3)` (anche qualora `x` fosse in modalità `extended` `(17, 5)`), il target `y = [goe]`, e tutti i metadati (`sample_id`, `dataset_id`, `performance_id`, `athlete_id`, `element_label`, `goe`, `split`, `is_augmented`, `parent_sample_id`, `severity_tier`, `perturbation_joint`, `perturbation_angle_deg`).
   - `process_and_save_dataset_graphs(samples, split_dict, data_cfg, graph_cfg)`: esegue l'intera catena `extract_stable_window → canonicalize_window_3d → build_pyg_graph` su tutti i campioni suddivisi in `train`, `val`, `test` e salva `train_graphs.pt`, `val_graphs.pt` e `test_graphs.pt` in `data/processed/`.

---

## 2. Fase 2 — Armonizzazione, Audit di FineFS Reale, Loader e Split Anti-Leakage

### 2.1 Schema Dati Unificato (`src/data/harmonization.py`)
In [src/data/harmonization.py](src/data/harmonization.py) abbiamo definito la dataclass **`UnifiedSample`**, che rappresenta in modo indipendente dalla sorgente (FineFS reale o generatore sintetico mock) un singolo elemento tecnico estratto:
- identificativi: `sample_id`, `dataset_id`, `performance_id`, `athlete_id` (`Optional[str]`), `element_label`;
- finestra nella clip sorgente: `start_frame`, `end_frame` (con `end_frame >= start_frame`);
- dati 3D: `keypoints` `(T_seg, 17, 3)` `float32`, `keypoint_confidence` `(T_seg, 17)` `float32` oppure `None`;
- etichetta qualità e tracciabilità: `goe` (`float` finito), `source_metadata` (`dict`).

Il metodo `validate()` controlla automaticamente tutte le shape, i tipi e la finitezza prima di salvare (`save_unified_samples`) o dopo aver ricaricato (`load_unified_samples`) il file pickle.

### 2.2 Acquisizione e Ispezione del Dataset Reale FineFS (`data/raw/finefs/`)
Per il progetto servivano solo i file di **annotazione** e di **scheletri 3D** (evitando di scaricare decine di gigabyte di video RGB grezzi non necessari):
1. Gli archivi delle annotazioni (`1167` file `.json`) sono stati scaricati e posizionati in `data/raw/finefs/annotation/` (`0.json` … `1166.json`).
2. L'archivio degli scheletri (`skeleton.zip`, ~828 MB) presentava un blocco anti-bot/CAPTCHA su Google Drive; in conformità con le regole di sicurezza non lo abbiamo aggirato ma abbiamo interrotto il download automatico e l'utente ha scaricato ed estratto manualmente i `1167` file `.npz` in `data/raw/finefs/skeleton/` (`0.npz` … `1166.npz`).

#### Le 4 Scoperte Critiche dall'Audit Reale sui 1167 File
Quando abbiamo ispezionato davvero i file estratti tramite [src/data/loaders.py](src/data/loaders.py) (`inspect_npz_file`, `audit_finefs_directory`), abbiamo scoperto quattro aspetti essenziali che hanno guidato l'implementazione:

1. **Integrità perfetta dei `.npz`:**  
   Tutti e `1167` i file `.npz` in `skeleton/` esistono, si aprono senza errori, contengono l'array sotto la chiave `"reconstruction"` con shape `(T, 17, 3)` in `float32`, **zero `NaN`/`Inf`**, e coordinate 3D reali non banali (`num_3d_nonzero_npz = 1167`). L'unica particolarità da gestire in fase di lettura era il nome chiave `"reconstruction"` e l'eventuale variazione di shape (`extract_3d_keypoints_from_array` gestisce in automatico `(T,17,3)`, `(T,17,4)`, `(M,T,17,3/4)` prendendo la persona principale, e `(T,51)`).

2. **Scoperta 1 — La convenzione anatomica reale di FineFS è `h36m17` (non `coco17`):**  
   Le specifiche preliminari usavano provvisoriamente l'indicizzazione `coco17` in attesa di ispezionare i dati reali. Verificando geometricamente le lunghezze ossee, le simmetrie e l'allineamento tra vettore anche e vettore spalle nei `.npz` reali, è emerso che FineFS usa lo standard **Human3.6M a 17 giunti (`h36m17`)** con **asse verticale `Z` (indice `2`)**.  
   Senza questa correzione, `canonicalize_frame_3d` avrebbe scambiato orecchie/spalle/anche e gli archi del grafo avrebbero connesso articolazioni inesistenti.  
   Abbiamo quindi aggiornato [configs/data.yaml](configs/data.yaml), [configs/graph.yaml](configs/graph.yaml), il generatore mock in [src/data/loaders.py](src/data/loaders.py) e i test relativi alla mappa reale `h36m17`:

   | Indice | Giunto (`h36m17`) | Indice | Giunto (`h36m17`) |
   | :---: | :--- | :---: | :--- |
   | `0` | `pelvis` (radice) | `9` | `neck` |
   | `1` | `right_hip` | `10` | `head` |
   | `2` | `right_knee` | `11` | `left_shoulder` |
   | `3` | `right_ankle` | `12` | `left_elbow` |
   | `4` | `left_hip` | `13` | `left_wrist` |
   | `5` | `left_knee` | `14` | `right_shoulder` |
   | `6` | `left_ankle` | `15` | `right_elbow` |
   | `7` | `spine` | `16` | `right_wrist` |
   | `8` | `thorax` | | |

   Esempio reale: in `canonicalization`, il bacino è definito da `pelvis_indices: [4, 1]` (`left_hip`, `right_hip`), le spalle da `shoulder_indices: [11, 14]` (`left_shoulder`, `right_shoulder`), e il piano orizzontale su cui ruota lo yaw è `horizontal_axes: [0, 1]` (`X, Y`).

3. **Scoperta 2 — Il campo `"goe"` nei `.json` è pesato sul Base Value, mentre `"judge_score"` è sulla scala ISU `[-5, +5]`:**  
   Guardando ad esempio [data/raw/finefs/annotation/0.json](data/raw/finefs/annotation/0.json), ciascun elemento in `executed_element` contiene:
   - `"bv"`: Base Value (es. `2.8` per un `FCSp3`);
   - `"judge_score"`: i 9 voti grezzi dei giudici nella scala ISU da `-5` a `+5` (es. `[1, 2, 2, 2, 1, 2, 1, 2, 2]`);
   - `"goe"`: il punteggio già moltiplicato per `0.1 * bv` (es. `0.48`), che dipende quindi dal livello di difficoltà dell'elemento e risulta schiacciato intorno a valori piccoli (`<= 1.5`).  
   Poiché [configs/training.yaml](configs/training.yaml) adotta soglie nella scala ISU pura (`goe_pos_threshold: 1.5`, `goe_neg_threshold: 0.0`), in `load_finefs_samples()` abbiamo implementato `_compute_normalized_judge_goe()`:
   - scarta elementi con `bv <= 0` o con tutti i voti sentinella `-6`;
   - filtra i voti validi in `[-5.0, 5.0]` (scartando rari outlier di OCR presenti nei `.json`);
   - calcola la **media troncata ISU** (`np.mean(valid_marks[1:-1])` togliendo minimo e massimo quando ci sono almeno 5 voti validi, altrimenti media semplice o fallback `raw_goe / (0.1 * bv)`).

4. **Scoperta 3 — Selezione delle trottole pure e parsing degli intervalli (`map_finefs_spin_label`, `_parse_finefs_time_range`):**  
   Ogni file `.json` descrive un intero programma di gara con salti (`jump`), sequenze (`sequence`) e trottole (`spin`).  
   - `map_finefs_spin_label(element_code, include_change_foot=False)` filtra i soli elementi `"spin"` delle 3 classi target senza cambio piede:
     - `FCSP*` / `CSP*` → **`Camel Spin`**
     - `FSSP*` / `SSP*` → **`Sit Spin`**
     - `LSP*` → **`Layback Spin`**
     *(I codici con cambio piede `CCSP*`/`CSSP*` sono esclusi di default perché nella finestra temporale centrale mescolerebbero due pose e due gambe d'appoggio diverse).*
   - `_parse_finefs_time_range(time_str, fps=25)` trasforma la stringa `"m-s,m-s"` (es. `"0-52,1-07"`) negli indici di frame `[start_frame, end_frame)` dentro il corrispondente `.npz`.

5. **Scoperta 4 — Assenza di `athlete_id` nei `.json` e Split Anti-Leakage per `performance_id` (`create_grouped_splits`):**  
   I file `.json` pubblici di FineFS anonimizzano il pattinatore (`athlete_id = None`, flag `athlete_id_available: false` nel report).  
   In `create_grouped_splits()` di [src/data/loaders.py](src/data/loaders.py):
   - quando `athlete_id` è disponibile (es. nei test sintetici mock) raggruppa per atleta (`ath_{athlete_id}`);
   - sul dataset reale FineFS raggruppa per **intera esibizione** (`perf_{performance_id}`, es. `finefs_0`), impedendo che due trottole dello stesso programma vadano una in `train` e una in `val`/`test`;
   - valuta fino a `50` permutazioni casuali guidate dal `seed` (`42`) scegliendo quella con migliore copertura sopra/sotto la mediana del GOE in tutte le coppie `(split, classe)`;
   - esegue infine `assert` rigorosi di disgiunzione (`isdisjoint`) su `sample_id`, `performance_id` e `athlete_id`.

---

### 2.3 Risultati Esatti della Materializzazione sul Dataset Reale FineFS
Eseguendo l'intera pipeline di Fase 1 + Fase 2 sui dati reali in `data/raw/finefs/`, sono stati generati e verificati su disco i seguenti artefatti:

#### A. Inventario e Caricamento
- **[reports/finefs_inventory.json](reports/finefs_inventory.json)** (`1,209,740` byte, `status: "ready"`):
  - File `.npz` trovati: `1167` (`1167` validi, `0` invalidi, `1167` 3D non nulli).
  - File annotazione `.json` scansionati: `1167`.
  - Campioni `UnifiedSample` caricati da `load_finefs_samples`: **`1583`** totali (`1` solo segmento scartato per `segment_out_of_bounds_or_short`):
    - **`Camel Spin`**: `696`
    - **`Layback Spin`**: `470`
    - **`Sit Spin`**: `417`
- **[data/intermediate/unified_samples.pkl](data/intermediate/unified_samples.pkl)** (`104,969,494` byte, ~100.1 MB): contiene tutte le `1583` trottole validate in formato `UnifiedSample`.

#### B. Split Anti-Leakage ([data/splits/grouped_splits.json](data/splits/grouped_splits.json), `44,606` byte)
| Split | N° Campioni | N° Esibizioni Distinte | Camel Spin | Layback Spin | Sit Spin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **1102** | `776` | `481` | `328` | `293` |
| **`val`** | **235** | `166` | `115` | `65` | `55` |
| **`test`** | **246** | `166` | `100` | `77` | `69` |
| **Totale** | **1583** | `1108` *(con almeno 1 trottola target)* | `696` | `470` | `417` |

#### C. Grafi PyG Serializzati in `data/processed/`
Tutte e `1583` le trottole hanno superato sia `extract_stable_window` sia `canonicalize_window_3d` con **zero scarti** (`discarded_in_preprocessing: 0` in ogni split):

| File Serializzato | Dimensione | Grafi Salvati | Scartati | `GOE >= 1.5` (Camel / Layback / Sit) | `GOE <= 0.0` (Camel / Layback / Sit) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`data/processed/train_graphs.pt`** | `3,392,589` B (~3.24 MB) | **1102** | `0` | `318` / `247` / `156` (**721** tot.) | `21` / `13` / `32` (**66** tot.) |
| **`data/processed/val_graphs.pt`** | `719,959` B (~703 KB) | **235** | `0` | `71` / `52` / `33` (**156** tot.) | `4` / `2` / `2` (**8** tot.) |
| **`data/processed/test_graphs.pt`** | `755,289` B (~738 KB) | **246** | `0` | `67` / `63` / `35` (**165** tot.) | `4` / `2` / `2` (**8** tot.) |

> **Osservazione chiave per la prossima Fase 3:**  
> Nel training set reale ci sono **721** trottole con qualità alta (`GOE >= 1.5`) ma solo **66** trottole con qualità bassa/negativa (`GOE <= 0.0`), perché nelle competizioni internazionali incluse in FineFS gli errori gravi sono relativamente rari.  
> Questo conferma esattamente perché l'architettura prevede la **data augmentation biomeccanica con rotazioni di Rodrigues** (Fase 3): generare negativi sintetici controllati di severità 1/2/3 a partire dalle esecuzioni corrette è indispensabile per dare alla rete Siamese esempi negativi abbondanti e graduali.

---

## 3. Fase 3 — Augmentation Geometrica di Rodrigues, Triplet Sampling, Loss Adattiva e Training Siamese GATv2

In Fase 3 abbiamo costruito il cuore neurale del sistema: generazione controllata di errori biomeccanici su pose 3D reali, costruzione deterministica delle triplette `(Anchor, Positive, Negative, margin_alpha)`, addestramento con rete **Siamese GATv2** e prevenzione sistematica del collasso dello spazio latente.

### 3.1 Scelte Architetturali e Diagnostiche Empiriche su FineFS Reale
Durante la messa a punto sperimentale su [configs/training.yaml](configs/training.yaml) e sui grafi reali (`train_graphs.pt` = `1102`, `val_graphs.pt` = `235`), la validazione ha evidenziato due fenomeni fondamentali che hanno guidato la configurazione finale:

1. **Impatto del `dropout` su grafi scheletrici compatti (17 nodi) con Siamese Triplet Loss (`dropout: 0.0`):**  
   In `GATv2Encoder`, ogni grafo ha solo `17` nodi e `32` archi diretti più `17` self-loop, normalizzati con `LayerNorm` prima di `ELU`. Applicando maschere di dropout casuali e indipendenti ai rami `(A, P, N)`, il rumore stocastico tra maschere risultava più grande della separazione iniziale `(d_an - d_ap)`, impedendo l'overfit su batch fisso di `16` triplette (`train_acc ~ 0.50` anche quando `eval_acc == 1.000`) e inducendo AdamW a collassare `z` verso un vettore costante.  
   Validando `dropout: 0.0` (come previsto dalla specifica padre per il tuning su validation), il test di overfit su `16` triplette converge stabilmente senza collasso.

2. **Prevenzione del collasso latente su FineFS reale (`Spin Variants` & Rotazioni Orarie/Antiorarie):**  
   Sul dataset reale FineFS, la stessa classe di trottola (es. *Camel Spin* o *Sit Spin*) include varianti stilistiche legittime e pattinatori che ruotano sia in senso orario che antiorario. Con un campionamento ingenuo:
   - un positivo `P` pescato da una variante diversa ha distanza euclidea grezza elevata rispetto all'àncora `A`, mentre un negativo sintetico `self_synth` ottenuto perturbando lievemente `A` stessa le assomiglia molto di più a livello globale;
   - una baseline puramente geometrica sulle coordinate allineate (`Raw Coord Fro`) raggiunge solo `0.325` di accuracy sulle triplette di validazione (`Raw Angle L2` = `0.565`), e un training Siamese non regolarizzato collassava rapidamente (`embedding_std -> 0.004`, `val_triplet_acc = 0.405`).
   
   Per risolvere strutturalmente il problema senza toccare in alcun modo la valutazione su `val`/`test`, sono stati introdotti tre accorgimenti sinergici in [configs/training.yaml](configs/training.yaml), [src/training/triplet_sampler.py](src/training/triplet_sampler.py) e [src/training/losses.py](src/training/losses.py):
   - **`positive_top_k: 8` (attivo SOLO su `split == "train"`):** `_select_angle_nearest_positives()` filtra i candidati positivi `P` (sempre con `performance_id` e atleta distinto quando disponibile) restringendo il campionamento ai `8` più vicini ad `A` nello spazio degli angoli articolari `edge_attr[:, 4]` (`cos(theta)`), evitando conflitti tra sottovarianti incompatibili in addestramento; in `val` e `test` il campionamento rimane completamente uniforme e imparziale.
   - **Bilanciamento dei negativi (`real_negative_prob: 0.70`, `self_synth_prob: 0.20`):** privilegia i negativi reali di bassa qualità esecutiva (`70%`) e, tra i negativi sintetici (`30%`), riduce l'eccesso di perturbazioni della stessa ancora (`20%` `self_synth`, `80%` `cross_synth` da altri esempi positivi).
   - **Hinge sulla varianza latente anti-collasso (`var_reg_weight: 0.35`, `var_reg_target: 1.0`, attivo SOLO su `train_criterion`):** penalizza il calo della deviazione standard per dimensione sotto `1.0` su `[z_A, z_P, z_N]`, lasciando invariata e priva di regolarizzazione `eval_criterion` così da misurare in validazione la sola `triplet_loss` pura.

---

### 3.2 Moduli Implementati in Fase 3

1. **[src/training/augmentation.py](src/training/augmentation.py) — Errori Biomeccanici via Rotazioni SO(3) di Rodrigues:**
   - `rodrigues_rotation_matrix(axis, angle_rad)`: calcola la matrice ortogonale `3x3` $R = I + \sin(\theta)K + (1 - \cos(\theta))K^2$.
   - `rotate_kinematic_subtree(P_bar, joint_idx, child_idx, axis, angle_deg, graph_cfg)`: legge le catene cinematiche `H36M-17` da `graph_cfg["kinematic_subtrees"]` (es. `"4->5": [5, 6]`) e ruota rigidamente il sotto-albero attorno al perno `P_bar[joint_idx]` preservando esattamente tutte le lunghezze ossee.
   - `_select_weighted_subtree_keys()` e `_orthogonal_unit_axis()`: privilegiano dinamicamente dai `joint_names` le articolazioni degli arti inferiori (`hip`/`knee`, peso `2.5`) e superiori (`shoulder`/`elbow`, peso `1.2`) rispetto al tronco (`0.3`), generando assi casuali ortogonali all'osso.
   - `generate_synthetic_negatives(train_graphs, data_cfg, graph_cfg, train_cfg)`: blocca con `ValueError` qualsiasi grafo con `split != "train"` (barriera anti-leakage); per ciascuno dei `721` grafi positivi reali del solo `train` genera `3` negativi sintetici (`tier` `1`, `2`, `3` con perturbazioni `[8, 15]°`, `[16, 28]°`, `[29, 42]°` più rumore gaussiano `noise_std = 0.005`), li ri-canonicalizza con `canonicalize_frame_3d()` (con fallback privo di rumore in caso di `DegenerateSkeletonError`) e ricostruisce i grafi PyG (`2163` grafi sintetici totali su FineFS).

2. **[src/training/triplet_sampler.py](src/training/triplet_sampler.py) — Partizionamento per GOE e Dataset Deterministico:**
   - `resolve_class_thresholds(real_class_graphs, train_cfg)`: calcola `(tau_neg, tau_pos)` per ciascuna classe usando le soglie `goe_neg_threshold = 0.0` e `goe_pos_threshold = 1.5` (con fallback automatico ai quantili `0.35` / `0.70` se il pool è troppo ristretto).
   - `partition_by_goe_tiers(graphs, train_cfg, fit_thresholds=None)`: separa i grafi per classe in `pos`, `mid`, `neg_real`, `neg_synth`; `val` e `test` ricevono sempre le soglie stimate sul solo training set.
   - `TripletDataset`: genera un set deterministico di `num_triplets` tuple `(A, P, N, margin_alpha)` all'interno della medesima classe (`element_label`):
     - verifica esplicitamente l'assenza di grafi aumentati nei pool reali di `val` e `test`;
     - sceglie `P` da `performance_id` e `athlete_id` distinti rispetto ad `A`, preferendo `|GOE(A) - GOE(P)| <= max_pos_goe_diff` (`1.5`) e, solo su `split == "train"`, applicando `_select_angle_nearest_positives(..., positive_top_k=8)`;
     - sceglie `N` da `neg_real` con probabilità `0.70` (mappando `delta_goe` sui tier `1`, `2`, `3` per il margine adattivo `0.20 / 0.60 / 1.00`) o da `neg_synth` con probabilità `0.30` (di cui `self_synth_prob = 0.20` dallo stesso `parent_sample_id` e `0.80` `cross_synth` con fallback su `neg_synth_pool`).

3. **[src/training/losses.py](src/training/losses.py) — Loss Adattiva Numericamente Sicura:**
   - `AdaptiveTripletMarginLoss(eps=1e-12, var_reg_weight=0.0, var_reg_target=1.0)`: calcola $d_{AP} = \sqrt{\|z_A - z_P\|_2^2 + \varepsilon}$ e $d_{AN} = \sqrt{\|z_A - z_N\|_2^2 + \varepsilon}$ (evitando derivate `NaN`/`inf` a distanza zero), applica `F.relu(d_ap - d_an + margins)` e opzionalmente aggiunge la regolarizzazione di varianza `var_reg_weight * mean(F.relu(var_reg_target - std_per_dim))`. Restituisce `(loss, stats)` dove `stats["loss"]` registra sempre la `triplet_loss` pura.

4. **[src/models/encoder.py](src/models/encoder.py) e [src/models/siamese_gat.py](src/models/siamese_gat.py) — Rete Siamese GATv2:**
   - `GATv2Encoder`: `GATv2Conv` multi-testa (`in_channels=3`, `hidden_dim=32`, `heads=4`, `edge_dim=5`, `concat=True`, `add_self_loops=True`, `share_weights=False`) → `LayerNorm(128)` → `ELU` → `GATv2Conv` (`concat=False`, `heads=4`) → `LayerNorm(32)` → `ELU` → `global_mean_pool` → MLP `proj` (`32 -> 32 -> ELU -> 32`). Supporta `return_attention_weights=True` restituendo `{"layer1": (edge_idx1, alpha1), "layer2": (edge_idx2, alpha2)}` per l'explainability in Fase 4 e 5.
   - `SiameseGATv2`: incapsula **un'unica istanza condivisa** `self.encoder` usata da `encode`, `forward_triplet` e `compute_distance`.

5. **[src/training/train.py](src/training/train.py) — Loop di Addestramento, Validazione e Checkpoint:**
   - `train_one_epoch`: forward Siamese, calcolo `train_criterion` (con regolarizzazione di varianza), `loss.backward()`, gradient clipping (`max_norm = 1.0`) e passo `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`).
   - `evaluate_triplets`: valutazione in `@torch.no_grad()` con `eval_criterion` puro (`var_reg_weight = 0.0`) e monitoraggio esplicito di `embedding_std` su tutti gli embedding di ancoraggio `z_A`.
   - `run_training_pipeline` & `main()`: genera le augmentation una sola volta sul solo train, ricostruisce `TripletDataset(..., seed=seed + epoch)` ad ogni epoca, salva `checkpoints/best_model.pt` solo quando migliora `val_triplet_acc` (o a parità migliora `val_loss`) e `embedding_std > 1e-6`, applica early stopping (`patience = 15`) e salva la cronologia completa in [reports/training_history.json](reports/training_history.json).

---

### 3.3 Risultati del Training Reale su FineFS (`cuda`)
Eseguendo `python -m src.training.train` su **NVIDIA GeForce RTX 3050 Laptop GPU** (`device = "cuda"`) a partire da `data/processed/train_graphs.pt` e `data/processed/val_graphs.pt` (con **`test_graphs.pt` completamente intatto** per la Fase 4):

- **Dati utilizzati:** `1102` grafi reali di train + **`2163` grafi negativi sintetici** (`721` positivi × `3` tier); **`235` grafi reali di validazione** (`0` sintetici).
- **Soglie GOE stimate su `train` (`[tau_neg, tau_pos]`):** `Camel Spin: [0.0, 1.5]`, `Layback Spin: [0.0, 1.5]`, `Sit Spin: [0.0, 1.5]`.
- **Andamento Training (`62` epoche completate, early stopping attivato 15 epoche dopo l'epoca migliore `47`):**
  - Epoca `1`: `train_loss = 0.6637`, `train_triplet_acc = 0.5583`, `val_loss = 0.7853`, `val_triplet_acc = 0.4650`, `embedding_std = 0.1028`.
  - **Miglior epoca (`47`):**
    - `train_loss = 0.4849` | `train_triplet_acc = 0.7117` | `train_margin_satisfied_acc = 0.4533` (`mean_d_ap = 1.2028`, `mean_d_an = 1.8667`)
    - **`val_loss = 0.7290`** | **`val_triplet_acc = 0.6500` (`65.0%`)** | `val_margin_satisfied_acc = 0.3750` (`mean_d_ap = 1.8857`, `mean_d_an = 2.2023`)
    - **`embedding_std = 0.2889`** (`0.28888118267059326`, spazio latente attivo e privo di collasso)
  - Epoca finale (`62`): `train_loss = 0.4298`, `train_triplet_acc = 0.7250`, `val_loss = 0.7982`, `val_triplet_acc = 0.6050`, `embedding_std = 0.3157`.
- **Artefatti salvati su disco:**
  - `checkpoints/best_model.pt` (contenente `epoch`, `model_state_dict`, `val_loss`, `val_triplet_acc`, `embedding_std`, `goe_thresholds`, `train_cfg`, `graph_cfg`, `data_cfg`)
  - [reports/training_history.json](reports/training_history.json) (`62` record epoca per epoca)

---

## 4. Suite di Test Automatici Verificata (`pytest tests/ -v`)

Al termine delle modifiche di Fase 3, è stata eseguita l'intera suite di test (`30 passed, 148 warnings in 24.89s`, `0` errori, `0` skip):

- **[tests/test_config_and_utils.py](tests/test_config_and_utils.py)** (`5` test):
  - `test_seed_everything_reproducibility`
  - `test_load_all_configs_valid`
  - `test_load_all_configs_validation_errors`
  - `test_hardware_helpers_contract`
  - `test_env_check_main_writes_profile`
- **[tests/test_dataset.py](tests/test_dataset.py)** (`7` test, inclusa verifica diretta sui file reali):
  - `test_unified_sample_validation`
  - `test_extract_3d_keypoints_shapes`
  - `test_grouped_splits_zero_leakage`
  - `test_save_and_load_unified_samples`
  - `test_inspect_npz_and_audit_directory`
  - `test_end_to_end_mock_to_pyg_graphs`
  - `test_load_real_finefs_samples_and_splits`
- **[tests/test_graph.py](tests/test_graph.py)** (`4` test):
  - `test_topology_paired_ordering`
  - `test_edge_features_geometry`
  - `test_pyg_data_and_batching`
  - `test_extended_node_features_preserves_pos`
- **[tests/test_normalization.py](tests/test_normalization.py)** (`6` test):
  - `test_root_centering_at_origin`
  - `test_unit_torso_scale`
  - `test_yaw_alignment_along_positive_x`
  - `test_exact_invariance_to_translation_scale_and_z_rotation`
  - `test_rotating_spin_window_preservation`
  - `test_extract_stable_window_filtering`
- **[tests/test_triplets.py](tests/test_triplets.py)** (`8` test — Gate 3):
  - `test_rodrigues_subtree_preserves_bone_lengths`
  - `test_triplet_dataset_intra_class_and_no_val_augmentation`
  - `test_adaptive_triplet_loss_backward_and_zero_distance_safety`
  - `test_pyg_dataloader_on_triplet_dataset`
  - `test_siamese_shared_weights_and_edge_attr_sensitivity`
  - `test_attention_weights_extraction_shape`
  - `test_single_batch_overfit` (convergenza stretta sia su `final_stats` che su `eval_stats`: `loss < 0.15` e `triplet_acc >= 0.90`)
  - `test_run_training_pipeline_end_to_end_mock`

*(Nota ambiente Windows: durante lo sviluppo, la sandbox blocca le operazioni di scrittura diretta nei percorsi del workspace con `PermissionError` / `RuntimeError`; per questo i passaggi che scrivono file generati in `data/`, `checkpoints/` e `reports/` vengono eseguiti con permessi di scrittura abilitati, mentre `pytest` gira pulito grazie a `-p no:cacheprovider` in `pyproject.toml`.)*

---

## 5. Prossimi Aggiornamenti (Sezioni Riservate alle Fasi 4 e 5)

### Fase 4 — Baseline Geometriche, Metriche AQA, Calibrazione $Q$ e Inferenza
*(Da compilare al completamento della Fase 4)*

### Fase 5 — Visualizzazione 3D, Figure Automatiche e Dashboard Streamlit
*(Da compilare al completamento della Fase 5)*

