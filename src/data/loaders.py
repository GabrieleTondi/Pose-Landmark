import csv
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.data.harmonization import UnifiedSample


def _rodrigues_z(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Ruota un array (N, 3) attorno all'asse verticale Z."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return points @ rot.T


def _rodrigues_axis(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Ruota un vettore 3D attorno a un asse unitario mediante formula di Rodrigues."""
    k = axis / (float(np.linalg.norm(axis)) + 1e-12)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return vec * c + np.cross(k, vec) * s + k * float(np.dot(k, vec)) * (1.0 - c)


def _rotate_limb_subtree(
    pose: np.ndarray,
    pivot_idx: int,
    descendant_indices: List[int],
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Applica rotazione rigida ai joint discendenti attorno al joint pivot."""
    out = pose.copy()
    pivot = out[pivot_idx].copy()
    angle_rad = float(np.deg2rad(angle_deg))
    for idx in descendant_indices:
        rel = out[idx] - pivot
        out[idx] = pivot + _rodrigues_axis(rel, axis, angle_rad)
    return out


def _base_element_template(element_label: str) -> np.ndarray:
    """
    Restituisce un template Human3.6M-17 3D anatomicamente plausibile (busto ~0.50m)
    coerente con configs/data.yaml e configs/graph.yaml:
    0=pelvis, 1=r_hip, 2=r_knee, 3=r_ankle, 4=l_hip, 5=l_knee, 6=l_ankle,
    7=spine, 8=thorax, 9=neck, 10=head,
    11=l_shoulder, 12=l_elbow, 13=l_wrist, 14=r_shoulder, 15=r_elbow, 16=r_wrist.
    """
    base = np.array(
        [
            [0.00, 0.00, 0.00],   # 0: pelvis
            [0.11, 0.00, 0.00],   # 1: right_hip
            [0.12, 0.02, -0.42],  # 2: right_knee
            [0.12, 0.00, -0.84],  # 3: right_ankle
            [-0.11, 0.00, 0.00],  # 4: left_hip
            [-0.12, 0.02, -0.42], # 5: left_knee (gamba d'appoggio)
            [-0.12, 0.00, -0.84], # 6: left_ankle
            [0.00, -0.02, 0.25],  # 7: spine
            [0.00, 0.00, 0.50],   # 8: thorax
            [0.00, 0.02, 0.62],   # 9: neck
            [0.00, 0.03, 0.75],   # 10: head
            [-0.17, 0.00, 0.50],  # 11: left_shoulder
            [-0.35, 0.10, 0.45],  # 12: left_elbow
            [-0.50, 0.22, 0.48],  # 13: left_wrist
            [0.17, 0.00, 0.50],   # 14: right_shoulder
            [0.35, 0.10, 0.45],   # 15: right_elbow
            [0.50, 0.22, 0.48],   # 16: right_wrist
        ],
        dtype=np.float64,
    )

    label_lower = element_label.lower()
    if "camel" in label_lower:
        # Busto inclinato avanti in posizione T, gamba destra libera tesa orizzontale indietro
        camel = base.copy()
        for idx in (7, 8, 9, 10, 11, 14):
            y_old = camel[idx, 1]
            z_old = camel[idx, 2]
            camel[idx, 1] = y_old + 0.90 * z_old
            camel[idx, 2] = 0.18 * z_old
        camel[12] = [-0.36, 0.30, 0.14]
        camel[13] = [-0.55, 0.25, 0.18]
        camel[15] = [0.36, 0.30, 0.14]
        camel[16] = [0.55, 0.25, 0.18]
        camel[2] = [0.12, -0.42, 0.02]
        camel[3] = [0.12, -0.84, 0.04]
        return camel

    if "sit" in label_lower:
        # Ginocchio sinistro piegato, gamba destra libera distesa avanti, braccia raccolte in avanti
        sit = base.copy()
        sit[5] = [-0.12, 0.34, -0.05]
        sit[6] = [-0.12, 0.02, -0.47]
        sit[2] = [0.12, 0.42, -0.02]
        sit[3] = [0.12, 0.84, -0.04]
        sit[12] = [-0.24, 0.28, 0.32]
        sit[13] = [-0.16, 0.48, 0.20]
        sit[15] = [0.24, 0.28, 0.32]
        sit[16] = [0.16, 0.48, 0.20]
        return sit

    if "layback" in label_lower:
        # Busto, spalle e testa inarcati indietro, braccia aperte, gamba libera dietro in attitude
        lay = base.copy()
        lay[7] = [0.00, -0.10, 0.24]
        lay[8] = [0.00, -0.22, 0.46]
        lay[9] = [0.00, -0.34, 0.54]
        lay[10] = [0.00, -0.46, 0.56]
        lay[11] = [-0.17, -0.20, 0.46]
        lay[14] = [0.17, -0.20, 0.46]
        lay[12] = [-0.32, -0.10, 0.56]
        lay[13] = [-0.46, 0.04, 0.72]
        lay[15] = [0.32, -0.10, 0.56]
        lay[16] = [0.46, 0.04, 0.72]
        lay[2] = [0.14, -0.34, -0.22]
        lay[3] = [0.16, -0.58, -0.02]
        return lay

    return base


def inspect_npz_file(npz_path: Union[str, Path]) -> Dict[str, Any]:
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
    file_path = Path(npz_path)
    try:
        with np.load(file_path, allow_pickle=True) as archive:
            keys = list(archive.files)
            shapes: Dict[str, Tuple[int, ...]] = {}
            dtypes: Dict[str, str] = {}
            has_nan = False
            is_3d_nonzero = False

            for key in keys:
                arr = archive[key]
                shapes[key] = tuple(int(dim) for dim in arr.shape)
                dtypes[key] = str(arr.dtype)

                if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                    if not np.isfinite(arr).all():
                        has_nan = True

                    z_vals: Optional[np.ndarray] = None
                    if arr.ndim >= 2 and arr.shape[-1] in (3, 4):
                        z_vals = arr[..., 2]
                    elif arr.ndim == 2 and arr.shape[-1] == 51:
                        z_vals = arr.reshape(arr.shape[0], 17, 3)[..., 2]

                    if z_vals is not None:
                        finite_z = z_vals[np.isfinite(z_vals)]
                        if finite_z.size > 1 and float(np.var(finite_z)) > 1e-6:
                            is_3d_nonzero = True

            return {
                "valid": len(keys) > 0,
                "keys": keys,
                "shapes": shapes,
                "dtypes": dtypes,
                "has_nan": has_nan,
                "is_3d_nonzero": is_3d_nonzero,
                "error": None if len(keys) > 0 else "Archivio .npz vuoto.",
            }
    except Exception as exc:
        return {
            "valid": False,
            "keys": [],
            "shapes": {},
            "dtypes": {},
            "has_nan": False,
            "is_3d_nonzero": False,
            "error": str(exc),
        }


def _inspect_annotation_file(ann_path: Path) -> Dict[str, Any]:
    """Ispeziona in sicurezza un file di annotazione (.json/.csv/.pkl/.txt)."""
    suffix = ann_path.suffix.lower()
    info: Dict[str, Any] = {"file": str(ann_path), "extension": suffix}
    try:
        if suffix == ".json":
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            info["type"] = type(data).__name__
            if isinstance(data, dict):
                info["keys"] = list(data.keys())[:30]
            elif isinstance(data, list):
                info["num_entries"] = len(data)
        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with ann_path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                rows = list(reader)
            info["num_rows"] = len(rows)
            info["header"] = rows[0] if rows else []
        elif suffix in (".pkl", ".pickle"):
            with ann_path.open("rb") as handle:
                obj = pickle.load(handle)
            info["type"] = type(obj).__name__
            if isinstance(obj, (dict, list, tuple)):
                info["len"] = len(obj)
        else:
            lines = ann_path.read_text(encoding="utf-8", errors="replace").splitlines()
            info["num_lines"] = len(lines)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def audit_finefs_directory(raw_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Scansiona raw_dir, ispeziona i file .npz e i file di annotazione (.json/.csv/.pkl),
    e restituisce un report completo salvabile in reports/finefs_inventory.json.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return {
            "raw_dir": str(raw_path),
            "directory_exists": False,
            "status": "missing_raw_directory",
            "num_npz_files": 0,
            "num_valid_npz": 0,
            "num_invalid_npz": 0,
            "num_annotation_files": 0,
            "npz_files": [],
            "annotation_files": [],
        }

    npz_paths = sorted(raw_path.rglob("*.npz"))
    ann_extensions = {".json", ".csv", ".tsv", ".pkl", ".pickle", ".txt"}
    ann_paths = sorted(
        p for p in raw_path.rglob("*") if p.is_file() and p.suffix.lower() in ann_extensions
    )

    npz_reports: List[Dict[str, Any]] = []
    num_valid = 0
    num_3d_nonzero = 0
    for p in npz_paths:
        rep = inspect_npz_file(p)
        rep["relative_path"] = str(p.relative_to(raw_path))
        if rep["valid"]:
            num_valid += 1
        if rep["is_3d_nonzero"]:
            num_3d_nonzero += 1
        npz_reports.append(rep)

    ann_reports: List[Dict[str, Any]] = []
    for p in ann_paths:
        rep = _inspect_annotation_file(p)
        rep["relative_path"] = str(p.relative_to(raw_path))
        ann_reports.append(rep)

    if len(npz_paths) == 0:
        status = "empty"
    elif num_valid > 0:
        status = "ready"
    else:
        status = "invalid_files"

    return {
        "raw_dir": str(raw_path),
        "directory_exists": True,
        "status": status,
        "num_npz_files": len(npz_paths),
        "num_valid_npz": num_valid,
        "num_invalid_npz": len(npz_paths) - num_valid,
        "num_3d_nonzero_npz": num_3d_nonzero,
        "num_annotation_files": len(ann_paths),
        "npz_files": npz_reports,
        "annotation_files": ann_reports,
    }


def extract_3d_keypoints_from_array(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Normalizza le diverse shape possibili degli array HRNet/FineFS in:
    - keypoints_3d: np.ndarray di shape (T, 17, 3) in float32
    - confidence: Optional[np.ndarray] di shape (T, 17) in float32
    Gestisce i casi:
    - shape (T, 17, 3) -> keypoints=arr, conf=None
    - shape (T, 17, 4) -> keypoints=arr[:, :, :3], conf=arr[:, :, 3]
    - shape (1, T, 17, 3) o (M, T, 17, 3) -> prende la persona principale arr[0]
    - shape (1, T, 17, 4) o (M, T, 17, 4) -> prende arr[0, :, :, :3], arr[0, :, :, 3]
    - shape (T, 51) -> reshape a (T, 17, 3)
    Solleva ValueError per qualsiasi altra shape non riconducibile a 17 joint 3D.
    """
    data = np.asarray(arr)

    if data.ndim == 3 and data.shape[0] >= 1 and data.shape[1] == 17 and data.shape[2] == 3:
        return np.ascontiguousarray(data, dtype=np.float32), None

    if data.ndim == 3 and data.shape[0] >= 1 and data.shape[1] == 17 and data.shape[2] == 4:
        kpts = np.ascontiguousarray(data[:, :, :3], dtype=np.float32)
        conf = np.ascontiguousarray(data[:, :, 3], dtype=np.float32)
        return kpts, conf

    if (
        data.ndim == 4
        and data.shape[0] >= 1
        and data.shape[1] >= 1
        and data.shape[2] == 17
        and data.shape[3] in (3, 4)
    ):
        person0 = data[0]
        if person0.shape[2] == 3:
            return np.ascontiguousarray(person0, dtype=np.float32), None
        kpts = np.ascontiguousarray(person0[:, :, :3], dtype=np.float32)
        conf = np.ascontiguousarray(person0[:, :, 3], dtype=np.float32)
        return kpts, conf

    if data.ndim == 2 and data.shape[0] >= 1 and data.shape[1] == 51:
        kpts = np.ascontiguousarray(data.reshape(data.shape[0], 17, 3), dtype=np.float32)
        return kpts, None

    raise ValueError(
        f"Shape array non riconducibile a 17 joint 3D: {data.shape}. "
        "Attese (T,17,3), (T,17,4), (M,T,17,3/4) o (T,51)."
    )


def map_finefs_spin_label(
    element_code: str,
    include_change_foot: bool = False,
) -> Optional[str]:
    """Mappa i codici ISU di FineFS alle tre classi target di trottole dell'MVP."""
    code = str(element_code).strip().upper()
    camel_prefixes = ("FCSP", "CSP", "CCSP") if include_change_foot else ("FCSP", "CSP")
    sit_prefixes = ("FSSP", "SSP", "CSSP") if include_change_foot else ("FSSP", "SSP")
    layback_prefixes = ("LSP",)

    if code.startswith(camel_prefixes):
        return "Camel Spin"
    if code.startswith(sit_prefixes):
        return "Sit Spin"
    if code.startswith(layback_prefixes):
        return "Layback Spin"
    return None


def _parse_finefs_time_range(time_str: str, fps: int) -> Tuple[int, int]:
    """Converte la stringa FineFS 'm-s,m-s' in indici di frame [start_frame, end_frame)."""
    parts = [p.strip() for p in str(time_str).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Formato intervallo temporale FineFS non valido: {time_str}")

    def _to_sec(token: str) -> int:
        segs = token.split("-")
        if len(segs) != 2:
            raise ValueError(f"Timestamp '{token}' non in formato mm-ss.")
        return int(segs[0]) * 60 + int(segs[1])

    start_sec = _to_sec(parts[0])
    end_sec = _to_sec(parts[1])
    return int(start_sec * fps), int(end_sec * fps)


def _compute_normalized_judge_goe(elem_meta: Dict[str, Any]) -> Optional[float]:
    """
    Estrae il grado di qualità normalizzato sulla scala ISU [-5.0, +5.0] a partire dai voti
    dei giudici ('judge_score', media tagliata escludendo eventuali sentinelle/outlier),
    con fallback su goe / (0.1 * bv). Restituisce None per elementi invalidi/non valutati.
    """
    bv = float(elem_meta.get("bv") or 0.0)
    raw_goe = elem_meta.get("goe")
    judge_scores = elem_meta.get("judge_score", [])

    if bv <= 0.0:
        return None
    if isinstance(judge_scores, list) and len(judge_scores) > 0:
        if all(int(x) == -6 for x in judge_scores):
            return None
        valid_marks = sorted(float(x) for x in judge_scores if -5.0 <= float(x) <= 5.0)
        if len(valid_marks) >= 5:
            return float(np.mean(valid_marks[1:-1]))
        if len(valid_marks) >= 1:
            return float(np.mean(valid_marks))

    if raw_goe is not None and np.isfinite(float(raw_goe)):
        approx_mark = float(raw_goe) / (0.1 * bv)
        return float(np.clip(approx_mark, -5.0, 5.0))
    return None


def load_finefs_samples(
    raw_dir: Union[str, Path],
    data_cfg: Dict[str, Any],
    include_change_foot: bool = False,
    fps: int = 25,
) -> Tuple[List[UnifiedSample], Dict[str, Any]]:
    """
    Carica i segmenti reali del dataset FineFS da raw_dir (annotation/*.json + skeleton/*.npz),
    convertendoli in UnifiedSample validati per le classi in data_cfg['dataset']['target_elements'].
    Restituisce (samples, report).
    """
    raw_path = Path(raw_dir)
    ann_dir = raw_path / "annotation"
    skel_dir = raw_path / "skeleton"
    if not ann_dir.is_dir() or not skel_dir.is_dir():
        raise FileNotFoundError(
            f"Directory FineFS incomplete in {raw_path}: attese 'annotation/' e 'skeleton/'."
        )

    target_elements = set(data_cfg["dataset"]["target_elements"])
    min_valid_frames = int(data_cfg["windowing"].get("min_valid_frames", 5))
    ann_files = sorted(
        ann_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )

    samples: List[UnifiedSample] = []
    discarded: Dict[str, int] = {}
    kept_per_class: Dict[str, int] = {}

    def _bump(reason: str) -> None:
        discarded[reason] = discarded.get(reason, 0) + 1

    for ann_path in ann_files:
        stem = ann_path.stem
        npz_path = skel_dir / f"{stem}.npz"
        if not npz_path.is_file():
            _bump("missing_skeleton_npz")
            continue

        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as archive:
            if "reconstruction" in archive.files:
                raw_arr = archive["reconstruction"]
            elif len(archive.files) == 1:
                raw_arr = archive[archive.files[0]]
            else:
                _bump("ambiguous_npz_keys")
                continue

        full_kpts, full_conf = extract_3d_keypoints_from_array(raw_arr)
        n_frames = int(full_kpts.shape[0])
        performance_id = f"finefs_{stem}"
        competition = str(ann_data.get("competition", "unknown"))
        competition_index = ann_data.get("competition_index")

        executed = ann_data.get("executed_element", {})
        if not isinstance(executed, dict):
            continue

        for elem_key, elem_meta in executed.items():
            if not isinstance(elem_meta, dict):
                continue
            if elem_meta.get("coarse_class") != "spin":
                continue

            raw_code = str(elem_meta.get("element", ""))
            mapped_label = map_finefs_spin_label(
                raw_code, include_change_foot=include_change_foot
            )
            if mapped_label is None or mapped_label not in target_elements:
                continue

            norm_goe = _compute_normalized_judge_goe(elem_meta)
            if norm_goe is None or not np.isfinite(norm_goe):
                _bump("invalid_goe_or_bv")
                continue

            try:
                sf, ef = _parse_finefs_time_range(str(elem_meta.get("time", "")), fps=fps)
            except ValueError:
                _bump("invalid_time_format")
                continue

            sf_clamped = max(0, sf)
            ef_clamped = min(n_frames, ef)
            if ef_clamped - sf_clamped < min_valid_frames:
                _bump("segment_out_of_bounds_or_short")
                continue

            seg_kpts = full_kpts[sf_clamped:ef_clamped]
            if not np.isfinite(seg_kpts).all():
                _bump("nan_in_segment")
                continue
            seg_conf = full_conf[sf_clamped:ef_clamped] if full_conf is not None else None

            sample = UnifiedSample(
                sample_id=f"{performance_id}_{elem_key}",
                dataset_id="finefs",
                performance_id=performance_id,
                athlete_id=None,  # FineFS anonimizza gli atleti nei .json
                element_label=mapped_label,
                start_frame=sf_clamped,
                end_frame=ef_clamped - 1,
                keypoints=seg_kpts,
                keypoint_confidence=seg_conf,
                goe=float(norm_goe),
                source_metadata={
                    "raw_file": str(npz_path.relative_to(raw_path)),
                    "annotation_file": str(ann_path.relative_to(raw_path)),
                    "raw_label": raw_code,
                    "element_key": elem_key,
                    "time_str": elem_meta.get("time"),
                    "bv": float(elem_meta.get("bv", 0.0)),
                    "bv_weighted_goe": float(elem_meta.get("goe", 0.0)),
                    "judge_score": list(elem_meta.get("judge_score", [])),
                    "competition": competition,
                    "competition_index": competition_index,
                    "goe_scale": "isu_judge_trimmed_mean_-5_to_+5",
                },
            )
            sample.validate()
            samples.append(sample)
            kept_per_class[mapped_label] = kept_per_class.get(mapped_label, 0) + 1

    report = {
        "raw_dir": str(raw_path),
        "num_annotation_files_scanned": len(ann_files),
        "num_samples_loaded": len(samples),
        "kept_per_class": kept_per_class,
        "discarded_counts": discarded,
        "include_change_foot": include_change_foot,
        "fps": fps,
        "athlete_id_available": False,
    }
    return samples, report


def generate_mock_finefs_samples(
    num_performances: int = 30,
    elements: Tuple[str, ...] = ("Camel Spin", "Sit Spin", "Layback Spin"),
    seed: int = 42,
) -> List[UnifiedSample]:
    """
    Genera una collezione di UnifiedSample biomeccanicamente plausibili a 17 joint (standard H36M-17)
    con diversi atleti, rotazioni di piroetta frame-by-frame, traslazioni, scale e punteggi GOE in [-3.0, +3.0]
    correlati alla precisione degli angoli articolari.
    Usato per i test automatici pytest e per verificare la pipeline end-to-end.
    """
    rng = np.random.default_rng(seed)
    samples: List[UnifiedSample] = []
    num_frames = 25
    omega = float(2.0 * np.pi / 9.0)  # giro completo sulla finestra da 9 frame

    for p in range(num_performances):
        athlete_id = f"ath_{p // 2:03d}"
        performance_id = f"perf_{p:04d}"

        for c_idx, element_label in enumerate(elements):
            template = _base_element_template(element_label)
            goe = float(np.round(rng.uniform(-3.0, 3.0), 3))

            # Deviazione biomeccanica reale correlata al GOE sulla gamba libera destra (1->2->3 in H36M-17)
            angular_dev_deg = 5.0 * abs(goe) if goe < 0.0 else max(0.0, 1.5 - goe) * 2.0
            pose_elem = template.copy()
            if angular_dev_deg > 0.0:
                sign = 1.0 if c_idx % 2 == 0 else -1.0
                pose_elem = _rotate_limb_subtree(
                    pose_elem,
                    pivot_idx=1,
                    descendant_indices=[2, 3],
                    axis=np.array([1.0, 0.2, 0.0], dtype=np.float64),
                    angle_deg=sign * angular_dev_deg,
                )
                pose_elem = _rotate_limb_subtree(
                    pose_elem,
                    pivot_idx=2,
                    descendant_indices=[3],
                    axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                    angle_deg=-0.5 * sign * angular_dev_deg,
                )

            scale_p = float(rng.uniform(0.8, 1.5))
            trans_p = rng.uniform(-2.0, 2.0, size=(1, 3))
            phase0 = float(rng.uniform(-np.pi, np.pi))

            frames = np.zeros((num_frames, 17, 3), dtype=np.float32)
            for t in range(num_frames):
                frame_pose = pose_elem.copy()
                # Transitorio di ingresso (primi 4 frame) e uscita (ultimi 4 frame) sul braccio SX (11->12->13)
                if t < 4 or t >= num_frames - 4:
                    transient_deg = 12.0 if t < 4 else -12.0
                    frame_pose = _rotate_limb_subtree(
                        frame_pose,
                        pivot_idx=11,
                        descendant_indices=[12, 13],
                        axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                        angle_deg=transient_deg,
                    )

                rotated = _rodrigues_z(frame_pose, phase0 + omega * float(t))
                noise = rng.normal(0.0, 0.002, size=(17, 3))
                frames[t] = (scale_p * rotated + trans_p + noise).astype(np.float32)

            conf = rng.uniform(0.75, 0.99, size=(num_frames, 17)).astype(np.float32)
            start_frame = p * 100 + c_idx * 30
            end_frame = start_frame + num_frames - 1
            sample = UnifiedSample(
                sample_id=f"{performance_id}_elem_{c_idx:02d}",
                dataset_id="synthetic_mock",
                performance_id=performance_id,
                athlete_id=athlete_id,
                element_label=element_label,
                start_frame=start_frame,
                end_frame=end_frame,
                keypoints=frames,
                keypoint_confidence=conf,
                goe=goe,
                source_metadata={
                    "raw_file": f"mock://{performance_id}.npz",
                    "raw_label": element_label,
                    "synthetic_mock": True,
                    "angular_deviation_deg": float(angular_dev_deg),
                    "scale_factor": scale_p,
                },
            )
            sample.validate()
            samples.append(sample)

    return samples


def _split_coverage_score(
    group_split_map: Dict[str, str],
    samples: List[UnifiedSample],
    sample_groups: Dict[str, str],
    class_medians: Dict[str, float],
) -> Tuple[int, int]:
    """Conta quante coppie (split, classe) hanno sia GOE >= mediana sia GOE < mediana."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for s in samples:
        split_name = group_split_map[sample_groups[s.sample_id]]
        buckets.setdefault((split_name, s.element_label), []).append(float(s.goe))

    total_checks = 0
    satisfied = 0
    for split_name in ("train", "val", "test"):
        for cls_name, med in class_medians.items():
            vals = buckets.get((split_name, cls_name), [])
            total_checks += 1
            if not vals:
                continue
            has_hi = any(v >= med for v in vals)
            has_lo = any(v < med for v in vals)
            if has_hi and has_lo:
                satisfied += 1
    return satisfied, total_checks


def create_grouped_splits(
    samples: List[UnifiedSample],
    data_cfg: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Suddivide i sample_id in {'train': [...], 'val': [...], 'test': [...]}
    garantendo che tutti i campioni con lo stesso athlete_id e performance_id
    cadano nello stesso identico split.
    """
    if len(samples) == 0:
        raise ValueError("Impossibile creare split su lista samples vuota.")

    perf_to_group: Dict[str, str] = {}
    sample_groups: Dict[str, str] = {}
    groups_dict: Dict[str, List[UnifiedSample]] = {}

    for s in samples:
        group_id = f"ath_{s.athlete_id}" if s.athlete_id else f"perf_{s.performance_id}"
        prev_group = perf_to_group.get(s.performance_id)
        if prev_group is not None and prev_group != group_id:
            raise ValueError(
                f"Inconsistenza gruppo per performance_id={s.performance_id}: "
                f"{prev_group} vs {group_id}"
            )
        perf_to_group[s.performance_id] = group_id
        sample_groups[s.sample_id] = group_id
        groups_dict.setdefault(group_id, []).append(s)

    unique_groups = sorted(groups_dict.keys())
    n_groups = len(unique_groups)
    if n_groups < 3:
        raise ValueError(
            f"Servono almeno 3 gruppi distinti (atleti/esibizioni) per split train/val/test, "
            f"trovati {n_groups}."
        )

    splits_cfg = data_cfg["splits"]
    val_ratio = float(splits_cfg["val_ratio"])
    test_ratio = float(splits_cfg["test_ratio"])
    seed = int(splits_cfg.get("seed", 42))

    n_val = max(1, int(round(n_groups * val_ratio)))
    n_test = max(1, int(round(n_groups * test_ratio)))
    n_train = n_groups - n_val - n_test
    while n_train < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n_groups - n_val - n_test
    assert n_train >= 1 and n_val >= 1 and n_test >= 1 and (n_train + n_val + n_test == n_groups)

    class_goes: Dict[str, List[float]] = {}
    for s in samples:
        class_goes.setdefault(s.element_label, []).append(float(s.goe))
    class_medians = {k: float(np.median(v)) for k, v in class_goes.items()}

    rng = np.random.default_rng(seed)
    best_score = -1
    best_group_split: Optional[Dict[str, str]] = None

    for _ in range(50):
        perm = rng.permutation(unique_groups).tolist()
        train_g = perm[:n_train]
        val_g = perm[n_train : n_train + n_val]
        test_g = perm[n_train + n_val :]

        candidate_map: Dict[str, str] = {}
        for g in train_g:
            candidate_map[g] = "train"
        for g in val_g:
            candidate_map[g] = "val"
        for g in test_g:
            candidate_map[g] = "test"

        satisfied, total_checks = _split_coverage_score(
            candidate_map, samples, sample_groups, class_medians
        )
        if satisfied > best_score:
            best_score = satisfied
            best_group_split = candidate_map
        if satisfied == total_checks:
            break

    assert best_group_split is not None
    split_dict: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for s in samples:
        split_name = best_group_split[sample_groups[s.sample_id]]
        split_dict[split_name].append(s.sample_id)

    train_ids = set(split_dict["train"])
    val_ids = set(split_dict["val"])
    test_ids = set(split_dict["test"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_perfs = {s.performance_id for s in samples if s.sample_id in train_ids}
    val_perfs = {s.performance_id for s in samples if s.sample_id in val_ids}
    test_perfs = {s.performance_id for s in samples if s.sample_id in test_ids}
    assert train_perfs.isdisjoint(val_perfs), "LEAKAGE RILEVATO: train e val condividono performance_id!"
    assert train_perfs.isdisjoint(test_perfs), "LEAKAGE RILEVATO: train e test condividono performance_id!"
    assert val_perfs.isdisjoint(test_perfs), "LEAKAGE RILEVATO: val e test condividono performance_id!"

    train_aths = {s.athlete_id for s in samples if s.sample_id in train_ids and s.athlete_id is not None}
    val_aths = {s.athlete_id for s in samples if s.sample_id in val_ids and s.athlete_id is not None}
    test_aths = {s.athlete_id for s in samples if s.sample_id in test_ids and s.athlete_id is not None}
    assert train_aths.isdisjoint(val_aths), "LEAKAGE RILEVATO: train e val condividono athlete_id!"
    assert train_aths.isdisjoint(test_aths), "LEAKAGE RILEVATO: train e test condividono athlete_id!"
    assert val_aths.isdisjoint(test_aths), "LEAKAGE RILEVATO: val e test condividono athlete_id!"

    return split_dict
