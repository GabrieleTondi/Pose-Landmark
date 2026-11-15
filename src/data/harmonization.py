from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class UnifiedSample:
    """Schema unificato per un segmento di elemento tecnico (17 joint 3D + metadati + GOE)."""

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
        if not isinstance(self.sample_id, str) or len(self.sample_id.strip()) == 0:
            raise ValueError("UnifiedSample.sample_id deve essere una stringa non vuota.")
        if not isinstance(self.performance_id, str) or len(self.performance_id.strip()) == 0:
            raise ValueError("UnifiedSample.performance_id deve essere una stringa non vuota.")
        if not isinstance(self.dataset_id, str) or len(self.dataset_id.strip()) == 0:
            raise ValueError("UnifiedSample.dataset_id deve essere una stringa non vuota.")
        if not isinstance(self.element_label, str) or len(self.element_label.strip()) == 0:
            raise ValueError("UnifiedSample.element_label deve essere una stringa non vuota.")

        if not isinstance(self.keypoints, np.ndarray):
            raise ValueError("UnifiedSample.keypoints deve essere un numpy.ndarray.")
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)

        if (
            self.keypoints.ndim != 3
            or self.keypoints.shape[1] != 17
            or self.keypoints.shape[2] != 3
        ):
            raise ValueError(
                f"UnifiedSample.keypoints deve avere shape (T_seg, 17, 3), "
                f"trovata {self.keypoints.shape}."
            )
        if self.keypoints.shape[0] < 1:
            raise ValueError("UnifiedSample.keypoints deve contenere almeno 1 frame (T_seg >= 1).")
        if int(self.end_frame) < int(self.start_frame):
            raise ValueError(
                f"end_frame ({self.end_frame}) non può essere inferiore a "
                f"start_frame ({self.start_frame})."
            )

        if self.keypoint_confidence is not None:
            if not isinstance(self.keypoint_confidence, np.ndarray):
                raise ValueError("UnifiedSample.keypoint_confidence deve essere numpy.ndarray o None.")
            self.keypoint_confidence = np.asarray(self.keypoint_confidence, dtype=np.float32)
            expected_conf_shape = (self.keypoints.shape[0], 17)
            if self.keypoint_confidence.shape != expected_conf_shape:
                raise ValueError(
                    f"UnifiedSample.keypoint_confidence deve avere shape {expected_conf_shape}, "
                    f"trovata {self.keypoint_confidence.shape}."
                )

        if self.goe is None:
            raise ValueError("UnifiedSample.goe non può essere None.")
        goe_val = float(self.goe)
        if not np.isfinite(goe_val):
            raise ValueError(f"UnifiedSample.goe deve essere finito, trovato {self.goe}.")
        self.goe = goe_val

        if not isinstance(self.source_metadata, dict):
            raise ValueError("UnifiedSample.source_metadata deve essere un dizionario.")


def save_unified_samples(samples: List[UnifiedSample], output_path: Union[str, Path]) -> None:
    """Valida ogni campione e salva la lista con pickle in output_path."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample.validate()
    with out_path.open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_unified_samples(input_path: Union[str, Path]) -> List[UnifiedSample]:
    """Carica la lista di UnifiedSample da input_path e riesegue validate()."""
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"File UnifiedSample non trovato: {in_path}")
    with in_path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"Contenuto non valido in {in_path}: attesa List[UnifiedSample].")
    for sample in samples:
        if not isinstance(sample, UnifiedSample):
            raise ValueError(f"Elemento non valido in {in_path}: atteso UnifiedSample.")
        sample.validate()
    return samples

