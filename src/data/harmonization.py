from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class UnifiedSample:
    """Unified schema for a technical element segment (17 3D joints + metadata + GOE)."""

    sample_id: str
    dataset_id: str                           # e.g., "finefs" or "synthetic_mock"
    performance_id: str                       # Performance ID (used for Group Split)
    athlete_id: Optional[str]                 # Athlete ID (if available, otherwise None)
    element_label: str                        # e.g., "Camel Spin", "Sit Spin", "Layback Spin"
    start_frame: int
    end_frame: int
    keypoints: np.ndarray                     # Shape (T_seg, 17, 3), dtype float32
    keypoint_confidence: Optional[np.ndarray] # Shape (T_seg, 17), dtype float32 or None
    goe: float                                # Actual GOE score of the single element
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if fields or shapes violate unified schema constraints."""
        if not isinstance(self.sample_id, str) or len(self.sample_id.strip()) == 0:
            raise ValueError("UnifiedSample.sample_id must be a non-empty string.")
        if not isinstance(self.performance_id, str) or len(self.performance_id.strip()) == 0:
            raise ValueError("UnifiedSample.performance_id must be a non-empty string.")
        if not isinstance(self.dataset_id, str) or len(self.dataset_id.strip()) == 0:
            raise ValueError("UnifiedSample.dataset_id must be a non-empty string.")
        if not isinstance(self.element_label, str) or len(self.element_label.strip()) == 0:
            raise ValueError("UnifiedSample.element_label must be a non-empty string.")

        if not isinstance(self.keypoints, np.ndarray):
            raise ValueError("UnifiedSample.keypoints must be a numpy.ndarray.")
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)

        if (
            self.keypoints.ndim != 3
            or self.keypoints.shape[1] != 17
            or self.keypoints.shape[2] != 3
        ):
            raise ValueError(
                f"UnifiedSample.keypoints must have shape (T_seg, 17, 3), "
                f"got {self.keypoints.shape}."
            )
        if self.keypoints.shape[0] < 1:
            raise ValueError("UnifiedSample.keypoints must contain at least 1 frame (T_seg >= 1).")
        if int(self.end_frame) < int(self.start_frame):
            raise ValueError(
                f"end_frame ({self.end_frame}) cannot be less than "
                f"start_frame ({self.start_frame})."
            )

        if self.keypoint_confidence is not None:
            if not isinstance(self.keypoint_confidence, np.ndarray):
                raise ValueError("UnifiedSample.keypoint_confidence must be numpy.ndarray or None.")
            self.keypoint_confidence = np.asarray(self.keypoint_confidence, dtype=np.float32)
            expected_conf_shape = (self.keypoints.shape[0], 17)
            if self.keypoint_confidence.shape != expected_conf_shape:
                raise ValueError(
                    f"UnifiedSample.keypoint_confidence must have shape {expected_conf_shape}, "
                    f"got {self.keypoint_confidence.shape}."
                )

        if self.goe is None:
            raise ValueError("UnifiedSample.goe cannot be None.")
        goe_val = float(self.goe)
        if not np.isfinite(goe_val):
            raise ValueError(f"UnifiedSample.goe must be finite, got {self.goe}.")
        self.goe = goe_val

        if not isinstance(self.source_metadata, dict):
            raise ValueError("UnifiedSample.source_metadata must be a dictionary.")


def save_unified_samples(samples: List[UnifiedSample], output_path: Union[str, Path]) -> None:
    """Validate each sample and save the list with pickle to output_path."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample.validate()
    with out_path.open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_unified_samples(input_path: Union[str, Path]) -> List[UnifiedSample]:
    """Load the list of UnifiedSample from input_path and re-run validate()."""
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"UnifiedSample file not found: {in_path}")
    with in_path.open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"Invalid content in {in_path}: expected List[UnifiedSample].")
    for sample in samples:
        if not isinstance(sample, UnifiedSample):
            raise ValueError(f"Invalid element in {in_path}: expected UnifiedSample.")
        sample.validate()
    return samples

