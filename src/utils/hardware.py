import warnings
from typing import Any, Dict

import torch


def get_compute_device(prefer_gpu: bool = True) -> torch.device:
    """Restituisce torch.device('cuda') se disponibile e funzionante, altrimenti torch.device('cpu')."""
    if prefer_gpu and torch.cuda.is_available():
        try:
            a = torch.ones((2, 2), device="cuda")
            b = torch.ones((2, 2), device="cuda")
            _ = a @ b
            return torch.device("cuda")
        except Exception as exc:
            warnings.warn(
                f"CUDA disponibile ma micro-test GPU fallito ({exc}); fallback su CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
    return torch.device("cpu")


def get_hardware_profile() -> Dict[str, Any]:
    """Restituisce un dizionario serializzabile JSON con i dettagli CPU/GPU/VRAM/CUDA."""
    cuda_ok = bool(torch.cuda.is_available())
    device_name = "CPU"
    total_vram_mb = 0.0
    cuda_version = None

    if cuda_ok:
        try:
            cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
            device_name = str(torch.cuda.get_device_name(0))
            total_vram_mb = round(
                float(torch.cuda.get_device_properties(0).total_memory) / (1024**2),
                2,
            )
        except Exception as exc:
            warnings.warn(
                f"Impossibile leggere proprietà GPU CUDA ({exc}); report fallback CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            cuda_ok = False
            cuda_version = None
            device_name = "CPU"
            total_vram_mb = 0.0

    return {
        "pytorch_version": str(torch.__version__),
        "cuda_available": cuda_ok,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "total_vram_mb": total_vram_mb,
    }


def get_vram_usage_mb() -> Dict[str, float]:
    """Restituisce {'allocated_mb': ..., 'reserved_mb': ..., 'max_allocated_mb': ...} (0.0 se su CPU)."""
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }
    scale = float(1024**2)
    return {
        "allocated_mb": round(float(torch.cuda.memory_allocated(0)) / scale, 4),
        "reserved_mb": round(float(torch.cuda.memory_reserved(0)) / scale, 4),
        "max_allocated_mb": round(float(torch.cuda.max_memory_allocated(0)) / scale, 4),
    }

