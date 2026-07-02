"""Configuration helpers for the PANNs pipeline."""

from __future__ import annotations

import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


SERVER_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("PIPELINE_MODELS_DIR") or (SERVER_DIR / "models"))
LOG_DIR = Path(os.environ.get("PIPELINE_LOG_DIR") or (SERVER_DIR / "logs"))

PANNS_MODEL_PATH = Path(os.environ.get("PANNS_MODEL_PATH") or (MODELS_DIR / "panns_cnn14_semantic.pt"))
PANNS_LABELS_JSON = Path(os.environ.get("PANNS_LABELS_JSON") or (MODELS_DIR / "semantic_labels.json"))

PANNS_TARGET_SAMPLE_RATE = _env_int("PANNS_TARGET_SAMPLE_RATE", 32000)
PANNS_CLIP_SECONDS = _env_float("PANNS_CLIP_SECONDS", 10.0)
PANNS_CLIP_HOP_SECONDS = _env_float("PANNS_CLIP_HOP_SECONDS", 5.0)
PANNS_N_FFT = _env_int("PANNS_N_FFT", 1024)
PANNS_HOP_LENGTH = _env_int("PANNS_HOP_LENGTH", 320)
PANNS_MEL_BINS = _env_int("PANNS_MEL_BINS", 64)
PANNS_FMIN = _env_float("PANNS_FMIN", 50.0)
PANNS_FMAX = _env_float("PANNS_FMAX", 14000.0)
PANNS_DROPOUT = _env_float("PANNS_DROPOUT", 0.2)
PANNS_DEFAULT_THRESHOLD = _env_float("PANNS_DEFAULT_THRESHOLD", 0.5)

DEFAULT_SEMANTIC_LABELS = [
    "speech",
    "crowd",
    "music",
    "dog",
    "cat",
    "bird",
    "vehicle_horn",
    "traffic_road",
    "car_bus_truck",
    "sirens",
    "rail",
    "aircraft",
    "engine_motion",
    "alarms_buzzer",
    "phone_ring",
    "wind_rain",
    "door_knock",
    "glass_break",
    "explosion_gunshot",
]


def load_semantic_labels(path: str | os.PathLike[str] | None = None) -> list[str]:
    """Load the semantic label order expected by the runtime decision layer."""
    target_path = Path(path) if path is not None else PANNS_LABELS_JSON
    if target_path.exists():
        import json

        with target_path.open("r", encoding="utf-8") as handle:
            return list(json.load(handle))
    return list(DEFAULT_SEMANTIC_LABELS)
