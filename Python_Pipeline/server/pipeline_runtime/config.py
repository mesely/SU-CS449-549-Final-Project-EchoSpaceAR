"""Central configuration for the real-time audio pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from load_env import load_env


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str_set(name: str, default: tuple[str, ...]) -> set[str]:
    raw = os.environ.get(name)
    if not raw:
        return set(default)
    return {token.strip() for token in raw.split(",") if token.strip()}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


# Load environment variables before optional integrations initialize.
SERVER_DIR = Path(__file__).resolve().parent.parent
load_env(".env")
load_env(str(SERVER_DIR / ".env"))


# Configure a macOS-safe backend before pyplot is imported anywhere else.
import matplotlib

if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass


try:
    from google import genai

    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

try:
    import tensorflow as tf
    import tensorflow_hub as hub

    TF_AVAILABLE = True
except Exception:
    tf = None
    hub = None
    TF_AVAILABLE = False


BASE_DIR = str(SERVER_DIR)
LOG_DIR = os.environ.get("PIPELINE_LOG_DIR") or os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.environ.get("PIPELINE_MODELS_DIR") or os.path.join(BASE_DIR, "models")

for path in (LOG_DIR, MODELS_DIR):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


REDUCED_LABEL_SET = [
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
    "Silence",
    "other",
]


def resolve_reduced_model_dir() -> str:
    """Find the reduced YAMNet SavedModel directory."""
    env_dir = os.environ.get("REDUCED_MODEL_DIR")
    if env_dir:
        return env_dir

    models_dir_candidate = os.path.join(MODELS_DIR, "reduced_yamnet_savedmodel")
    local_candidate = os.path.join(BASE_DIR, "reduced_yamnet_savedmodel")

    def has_saved_model(path: str) -> bool:
        return os.path.exists(os.path.join(path, "saved_model.pb")) or os.path.exists(
            os.path.join(path, "saved_model.pbtxt")
        )

    return models_dir_candidate if has_saved_model(models_dir_candidate) else local_candidate


def resolve_reduced_labels_json() -> str:
    """Find the label file that matches the reduced YAMNet model."""
    env_path = os.environ.get("REDUCED_LABELS_JSON")
    if env_path:
        return env_path

    models_path = os.path.join(MODELS_DIR, "reduced_labels.json")
    local_path = os.path.join(BASE_DIR, "reduced_labels.json")
    return models_path if os.path.exists(models_path) else local_path


REDUCED_MODEL_DIR = resolve_reduced_model_dir()
REDUCED_LABELS_JSON = resolve_reduced_labels_json()

USE_REDUCED = True
CLASSIFIER_ENABLED = TF_AVAILABLE


DATA_COLLECTION_MODE = True
WIDE_CSV_PATH = os.path.join(LOG_DIR, "classification_probs.csv")
DECISION_CSV_PATH = os.path.join(LOG_DIR, "decision_log.csv")
TOPK_OVERLAY = 5
DECISION_TOPK = 5
PRINT_ALL_TO_CONSOLE = False


PLOT_UPDATE_INTERVAL_S = _env_float("PLOT_UPDATE_INTERVAL_S", 0.1)
CLASSIFY_WINDOW_S = _env_float("CLASSIFY_WINDOW_S", 1.0)
CLASSIFY_HOP_S = _env_float("CLASSIFY_HOP_S", 0.5)


# mono + stereo: mono baseline'i bozma, stereo gelirse side-channel'i yasat.
STEREO_SIDE_CHANNEL_ENABLED = _env_bool("STEREO_SIDE_CHANNEL_ENABLED", True)
SPATIAL_CENTER_ILD_DB = _env_float("SPATIAL_CENTER_ILD_DB", 1.5)
SPATIAL_DIRECTION_MIN_ILD_DB = _env_float("SPATIAL_DIRECTION_MIN_ILD_DB", 3.0)
SPATIAL_MAX_ILD_DB = _env_float("SPATIAL_MAX_ILD_DB", 12.0)
SPATIAL_MAX_GCC_DELAY_S = _env_float("SPATIAL_MAX_GCC_DELAY_S", 0.0015)
SPATIAL_CENTER_CORRELATION = _env_float("SPATIAL_CENTER_CORRELATION", 0.85)


STT_ENABLED = True
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"
STT_WINDOW_S = 5.0
STT_HOP_S = 1.0
STT_MIN_TEXT_LEN = 1
STT_CSV_PATH = os.path.join(LOG_DIR, "transcription_log.csv")


SPEECH_GATE_ENABLED = True
SPEECH_LABEL = "speech"
MODEL_SILENCE_LABEL = "Silence"
SPEECH_ON_THRESH = _env_float("SPEECH_ON_THRESH", 0.40)
SPEECH_OFF_THRESH = _env_float("SPEECH_OFF_THRESH", 0.28)
SPEECH_MIN_ON_S = _env_float("SPEECH_MIN_ON_S", 0.40)
SPEECH_MIN_OFF_S = _env_float("SPEECH_MIN_OFF_S", 0.40)
SPEECH_ENERGY_MARGIN_DB = _env_float("SPEECH_ENERGY_MARGIN_DB", 4.0)
WHISPER_LANGUAGE = _env_str("WHISPER_LANGUAGE", "")
REQUIRE_TOP_IS_SPEECH = False

PREROLL_S = 1.5
MAX_SESSION_S = 5.0
DECODE_HOP_S = 1.5


LLM_ENABLED = True
LLM_DRY_RUN = False
GEMINI_MODEL_NAME = "gemini-2.5-flash"
LLM_CSV_PATH = os.path.join(LOG_DIR, "llm_events.csv")
LLM_JSONL_PATH = os.path.join(LOG_DIR, "llm_events.jsonl")
LLM_MAX_EVENTS_PER_WINDOW = 40
LLM_MAX_STT_CHARS = 600


IMPORTANT_SOUND_LABELS = {
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
    "door_knock",
    "glass_break",
    "explosion_gunshot",
}
IMPORTANT_SOUND_THRESH = 0.55
SOUND_LLM_WINDOW_S = 5.0
SOUND_LLM_COOLDOWN_S = 10.0
GLOBAL_LLM_COOLDOWN_S = 2.0


# EMA + hysteresis + OOD gate: ham p(c|z) skorlarini HUD kararina tasiyan katman.
DECISION_EMA_LAMBDA = _env_float("DECISION_EMA_LAMBDA", 0.65)
DECISION_ENTER_THRESH = _env_float("DECISION_ENTER_THRESH", 0.42)
DECISION_EXIT_THRESH = _env_float("DECISION_EXIT_THRESH", 0.30)
DECISION_MIN_HOLD_S = _env_float("DECISION_MIN_HOLD_S", 1.0)
DECISION_SILENCE_MARGIN_DB = _env_float("DECISION_SILENCE_MARGIN_DB", 4.5)
DECISION_IDLE_CONF_THRESH = _env_float("DECISION_IDLE_CONF_THRESH", 0.32)
DECISION_AWARENESS_CONF_THRESH = _env_float("DECISION_AWARENESS_CONF_THRESH", 0.18)
DECISION_OOD_ENTROPY_THRESH = _env_float("DECISION_OOD_ENTROPY_THRESH", 0.86)
DECISION_OOD_MIN_TOP_PROB = _env_float("DECISION_OOD_MIN_TOP_PROB", 0.30)
DECISION_OOD_MIN_MARGIN = _env_float("DECISION_OOD_MIN_MARGIN", 0.05)
DECISION_PRIORITY_SWITCH_MARGIN = _env_float("DECISION_PRIORITY_SWITCH_MARGIN", 0.08)
DECISION_NOISE_FLOOR_INIT_DBFS = _env_float("DECISION_NOISE_FLOOR_INIT_DBFS", -72.0)


IDLE_LABELS = _env_str_set(
    "IDLE_LABELS",
    (
        "traffic_road",
        "wind_rain",
        "engine_motion",
        "crowd",
        "Silence",
    ),
)
NON_ACTIONABLE_LABELS = _env_str_set(
    "NON_ACTIONABLE_LABELS",
    (
        "speech",
        "music",
        "dog",
        "cat",
        "bird",
        "car_bus_truck",
        "rail",
        "aircraft",
        "other",
    ),
)
ACTIONABLE_PRIORITY_MAP = {
    "sirens": "critical",
    "explosion_gunshot": "critical",
    "vehicle_horn": "high",
    "glass_break": "high",
    "alarms_buzzer": "high",
    "phone_ring": "high",
    "door_knock": "medium",
}
ACTIONABLE_LABELS = set(ACTIONABLE_PRIORITY_MAP)
CRITICAL_LABELS = {label for label, priority in ACTIONABLE_PRIORITY_MAP.items() if priority == "critical"}


DEBUG_STT = False
