"""Central configuration for the real-time audio pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from current_pipeline.b_config import load_env


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


SEMANTIC_LABEL_SET = [
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


def resolve_yamnet_model_dir() -> str:
    """Find the semantic YAMNet SavedModel directory."""
    env_dir = os.environ.get("YAMNET_MODEL_DIR") or os.environ.get("REDUCED_MODEL_DIR")
    if env_dir:
        return env_dir

    models_dir_candidate = os.path.join(MODELS_DIR, "yamnet_savedmodel")
    local_candidate = os.path.join(BASE_DIR, "yamnet_savedmodel")

    def has_saved_model(path: str) -> bool:
        return os.path.exists(os.path.join(path, "saved_model.pb")) or os.path.exists(
            os.path.join(path, "saved_model.pbtxt")
        )

    legacy_models_dir_candidate = os.path.join(MODELS_DIR, "reduced_yamnet_savedmodel")
    legacy_local_candidate = os.path.join(BASE_DIR, "reduced_yamnet_savedmodel")
    if has_saved_model(models_dir_candidate):
        return models_dir_candidate
    if has_saved_model(local_candidate):
        return local_candidate
    if has_saved_model(legacy_models_dir_candidate):
        return legacy_models_dir_candidate
    return legacy_local_candidate


def resolve_semantic_labels_json() -> str:
    """Find the label file that matches the semantic YAMNet model."""
    env_path = os.environ.get("YAMNET_LABELS_JSON") or os.environ.get("REDUCED_LABELS_JSON")
    if env_path:
        return env_path

    models_path = os.path.join(MODELS_DIR, "semantic_labels.json")
    local_path = os.path.join(BASE_DIR, "semantic_labels.json")
    if not os.path.exists(models_path) and not os.path.exists(local_path):
        models_path = os.path.join(MODELS_DIR, "reduced_labels.json")
        local_path = os.path.join(BASE_DIR, "reduced_labels.json")
    return models_path if os.path.exists(models_path) else local_path


YAMNET_MODEL_DIR = resolve_yamnet_model_dir()
SEMANTIC_LABELS_JSON = resolve_semantic_labels_json()

USE_SEMANTIC_YAMNET = True
CLASSIFIER_ENABLED = TF_AVAILABLE


DATA_COLLECTION_MODE = True
WIDE_CSV_PATH = os.path.join(LOG_DIR, "classification_probs.csv")
TOPK_OVERLAY = 5
PRINT_ALL_TO_CONSOLE = False


STT_ENABLED = True
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"
STT_WINDOW_S = 5.0
STT_HOP_S = 1.0
STT_MIN_TEXT_LEN = 1
STT_CSV_PATH = os.path.join(LOG_DIR, "transcription_log.csv")


SPEECH_GATE_ENABLED = True
SPEECH_LABEL = "speech"
SPEECH_ON_THRESH = 0.40
SPEECH_OFF_THRESH = 0.30
SPEECH_MIN_ON_S = 0.50
SPEECH_MIN_OFF_S = 0.00
WHISPER_LANGUAGE = "en"
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


DEBUG_STT = False
