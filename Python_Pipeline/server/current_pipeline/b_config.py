"""Central configuration and small shared helpers for the real-time pipeline."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_env(path: str = ".env") -> None:
    """Populate os.environ with simple KEY=VALUE pairs from a .env file."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
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


def resample_linear(samples: np.ndarray, input_sr: float, output_sr: int) -> np.ndarray:
    """Linearly resample a mono waveform to the target sample rate."""
    if samples.size == 0 or input_sr == output_sr:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / float(input_sr)
    source_times = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_count = int(round(duration * output_sr))
    target_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32, copy=False)


def unix_to_local_iso(timestamp_unix: float) -> str:
    """Format a Unix timestamp in local time with millisecond precision."""
    return datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).astimezone().isoformat(
        timespec="milliseconds"
    )


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
    # "Silence" removed — energy-based detection in decision layer
    # "other" removed — no garbage-collector class; unlabelled YAMNet classes
    #                    contribute to nothing rather than inflating a catch-all
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

    models_candidate = os.path.join(MODELS_DIR, "semantic_labels.json")
    local_candidate = os.path.join(os.path.dirname(__file__), "semantic_labels.json")
    if not os.path.exists(models_candidate) and not os.path.exists(local_candidate):
        models_candidate = os.path.join(MODELS_DIR, "reduced_labels.json")
        local_candidate = os.path.join(os.path.dirname(__file__), "reduced_labels.json")
    return models_candidate if os.path.exists(models_candidate) else local_candidate


def resolve_spectral_model_path() -> str:
    """Find the lightweight spectral backbone artifact."""
    env_path = os.environ.get("SPECTRAL_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "spectral_backbone.joblib")


def resolve_learned_head_model_path() -> str:
    """Find the learnable semantic-head artifact."""
    env_path = os.environ.get("LEARNED_HEAD_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "yamnet_learned_head.joblib")


def resolve_beme_head_model_path() -> str:
    """Find the BEME semantic-head artifact."""
    env_path = os.environ.get("BEME_HEAD_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "yamnet_beme_head.joblib")


def resolve_beme_stereo_head_model_path() -> str:
    """Find the stereo-aware BEME semantic-head artifact."""
    env_path = os.environ.get("BEME_STEREO_HEAD_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "yamnet_beme_stereo_head.joblib")


def resolve_pairwise_refiner_model_path() -> str:
    """Find the confusion-aware pairwise relabel artifact."""
    env_path = os.environ.get("PAIRWISE_REFINER_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "semantic_pairwise_refiner.joblib")


def resolve_panns_model_path() -> str:
    """Find the exported PANNs semantic checkpoint."""
    env_path = os.environ.get("PANNS_MODEL_PATH")
    if env_path:
        return env_path
    return os.path.join(MODELS_DIR, "panns_cnn14_semantic.pt")


YAMNET_MODEL_DIR = resolve_yamnet_model_dir()
SEMANTIC_LABELS_JSON = resolve_semantic_labels_json()
SPECTRAL_MODEL_PATH = resolve_spectral_model_path()
LEARNED_HEAD_MODEL_PATH = resolve_learned_head_model_path()
BEME_HEAD_MODEL_PATH = resolve_beme_head_model_path()
BEME_STEREO_HEAD_MODEL_PATH = resolve_beme_stereo_head_model_path()
PAIRWISE_REFINER_MODEL_PATH = resolve_pairwise_refiner_model_path()
PANNS_MODEL_PATH = resolve_panns_model_path()

# Runtime classifier contract:
# - "semantic": 19-class semantic YAMNet head used by the current evaluation/runtime path
# - "learned": full YAMNet embeddings + learnable semantic residual head
# - "beme": YAMNet embeddings + semantic projection features routed through AutoBEME
#           optionally switches to a stereo-aware AutoBEME artifact when stereo
#           frames are available
# - "full": full TF-Hub YAMNet projected back into the 19-class semantic space
# - "spectral": NumPy/SciPy + sklearn spectral fallback backbone
# - "panns": separate PANNs-style semantic backbone under panns_pipeline/
# - "hybrid": fuse all available semantic backbones
CLASSIFIER_MODE = _env_str("CLASSIFIER_MODE", "spectral" if not TF_AVAILABLE else "semantic").strip().lower()
if CLASSIFIER_MODE == "reduced":
    CLASSIFIER_MODE = "semantic"
if CLASSIFIER_MODE not in {"semantic", "learned", "beme", "full", "spectral", "hybrid", "panns"}:
    CLASSIFIER_MODE = "spectral" if not TF_AVAILABLE else "semantic"
USE_SEMANTIC_YAMNET = CLASSIFIER_MODE == "semantic"
CLASSIFIER_ENABLED = (
    TF_AVAILABLE
    or os.path.exists(SPECTRAL_MODEL_PATH)
    or os.path.exists(LEARNED_HEAD_MODEL_PATH)
    or os.path.exists(BEME_HEAD_MODEL_PATH)
    or os.path.exists(BEME_STEREO_HEAD_MODEL_PATH)
    or os.path.exists(PANNS_MODEL_PATH)
)
BEME_STEREO_BLEND_ALPHA = _env_float("BEME_STEREO_BLEND_ALPHA", 0.10)
BEME_STEREO_UPLIFT_ONLY = _env_bool("BEME_STEREO_UPLIFT_ONLY", True)
BEME_STEREO_GATE_ENABLED = _env_bool("BEME_STEREO_GATE_ENABLED", True)
BEME_STEREO_GATE_TARGET_ONLY = _env_bool("BEME_STEREO_GATE_TARGET_ONLY", True)
BEME_STEREO_GATE_TARGET_LABELS = _env_str_set(
    "BEME_STEREO_GATE_TARGET_LABELS",
    ("glass_break", "door_knock", "dog", "music", "speech"),
)
BEME_STEREO_GATE_TOPK = max(1, _env_int("BEME_STEREO_GATE_TOPK", 3))
BEME_STEREO_GATE_TOP1_MAX = _env_float("BEME_STEREO_GATE_TOP1_MAX", 0.52)
BEME_STEREO_GATE_MARGIN_MAX = _env_float("BEME_STEREO_GATE_MARGIN_MAX", 0.12)
BEME_STEREO_GATE_ENTROPY_MIN = _env_float("BEME_STEREO_GATE_ENTROPY_MIN", 0.72)
BEME_STEREO_GATE_MIN_TARGET_PROB = _env_float("BEME_STEREO_GATE_MIN_TARGET_PROB", 0.10)
BEME_STEREO_GATE_MIN_STEREO_DELTA = _env_float("BEME_STEREO_GATE_MIN_STEREO_DELTA", 0.015)
BEME_STEREO_GATE_MIN_DIRECTION_CONF = _env_float("BEME_STEREO_GATE_MIN_DIRECTION_CONF", 0.12)
BEME_STEREO_GATE_MIN_DECORRELATION = _env_float("BEME_STEREO_GATE_MIN_DECORRELATION", 0.10)
BEME_STEREO_GATE_MIN_ILD_DB = _env_float("BEME_STEREO_GATE_MIN_ILD_DB", 0.75)
BEME_STEREO_GATE_MIN_SCORE = _env_float("BEME_STEREO_GATE_MIN_SCORE", 0.38)
HYBRID_SEMANTIC_WEIGHT = _env_float("HYBRID_SEMANTIC_WEIGHT", _env_float("HYBRID_REDUCED_WEIGHT", 0.45))
HYBRID_LEARNED_WEIGHT = _env_float("HYBRID_LEARNED_WEIGHT", 0.60)
HYBRID_FULL_WEIGHT = _env_float("HYBRID_FULL_WEIGHT", 0.20)
HYBRID_SPECTRAL_WEIGHT = _env_float("HYBRID_SPECTRAL_WEIGHT", 0.35)


DATA_COLLECTION_MODE = True
WIDE_CSV_PATH = os.path.join(LOG_DIR, "classification_probs.csv")
DECISION_CSV_PATH = os.path.join(LOG_DIR, "decision_log.csv")
TOPK_OVERLAY = 5
DECISION_TOPK = 5
PRINT_ALL_TO_CONSOLE = False


PLOT_UPDATE_INTERVAL_S = _env_float("PLOT_UPDATE_INTERVAL_S", 0.1)
CLASSIFY_WINDOW_S = _env_float("CLASSIFY_WINDOW_S", 1.0)
CLASSIFY_HOP_S = _env_float("CLASSIFY_HOP_S", 0.5)


# Class-aware temporal frontend:
# fixed 1.0s windows are still the runtime contract, but we inspect shorter
# subwindows inside that second to avoid smearing short events and to prevent
# sustained classes from winning on one unstable spike.
TEMPORAL_FRONTEND_ENABLED = _env_bool("TEMPORAL_FRONTEND_ENABLED", False)
TEMPORAL_FAMILY_BY_LABEL = {
    "alarms_buzzer": "transient",
    "phone_ring": "transient",
    "glass_break": "transient",
    "explosion_gunshot": "transient",
    "door_knock": "transient",
    "wind_rain": "sustained",
    "rail": "sustained",
    "speech": "sustained",
    "music": "sustained",
}
TEMPORAL_FRONTEND_BRANCHES = (
    {"name": "micro", "window_s": 0.20, "hop_s": 0.10},
    {"name": "short", "window_s": 0.35, "hop_s": 0.175},
    {"name": "half", "window_s": 0.50, "hop_s": 0.25},
)
TEMPORAL_TRANSIENT_CONFIG = {
    "alarms_buzzer": {"alpha": 0.30, "margin": 0.012, "min_peak": 0.050, "support_weight": 0.22},
    "phone_ring": {"alpha": 0.28, "margin": 0.012, "min_peak": 0.045, "support_weight": 0.22},
    "glass_break": {"alpha": 0.34, "margin": 0.015, "min_peak": 0.055, "support_weight": 0.18},
    "explosion_gunshot": {"alpha": 0.34, "margin": 0.015, "min_peak": 0.055, "support_weight": 0.18},
    "door_knock": {"alpha": 0.26, "margin": 0.012, "min_peak": 0.045, "support_weight": 0.18},
}
TEMPORAL_SUSTAINED_PULL_ALPHA = _env_float("TEMPORAL_SUSTAINED_PULL_ALPHA", 0.28)
TEMPORAL_SUSTAINED_BOOST_ALPHA = _env_float("TEMPORAL_SUSTAINED_BOOST_ALPHA", 0.10)
TEMPORAL_SUSTAINED_PEAK_MARGIN = _env_float("TEMPORAL_SUSTAINED_PEAK_MARGIN", 0.030)

TEMPORAL_CLIP_AGGREGATION_ENABLED = _env_bool("TEMPORAL_CLIP_AGGREGATION_ENABLED", False)
TEMPORAL_CLIP_MIN_REL_SUPPORT = _env_float("TEMPORAL_CLIP_MIN_REL_SUPPORT", 0.58)
TEMPORAL_CLIP_MIN_ABS_SUPPORT = _env_float("TEMPORAL_CLIP_MIN_ABS_SUPPORT", 0.035)
TEMPORAL_CLIP_TRANSIENT_PEAK_WEIGHT = _env_float("TEMPORAL_CLIP_TRANSIENT_PEAK_WEIGHT", 0.68)
TEMPORAL_CLIP_TRANSIENT_TOPK_WEIGHT = _env_float("TEMPORAL_CLIP_TRANSIENT_TOPK_WEIGHT", 0.22)
TEMPORAL_CLIP_TRANSIENT_RUN_WEIGHT = _env_float("TEMPORAL_CLIP_TRANSIENT_RUN_WEIGHT", 0.10)
TEMPORAL_CLIP_SUSTAINED_PEAK_WEIGHT = _env_float("TEMPORAL_CLIP_SUSTAINED_PEAK_WEIGHT", 0.26)
TEMPORAL_CLIP_SUSTAINED_TOPK_WEIGHT = _env_float("TEMPORAL_CLIP_SUSTAINED_TOPK_WEIGHT", 0.24)
TEMPORAL_CLIP_SUSTAINED_RUN_WEIGHT = _env_float("TEMPORAL_CLIP_SUSTAINED_RUN_WEIGHT", 0.50)


# Multi-resolution classifier front-end:
# keep the 1.0s main window as the semantic anchor, but let short transient
# classes borrow evidence from smaller overlapping windows inside that same
# interval. Sustained classes keep the main-window score as their default so
# they do not become onset-noise detectors.
MULTIRES_FRONTEND_ENABLED = _env_bool("MULTIRES_FRONTEND_ENABLED", False)
MULTIRES_FAMILY_BY_LABEL = {
    "alarms_buzzer": "transient",
    "phone_ring": "transient",
    "glass_break": "transient",
    "explosion_gunshot": "transient",
    "wind_rain": "sustained",
    "rail": "sustained",
    "speech": "sustained",
    "music": "sustained",
}
MULTIRES_BRANCHES = (
    {"name": "micro", "window_s": 0.20, "hop_s": 0.10},
    {"name": "short", "window_s": 0.35, "hop_s": 0.175},
)
MULTIRES_TRANSIENT_CONFIG = {
    "alarms_buzzer": {"alpha": 0.52, "margin": 0.016, "min_peak": 0.050, "support_weight": 0.35},
    "phone_ring": {"alpha": 0.50, "margin": 0.016, "min_peak": 0.045, "support_weight": 0.34},
    "glass_break": {"alpha": 0.58, "margin": 0.020, "min_peak": 0.060, "support_weight": 0.32},
    "explosion_gunshot": {"alpha": 0.60, "margin": 0.022, "min_peak": 0.060, "support_weight": 0.30},
}
MULTIRES_ONSET_ATTACK_S = _env_float("MULTIRES_ONSET_ATTACK_S", 0.08)
MULTIRES_ONSET_RELEASE_S = _env_float("MULTIRES_ONSET_RELEASE_S", 0.24)
MULTIRES_ONSET_ALPHA_BOOST = _env_float("MULTIRES_ONSET_ALPHA_BOOST", 0.30)


# Confusion-aware semantic relabeling:
# pairwise specialists are trained offline on the same semantic label set and
# only intervene when the main head is uncertain inside known confusion pairs.
CONFUSION_PAIR_REFINEMENT_ENABLED = _env_bool("CONFUSION_PAIR_REFINEMENT_ENABLED", True)
RAIL_SPEECH_REFINEMENT_ENABLED = _env_bool("RAIL_SPEECH_REFINEMENT_ENABLED", True)
GLASS_IMPULSE_REFINEMENT_ENABLED = _env_bool("GLASS_IMPULSE_REFINEMENT_ENABLED", True)
RAIL_SPEECH_TRIGGER_MARGIN = _env_float("RAIL_SPEECH_TRIGGER_MARGIN", 0.22)
RAIL_SPEECH_MIN_SPEECH_PROB = _env_float("RAIL_SPEECH_MIN_SPEECH_PROB", 0.05)
RAIL_SPEECH_ALPHA = _env_float("RAIL_SPEECH_ALPHA", 0.68)
RAIL_SPEECH_VOICING_MAX = _env_float("RAIL_SPEECH_VOICING_MAX", 0.60)
RAIL_SPEECH_TONALITY_MAX = _env_float("RAIL_SPEECH_TONALITY_MAX", 0.55)
RAIL_SPEECH_RAIL_BOOST = _env_float("RAIL_SPEECH_RAIL_BOOST", 0.022)
RAIL_SPEECH_MEMORY_REFINEMENT_ENABLED = _env_bool("RAIL_SPEECH_MEMORY_REFINEMENT_ENABLED", False)
RAIL_SPEECH_MEMORY_WINDOWS = int(round(_env_float("RAIL_SPEECH_MEMORY_WINDOWS", 3.0)))
RAIL_SPEECH_MEMORY_MIN_RAIL = _env_float("RAIL_SPEECH_MEMORY_MIN_RAIL", 0.070)
RAIL_SPEECH_MEMORY_MAX_VOICING = _env_float("RAIL_SPEECH_MEMORY_MAX_VOICING", 0.32)
RAIL_SPEECH_MEMORY_MAX_TONALITY = _env_float("RAIL_SPEECH_MEMORY_MAX_TONALITY", 0.46)
RAIL_SPEECH_MEMORY_ALPHA = _env_float("RAIL_SPEECH_MEMORY_ALPHA", 0.42)
RAIL_SPEECH_MEMORY_BOOST = _env_float("RAIL_SPEECH_MEMORY_BOOST", 0.022)
GLASS_IMPULSE_TRIGGER_MARGIN = _env_float("GLASS_IMPULSE_TRIGGER_MARGIN", 0.18)
GLASS_IMPULSE_MIN_COMPETITOR_PROB = _env_float("GLASS_IMPULSE_MIN_COMPETITOR_PROB", 0.05)
GLASS_IMPULSE_MIN_GLASS_PROB = _env_float("GLASS_IMPULSE_MIN_GLASS_PROB", 0.012)
GLASS_IMPULSE_ALPHA = _env_float("GLASS_IMPULSE_ALPHA", 0.58)
GLASS_IMPULSE_MIN_SCORE = _env_float("GLASS_IMPULSE_MIN_SCORE", 0.62)
GLASS_IMPULSE_GLASS_BOOST = _env_float("GLASS_IMPULSE_GLASS_BOOST", 0.015)
GLASS_EVENT_CONFIRM_ENABLED = _env_bool("GLASS_EVENT_CONFIRM_ENABLED", False)
GLASS_EVENT_MEMORY_WINDOWS = int(round(_env_float("GLASS_EVENT_MEMORY_WINDOWS", 2.0)))
GLASS_EVENT_MIN_BURST = _env_float("GLASS_EVENT_MIN_BURST", 0.42)
GLASS_EVENT_DECAY_MIN = _env_float("GLASS_EVENT_DECAY_MIN", 0.08)
GLASS_EVENT_ALPHA = _env_float("GLASS_EVENT_ALPHA", 0.38)
GLASS_EVENT_BOOST = _env_float("GLASS_EVENT_BOOST", 0.018)

# Backward-compatible aliases for existing classifier fusion code.
TRANSIENT_MULTIRES_ENABLED = MULTIRES_FRONTEND_ENABLED
TRANSIENT_MULTIRES_WINDOW_S = float(MULTIRES_BRANCHES[-1]["window_s"])
TRANSIENT_MULTIRES_HOP_S = float(MULTIRES_BRANCHES[-1]["hop_s"])
TRANSIENT_MULTIRES_CONFIG = {
    label: {
        "alpha": float(cfg.get("alpha", 0.5)),
        "margin": float(cfg.get("margin", 0.0)),
        "min_peak": float(cfg.get("min_peak", 0.0)),
    }
    for label, cfg in MULTIRES_TRANSIENT_CONFIG.items()
}


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
MODEL_SILENCE_LABEL = None  # Silence removed as a model class; energy-based only
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
DECISION_IDLE_SUSTAIN_THRESH = _env_float("DECISION_IDLE_SUSTAIN_THRESH", 0.28)
DECISION_AWARENESS_CONF_THRESH = _env_float("DECISION_AWARENESS_CONF_THRESH", 0.18)
DECISION_AWARE_SUSTAIN_THRESH = _env_float("DECISION_AWARE_SUSTAIN_THRESH", 0.10)
DECISION_OOD_ENTROPY_THRESH = _env_float("DECISION_OOD_ENTROPY_THRESH", 0.86)
DECISION_OOD_MIN_TOP_PROB = _env_float("DECISION_OOD_MIN_TOP_PROB", 0.30)
DECISION_OOD_MIN_MARGIN = _env_float("DECISION_OOD_MIN_MARGIN", 0.05)
DECISION_OOD_DEBOUNCE_WINDOWS = max(1, int(_env_float("DECISION_OOD_DEBOUNCE_WINDOWS", 3.0)))
DECISION_PRIORITY_SWITCH_MARGIN = _env_float("DECISION_PRIORITY_SWITCH_MARGIN", 0.08)
DECISION_ACTIONABLE_TOP_MARGIN = _env_float("DECISION_ACTIONABLE_TOP_MARGIN", 0.05)
DECISION_NOISE_FLOOR_INIT_DBFS = _env_float("DECISION_NOISE_FLOOR_INIT_DBFS", -72.0)
ACTIVE_SEMANTIC_LABELS = _env_str_set("ACTIVE_SEMANTIC_LABELS", ())


IDLE_LABELS = _env_str_set(
    "IDLE_LABELS",
    (
        "traffic_road",
        "wind_rain",
        "engine_motion",
        "crowd",
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

# Class-specific enter thresholds:
# Multi-label audio events are often calibrated per class rather than with one
# global decision threshold.  This especially matters for transient classes
# such as glass/knock/gunshot, which tend to produce asymmetric false alerts.
ACTIONABLE_ENTER_THRESHOLDS = {
    "glass_break": 0.60,
    "door_knock": 0.60,
    "explosion_gunshot": 0.52,
    "phone_ring": 0.48,
    "vehicle_horn": 0.55,
}

# Narrow burst-confirm path:
# only rescue short impact alerts when repeated raw evidence exists but the
# semantic top label is a non-actionable/context class or the frame is about to
# be suppressed as OOD.
BURST_CONFIRM_LABELS = _env_str_set(
    "BURST_CONFIRM_LABELS",
    (
        "glass_break",
    ),
)
BURST_CONFIRM_HISTORY_WINDOWS = max(1, int(_env_float("BURST_CONFIRM_HISTORY_WINDOWS", 3.0)))
BURST_CONFIRM_MIN_HITS = max(1, int(_env_float("BURST_CONFIRM_MIN_HITS", 2.0)))
BURST_CONFIRM_MAX_NON_ACTIONABLE_GAP = _env_float("BURST_CONFIRM_MAX_NON_ACTIONABLE_GAP", 0.08)
BURST_CONFIRM_ALLOW_OOD_RESCUE = _env_bool("BURST_CONFIRM_ALLOW_OOD_RESCUE", False)
BURST_CONFIRM_GLASS_BREAK_RAW_PROB_MIN = _env_float("BURST_CONFIRM_GLASS_BREAK_RAW_PROB_MIN", 0.14)
BURST_CONFIRM_GLASS_BREAK_EMA_PROB_MIN = _env_float("BURST_CONFIRM_GLASS_BREAK_EMA_PROB_MIN", 0.20)
BURST_CONFIRM_EXPLOSION_GUNSHOT_RAW_PROB_MIN = _env_float("BURST_CONFIRM_EXPLOSION_GUNSHOT_RAW_PROB_MIN", 0.10)
BURST_CONFIRM_EXPLOSION_GUNSHOT_EMA_PROB_MIN = _env_float("BURST_CONFIRM_EXPLOSION_GUNSHOT_EMA_PROB_MIN", 0.20)
BURST_CONFIRM_RULES = {
    "glass_break": {
        "raw_prob_min": BURST_CONFIRM_GLASS_BREAK_RAW_PROB_MIN,
        "ema_prob_min": BURST_CONFIRM_GLASS_BREAK_EMA_PROB_MIN,
    },
    "explosion_gunshot": {
        "raw_prob_min": BURST_CONFIRM_EXPLOSION_GUNSHOT_RAW_PROB_MIN,
        "ema_prob_min": BURST_CONFIRM_EXPLOSION_GUNSHOT_EMA_PROB_MIN,
    },
}

# Start-only actionable reroutes:
# keep the event actionable, but change the sibling label when raw evidence
# clearly says the current candidate is the wrong family member.
ACTIONABLE_START_REROUTES = {
    "door_knock": [
        {
            "to": "alarms_buzzer",
            "raw_prob_min": 0.008,
            "ema_prob_min": 0.05,
            "history": 1,
            "raw_max_min": 0.008,
        }
    ],
}

# Event-level actionable family arbitration:
# keep semantically similar alert siblings from flapping or starting under the
# wrong label when their short raw history clearly favors one family member.
DEBUG_STT = False
