"""Build a corrected YAMNet-based SavedModel with fixed label mapping.

Changes vs the earlier head builder:
  - alarms_buzzer: added Doorbell, Ding-dong, Chime, Microwave oven
  - door_knock: removed Doorbell, Ding-dong (moved to alarms_buzzer)
  - Silence: removed as a model class (energy-based detection in decision layer)
  - other/REST: removed (no garbage-collector class; unlabelled YAMNet classes
    contribute to nothing rather than inflating a single catch-all)
  - temporal pooling: class-dependent instead of a single mean/LSE for every label
    - transient alerts use top-k LSE over the strongest windows
    - sustained classes use softer full-window LSE
  - output: {"probs": [K], "embeddings": [512]} — embeddings exported for
    future learnable-head training without re-running the backbone
  - L1 normalisation NOT applied here; classification.py normalisation removed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except Exception:  # pragma: no cover - optional dependency
    tf = None
    hub = None

from current_pipeline.b_config import (
    BEME_HEAD_MODEL_PATH,
    BEME_STEREO_HEAD_MODEL_PATH,
    MODELS_DIR,
    PAIRWISE_REFINER_MODEL_PATH,
    SEMANTIC_LABEL_SET,
    YAMNET_MODEL_DIR,
    resample_linear,
)
from current_pipeline.c_audio_core import (
    downmix_to_mono,
    ensure_frame_major,
    extract_spectral_feature_vector,
    resample_frame_major_audio,
    summarize_spatial_audio,
)
from current_pipeline.i_beme import make_autobeme, predict_autobeme_proba


# ---------------------------------------------------------------------------
# Label map — 19 semantic classes, no Silence, no other/REST
# Key fix: Doorbell/Ding-dong/Microwave oven → alarms_buzzer
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, list[str | tuple[str, float]]] = {
    "speech": [
        "Speech",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Whispering",
    ],
    "crowd": [
        "Chatter",
        "Crowd",
        "Hubbub, speech noise, speech babble",
        "Cheering",
        "Applause",
        "Children playing",
    ],
    "music": [
        "Music",
        "Musical instrument",
        "Song",
        "Background music",
        "Theme music",
        "Soundtrack music",
        "Jingle (music)",
        "Vocal music",
        "A capella",
    ],
    "dog": ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"],
    "cat": ["Cat", "Purr", "Meow", "Hiss", "Caterwaul"],
    "bird": [
        "Bird",
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
        "Squawk",
        "Pigeon, dove",
        "Coo",
        "Owl",
        "Hoot",
        "Bird flight, flapping wings",
        "Crow",
        "Caw",
    ],
    "vehicle_horn": [
        "Vehicle horn, car horn, honking",
        "Toot",
        "Air horn, truck horn",
        "Foghorn",
    ],
    "traffic_road": [
        "Motor vehicle (road)",
        "Traffic noise, roadway noise",
        "Car passing by",
        "Race car, auto racing",
        "Skidding",
        "Tire squeal",
    ],
    "car_bus_truck": ["Car", "Bus", "Truck", "Ice cream truck, ice cream van"],
    "sirens": [
        "Siren",
        "Civil defense siren",
        "Emergency vehicle",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Car alarm",
    ],
    "rail": [
        "Rail transport",
        "Train",
        "Train whistle",
        "Train horn",
        "Railroad car, train wagon",
        "Train wheels squealing",
        "Subway, metro, underground",
    ],
    "aircraft": [
        "Aircraft",
        "Aircraft engine",
        "Jet engine",
        "Propeller, airscrew",
        "Helicopter",
        "Fixed-wing aircraft, airplane",
    ],
    "engine_motion": [
        "Engine",
        "Light engine (high frequency)",
        "Medium engine (mid frequency)",
        "Heavy engine (low frequency)",
        "Engine knocking",
        "Engine starting",
        "Idling",
        "Accelerating, revving, vroom",
    ],
    # FIXED: Doorbell / Ding-dong / Chime / Microwave oven moved here from door_knock / REST
    "alarms_buzzer": [
        ("Alarm", 0.9),
        ("Alarm clock", 1.0),
        ("Buzzer", 1.2),
        ("Smoke detector, smoke alarm", 1.0),
        ("Fire alarm", 1.0),
        ("Doorbell", 3.0),
        ("Ding-dong", 2.6),
        ("Chime", 2.0),
        ("Microwave oven", 3.2),
    ],
    "phone_ring": [
        ("Telephone bell ringing", 2.8),
        ("Ringtone", 3.0),
        ("Telephone", 0.8),
        "Telephone dialing, DTMF",
        "Dial tone",
        "Busy signal",
    ],
    "wind_rain": [
        "Wind",
        "Rustling leaves",
        "Wind noise (microphone)",
        "Thunderstorm",
        "Thunder",
        "Rain",
        "Raindrop",
        "Rain on surface",
    ],
    # FIXED: Doorbell / Ding-dong removed; "Door" kept (generic door event)
    "door_knock": [
        ("Door", 0.45),
        ("Knock", 2.0),
        ("Tap", 1.3),
        ("Slam", 1.6),
        "Sliding door",
        "Cupboard open or close",
        "Drawer open or close",
    ],
    "glass_break": [
        ("Glass", 0.40),
        ("Shatter", 2.8),
        ("Smash, crash", 1.8),
        ("Breaking", 1.8),
        ("Chink, clink", 0.55),
        ("Crack", 1.6),
    ],
    "explosion_gunshot": [
        ("Explosion", 1.8),
        ("Gunshot, gunfire", 2.5),
        "Machine gun",
        "Fusillade",
        "Artillery fire",
        "Cap gun",
        ("Fireworks", 2.0),
        "Firecracker",
        "Burst, pop",
        "Eruption",
        ("Boom", 1.5),
    ],
}

DEFAULT_LSE_BETA = 6.0
DEFAULT_POOL_MODE = "lse"
EXPORT_DIR = "yamnet_savedmodel"
LABELS_JSON_MODELS = os.path.join(MODELS_DIR, "semantic_labels.json")
POOLING_CONFIG_JSON_MODELS = os.path.join(MODELS_DIR, "yamnet_pooling_config.json")
SPECTRAL_MODEL_PATH = os.path.join(MODELS_DIR, "spectral_backbone.joblib")
LEARNED_HEAD_MODEL_PATH = os.path.join(MODELS_DIR, "yamnet_learned_head.joblib")
BEME_SEMANTIC_HEAD_PATH = BEME_HEAD_MODEL_PATH
BEME_STEREO_SEMANTIC_HEAD_PATH = BEME_STEREO_HEAD_MODEL_PATH
PAIRWISE_REFINER_OUTPUT_PATH = PAIRWISE_REFINER_MODEL_PATH
DEFAULT_SONYC_ROOT = str(Path(__file__).resolve().parents[2] / "Sony_Data")
CALIBRATION_ALPHA_GRID = (0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
CALIBRATION_THRESHOLD_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)

# Class-aware temporal pooling. The goal is to stop short transient alerts from
# being washed out by the same pooling recipe used for long ambient textures.
TEMPORAL_POOL_CONFIG: dict[str, dict[str, float | int | str]] = {
    "alarms_buzzer": {"mode": "topk_lse", "beta": 12.0, "top_k": 3},
    "glass_break": {"mode": "topk_lse", "beta": 12.0, "top_k": 3},
    "phone_ring": {"mode": "topk_lse", "beta": 10.0, "top_k": 4},
    "dog": {"mode": "topk_lse", "beta": 8.0, "top_k": 3},
    "wind_rain": {"mode": "lse", "beta": 4.0, "top_k": 0},
    "speech": {"mode": "lse", "beta": 4.0, "top_k": 0},
    "rail": {"mode": "lse", "beta": 3.5, "top_k": 0},
    "music": {"mode": "lse", "beta": 5.0, "top_k": 0},
}

PAIRWISE_REFINER_CONFIG: list[dict[str, object]] = [
    {"name": "alarms_buzzer__phone_ring", "labels": ("alarms_buzzer", "phone_ring"), "blend_alpha": 0.72, "trigger_margin": 0.17, "min_top_prob": 0.10, "min_confidence": 0.58},
    {"name": "glass_break__explosion_gunshot", "labels": ("glass_break", "explosion_gunshot"), "blend_alpha": 0.72, "trigger_margin": 0.16, "min_top_prob": 0.09, "min_confidence": 0.58},
    {"name": "door_knock__alarms_buzzer", "labels": ("door_knock", "alarms_buzzer"), "blend_alpha": 0.68, "trigger_margin": 0.18, "min_top_prob": 0.08, "min_confidence": 0.56},
    {"name": "rail__wind_rain", "labels": ("rail", "wind_rain"), "blend_alpha": 0.62, "trigger_margin": 0.16, "min_top_prob": 0.08, "min_confidence": 0.56},
    {"name": "glass_break__music", "labels": ("glass_break", "music"), "blend_alpha": 0.54, "trigger_margin": 0.14, "min_top_prob": 0.08, "min_confidence": 0.60},
    {"name": "rail__speech", "labels": ("rail", "speech"), "blend_alpha": 0.50, "trigger_margin": 0.14, "min_top_prob": 0.08, "min_confidence": 0.60},
]

SONYC_LABEL_ALIASES = {
    "glass_break": ("Shatter", "Crack", "Breaking"),
    "alarms_buzzer": (
        "Alarm",
        "Alarm_clock",
        "Buzzer",
        "Doorbell",
        "Ding-dong",
        "Chime",
        "Microwave_oven",
        "Smoke_detector_and_smoke_alarm",
        "Fire_alarm",
    ),
    "phone_ring": ("Ringtone", "Telephone", "Telephone_bell_ringing", "Dial_tone", "Busy_signal"),
    "door_knock": ("Knock", "Tap", "Slam", "Door", "Cupboard_open_or_close", "Drawer_open_or_close", "Sliding_door"),
    "explosion_gunshot": ("Fireworks", "Gunshot_and_gunfire", "Boom", "Explosion", "Burst_pop"),
    "cat": ("Meow", "Cat"),
    "dog": ("Bark", "Dog"),
    "rail": ("Subway_and_metro_and_underground", "Rail_transport", "Train", "Train_horn", "Train_whistle"),
    "wind_rain": ("Wind", "Rain", "Thunder", "Thunderstorm", "Rain_on_surface"),
    "speech": (
        "Speech",
        "Conversation",
        "Narration_monologue",
        "Male_speech_and_man_speaking",
        "Female_speech_and_woman_speaking",
        "Child_speech_and_kid_speaking",
        "Whispering",
        "Speech_synthesizer",
    ),
    "music": (
        "Music",
        "Musical_instrument",
        "Song",
        "Background_music",
        "Theme_music",
        "Soundtrack_music",
        "Vocal_music",
        "Jingle_music",
    ),
}
SONYC_LABEL_MAP = {
    "_".join(str(alias).lower().replace("&", " and ").replace("-", " ").replace("/", " ").replace(",", " ").replace("(", " ").replace(")", " ").replace(".", " ").replace(":", " ").replace("'", "").split()): label
    for label, aliases in SONYC_LABEL_ALIASES.items()
    for alias in aliases
}


def normalize_sonyc_label(raw_label: str) -> str:
    normalized = str(raw_label or "").strip().lower().replace("&", " and ")
    for token in ("(", ")", ",", "-", ".", "/", ":"):
        normalized = normalized.replace(token, " ")
    normalized = normalized.replace("'", "")
    return "_".join(part for part in normalized.split() if part)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_source_audio_frames(path: str) -> tuple[np.ndarray, int]:
    sample_rate, samples = wavfile.read(path)
    audio = np.asarray(samples)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_value)
    else:
        audio = audio.astype(np.float32, copy=False)
    return ensure_frame_major(audio, channels_hint=(2 if audio.ndim == 2 else 1)), int(sample_rate)


def load_source_audio(path: str) -> tuple[np.ndarray, int]:
    frames, sample_rate = load_source_audio_frames(path)
    return downmix_to_mono(frames), int(sample_rate)


def crop_or_pad(samples: np.ndarray, start_sample: int, target_samples: int) -> np.ndarray:
    if start_sample >= samples.size:
        return np.zeros(target_samples, dtype=np.float32)
    segment = samples[max(0, start_sample) : max(0, start_sample) + target_samples]
    if segment.size >= target_samples:
        return segment[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: segment.size] = segment.astype(np.float32, copy=False)
    return padded


def iter_annotation_paths(sonyc_root: str, split_names: list[str]) -> list[str]:
    annotation_root = Path(sonyc_root) / "extracted" / "SONYC_FSD_SED.annotations"
    paths: list[str] = []
    for split_name in split_names:
        split_dir = annotation_root / split_name
        paths.extend(str(path) for path in sorted(split_dir.glob("*.jams")))
    return paths


def load_vocab(sonyc_root: str) -> list[str]:
    return list(load_json(os.path.join(sonyc_root, "vocab.json")))


def iter_training_examples(sonyc_root: str, split_names: list[str], max_per_label: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    source_root = Path(sonyc_root) / "extracted" / "SONYC_FSD_SED.source"
    vocab = load_vocab(sonyc_root)
    annotation_paths = iter_annotation_paths(sonyc_root, split_names)
    rng = random.Random(seed)
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in SEMANTIC_LABEL_SET}
    active_labels = set(SONYC_LABEL_ALIASES.keys())

    for annotation_path in annotation_paths:
        if active_labels and all(len(by_label[label]) >= max_per_label for label in active_labels):
            break
        data = load_json(annotation_path)
        events = data.get("annotations", [{}])[0].get("data", [])
        for item in events:
            value = item.get("value", {})
            if str(value.get("role", "")) != "foreground":
                continue
            raw_label = str(value.get("label", ""))
            if raw_label.isdigit():
                raw_index = int(raw_label)
                if 0 <= raw_index < len(vocab):
                    raw_label = vocab[raw_index]
            semantic_label = SONYC_LABEL_MAP.get(normalize_sonyc_label(raw_label))
            if semantic_label not in by_label or len(by_label[semantic_label]) >= max_per_label:
                continue

            source_rel_path = str(value.get("source_file", "")).replace("\\", "/")
            marker_index = max(source_rel_path.find("sonyc_background/"), source_rel_path.find("fsd50k_foreground/"))
            if marker_index < 0:
                continue
            source_path = source_root / source_rel_path[marker_index:]
            if not source_path.exists():
                continue

            samples, sample_rate = load_source_audio(str(source_path))
            start_sample = int(round(float(value.get("source_time", 0.0) or 0.0) * sample_rate))
            duration = float(value.get("event_duration", item.get("duration", 0.0)) or 0.0)
            target_samples = max(1, int(round(duration * sample_rate)))
            snippet = crop_or_pad(samples, start_sample, target_samples)
            if sample_rate != 16000:
                snippet = resample_linear(snippet, sample_rate, 16000)
            if snippet.size < 1600:
                continue
            by_label[semantic_label].append(extract_spectral_feature_vector(snippet, sample_rate=16000))
            if len(by_label[semantic_label]) >= max_per_label and semantic_label in active_labels:
                active_labels.remove(semantic_label)

    labels = [label for label in SEMANTIC_LABEL_SET if by_label[label]]
    for label in labels:
        rng.shuffle(by_label[label])
        by_label[label] = by_label[label][:max_per_label]

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label_index, label in enumerate(labels):
        for feature in by_label[label]:
            target = np.zeros(len(labels), dtype=np.float32)
            target[label_index] = 1.0
            features.append(feature)
            targets.append(target)
    if not features:
        raise RuntimeError("Spectral backbone icin hic egitim ornegi toplanamadi.")

    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.vstack(targets).astype(np.float32, copy=False),
        labels,
        {label: len(by_label[label]) for label in labels},
    )


def build_spectral_backbone(sonyc_root: str, split_names: list[str], max_per_label: int, seed: int, output_path: str) -> None:
    features, targets, labels, counts = iter_training_examples(
        sonyc_root=sonyc_root,
        split_names=split_names,
        max_per_label=max_per_label,
        seed=seed,
    )
    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_std = np.std(features, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (features - feature_mean) / feature_std

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1200,
            solver="lbfgs",
        )
    )
    classifier.fit(normalized, targets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": labels,
            "train_counts": counts,
            "source": "SONYC_FSD_SED source-event spectral backbone",
        },
        output_path,
    )
    print(f"[OK] Spectral backbone -> {output_path}")
    for label in labels:
        print(f"[OK] {label:<20} {counts[label]}")


def collect_feature_bank(
    sonyc_root: str,
    split_names: list[str],
    max_per_label: int,
    seed: int,
    target_labels: set[str],
) -> dict[str, list[np.ndarray]]:
    source_root = Path(sonyc_root) / "extracted" / "SONYC_FSD_SED.source"
    vocab = load_vocab(sonyc_root)
    annotation_paths = iter_annotation_paths(sonyc_root, split_names)
    rng = random.Random(seed)
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in target_labels}

    for annotation_path in annotation_paths:
        if all(len(by_label[label]) >= max_per_label for label in target_labels):
            break
        data = load_json(annotation_path)
        events = data.get("annotations", [{}])[0].get("data", [])
        for item in events:
            value = item.get("value", {})
            if str(value.get("role", "")) != "foreground":
                continue
            raw_label = str(value.get("label", ""))
            if raw_label.isdigit():
                raw_index = int(raw_label)
                if 0 <= raw_index < len(vocab):
                    raw_label = vocab[raw_index]
            semantic_label = SONYC_LABEL_MAP.get(normalize_sonyc_label(raw_label))
            if semantic_label not in by_label or len(by_label[semantic_label]) >= max_per_label:
                continue

            source_rel_path = str(value.get("source_file", "")).replace("\\", "/")
            marker_index = max(source_rel_path.find("sonyc_background/"), source_rel_path.find("fsd50k_foreground/"))
            if marker_index < 0:
                continue
            source_path = source_root / source_rel_path[marker_index:]
            if not source_path.exists():
                continue

            samples, sample_rate = load_source_audio(str(source_path))
            start_sample = int(round(float(value.get("source_time", 0.0) or 0.0) * sample_rate))
            duration = float(value.get("event_duration", item.get("duration", 0.0)) or 0.0)
            target_samples = max(1, int(round(duration * sample_rate)))
            snippet = crop_or_pad(samples, start_sample, target_samples)
            if sample_rate != 16000:
                snippet = resample_linear(snippet, sample_rate, 16000)
            if snippet.size < 1600:
                continue
            by_label[semantic_label].append(extract_spectral_feature_vector(snippet, sample_rate=16000))

    for label in target_labels:
        rng.shuffle(by_label[label])
        by_label[label] = by_label[label][:max_per_label]
    return by_label


def collect_feature_bank_from_manifest(
    manifest_path: str,
    max_per_label: int,
    seed: int,
    target_labels: set[str],
) -> dict[str, list[np.ndarray]]:
    rng = random.Random(seed)
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in target_labels}
    if not os.path.exists(manifest_path):
        return by_label

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_labels = str(row.get("labels", "") or row.get("label", "")).strip()
            if not raw_labels:
                continue
            labels = [item.strip() for item in raw_labels.split(";") if item.strip()]
            matched = [label for label in labels if label in target_labels and len(by_label[label]) < max_per_label]
            if not matched:
                continue
            audio_path = str(row.get("audio_path", "")).strip()
            if not audio_path or not os.path.exists(audio_path):
                continue
            samples, sample_rate = load_source_audio(audio_path)
            if sample_rate != 16000:
                samples = resample_linear(samples, sample_rate, 16000)
            feature = extract_spectral_feature_vector(samples, sample_rate=16000)
            for label in matched:
                by_label[label].append(feature)

    for label in target_labels:
        rng.shuffle(by_label[label])
        by_label[label] = by_label[label][:max_per_label]
    return by_label


def build_pairwise_refiner(sonyc_root: str, split_names: list[str], max_per_label: int, seed: int, output_path: str) -> None:
    target_labels = {label for item in PAIRWISE_REFINER_CONFIG for label in item["labels"]}
    manifest_candidate = str(Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "sonyc_fsd_sed_manifest.csv")
    feature_bank = collect_feature_bank_from_manifest(
        manifest_path=manifest_candidate,
        max_per_label=max_per_label,
        seed=seed,
        target_labels=target_labels,
    )
    if sum(len(items) for items in feature_bank.values()) == 0:
        feature_bank = collect_feature_bank(
            sonyc_root=sonyc_root,
            split_names=split_names,
            max_per_label=max_per_label,
            seed=seed,
            target_labels=target_labels,
        )

    pair_models: list[dict[str, object]] = []
    for item in PAIRWISE_REFINER_CONFIG:
        left_label, right_label = item["labels"]
        left_features = list(feature_bank.get(left_label, []))
        right_features = list(feature_bank.get(right_label, []))
        if not left_features or not right_features:
            print(f"[WARN] pair refiner skipped: {left_label} vs {right_label} (missing data)")
            continue

        balanced_count = max(min(len(left_features), len(right_features)), 1)
        rng = random.Random(f"{seed}:{left_label}:{right_label}")
        if len(left_features) > balanced_count:
            rng.shuffle(left_features)
            left_features = left_features[:balanced_count]
        if len(right_features) > balanced_count:
            rng.shuffle(right_features)
            right_features = right_features[:balanced_count]

        features = np.vstack(left_features + right_features).astype(np.float32, copy=False)
        targets = np.concatenate(
            [
                np.zeros(len(left_features), dtype=np.int32),
                np.ones(len(right_features), dtype=np.int32),
            ]
        )

        feature_mean = np.mean(features, axis=0).astype(np.float32)
        feature_std = np.std(features, axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        normalized = (features - feature_mean) / feature_std

        classifier = LogisticRegression(
            C=2.4,
            class_weight="balanced",
            max_iter=1400,
            solver="lbfgs",
        )
        classifier.fit(normalized, targets)
        pair_models.append(
            {
                "name": str(item["name"]),
                "labels": [left_label, right_label],
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "classifier": classifier,
                "blend_alpha": float(item["blend_alpha"]),
                "trigger_margin": float(item["trigger_margin"]),
                "min_top_prob": float(item["min_top_prob"]),
                "min_confidence": float(item["min_confidence"]),
                "sample_count": int(features.shape[0]),
            }
        )
        print(f"[OK] pair refiner {left_label:<20} vs {right_label:<20} samples={features.shape[0]}")

    if not pair_models:
        raise RuntimeError("No pairwise refiners could be trained.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"pairs": pair_models, "feature_source": "SONYC_FSD_SED source-event hard negatives"}, output_path)
    print(f"[OK] Pairwise refiner -> {output_path}")


def _load_yamnet_backbone_and_projection() -> tuple[object, np.ndarray, list[str]]:
    if tf is None or hub is None:
        raise RuntimeError("TensorFlow/TensorFlow Hub not available for learned semantic-head training.")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    yamnet_names = _read_class_map(class_map_path)
    projection, labels = build_projection_matrix(yamnet_names, LABEL_MAP)
    return yamnet, projection.astype(np.float32, copy=False), labels


def _extract_yamnet_projection_and_embedding(
    yamnet: object,
    projection: np.ndarray,
    mono_audio_16k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    waveform = tf.convert_to_tensor(np.asarray(mono_audio_16k, dtype=np.float32), dtype=tf.float32)
    scores, embeddings, _spectrogram = yamnet(waveform)
    pooled_scores = tf.reduce_mean(scores, axis=0).numpy().astype(np.float32, copy=False)
    semantic_projection = pooled_scores @ projection
    embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32, copy=False)
    return np.clip(semantic_projection, 0.0, 1.0).astype(np.float32, copy=False), embedding_mean


def _extract_yamnet_semantic_features(yamnet: object, projection: np.ndarray, mono_audio_16k: np.ndarray) -> np.ndarray:
    semantic_projection, embedding_mean = _extract_yamnet_projection_and_embedding(yamnet, projection, mono_audio_16k)
    return np.concatenate([embedding_mean, semantic_projection]).astype(np.float32, copy=False)


def _extract_yamnet_stereo_semantic_features(
    yamnet: object,
    projection: np.ndarray,
    frames_audio_16k: np.ndarray,
    feature_mode: str = "full",
) -> np.ndarray:
    frames = ensure_frame_major(frames_audio_16k, channels_hint=2)
    mono_audio_16k = downmix_to_mono(frames)
    projected_baseline, embedding_mean = _extract_yamnet_projection_and_embedding(yamnet, projection, mono_audio_16k)
    base_feature = np.concatenate([embedding_mean, projected_baseline]).astype(np.float32, copy=False)
    compact_base_feature = projected_baseline.astype(np.float32, copy=False)

    def _compose_feature(
        left_projection: np.ndarray | None = None,
        right_projection: np.ndarray | None = None,
        stereo_delta: np.ndarray | None = None,
        stereo_stats: np.ndarray | None = None,
    ) -> np.ndarray:
        left_projection = (
            np.asarray(left_projection, dtype=np.float32)
            if left_projection is not None
            else np.zeros_like(projected_baseline, dtype=np.float32)
        )
        right_projection = (
            np.asarray(right_projection, dtype=np.float32)
            if right_projection is not None
            else np.zeros_like(projected_baseline, dtype=np.float32)
        )
        stereo_delta = (
            np.asarray(stereo_delta, dtype=np.float32)
            if stereo_delta is not None
            else np.zeros_like(projected_baseline, dtype=np.float32)
        )
        stereo_stats = (
            np.asarray(stereo_stats, dtype=np.float32)
            if stereo_stats is not None
            else np.zeros(8, dtype=np.float32)
        )
        mode = str(feature_mode or "full").strip().lower()
        if mode == "delta_stats":
            return np.concatenate([base_feature, stereo_delta, stereo_stats]).astype(np.float32, copy=False)
        if mode == "stats_only":
            return np.concatenate([base_feature, stereo_stats]).astype(np.float32, copy=False)
        if mode == "compact":
            return np.concatenate(
                [compact_base_feature, left_projection, right_projection, stereo_delta, stereo_stats]
            ).astype(np.float32, copy=False)
        if mode == "compact_delta_stats":
            return np.concatenate([compact_base_feature, stereo_delta, stereo_stats]).astype(np.float32, copy=False)
        if mode == "compact_stats":
            return np.concatenate([compact_base_feature, stereo_stats]).astype(np.float32, copy=False)
        return np.concatenate([base_feature, left_projection, right_projection, stereo_delta, stereo_stats]).astype(
            np.float32,
            copy=False,
        )

    if frames.shape[0] == 0 or frames.shape[1] < 2:
        return _compose_feature()

    left_projection, _left_embedding = _extract_yamnet_projection_and_embedding(yamnet, projection, frames[:, 0])
    right_projection, _right_embedding = _extract_yamnet_projection_and_embedding(yamnet, projection, frames[:, 1])
    stereo_delta = np.abs(left_projection - right_projection).astype(np.float32, copy=False)
    spatial = summarize_spatial_audio(frames, 16000.0)
    stereo_stats = np.array(
        [
            float(spatial.ild_db),
            float(spatial.ipd_rad),
            float(spatial.gcc_delay_s),
            float(spatial.correlation),
            float(spatial.left_rms),
            float(spatial.right_rms),
            float(abs(spatial.left_rms - spatial.right_rms)),
            float((spatial.left_rms + spatial.right_rms) * 0.5),
        ],
        dtype=np.float32,
    )
    return _compose_feature(
        left_projection=left_projection,
        right_projection=right_projection,
        stereo_delta=stereo_delta,
        stereo_stats=stereo_stats,
    )


def _format_label_cell(labels: list[str]) -> str:
    return ";".join(label for label in labels if label)


def _collect_rendered_manifest_rows(
    manifest_path: str,
    stereo_aware: bool = False,
    stereo_feature_mode: str = "full",
) -> tuple[list[dict[str, object]], list[str], Counter[str]]:
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"Rendered training manifest not found: {manifest_path}")

    yamnet, projection, labels = _load_yamnet_backbone_and_projection()
    label_set = set(labels)
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            raw_labels = str(row.get("labels", "") or row.get("label", "")).strip()
            if not raw_labels:
                continue
            positive_labels = [item.strip() for item in raw_labels.split(";") if item.strip() in label_set]
            if not positive_labels:
                continue
            audio_path = str(row.get("audio_path", "")).strip()
            if not audio_path or not os.path.exists(audio_path):
                continue
            if stereo_aware:
                channels = int(str(row.get("channels", "1") or "1"))
                if channels < 2:
                    continue
                frames, sample_rate = load_source_audio_frames(audio_path)
                if sample_rate != 16000:
                    frames = resample_frame_major_audio(frames, sample_rate, 16000)
                if frames.shape[0] < 1600:
                    continue
                feature = _extract_yamnet_stereo_semantic_features(
                    yamnet,
                    projection,
                    frames,
                    feature_mode=stereo_feature_mode,
                )
            else:
                samples, sample_rate = load_source_audio(audio_path)
                if sample_rate != 16000:
                    samples = resample_linear(samples, sample_rate, 16000)
                if samples.size < 1600:
                    continue
                feature = _extract_yamnet_semantic_features(yamnet, projection, samples)
            unique_positive = sorted(set(positive_labels))
            rows.append(
                {
                    "audio_path": audio_path,
                    "feature": feature,
                    "positive_labels": unique_positive,
                }
            )
            counts.update(unique_positive)
            if row_index == 1 or row_index % 100 == 0:
                print(f"[RENDERED_HEAD] loaded {row_index} manifest rows...")

    return rows, labels, counts


def _build_rendered_dataset_from_rows(
    rows: list[dict[str, object]],
    labels: list[str],
    counts: Counter[str],
    seed: int,
    min_examples_per_label: int,
    allowed_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int], list[str]]:
    if allowed_labels is None:
        trained_labels = [label for label in labels if counts[label] >= int(min_examples_per_label)]
    else:
        allowed = set(allowed_labels)
        trained_labels = [label for label in labels if label in allowed]

    if not trained_labels:
        raise RuntimeError("Rendered manifest did not provide enough positive coverage for any semantic label.")

    label_to_index = {label: index for index, label in enumerate(trained_labels)}
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    audio_paths: list[str] = []
    shuffled_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_rows)
    for item in shuffled_rows:
        positive = [label for label in item["positive_labels"] if label in label_to_index]
        if not positive:
            continue
        target = np.zeros(len(trained_labels), dtype=np.float32)
        for label in positive:
            target[label_to_index[label]] = 1.0
        features.append(np.asarray(item["feature"], dtype=np.float32))
        targets.append(target)
        audio_paths.append(str(item["audio_path"]))

    if not features:
        raise RuntimeError("Rendered manifest head dataset ended up empty after label filtering.")

    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.vstack(targets).astype(np.float32, copy=False),
        trained_labels,
        {label: int(counts[label]) for label in labels if counts[label] > 0},
        audio_paths,
    )


def _predict_semantic_savedmodel_probabilities(audio_paths: list[str], labels: list[str]) -> np.ndarray:
    if tf is None:
        raise RuntimeError("TensorFlow is required for rendered-manifest calibration.")
    if not (
        os.path.exists(os.path.join(YAMNET_MODEL_DIR, "saved_model.pb"))
        or os.path.exists(os.path.join(YAMNET_MODEL_DIR, "saved_model.pbtxt"))
    ):
        raise RuntimeError(f"Semantic SavedModel not found: {YAMNET_MODEL_DIR}")

    model = tf.saved_model.load(YAMNET_MODEL_DIR)
    infer = model.__call__.get_concrete_function()
    probabilities: list[np.ndarray] = []
    total = len(audio_paths)
    for index, audio_path in enumerate(audio_paths, start=1):
        samples, sample_rate = load_source_audio(audio_path)
        if sample_rate != 16000:
            samples = resample_linear(samples, sample_rate, 16000)
        output = infer(waveform_16k=tf.convert_to_tensor(samples, dtype=tf.float32))
        probs = np.asarray(output["probs"].numpy(), dtype=np.float32).reshape(-1)
        if probs.shape[0] != len(labels):
            raise RuntimeError(
                "Semantic SavedModel output width does not match expected label count. "
                f"expected={len(labels)} got={probs.shape[0]}"
            )
        probabilities.append(probs)
        if index == 1 or index % 50 == 0 or index == total:
            print(f"[RENDERED_HEAD] calibrated semantic pass {index}/{total} complete.")
    return np.vstack(probabilities).astype(np.float32, copy=False)


def _calibrate_rendered_beme_fusion(
    classifier: object,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_audio_paths: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    labels: list[str],
    trained_labels: list[str],
    default_blend_alpha: float,
) -> tuple[dict[str, float], dict[str, bool], dict[str, float], dict[str, dict[str, float]]]:
    if calibration_features.shape[0] == 0:
        return {}, {}, {}, {}

    label_count = len(labels)
    projected_probs = calibration_features[:, -label_count:].astype(np.float32, copy=False)
    normalized = ((calibration_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)).astype(
        np.float32,
        copy=False,
    )
    learned_probs = predict_autobeme_proba(classifier, normalized)
    learned_probs = np.asarray(learned_probs, dtype=np.float32)
    if learned_probs.ndim == 1:
        learned_probs = learned_probs.reshape(-1, 1)
    semantic_probs = _predict_semantic_savedmodel_probabilities(calibration_audio_paths, labels)

    trained_index_by_label = {label: idx for idx, label in enumerate(trained_labels)}
    output_index_by_label = {label: idx for idx, label in enumerate(labels)}
    per_label_blend_alpha: dict[str, float] = {}
    per_label_uplift_only: dict[str, bool] = {}
    per_label_decision_threshold: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}

    for label in trained_labels:
        trained_index = trained_index_by_label[label]
        output_index = output_index_by_label[label]
        y_true = calibration_targets[:, trained_index].astype(np.int32)
        positives = int(np.sum(y_true))
        negatives = int(y_true.shape[0] - positives)
        if positives == 0 or negatives == 0:
            continue

        semantic = semantic_probs[:, output_index]
        projected = projected_probs[:, output_index]
        learned = learned_probs[:, trained_index]
        delta = learned - projected

        best_score = -1.0
        best_alpha = float(default_blend_alpha)
        best_uplift_only = True
        best_threshold = 0.50

        for uplift_only in (True, False):
            for alpha in CALIBRATION_ALPHA_GRID:
                fused = np.asarray(semantic, dtype=np.float32).copy()
                if alpha > 0.0:
                    if uplift_only:
                        positive_mask = delta > 0.0
                        fused[positive_mask] = fused[positive_mask] + (float(alpha) * delta[positive_mask])
                    else:
                        fused = fused + (float(alpha) * delta)
                fused = np.clip(fused, 0.0, 1.0)
                for threshold in CALIBRATION_THRESHOLD_GRID:
                    score = float(f1_score(y_true, fused >= float(threshold), zero_division=0))
                    is_better = score > (best_score + 1e-9)
                    same_score = abs(score - best_score) <= 1e-9
                    if same_score:
                        is_more_conservative = (
                            (best_uplift_only is False and uplift_only is True)
                            or (
                                bool(uplift_only) == bool(best_uplift_only)
                                and (
                                    float(alpha) < (best_alpha - 1e-9)
                                    or (
                                        abs(float(alpha) - best_alpha) <= 1e-9
                                        and abs(float(threshold) - 0.50) < abs(best_threshold - 0.50)
                                    )
                                )
                            )
                        )
                        is_better = is_better or is_more_conservative
                    if not is_better:
                        continue
                    best_score = score
                    best_alpha = float(alpha)
                    best_uplift_only = bool(uplift_only)
                    best_threshold = float(threshold)

        baseline_f1 = float(f1_score(y_true, semantic >= 0.50, zero_division=0))
        per_label_blend_alpha[label] = best_alpha
        per_label_uplift_only[label] = best_uplift_only
        per_label_decision_threshold[label] = best_threshold
        summary[label] = {
            "baseline_f1": baseline_f1,
            "calibrated_f1": float(best_score),
            "positives": float(positives),
            "negatives": float(negatives),
            "alpha": best_alpha,
            "uplift_only": float(1.0 if best_uplift_only else 0.0),
            "threshold": best_threshold,
        }
        print(
            "[RENDERED_HEAD] calibration "
            f"{label:<20} baseline_f1={baseline_f1:.4f} -> calibrated_f1={best_score:.4f} "
            f"(alpha={best_alpha:.2f}, uplift_only={best_uplift_only}, threshold={best_threshold:.2f})"
        )

    return per_label_blend_alpha, per_label_uplift_only, per_label_decision_threshold, summary


def build_rendered_training_manifest(
    sonyc_root: str,
    split_names: list[str],
    per_label: int,
    seed: int,
    render_dir: str,
    manifest_path: str,
) -> tuple[str, dict[str, int]]:
    from current_pipeline.g_evaluate import choose_sonyc_scenes, parse_sonyc_scene, render_sonyc_scene

    source_root = os.path.join(sonyc_root, "extracted", "SONYC_FSD_SED.source")
    if not os.path.isdir(source_root):
        raise RuntimeError(f"SONYC source directory not found: {source_root}")

    vocab = load_vocab(sonyc_root)
    annotation_paths = iter_annotation_paths(sonyc_root, split_names)
    scene_specs = []
    for annotation_index, annotation_path in enumerate(annotation_paths, start=1):
        scene = parse_sonyc_scene(annotation_path, vocab)
        if scene is not None:
            scene_specs.append(scene)
        if annotation_index == 1 or annotation_index % 10000 == 0:
            print(f"[RENDERED_HEAD] parsed {annotation_index}/{len(annotation_paths)} annotations...")

    selected_scenes, coverage = choose_sonyc_scenes(
        scenes=scene_specs,
        source_root=source_root,
        per_label=per_label,
        seed=seed,
    )
    if not selected_scenes:
        raise RuntimeError("No train/val SONYC scenes could be selected for rendered-manifest head training.")

    render_root = Path(render_dir).resolve()
    mono_dir = render_root / "mono"
    stereo_dir = render_root / "stereo"
    mono_dir.mkdir(parents=True, exist_ok=True)
    stereo_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, np.ndarray] = {}

    manifest_output = Path(manifest_path).resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "audio_path",
                "label",
                "labels",
                "source_label",
                "dataset",
                "environment",
                "sensor",
                "channels",
            ],
        )
        writer.writeheader()

        total_renders = len(selected_scenes) * 2
        render_index = 0
        for scene in selected_scenes:
            for stereo_flag, dataset_name, output_dir, channels in (
                (False, "SONYC TrainVal Rendered Mono", mono_dir, "1"),
                (True, "SONYC TrainVal Rendered Stereo", stereo_dir, "2"),
            ):
                render_index += 1
                output_path = output_dir / f"{scene.clip_id}.wav"
                render_sonyc_scene(
                    scene=scene,
                    source_root=source_root,
                    output_path=str(output_path),
                    stereo=stereo_flag,
                    seed=seed,
                    cache=source_cache,
                )
                writer.writerow(
                    {
                        "clip_id": f"{scene.clip_id}_{'stereo' if stereo_flag else 'mono'}",
                        "audio_path": str(output_path),
                        "label": scene.positive_labels[0],
                        "labels": _format_label_cell(scene.positive_labels),
                        "source_label": _format_label_cell(scene.source_labels),
                        "dataset": dataset_name,
                        "environment": "urban_synthetic_trainval",
                        "sensor": "sonyc_fsd_sed",
                        "channels": channels,
                    }
                )
                if render_index == 1 or render_index % 25 == 0 or render_index == total_renders:
                    print(f"[RENDERED_HEAD] render {render_index}/{total_renders} complete.")

    return str(manifest_output), coverage


def build_rendered_manifest_head_dataset(
    manifest_path: str,
    seed: int,
    min_examples_per_label: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]:
    rows, labels, counts = _collect_rendered_manifest_rows(manifest_path)
    features, targets, trained_labels, count_map, _audio_paths = _build_rendered_dataset_from_rows(
        rows=rows,
        labels=labels,
        counts=counts,
        seed=seed,
        min_examples_per_label=min_examples_per_label,
    )
    return features, targets, labels, trained_labels, count_map


def build_rendered_manifest_stereo_head_dataset(
    manifest_path: str,
    seed: int,
    min_examples_per_label: int,
    stereo_feature_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]:
    rows, labels, counts = _collect_rendered_manifest_rows(
        manifest_path,
        stereo_aware=True,
        stereo_feature_mode=stereo_feature_mode,
    )
    features, targets, trained_labels, count_map, _audio_paths = _build_rendered_dataset_from_rows(
        rows=rows,
        labels=labels,
        counts=counts,
        seed=seed,
        min_examples_per_label=min_examples_per_label,
    )
    return features, targets, labels, trained_labels, count_map


def build_learned_head_dataset(
    sonyc_root: str,
    split_names: list[str],
    max_per_label: int,
    seed: int,
    max_annotation_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]:
    source_root = Path(sonyc_root) / "extracted" / "SONYC_FSD_SED.source"
    vocab = load_vocab(sonyc_root)
    annotation_paths = iter_annotation_paths(sonyc_root, split_names)
    rng = random.Random(seed)
    yamnet, projection, labels = _load_yamnet_backbone_and_projection()
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    active_labels = set(SONYC_LABEL_ALIASES.keys())

    for annotation_index, annotation_path in enumerate(annotation_paths, start=1):
        if max_annotation_files is not None and annotation_index > max_annotation_files:
            break
        if active_labels and all(len(by_label[label]) >= max_per_label for label in active_labels):
            break
        if annotation_index == 1 or annotation_index % 2000 == 0:
            current_counts = {label: len(by_label[label]) for label in active_labels if len(by_label[label]) > 0}
            print(
                f"[LEARNED_HEAD] scanned {annotation_index}/{len(annotation_paths)} annotations; "
                f"active={sorted(active_labels)} counts={current_counts}"
            )
        data = load_json(annotation_path)
        events = data.get("annotations", [{}])[0].get("data", [])
        for item in events:
            value = item.get("value", {})
            if str(value.get("role", "")) != "foreground":
                continue
            raw_label = str(value.get("label", ""))
            if raw_label.isdigit():
                raw_index = int(raw_label)
                if 0 <= raw_index < len(vocab):
                    raw_label = vocab[raw_index]
            semantic_label = SONYC_LABEL_MAP.get(normalize_sonyc_label(raw_label))
            if semantic_label not in by_label or len(by_label[semantic_label]) >= max_per_label:
                continue

            source_rel_path = str(value.get("source_file", "")).replace("\\", "/")
            marker_index = max(source_rel_path.find("sonyc_background/"), source_rel_path.find("fsd50k_foreground/"))
            if marker_index < 0:
                continue
            source_path = source_root / source_rel_path[marker_index:]
            if not source_path.exists():
                continue

            samples, sample_rate = load_source_audio(str(source_path))
            start_sample = int(round(float(value.get("source_time", 0.0) or 0.0) * sample_rate))
            duration = float(value.get("event_duration", item.get("duration", 0.0)) or 0.0)
            target_samples = max(1, int(round(duration * sample_rate)))
            snippet = crop_or_pad(samples, start_sample, target_samples)
            if sample_rate != 16000:
                snippet = resample_linear(snippet, sample_rate, 16000)
            if snippet.size < 1600:
                continue

            by_label[semantic_label].append(_extract_yamnet_semantic_features(yamnet, projection, snippet))
            if len(by_label[semantic_label]) >= max_per_label and semantic_label in active_labels:
                active_labels.remove(semantic_label)

    trained_labels = [label for label in labels if len(by_label[label]) >= max(8, min(16, max_per_label // 3 if max_per_label > 0 else 8))]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label in trained_labels:
        rng.shuffle(by_label[label])
        by_label[label] = by_label[label][:max_per_label]

    for label_index, label in enumerate(trained_labels):
        for feature in by_label[label]:
            target = np.zeros(len(trained_labels), dtype=np.float32)
            target[label_index] = 1.0
            features.append(feature)
            targets.append(target)

    if not features:
        raise RuntimeError("Learned semantic head icin yeterli egitim ornegi toplanamadi.")

    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.vstack(targets).astype(np.float32, copy=False),
        labels,
        trained_labels,
        {label: len(by_label[label]) for label in labels if by_label[label]},
    )


def build_learned_semantic_head(
    sonyc_root: str,
    split_names: list[str],
    max_per_label: int,
    seed: int,
    output_path: str,
    blend_alpha: float,
    max_annotation_files: int | None = None,
) -> None:
    features, targets, labels, trained_labels, counts = build_learned_head_dataset(
        sonyc_root=sonyc_root,
        split_names=split_names,
        max_per_label=max_per_label,
        seed=seed,
        max_annotation_files=max_annotation_files,
    )
    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_std = np.std(features, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (features - feature_mean) / feature_std

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.5,
            class_weight="balanced",
            max_iter=1400,
            solver="lbfgs",
        )
    )
    classifier.fit(normalized, targets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": labels,
            "trained_labels": trained_labels,
            "train_counts": counts,
            "blend_alpha": float(blend_alpha),
            "uplift_only": True,
            "source": "YAMNet embeddings + semantic projection residual head",
        },
        output_path,
    )
    print(f"[OK] Learned semantic head -> {output_path}")
    print(f"[OK] trained labels: {trained_labels}")
    for label in trained_labels:
        print(f"[OK] {label:<20} {counts.get(label, 0)}")


def build_beme_semantic_head(
    sonyc_root: str,
    split_names: list[str],
    max_per_label: int,
    seed: int,
    output_path: str,
    blend_alpha: float,
    beme_mode: str,
    n_funds: int,
    max_annotation_files: int | None = None,
) -> None:
    features, targets, labels, trained_labels, counts = build_learned_head_dataset(
        sonyc_root=sonyc_root,
        split_names=split_names,
        max_per_label=max_per_label,
        seed=seed,
        max_annotation_files=max_annotation_files,
    )
    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_std = np.std(features, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = ((features - feature_mean) / feature_std).astype(np.float32, copy=False)

    print(
        "[*] Training BEME semantic head "
        f"(mode={beme_mode}, n_funds={n_funds}, samples={normalized.shape[0]}, classes={targets.shape[1]})"
    )
    classifier = make_autobeme(mode=beme_mode, n_funds=n_funds)
    classifier.fit(normalized, targets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": labels,
            "trained_labels": trained_labels,
            "train_counts": counts,
            "blend_alpha": float(blend_alpha),
            "uplift_only": False,
            "beme_mode": str(beme_mode),
            "n_funds": int(n_funds),
            "source": "YAMNet embeddings + semantic projection BEME head",
        },
        output_path,
    )
    print(f"[OK] BEME semantic head -> {output_path}")
    print(f"[OK] trained labels: {trained_labels}")
    for label in trained_labels:
        print(f"[OK] {label:<20} {counts.get(label, 0)}")


def build_rendered_manifest_beme_semantic_head(
    sonyc_root: str,
    split_names: list[str],
    render_per_label: int,
    seed: int,
    output_path: str,
    blend_alpha: float,
    beme_mode: str,
    n_funds: int,
    train_manifest_path: str,
    render_dir: str,
    min_examples_per_label: int,
    calibration_split_names: list[str] | None = None,
    calibration_manifest_path: str | None = None,
    calibration_render_dir: str | None = None,
) -> None:
    manifest_path, coverage = build_rendered_training_manifest(
        sonyc_root=sonyc_root,
        split_names=split_names,
        per_label=render_per_label,
        seed=seed,
        render_dir=render_dir,
        manifest_path=train_manifest_path,
    )
    print(f"[RENDERED_HEAD] training manifest -> {manifest_path}")
    print(f"[RENDERED_HEAD] coverage -> {coverage}")

    features, targets, labels, trained_labels, counts = build_rendered_manifest_head_dataset(
        manifest_path=manifest_path,
        seed=seed,
        min_examples_per_label=min_examples_per_label,
    )
    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_std = np.std(features, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = ((features - feature_mean) / feature_std).astype(np.float32, copy=False)

    print(
        "[*] Training rendered-manifest BEME semantic head "
        f"(mode={beme_mode}, n_funds={n_funds}, samples={normalized.shape[0]}, classes={targets.shape[1]})"
    )
    classifier = make_autobeme(mode=beme_mode, n_funds=n_funds)
    classifier.fit(normalized, targets)

    per_label_blend_alpha: dict[str, float] = {}
    per_label_uplift_only: dict[str, bool] = {}
    per_label_decision_threshold: dict[str, float] = {}
    calibration_summary: dict[str, dict[str, float]] = {}
    calibration_manifest = None
    if calibration_split_names and calibration_manifest_path and calibration_render_dir:
        calibration_manifest, calibration_coverage = build_rendered_training_manifest(
            sonyc_root=sonyc_root,
            split_names=calibration_split_names,
            per_label=render_per_label,
            seed=seed + 101,
            render_dir=calibration_render_dir,
            manifest_path=calibration_manifest_path,
        )
        print(f"[RENDERED_HEAD] calibration manifest -> {calibration_manifest}")
        print(f"[RENDERED_HEAD] calibration coverage -> {calibration_coverage}")
        calibration_rows, calibration_labels, calibration_counts = _collect_rendered_manifest_rows(calibration_manifest)
        if calibration_labels != labels:
            raise RuntimeError("Rendered calibration labels do not match training label order.")
        (
            calibration_features,
            calibration_targets,
            _calibration_trained_labels,
            _calibration_count_map,
            calibration_audio_paths,
        ) = _build_rendered_dataset_from_rows(
            rows=calibration_rows,
            labels=calibration_labels,
            counts=calibration_counts,
            seed=seed,
            min_examples_per_label=1,
            allowed_labels=trained_labels,
        )
        (
            per_label_blend_alpha,
            per_label_uplift_only,
            per_label_decision_threshold,
            calibration_summary,
        ) = _calibrate_rendered_beme_fusion(
            classifier=classifier,
            calibration_features=calibration_features,
            calibration_targets=calibration_targets,
            calibration_audio_paths=calibration_audio_paths,
            feature_mean=feature_mean,
            feature_std=feature_std,
            labels=labels,
            trained_labels=trained_labels,
            default_blend_alpha=float(blend_alpha),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": labels,
            "trained_labels": trained_labels,
            "train_counts": counts,
            "blend_alpha": float(blend_alpha),
            "uplift_only": True,
            "per_label_blend_alpha": per_label_blend_alpha,
            "per_label_uplift_only": per_label_uplift_only,
            "per_label_decision_threshold": per_label_decision_threshold,
            "beme_mode": str(beme_mode),
            "n_funds": int(n_funds),
            "train_manifest_path": manifest_path,
            "calibration_manifest_path": calibration_manifest,
            "calibration_summary": calibration_summary,
            "source": "Rendered SONYC clips -> YAMNet embeddings + semantic projection BEME head with per-label calibration",
        },
        output_path,
    )
    print(f"[OK] Rendered-manifest BEME semantic head -> {output_path}")
    print(f"[OK] trained labels: {trained_labels}")
    for label in trained_labels:
        print(f"[OK] {label:<20} {counts.get(label, 0)}")


def build_rendered_manifest_stereo_beme_semantic_head(
    sonyc_root: str,
    split_names: list[str],
    render_per_label: int,
    seed: int,
    output_path: str,
    blend_alpha: float,
    beme_mode: str,
    n_funds: int,
    train_manifest_path: str,
    render_dir: str,
    min_examples_per_label: int,
    stereo_feature_mode: str,
) -> None:
    manifest_path, coverage = build_rendered_training_manifest(
        sonyc_root=sonyc_root,
        split_names=split_names,
        per_label=render_per_label,
        seed=seed,
        render_dir=render_dir,
        manifest_path=train_manifest_path,
    )
    print(f"[RENDERED_STEREO_HEAD] training manifest -> {manifest_path}")
    print(f"[RENDERED_STEREO_HEAD] coverage -> {coverage}")

    features, targets, labels, trained_labels, counts = build_rendered_manifest_stereo_head_dataset(
        manifest_path=manifest_path,
        seed=seed,
        min_examples_per_label=min_examples_per_label,
        stereo_feature_mode=stereo_feature_mode,
    )
    feature_mean = np.mean(features, axis=0).astype(np.float32)
    feature_std = np.std(features, axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = ((features - feature_mean) / feature_std).astype(np.float32, copy=False)

    print(
        "[*] Training rendered-manifest stereo-aware BEME semantic head "
        f"(mode={beme_mode}, stereo_feature_mode={stereo_feature_mode}, "
        f"n_funds={n_funds}, samples={normalized.shape[0]}, classes={targets.shape[1]})"
    )
    classifier = make_autobeme(mode=beme_mode, n_funds=n_funds)
    classifier.fit(normalized, targets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": labels,
            "trained_labels": trained_labels,
            "train_counts": counts,
            "blend_alpha": float(blend_alpha),
            "uplift_only": True,
            "beme_mode": str(beme_mode),
            "n_funds": int(n_funds),
            "stereo_aware": True,
            "stereo_feature_mode": str(stereo_feature_mode),
            "train_manifest_path": manifest_path,
            "source": "Rendered SONYC stereo clips -> stereo-aware YAMNet projection BEME head",
        },
        output_path,
    )
    print(f"[OK] Rendered-manifest stereo-aware BEME semantic head -> {output_path}")
    print(f"[OK] trained labels: {trained_labels}")
    for label in trained_labels:
        print(f"[OK] {label:<20} {counts.get(label, 0)}")


# ---------------------------------------------------------------------------
# Pooling matrix helpers
# ---------------------------------------------------------------------------

def load_yamnet_names() -> list[str]:
    """Read 521 YAMNet display names from the local SavedModel asset."""
    asset_csv = os.path.join(EXPORT_DIR, "assets", "yamnet_class_map.csv")
    if os.path.exists(asset_csv):
        return _read_class_map(asset_csv)
    # Fallback: download from TF-Hub (requires network)
    try:
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        return _read_class_map(yamnet_model.class_map_path().numpy().decode())
    except Exception as exc:
        raise RuntimeError(f"Cannot load YAMNet class map: {exc}")


def _read_class_map(path: str) -> list[str]:
    names: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.append(row["display_name"])
    if len(names) != 521:
        raise ValueError(f"Expected 521 YAMNet classes, got {len(names)}")
    return names


def build_projection_matrix(
    yamnet_names: list[str],
    label_map: dict[str, list[str | tuple[str, float]]],
) -> tuple[np.ndarray, list[str]]:
    """Return a [521, K] float32 projection matrix and the ordered label list.

    Each column sums to 1 (weighted mean aggregation across contributing YAMNet classes).
    Unlabelled YAMNet classes contribute to nothing — no REST/other bucket.
    """
    labels = list(label_map.keys())
    name_to_idx = {name: i for i, name in enumerate(yamnet_names)}
    matrix = np.zeros((521, len(labels)), dtype=np.float32)

    for col, label in enumerate(labels):
        sources = label_map[label]
        total_weight = 0.0
        for src_entry in sources:
            if isinstance(src_entry, tuple):
                src, weight = src_entry
            else:
                src, weight = src_entry, 1.0
            if src not in name_to_idx:
                print(f"[WARN] '{src}' not in YAMNet 521 — skipped")
                continue
            matrix[name_to_idx[src], col] = float(weight)
            total_weight += float(weight)
        if total_weight > 0.0:
            matrix[:, col] /= total_weight

    assigned = int((matrix.sum(axis=1) > 0).sum())
    print(f"[INFO] {assigned}/521 YAMNet classes assigned; "
          f"{521 - assigned} contribute to nothing (no other bucket)")
    return matrix, labels


def build_temporal_pooling_metadata(labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float | int | str | None]]]:
    """Build per-class pooling parameters and a small JSON export."""
    betas: list[float] = []
    top_ks: list[int] = []
    use_top_k: list[bool] = []
    metadata: dict[str, dict[str, float | int | str | None]] = {}

    for label in labels:
        raw = TEMPORAL_POOL_CONFIG.get(label, {})
        mode = str(raw.get("mode", DEFAULT_POOL_MODE)).strip().lower()
        beta = float(raw.get("beta", DEFAULT_LSE_BETA))
        top_k = int(raw.get("top_k", 0))
        top_k_enabled = mode == "topk_lse" and top_k > 0

        betas.append(beta)
        top_ks.append(top_k if top_k_enabled else 1)
        use_top_k.append(top_k_enabled)
        metadata[label] = {
            "mode": "topk_lse" if top_k_enabled else "lse",
            "beta": beta,
            "top_k": top_k if top_k_enabled else None,
        }

    return (
        np.asarray(betas, dtype=np.float32),
        np.asarray(top_ks, dtype=np.int32),
        np.asarray(use_top_k, dtype=np.bool_),
        metadata,
    )


# ---------------------------------------------------------------------------
# TF SavedModel module
# ---------------------------------------------------------------------------
if tf is not None:

    class YamnetHeadModule(tf.Module):
        """Full YAMNet backbone + corrected projection + class-aware temporal pooling."""

        def __init__(
            self,
            projection: np.ndarray,
            labels: list[str],
            temporal_pool_config: dict[str, dict[str, float | int | str | None]],
        ) -> None:
            super().__init__()
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            self.projection = tf.constant(projection, dtype=tf.float32)
            pool_betas, pool_top_ks, pool_use_topk, _ = build_temporal_pooling_metadata(labels)
            self.pool_betas = tf.constant(pool_betas, dtype=tf.float32)
            self.pool_top_ks = tf.constant(pool_top_ks, dtype=tf.int32)
            self.pool_use_topk = tf.constant(pool_use_topk, dtype=tf.bool)
            self.max_top_k = max(int(item.get("top_k") or 0) for item in temporal_pool_config.values()) if temporal_pool_config else 1
            self.max_top_k = max(self.max_top_k, 1)

        @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="waveform_16k")])
        def __call__(self, waveform_16k: tf.Tensor) -> dict[str, tf.Tensor]:
            scores, embeddings, _spectrogram = self.yamnet(waveform_16k)
            projected = tf.matmul(scores, self.projection)
            col_max = tf.reduce_max(projected, axis=0, keepdims=True)
            shifted = projected - col_max
            lse_all = tf.squeeze(col_max, 0) + tf.math.log(
                tf.reduce_mean(tf.exp(tf.reshape(self.pool_betas, [1, -1]) * shifted), axis=0) + 1e-9
            ) / self.pool_betas

            projected_by_class = tf.transpose(projected)
            top_k_limit = tf.maximum(
                1,
                tf.minimum(tf.shape(projected_by_class)[1], tf.constant(self.max_top_k, dtype=tf.int32)),
            )
            top_values, _ = tf.math.top_k(projected_by_class, k=top_k_limit, sorted=True)
            safe_top_ks = tf.minimum(self.pool_top_ks, top_k_limit)
            safe_top_ks = tf.where(self.pool_use_topk, safe_top_ks, tf.ones_like(safe_top_ks))
            top_mask = tf.sequence_mask(safe_top_ks, maxlen=top_k_limit, dtype=top_values.dtype)
            masked_top = tf.where(
                top_mask > 0,
                top_values,
                tf.fill(tf.shape(top_values), tf.constant(-1e9, dtype=top_values.dtype)),
            )
            top_max = tf.reduce_max(masked_top, axis=1, keepdims=True)
            top_shifted = masked_top - top_max
            top_exp = tf.exp(tf.reshape(self.pool_betas, [-1, 1]) * top_shifted) * top_mask
            top_counts = tf.cast(tf.maximum(safe_top_ks, 1), top_values.dtype)
            lse_topk = tf.squeeze(top_max, 1) + tf.math.log(
                tf.reduce_sum(top_exp, axis=1) / top_counts + 1e-9
            ) / self.pool_betas

            pooled = tf.where(self.pool_use_topk, lse_topk, lse_all)
            probs = tf.clip_by_value(pooled, 0.0, 1.0)
            emb_mean = tf.reduce_mean(embeddings, axis=0)
            return {"probs": probs, "embeddings": emb_mean}


    def export(
        module: YamnetHeadModule,
        export_dir: str,
        labels: list[str],
        pooling_metadata: dict[str, dict[str, float | int | str | None]],
    ) -> None:
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(os.path.dirname(LABELS_JSON_MODELS), exist_ok=True)
        with open(LABELS_JSON_MODELS, "w", encoding="utf-8") as fh:
            json.dump(labels, fh, indent=2)
        print(f"[OK] labels ({len(labels)} classes) -> {LABELS_JSON_MODELS}")
        with open(POOLING_CONFIG_JSON_MODELS, "w", encoding="utf-8") as fh:
            json.dump(pooling_metadata, fh, indent=2)
        print(f"[OK] pooling metadata -> {POOLING_CONFIG_JSON_MODELS}")
        tf.saved_model.save(module, export_dir, signatures=module.__call__.get_concrete_function())
        print(f"[OK] SavedModel -> {export_dir}")

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_bytes = converter.convert()
            tflite_path = os.path.join(export_dir, "yamnet_semantic.tflite")
            with open(tflite_path, "wb") as fh:
                fh.write(tflite_bytes)
            print(f"[OK] TFLite   -> {tflite_path}")
        except Exception as exc:
            print(f"[WARN] TFLite conversion skipped: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builder",
        choices=("yamnet", "spectral", "learned_head", "beme_head", "rendered_beme_head", "rendered_stereo_beme_head", "pairwise", "all"),
        default=("yamnet" if tf is not None else "spectral"),
    )
    parser.add_argument("--sonyc-root", default=DEFAULT_SONYC_ROOT, help="Root folder containing Sony_Data assets.")
    parser.add_argument("--train-splits", default="train,val", help="Comma-separated SONYC annotation splits for spectral training.")
    parser.add_argument("--max-per-label", type=int, default=450, help="Maximum number of source events per semantic label.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed for sampling.")
    parser.add_argument("--spectral-output", default=SPECTRAL_MODEL_PATH, help="Output joblib path for the spectral artifact.")
    parser.add_argument("--learned-head-output", default=LEARNED_HEAD_MODEL_PATH, help="Output joblib path for the learnable semantic-head artifact.")
    parser.add_argument("--learned-head-alpha", type=float, default=0.65, help="Blend factor used to mix learned head probabilities with the semantic baseline.")
    parser.add_argument("--beme-head-output", default=BEME_SEMANTIC_HEAD_PATH, help="Output joblib path for the BEME semantic-head artifact.")
    parser.add_argument(
        "--beme-stereo-head-output",
        default=BEME_STEREO_SEMANTIC_HEAD_PATH,
        help="Output joblib path for the stereo-aware rendered BEME semantic-head artifact.",
    )
    parser.add_argument("--beme-head-alpha", type=float, default=0.55, help="Blend factor used to mix BEME head probabilities with the semantic baseline.")
    parser.add_argument("--beme-mode", choices=("recall", "precision", "balanced"), default="balanced", help="AutoBEME operating mode for the embedding-level semantic head.")
    parser.add_argument("--beme-n-funds", type=int, default=12, help="Number of BEME agents used in the semantic head.")
    parser.add_argument(
        "--beme-stereo-feature-mode",
        choices=("full", "delta_stats", "stats_only", "compact", "compact_delta_stats", "compact_stats"),
        default="full",
        help="Stereo-aware feature recipe used by rendered_stereo_beme_head.",
    )
    parser.add_argument(
        "--rendered-train-manifest",
        default=str(Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "tmp_eval" / "sonyc_trainval_rendered_manifest.csv"),
        help="Output CSV path for the rendered train/val clip manifest used by rendered_beme_head.",
    )
    parser.add_argument(
        "--rendered-train-dir",
        default=str(Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "tmp_eval" / "sonyc_trainval_rendered"),
        help="Directory where rendered train/val clips are cached for rendered_beme_head and rendered_stereo_beme_head.",
    )
    parser.add_argument(
        "--rendered-calibration-splits",
        default="",
        help="Optional comma-separated SONYC annotation splits used to build an experimental rendered calibration manifest. Leave empty to keep the safer uncalibrated default.",
    )
    parser.add_argument(
        "--rendered-calibration-manifest",
        default=str(Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "tmp_eval" / "sonyc_val_rendered_manifest.csv"),
        help="Output CSV path for the rendered validation manifest used for per-label BEME calibration.",
    )
    parser.add_argument(
        "--rendered-calibration-dir",
        default=str(Path(__file__).resolve().parents[1] / "logs" / "evaluation" / "tmp_eval" / "sonyc_val_rendered"),
        help="Directory where rendered validation clips are cached for per-label BEME calibration.",
    )
    parser.add_argument(
        "--rendered-per-label",
        type=int,
        default=25,
        help="Balanced scene quota per semantic label when building the rendered train/val manifest.",
    )
    parser.add_argument(
        "--rendered-min-positive",
        type=int,
        default=8,
        help="Minimum positive clip count required for a semantic label to be trainable in rendered_beme_head or rendered_stereo_beme_head.",
    )
    parser.add_argument("--max-annotation-files", type=int, default=6000, help="Optional cap on scanned SONYC annotation files for learned semantic-head training. Use 0 for no cap.")
    parser.add_argument("--pairwise-output", default=PAIRWISE_REFINER_OUTPUT_PATH, help="Output joblib path for the pairwise confusion refiner.")
    return parser.parse_args()


def build_yamnet_savedmodel() -> None:
    if tf is None or hub is None:
        raise RuntimeError("TensorFlow/TensorFlow Hub not available for YAMNet export.")

    print("[*] Loading YAMNet class map …")
    names = load_yamnet_names()

    print("[*] Building projection matrix …")
    projection, labels = build_projection_matrix(names, LABEL_MAP)
    _pool_betas, _pool_top_ks, _pool_use_topk, pooling_metadata = build_temporal_pooling_metadata(labels)

    print("[*] Constructing YamnetHeadModule …")
    module = YamnetHeadModule(projection, labels=labels, temporal_pool_config=pooling_metadata)

    # Warm-up to trace the tf.function before export
    _ = module(tf.zeros([16000], tf.float32))
    print("[*] Warm-up done.")

    export(module, EXPORT_DIR, labels, pooling_metadata)
    print("[DONE] Rebuild complete. Run evaluate_model.py to verify.")


def main() -> None:
    args = parse_args()
    split_names = [item.strip() for item in args.train_splits.split(",") if item.strip()]
    if args.builder == "spectral":
        build_spectral_backbone(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.spectral_output),
        )
        return

    if args.builder == "learned_head":
        build_learned_semantic_head(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.learned_head_output),
            blend_alpha=float(args.learned_head_alpha),
            max_annotation_files=(None if int(args.max_annotation_files) <= 0 else int(args.max_annotation_files)),
        )
        return

    if args.builder == "beme_head":
        build_beme_semantic_head(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.beme_head_output),
            blend_alpha=float(args.beme_head_alpha),
            beme_mode=str(args.beme_mode),
            n_funds=max(2, int(args.beme_n_funds)),
            max_annotation_files=(None if int(args.max_annotation_files) <= 0 else int(args.max_annotation_files)),
        )
        return

    if args.builder == "rendered_beme_head":
        calibration_split_names = [item.strip() for item in args.rendered_calibration_splits.split(",") if item.strip()]
        build_rendered_manifest_beme_semantic_head(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            render_per_label=max(1, int(args.rendered_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.beme_head_output),
            blend_alpha=float(args.beme_head_alpha),
            beme_mode=str(args.beme_mode),
            n_funds=max(2, int(args.beme_n_funds)),
            train_manifest_path=os.path.abspath(args.rendered_train_manifest),
            render_dir=os.path.abspath(args.rendered_train_dir),
            min_examples_per_label=max(1, int(args.rendered_min_positive)),
            calibration_split_names=calibration_split_names or None,
            calibration_manifest_path=os.path.abspath(args.rendered_calibration_manifest),
            calibration_render_dir=os.path.abspath(args.rendered_calibration_dir),
        )
        return

    if args.builder == "rendered_stereo_beme_head":
        build_rendered_manifest_stereo_beme_semantic_head(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            render_per_label=max(1, int(args.rendered_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.beme_stereo_head_output),
            blend_alpha=float(args.beme_head_alpha),
            beme_mode=str(args.beme_mode),
            n_funds=max(2, int(args.beme_n_funds)),
            train_manifest_path=os.path.abspath(args.rendered_train_manifest),
            render_dir=os.path.abspath(args.rendered_train_dir),
            min_examples_per_label=max(1, int(args.rendered_min_positive)),
            stereo_feature_mode=str(args.beme_stereo_feature_mode),
        )
        return

    if args.builder == "pairwise":
        build_pairwise_refiner(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.pairwise_output),
        )
        return

    if args.builder == "all":
        build_spectral_backbone(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.spectral_output),
        )
        build_learned_semantic_head(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.learned_head_output),
            blend_alpha=float(args.learned_head_alpha),
            max_annotation_files=(None if int(args.max_annotation_files) <= 0 else int(args.max_annotation_files)),
        )
        build_pairwise_refiner(
            sonyc_root=os.path.abspath(args.sonyc_root),
            split_names=split_names,
            max_per_label=max(1, int(args.max_per_label)),
            seed=int(args.seed),
            output_path=os.path.abspath(args.pairwise_output),
        )

    build_yamnet_savedmodel()


if __name__ == "__main__":
    main()
