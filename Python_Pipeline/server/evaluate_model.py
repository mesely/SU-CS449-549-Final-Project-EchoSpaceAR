"""Evaluate and compare the old and new EchoSpace audio pipelines.

This script intentionally stays in a single file so it is easy to audit,
version, and rerun. It compares the current pipeline files against their
`_old` backups on the same audio clips and writes shared CSV histories plus
comparison figures.

What it compares:
  - legacy raw YAMNet-style pipeline (`*_old` behavior)
  - current pipeline with decision layer + stereo side-channel

What it measures:
  - multilabel raw-event hit rate and Macro-F1@K
  - primary-label confusion and calibration (ECE)
  - decision-target hit rate for the HUD layer
  - flapping / label transitions per minute
  - false alerts per minute
  - speech-gate stability
  - decision / spatial payload presence
  - silence / idle / other / ood output ratios

Required manifest columns:
  - audio_path
  - either `labels` or `label`

Optional manifest columns:
  - clip_id
  - dataset
  - source_label
  - participant_id
  - condition
  - environment
  - sensor
  - channels
  - snr_db
  - annotator_confidence
  - onset_time
  - prediction_time
  - ui_render_time
  - user_response_time

The script expects WAV files. Mono and stereo WAV files are both supported.
FSD50K-style multilabel manifests are supported through a semicolon-separated
`labels` column after mapping labels into the reduced EchoSpace label set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

import pipeline_runtime.config as current_config
import pipeline_runtime.config_old as legacy_config
from pipeline_runtime.classification import YamnetClassifier
from pipeline_runtime.config import LOG_DIR, REDUCED_LABEL_SET, tf
from pipeline_runtime.decision_layer import DecisionSnapshot, PriorityDecisionLayer
from pipeline_runtime.spatial_audio import SpatialSnapshot, downmix_to_mono, ensure_frame_major, summarize_spatial_audio
from pipeline_runtime.utils import resample_linear

import matplotlib.pyplot as plt

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional runtime dependency
    sf = None


EVAL_DIR = os.path.join(LOG_DIR, "evaluation")
SUMMARY_CSV = os.path.join(EVAL_DIR, "model_benchmark_results.csv")
PREDICTIONS_CSV = os.path.join(EVAL_DIR, "model_benchmark_predictions.csv")
CORE_METRICS_PNG = os.path.join(EVAL_DIR, "1_overall_core_metrics.png")
CONFUSION_PAIR_PNG = os.path.join(EVAL_DIR, "2_confusion_mono.png")
BEHAVIOR_SUMMARY_PNG = os.path.join(EVAL_DIR, "3_behavior_summary.png")
ARHUD_DIFF_PNG = os.path.join(EVAL_DIR, "4_arhud_diff.png")
STEREO_MONO_DIFF_PNG = os.path.join(EVAL_DIR, "5_stereo_mono_difference.png")
SONYC_SAMPLE_RATE = 16000
SONYC_RENDER_SECONDS = 10.0
SONYC_MONO_DATASET_NAME = "SONYC-FSD-SED Mono"
SONYC_STEREO_DATASET_NAME = "SONYC-FSD-SED Stereo"
SONYC_RENDER_DIR = os.path.join(EVAL_DIR, "sonyc_rendered")
SONYC_MANIFEST_CSV = os.path.join(EVAL_DIR, "sonyc_fsd_sed_manifest.csv")
SONYC_SOURCE_SUBDIR = os.path.join("extracted", "SONYC_FSD_SED.source")
SONYC_TEST_ANNOTATION_SUBDIR = os.path.join("extracted", "SONYC_FSD_SED_add_test.annotations", "test_past_year")
SONYC_VOCAB_FILENAME = "vocab.json"
SONYC_LABEL_MAP = {
    "Shatter": "glass_break",
    "Crack": "glass_break",
    "Ringtone": "phone_ring",
    "Knock": "door_knock",
    "Doorbell": "alarms_buzzer",
    "Microwave_oven": "alarms_buzzer",
    "Fireworks": "explosion_gunshot",
    "Gunshot_and_gunfire": "explosion_gunshot",
    "Boom": "explosion_gunshot",
    "Meow": "cat",
    "Bark": "dog",
    "Subway_and_metro_and_underground": "rail",
    "Wind": "wind_rain",
    "Male_speech_and_man_speaking": "speech",
    "Female_speech_and_woman_speaking": "speech",
    "Child_speech_and_kid_speaking": "speech",
    "Whispering": "speech",
    "Speech_synthesizer": "speech",
    "Bass_drum": "music",
    "Hi-hat": "music",
    "Electric_guitar": "music",
    "Bass_guitar": "music",
    "Harmonica": "music",
    "Trumpet": "music",
    "Acoustic_guitar": "music",
    "Piano": "music",
    "Snare_drum": "music",
    "Bowed_string_instrument": "music",
    "Tabla": "music",
    "Harp": "music",
    "Female_singing": "music",
    "Male_singing": "music",
    "Tambourine": "music",
    "Crash_cymbal": "music",
    "Drum_kit": "music",
    "Accordion": "music",
    "Organ": "music",
    "Cowbell": "music",
    "Rattle_(instrument)": "music",
    "Gong": "music",
}
SONYC_LABEL_ORDER = [
    "glass_break",
    "alarms_buzzer",
    "door_knock",
    "explosion_gunshot",
    "phone_ring",
    "speech",
    "music",
    "dog",
    "cat",
    "rail",
    "wind_rain",
]

OPTIONAL_METADATA_COLUMNS = [
    "participant_id",
    "condition",
    "environment",
    "sensor",
    "channels",
    "snr_db",
    "annotator_confidence",
]

NEUTRAL_OUTPUT_LABELS = {"silence", "idle", "other", "ood", "Silence"}
DEFAULT_LABEL_SEPARATOR = ";"
SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "run_id",
    "model_label",
    "pipeline_variant",
    "dataset_name",
    "metric_scope",
    "class_label",
    "sample_count",
    "correct_count",
    "total_audio_minutes",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "topk_hit_rate",
    "decision_target_hit_rate",
    "actionable_hud_hit_rate",
    "non_actionable_suppression_rate",
    "raw_accuracy",
    "raw_precision",
    "raw_recall",
    "raw_f1",
    "avg_confidence",
    "raw_avg_confidence",
    "ece",
    "false_alerts_per_minute",
    "flapping_transitions_per_min",
    "speech_gate_toggles_per_min",
    "speech_gate_open_ratio",
    "payload_decision_rate",
    "payload_spatial_rate",
    "payload_raw_top5_rate",
    "paired_dominant_agreement_rate",
    "mono_raw_match_rate",
    "mono_spatial_valid_rate",
    "stereo_spatial_active_rate",
    "silence_rate",
    "idle_rate",
    "other_rate",
    "ood_rate",
    "onset_latency_ms_mean",
    "onset_latency_ms_p50",
    "ui_latency_ms_mean",
    "ui_latency_ms_p50",
    "human_response_ms_mean",
    "human_response_ms_p50",
]
PREDICTION_FIELDNAMES = [
    "timestamp_utc",
    "run_id",
    "model_label",
    "pipeline_variant",
    "dataset_name",
    "clip_id",
    "audio_path",
    "source_label",
    "target_label",
    "target_labels",
    "decision_target_labels",
    "raw_predicted_label",
    "raw_predicted_confidence",
    "raw_top3_labels",
    "dominant_label",
    "dominant_confidence",
    "dominant_top3_labels",
    "window_index",
    "window_start_s",
    "window_end_s",
    "duration_seconds",
    "is_correct",
    "is_raw_correct",
    "matches_any_target",
    "matches_decision_target",
    "input_channels",
    "spatial_mode",
    "spatial_direction",
    "spatial_direction_confidence",
    "spl_dbfs",
    "decision_state",
    "decision_priority",
    "speech_prob",
    "speech_gate_state",
    "speech_gate_opened",
    "speech_gate_closed",
    "payload_has_decision",
    "payload_has_spatial",
    "payload_has_raw_top5",
    "participant_id",
    "condition",
    "environment",
    "sensor",
    "channels",
    "snr_db",
    "annotator_confidence",
    "onset_time",
    "prediction_time",
    "ui_render_time",
    "user_response_time",
]


@dataclass
class SampleRecord:
    """One labeled audio sample loaded from the manifest."""

    clip_id: str
    audio_path: str
    label: str
    source_label: str
    dataset_name: str
    positive_labels: tuple[str, ...] = ()
    decision_target_labels: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    onset_time: float | None = None
    prediction_time: float | None = None
    ui_render_time: float | None = None
    user_response_time: float | None = None


@dataclass
class WindowRecord:
    """One pipeline output window inside one audio clip."""

    sample: SampleRecord
    pipeline_variant: str
    window_index: int
    window_start_s: float
    window_end_s: float
    raw_predicted_label: str
    raw_predicted_confidence: float
    raw_top3: list[tuple[str, float]]
    dominant_label: str
    dominant_confidence: float
    dominant_top3: list[tuple[str, float]]
    decision_state: str
    decision_priority: str
    speech_prob: float
    speech_gate_state: str
    speech_gate_opened: bool
    speech_gate_closed: bool
    payload_has_decision: bool
    payload_has_spatial: bool
    payload_has_raw_top5: bool
    spatial_mode: str
    spatial_direction: str
    spatial_direction_confidence: float
    spl_dbfs: float
    duration_seconds: float
    input_channels: int


@dataclass
class ClipResult:
    """Aggregate one clip into one comparable result row per pipeline variant."""

    sample: SampleRecord
    pipeline_variant: str
    predicted_label: str
    predicted_confidence: float
    dominant_top3: list[tuple[str, float]]
    raw_predicted_label: str
    raw_predicted_confidence: float
    raw_top3: list[tuple[str, float]]
    raw_eval_labels: tuple[str, ...]
    duration_seconds: float
    windows: list[WindowRecord]


@dataclass
class CrossVariantMetrics:
    """Metrics that depend on looking at both variants on aligned windows."""

    paired_dominant_agreement_rate: float | str
    mono_raw_match_rate: float | str


@dataclass
class SonicEventSpec:
    """One source event used to synthesize a SONYC-FSD-SED soundscape."""

    role: str
    raw_label: str
    reduced_label: str | None
    source_rel_path: str
    source_time_s: float
    source_duration_s: float
    output_time_s: float
    output_duration_s: float
    snr_db: float


@dataclass
class SonicSceneSpec:
    """One selected SONYC-FSD-SED scene with mapped labels and source events."""

    clip_id: str
    annotation_path: str
    positive_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    background_event: SonicEventSpec
    foreground_events: tuple[SonicEventSpec, ...]


@dataclass
class LegacyWindowOutput:
    """Legacy pipeline output for one window."""

    raw_top5: list[dict]
    dominant_label: str
    dominant_prob: float
    speech_prob: float
    gate_state: str
    gate_opened: bool
    gate_closed: bool
    payload: dict


@dataclass
class CurrentWindowOutput:
    """Current pipeline output for one window."""

    raw_top5: list[dict]
    decision: DecisionSnapshot
    spatial: SpatialSnapshot
    gate_state: str
    gate_opened: bool
    gate_closed: bool
    payload: dict


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=False, help="CSV manifest with audio_path and label columns.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional root folder for relative audio paths. Defaults to the manifest folder.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Friendly dataset name used in CSV rows and charts. Defaults to manifest filename.",
    )
    parser.add_argument(
        "--model-label",
        default="Recent Model",
        help="Base model label written into the shared benchmark CSV.",
    )
    parser.add_argument(
        "--legacy-variant-label",
        default="old_pipeline",
        help="Variant name used for the `_old` pipeline rows.",
    )
    parser.add_argument(
        "--current-variant-label",
        default="current_pipeline",
        help="Variant name used for the current pipeline rows.",
    )
    parser.add_argument(
        "--path-column",
        default="audio_path",
        help="Manifest column containing each audio file path.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Fallback manifest column containing the primary target label.",
    )
    parser.add_argument(
        "--labels-column",
        default="labels",
        help="Optional manifest column containing semicolon-separated multilabel targets.",
    )
    parser.add_argument(
        "--label-separator",
        default=DEFAULT_LABEL_SEPARATOR,
        help="Separator used by the multilabel target column.",
    )
    parser.add_argument(
        "--clip-id-column",
        default="clip_id",
        help="Optional manifest column used as a stable sample identifier.",
    )
    parser.add_argument(
        "--dataset-column",
        default="dataset",
        help="Optional manifest column for dataset name overrides per row.",
    )
    parser.add_argument(
        "--source-label-column",
        default="source_label",
        help="Optional manifest column holding the original dataset label before mapping.",
    )
    parser.add_argument(
        "--label-map-json",
        default=None,
        help="Optional JSON file that maps source labels into the reduced label space.",
    )
    parser.add_argument(
        "--summary-csv",
        default=SUMMARY_CSV,
        help="Shared CSV that stores overall and per-class benchmark rows.",
    )
    parser.add_argument(
        "--predictions-csv",
        default=PREDICTIONS_CSV,
        help="Shared CSV that stores one row per evaluated window.",
    )
    parser.add_argument(
        "--plot-path",
        default=CORE_METRICS_PNG,
        help="PNG path for the core outcome comparison chart.",
    )
    parser.add_argument(
        "--behavior-plot-path",
        default=BEHAVIOR_SUMMARY_PNG,
        help="PNG path for the pipeline behavior summary chart.",
    )
    parser.add_argument(
        "--confusion-plot-path",
        default=CONFUSION_PAIR_PNG,
        help="PNG path for the mono confusion comparison chart.",
    )
    parser.add_argument(
        "--arhud-plot-path",
        default=ARHUD_DIFF_PNG,
        help="PNG path for the AR-HUD expectation chart.",
    )
    parser.add_argument(
        "--stereo-mono-plot-path",
        default=STEREO_MONO_DIFF_PNG,
        help="PNG path for the mono-vs-stereo comparison chart.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(current_config.CLASSIFY_WINDOW_S),
        help="Sliding-window size in seconds.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=float(current_config.CLASSIFY_HOP_S),
        help="Sliding-window hop size in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick dry-runs.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins used to compute expected calibration error.",
    )
    parser.add_argument(
        "--raw-topk",
        type=int,
        default=3,
        help="Top-K raw labels used for multilabel hit/F1 evaluation.",
    )
    parser.add_argument(
        "--clear-old-pngs",
        action="store_true",
        help="Delete old PNG artifacts under the evaluation folder before writing new ones.",
    )
    parser.add_argument(
        "--sonyc-root",
        default=None,
        help="Optional SONYC-FSD-SED dataset root. When set, the manifest is built automatically.",
    )
    parser.add_argument(
        "--sonyc-split",
        default="test_past_year",
        help="SONYC-FSD-SED split folder used under extracted annotations.",
    )
    parser.add_argument(
        "--sonyc-per-label",
        type=int,
        default=25,
        help="Balanced quota per mapped reduced label for SONYC scene selection.",
    )
    parser.add_argument(
        "--sonyc-seed",
        type=int,
        default=449,
        help="Deterministic random seed for SONYC scene selection and stereo panning.",
    )
    parser.add_argument(
        "--sonyc-render-dir",
        default=SONYC_RENDER_DIR,
        help="Cache directory for rendered SONYC WAV files.",
    )
    parser.add_argument(
        "--sonyc-manifest-out",
        default=SONYC_MANIFEST_CSV,
        help="Output CSV path for the generated SONYC manifest.",
    )
    args = parser.parse_args()
    if not args.manifest and not args.sonyc_root:
        parser.error("--manifest veya --sonyc-root verilmelidir.")
    return args


def ensure_parent_dir(path: str) -> None:
    """Create the parent folder for a file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def slugify(value: str) -> str:
    """Create a stable filesystem-friendly name."""
    lowered = value.lower()
    chars = [ch if ch.isalnum() else "-" for ch in lowered]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def safe_div(numerator: float, denominator: float) -> float:
    """Avoid repeated zero-division checks in metric calculations."""
    return numerator / denominator if denominator else 0.0


def unique_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate labels while preserving their original order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return tuple(ordered)


def split_label_cell(raw_value: str, separator: str) -> tuple[str, ...]:
    """Parse a label cell into a stable tuple of labels."""
    if not raw_value:
        return ()

    normalized = raw_value.replace("|", separator)
    if separator != ",":
        normalized = normalized.replace(",", separator)
    parts = [item.strip() for item in normalized.split(separator)]
    return unique_labels(part for part in parts if part)


def format_labels(labels: Iterable[str]) -> str:
    """Serialize labels into a compact semicolon-separated string."""
    return DEFAULT_LABEL_SEPARATOR.join(unique_labels(labels))


def collapse_label_for_decision(label: str) -> str:
    """Map raw event labels into the HUD-oriented decision target space."""
    if label in {"silence", current_config.MODEL_SILENCE_LABEL}:
        return "silence"
    if label == "ood":
        return "ood"
    if label in current_config.ACTIONABLE_LABELS:
        return label
    if label in current_config.IDLE_LABELS:
        return "idle"
    return "other"


def target_labels_for_sample(sample: SampleRecord) -> set[str]:
    """Return the raw positive label set for one sample."""
    return set(sample.positive_labels or (sample.label,))


def decision_target_set(sample: SampleRecord) -> set[str]:
    """Return the collapsed decision-target set for one sample."""
    return set(sample.decision_target_labels or (collapse_label_for_decision(sample.label),))


def clean_old_png_artifacts(directory: str) -> list[str]:
    """Delete stale PNG outputs before a fresh benchmark run."""
    removed = []
    if not os.path.isdir(directory):
        return removed
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".png"):
            continue
        path = os.path.join(directory, filename)
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            continue
    return removed


def parse_optional_float(value: str | None) -> float | None:
    """Convert an optional string to float without crashing on blanks."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped.replace(",", "."))
    except ValueError:
        return None


def load_label_map(path: str | None) -> dict[str, str]:
    """Load an optional source-label to reduced-label mapping."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(key): str(value) for key, value in mapping.items()}


def source_rel_path(raw_source_path: str) -> str:
    """Convert SONYC scratch paths into repo-local relative source paths."""
    normalized = raw_source_path.replace("\\", "/")
    for marker in ("sonyc_background/", "fsd50k_foreground/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index:]
    raise ValueError(f"Kaynak yol eslenemedi: {raw_source_path}")


def load_sonyc_vocab(sonyc_root: str) -> list[str]:
    """Load the SONYC-FSD-SED foreground vocabulary."""
    vocab_path = os.path.join(sonyc_root, SONYC_VOCAB_FILENAME)
    with open(vocab_path, "r", encoding="utf-8") as handle:
        return list(json.load(handle))


def iter_sonyc_annotation_paths(sonyc_root: str, split_name: str) -> list[str]:
    """Collect all extracted SONYC annotation paths for one split."""
    extracted_root = os.path.join(sonyc_root, "extracted", "SONYC_FSD_SED_add_test.annotations", split_name)
    if not os.path.isdir(extracted_root):
        raise RuntimeError(
            f"SONYC anotasyon klasoru bulunamadi: {extracted_root}. "
            "Lutfen add_test annotations arsivini acilmis halde tut."
        )
    return [str(path) for path in sorted(Path(extracted_root).glob("*.jams"))]


def load_json(path: str) -> dict:
    """Load one JSON/JAMS document from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_sonyc_scene(annotation_path: str, vocab: list[str]) -> SonicSceneSpec | None:
    """Parse one SONYC-FSD-SED JAMS file into a compact scene description."""
    data = load_json(annotation_path)
    events = data.get("annotations", [{}])[0].get("data", [])

    background_event: SonicEventSpec | None = None
    foreground_events: list[SonicEventSpec] = []
    mapped_labels: list[str] = []
    mapped_source_labels: list[str] = []

    for item in events:
        value = item.get("value", {})
        role = str(value.get("role", ""))
        raw_label_id = str(value.get("label", ""))
        raw_label = raw_label_id
        if role == "foreground" and raw_label_id.isdigit():
            raw_index = int(raw_label_id)
            if 0 <= raw_index < len(vocab):
                raw_label = vocab[raw_index]
        reduced_label = SONYC_LABEL_MAP.get(raw_label)

        spec = SonicEventSpec(
            role=role,
            raw_label=raw_label,
            reduced_label=reduced_label,
            source_rel_path=source_rel_path(str(value.get("source_file", ""))),
            source_time_s=float(value.get("source_time", 0.0) or 0.0),
            source_duration_s=float(value.get("event_duration", item.get("duration", 0.0)) or 0.0),
            output_time_s=float(item.get("time", 0.0) or 0.0),
            output_duration_s=float(item.get("duration", 0.0) or 0.0),
            snr_db=float(value.get("snr", 0.0) or 0.0),
        )

        if role == "background":
            background_event = spec
            continue

        if role != "foreground":
            continue

        foreground_events.append(spec)
        if reduced_label:
            mapped_labels.append(reduced_label)
            mapped_source_labels.append(raw_label)

    if background_event is None or not mapped_labels:
        return None

    clip_id = os.path.splitext(os.path.basename(annotation_path))[0]
    return SonicSceneSpec(
        clip_id=clip_id,
        annotation_path=annotation_path,
        positive_labels=unique_labels(mapped_labels),
        source_labels=unique_labels(mapped_source_labels),
        background_event=background_event,
        foreground_events=tuple(foreground_events),
    )


def scene_sources_exist(scene: SonicSceneSpec, source_root: str) -> bool:
    """Check that every referenced audio file exists locally."""
    background_path = os.path.join(source_root, scene.background_event.source_rel_path)
    if not os.path.exists(background_path):
        return False
    return all(
        os.path.exists(os.path.join(source_root, event.source_rel_path))
        for event in scene.foreground_events
        if event.reduced_label is not None
    )


def choose_sonyc_scenes(
    scenes: list[SonicSceneSpec],
    source_root: str,
    per_label: int,
    seed: int,
) -> tuple[list[SonicSceneSpec], dict[str, int]]:
    """Choose a balanced subset of SONYC scenes by reduced label."""
    support = Counter(label for scene in scenes for label in scene.positive_labels)
    labels = [label for label in SONYC_LABEL_ORDER if support[label] > 0]
    by_label: dict[str, list[SonicSceneSpec]] = {label: [] for label in labels}
    for scene in scenes:
        for label in scene.positive_labels:
            if label in by_label:
                by_label[label].append(scene)

    rng = random.Random(seed)
    for scene_list in by_label.values():
        rng.shuffle(scene_list)

    selected_by_id: dict[str, SonicSceneSpec] = {}
    coverage = Counter()
    for label in sorted(labels, key=lambda item: (support[item], item)):
        for scene in by_label[label]:
            if coverage[label] >= per_label:
                break
            if scene.clip_id in selected_by_id:
                continue
            if not scene_sources_exist(scene, source_root):
                continue
            selected_by_id[scene.clip_id] = scene
            for covered_label in scene.positive_labels:
                coverage[covered_label] += 1

    underfilled_labels = [label for label in labels if coverage[label] < per_label]
    if underfilled_labels:
        for label in underfilled_labels:
            for scene in by_label[label]:
                if coverage[label] >= per_label:
                    break
                if scene.clip_id in selected_by_id or not scene_sources_exist(scene, source_root):
                    continue
                selected_by_id[scene.clip_id] = scene
                for covered_label in scene.positive_labels:
                    coverage[covered_label] += 1

    return sorted(selected_by_id.values(), key=lambda scene: scene.clip_id), {label: coverage[label] for label in labels}


def stretch_to_num_samples(samples: np.ndarray, target_samples: int) -> np.ndarray:
    """Linearly stretch a mono segment to a fixed sample count."""
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if samples.size == target_samples:
        return samples.astype(np.float32, copy=False)

    source_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32, copy=False)


def crop_or_pad(samples: np.ndarray, start_sample: int, target_samples: int) -> np.ndarray:
    """Extract a fixed-length segment from a mono waveform."""
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if start_sample >= samples.size:
        return np.zeros(target_samples, dtype=np.float32)

    segment = samples[max(start_sample, 0) : max(start_sample, 0) + target_samples]
    if segment.size >= target_samples:
        return segment.astype(np.float32, copy=False)

    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: segment.size] = segment.astype(np.float32, copy=False)
    return padded


def rms(samples: np.ndarray) -> float:
    """Compute a stable RMS value for gain matching."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64))) + 1e-9))


def deterministic_pan(scene_id: str, event_index: int, seed: int) -> float:
    """Generate a stable left-right pan for synthetic stereo rendering."""
    digest = hashlib.sha1(f"{scene_id}:{event_index}:{seed}".encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / float(0xFFFFFFFF)
    return -0.78 + (1.56 * unit)


def apply_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """Pan a mono segment into stereo with constant-power gains."""
    clipped_pan = max(-1.0, min(1.0, pan))
    left_gain = float(np.sqrt((1.0 - clipped_pan) * 0.5))
    right_gain = float(np.sqrt((1.0 + clipped_pan) * 0.5))
    return np.stack([mono * left_gain, mono * right_gain], axis=1)


def maybe_shift_channel(stereo: np.ndarray, pan: float, sample_rate: int) -> np.ndarray:
    """Inject a tiny inter-channel delay so the spatial probe is less degenerate."""
    if stereo.size == 0 or abs(pan) < 0.15:
        return stereo

    delay_samples = max(1, min(int(round(abs(pan) * sample_rate * 0.00025)), 6))
    shifted = stereo.copy()
    if pan > 0:
        shifted[delay_samples:, 0] = shifted[:-delay_samples, 0]
        shifted[:delay_samples, 0] = 0.0
    else:
        shifted[delay_samples:, 1] = shifted[:-delay_samples, 1]
        shifted[:delay_samples, 1] = 0.0
    return shifted


def render_sonyc_scene(
    scene: SonicSceneSpec,
    source_root: str,
    output_path: str,
    stereo: bool,
    seed: int,
    cache: dict[str, np.ndarray],
) -> None:
    """Render one SONYC scene into a reusable WAV file."""
    ensure_parent_dir(output_path)
    if os.path.exists(output_path):
        return

    target_length = int(round(SONYC_RENDER_SECONDS * SONYC_SAMPLE_RATE))

    def load_source_mono(source_rel_path: str) -> np.ndarray:
        cache_key = source_rel_path
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        absolute_path = os.path.join(source_root, source_rel_path)
        if sf is not None:
            samples, sample_rate = sf.read(absolute_path, always_2d=True, dtype="float32")
            mono = np.mean(np.asarray(samples, dtype=np.float32), axis=1)
            input_sr = int(sample_rate)
        else:
            raw_audio = tf.io.read_file(absolute_path)
            waveform, sample_rate = tf.audio.decode_wav(raw_audio, desired_channels=1)
            mono = waveform.numpy().reshape(-1).astype(np.float32, copy=False)
            input_sr = int(sample_rate.numpy())
        if input_sr != SONYC_SAMPLE_RATE:
            mono = resample_linear(mono, input_sr, SONYC_SAMPLE_RATE)
        cache[cache_key] = mono
        return mono

    background_wave = load_source_mono(scene.background_event.source_rel_path)
    background_start = int(round(scene.background_event.source_time_s * SONYC_SAMPLE_RATE))
    background_clip = crop_or_pad(background_wave, background_start, target_length)
    background_rms = max(rms(background_clip), 1e-4)

    if stereo:
        mix = np.stack([background_clip, background_clip], axis=1)
    else:
        mix = background_clip.copy()

    mapped_events = [event for event in scene.foreground_events if event.reduced_label is not None]
    for event_index, event in enumerate(mapped_events):
        source_wave = load_source_mono(event.source_rel_path)
        source_start = int(round(event.source_time_s * SONYC_SAMPLE_RATE))
        source_length = max(1, int(round(event.source_duration_s * SONYC_SAMPLE_RATE)))
        source_segment = crop_or_pad(source_wave, source_start, source_length)
        event_length = max(1, int(round(event.output_duration_s * SONYC_SAMPLE_RATE)))
        rendered_segment = stretch_to_num_samples(source_segment, event_length)

        event_rms = max(rms(rendered_segment), 1e-4)
        target_event_rms = background_rms * (10.0 ** (event.snr_db / 20.0))
        gain = target_event_rms / event_rms
        rendered_segment = rendered_segment * gain

        start_index = int(round(event.output_time_s * SONYC_SAMPLE_RATE))
        if start_index >= target_length:
            continue
        end_index = min(start_index + rendered_segment.size, target_length)
        rendered_segment = rendered_segment[: end_index - start_index]

        if stereo:
            panned = apply_pan(rendered_segment, deterministic_pan(scene.clip_id, event_index, seed))
            panned = maybe_shift_channel(panned, deterministic_pan(scene.clip_id, event_index, seed), SONYC_SAMPLE_RATE)
            mix[start_index:end_index] += panned
        else:
            mix[start_index:end_index] += rendered_segment

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix = mix / peak * 0.95

    tensor = tf.convert_to_tensor(mix.reshape((-1, 2 if stereo else 1)), dtype=tf.float32)
    encoded = tf.audio.encode_wav(tensor, sample_rate=SONYC_SAMPLE_RATE)
    tf.io.write_file(output_path, encoded)


def build_sonyc_manifest(args: argparse.Namespace) -> str:
    """Render a balanced SONYC benchmark subset and write a manifest CSV."""
    sonyc_root = os.path.abspath(args.sonyc_root)
    source_root = os.path.join(sonyc_root, SONYC_SOURCE_SUBDIR)
    if not os.path.isdir(source_root):
        raise RuntimeError(
            f"SONYC source klasoru bulunamadi: {source_root}. "
            "Kaynak arsivin en azindan kullanacagimiz dosyalari acilmis olmali."
        )

    print("[SONYC] Vocab ve anotasyonlar okunuyor...")
    vocab = load_sonyc_vocab(sonyc_root)
    annotation_paths = iter_sonyc_annotation_paths(sonyc_root, args.sonyc_split)
    scene_specs = []
    for index, annotation_path in enumerate(annotation_paths, start=1):
        scene = parse_sonyc_scene(annotation_path, vocab)
        if scene is not None:
            scene_specs.append(scene)
        if index % 10000 == 0:
            print(f"[SONYC] {index}/{len(annotation_paths)} anotasyon tarandi.")

    selected_scenes, coverage = choose_sonyc_scenes(
        scenes=scene_specs,
        source_root=source_root,
        per_label=args.sonyc_per_label,
        seed=args.sonyc_seed,
    )
    if not selected_scenes:
        raise RuntimeError("SONYC benchmarki icin kullanilabilir hic sahne secilemedi.")

    print(f"[SONYC] Secilen baz sahne sayisi: {len(selected_scenes)}")
    for label in SONYC_LABEL_ORDER:
        if label in coverage:
            print(f"[SONYC] {label:<20} -> {coverage[label]}")

    render_root = os.path.abspath(args.sonyc_render_dir)
    mono_dir = os.path.join(render_root, "mono")
    stereo_dir = os.path.join(render_root, "stereo")
    os.makedirs(mono_dir, exist_ok=True)
    os.makedirs(stereo_dir, exist_ok=True)
    source_cache: dict[str, np.ndarray] = {}

    manifest_path = os.path.abspath(args.sonyc_manifest_out)
    ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
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
                (False, SONYC_MONO_DATASET_NAME, mono_dir, "1"),
                (True, SONYC_STEREO_DATASET_NAME, stereo_dir, "2"),
            ):
                render_index += 1
                output_path = os.path.join(output_dir, f"{scene.clip_id}.wav")
                render_sonyc_scene(
                    scene=scene,
                    source_root=source_root,
                    output_path=output_path,
                    stereo=stereo_flag,
                    seed=args.sonyc_seed,
                    cache=source_cache,
                )
                writer.writerow(
                    {
                        "clip_id": f"{scene.clip_id}_{'stereo' if stereo_flag else 'mono'}",
                        "audio_path": output_path,
                        "label": scene.positive_labels[0],
                        "labels": format_labels(scene.positive_labels),
                        "source_label": format_labels(scene.source_labels),
                        "dataset": dataset_name,
                        "environment": "urban_synthetic",
                        "sensor": "sonyc_fsd_sed",
                        "channels": channels,
                    }
                )
                if render_index == 1 or render_index % 25 == 0 or render_index == total_renders:
                    print(f"[SONYC] Render {render_index}/{total_renders} tamamlandi.")

    return manifest_path


def build_sample_records(args: argparse.Namespace, label_map: dict[str, str]) -> list[SampleRecord]:
    """Read the manifest and normalize all sample rows into one structure."""
    manifest_path = os.path.abspath(args.manifest)
    dataset_root = os.path.abspath(args.dataset_root or os.path.dirname(manifest_path))
    default_dataset_name = args.dataset_name or os.path.splitext(os.path.basename(manifest_path))[0]

    records: list[SampleRecord] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            raw_path = (row.get(args.path_column) or "").strip()
            raw_label = (row.get(args.label_column) or "").strip()
            raw_label_list = (row.get(args.labels_column) or raw_label).strip()
            if not raw_path or not raw_label_list:
                continue

            source_labels = split_label_cell((row.get(args.source_label_column) or raw_label_list).strip(), args.label_separator)
            raw_labels = split_label_cell(raw_label_list, args.label_separator)

            mapped_labels = []
            for label in raw_labels:
                mapped = label_map.get(label, label)
                if mapped not in REDUCED_LABEL_SET:
                    print(
                        f"[WARN] Satir {row_index}: '{mapped}' reduced label setinde yok, atlandi.",
                        file=sys.stderr,
                    )
                    continue
                mapped_labels.append(mapped)

            mapped_labels = list(unique_labels(mapped_labels))
            if not mapped_labels:
                continue

            audio_path = raw_path if os.path.isabs(raw_path) else os.path.join(dataset_root, raw_path)
            clip_id = (row.get(args.clip_id_column) or f"clip-{row_index:05d}").strip()
            dataset_name = (row.get(args.dataset_column) or default_dataset_name).strip() or default_dataset_name

            metadata = {}
            for column_name in OPTIONAL_METADATA_COLUMNS:
                value = (row.get(column_name) or "").strip()
                if value:
                    metadata[column_name] = value

            records.append(
                SampleRecord(
                    clip_id=clip_id,
                    audio_path=os.path.abspath(audio_path),
                    label=mapped_labels[0],
                    source_label=format_labels(source_labels or raw_labels),
                    dataset_name=dataset_name,
                    positive_labels=tuple(mapped_labels),
                    decision_target_labels=unique_labels(collapse_label_for_decision(label) for label in mapped_labels),
                    metadata=metadata,
                    onset_time=parse_optional_float(row.get("onset_time")),
                    prediction_time=parse_optional_float(row.get("prediction_time")),
                    ui_render_time=parse_optional_float(row.get("ui_render_time")),
                    user_response_time=parse_optional_float(row.get("user_response_time")),
                )
            )

            if args.limit is not None and len(records) >= args.limit:
                break

    if not records:
        raise RuntimeError("Manifestten kullanilabilir hic ornek okunamadi.")
    return records


def load_audio_frames(path: str) -> tuple[np.ndarray, int]:
    """Load WAV audio and preserve the original channel count."""
    if sf is not None:
        samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        frames = ensure_frame_major(np.asarray(samples, dtype=np.float32))
    else:
        raw_audio = tf.io.read_file(path)
        waveform, sample_rate = tf.audio.decode_wav(raw_audio)
        frames = ensure_frame_major(waveform.numpy().astype(np.float32, copy=False))
        sample_rate = int(sample_rate.numpy())
    if frames.size == 0:
        raise RuntimeError(f"Bos ses dosyasi: {path}")
    return frames, int(sample_rate)


def iterate_windows(frames: np.ndarray, sample_rate: int, window_seconds: float, hop_seconds: float):
    """Yield overlapping frame windows plus their relative times."""
    total_frames = int(frames.shape[0])
    window_size = max(1, int(round(sample_rate * float(window_seconds))))
    hop_size = max(1, int(round(sample_rate * float(hop_seconds))))

    if total_frames <= window_size:
        yield 0, 0.0, float(total_frames) / float(sample_rate), frames
        return

    start_indices = list(range(0, max(total_frames - window_size + 1, 1), hop_size))
    last_start = max(total_frames - window_size, 0)
    if not start_indices or start_indices[-1] != last_start:
        start_indices.append(last_start)

    for window_index, start in enumerate(start_indices):
        end = min(start + window_size, total_frames)
        yield window_index, float(start) / float(sample_rate), float(end) / float(sample_rate), frames[start:end]


def topk_from_probabilities(labels: list[str], probabilities: np.ndarray, limit: int) -> list[dict]:
    """Convert sorted probabilities into serializable top-k rows."""
    order = np.argsort(probabilities)[::-1]
    return [
        {"label": labels[index], "prob": float(probabilities[index])}
        for index in order[: min(limit, len(order))]
    ]


def label_score_pairs_from_counter(counter: Counter[str], confidence_sums: dict[str, float]) -> list[tuple[str, float]]:
    """Rank labels first by hit count, then by summed confidence."""
    return sorted(
        counter.keys(),
        key=lambda label: (counter[label], confidence_sums[label]),
        reverse=True,
    )


def average_topk(labels: list[str], probabilities: np.ndarray, limit: int) -> list[tuple[str, float]]:
    """Return a stable top-k list from one averaged probability vector."""
    order = np.argsort(probabilities)[::-1]
    return [(labels[index], float(probabilities[index])) for index in order[: min(limit, len(order))]]


def compute_ece_from_arrays(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int,
) -> tuple[float, list[dict[str, float]]]:
    """Compute expected calibration error and reliability bins."""
    if confidences.size == 0:
        return 0.0, []

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins: list[dict[str, float]] = []
    total = int(confidences.size)

    for index in range(num_bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == num_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        count = int(np.sum(mask))
        if count == 0:
            bins.append(
                {
                    "bin_left": left,
                    "bin_right": right,
                    "count": 0,
                    "accuracy": 0.0,
                    "avg_confidence": 0.0,
                }
            )
            continue

        bin_accuracy = float(np.mean(correctness[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += abs(bin_accuracy - bin_confidence) * (count / total)
        bins.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "accuracy": bin_accuracy,
                "avg_confidence": bin_confidence,
            }
        )

    return ece, bins


def collect_latency_ms(samples: list[SampleRecord], start_field: str, end_field: str) -> list[float]:
    """Collect optional latencies in milliseconds from manifest timestamps."""
    values_ms = []
    for sample in samples:
        start_value = getattr(sample, start_field)
        end_value = getattr(sample, end_field)
        if start_value is None or end_value is None:
            continue
        values_ms.append((end_value - start_value) * 1000.0)
    return values_ms


def summarize_latency(values_ms: list[float]) -> tuple[str | float, str | float]:
    """Return mean and median latency if the values exist."""
    if not values_ms:
        return "", ""
    array = np.asarray(values_ms, dtype=np.float64)
    return float(np.mean(array)), float(np.median(array))


def prepare_csv_for_append(path: str, fieldnames: list[str]) -> None:
    """Rotate an incompatible CSV so the shared benchmark file stays consistent."""
    ensure_parent_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, [])

    if existing_header == fieldnames:
        return

    backup_path = f"{path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    os.replace(path, backup_path)
    print(f"[WARN] Eski CSV basligi uyusmadi, yedeklendi: {backup_path}")


def append_summary_rows(csv_path: str, rows: list[dict[str, str | float | int]]) -> None:
    """Append metric rows into the shared benchmark CSV."""
    prepare_csv_for_append(csv_path, SUMMARY_FIELDNAMES)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_prediction_rows(
    csv_path: str,
    window_records: list[WindowRecord],
    run_id: str,
    model_label: str,
    run_timestamp_unix: float,
) -> None:
    """Append one row per evaluated window for detailed inspection."""
    prepare_csv_for_append(csv_path, PREDICTION_FIELDNAMES)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))

    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for record in window_records:
            row = {
                "timestamp_utc": timestamp_iso,
                "run_id": run_id,
                "model_label": model_label,
                "pipeline_variant": record.pipeline_variant,
                "dataset_name": record.sample.dataset_name,
                "clip_id": record.sample.clip_id,
                "audio_path": record.sample.audio_path,
                "source_label": record.sample.source_label,
                "target_label": record.sample.label,
                "target_labels": format_labels(record.sample.positive_labels),
                "decision_target_labels": format_labels(record.sample.decision_target_labels),
                "raw_predicted_label": record.raw_predicted_label,
                "raw_predicted_confidence": f"{record.raw_predicted_confidence:.6f}",
                "raw_top3_labels": json.dumps(record.raw_top3, ensure_ascii=False),
                "dominant_label": record.dominant_label,
                "dominant_confidence": f"{record.dominant_confidence:.6f}",
                "dominant_top3_labels": json.dumps(record.dominant_top3, ensure_ascii=False),
                "window_index": record.window_index,
                "window_start_s": f"{record.window_start_s:.6f}",
                "window_end_s": f"{record.window_end_s:.6f}",
                "duration_seconds": f"{record.duration_seconds:.6f}",
                "is_correct": int(record.dominant_label in decision_target_set(record.sample)),
                "is_raw_correct": int(record.raw_predicted_label in target_labels_for_sample(record.sample)),
                "matches_any_target": int(record.raw_predicted_label in target_labels_for_sample(record.sample)),
                "matches_decision_target": int(record.dominant_label in decision_target_set(record.sample)),
                "input_channels": record.input_channels,
                "spatial_mode": record.spatial_mode,
                "spatial_direction": record.spatial_direction,
                "spatial_direction_confidence": f"{record.spatial_direction_confidence:.6f}",
                "spl_dbfs": f"{record.spl_dbfs:.6f}",
                "decision_state": record.decision_state,
                "decision_priority": record.decision_priority,
                "speech_prob": f"{record.speech_prob:.6f}",
                "speech_gate_state": record.speech_gate_state,
                "speech_gate_opened": int(record.speech_gate_opened),
                "speech_gate_closed": int(record.speech_gate_closed),
                "payload_has_decision": int(record.payload_has_decision),
                "payload_has_spatial": int(record.payload_has_spatial),
                "payload_has_raw_top5": int(record.payload_has_raw_top5),
                "onset_time": record.sample.onset_time if record.sample.onset_time is not None else "",
                "prediction_time": record.sample.prediction_time if record.sample.prediction_time is not None else "",
                "ui_render_time": record.sample.ui_render_time if record.sample.ui_render_time is not None else "",
                "user_response_time": record.sample.user_response_time if record.sample.user_response_time is not None else "",
            }
            for metadata_key in OPTIONAL_METADATA_COLUMNS:
                row[metadata_key] = record.sample.metadata.get(metadata_key, "")
            writer.writerow(row)


def read_summary_rows(csv_path: str) -> list[dict[str, str]]:
    """Read the shared summary CSV if it already exists."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display_label(row: dict[str, str]) -> str:
    """Build a compact legend label for charts."""
    variant = row.get("pipeline_variant", "")
    model_label = row.get("model_label", "")
    return f"{model_label} [{variant}]" if variant else model_label


def latest_overall_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the latest row per dataset/model/variant triple for plots."""
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in summary_rows:
        if row["metric_scope"] != "overall":
            continue
        key = (row["dataset_name"], row["model_label"], row.get("pipeline_variant", ""))
        previous = latest.get(key)
        if previous is None or row["timestamp_utc"] > previous["timestamp_utc"]:
            latest[key] = row
    return list(latest.values())


def plot_history(summary_rows: list[dict[str, str]], plot_path: str) -> None:
    """Generate a multilabel-aware classification comparison chart."""
    overall_rows = latest_overall_rows(summary_rows)
    if not overall_rows:
        return

    ensure_parent_dir(plot_path)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        grouped_rows[row["dataset_name"]].append(row)

    dataset_names = sorted(grouped_rows.keys())
    legend_labels = sorted({display_label(row) for row in overall_rows})
    color_map = {label: plt.cm.Set2(index % 8) for index, label in enumerate(legend_labels)}
    metrics = [
        ("accuracy", "Raw Top-1 Hit Orani", 0.0, 1.08),
        ("topk_hit_rate", "Raw Top-K Hit Orani", 0.0, 1.08),
        ("f1", "Multilabel Macro F1@K", 0.0, 1.08),
        ("decision_target_hit_rate", "HUD Karar Hit Orani", 0.0, 1.08),
        ("ece", "ECE", 0.0, 1.08),
        ("false_alerts_per_minute", "Yanlis Alarm / Dakika", 0.0, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, (metric_key, metric_title, y_min, y_max) in zip(axes_flat, metrics):
        max_bars = max(len(grouped_rows[name]) for name in dataset_names)
        bar_width = 0.72 / max(max_bars, 1)
        x_positions = np.arange(len(dataset_names))
        max_value = 0.0

        for bar_index in range(max_bars):
            positions = []
            values = []
            labels = []
            colors = []
            for dataset_index, dataset_name in enumerate(dataset_names):
                dataset_rows = sorted(
                    grouped_rows[dataset_name],
                    key=lambda row: display_label(row),
                )
                if bar_index >= len(dataset_rows):
                    continue
                row = dataset_rows[bar_index]
                raw_value = row.get(metric_key, "")
                value = float(raw_value) if raw_value not in ("", None) else 0.0
                legend_label = display_label(row)
                positions.append(dataset_index - 0.36 + (bar_index + 0.5) * bar_width)
                values.append(value)
                labels.append(legend_label)
                colors.append(color_map[legend_label])
                max_value = max(max_value, value)

            if not positions:
                continue

            bars = axis.bar(positions, values, width=bar_width, color=colors, edgecolor="white", linewidth=1.0)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(0.01, max_value * 0.03),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.set_title(metric_title)
        axis.set_ylabel("Skor")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(dataset_names, rotation=0)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(y_min, (y_max if y_max is not None else max(1.0, max_value * 1.35 + 0.05)))

    for axis in axes_flat[len(metrics):]:
        axis.axis("off")

    handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=8, label=label)
        for label in legend_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False)
    fig.suptitle("Multilabel siniflandirma ve karar ozeti", fontsize=16)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_history(summary_rows: list[dict[str, str]], plot_path: str) -> None:
    """Generate a behavior-focused comparison chart from the shared CSV."""
    overall_rows = latest_overall_rows(summary_rows)
    if not overall_rows:
        return

    ensure_parent_dir(plot_path)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        grouped_rows[row["dataset_name"]].append(row)

    dataset_names = sorted(grouped_rows.keys())
    legend_labels = sorted({display_label(row) for row in overall_rows})
    color_map = {label: plt.cm.Accent(index % 8) for index, label in enumerate(legend_labels)}
    metrics = [
        ("flapping_transitions_per_min", "Flapping / Dakika", 0.0, None),
        ("speech_gate_toggles_per_min", "Speech Gate Toggle / Dakika", 0.0, None),
        ("payload_decision_rate", "Payload Decision Orani", 0.0, 1.08),
        ("payload_spatial_rate", "Payload Spatial Orani", 0.0, 1.08),
        ("mono_raw_match_rate", "Mono Raw Eslesme Orani", 0.0, 1.08),
        ("other_rate", "Other Cikis Orani", 0.0, 1.08),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, (metric_key, metric_title, y_min, y_max) in zip(axes_flat, metrics):
        max_bars = max(len(grouped_rows[name]) for name in dataset_names)
        bar_width = 0.72 / max(max_bars, 1)
        x_positions = np.arange(len(dataset_names))
        max_value = 0.0

        for bar_index in range(max_bars):
            positions = []
            values = []
            colors = []
            for dataset_index, dataset_name in enumerate(dataset_names):
                dataset_rows = sorted(grouped_rows[dataset_name], key=lambda row: display_label(row))
                if bar_index >= len(dataset_rows):
                    continue
                row = dataset_rows[bar_index]
                raw_value = row.get(metric_key, "")
                value = float(raw_value) if raw_value not in ("", None) else 0.0
                legend_label = display_label(row)
                positions.append(dataset_index - 0.36 + (bar_index + 0.5) * bar_width)
                values.append(value)
                colors.append(color_map[legend_label])
                max_value = max(max_value, value)

            if not positions:
                continue

            bars = axis.bar(positions, values, width=bar_width, color=colors, edgecolor="white", linewidth=1.0)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(0.01, max_value * 0.03 if max_value > 0 else 0.02),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.set_title(metric_title)
        axis.set_ylabel("Skor")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(dataset_names, rotation=0)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(y_min, (y_max if y_max is not None else max(1.0, max_value * 1.35 + 0.05)))

    for axis in axes_flat[len(metrics):]:
        axis.axis("off")

    handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=8, label=label)
        for label in legend_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False)
    fig.suptitle("Pipeline davranis karsilastirmasi", fontsize=16)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_confusion_matrix(clip_results: list[ClipResult]) -> tuple[list[str], np.ndarray]:
    """Build a primary-label confusion matrix from raw top-1 predictions."""
    labels = [
        label
        for label in sorted(set(REDUCED_LABEL_SET))
        if any(result.sample.label == label or result.raw_predicted_label == label for result in clip_results)
    ]
    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)

    for result in clip_results:
        true_index = index_by_label[result.sample.label]
        pred_index = index_by_label[result.raw_predicted_label]
        matrix[true_index, pred_index] += 1

    return labels, matrix


def write_confusion_matrix_csv(path: str, labels: list[str], matrix: np.ndarray) -> None:
    """Write the confusion matrix as a CSV table."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def plot_confusion_matrix(path: str, labels: list[str], matrix: np.ndarray, title: str) -> None:
    """Render a confusion matrix heatmap."""
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), max(6, len(labels) * 0.55)))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Tahmin etiketi")
    ax.set_ylabel("Gercek etiket")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if len(labels) <= 20:
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if value == 0:
                    continue
                ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_curve(
    path: str,
    calibration_bins: list[dict[str, float]],
    ece: float,
    title: str,
) -> None:
    """Render a reliability curve using the computed calibration bins."""
    ensure_parent_dir(path)
    if not calibration_bins:
        return

    centers = [(item["bin_left"] + item["bin_right"]) / 2.0 for item in calibration_bins]
    accuracies = [item["accuracy"] for item in calibration_bins]
    confidences = [item["avg_confidence"] for item in calibration_bins]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Mukemmel kalibrasyon")
    ax.plot(centers, accuracies, marker="o", linewidth=2, label="Gercek dogruluk")
    ax.bar(centers, confidences, width=0.08, alpha=0.25, label="Ort. guven")
    ax.set_title(f"{title}\nECE={ece:.4f}")
    ax.set_xlabel("Guven bin merkezi")
    ax.set_ylabel("Skor")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_state_distribution(path: str, window_records: list[WindowRecord], title: str) -> None:
    """Render a compact state-ratio chart for the new pipeline outputs."""
    if not window_records:
        return
    ensure_parent_dir(path)

    total = float(len(window_records))
    categories = [
        ("silence", sum(record.dominant_label in {"silence", "Silence"} for record in window_records) / total),
        ("idle", sum(record.dominant_label == "idle" for record in window_records) / total),
        ("other", sum(record.dominant_label == "other" for record in window_records) / total),
        ("ood", sum(record.dominant_label == "ood" for record in window_records) / total),
        (
            "active",
            sum(record.dominant_label not in NEUTRAL_OUTPUT_LABELS for record in window_records) / total,
        ),
    ]
    colors = ["#9ca3af", "#60a5fa", "#f59e0b", "#ef4444", "#10b981"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([item[0] for item in categories], [item[1] for item in categories], color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Oran")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for index, (_, value) in enumerate(categories):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_class_report(summary_rows: list[dict[str, str | float | int]], plot_path: str) -> None:
    """Render a grouped multilabel class report without overlapping labels."""
    class_rows = [row for row in summary_rows if row.get("metric_scope") == "class"]
    if not class_rows:
        return

    grouped: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    for row in class_rows:
        grouped[str(row["dataset_name"])].append(row)

    dataset_names = sorted(grouped.keys())
    metrics = [("precision", "Precision@K"), ("recall", "Recall@K"), ("f1", "F1@K")]
    fig, axes = plt.subplots(
        len(dataset_names),
        len(metrics),
        figsize=(18, max(4.5, 4.4 * len(dataset_names))),
        constrained_layout=True,
        squeeze=False,
    )

    legend_handles = {}
    for row_index, dataset_name in enumerate(dataset_names):
        dataset_rows = grouped[dataset_name]
        variant_names = sorted({str(row["pipeline_variant"]) for row in dataset_rows})
        support_by_label = defaultdict(int)
        for row in dataset_rows:
            support_by_label[str(row["class_label"])] = max(
                support_by_label[str(row["class_label"])],
                int(float(row["sample_count"])),
            )

        top_labels = [
            label
            for label, _support in sorted(
                support_by_label.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if support_by_label[label] > 0
        ][:12]
        if not top_labels:
            top_labels = sorted(support_by_label.keys())[:12]

        y_positions = np.arange(len(top_labels))
        bar_height = 0.72 / max(len(variant_names), 1)
        color_map = {name: plt.cm.Set2(index % 8) for index, name in enumerate(variant_names)}

        for col_index, (metric_key, metric_title) in enumerate(metrics):
            axis = axes[row_index][col_index]
            max_value = 0.0
            for variant_index, variant_name in enumerate(variant_names):
                offsets = y_positions - 0.36 + (variant_index + 0.5) * bar_height
                values = []
                for label in top_labels:
                    matching_row = next(
                        (
                            row
                            for row in dataset_rows
                            if str(row["pipeline_variant"]) == variant_name and str(row["class_label"]) == label
                        ),
                        None,
                    )
                    value = float(matching_row[metric_key]) if matching_row and matching_row[metric_key] not in ("", None) else 0.0
                    values.append(value)
                    max_value = max(max_value, value)

                bars = axis.barh(
                    offsets,
                    values,
                    height=bar_height,
                    color=color_map[variant_name],
                    edgecolor="white",
                    linewidth=0.8,
                    label=variant_name,
                )
                legend_handles[variant_name] = bars[0]
                for bar, value in zip(bars, values):
                    axis.text(
                        min(value + 0.015, 1.03),
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{value:.2f}",
                        va="center",
                        ha="left",
                        fontsize=8,
                    )

            axis.set_yticks(y_positions)
            axis.set_yticklabels(top_labels)
            axis.invert_yaxis()
            axis.set_xlim(0.0, max(1.0, max_value * 1.15 + 0.04))
            axis.grid(True, axis="x", linestyle="--", alpha=0.3)
            axis.set_title(f"{dataset_name} - {metric_title}")
            if col_index == 0:
                axis.set_ylabel("Etiket")
            else:
                axis.set_ylabel("")
            axis.set_xlabel("Skor")

    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper center",
        ncol=min(len(legend_handles), 4),
        frameon=False,
    )
    fig.suptitle("Multilabel sinif bazli rapor", fontsize=16)
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def overall_row_for(
    summary_rows: list[dict[str, str | float | int]],
    dataset_name: str,
    pipeline_variant: str,
) -> dict[str, str | float | int] | None:
    """Find one overall row for the given dataset and variant."""
    return next(
        (
            row
            for row in summary_rows
            if row.get("metric_scope") == "overall"
            and str(row.get("dataset_name")) == dataset_name
            and str(row.get("pipeline_variant")) == pipeline_variant
        ),
        None,
    )


def plot_core_run_summary(summary_rows: list[dict[str, str | float | int]], plot_path: str) -> None:
    """Plot the main decision metrics for the mono benchmark."""
    legacy_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "old_pipeline")
    current_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "current_pipeline")
    if legacy_row is None or current_row is None:
        return

    metrics = [
        ("Raw Top-1", "accuracy"),
        ("Raw Top-3", "topk_hit_rate"),
        ("Macro F1", "f1"),
        ("HUD Hit", "decision_target_hit_rate"),
        ("Aksiyon Hit", "actionable_hud_hit_rate"),
    ]
    labels = [item[0] for item in metrics]
    old_values = [float(legacy_row[item[1]]) for item in metrics]
    new_values = [float(current_row[item[1]]) for item in metrics]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.5))
    old_bars = ax.bar(x - width / 2, old_values, width=width, color="#8aa4d6", label="Eski pipeline")
    new_bars = ax.bar(x + width / 2, new_values, width=width, color="#2f6fed", label="Yeni pipeline")

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Temel sonuc ozeti")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (old_bars, new_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_decision_confusion(clip_results: list[ClipResult]) -> tuple[list[str], np.ndarray]:
    """Build a compact confusion matrix in the HUD decision space."""
    labels = []
    for result in clip_results:
        labels.append(primary_hud_target(result.sample))
        labels.append(result.predicted_label)
    label_order = [label for label in ("glass_break", "alarms_buzzer", "door_knock", "explosion_gunshot", "idle", "other", "ood", "silence") if label in labels]
    index_by_label = {label: index for index, label in enumerate(label_order)}
    matrix = np.zeros((len(label_order), len(label_order)), dtype=np.int32)

    for result in clip_results:
        true_label = primary_hud_target(result.sample)
        pred_label = result.predicted_label
        if true_label not in index_by_label or pred_label not in index_by_label:
            continue
        matrix[index_by_label[true_label], index_by_label[pred_label]] += 1
    return label_order, matrix


def plot_confusion_pair(
    legacy_results: list[ClipResult],
    current_results: list[ClipResult],
    plot_path: str,
) -> None:
    """Plot old/new decision-space confusion matrices side by side."""
    legacy_labels, legacy_matrix = build_decision_confusion(legacy_results)
    current_labels, current_matrix = build_decision_confusion(current_results)
    labels = list(dict.fromkeys([*legacy_labels, *current_labels]))
    if not labels:
        return

    def align_matrix(matrix_labels: list[str], matrix: np.ndarray) -> np.ndarray:
        aligned = np.zeros((len(labels), len(labels)), dtype=np.int32)
        lookup = {label: idx for idx, label in enumerate(matrix_labels)}
        for true_label in matrix_labels:
            for pred_label in matrix_labels:
                aligned[labels.index(true_label), labels.index(pred_label)] = matrix[lookup[true_label], lookup[pred_label]]
        return aligned

    legacy_aligned = align_matrix(legacy_labels, legacy_matrix)
    current_aligned = align_matrix(current_labels, current_matrix)
    vmax = max(int(legacy_aligned.max()), int(current_aligned.max()), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], legacy_aligned, "Eski pipeline"),
        (axes[1], current_aligned, "Yeni pipeline"),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if value > 0:
                    ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8, color="#0f172a")

    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.suptitle("Mono karar uzayinda karisiklik matrisi", fontsize=15)
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_run_summary(summary_rows: list[dict[str, str | float | int]], plot_path: str) -> None:
    """Plot the stability metrics that matter for the UX side."""
    legacy_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "old_pipeline")
    current_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "current_pipeline")
    if legacy_row is None or current_row is None:
        return

    metrics = [
        ("Flapping/dk", "flapping_transitions_per_min"),
        ("Gate toggle/dk", "speech_gate_toggles_per_min"),
        ("Yanlis alarm/dk", "false_alerts_per_minute"),
        ("Other orani", "other_rate"),
    ]
    labels = [item[0] for item in metrics]
    old_values = [float(legacy_row[item[1]]) for item in metrics]
    new_values = [float(current_row[item[1]]) for item in metrics]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.5))
    old_bars = ax.bar(x - width / 2, old_values, width=width, color="#b7c5e0", label="Eski pipeline")
    new_bars = ax.bar(x + width / 2, new_values, width=width, color="#4c84ff", label="Yeni pipeline")

    ax.set_title("Davranis ve stabilite ozeti")
    ax.set_ylabel("Deger")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    ymax = max(max(old_values), max(new_values), 1.0)
    ax.set_ylim(0.0, ymax * 1.22)
    for bars in (old_bars, new_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + ymax * 0.03, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_arhud_summary(summary_rows: list[dict[str, str | float | int]], plot_path: str) -> None:
    """Plot the research-facing HUD expectations in one compact figure."""
    legacy_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "old_pipeline")
    current_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "current_pipeline")
    if legacy_row is None or current_row is None:
        return

    metrics = [
        ("Aksiyon yakalama", "actionable_hud_hit_rate"),
        ("Notr bastirma", "non_actionable_suppression_rate"),
        ("Genel HUD hit", "decision_target_hit_rate"),
    ]
    labels = [item[0] for item in metrics]
    old_values = [float(legacy_row[item[1]]) for item in metrics]
    new_values = [float(current_row[item[1]]) for item in metrics]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.2))
    old_bars = ax.bar(x - width / 2, old_values, width=width, color="#c3d2ea", label="Eski pipeline")
    new_bars = ax.bar(x + width / 2, new_values, width=width, color="#2563eb", label="Yeni pipeline")

    ax.set_ylim(0.0, 1.05)
    ax.set_title("AR-HUD farki")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (old_bars, new_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stereo_mono_summary(summary_rows: list[dict[str, str | float | int]], plot_path: str) -> None:
    """Compare mono preservation and stereo readiness for the new pipeline only."""
    mono_row = overall_row_for(summary_rows, SONYC_MONO_DATASET_NAME, "current_pipeline")
    stereo_row = overall_row_for(summary_rows, SONYC_STEREO_DATASET_NAME, "current_pipeline")
    if mono_row is None or stereo_row is None:
        return

    metrics = [
        ("Raw Top-1", "accuracy"),
        ("HUD Hit", "decision_target_hit_rate"),
        ("Aksiyon Hit", "actionable_hud_hit_rate"),
        ("Spatial aktif", "stereo_spatial_active_rate"),
    ]
    mono_values = [float(mono_row[item[1]] or 0.0) for item in metrics]
    stereo_values = [float(stereo_row[item[1]] or 0.0) for item in metrics]
    labels = [item[0] for item in metrics]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    mono_bars = ax.bar(x - width / 2, mono_values, width=width, color="#93c5fd", label="Mono render")
    stereo_bars = ax.bar(x + width / 2, stereo_values, width=width, color="#1d4ed8", label="Sentetik stereo render")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Mono - stereo farki")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (mono_bars, stereo_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_parent_dir(plot_path)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


class LegacySpeechGateEmulator:
    """Replica of the old speech-gate behavior for offline evaluation."""

    def __init__(self) -> None:
        self.gate_state = "IDLE"

    def update(
        self,
        raw_label: str,
        raw_confidence: float,
        labels: list[str],
        probabilities: np.ndarray,
    ) -> tuple[str, float, bool, bool]:
        try:
            speech_index = labels.index(legacy_config.SPEECH_LABEL)
            speech_prob = float(probabilities[speech_index])
            top_is_speech = labels[int(np.argmax(probabilities))] == legacy_config.SPEECH_LABEL
        except Exception:
            speech_prob = raw_confidence if raw_label == legacy_config.SPEECH_LABEL else 0.0
            top_is_speech = raw_label == legacy_config.SPEECH_LABEL

        cond_on = speech_prob >= legacy_config.SPEECH_ON_THRESH and (
            top_is_speech if legacy_config.REQUIRE_TOP_IS_SPEECH else True
        )
        cond_off = speech_prob < legacy_config.SPEECH_OFF_THRESH or (
            legacy_config.REQUIRE_TOP_IS_SPEECH and not top_is_speech
        )

        gate_opened = False
        gate_closed = False
        if self.gate_state == "IDLE":
            if cond_on:
                self.gate_state = "RECORDING"
                gate_opened = True
            return self.gate_state, speech_prob, gate_opened, gate_closed

        if self.gate_state == "RECORDING":
            if cond_off:
                self.gate_state = "COOLDOWN"
                gate_closed = True
            return self.gate_state, speech_prob, gate_opened, gate_closed

        if self.gate_state == "COOLDOWN" and not cond_off:
            self.gate_state = "IDLE"

        return self.gate_state, speech_prob, gate_opened, gate_closed


class CurrentSpeechGateEmulator:
    """Replica of the current speech-gate behavior for offline evaluation."""

    def __init__(self) -> None:
        self.gate_state = "IDLE"
        self._speech_on_since: float | None = None
        self._speech_off_since: float | None = None

    def update(
        self,
        decision: DecisionSnapshot,
        raw_top_label: str,
        spl_dbfs: float,
        now_ts: float,
    ) -> tuple[str, bool, bool]:
        energy_ok = spl_dbfs >= (decision.noise_floor_dbfs + current_config.SPEECH_ENERGY_MARGIN_DB)
        top_is_speech = raw_top_label == current_config.SPEECH_LABEL
        speech_prob = float(decision.speech_prob)
        cond_on = speech_prob >= current_config.SPEECH_ON_THRESH and energy_ok and (
            top_is_speech if current_config.REQUIRE_TOP_IS_SPEECH else True
        )
        cond_off = (speech_prob < current_config.SPEECH_OFF_THRESH) or (not energy_ok) or (
            current_config.REQUIRE_TOP_IS_SPEECH and not top_is_speech
        )

        if cond_on:
            if self._speech_on_since is None:
                self._speech_on_since = now_ts
            self._speech_off_since = None
        else:
            self._speech_on_since = None

        if cond_off:
            if self._speech_off_since is None:
                self._speech_off_since = now_ts
        else:
            self._speech_off_since = None

        gate_opened = False
        gate_closed = False
        if self.gate_state == "IDLE":
            if self._speech_on_since is not None and (now_ts - self._speech_on_since) >= current_config.SPEECH_MIN_ON_S:
                self.gate_state = "RECORDING"
                gate_opened = True
            return self.gate_state, gate_opened, gate_closed

        if self.gate_state == "RECORDING":
            if self._speech_off_since is not None and (
                (now_ts - self._speech_off_since) >= current_config.SPEECH_MIN_OFF_S
            ):
                self.gate_state = "COOLDOWN"
                gate_closed = True
            return self.gate_state, gate_opened, gate_closed

        if self.gate_state == "COOLDOWN" and not cond_off:
            self.gate_state = "IDLE"

        return self.gate_state, gate_opened, gate_closed


def build_legacy_payload(
    window_end: float,
    classify_window_s: float,
    top5: list[dict],
    dominant_label: str,
    dominant_prob: float,
    spl_dbfs: float,
) -> dict:
    """Mirror the old HTTP bridge payload shape."""
    return {
        "window_start_unix": window_end - classify_window_s,
        "window_end_unix": window_end,
        "top5": top5,
        "dominant_label": dominant_label,
        "dominant_prob": dominant_prob,
        "spl_dbfs": float(spl_dbfs),
    }


def build_current_payload(
    window_end: float,
    classify_window_s: float,
    top5: list[dict],
    dominant_label: str,
    dominant_prob: float,
    spl_dbfs: float,
    decision: dict | None,
    spatial: dict | None,
    raw_top5: list[dict] | None,
) -> dict:
    """Mirror the current HTTP bridge payload shape."""
    payload = {
        "window_start_unix": window_end - classify_window_s,
        "window_end_unix": window_end,
        "top5": top5,
        "dominant_label": dominant_label,
        "dominant_prob": dominant_prob,
        "spl_dbfs": float(spl_dbfs),
    }
    if decision is not None:
        payload["decision"] = decision
    if spatial is not None:
        payload["spatial"] = spatial
    if raw_top5 is not None:
        payload["raw_top5"] = raw_top5
    return payload


def run_legacy_window(
    labels: list[str],
    probabilities: np.ndarray,
    spl_dbfs: float,
    timestamp_s: float,
    classify_window_s: float,
    gate: LegacySpeechGateEmulator,
) -> LegacyWindowOutput:
    """Evaluate one window with the old pipeline behavior."""
    raw_top5 = topk_from_probabilities(labels, probabilities, limit=5)
    raw_top_label = raw_top5[0]["label"]
    raw_top_prob = float(raw_top5[0]["prob"])
    gate_state, speech_prob, gate_opened, gate_closed = gate.update(raw_top_label, raw_top_prob, labels, probabilities)
    payload = build_legacy_payload(
        window_end=timestamp_s,
        classify_window_s=classify_window_s,
        top5=raw_top5,
        dominant_label=raw_top_label,
        dominant_prob=raw_top_prob,
        spl_dbfs=spl_dbfs,
    )
    return LegacyWindowOutput(
        raw_top5=raw_top5,
        dominant_label=raw_top_label,
        dominant_prob=raw_top_prob,
        speech_prob=speech_prob,
        gate_state=gate_state,
        gate_opened=gate_opened,
        gate_closed=gate_closed,
        payload=payload,
    )


def run_current_window(
    labels: list[str],
    probabilities: np.ndarray,
    frames_window: np.ndarray,
    sample_rate: int,
    spl_dbfs: float,
    timestamp_s: float,
    classify_window_s: float,
    decision_layer: PriorityDecisionLayer,
    gate: CurrentSpeechGateEmulator,
) -> CurrentWindowOutput:
    """Evaluate one window with the current pipeline behavior."""
    raw_top5 = topk_from_probabilities(labels, probabilities, limit=current_config.TOPK_OVERLAY)
    spatial = summarize_spatial_audio(frames_window, float(sample_rate))
    decision = decision_layer.update(labels, probabilities, spl_dbfs, spatial, timestamp_s)
    gate_state, gate_opened, gate_closed = gate.update(decision, raw_top5[0]["label"], spl_dbfs, timestamp_s)
    payload = build_current_payload(
        window_end=timestamp_s,
        classify_window_s=classify_window_s,
        top5=decision.top5,
        dominant_label=decision.label,
        dominant_prob=decision.confidence,
        spl_dbfs=spl_dbfs,
        decision=decision.to_dict(),
        spatial=spatial.to_dict(),
        raw_top5=raw_top5,
    )
    return CurrentWindowOutput(
        raw_top5=raw_top5,
        decision=decision,
        spatial=spatial,
        gate_state=gate_state,
        gate_opened=gate_opened,
        gate_closed=gate_closed,
        payload=payload,
    )


def aggregate_clip_result(
    sample: SampleRecord,
    pipeline_variant: str,
    window_records: list[WindowRecord],
    labels: list[str],
    raw_prob_sum: np.ndarray,
    raw_topk: int,
) -> ClipResult:
    """Collapse one clip's windows into one clip-level result."""
    dominant_counter: Counter[str] = Counter()
    dominant_conf_sums: defaultdict[str, float] = defaultdict(float)
    for record in window_records:
        dominant_counter[record.dominant_label] += 1
        dominant_conf_sums[record.dominant_label] += float(record.dominant_confidence)

    ranked_dominant = label_score_pairs_from_counter(dominant_counter, dominant_conf_sums)
    predicted_label = ranked_dominant[0]
    predicted_confidence = safe_div(dominant_conf_sums[predicted_label], dominant_counter[predicted_label])
    dominant_top3 = [
        (
            label,
            safe_div(dominant_conf_sums[label], dominant_counter[label]),
        )
        for label in ranked_dominant[:3]
    ]

    averaged_probs = raw_prob_sum / max(len(window_records), 1)
    raw_top3 = average_topk(labels, averaged_probs, limit=3)
    raw_predicted_label, raw_predicted_confidence = raw_top3[0]
    raw_eval_labels = tuple(label for label, _score in average_topk(labels, averaged_probs, limit=max(1, raw_topk)))

    return ClipResult(
        sample=sample,
        pipeline_variant=pipeline_variant,
        predicted_label=predicted_label,
        predicted_confidence=float(predicted_confidence),
        dominant_top3=dominant_top3,
        raw_predicted_label=raw_predicted_label,
        raw_predicted_confidence=float(raw_predicted_confidence),
        raw_top3=raw_top3,
        raw_eval_labels=raw_eval_labels,
        duration_seconds=float(window_records[-1].window_end_s if window_records else 0.0),
        windows=window_records,
    )


def evaluate_sample_pair(
    classifier: YamnetClassifier,
    sample: SampleRecord,
    args: argparse.Namespace,
) -> tuple[ClipResult, ClipResult]:
    """Run the old and new pipeline variants on one audio sample."""
    frames, sample_rate = load_audio_frames(sample.audio_path)

    legacy_gate = LegacySpeechGateEmulator()
    current_gate = CurrentSpeechGateEmulator()
    decision_layer = PriorityDecisionLayer()

    legacy_windows: list[WindowRecord] = []
    current_windows: list[WindowRecord] = []
    raw_prob_sum: np.ndarray | None = None
    label_names: list[str] | None = None

    for window_index, window_start_s, window_end_s, frames_window in iterate_windows(
        frames,
        sample_rate,
        args.window_seconds,
        args.hop_seconds,
    ):
        mono_window = downmix_to_mono(frames_window)
        spl_dbfs = 20.0 * np.log10(float(np.sqrt(np.mean(mono_window.astype(np.float64) ** 2) + 1e-9)) + 1e-9)
        labels, probabilities = classifier.predict_all(mono_window, sample_rate)

        if raw_prob_sum is None:
            raw_prob_sum = np.zeros_like(probabilities, dtype=np.float64)
            label_names = list(labels)
        raw_prob_sum += np.asarray(probabilities, dtype=np.float64)

        legacy = run_legacy_window(
            labels=labels,
            probabilities=probabilities,
            spl_dbfs=spl_dbfs,
            timestamp_s=window_end_s,
            classify_window_s=args.window_seconds,
            gate=legacy_gate,
        )
        current = run_current_window(
            labels=labels,
            probabilities=probabilities,
            frames_window=frames_window,
            sample_rate=sample_rate,
            spl_dbfs=spl_dbfs,
            timestamp_s=window_end_s,
            classify_window_s=args.window_seconds,
            decision_layer=decision_layer,
            gate=current_gate,
        )

        input_channels = int(frames_window.shape[1]) if frames_window.ndim == 2 else 1
        duration_seconds = window_end_s - window_start_s
        legacy_windows.append(
            WindowRecord(
                sample=sample,
                pipeline_variant=args.legacy_variant_label,
                window_index=window_index,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                raw_predicted_label=legacy.raw_top5[0]["label"],
                raw_predicted_confidence=float(legacy.raw_top5[0]["prob"]),
                raw_top3=[(item["label"], float(item["prob"])) for item in legacy.raw_top5[:3]],
                dominant_label=legacy.dominant_label,
                dominant_confidence=float(legacy.dominant_prob),
                dominant_top3=[(item["label"], float(item["prob"])) for item in legacy.raw_top5[:3]],
                decision_state="raw",
                decision_priority="none",
                speech_prob=float(legacy.speech_prob),
                speech_gate_state=legacy.gate_state,
                speech_gate_opened=legacy.gate_opened,
                speech_gate_closed=legacy.gate_closed,
                payload_has_decision="decision" in legacy.payload,
                payload_has_spatial="spatial" in legacy.payload,
                payload_has_raw_top5="raw_top5" in legacy.payload,
                spatial_mode="mono" if input_channels <= 1 else "not_available",
                spatial_direction="unknown",
                spatial_direction_confidence=0.0,
                spl_dbfs=float(spl_dbfs),
                duration_seconds=float(duration_seconds),
                input_channels=input_channels,
            )
        )
        current_windows.append(
            WindowRecord(
                sample=sample,
                pipeline_variant=args.current_variant_label,
                window_index=window_index,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                raw_predicted_label=current.raw_top5[0]["label"],
                raw_predicted_confidence=float(current.raw_top5[0]["prob"]),
                raw_top3=[(item["label"], float(item["prob"])) for item in current.raw_top5[:3]],
                dominant_label=current.decision.label,
                dominant_confidence=float(current.decision.confidence),
                dominant_top3=[(item["label"], float(item["prob"])) for item in current.decision.top5[:3]],
                decision_state=current.decision.state,
                decision_priority=current.decision.priority,
                speech_prob=float(current.decision.speech_prob),
                speech_gate_state=current.gate_state,
                speech_gate_opened=current.gate_opened,
                speech_gate_closed=current.gate_closed,
                payload_has_decision="decision" in current.payload,
                payload_has_spatial="spatial" in current.payload,
                payload_has_raw_top5="raw_top5" in current.payload,
                spatial_mode=current.spatial.mode,
                spatial_direction=current.spatial.direction,
                spatial_direction_confidence=float(current.spatial.direction_confidence),
                spl_dbfs=float(spl_dbfs),
                duration_seconds=float(duration_seconds),
                input_channels=input_channels,
            )
        )

    if raw_prob_sum is None or label_names is None:
        raise RuntimeError(f"Degerlendirilebilir pencere olusmadi: {sample.audio_path}")

    legacy_clip = aggregate_clip_result(
        sample,
        args.legacy_variant_label,
        legacy_windows,
        label_names,
        raw_prob_sum,
        args.raw_topk,
    )
    current_clip = aggregate_clip_result(
        sample,
        args.current_variant_label,
        current_windows,
        label_names,
        raw_prob_sum,
        args.raw_topk,
    )
    return legacy_clip, current_clip


def evaluate_samples(
    classifier: YamnetClassifier,
    samples: list[SampleRecord],
    args: argparse.Namespace,
) -> tuple[list[ClipResult], list[WindowRecord]]:
    """Run paired old/new evaluation for each sample in the manifest."""
    clip_results: list[ClipResult] = []
    all_windows: list[WindowRecord] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        try:
            legacy_clip, current_clip = evaluate_sample_pair(classifier, sample, args)
            clip_results.extend([legacy_clip, current_clip])
            all_windows.extend(legacy_clip.windows)
            all_windows.extend(current_clip.windows)
        except Exception as exc:
            print(f"[WARN] Tahmin atlandi: {sample.audio_path} -> {exc}", file=sys.stderr)
            continue

        if index == 1 or index % 10 == 0 or index == total:
            print(f"[EVAL] {index}/{total} ornek tamamlandi.")

    if not clip_results:
        raise RuntimeError("Degerlendirme sonunda hic clip sonucu olusmadi.")
    return clip_results, all_windows


def compute_cross_variant_metrics(
    legacy_windows: list[WindowRecord],
    current_windows: list[WindowRecord],
) -> CrossVariantMetrics:
    """Compare aligned windows emitted by the old and current pipelines."""
    if not legacy_windows or not current_windows:
        return CrossVariantMetrics("", "")

    paired_count = min(len(legacy_windows), len(current_windows))
    dominant_matches = 0
    mono_matches = 0
    mono_count = 0

    for legacy_record, current_record in zip(legacy_windows[:paired_count], current_windows[:paired_count]):
        if legacy_record.dominant_label == current_record.dominant_label:
            dominant_matches += 1

        if legacy_record.input_channels == 1 and current_record.input_channels == 1:
            mono_count += 1
            conf_match = abs(legacy_record.raw_predicted_confidence - current_record.raw_predicted_confidence) < 1e-6
            if legacy_record.raw_predicted_label == current_record.raw_predicted_label and conf_match:
                mono_matches += 1

    return CrossVariantMetrics(
        paired_dominant_agreement_rate=safe_div(dominant_matches, paired_count),
        mono_raw_match_rate=(safe_div(mono_matches, mono_count) if mono_count else ""),
    )


def raw_top1_hit(result: ClipResult) -> bool:
    """Return whether the raw top-1 label hits any positive target label."""
    return result.raw_predicted_label in target_labels_for_sample(result.sample)


def raw_topk_hit(result: ClipResult) -> bool:
    """Return whether the raw top-k set overlaps any positive target label."""
    return bool(set(result.raw_eval_labels) & target_labels_for_sample(result.sample))


def decision_target_hit(result: ClipResult) -> bool:
    """Return whether the decision output matches the HUD-oriented target set."""
    return result.predicted_label in decision_target_set(result.sample)


def actionable_target_labels(sample: SampleRecord) -> set[str]:
    """Return only the actionable target labels for one sample."""
    return decision_target_set(sample) & current_config.ACTIONABLE_LABELS


def is_non_actionable_only(sample: SampleRecord) -> bool:
    """Return whether the sample should collapse to a neutral HUD state."""
    targets = decision_target_set(sample)
    return not actionable_target_labels(sample) and bool(targets) and targets.issubset(NEUTRAL_OUTPUT_LABELS)


def actionable_decision_hit(result: ClipResult) -> bool:
    """Return whether the predicted HUD label hits an actionable target."""
    return result.predicted_label in actionable_target_labels(result.sample)


def neutral_suppression_hit(result: ClipResult) -> bool:
    """Return whether the decision output stays in the intended neutral state."""
    return result.predicted_label in decision_target_set(result.sample)


def primary_hud_target(sample: SampleRecord) -> str:
    """Choose one readable decision target label for compact confusion plots."""
    actionable_targets = actionable_target_labels(sample)
    if actionable_targets:
        return sorted(actionable_targets)[0]
    targets = decision_target_set(sample)
    for candidate in ("idle", "other", "silence", "ood"):
        if candidate in targets:
            return candidate
    return sorted(targets)[0] if targets else collapse_label_for_decision(sample.label)


def multilabel_sample_prf(result: ClipResult) -> tuple[float, float, float]:
    """Compute sample-wise multilabel precision/recall/F1 from the raw top-k set."""
    positives = target_labels_for_sample(result.sample)
    predicted = set(result.raw_eval_labels)
    tp = len(predicted & positives)
    precision = safe_div(tp, len(predicted))
    recall = safe_div(tp, len(positives))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def summarize_variant_rows(
    clip_results: list[ClipResult],
    all_window_records: list[WindowRecord],
    samples: list[SampleRecord],
    dataset_name: str,
    run_id: str,
    model_label: str,
    pipeline_variant: str,
    run_timestamp_unix: float,
    ece_bins: int,
    cross_metrics: CrossVariantMetrics,
) -> tuple[list[dict[str, str | float | int]], float, list[dict[str, float]]]:
    """Build multilabel raw-event rows plus decision-target overall rows."""
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))
    total = len(clip_results)
    total_audio_minutes = safe_div(sum(result.duration_seconds for result in clip_results), 60.0)

    raw_top1_correctness = np.array([1.0 if raw_top1_hit(result) else 0.0 for result in clip_results], dtype=np.float64)
    ece_confidences = np.array([result.raw_predicted_confidence for result in clip_results], dtype=np.float64)
    ece, calibration_bins = compute_ece_from_arrays(ece_confidences, raw_top1_correctness, ece_bins)

    class_support = Counter(label for result in clip_results for label in target_labels_for_sample(result.sample))
    raw_pred_counter = Counter(label for result in clip_results for label in result.raw_eval_labels)
    raw_tp_counter = Counter(
        label
        for result in clip_results
        for label in (set(result.raw_eval_labels) & target_labels_for_sample(result.sample))
    )

    primary_true_counter = Counter(result.sample.label for result in clip_results)
    primary_pred_counter = Counter(result.raw_predicted_label for result in clip_results)
    primary_tp_counter = Counter(
        result.sample.label for result in clip_results if result.raw_predicted_label == result.sample.label
    )

    multilabel_macro_precision_values = []
    multilabel_macro_recall_values = []
    multilabel_macro_f1_values = []
    primary_precision_values = []
    primary_recall_values = []
    primary_f1_values = []
    labels_in_run = sorted(
        set(class_support)
        | set(raw_pred_counter)
        | {result.predicted_label for result in clip_results if result.predicted_label not in NEUTRAL_OUTPUT_LABELS}
    )

    per_class_rows: list[dict[str, str | float | int]] = []
    for label in labels_in_run:
        tp = raw_tp_counter[label]
        fp = raw_pred_counter[label] - tp
        fn = class_support[label] - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        primary_tp = primary_tp_counter[label]
        primary_fp = primary_pred_counter[label] - primary_tp
        primary_fn = primary_true_counter[label] - primary_tp
        primary_precision = safe_div(primary_tp, primary_tp + primary_fp)
        primary_recall = safe_div(primary_tp, primary_tp + primary_fn)
        primary_f1 = safe_div(2 * primary_precision * primary_recall, primary_precision + primary_recall)

        if class_support[label] > 0:
            multilabel_macro_precision_values.append(precision)
            multilabel_macro_recall_values.append(recall)
            multilabel_macro_f1_values.append(f1)
        if primary_true_counter[label] > 0:
            primary_precision_values.append(primary_precision)
            primary_recall_values.append(primary_recall)
            primary_f1_values.append(primary_f1)

        per_class_rows.append(
            {
                "timestamp_utc": timestamp_iso,
                "run_id": run_id,
                "model_label": model_label,
                "pipeline_variant": pipeline_variant,
                "dataset_name": dataset_name,
                "metric_scope": "class",
                "class_label": label,
                "sample_count": class_support[label],
                "correct_count": tp,
                "total_audio_minutes": "",
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "topk_hit_rate": "",
                "decision_target_hit_rate": "",
                "actionable_hud_hit_rate": "",
                "non_actionable_suppression_rate": "",
                "raw_accuracy": safe_div(primary_tp, primary_true_counter[label]),
                "raw_precision": primary_precision,
                "raw_recall": primary_recall,
                "raw_f1": primary_f1,
                "avg_confidence": safe_div(
                    sum(result.raw_predicted_confidence for result in clip_results if label in target_labels_for_sample(result.sample)),
                    class_support[label],
                ),
                "raw_avg_confidence": safe_div(
                    sum(result.raw_predicted_confidence for result in clip_results if result.sample.label == label),
                    primary_true_counter[label],
                ),
                "ece": "",
                "false_alerts_per_minute": "",
                "flapping_transitions_per_min": "",
                "speech_gate_toggles_per_min": "",
                "speech_gate_open_ratio": "",
                "payload_decision_rate": "",
                "payload_spatial_rate": "",
                "payload_raw_top5_rate": "",
                "paired_dominant_agreement_rate": "",
                "mono_raw_match_rate": "",
                "mono_spatial_valid_rate": "",
                "stereo_spatial_active_rate": "",
                "silence_rate": "",
                "idle_rate": "",
                "other_rate": "",
                "ood_rate": "",
                "onset_latency_ms_mean": "",
                "onset_latency_ms_p50": "",
                "ui_latency_ms_mean": "",
                "ui_latency_ms_p50": "",
                "human_response_ms_mean": "",
                "human_response_ms_p50": "",
            }
        )

    windows_by_clip: dict[str, list[WindowRecord]] = defaultdict(list)
    for record in all_window_records:
        windows_by_clip[record.sample.clip_id].append(record)
    for records in windows_by_clip.values():
        records.sort(key=lambda item: item.window_index)

    transitions = sum(
        sum(
            1
            for prev, curr in zip(records, records[1:])
            if prev.dominant_label != curr.dominant_label
        )
        for records in windows_by_clip.values()
    )

    wrong_alert_starts = 0
    for records in windows_by_clip.values():
        previous_wrong_label = None
        for record in records:
            target_decisions = decision_target_set(record.sample)
            is_wrong_alert = record.dominant_label not in NEUTRAL_OUTPUT_LABELS and record.dominant_label not in target_decisions
            current_wrong_label = record.dominant_label if is_wrong_alert else None
            if current_wrong_label is not None and current_wrong_label != previous_wrong_label:
                wrong_alert_starts += 1
            previous_wrong_label = current_wrong_label

    total_window_count = max(len(all_window_records), 1)
    payload_decision_rate = safe_div(sum(record.payload_has_decision for record in all_window_records), total_window_count)
    payload_spatial_rate = safe_div(sum(record.payload_has_spatial for record in all_window_records), total_window_count)
    payload_raw_top5_rate = safe_div(sum(record.payload_has_raw_top5 for record in all_window_records), total_window_count)
    speech_gate_toggles = sum(record.speech_gate_opened or record.speech_gate_closed for record in all_window_records)
    speech_gate_open_ratio = safe_div(
        sum(record.speech_gate_state == "RECORDING" for record in all_window_records),
        total_window_count,
    )

    mono_records = [record for record in all_window_records if record.input_channels == 1]
    stereo_records = [record for record in all_window_records if record.input_channels > 1]
    mono_spatial_valid_rate = (
        safe_div(
            sum(record.spatial_mode == "mono" and record.spatial_direction in {"unknown", ""} for record in mono_records),
            len(mono_records),
        )
        if mono_records
        else ""
    )
    stereo_spatial_active_rate = (
        safe_div(sum(record.spatial_mode == "stereo" for record in stereo_records), len(stereo_records))
        if stereo_records
        else ""
    )

    onset_latency_ms = collect_latency_ms(samples, "onset_time", "prediction_time")
    ui_latency_ms = collect_latency_ms(samples, "prediction_time", "ui_render_time")
    human_response_ms = collect_latency_ms(samples, "ui_render_time", "user_response_time")
    onset_mean, onset_p50 = summarize_latency(onset_latency_ms)
    ui_mean, ui_p50 = summarize_latency(ui_latency_ms)
    human_mean, human_p50 = summarize_latency(human_response_ms)

    sample_precisions = []
    sample_recalls = []
    sample_f1s = []
    for result in clip_results:
        precision, recall, f1 = multilabel_sample_prf(result)
        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1s.append(f1)

    actionable_results = [result for result in clip_results if actionable_target_labels(result.sample)]
    non_actionable_results = [result for result in clip_results if is_non_actionable_only(result.sample)]

    overall_row = {
        "timestamp_utc": timestamp_iso,
        "run_id": run_id,
        "model_label": model_label,
        "pipeline_variant": pipeline_variant,
        "dataset_name": dataset_name,
        "metric_scope": "overall",
        "class_label": "__overall__",
        "sample_count": total,
        "correct_count": int(np.sum(raw_top1_correctness)),
        "total_audio_minutes": total_audio_minutes,
        "accuracy": safe_div(np.sum(raw_top1_correctness), total),
        "precision": safe_div(sum(multilabel_macro_precision_values), len(multilabel_macro_precision_values)),
        "recall": safe_div(sum(multilabel_macro_recall_values), len(multilabel_macro_recall_values)),
        "f1": safe_div(sum(multilabel_macro_f1_values), len(multilabel_macro_f1_values)),
        "topk_hit_rate": safe_div(sum(1 for result in clip_results if raw_topk_hit(result)), total),
        "decision_target_hit_rate": safe_div(sum(1 for result in clip_results if decision_target_hit(result)), total),
        "actionable_hud_hit_rate": safe_div(
            sum(1 for result in actionable_results if actionable_decision_hit(result)),
            len(actionable_results),
        ),
        "non_actionable_suppression_rate": safe_div(
            sum(1 for result in non_actionable_results if neutral_suppression_hit(result)),
            len(non_actionable_results),
        ),
        "raw_accuracy": safe_div(sum(1 for result in clip_results if result.raw_predicted_label == result.sample.label), total),
        "raw_precision": safe_div(sum(primary_precision_values), len(primary_precision_values)),
        "raw_recall": safe_div(sum(primary_recall_values), len(primary_recall_values)),
        "raw_f1": safe_div(sum(primary_f1_values), len(primary_f1_values)),
        "avg_confidence": safe_div(sum(result.raw_predicted_confidence for result in clip_results), total),
        "raw_avg_confidence": safe_div(sum(result.raw_predicted_confidence for result in clip_results), total),
        "ece": ece,
        "false_alerts_per_minute": safe_div(wrong_alert_starts, total_audio_minutes),
        "flapping_transitions_per_min": safe_div(transitions, total_audio_minutes),
        "speech_gate_toggles_per_min": safe_div(speech_gate_toggles, total_audio_minutes),
        "speech_gate_open_ratio": speech_gate_open_ratio,
        "payload_decision_rate": payload_decision_rate,
        "payload_spatial_rate": payload_spatial_rate,
        "payload_raw_top5_rate": payload_raw_top5_rate,
        "paired_dominant_agreement_rate": cross_metrics.paired_dominant_agreement_rate,
        "mono_raw_match_rate": cross_metrics.mono_raw_match_rate,
        "mono_spatial_valid_rate": mono_spatial_valid_rate,
        "stereo_spatial_active_rate": stereo_spatial_active_rate,
        "silence_rate": safe_div(
            sum(record.dominant_label in {"silence", "Silence"} for record in all_window_records),
            total_window_count,
        ),
        "idle_rate": safe_div(sum(record.dominant_label == "idle" for record in all_window_records), total_window_count),
        "other_rate": safe_div(sum(record.dominant_label == "other" for record in all_window_records), total_window_count),
        "ood_rate": safe_div(sum(record.dominant_label == "ood" for record in all_window_records), total_window_count),
        "onset_latency_ms_mean": onset_mean,
        "onset_latency_ms_p50": onset_p50,
        "ui_latency_ms_mean": ui_mean,
        "ui_latency_ms_p50": ui_p50,
        "human_response_ms_mean": human_mean,
        "human_response_ms_p50": human_p50,
    }
    return [overall_row, *per_class_rows], ece, calibration_bins

def export_dataset_artifacts(
    clip_results: list[ClipResult],
    window_records: list[WindowRecord],
    dataset_name: str,
    model_label: str,
    pipeline_variant: str,
    run_id: str,
    ece: float,
    calibration_bins: list[dict[str, float]],
) -> list[str]:
    """Write lightweight per-dataset artifacts for later drill-down."""
    _ = window_records
    _ = ece
    _ = calibration_bins
    dataset_slug = slugify(dataset_name)
    model_slug = slugify(model_label)
    variant_slug = slugify(pipeline_variant)
    artifact_prefix = os.path.join(EVAL_DIR, f"{run_id}_{dataset_slug}_{model_slug}_{variant_slug}")

    labels, matrix = build_confusion_matrix(clip_results)
    confusion_csv = f"{artifact_prefix}_confusion_matrix.csv"

    write_confusion_matrix_csv(confusion_csv, labels, matrix)
    return [confusion_csv]


def print_run_summary(rows: Iterable[dict[str, str | float | int]]) -> None:
    """Print the overall rows for terminal visibility."""
    print("")
    for overall_row in (row for row in rows if row["metric_scope"] == "overall"):
        print("[SONUC]")
        print(f"Model etiketi           : {overall_row['model_label']}")
        print(f"Pipeline varyanti       : {overall_row['pipeline_variant']}")
        print(f"Veri kumesi             : {overall_row['dataset_name']}")
        print(f"Clip sayisi             : {overall_row['sample_count']}")
        print(f"Raw Top-1 hit           : {float(overall_row['accuracy']):.4f}")
        print(f"Raw Top-K hit           : {float(overall_row['topk_hit_rate']):.4f}")
        print(f"Multilabel Macro F1@K   : {float(overall_row['f1']):.4f}")
        print(f"HUD karar hit           : {float(overall_row['decision_target_hit_rate']):.4f}")
        print(f"Aksiyon HUD hit         : {float(overall_row['actionable_hud_hit_rate']):.4f}")
        print(f"Notr bastirma orani     : {float(overall_row['non_actionable_suppression_rate']):.4f}")
        print(f"Primary-label accuracy  : {float(overall_row['raw_accuracy']):.4f}")
        print(f"ECE                     : {float(overall_row['ece']):.4f}")
        print(f"Flapping / dakika       : {float(overall_row['flapping_transitions_per_min']):.4f}")
        print(f"Speech gate / dakika    : {float(overall_row['speech_gate_toggles_per_min']):.4f}")
        print(f"Yanlis alarm / dakika   : {float(overall_row['false_alerts_per_minute']):.4f}")
        if overall_row["paired_dominant_agreement_rate"] != "":
            print(f"Varyant anlasma orani   : {float(overall_row['paired_dominant_agreement_rate']):.4f}")
        if overall_row["mono_raw_match_rate"] != "":
            print(f"Mono raw eslesme orani  : {float(overall_row['mono_raw_match_rate']):.4f}")
        print("")


def main() -> None:
    """Entry point for one complete benchmark pass."""
    args = parse_args()
    run_timestamp_unix = time.time()
    run_id = uuid.uuid4().hex[:12]
    label_map = load_label_map(args.label_map_json)

    if args.clear_old_pngs:
        removed_pngs = clean_old_png_artifacts(EVAL_DIR)
        if removed_pngs:
            print(f"[TEMIZLIK] Silinen eski PNG sayisi: {len(removed_pngs)}")

    if args.sonyc_root:
        print("[SONYC] Otomatik manifest ve render seti hazirlaniyor...")
        args.manifest = build_sonyc_manifest(args)
        if not args.dataset_name:
            args.dataset_name = "sonyc_fsd_sed"
        print(f"[SONYC] Manifest hazir: {os.path.abspath(args.manifest)}")

    print("[EVAL] Manifest okunuyor...")
    samples = build_sample_records(args, label_map)
    print(f"[EVAL] Kullanilacak ornek sayisi: {len(samples)}")

    print("[EVAL] Model yukleniyor...")
    try:
        classifier = YamnetClassifier()
    except Exception as exc:
        raise SystemExit(
            "YAMNet siniflandirici baslatilamadi. TensorFlow / model bagimliliklarini "
            f"kontrol et. Asil hata: {exc}"
        ) from exc

    print("[EVAL] Eski ve yeni pipeline birlikte calistiriliyor...")
    clip_results, window_records = evaluate_samples(classifier, samples, args)

    rows_to_append: list[dict[str, str | float | int]] = []
    artifact_paths: list[str] = []
    grouped_clips: dict[tuple[str, str], list[ClipResult]] = defaultdict(list)
    grouped_windows: dict[tuple[str, str], list[WindowRecord]] = defaultdict(list)
    samples_by_dataset: dict[str, list[SampleRecord]] = defaultdict(list)

    for sample in samples:
        samples_by_dataset[sample.dataset_name].append(sample)
    for result in clip_results:
        grouped_clips[(result.sample.dataset_name, result.pipeline_variant)].append(result)
    for record in window_records:
        grouped_windows[(record.sample.dataset_name, record.pipeline_variant)].append(record)

    dataset_names = sorted({sample.dataset_name for sample in samples})
    for dataset_name in dataset_names:
        legacy_windows = grouped_windows.get((dataset_name, args.legacy_variant_label), [])
        current_windows = grouped_windows.get((dataset_name, args.current_variant_label), [])
        cross_metrics = compute_cross_variant_metrics(legacy_windows, current_windows)

        for pipeline_variant in (args.legacy_variant_label, args.current_variant_label):
            dataset_clip_results = grouped_clips.get((dataset_name, pipeline_variant), [])
            dataset_window_records = grouped_windows.get((dataset_name, pipeline_variant), [])
            if not dataset_clip_results:
                continue

            summary_rows, ece, calibration_bins = summarize_variant_rows(
                clip_results=dataset_clip_results,
                all_window_records=dataset_window_records,
                samples=samples_by_dataset[dataset_name],
                dataset_name=dataset_name,
                run_id=run_id,
                model_label=args.model_label,
                pipeline_variant=pipeline_variant,
                run_timestamp_unix=run_timestamp_unix,
                ece_bins=args.ece_bins,
                cross_metrics=cross_metrics,
            )
            rows_to_append.extend(summary_rows)
            artifact_paths.extend(
                export_dataset_artifacts(
                    clip_results=dataset_clip_results,
                    window_records=dataset_window_records,
                    dataset_name=dataset_name,
                    model_label=args.model_label,
                    pipeline_variant=pipeline_variant,
                    run_id=run_id,
                    ece=ece,
                    calibration_bins=calibration_bins,
                )
            )

    append_summary_rows(args.summary_csv, rows_to_append)
    append_prediction_rows(args.predictions_csv, window_records, run_id, args.model_label, run_timestamp_unix)

    plot_core_run_summary(rows_to_append, args.plot_path)
    plot_behavior_run_summary(rows_to_append, args.behavior_plot_path)
    plot_arhud_summary(rows_to_append, args.arhud_plot_path)
    plot_stereo_mono_summary(rows_to_append, args.stereo_mono_plot_path)
    mono_legacy_results = grouped_clips.get((SONYC_MONO_DATASET_NAME, args.legacy_variant_label), [])
    mono_current_results = grouped_clips.get((SONYC_MONO_DATASET_NAME, args.current_variant_label), [])
    plot_confusion_pair(mono_legacy_results, mono_current_results, args.confusion_plot_path)
    print_run_summary(rows_to_append)

    print(f"[KAYIT] Ozet CSV             : {os.path.abspath(args.summary_csv)}")
    print(f"[KAYIT] Detay pencere CSV    : {os.path.abspath(args.predictions_csv)}")
    print(f"[KAYIT] Genel ozet png       : {os.path.abspath(args.plot_path)}")
    print(f"[KAYIT] Confusion png        : {os.path.abspath(args.confusion_plot_path)}")
    print(f"[KAYIT] Davranis png         : {os.path.abspath(args.behavior_plot_path)}")
    print(f"[KAYIT] AR-HUD png           : {os.path.abspath(args.arhud_plot_path)}")
    print(f"[KAYIT] Stereo-mono png      : {os.path.abspath(args.stereo_mono_plot_path)}")
    print(f"[KAYIT] Run ID               : {run_id}")
    for artifact_path in artifact_paths:
        print(f"[KAYIT] Artefakt             : {os.path.abspath(artifact_path)}")


if __name__ == "__main__":
    main()
