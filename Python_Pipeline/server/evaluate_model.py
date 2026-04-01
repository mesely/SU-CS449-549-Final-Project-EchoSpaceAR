"""Evaluate and compare the old and new EchoSpace audio pipelines.

This script intentionally stays in a single file so it is easy to audit,
version, and rerun. It compares the current pipeline files against their
`_old` backups on the same audio clips and writes shared CSV histories plus
comparison figures.

What it compares:
  - legacy raw YAMNet-style pipeline (`*_old` behavior)
  - current pipeline with decision layer + stereo side-channel

What it measures:
  - overall and per-class accuracy / precision / recall / F1
  - raw classifier consistency for mono input
  - confusion matrix and calibration (ECE)
  - flapping / label transitions per minute
  - false alerts per minute
  - speech-gate stability
  - decision / spatial payload presence
  - silence / idle / other / ood output ratios

Required manifest columns:
  - audio_path
  - label

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
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import pipeline_runtime.config as current_config
import pipeline_runtime.config_old as legacy_config
from pipeline_runtime.classification import YamnetClassifier
from pipeline_runtime.config import LOG_DIR, REDUCED_LABEL_SET, tf
from pipeline_runtime.decision_layer import DecisionSnapshot, PriorityDecisionLayer
from pipeline_runtime.spatial_audio import SpatialSnapshot, downmix_to_mono, ensure_frame_major, summarize_spatial_audio

import matplotlib.pyplot as plt


EVAL_DIR = os.path.join(LOG_DIR, "evaluation")
SUMMARY_CSV = os.path.join(EVAL_DIR, "model_benchmark_results.csv")
PREDICTIONS_CSV = os.path.join(EVAL_DIR, "model_benchmark_predictions.csv")
CLASSIFICATION_COMPARISON_PNG = os.path.join(EVAL_DIR, "model_benchmark_comparison.png")
PIPELINE_BEHAVIOR_PNG = os.path.join(EVAL_DIR, "pipeline_behavior_comparison.png")

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
    "raw_predicted_label",
    "raw_predicted_confidence",
    "dominant_label",
    "dominant_confidence",
    "window_index",
    "window_start_s",
    "window_end_s",
    "duration_seconds",
    "is_correct",
    "is_raw_correct",
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
    dominant_label: str
    dominant_confidence: float
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
    duration_seconds: float
    windows: list[WindowRecord]


@dataclass
class CrossVariantMetrics:
    """Metrics that depend on looking at both variants on aligned windows."""

    paired_dominant_agreement_rate: float | str
    mono_raw_match_rate: float | str


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
    parser.add_argument("--manifest", required=True, help="CSV manifest with audio_path and label columns.")
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
        help="Manifest column containing the target label.",
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
        default=CLASSIFICATION_COMPARISON_PNG,
        help="PNG path for the classification comparison chart.",
    )
    parser.add_argument(
        "--behavior-plot-path",
        default=PIPELINE_BEHAVIOR_PNG,
        help="PNG path for the pipeline-behavior comparison chart.",
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
    return parser.parse_args()


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
            if not raw_path or not raw_label:
                continue

            source_label = (row.get(args.source_label_column) or raw_label).strip()
            mapped_label = label_map.get(raw_label, raw_label)
            if mapped_label not in REDUCED_LABEL_SET:
                print(
                    f"[WARN] Satir {row_index}: '{mapped_label}' reduced label setinde yok, atlandi.",
                    file=sys.stderr,
                )
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
                    label=mapped_label,
                    source_label=source_label,
                    dataset_name=dataset_name,
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
    raw_audio = tf.io.read_file(path)
    waveform, sample_rate = tf.audio.decode_wav(raw_audio)
    frames = ensure_frame_major(waveform.numpy().astype(np.float32, copy=False))
    if frames.size == 0:
        raise RuntimeError(f"Bos ses dosyasi: {path}")
    return frames, int(sample_rate.numpy())


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
                "raw_predicted_label": record.raw_predicted_label,
                "raw_predicted_confidence": f"{record.raw_predicted_confidence:.6f}",
                "dominant_label": record.dominant_label,
                "dominant_confidence": f"{record.dominant_confidence:.6f}",
                "window_index": record.window_index,
                "window_start_s": f"{record.window_start_s:.6f}",
                "window_end_s": f"{record.window_end_s:.6f}",
                "duration_seconds": f"{record.duration_seconds:.6f}",
                "is_correct": int(record.dominant_label == record.sample.label),
                "is_raw_correct": int(record.raw_predicted_label == record.sample.label),
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
    """Generate a classification comparison chart from the shared benchmark CSV."""
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
        ("accuracy", "Uctan Uca Dogruluk", 0.0, 1.08),
        ("f1", "Uctan Uca Makro F1", 0.0, 1.08),
        ("ece", "ECE", 0.0, 1.08),
        ("false_alerts_per_minute", "Yanlis Alarm / Dakika", 0.0, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
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
        axis.set_xticklabels(dataset_names)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(y_min, (y_max if y_max is not None else max(1.0, max_value * 1.35 + 0.05)))

    handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=8, label=label)
        for label in legend_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False)
    fig.suptitle("Model ve pipeline karsilastirma ozeti", fontsize=16)
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
        ("mono_raw_match_rate", "Mono Raw Eslesme Orani", 0.0, 1.08),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
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
        axis.set_xticklabels(dataset_names)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(y_min, (y_max if y_max is not None else max(1.0, max_value * 1.35 + 0.05)))

    handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=8, label=label)
        for label in legend_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False)
    fig.suptitle("Pipeline davranis karsilastirmasi", fontsize=16)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_confusion_matrix(clip_results: list[ClipResult]) -> tuple[list[str], np.ndarray]:
    """Build a confusion matrix ordered by the active label set in this run."""
    labels = [
        label
        for label in sorted(set(REDUCED_LABEL_SET + ["silence", "idle", "ood"]))
        if any(result.sample.label == label or result.predicted_label == label for result in clip_results)
    ]
    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)

    for result in clip_results:
        true_index = index_by_label[result.sample.label]
        pred_index = index_by_label[result.predicted_label]
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
    image = ax.imshow(matrix, cmap="YlOrRd")
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

    return ClipResult(
        sample=sample,
        pipeline_variant=pipeline_variant,
        predicted_label=predicted_label,
        predicted_confidence=float(predicted_confidence),
        dominant_top3=dominant_top3,
        raw_predicted_label=raw_predicted_label,
        raw_predicted_confidence=float(raw_predicted_confidence),
        raw_top3=raw_top3,
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
                dominant_label=legacy.dominant_label,
                dominant_confidence=float(legacy.dominant_prob),
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
                dominant_label=current.decision.label,
                dominant_confidence=float(current.decision.confidence),
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

    legacy_clip = aggregate_clip_result(sample, args.legacy_variant_label, legacy_windows, label_names, raw_prob_sum)
    current_clip = aggregate_clip_result(sample, args.current_variant_label, current_windows, label_names, raw_prob_sum)
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
    """Build overall + per-class rows for one dataset and one pipeline variant."""
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))
    total = len(clip_results)
    total_audio_minutes = safe_div(sum(result.duration_seconds for result in clip_results), 60.0)

    correctness = np.array(
        [1.0 if result.predicted_label == result.sample.label else 0.0 for result in clip_results],
        dtype=np.float64,
    )
    confidences = np.array([result.predicted_confidence for result in clip_results], dtype=np.float64)
    ece, calibration_bins = compute_ece_from_arrays(confidences, correctness, ece_bins)

    true_counter = Counter(result.sample.label for result in clip_results)
    pred_counter = Counter(result.predicted_label for result in clip_results)
    tp_counter = Counter(result.sample.label for result in clip_results if result.predicted_label == result.sample.label)

    raw_true_counter = Counter(result.sample.label for result in clip_results)
    raw_pred_counter = Counter(result.raw_predicted_label for result in clip_results)
    raw_tp_counter = Counter(
        result.sample.label for result in clip_results if result.raw_predicted_label == result.sample.label
    )

    correct = int(np.sum(correctness))
    raw_correct = sum(1 for result in clip_results if result.raw_predicted_label == result.sample.label)
    avg_confidence = safe_div(sum(result.predicted_confidence for result in clip_results), total)
    raw_avg_confidence = safe_div(sum(result.raw_predicted_confidence for result in clip_results), total)

    precision_values = []
    recall_values = []
    f1_values = []
    raw_precision_values = []
    raw_recall_values = []
    raw_f1_values = []
    labels_in_run = sorted(
        {
            result.sample.label
            for result in clip_results
        }
        | {result.predicted_label for result in clip_results}
        | {result.raw_predicted_label for result in clip_results}
    )

    per_class_rows: list[dict[str, str | float | int]] = []
    for label in labels_in_run:
        tp = tp_counter[label]
        fp = pred_counter[label] - tp
        fn = true_counter[label] - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        raw_tp = raw_tp_counter[label]
        raw_fp = raw_pred_counter[label] - raw_tp
        raw_fn = raw_true_counter[label] - raw_tp
        raw_precision = safe_div(raw_tp, raw_tp + raw_fp)
        raw_recall = safe_div(raw_tp, raw_tp + raw_fn)
        raw_f1 = safe_div(2 * raw_precision * raw_recall, raw_precision + raw_recall)

        if true_counter[label] > 0:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            raw_precision_values.append(raw_precision)
            raw_recall_values.append(raw_recall)
            raw_f1_values.append(raw_f1)

        per_class_rows.append(
            {
                "timestamp_utc": timestamp_iso,
                "run_id": run_id,
                "model_label": model_label,
                "pipeline_variant": pipeline_variant,
                "dataset_name": dataset_name,
                "metric_scope": "class",
                "class_label": label,
                "sample_count": true_counter[label],
                "correct_count": tp,
                "total_audio_minutes": "",
                "accuracy": safe_div(tp, true_counter[label]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "raw_accuracy": safe_div(raw_tp, raw_true_counter[label]),
                "raw_precision": raw_precision,
                "raw_recall": raw_recall,
                "raw_f1": raw_f1,
                "avg_confidence": safe_div(
                    sum(result.predicted_confidence for result in clip_results if result.sample.label == label),
                    true_counter[label],
                ),
                "raw_avg_confidence": safe_div(
                    sum(result.raw_predicted_confidence for result in clip_results if result.sample.label == label),
                    raw_true_counter[label],
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
            is_wrong_alert = record.dominant_label not in NEUTRAL_OUTPUT_LABELS and record.dominant_label != record.sample.label
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

    overall_row = {
        "timestamp_utc": timestamp_iso,
        "run_id": run_id,
        "model_label": model_label,
        "pipeline_variant": pipeline_variant,
        "dataset_name": dataset_name,
        "metric_scope": "overall",
        "class_label": "__overall__",
        "sample_count": total,
        "correct_count": correct,
        "total_audio_minutes": total_audio_minutes,
        "accuracy": safe_div(correct, total),
        "precision": safe_div(sum(precision_values), len(precision_values)),
        "recall": safe_div(sum(recall_values), len(recall_values)),
        "f1": safe_div(sum(f1_values), len(f1_values)),
        "raw_accuracy": safe_div(raw_correct, total),
        "raw_precision": safe_div(sum(raw_precision_values), len(raw_precision_values)),
        "raw_recall": safe_div(sum(raw_recall_values), len(raw_recall_values)),
        "raw_f1": safe_div(sum(raw_f1_values), len(raw_f1_values)),
        "avg_confidence": avg_confidence,
        "raw_avg_confidence": raw_avg_confidence,
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
    """Write per-dataset artifacts such as confusion matrices and reliability plots."""
    dataset_slug = slugify(dataset_name)
    model_slug = slugify(model_label)
    variant_slug = slugify(pipeline_variant)
    artifact_prefix = os.path.join(EVAL_DIR, f"{run_id}_{dataset_slug}_{model_slug}_{variant_slug}")

    labels, matrix = build_confusion_matrix(clip_results)
    confusion_csv = f"{artifact_prefix}_confusion_matrix.csv"
    confusion_png = f"{artifact_prefix}_confusion_matrix.png"
    reliability_png = f"{artifact_prefix}_reliability_curve.png"
    states_png = f"{artifact_prefix}_state_distribution.png"

    write_confusion_matrix_csv(confusion_csv, labels, matrix)
    plot_confusion_matrix(confusion_png, labels, matrix, f"{dataset_name} - {model_label} [{pipeline_variant}]")
    plot_reliability_curve(
        reliability_png,
        calibration_bins,
        ece,
        f"{dataset_name} - {model_label} [{pipeline_variant}]",
    )
    plot_state_distribution(states_png, window_records, f"{dataset_name} - {pipeline_variant} durum dagilimi")

    return [confusion_csv, confusion_png, reliability_png, states_png]


def print_run_summary(rows: Iterable[dict[str, str | float | int]]) -> None:
    """Print the overall rows for terminal visibility."""
    print("")
    for overall_row in (row for row in rows if row["metric_scope"] == "overall"):
        print("[SONUC]")
        print(f"Model etiketi           : {overall_row['model_label']}")
        print(f"Pipeline varyanti       : {overall_row['pipeline_variant']}")
        print(f"Veri kumesi             : {overall_row['dataset_name']}")
        print(f"Clip sayisi             : {overall_row['sample_count']}")
        print(f"Uctan uca dogruluk      : {float(overall_row['accuracy']):.4f}")
        print(f"Uctan uca Makro F1      : {float(overall_row['f1']):.4f}")
        print(f"Raw dogruluk            : {float(overall_row['raw_accuracy']):.4f}")
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

    summary_history = read_summary_rows(args.summary_csv)
    plot_history(summary_history, args.plot_path)
    plot_behavior_history(summary_history, args.behavior_plot_path)
    print_run_summary(rows_to_append)

    print(f"[KAYIT] Ozet CSV             : {os.path.abspath(args.summary_csv)}")
    print(f"[KAYIT] Detay pencere CSV    : {os.path.abspath(args.predictions_csv)}")
    print(f"[KAYIT] Siniflama grafigi    : {os.path.abspath(args.plot_path)}")
    print(f"[KAYIT] Davranis grafigi     : {os.path.abspath(args.behavior_plot_path)}")
    print(f"[KAYIT] Run ID               : {run_id}")
    for artifact_path in artifact_paths:
        print(f"[KAYIT] Artefakt             : {os.path.abspath(artifact_path)}")


if __name__ == "__main__":
    main()
