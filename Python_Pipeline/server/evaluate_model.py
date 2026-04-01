"""Evaluate the current classifier with a research-oriented benchmark pipeline.

This script is intentionally kept in a single file so it is easy to run, audit,
and extend. It is designed around the evaluation needs described in the
EchoSpace research notes:

  - shared CSV history for model-to-model comparison
  - per-sample prediction logging
  - macro-F1, per-class recall, confusion matrix, and ECE
  - false alerts per minute
  - optional onset/UI/human latency summaries when timestamps are available
  - reusable plots that can later compare "Recent Model" vs improved models

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

The audio loader expects WAV files so TensorFlow can decode them directly.
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

# Import config before pyplot so the runtime can pick a safe backend first.
from pipeline_runtime.config import LOG_DIR, REDUCED_LABEL_SET, tf
import matplotlib.pyplot as plt

from pipeline_runtime.classification import YamnetClassifier


EVAL_DIR = os.path.join(LOG_DIR, "evaluation")
SUMMARY_CSV = os.path.join(EVAL_DIR, "model_benchmark_results.csv")
PREDICTIONS_CSV = os.path.join(EVAL_DIR, "model_benchmark_predictions.csv")
COMPARISON_PNG = os.path.join(EVAL_DIR, "model_benchmark_comparison.png")

OPTIONAL_METADATA_COLUMNS = [
    "participant_id",
    "condition",
    "environment",
    "sensor",
    "channels",
    "snr_db",
    "annotator_confidence",
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
class PredictionRecord:
    """Prediction result for one sample."""

    sample: SampleRecord
    predicted_label: str
    predicted_confidence: float
    top3: list[tuple[str, float]]
    duration_seconds: float


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
        help="Model label written into the shared benchmark CSV.",
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
        help="Shared CSV that stores one row per predicted sample.",
    )
    parser.add_argument(
        "--plot-path",
        default=COMPARISON_PNG,
        help="PNG path for the auto-generated comparison chart.",
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


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file through TensorFlow without adding extra dependencies."""
    raw_audio = tf.io.read_file(path)
    waveform, sample_rate = tf.audio.decode_wav(raw_audio, desired_channels=1)
    mono_audio = tf.squeeze(waveform, axis=-1).numpy().astype(np.float32, copy=False)
    if mono_audio.size == 0:
        raise RuntimeError(f"Bos ses dosyasi: {path}")
    return mono_audio, int(sample_rate.numpy())


def evaluate_samples(classifier: YamnetClassifier, samples: list[SampleRecord]) -> list[PredictionRecord]:
    """Run predictions for each sample in the manifest."""
    predictions: list[PredictionRecord] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        try:
            audio, sample_rate = load_audio(sample.audio_path)
            labels, probabilities = classifier.predict_all(audio, sample_rate)
            order = np.argsort(probabilities)[::-1]
            top3 = [(labels[i], float(probabilities[i])) for i in order[:3]]
            predicted_label, predicted_confidence = top3[0]
            duration_seconds = float(audio.size) / float(sample_rate)
            predictions.append(
                PredictionRecord(
                    sample=sample,
                    predicted_label=predicted_label,
                    predicted_confidence=predicted_confidence,
                    top3=top3,
                    duration_seconds=duration_seconds,
                )
            )
        except Exception as exc:
            print(f"[WARN] Tahmin atlandi: {sample.audio_path} -> {exc}", file=sys.stderr)
            continue

        if index == 1 or index % 25 == 0 or index == total:
            print(f"[EVAL] {index}/{total} ornek tamamlandi.")

    if not predictions:
        raise RuntimeError("Degerlendirme sonunda hic tahmin kaydi olusmadi.")
    return predictions


def compute_ece(predictions: list[PredictionRecord], num_bins: int) -> tuple[float, list[dict[str, float]]]:
    """Compute expected calibration error and the reliability-curve bins."""
    confidences = np.array([record.predicted_confidence for record in predictions], dtype=np.float64)
    correctness = np.array(
        [1.0 if record.predicted_label == record.sample.label else 0.0 for record in predictions],
        dtype=np.float64,
    )
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bins: list[dict[str, float]] = []
    total = len(predictions)

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


def collect_latency_ms(
    predictions: list[PredictionRecord],
    start_field: str,
    end_field: str,
) -> list[float]:
    """Collect optional latencies in milliseconds from manifest timestamps."""
    values_ms = []
    for record in predictions:
        start_value = getattr(record.sample, start_field)
        end_value = getattr(record.sample, end_field)
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


def summarize_predictions(
    predictions: list[PredictionRecord],
    dataset_name: str,
    run_id: str,
    model_label: str,
    run_timestamp_unix: float,
    ece: float,
) -> list[dict[str, str | float | int]]:
    """Convert raw predictions into overall + per-class metric rows."""
    labels_in_run = [
        label
        for label in REDUCED_LABEL_SET
        if any(record.sample.label == label or record.predicted_label == label for record in predictions)
    ]
    total = len(predictions)
    correct = sum(1 for record in predictions if record.predicted_label == record.sample.label)
    average_confidence = safe_div(sum(record.predicted_confidence for record in predictions), total)
    total_audio_minutes = safe_div(sum(record.duration_seconds for record in predictions), 60.0)

    true_counter = Counter(record.sample.label for record in predictions)
    pred_counter = Counter(record.predicted_label for record in predictions)
    tp_counter = Counter(
        record.sample.label for record in predictions if record.predicted_label == record.sample.label
    )

    per_class_rows = []
    precision_values = []
    recall_values = []
    f1_values = []
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))

    false_alerts_per_minute = safe_div(total - correct, total_audio_minutes)
    onset_latency_ms = collect_latency_ms(predictions, "onset_time", "prediction_time")
    ui_latency_ms = collect_latency_ms(predictions, "prediction_time", "ui_render_time")
    human_response_ms = collect_latency_ms(predictions, "ui_render_time", "user_response_time")

    onset_mean, onset_p50 = summarize_latency(onset_latency_ms)
    ui_mean, ui_p50 = summarize_latency(ui_latency_ms)
    human_mean, human_p50 = summarize_latency(human_response_ms)

    for label in labels_in_run:
        tp = tp_counter[label]
        fp = pred_counter[label] - tp
        fn = true_counter[label] - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        class_accuracy = safe_div(tp, true_counter[label])

        if true_counter[label] > 0:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        per_class_rows.append(
            {
                "timestamp_utc": timestamp_iso,
                "run_id": run_id,
                "model_label": model_label,
                "dataset_name": dataset_name,
                "metric_scope": "class",
                "class_label": label,
                "sample_count": true_counter[label],
                "correct_count": tp,
                "total_audio_minutes": "",
                "accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_confidence": safe_div(
                    sum(
                        record.predicted_confidence
                        for record in predictions
                        if record.sample.label == label
                    ),
                    true_counter[label],
                ),
                "ece": "",
                "false_alerts_per_minute": "",
                "onset_latency_ms_mean": "",
                "onset_latency_ms_p50": "",
                "ui_latency_ms_mean": "",
                "ui_latency_ms_p50": "",
                "human_response_ms_mean": "",
                "human_response_ms_p50": "",
            }
        )

    overall_row = {
        "timestamp_utc": timestamp_iso,
        "run_id": run_id,
        "model_label": model_label,
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
        "avg_confidence": average_confidence,
        "ece": ece,
        "false_alerts_per_minute": false_alerts_per_minute,
        "onset_latency_ms_mean": onset_mean,
        "onset_latency_ms_p50": onset_p50,
        "ui_latency_ms_mean": ui_mean,
        "ui_latency_ms_p50": ui_p50,
        "human_response_ms_mean": human_mean,
        "human_response_ms_p50": human_p50,
    }

    return [overall_row, *per_class_rows]


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
    fieldnames = [
        "timestamp_utc",
        "run_id",
        "model_label",
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
        "avg_confidence",
        "ece",
        "false_alerts_per_minute",
        "onset_latency_ms_mean",
        "onset_latency_ms_p50",
        "ui_latency_ms_mean",
        "ui_latency_ms_p50",
        "human_response_ms_mean",
        "human_response_ms_p50",
    ]

    prepare_csv_for_append(csv_path, fieldnames)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_prediction_rows(
    csv_path: str,
    predictions: list[PredictionRecord],
    run_id: str,
    model_label: str,
    run_timestamp_unix: float,
) -> None:
    """Append one row per evaluated sample for detailed inspection."""
    fieldnames = [
        "timestamp_utc",
        "run_id",
        "model_label",
        "dataset_name",
        "clip_id",
        "audio_path",
        "source_label",
        "target_label",
        "predicted_label",
        "predicted_confidence",
        "top3_labels",
        "duration_seconds",
        "is_correct",
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

    prepare_csv_for_append(csv_path, fieldnames)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))

    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for record in predictions:
            row = {
                "timestamp_utc": timestamp_iso,
                "run_id": run_id,
                "model_label": model_label,
                "dataset_name": record.sample.dataset_name,
                "clip_id": record.sample.clip_id,
                "audio_path": record.sample.audio_path,
                "source_label": record.sample.source_label,
                "target_label": record.sample.label,
                "predicted_label": record.predicted_label,
                "predicted_confidence": f"{record.predicted_confidence:.6f}",
                "top3_labels": json.dumps(record.top3, ensure_ascii=False),
                "duration_seconds": f"{record.duration_seconds:.6f}",
                "is_correct": int(record.predicted_label == record.sample.label),
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


def latest_overall_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the latest row per dataset/model pair for comparison plots."""
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in summary_rows:
        if row["metric_scope"] != "overall":
            continue
        key = (row["dataset_name"], row["model_label"])
        previous = latest.get(key)
        if previous is None or row["timestamp_utc"] > previous["timestamp_utc"]:
            latest[key] = row
    return list(latest.values())


def plot_history(summary_rows: list[dict[str, str]], plot_path: str) -> None:
    """Generate a comparison chart from the shared benchmark CSV."""
    overall_rows = latest_overall_rows(summary_rows)
    if not overall_rows:
        return

    ensure_parent_dir(plot_path)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        grouped_rows[row["dataset_name"]].append(row)

    dataset_names = sorted(grouped_rows.keys())
    model_labels = sorted({row["model_label"] for row in overall_rows})
    color_map = {label: plt.cm.Set2(index % 8) for index, label in enumerate(model_labels)}
    metrics = [
        ("accuracy", "Top-1 Dogruluk", 0.0, 1.08),
        ("f1", "Makro F1", 0.0, 1.08),
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
                dataset_rows = sorted(grouped_rows[dataset_name], key=lambda row: row["model_label"])
                if bar_index >= len(dataset_rows):
                    continue
                row = dataset_rows[bar_index]
                value = float(row[metric_key]) if row[metric_key] not in ("", None) else 0.0
                positions.append(dataset_index - 0.36 + (bar_index + 0.5) * bar_width)
                values.append(value)
                labels.append(row["model_label"])
                colors.append(color_map[row["model_label"]])
                max_value = max(max_value, value)

            if not positions:
                continue

            bars = axis.bar(positions, values, width=bar_width, color=colors, edgecolor="white", linewidth=1.0)
            for bar, value, label in zip(bars, values, labels):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(0.01, max_value * 0.03),
                    f"{label}\n{value:.3f}",
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
        for label in model_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False)
    fig.suptitle("Model karsilastirma ozeti", fontsize=16)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_confusion_matrix(predictions: list[PredictionRecord]) -> tuple[list[str], np.ndarray]:
    """Build a confusion matrix ordered by the reduced label set."""
    labels = [
        label
        for label in REDUCED_LABEL_SET
        if any(record.sample.label == label or record.predicted_label == label for record in predictions)
    ]
    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)

    for record in predictions:
        true_index = index_by_label[record.sample.label]
        pred_index = index_by_label[record.predicted_label]
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


def plot_latency_overview(
    path: str,
    onset_latency_ms: list[float],
    ui_latency_ms: list[float],
    human_response_ms: list[float],
    title: str,
) -> None:
    """Render latency histograms if timestamp columns are available."""
    series = [
        ("Onset latency", onset_latency_ms, "#1d4d6c"),
        ("UI latency", ui_latency_ms, "#0f766e"),
        ("Human response", human_response_ms, "#d97706"),
    ]
    available = [(name, values, color) for name, values, color in series if values]
    if not available:
        return

    ensure_parent_dir(path)
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.2 * len(available)), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    for axis, (name, values, color) in zip(axes, available):
        axis.hist(values, bins=min(20, max(6, int(np.sqrt(len(values))))), color=color, alpha=0.8)
        axis.set_title(f"{name} dagilimi")
        axis.set_xlabel("Milisaniye")
        axis.set_ylabel("Ornek")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=15)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_dataset_artifacts(
    predictions: list[PredictionRecord],
    dataset_name: str,
    model_label: str,
    run_id: str,
    ece: float,
    calibration_bins: list[dict[str, float]],
) -> list[str]:
    """Write per-dataset artifacts such as confusion matrices and calibration plots."""
    dataset_slug = slugify(dataset_name)
    model_slug = slugify(model_label)
    artifact_prefix = os.path.join(EVAL_DIR, f"{run_id}_{dataset_slug}_{model_slug}")

    labels, matrix = build_confusion_matrix(predictions)
    confusion_csv = f"{artifact_prefix}_confusion_matrix.csv"
    confusion_png = f"{artifact_prefix}_confusion_matrix.png"
    reliability_png = f"{artifact_prefix}_reliability_curve.png"
    latency_png = f"{artifact_prefix}_latency_overview.png"

    write_confusion_matrix_csv(confusion_csv, labels, matrix)
    plot_confusion_matrix(confusion_png, labels, matrix, f"{dataset_name} - {model_label}")
    plot_reliability_curve(reliability_png, calibration_bins, ece, f"{dataset_name} - {model_label}")

    onset_latency_ms = collect_latency_ms(predictions, "onset_time", "prediction_time")
    ui_latency_ms = collect_latency_ms(predictions, "prediction_time", "ui_render_time")
    human_response_ms = collect_latency_ms(predictions, "ui_render_time", "user_response_time")
    plot_latency_overview(
        latency_png,
        onset_latency_ms,
        ui_latency_ms,
        human_response_ms,
        f"{dataset_name} - {model_label}",
    )

    artifact_paths = [confusion_csv, confusion_png, reliability_png]
    if os.path.exists(latency_png):
        artifact_paths.append(latency_png)
    return artifact_paths


def print_run_summary(rows: Iterable[dict[str, str | float | int]]) -> None:
    """Print the overall rows for terminal visibility."""
    print("")
    for overall_row in (row for row in rows if row["metric_scope"] == "overall"):
        print("[SONUC]")
        print(f"Model etiketi        : {overall_row['model_label']}")
        print(f"Veri kumesi          : {overall_row['dataset_name']}")
        print(f"Ornek sayisi         : {overall_row['sample_count']}")
        print(f"Top-1 dogruluk       : {float(overall_row['accuracy']):.4f}")
        print(f"Makro F1             : {float(overall_row['f1']):.4f}")
        print(f"Makro recall         : {float(overall_row['recall']):.4f}")
        print(f"ECE                  : {float(overall_row['ece']):.4f}")
        print(f"Yanlis alarm / dakika: {float(overall_row['false_alerts_per_minute']):.4f}")
        print(f"Ort. guven           : {float(overall_row['avg_confidence']):.4f}")
        if overall_row["onset_latency_ms_mean"] != "":
            print(f"Onset latency ort.   : {float(overall_row['onset_latency_ms_mean']):.2f} ms")
        if overall_row["ui_latency_ms_mean"] != "":
            print(f"UI latency ort.      : {float(overall_row['ui_latency_ms_mean']):.2f} ms")
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
    classifier = YamnetClassifier()

    print("[EVAL] Tahminler basliyor...")
    predictions = evaluate_samples(classifier, samples)

    grouped_predictions: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        grouped_predictions[prediction.sample.dataset_name].append(prediction)

    all_rows: list[dict[str, str | float | int]] = []
    artifact_paths: list[str] = []

    for dataset_name in sorted(grouped_predictions):
        dataset_predictions = grouped_predictions[dataset_name]
        ece, calibration_bins = compute_ece(dataset_predictions, args.ece_bins)
        all_rows.extend(
            summarize_predictions(
                predictions=dataset_predictions,
                dataset_name=dataset_name,
                run_id=run_id,
                model_label=args.model_label,
                run_timestamp_unix=run_timestamp_unix,
                ece=ece,
            )
        )
        artifact_paths.extend(
            export_dataset_artifacts(
                predictions=dataset_predictions,
                dataset_name=dataset_name,
                model_label=args.model_label,
                run_id=run_id,
                ece=ece,
                calibration_bins=calibration_bins,
            )
        )

    append_summary_rows(args.summary_csv, all_rows)
    append_prediction_rows(args.predictions_csv, predictions, run_id, args.model_label, run_timestamp_unix)

    summary_history = read_summary_rows(args.summary_csv)
    plot_history(summary_history, args.plot_path)
    print_run_summary(all_rows)

    print(f"[KAYIT] Ozet CSV       : {os.path.abspath(args.summary_csv)}")
    print(f"[KAYIT] Tahmin CSV     : {os.path.abspath(args.predictions_csv)}")
    print(f"[KAYIT] Karsilastirma  : {os.path.abspath(args.plot_path)}")
    print(f"[KAYIT] Run ID         : {run_id}")
    for artifact_path in artifact_paths:
        print(f"[KAYIT] Artefakt       : {os.path.abspath(artifact_path)}")


if __name__ == "__main__":
    main()
