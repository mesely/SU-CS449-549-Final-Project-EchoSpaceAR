"""Evaluate the current audio classifier and append comparable benchmark results.

This script is designed to create a durable evaluation trail:
  - it runs the current classifier on a labeled manifest
  - it appends overall + per-class metrics into one shared CSV
  - it writes per-sample predictions to a second CSV
  - it refreshes a comparison chart that can later include improved models

Expected manifest columns:
  - `audio_path`: absolute path, or a path relative to the manifest / dataset root
  - `label`: target label in the reduced label space

Audio files are expected to be WAV so they can be decoded through TensorFlow.

Optional columns:
  - `clip_id`
  - `dataset`
  - `source_label`

Example:
  python evaluate_model.py \
      --manifest /path/to/manifest.csv \
      --dataset-name ESC-50 \
      --model-label "Recent Model"

If your dataset labels do not match the reduced labels exactly, pass a JSON mapping:
  {
    "chainsaw": "engine_motion",
    "siren": "sirens",
    "car_horn": "vehicle_horn"
  }
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
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Import config before pyplot so the runtime can select a safe backend first.
from pipeline_runtime.config import LOG_DIR, REDUCED_LABEL_SET, tf
import matplotlib.pyplot as plt

from pipeline_runtime.classification import YamnetClassifier


EVAL_DIR = os.path.join(LOG_DIR, "evaluation")
SUMMARY_CSV = os.path.join(EVAL_DIR, "model_benchmark_results.csv")
PREDICTIONS_CSV = os.path.join(EVAL_DIR, "model_benchmark_predictions.csv")
COMPARISON_PNG = os.path.join(EVAL_DIR, "model_benchmark_comparison.png")


@dataclass
class SampleRecord:
    """One labeled audio sample loaded from the manifest."""

    clip_id: str
    audio_path: str
    label: str
    source_label: str
    dataset_name: str


@dataclass
class PredictionRecord:
    """Prediction result for one sample."""

    sample: SampleRecord
    predicted_label: str
    predicted_confidence: float
    top3: list[tuple[str, float]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation run."""
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
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    """Create the parent folder for a file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


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

            records.append(
                SampleRecord(
                    clip_id=clip_id,
                    audio_path=os.path.abspath(audio_path),
                    label=mapped_label,
                    source_label=source_label,
                    dataset_name=dataset_name,
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
    """Run top-k predictions for each sample in the manifest."""
    predictions: list[PredictionRecord] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        try:
            audio, sample_rate = load_audio(sample.audio_path)
            labels, probabilities = classifier.predict_all(audio, sample_rate)
            order = np.argsort(probabilities)[::-1]
            top3 = [(labels[i], float(probabilities[i])) for i in order[:3]]
            predicted_label, predicted_confidence = top3[0]
            predictions.append(
                PredictionRecord(
                    sample=sample,
                    predicted_label=predicted_label,
                    predicted_confidence=predicted_confidence,
                    top3=top3,
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


def safe_div(numerator: float, denominator: float) -> float:
    """Avoid repeated zero-division checks in metric calculations."""
    return numerator / denominator if denominator else 0.0


def summarize_predictions(
    predictions: list[PredictionRecord],
    dataset_name: str,
    run_id: str,
    model_label: str,
    run_timestamp_unix: float,
) -> list[dict[str, str | float | int]]:
    """Convert raw predictions into overall + per-class metric rows."""
    labels_in_run = sorted({record.sample.label for record in predictions} | {record.predicted_label for record in predictions})
    total = len(predictions)
    correct = sum(1 for record in predictions if record.predicted_label == record.sample.label)
    average_confidence = safe_div(sum(record.predicted_confidence for record in predictions), total)

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
        "accuracy": safe_div(correct, total),
        "precision": safe_div(sum(precision_values), len(precision_values)),
        "recall": safe_div(sum(recall_values), len(recall_values)),
        "f1": safe_div(sum(f1_values), len(f1_values)),
        "avg_confidence": average_confidence,
    }

    return [overall_row, *per_class_rows]


def append_summary_rows(csv_path: str, rows: list[dict[str, str | float | int]]) -> None:
    """Append metric rows into the shared benchmark CSV."""
    ensure_parent_dir(csv_path)
    fieldnames = [
        "timestamp_utc",
        "run_id",
        "model_label",
        "dataset_name",
        "metric_scope",
        "class_label",
        "sample_count",
        "correct_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "avg_confidence",
    ]

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
    ensure_parent_dir(csv_path)
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
        "is_correct",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_timestamp_unix))

    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for record in predictions:
            writer.writerow(
                {
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
                    "is_correct": int(record.predicted_label == record.sample.label),
                }
            )


def read_summary_rows(csv_path: str) -> list[dict[str, str]]:
    """Read the shared summary CSV if it already exists."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_history(summary_rows: list[dict[str, str]], plot_path: str) -> None:
    """Generate a comparison chart from the shared benchmark CSV."""
    overall_rows = [row for row in summary_rows if row["metric_scope"] == "overall"]
    if not overall_rows:
        return

    ensure_parent_dir(plot_path)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in overall_rows:
        grouped_rows[row["dataset_name"]].append(row)

    dataset_names = list(grouped_rows.keys())
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    metrics = [("accuracy", "Top-1 Dogruluk"), ("f1", "Makro F1")]

    for axis, (metric_key, metric_title) in zip(axes, metrics):
        max_bars = max(len(grouped_rows[name]) for name in dataset_names)
        bar_width = 0.72 / max(max_bars, 1)
        x_positions = np.arange(len(dataset_names))

        for bar_index in range(max_bars):
            labels = []
            values = []
            positions = []
            for dataset_index, dataset_name in enumerate(dataset_names):
                dataset_rows = grouped_rows[dataset_name]
                if bar_index >= len(dataset_rows):
                    continue
                row = dataset_rows[bar_index]
                positions.append(dataset_index - 0.36 + (bar_index + 0.5) * bar_width)
                values.append(float(row[metric_key]))
                labels.append(row["model_label"])

            if not positions:
                continue

            bars = axis.bar(positions, values, width=bar_width, label=labels[0] if len(set(labels)) == 1 else None)
            for bar, value, label in zip(bars, values, labels):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.01,
                    f"{label}\n{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.set_title(metric_title)
        axis.set_ylabel("Skor")
        axis.set_ylim(0.0, 1.08)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(dataset_names, rotation=0)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Model karsilastirma ozeti", fontsize=16)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_run_summary(rows: Iterable[dict[str, str | float | int]]) -> None:
    """Print overall rows for terminal visibility."""
    print("")
    for overall_row in (row for row in rows if row["metric_scope"] == "overall"):
        print("[SONUC]")
        print(f"Model etiketi : {overall_row['model_label']}")
        print(f"Veri kumesi   : {overall_row['dataset_name']}")
        print(f"Ornek sayisi  : {overall_row['sample_count']}")
        print(f"Dogruluk      : {float(overall_row['accuracy']):.4f}")
        print(f"Makro F1      : {float(overall_row['f1']):.4f}")
        print(f"Ort. guven    : {float(overall_row['avg_confidence']):.4f}")
        print("")


def main() -> None:
    """Entry point for one complete evaluation pass."""
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

    rows: list[dict[str, str | float | int]] = []
    for dataset_name in sorted(grouped_predictions):
        rows.extend(
            summarize_predictions(
                predictions=grouped_predictions[dataset_name],
                dataset_name=dataset_name,
                run_id=run_id,
                model_label=args.model_label,
                run_timestamp_unix=run_timestamp_unix,
            )
        )

    append_summary_rows(args.summary_csv, rows)
    append_prediction_rows(args.predictions_csv, predictions, run_id, args.model_label, run_timestamp_unix)

    summary_history = read_summary_rows(args.summary_csv)
    plot_history(summary_history, args.plot_path)
    print_run_summary(rows)

    print("")
    print(f"[KAYIT] Ozet CSV      : {os.path.abspath(args.summary_csv)}")
    print(f"[KAYIT] Tahmin CSV    : {os.path.abspath(args.predictions_csv)}")
    print(f"[KAYIT] Grafik dosyasi: {os.path.abspath(args.plot_path)}")
    print(f"[KAYIT] Run ID        : {run_id}")


if __name__ == "__main__":
    main()
