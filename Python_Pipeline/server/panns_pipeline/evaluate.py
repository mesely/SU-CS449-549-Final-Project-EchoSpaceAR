"""Evaluate a PANNs checkpoint on a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .classifier import PannsClassifier
from .config import PANNS_DEFAULT_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic PANNs checkpoint.")
    parser.add_argument("--manifest", required=True, help="CSV manifest with audio paths and labels.")
    parser.add_argument("--artifact", default=None, help="Checkpoint path. Defaults to PANNS_MODEL_PATH.")
    parser.add_argument("--labels-json", default=None, help="Optional label order JSON.")
    parser.add_argument("--thresholds-json", default=None, help="Optional per-class threshold JSON.")
    parser.add_argument("--output-json", default=None, help="Optional output metrics path.")
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def parse_labels(row: dict[str, str]) -> set[str]:
    for key in ("target_labels", "decision_target_labels", "labels"):
        raw = row.get(key, "").strip()
        if raw:
            if raw.startswith("["):
                try:
                    return {str(item).strip() for item in json.loads(raw) if str(item).strip()}
                except Exception:
                    pass
            return {token.strip() for token in raw.split(",") if token.strip()}
    for key in ("target_label", "label"):
        raw = row.get(key, "").strip()
        if raw:
            return {raw}
    return set()


def load_thresholds(path: str | None, labels: list[str]) -> dict[str, float]:
    if not path:
        return {label: PANNS_DEFAULT_THRESHOLD for label in labels}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {label: float(raw.get(label, PANNS_DEFAULT_THRESHOLD)) for label in labels}


def main() -> None:
    args = parse_args()
    classifier = PannsClassifier(artifact_path=args.artifact, labels_path=args.labels_json)
    thresholds = load_thresholds(args.thresholds_json, classifier.class_names)

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    sample_f1s: list[float] = []
    sample_precisions: list[float] = []
    sample_recalls: list[float] = []

    for row in rows:
        audio_path = row.get("audio_path") or row.get("path") or row.get("wav_path")
        if not audio_path:
            continue
        from scipy.io import wavfile

        sample_rate, audio = wavfile.read(audio_path)
        labels, probabilities = classifier.predict_all(audio, int(sample_rate))
        predicted = {
            label
            for label, probability in zip(labels, probabilities)
            if float(probability) >= float(thresholds.get(label, PANNS_DEFAULT_THRESHOLD))
        }
        target = parse_labels(row)
        if not target:
            continue
        tp = len(predicted & target)
        precision = safe_div(tp, len(predicted))
        recall = safe_div(tp, len(target))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1s.append(f1)

    metrics = {
        "sample_count": len(sample_f1s),
        "macro_precision": float(np.mean(sample_precisions)) if sample_precisions else 0.0,
        "macro_recall": float(np.mean(sample_recalls)) if sample_recalls else 0.0,
        "macro_f1": float(np.mean(sample_f1s)) if sample_f1s else 0.0,
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
