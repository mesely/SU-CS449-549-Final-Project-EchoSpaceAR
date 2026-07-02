"""Normalize raw PANNs checkpoints into a unified runtime artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    PANNS_DROPOUT,
    PANNS_FMAX,
    PANNS_FMIN,
    PANNS_HOP_LENGTH,
    PANNS_LABELS_JSON,
    PANNS_MEL_BINS,
    PANNS_MODEL_PATH,
    PANNS_N_FFT,
    PANNS_TARGET_SAMPLE_RATE,
    load_semantic_labels,
)
from .models import TORCH_AVAILABLE, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a unified PANNs runtime artifact.")
    parser.add_argument("--input", required=True, help="Input checkpoint path.")
    parser.add_argument("--output", default=str(PANNS_MODEL_PATH), help="Output artifact path.")
    parser.add_argument("--labels-json", default=str(PANNS_LABELS_JSON), help="Semantic labels JSON.")
    return parser.parse_args()


def main() -> None:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required to export the PANNs pipeline.")

    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    labels = checkpoint.get("labels") or load_semantic_labels(args.labels_json)
    artifact = {
        "state_dict": checkpoint.get("state_dict", checkpoint),
        "labels": labels,
        "n_fft": int(checkpoint.get("n_fft", PANNS_N_FFT)),
        "hop_length": int(checkpoint.get("hop_length", PANNS_HOP_LENGTH)),
        "n_mels": int(checkpoint.get("n_mels", PANNS_MEL_BINS)),
        "fmin": float(checkpoint.get("fmin", PANNS_FMIN)),
        "fmax": float(checkpoint.get("fmax", PANNS_FMAX)),
        "dropout": float(checkpoint.get("dropout", PANNS_DROPOUT)),
        "embedding_dim": int(checkpoint.get("embedding_dim", 2048)),
        "target_sample_rate": int(checkpoint.get("target_sample_rate", PANNS_TARGET_SAMPLE_RATE)),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    print(f"[PANNS][export] artifact -> {output_path}")
    print(f"[PANNS][export] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
