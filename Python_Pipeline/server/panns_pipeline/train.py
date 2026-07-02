"""Train a PANNs-style semantic classifier from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

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
from .models import PannsCNN14Semantic, TORCH_AVAILABLE, torch
from .preprocess import ensure_mono, extract_logmel, pad_or_trim, resample_linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic PANNs CNN14 checkpoint.")
    parser.add_argument("--manifest", required=True, help="CSV manifest with audio paths and labels.")
    parser.add_argument("--output", default=str(PANNS_MODEL_PATH), help="Output checkpoint path.")
    parser.add_argument("--labels-json", default=str(PANNS_LABELS_JSON), help="Semantic label order JSON.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-shift", action="store_true", help="Enable circular time-shift augmentation.")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Beta(alpha, alpha) mixup strength.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


@dataclass(slots=True)
class Record:
    audio_path: str
    label_names: tuple[str, ...]


def load_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_label_names(row: dict[str, str]) -> tuple[str, ...]:
    for key in ("target_labels", "decision_target_labels", "labels", "label_names"):
        raw = row.get(key, "").strip()
        if raw:
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    return tuple(str(item).strip() for item in parsed if str(item).strip())
                except Exception:
                    pass
            return tuple(token.strip() for token in raw.split(",") if token.strip())
    for key in ("target_label", "label"):
        raw = row.get(key, "").strip()
        if raw:
            return (raw,)
    raise ValueError(f"Cannot determine labels from row: {row}")


def read_audio(path: str) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(scale)
    else:
        audio = audio.astype(np.float32, copy=False)
    mono = ensure_mono(audio)
    if sample_rate != PANNS_TARGET_SAMPLE_RATE:
        mono = resample_linear(mono, sample_rate, PANNS_TARGET_SAMPLE_RATE)
    target_samples = int(round(PANNS_TARGET_SAMPLE_RATE * 10.0))
    return pad_or_trim(mono, target_samples)


def build_examples(rows: list[dict[str, str]], labels: list[str]) -> list[Record]:
    label_set = set(labels)
    records: list[Record] = []
    for row in rows:
        audio_path = row.get("audio_path") or row.get("path") or row.get("wav_path")
        if not audio_path:
            continue
        names = tuple(label for label in parse_label_names(row) if label in label_set)
        if names:
            records.append(Record(audio_path=audio_path, label_names=names))
    if not records:
        raise RuntimeError("No usable training records were found in the manifest.")
    return records


def encode_targets(label_names: tuple[str, ...], label_to_index: dict[str, int]) -> np.ndarray:
    target = np.zeros(len(label_to_index), dtype=np.float32)
    for label in label_names:
        target[label_to_index[label]] = 1.0
    return target


def maybe_time_shift(audio: np.ndarray) -> np.ndarray:
    max_shift = max(1, audio.shape[0] // 10)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(audio, shift).astype(np.float32, copy=False)


def collate_batch(
    records: list[Record],
    label_to_index: dict[str, int],
    enable_time_shift: bool,
    mixup_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        audio = read_audio(record.audio_path)
        if enable_time_shift:
            audio = maybe_time_shift(audio)
        feature = extract_logmel(
            audio,
            PANNS_TARGET_SAMPLE_RATE,
            n_fft=PANNS_N_FFT,
            hop_length=PANNS_HOP_LENGTH,
            n_mels=PANNS_MEL_BINS,
            fmin=PANNS_FMIN,
            fmax=PANNS_FMAX,
        )
        features.append(feature)
        targets.append(encode_targets(record.label_names, label_to_index))

    feature_batch = np.stack(features, axis=0).astype(np.float32, copy=False)
    target_batch = np.stack(targets, axis=0).astype(np.float32, copy=False)
    if mixup_alpha > 0.0 and feature_batch.shape[0] > 1:
        lam = np.random.beta(mixup_alpha, mixup_alpha, size=(feature_batch.shape[0], 1, 1)).astype(np.float32)
        order = np.random.permutation(feature_batch.shape[0])
        feature_batch = (lam * feature_batch) + ((1.0 - lam) * feature_batch[order])
        target_batch = (lam[:, 0, 0:1] * target_batch) + ((1.0 - lam[:, 0, 0:1]) * target_batch[order])
    return feature_batch, target_batch


def main() -> None:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required to train the PANNs pipeline.")

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = load_semantic_labels(args.labels_json)
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows = load_manifest(args.manifest)
    examples = build_examples(rows, labels)

    model = PannsCNN14Semantic(num_classes=len(labels), dropout=PANNS_DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        random.shuffle(examples)
        epoch_losses: list[float] = []
        for start in range(0, len(examples), args.batch_size):
            batch_records = examples[start : start + args.batch_size]
            feature_batch, target_batch = collate_batch(
                batch_records,
                label_to_index,
                enable_time_shift=args.time_shift,
                mixup_alpha=args.mixup_alpha,
            )
            logits, _embeddings = model(torch.from_numpy(feature_batch))
            loss = criterion(logits, torch.from_numpy(target_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"[PANNS][train] epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "n_fft": PANNS_N_FFT,
                    "hop_length": PANNS_HOP_LENGTH,
                    "n_mels": PANNS_MEL_BINS,
                    "fmin": PANNS_FMIN,
                    "fmax": PANNS_FMAX,
                    "dropout": PANNS_DROPOUT,
                    "embedding_dim": 2048,
                    "target_sample_rate": PANNS_TARGET_SAMPLE_RATE,
                    "best_train_loss": best_loss,
                },
                output_path,
            )
            print(f"[PANNS][train] best checkpoint -> {output_path}")


if __name__ == "__main__":
    main()
