"""Runtime classifier for the PANNs pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import (
    PANNS_CLIP_HOP_SECONDS,
    PANNS_CLIP_SECONDS,
    PANNS_DEFAULT_THRESHOLD,
    PANNS_DROPOUT,
    PANNS_FMAX,
    PANNS_FMIN,
    PANNS_HOP_LENGTH,
    PANNS_MEL_BINS,
    PANNS_MODEL_PATH,
    PANNS_N_FFT,
    PANNS_TARGET_SAMPLE_RATE,
    load_semantic_labels,
)
from .models import PannsCNN14Semantic, TORCH_AVAILABLE, torch
from .preprocess import ensure_mono, extract_logmel, iter_clip_spans, pad_or_trim, resample_linear


class PannsClassifier:
    """PANNs-style semantic classifier that matches the current runtime contract."""

    def __init__(self, artifact_path: str | None = None, labels_path: str | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for CLASSIFIER_MODE=panns.")

        self.mode = "panns"
        self.target_sr = PANNS_TARGET_SAMPLE_RATE
        self.clip_seconds = float(PANNS_CLIP_SECONDS)
        self.clip_hop_seconds = float(PANNS_CLIP_HOP_SECONDS)
        self.clip_samples = max(1, int(round(self.target_sr * self.clip_seconds)))
        self.clip_hop_samples = max(1, int(round(self.target_sr * self.clip_hop_seconds)))
        self.default_threshold = float(PANNS_DEFAULT_THRESHOLD)

        self.artifact_path = Path(artifact_path) if artifact_path is not None else Path(PANNS_MODEL_PATH)
        if not self.artifact_path.exists():
            raise RuntimeError(
                "PANNs artifact not found. Expected checkpoint at "
                f"{self.artifact_path}. Train/export a checkpoint into this path or set PANNS_MODEL_PATH."
            )

        checkpoint = torch.load(self.artifact_path, map_location="cpu")
        if hasattr(checkpoint, "state_dict"):
            checkpoint = checkpoint.state_dict()
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported PANNs artifact format: {type(checkpoint)!r}")

        artifact_labels = checkpoint.get("labels")
        self.class_names = list(artifact_labels or load_semantic_labels(labels_path))
        self.n_fft = int(checkpoint.get("n_fft", PANNS_N_FFT))
        self.hop_length = int(checkpoint.get("hop_length", PANNS_HOP_LENGTH))
        self.n_mels = int(checkpoint.get("n_mels", PANNS_MEL_BINS))
        self.fmin = float(checkpoint.get("fmin", PANNS_FMIN))
        self.fmax = float(checkpoint.get("fmax", PANNS_FMAX))

        self.model = PannsCNN14Semantic(
            num_classes=len(self.class_names),
            dropout=float(checkpoint.get("dropout", PANNS_DROPOUT)),
            embedding_dim=int(checkpoint.get("embedding_dim", 2048)),
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _prepare_batch(self, mono_audio: np.ndarray) -> np.ndarray:
        spans = iter_clip_spans(mono_audio.shape[0], self.clip_samples, self.clip_hop_samples)
        features: list[np.ndarray] = []
        for start, end in spans:
            clip = mono_audio[start:end]
            clip = pad_or_trim(clip, self.clip_samples)
            features.append(
                extract_logmel(
                    clip,
                    self.target_sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                )
            )
        return np.stack(features, axis=0).astype(np.float32, copy=False)

    def predict_all(self, mono_audio: np.ndarray, input_sr: int) -> tuple[list[str], np.ndarray]:
        """Return semantic probabilities aligned with the current decision layer."""
        mono = ensure_mono(np.asarray(mono_audio, dtype=np.float32))
        if input_sr != self.target_sr:
            mono = resample_linear(mono, input_sr, self.target_sr)
        batch = self._prepare_batch(mono)
        with torch.no_grad():
            logits, _embeddings = self.model(torch.from_numpy(batch))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
        merged = np.mean(probabilities, axis=0)
        return self.class_names, np.clip(merged, 0.0, 1.0).astype(np.float32, copy=False)
