"""YAMNet classifier adapters used by the visualizer."""

from __future__ import annotations

import csv
import json

import numpy as np

from .config import (
    CLASSIFIER_ENABLED,
    REDUCED_LABELS_JSON,
    REDUCED_MODEL_DIR,
    TF_AVAILABLE,
    USE_REDUCED,
    hub,
    tf,
)
from .utils import resample_linear


class YamnetClassifier:
    """
    Support either the reduced SavedModel or the full TF-Hub YAMNet model.
    """

    def __init__(self) -> None:
        if not CLASSIFIER_ENABLED or not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available; classifier disabled.")

        self.target_sr = 16000
        self.mode = "reduced" if USE_REDUCED else "full"

        if USE_REDUCED:
            print("Loading Reduced-YAMNet SavedModel...")
            model = tf.saved_model.load(REDUCED_MODEL_DIR)
            self.infer = model.__call__.get_concrete_function()
            with open(REDUCED_LABELS_JSON, "r", encoding="utf-8") as handle:
                self.class_names = json.load(handle)
            print("[OK] Reduced-YAMNet loaded.")
            return

        print("Loading full YAMNet from TF-Hub...")
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.yamnet.class_map_path().numpy().decode("utf-8")
        names = []
        with tf.io.gfile.GFile(class_map_path, "r") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                names.append(row["display_name"])
        self.class_names = names
        print("[OK] Full YAMNet loaded (521 classes).")

    def predict_all(self, mono_audio: np.ndarray, input_sr: int) -> tuple[list[str], np.ndarray]:
        """Return normalized probabilities for every label."""
        if input_sr != self.target_sr:
            mono_audio = resample_linear(mono_audio, input_sr, self.target_sr)
        waveform = tf.convert_to_tensor(mono_audio, dtype=tf.float32)

        if self.mode == "reduced":
            output = self.infer(waveform_16k=waveform)
            probabilities = output["probs"].numpy()
        else:
            scores, _embeddings, _spectrogram = self.yamnet(waveform)
            probabilities = tf.reduce_mean(scores, axis=0).numpy()

        probabilities = probabilities / (probabilities.sum() + 1e-9)
        return self.class_names, probabilities

    def predict_top(self, mono_audio: np.ndarray, input_sr: int) -> tuple[str, float]:
        """Return the top label and its probability."""
        labels, probabilities = self.predict_all(mono_audio, input_sr)
        index = int(np.argmax(probabilities))
        if 0 <= index < len(labels):
            return labels[index], float(probabilities[index])
        return "Unknown", float(probabilities[index])
