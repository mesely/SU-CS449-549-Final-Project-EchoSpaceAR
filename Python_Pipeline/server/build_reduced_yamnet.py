"""Build a reduced-label YAMNet SavedModel and optional TFLite export."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


REDUCED_MAP = {
    "speech": ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Whispering"],
    "crowd": ["Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Cheering", "Applause", "Children playing"],
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
    "vehicle_horn": ["Vehicle horn, car horn, honking", "Toot", "Air horn, truck horn", "Foghorn"],
    "traffic_road": ["Motor vehicle (road)", "Traffic noise, roadway noise", "Car passing by", "Race car, auto racing", "Skidding", "Tire squeal"],
    "car_bus_truck": ["Car", "Bus", "Truck", "Ice cream truck, ice cream van"],
    "sirens": ["Siren", "Civil defense siren", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Car alarm"],
    "rail": ["Rail transport", "Train", "Train whistle", "Train horn", "Railroad car, train wagon", "Train wheels squealing", "Subway, metro, underground"],
    "aircraft": ["Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane"],
    "engine_motion": ["Engine", "Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine knocking", "Engine starting", "Idling", "Accelerating, revving, vroom"],
    "alarms_buzzer": ["Alarm", "Alarm clock", "Buzzer", "Smoke detector, smoke alarm", "Fire alarm"],
    "phone_ring": ["Telephone bell ringing", "Ringtone", "Telephone", "Telephone dialing, DTMF", "Dial tone", "Busy signal"],
    "wind_rain": ["Wind", "Rustling leaves", "Wind noise (microphone)", "Thunderstorm", "Thunder", "Rain", "Raindrop", "Rain on surface"],
    "door_knock": ["Door", "Doorbell", "Ding-dong", "Knock", "Tap", "Slam", "Sliding door", "Cupboard open or close", "Drawer open or close"],
    "glass_break": ["Glass", "Shatter", "Smash, crash", "Breaking", "Chink, clink", "Crack"],
    "explosion_gunshot": ["Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Fireworks", "Firecracker", "Burst, pop", "Eruption", "Boom"],
    "Silence": ["Silence", "Inside, small room", "Inside, large room or hall"],
    "other": "REST",
}

AGGREGATION = "sum"
EXPORT_DIR = "reduced_yamnet_savedmodel"
LABELS_JSON = "reduced_labels.json"


def load_yamnet_names() -> list[str]:
    """Read the canonical 521 YAMNet labels from TF-Hub or a local fallback."""
    try:
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
        return _read_class_map(class_map_path)
    except Exception as exc:
        local_csv = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
        if tf.io.gfile.exists(local_csv):
            return _read_class_map(local_csv)
        raise RuntimeError(f"Could not load YAMNet class map: {exc}")


def _read_class_map(path: str) -> list[str]:
    names = []
    with tf.io.gfile.GFile(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names.append(row["display_name"])
    if len(names) != 521:
        raise ValueError(f"class count {len(names)} != 521")
    return names


def build_pooling_matrix(
    names: list[str],
    reduced_map: dict[str, list[str] | str],
    aggregation: str = "sum",
) -> tuple[np.ndarray, list[str]]:
    """Build a `[521, K]` mapping from YAMNet labels to reduced labels."""
    target_labels = list(reduced_map.keys())
    name_to_index = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(target_labels)), dtype=np.float32)

    assigned_indices = set()
    rest_index = None

    for target_index, target_label in enumerate(target_labels):
        specification = reduced_map[target_label]
        if specification == "REST":
            rest_index = target_index
            continue

        for source_name in specification:
            if source_name not in name_to_index:
                print(f"[WARN] '{source_name}' not in YAMNet names; skipping.")
                continue
            source_index = name_to_index[source_name]
            matrix[source_index, target_index] = 1.0
            assigned_indices.add(source_index)

    if rest_index is not None:
        for source_index in range(len(names)):
            if source_index not in assigned_indices:
                matrix[source_index, rest_index] = 1.0

    if aggregation == "mean":
        column_sums = matrix.sum(axis=0, keepdims=True)
        column_sums[column_sums == 0] = 1.0
        matrix = matrix / column_sums

    return matrix.astype(np.float32), target_labels


class ReducedModule(tf.Module):
    """Wrap the original YAMNet model and pool it into a smaller label set."""

    def __init__(self, pooling_matrix: np.ndarray) -> None:
        super().__init__()
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        self.pooling_matrix = tf.constant(pooling_matrix, tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="waveform_16k")])
    def __call__(self, waveform_16k):
        scores, _embeddings, _spectrogram = self.yamnet(waveform_16k)
        pooled_scores = tf.matmul(scores, self.pooling_matrix)
        probabilities = tf.reduce_mean(pooled_scores, axis=0)
        return {"probs": probabilities}


def export_saved_model(module: ReducedModule) -> None:
    """Persist the reduced SavedModel and an optional TFLite version."""
    tf.saved_model.save(module, EXPORT_DIR, signatures=module.__call__.get_concrete_function())
    print(f"[OK] SavedModel -> {EXPORT_DIR}")

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
        tflite_model = converter.convert()
        output_path = os.path.join(EXPORT_DIR, "reduced_yamnet.tflite")
        with open(output_path, "wb") as handle:
            handle.write(tflite_model)
        print(f"[OK] TFLite -> {output_path}")
    except Exception as exc:
        print(f"[WARN] TFLite conversion skipped: {exc}")


def main() -> None:
    """Build the reduced-label model and write its label metadata."""
    names = load_yamnet_names()
    pooling_matrix, labels = build_pooling_matrix(names, REDUCED_MAP, AGGREGATION)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(LABELS_JSON, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
    print(f"[OK] labels -> {LABELS_JSON}: {labels}")

    module = ReducedModule(pooling_matrix)
    _ = module(tf.zeros([16000], tf.float32))
    export_saved_model(module)


if __name__ == "__main__":
    main()
