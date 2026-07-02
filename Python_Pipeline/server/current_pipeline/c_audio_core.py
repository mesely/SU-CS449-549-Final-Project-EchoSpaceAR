"""Audio core: resampling, stereo summaries, and YAMNet classification."""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from dataclasses import dataclass

import joblib
import numpy as np
from scipy import signal

from .b_config import (
    BEME_HEAD_MODEL_PATH,
    BEME_STEREO_BLEND_ALPHA,
    BEME_STEREO_GATE_ENABLED,
    BEME_STEREO_GATE_ENTROPY_MIN,
    BEME_STEREO_HEAD_MODEL_PATH,
    BEME_STEREO_GATE_MARGIN_MAX,
    BEME_STEREO_GATE_MIN_DECORRELATION,
    BEME_STEREO_GATE_MIN_DIRECTION_CONF,
    BEME_STEREO_GATE_MIN_ILD_DB,
    BEME_STEREO_GATE_MIN_SCORE,
    BEME_STEREO_GATE_MIN_STEREO_DELTA,
    BEME_STEREO_GATE_MIN_TARGET_PROB,
    BEME_STEREO_GATE_TARGET_LABELS,
    BEME_STEREO_GATE_TARGET_ONLY,
    BEME_STEREO_GATE_TOP1_MAX,
    BEME_STEREO_GATE_TOPK,
    BEME_STEREO_UPLIFT_ONLY,
    CLASSIFIER_MODE,
    CLASSIFIER_ENABLED,
    CONFUSION_PAIR_REFINEMENT_ENABLED,
    GLASS_IMPULSE_ALPHA,
    GLASS_EVENT_ALPHA,
    GLASS_EVENT_BOOST,
    GLASS_EVENT_CONFIRM_ENABLED,
    GLASS_EVENT_DECAY_MIN,
    GLASS_EVENT_MEMORY_WINDOWS,
    GLASS_EVENT_MIN_BURST,
    GLASS_IMPULSE_GLASS_BOOST,
    GLASS_IMPULSE_MIN_COMPETITOR_PROB,
    GLASS_IMPULSE_MIN_GLASS_PROB,
    GLASS_IMPULSE_MIN_SCORE,
    GLASS_IMPULSE_REFINEMENT_ENABLED,
    GLASS_IMPULSE_TRIGGER_MARGIN,
    HYBRID_FULL_WEIGHT,
    HYBRID_LEARNED_WEIGHT,
    HYBRID_SEMANTIC_WEIGHT,
    HYBRID_SPECTRAL_WEIGHT,
    LEARNED_HEAD_MODEL_PATH,
    MULTIRES_BRANCHES,
    MULTIRES_FAMILY_BY_LABEL,
    MULTIRES_FRONTEND_ENABLED,
    MULTIRES_ONSET_ALPHA_BOOST,
    MULTIRES_ONSET_ATTACK_S,
    MULTIRES_ONSET_RELEASE_S,
    MULTIRES_TRANSIENT_CONFIG,
    PAIRWISE_REFINER_MODEL_PATH,
    RAIL_SPEECH_ALPHA,
    RAIL_SPEECH_MEMORY_ALPHA,
    RAIL_SPEECH_MEMORY_BOOST,
    RAIL_SPEECH_MEMORY_MAX_TONALITY,
    RAIL_SPEECH_MEMORY_MAX_VOICING,
    RAIL_SPEECH_MEMORY_MIN_RAIL,
    RAIL_SPEECH_MEMORY_REFINEMENT_ENABLED,
    RAIL_SPEECH_MEMORY_WINDOWS,
    RAIL_SPEECH_MIN_SPEECH_PROB,
    RAIL_SPEECH_RAIL_BOOST,
    RAIL_SPEECH_REFINEMENT_ENABLED,
    RAIL_SPEECH_TONALITY_MAX,
    RAIL_SPEECH_TRIGGER_MARGIN,
    RAIL_SPEECH_VOICING_MAX,
    SEMANTIC_LABELS_JSON,
    SPECTRAL_MODEL_PATH,
    SPATIAL_CENTER_CORRELATION,
    SPATIAL_CENTER_ILD_DB,
    SPATIAL_DIRECTION_MIN_ILD_DB,
    SPATIAL_MAX_GCC_DELAY_S,
    SPATIAL_MAX_ILD_DB,
    STEREO_SIDE_CHANNEL_ENABLED,
    TEMPORAL_FAMILY_BY_LABEL,
    TEMPORAL_FRONTEND_BRANCHES,
    TEMPORAL_FRONTEND_ENABLED,
    TEMPORAL_SUSTAINED_BOOST_ALPHA,
    TEMPORAL_SUSTAINED_PEAK_MARGIN,
    TEMPORAL_SUSTAINED_PULL_ALPHA,
    TEMPORAL_TRANSIENT_CONFIG,
    TF_AVAILABLE,
    USE_SEMANTIC_YAMNET,
    YAMNET_MODEL_DIR,
    hub,
    tf,
    resample_linear,
)
from .i_beme import ensure_beme_available, predict_autobeme_proba


EPS = 1e-9
SPECTRAL_FRAME_S = 0.025
SPECTRAL_HOP_S = 0.010
SPECTRAL_MEL_BINS = 48
SPECTRAL_FEATURE_STATS = ("mean", "std", "max", "p90")
YAMNET_LABEL_MAP: dict[str, list[str | tuple[str, float]]] = {
    "speech": [
        "Speech",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Whispering",
    ],
    "crowd": [
        "Chatter",
        "Crowd",
        "Hubbub, speech noise, speech babble",
        "Cheering",
        "Applause",
        "Children playing",
    ],
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
    "vehicle_horn": [
        "Vehicle horn, car horn, honking",
        "Toot",
        "Air horn, truck horn",
        "Foghorn",
    ],
    "traffic_road": [
        "Motor vehicle (road)",
        "Traffic noise, roadway noise",
        "Car passing by",
        "Race car, auto racing",
        "Skidding",
        "Tire squeal",
    ],
    "car_bus_truck": ["Car", "Bus", "Truck", "Ice cream truck, ice cream van"],
    "sirens": [
        "Siren",
        "Civil defense siren",
        "Emergency vehicle",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Car alarm",
    ],
    "rail": [
        "Rail transport",
        "Train",
        "Train whistle",
        "Train horn",
        "Railroad car, train wagon",
        "Train wheels squealing",
        "Subway, metro, underground",
    ],
    "aircraft": [
        "Aircraft",
        "Aircraft engine",
        "Jet engine",
        "Propeller, airscrew",
        "Helicopter",
        "Fixed-wing aircraft, airplane",
    ],
    "engine_motion": [
        "Engine",
        "Light engine (high frequency)",
        "Medium engine (mid frequency)",
        "Heavy engine (low frequency)",
        "Engine knocking",
        "Engine starting",
        "Idling",
        "Accelerating, revving, vroom",
    ],
    "alarms_buzzer": [
        ("Alarm", 1.0),
        ("Alarm clock", 1.0),
        ("Buzzer", 1.0),
        ("Smoke detector, smoke alarm", 1.0),
        ("Fire alarm", 1.0),
        ("Doorbell", 2.5),
        ("Ding-dong", 2.0),
        ("Chime", 1.5),
        ("Microwave oven", 3.0),
    ],
    "phone_ring": [
        ("Telephone bell ringing", 2.0),
        ("Ringtone", 2.5),
        "Telephone",
        "Telephone dialing, DTMF",
        "Dial tone",
        "Busy signal",
    ],
    "wind_rain": [
        "Wind",
        "Rustling leaves",
        "Wind noise (microphone)",
        "Thunderstorm",
        "Thunder",
        "Rain",
        "Raindrop",
        "Rain on surface",
    ],
    "door_knock": [
        "Door",
        "Knock",
        "Tap",
        "Slam",
        "Sliding door",
        "Cupboard open or close",
        "Drawer open or close",
    ],
    "glass_break": [
        "Glass",
        "Shatter",
        "Smash, crash",
        "Breaking",
        "Chink, clink",
        "Crack",
    ],
    "explosion_gunshot": [
        "Explosion",
        ("Gunshot, gunfire", 2.5),
        "Machine gun",
        "Fusillade",
        "Artillery fire",
        "Cap gun",
        ("Fireworks", 2.0),
        "Firecracker",
        "Burst, pop",
        "Eruption",
        ("Boom", 1.5),
    ],
}


@dataclass(slots=True)
class SpatialSnapshot:
    """Compact stereo/spatial summary published beside the mono decision."""

    mode: str
    channels: int
    direction: str
    direction_confidence: float
    ild_db: float
    ipd_rad: float
    gcc_delay_s: float
    correlation: float
    left_rms: float
    right_rms: float

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "channels": int(self.channels),
            "direction": self.direction,
            "direction_confidence": float(self.direction_confidence),
            "ild_db": float(self.ild_db),
            "ipd_rad": float(self.ipd_rad),
            "gcc_delay_s": float(self.gcc_delay_s),
            "correlation": float(self.correlation),
            "left_rms": float(self.left_rms),
            "right_rms": float(self.right_rms),
        }


@dataclass(slots=True)
class StereoGateDecision:
    score: float
    uncertainty_score: float
    target_score: float
    spatial_score: float
    target_mask: np.ndarray


def ensure_frame_major(samples: np.ndarray, channels_hint: int = 1) -> np.ndarray:
    """Normalize mono/stereo audio into `(frames, channels)` float32 form."""
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        channels = max(1, int(channels_hint))
        if channels > 1 and array.size % channels == 0:
            return array.reshape(-1, channels)
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Unsupported audio rank: {array.ndim}")
    if array.shape[0] <= 4 and array.shape[1] > array.shape[0]:
        return array.T.astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False)


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    """Mono baseline: keep the classic mean downmix, preserve stereo elsewhere."""
    frames = ensure_frame_major(samples)
    if frames.shape[1] == 1:
        return frames[:, 0].astype(np.float32, copy=False)
    return np.mean(frames, axis=1).astype(np.float32, copy=False)


def resample_frame_major_audio(samples: np.ndarray, input_sr: int, target_sr: int) -> np.ndarray:
    """Resample each channel while preserving `(frames, channels)` layout."""
    frames = ensure_frame_major(samples)
    if int(input_sr) == int(target_sr):
        return frames.astype(np.float32, copy=False)
    channels = [
        resample_linear(frames[:, channel_index].astype(np.float32, copy=False), input_sr, target_sr)
        for channel_index in range(int(frames.shape[1]))
    ]
    if not channels:
        return np.zeros((0, 1), dtype=np.float32)
    min_length = min(int(channel.shape[0]) for channel in channels)
    if min_length <= 0:
        return np.zeros((0, len(channels)), dtype=np.float32)
    return np.stack([channel[:min_length].astype(np.float32, copy=False) for channel in channels], axis=1)


def summarize_spatial_audio(samples: np.ndarray, sample_rate: float) -> SpatialSnapshot:
    """Summarize stereo cues while keeping the mono baseline classifier path."""
    frames = ensure_frame_major(samples)
    channels = int(frames.shape[1])

    if not STEREO_SIDE_CHANNEL_ENABLED or channels < 2:
        mono = downmix_to_mono(frames)
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + EPS))
        return SpatialSnapshot("mono", channels, "unknown", 0.0, 0.0, 0.0, 0.0, 1.0, mono_rms, mono_rms)

    left = frames[:, 0].astype(np.float32, copy=False)
    right = frames[:, 1].astype(np.float32, copy=False)
    left_rms = float(np.sqrt(np.mean(left.astype(np.float64) ** 2) + EPS))
    right_rms = float(np.sqrt(np.mean(right.astype(np.float64) ** 2) + EPS))
    ild_db = float(20.0 * np.log10((left_rms + EPS) / (right_rms + EPS)))
    ipd_rad = float(_dominant_ipd(left, right))
    gcc_delay_s = float(_gcc_phat_delay(left, right, sample_rate))
    correlation = float(_safe_corrcoef(left, right))
    direction, direction_conf = _infer_direction(ild_db, correlation)
    return SpatialSnapshot(
        "stereo",
        channels,
        direction,
        direction_conf,
        ild_db,
        ipd_rad,
        gcc_delay_s,
        correlation,
        left_rms,
        right_rms,
    )


def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 1.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < EPS or right_std < EPS:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _dominant_ipd(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    window = np.hanning(left.size).astype(np.float32, copy=False)
    left_fft = np.fft.rfft(left * window)
    right_fft = np.fft.rfft(right * window)
    if left_fft.size <= 1 or right_fft.size <= 1:
        return 0.0
    power = np.abs(left_fft) + np.abs(right_fft)
    dominant_index = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    if dominant_index <= 0:
        return 0.0
    return float(np.angle(left_fft[dominant_index] * np.conj(right_fft[dominant_index])))


def _gcc_phat_delay(left: np.ndarray, right: np.ndarray, sample_rate: float) -> float:
    if left.size == 0 or right.size == 0 or sample_rate <= 0:
        return 0.0
    n = 1
    target = left.size + right.size
    while n < target:
        n <<= 1
    left_fft = np.fft.rfft(left, n=n)
    right_fft = np.fft.rfft(right, n=n)
    cross = left_fft * np.conj(right_fft)
    cross /= np.abs(cross) + EPS
    corr = np.fft.irfft(cross, n=n)
    max_shift = max(1, min(int(sample_rate * SPATIAL_MAX_GCC_DELAY_S), n // 2))
    corr = np.concatenate((corr[-max_shift:], corr[: max_shift + 1]))
    shift = int(np.argmax(np.abs(corr)) - max_shift)
    return float(shift / float(sample_rate))


def _infer_direction(ild_db: float, correlation: float) -> tuple[str, float]:
    abs_ild = abs(float(ild_db))
    if abs_ild <= SPATIAL_CENTER_ILD_DB and correlation >= SPATIAL_CENTER_CORRELATION:
        return "center", float(min(1.0, correlation))
    if abs_ild < SPATIAL_DIRECTION_MIN_ILD_DB:
        return "unknown", float(abs_ild / max(SPATIAL_DIRECTION_MIN_ILD_DB, EPS))
    direction = "left" if ild_db > 0.0 else "right"
    ild_conf = min(1.0, abs_ild / max(SPATIAL_MAX_ILD_DB, EPS))
    decorrelation_bonus = float(np.clip(1.0 - max(correlation, 0.0), 0.0, 1.0))
    confidence = np.clip(0.65 * ild_conf + 0.35 * decorrelation_bonus, 0.0, 1.0)
    return direction, float(confidence)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = SPECTRAL_MEL_BINS) -> np.ndarray:
    def hz_to_mel(value: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + (value / 700.0))

    def mel_to_hz(value: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    mel_min = hz_to_mel(np.asarray([30.0], dtype=np.float64))[0]
    mel_max = hz_to_mel(np.asarray([sample_rate * 0.5], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for index in range(n_mels):
        left = max(0, bins[index])
        center = max(left + 1, bins[index + 1])
        right = max(center + 1, bins[index + 2])
        for bin_index in range(left, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, min(right, filters.shape[1])):
            filters[index, bin_index] = (right - bin_index) / max(right - center, 1)
    return filters


def extract_spectral_feature_vector(mono_audio_16k: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Convert a mono waveform into a compact log-mel + spectral summary vector."""
    mono = np.asarray(mono_audio_16k, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        stat_count = SPECTRAL_MEL_BINS * len(SPECTRAL_FEATURE_STATS)
        return np.zeros(stat_count + 10, dtype=np.float32)

    frame_length = max(256, int(round(sample_rate * SPECTRAL_FRAME_S)))
    hop_length = max(128, int(round(sample_rate * SPECTRAL_HOP_S)))
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1

    _freqs, _times, stft = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=max(frame_length - hop_length, 0),
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float32, copy=False)
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        magnitude = np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)

    mel_basis = _mel_filterbank(sample_rate, n_fft)
    mel_spec = np.maximum(mel_basis @ magnitude, EPS)
    log_mel = np.log(mel_spec).astype(np.float32, copy=False)
    mel_stats = [
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.max(log_mel, axis=1),
        np.percentile(log_mel, 90.0, axis=1),
    ]

    power = magnitude ** 2
    freq_axis = np.linspace(0.0, sample_rate * 0.5, magnitude.shape[0], dtype=np.float32)
    frame_energy = np.sum(power, axis=0) + EPS
    centroid = np.sum(freq_axis[:, None] * power, axis=0) / frame_energy
    bandwidth = np.sqrt(np.sum(((freq_axis[:, None] - centroid[None, :]) ** 2) * power, axis=0) / frame_energy)
    cumulative = np.cumsum(power, axis=0)
    rolloff_target = 0.85 * frame_energy
    rolloff = freq_axis[np.argmax(cumulative >= rolloff_target[None, :], axis=0)]
    flatness = np.exp(np.mean(np.log(power + EPS), axis=0)) / (np.mean(power, axis=0) + EPS)
    flux = np.sqrt(np.sum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]) ** 2, axis=0))
    zcr = np.mean(np.abs(np.diff(np.signbit(mono).astype(np.float32))), dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + EPS))
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    crest = peak / max(rms, EPS)

    global_stats = np.asarray(
        [
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(bandwidth)),
            float(np.mean(rolloff)),
            float(np.mean(flatness)),
            float(np.std(flatness)),
            float(np.mean(flux)),
            float(np.std(flux)),
            float(zcr),
            float(crest),
        ],
        dtype=np.float32,
    )
    return np.concatenate([*(item.astype(np.float32, copy=False) for item in mel_stats), global_stats]).astype(
        np.float32,
        copy=False,
    )


def _build_projection_matrix(
    yamnet_names: list[str],
    label_map: dict[str, list[str | tuple[str, float]]],
) -> tuple[list[str], np.ndarray]:
    labels = list(label_map.keys())
    name_to_index = {name: idx for idx, name in enumerate(yamnet_names)}
    projection = np.zeros((len(yamnet_names), len(labels)), dtype=np.float32)
    for column, label in enumerate(labels):
        total_weight = 0.0
        for source in label_map[label]:
            if isinstance(source, tuple):
                source_name, weight = source
            else:
                source_name, weight = source, 1.0
            source_index = name_to_index.get(source_name)
            if source_index is None:
                continue
            projection[source_index, column] += float(weight)
            total_weight += float(weight)
        if total_weight > 0.0:
            projection[:, column] /= total_weight
    return labels, projection


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32, copy=False)


def _apply_probability_threshold(probability: float, threshold: float) -> float:
    clipped_probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))
    clipped_threshold = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    logit_probability = np.log(clipped_probability / (1.0 - clipped_probability))
    logit_threshold = np.log(clipped_threshold / (1.0 - clipped_threshold))
    return float(1.0 / (1.0 + np.exp(-(logit_probability - logit_threshold))))


def _normalized_score_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), EPS, 1.0)
    total = float(np.sum(values))
    if total <= EPS or values.size <= 1:
        return 0.0
    distribution = values / total
    entropy = -float(np.sum(distribution * np.log(distribution + EPS)))
    return float(np.clip(entropy / max(np.log(values.size), EPS), 0.0, 1.0))


class _SemanticBackbone:
    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class _ProjectedYamnetMixin:
    def __init__(self) -> None:
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.yamnet.class_map_path().numpy().decode("utf-8")
        names: list[str] = []
        with tf.io.gfile.GFile(class_map_path, "r") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                names.append(row["display_name"])
        self.class_names, self.projection = _build_projection_matrix(names, YAMNET_LABEL_MAP)

    def _project_waveform(self, waveform_16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        waveform = tf.convert_to_tensor(waveform_16k, dtype=tf.float32)
        scores, embeddings, _spectrogram = self.yamnet(waveform)
        pooled_scores = tf.reduce_mean(scores, axis=0).numpy()
        projected = np.asarray(pooled_scores, dtype=np.float32) @ self.projection
        embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32, copy=False)
        return np.clip(projected, 0.0, 1.0).astype(np.float32, copy=False), embedding_mean

    def project_with_embeddings(self, mono_audio_16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._project_waveform(mono_audio_16k)

    def build_stereo_beme_features(
        self,
        mono_audio_16k: np.ndarray,
        frames_audio_16k: np.ndarray | None = None,
        feature_mode: str = "full",
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_baseline, embedding_mean = self.project_with_embeddings(mono_audio_16k)
        base_feature = np.concatenate([embedding_mean, projected_baseline]).astype(np.float32, copy=False)
        compact_base_feature = projected_baseline.astype(np.float32, copy=False)

        def _compose_feature(
            left_projection: np.ndarray | None = None,
            right_projection: np.ndarray | None = None,
            stereo_delta: np.ndarray | None = None,
            stereo_stats: np.ndarray | None = None,
        ) -> np.ndarray:
            left_projection = (
                np.asarray(left_projection, dtype=np.float32)
                if left_projection is not None
                else np.zeros_like(projected_baseline, dtype=np.float32)
            )
            right_projection = (
                np.asarray(right_projection, dtype=np.float32)
                if right_projection is not None
                else np.zeros_like(projected_baseline, dtype=np.float32)
            )
            stereo_delta = (
                np.asarray(stereo_delta, dtype=np.float32)
                if stereo_delta is not None
                else np.zeros_like(projected_baseline, dtype=np.float32)
            )
            stereo_stats = (
                np.asarray(stereo_stats, dtype=np.float32)
                if stereo_stats is not None
                else np.zeros(8, dtype=np.float32)
            )
            mode = str(feature_mode or "full").strip().lower()
            if mode == "delta_stats":
                return np.concatenate([base_feature, stereo_delta, stereo_stats]).astype(np.float32, copy=False)
            if mode == "stats_only":
                return np.concatenate([base_feature, stereo_stats]).astype(np.float32, copy=False)
            if mode == "compact":
                return np.concatenate(
                    [compact_base_feature, left_projection, right_projection, stereo_delta, stereo_stats]
                ).astype(np.float32, copy=False)
            if mode == "compact_delta_stats":
                return np.concatenate([compact_base_feature, stereo_delta, stereo_stats]).astype(np.float32, copy=False)
            if mode == "compact_stats":
                return np.concatenate([compact_base_feature, stereo_stats]).astype(np.float32, copy=False)
            return np.concatenate([base_feature, left_projection, right_projection, stereo_delta, stereo_stats]).astype(
                np.float32,
                copy=False,
            )

        if frames_audio_16k is None:
            return projected_baseline, _compose_feature()

        frames = ensure_frame_major(frames_audio_16k)
        if frames.shape[0] == 0 or frames.shape[1] < 2:
            return projected_baseline, _compose_feature()

        left = frames[:, 0].astype(np.float32, copy=False)
        right = frames[:, 1].astype(np.float32, copy=False)
        left_projection, _left_embedding = self._project_waveform(left)
        right_projection, _right_embedding = self._project_waveform(right)
        stereo_delta = np.abs(left_projection - right_projection).astype(np.float32, copy=False)
        spatial = summarize_spatial_audio(frames, 16000.0)
        stereo_stats = np.array(
            [
                float(spatial.ild_db),
                float(spatial.ipd_rad),
                float(spatial.gcc_delay_s),
                float(spatial.correlation),
                float(spatial.left_rms),
                float(spatial.right_rms),
                float(abs(spatial.left_rms - spatial.right_rms)),
                float((spatial.left_rms + spatial.right_rms) * 0.5),
            ],
            dtype=np.float32,
        )
        stereo_feature = _compose_feature(
            left_projection=left_projection,
            right_projection=right_projection,
            stereo_delta=stereo_delta,
            stereo_stats=stereo_stats,
        )
        return projected_baseline, stereo_feature


class _SemanticYamnetBackbone(_SemanticBackbone):
    def __init__(self, class_names: list[str]) -> None:
        model = tf.saved_model.load(YAMNET_MODEL_DIR)
        self.infer = model.__call__.get_concrete_function()
        self.class_names = class_names

    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        waveform = tf.convert_to_tensor(mono_audio_16k, dtype=tf.float32)
        output = self.infer(waveform_16k=waveform)
        return output["probs"].numpy()


class _FullProjectedYamnetBackbone(_ProjectedYamnetMixin, _SemanticBackbone):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        projected, _embedding_mean = self.project_with_embeddings(mono_audio_16k)
        return projected


class _LearnedSemanticHeadBackbone(_ProjectedYamnetMixin, _SemanticBackbone):
    def __init__(self, artifact_path: str) -> None:
        if not os.path.exists(artifact_path):
            raise RuntimeError(f"Learned semantic-head artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        self.artifact_path = artifact_path
        self.feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(artifact["feature_std"], dtype=np.float32), 1e-6)
        self.classifier = artifact["classifier"]
        self.trained_labels = list(artifact["trained_labels"])
        self.blend_alpha = float(artifact.get("blend_alpha", 0.65))
        self.uplift_only = bool(artifact.get("uplift_only", True))
        self.per_label_blend_alpha = {str(key): float(value) for key, value in artifact.get("per_label_blend_alpha", {}).items()}
        self.per_label_uplift_only = {str(key): bool(value) for key, value in artifact.get("per_label_uplift_only", {}).items()}
        self.per_label_decision_threshold = {
            str(key): float(value) for key, value in artifact.get("per_label_decision_threshold", {}).items()
        }
        self.trained_index_by_label = {label: idx for idx, label in enumerate(self.trained_labels)}
        self.output_labels = list(artifact.get("labels", list(YAMNET_LABEL_MAP.keys())))
        self.output_index_by_label = {label: idx for idx, label in enumerate(self.output_labels)}
        super().__init__()
        self.semantic_fusion_backbone = None
        if os.path.exists(os.path.join(YAMNET_MODEL_DIR, "saved_model.pb")) or os.path.exists(
            os.path.join(YAMNET_MODEL_DIR, "saved_model.pbtxt")
        ):
            try:
                self.semantic_fusion_backbone = _SemanticYamnetBackbone(self.output_labels)
            except Exception:
                self.semantic_fusion_backbone = None
        if self.class_names != self.output_labels:
            raise RuntimeError(
                "Learned semantic-head artifact labels do not match runtime semantic label order. "
                f"artifact={self.output_labels}, runtime={self.class_names}"
            )

    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        projected_baseline, embedding_mean = self.project_with_embeddings(mono_audio_16k)
        features = np.concatenate([embedding_mean, projected_baseline]).astype(np.float32, copy=False).reshape(1, -1)
        normalized = ((features - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(
            np.float32,
            copy=False,
        )
        raw_probabilities = self.classifier.predict_proba(normalized)
        raw_probabilities = np.asarray(raw_probabilities, dtype=np.float32)
        if raw_probabilities.ndim == 3:
            learned_probs = raw_probabilities[:, :, -1].reshape(1, -1)[0]
        elif raw_probabilities.ndim == 2:
            learned_probs = raw_probabilities[0]
        else:
            learned_probs = _sigmoid(np.asarray(self.classifier.decision_function(normalized), dtype=np.float32).reshape(-1))

        fusion_baseline = (
            self.semantic_fusion_backbone.predict(mono_audio_16k, frames_audio_16k)
            if self.semantic_fusion_backbone is not None
            else projected_baseline
        )
        fused = np.asarray(fusion_baseline, dtype=np.float32).copy()
        for label, trained_index in self.trained_index_by_label.items():
            output_index = self.output_index_by_label[label]
            projected = float(projected_baseline[output_index])
            baseline = float(fused[output_index])
            learned = float(learned_probs[trained_index])
            delta = learned - projected
            label_alpha = float(self.per_label_blend_alpha.get(label, self.blend_alpha))
            label_uplift_only = bool(self.per_label_uplift_only.get(label, self.uplift_only))
            fused_probability = baseline
            if label_alpha > 0.0 and (not label_uplift_only or delta > 0.0):
                fused_probability = float(np.clip(baseline + (label_alpha * delta), 0.0, 1.0))
            threshold = float(self.per_label_decision_threshold.get(label, 0.50))
            fused[output_index] = _apply_probability_threshold(fused_probability, threshold)
        return fused


class _BemeSemanticHeadBackbone(_ProjectedYamnetMixin, _SemanticBackbone):
    def __init__(self, artifact_path: str) -> None:
        if not os.path.exists(artifact_path):
            raise RuntimeError(f"BEME semantic-head artifact not found: {artifact_path}")
        ensure_beme_available()
        artifact = joblib.load(artifact_path)
        self.artifact_path = artifact_path
        self.feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(artifact["feature_std"], dtype=np.float32), 1e-6)
        self.classifier = artifact["classifier"]
        self.trained_labels = list(artifact["trained_labels"])
        self.stereo_aware = bool(artifact.get("stereo_aware", False))
        self.stereo_feature_mode = str(artifact.get("stereo_feature_mode", "full"))
        self.blend_alpha = float(artifact.get("blend_alpha", 0.55))
        self.uplift_only = bool(artifact.get("uplift_only", False))
        self.per_label_blend_alpha = {str(key): float(value) for key, value in artifact.get("per_label_blend_alpha", {}).items()}
        self.per_label_uplift_only = {str(key): bool(value) for key, value in artifact.get("per_label_uplift_only", {}).items()}
        self.per_label_decision_threshold = {
            str(key): float(value) for key, value in artifact.get("per_label_decision_threshold", {}).items()
        }
        self.trained_index_by_label = {label: idx for idx, label in enumerate(self.trained_labels)}
        self.output_labels = list(artifact.get("labels", list(YAMNET_LABEL_MAP.keys())))
        self.output_index_by_label = {label: idx for idx, label in enumerate(self.output_labels)}
        super().__init__()
        self.semantic_fusion_backbone = None
        if os.path.exists(os.path.join(YAMNET_MODEL_DIR, "saved_model.pb")) or os.path.exists(
            os.path.join(YAMNET_MODEL_DIR, "saved_model.pbtxt")
        ):
            try:
                self.semantic_fusion_backbone = _SemanticYamnetBackbone(self.output_labels)
            except Exception:
                self.semantic_fusion_backbone = None
        if self.class_names != self.output_labels:
            raise RuntimeError(
                "BEME semantic-head artifact labels do not match runtime semantic label order. "
                f"artifact={self.output_labels}, runtime={self.class_names}"
            )

    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        if self.stereo_aware:
            projected_baseline, feature_vector = self.build_stereo_beme_features(
                mono_audio_16k,
                frames_audio_16k,
                feature_mode=self.stereo_feature_mode,
            )
            features = feature_vector.reshape(1, -1).astype(np.float32, copy=False)
        else:
            projected_baseline, embedding_mean = self.project_with_embeddings(mono_audio_16k)
            features = np.concatenate([embedding_mean, projected_baseline]).astype(np.float32, copy=False).reshape(1, -1)
        normalized = ((features - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(
            np.float32,
            copy=False,
        )
        learned_probs = predict_autobeme_proba(self.classifier, normalized)[0]

        fusion_baseline = (
            self.semantic_fusion_backbone.predict(mono_audio_16k, frames_audio_16k)
            if self.semantic_fusion_backbone is not None
            else projected_baseline
        )
        fused = np.asarray(fusion_baseline, dtype=np.float32).copy()
        for label, trained_index in self.trained_index_by_label.items():
            output_index = self.output_index_by_label[label]
            projected = float(projected_baseline[output_index])
            baseline = float(fused[output_index])
            learned = float(learned_probs[trained_index])
            delta = learned - projected
            label_alpha = float(self.per_label_blend_alpha.get(label, self.blend_alpha))
            label_uplift_only = bool(self.per_label_uplift_only.get(label, self.uplift_only))
            fused_probability = baseline
            if label_alpha > 0.0 and (not label_uplift_only or delta > 0.0):
                fused_probability = float(np.clip(baseline + (label_alpha * delta), 0.0, 1.0))
            threshold = float(self.per_label_decision_threshold.get(label, 0.50))
            fused[output_index] = _apply_probability_threshold(fused_probability, threshold)
        return fused


class _SpectralBackbone(_SemanticBackbone):
    def __init__(self, artifact_path: str, expected_labels: list[str] | None = None) -> None:
        if not os.path.exists(artifact_path):
            raise RuntimeError(f"Spectral backbone artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
        self.classifier = artifact["classifier"]
        artifact_labels = list(artifact["labels"])
        self.class_names = list(expected_labels or artifact_labels)
        self.output_index_by_artifact = {label: idx for idx, label in enumerate(artifact_labels)}
        self.expected_index_by_label = {label: idx for idx, label in enumerate(self.class_names)}
        self.feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(artifact["feature_std"], dtype=np.float32), 1e-6)

    def predict(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        features = extract_spectral_feature_vector(mono_audio_16k, sample_rate=16000)
        normalized = ((features - self.feature_mean) / self.feature_std).reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(normalized)
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if probabilities.ndim == 3:
                probabilities = probabilities[:, :, -1].reshape(1, -1)
            if probabilities.ndim == 2 and probabilities.shape[1] == len(self.class_names):
                return probabilities[0]
            if probabilities.ndim == 2:
                base = probabilities[0]
                expanded = np.zeros(len(self.class_names), dtype=np.float32)
                for label, artifact_index in self.output_index_by_artifact.items():
                    expected_index = self.expected_index_by_label.get(label)
                    if expected_index is not None:
                        expanded[expected_index] = float(base[artifact_index])
                return expanded
        logits = np.asarray(self.classifier.decision_function(normalized), dtype=np.float32).reshape(-1)
        base = _sigmoid(logits)
        if len(base) == len(self.class_names):
            return base
        expanded = np.zeros(len(self.class_names), dtype=np.float32)
        for label, artifact_index in self.output_index_by_artifact.items():
            expected_index = self.expected_index_by_label.get(label)
            if expected_index is not None:
                expanded[expected_index] = float(base[artifact_index])
        return expanded


class _PairwiseSemanticRefiner:
    def __init__(self, artifact_path: str, class_names: list[str]) -> None:
        artifact = joblib.load(artifact_path)
        self.label_to_index = {label: idx for idx, label in enumerate(class_names)}
        self.pairs: list[dict[str, object]] = []
        for item in artifact.get("pairs", []):
            labels = list(item.get("labels", []))
            if len(labels) != 2:
                continue
            left_label, right_label = labels
            if left_label not in self.label_to_index or right_label not in self.label_to_index:
                continue
            self.pairs.append(
                {
                    "labels": labels,
                    "indices": (self.label_to_index[left_label], self.label_to_index[right_label]),
                    "classifier": item["classifier"],
                    "feature_mean": np.asarray(item["feature_mean"], dtype=np.float32),
                    "feature_std": np.maximum(np.asarray(item["feature_std"], dtype=np.float32), 1e-6),
                    "blend_alpha": float(item.get("blend_alpha", 0.6)),
                    "trigger_margin": float(item.get("trigger_margin", 0.15)),
                    "min_top_prob": float(item.get("min_top_prob", 0.08)),
                    "min_confidence": float(item.get("min_confidence", 0.58)),
                }
            )

    def refine(self, mono_audio_16k: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
        if not self.pairs:
            return probabilities

        current = np.asarray(probabilities, dtype=np.float32).copy()
        order = np.argsort(current)[::-1]
        top_labels = {labels[int(index)] for index in order[:3]}
        pair_feature: np.ndarray | None = None

        for pair in self.pairs:
            left_label, right_label = pair["labels"]
            if left_label not in top_labels and right_label not in top_labels:
                continue

            left_index, right_index = pair["indices"]
            left_score = float(current[left_index])
            right_score = float(current[right_index])
            top_prob = max(left_score, right_score)
            margin = abs(left_score - right_score)
            if top_prob < float(pair["min_top_prob"]) or margin > float(pair["trigger_margin"]):
                continue

            if pair_feature is None:
                pair_feature = extract_spectral_feature_vector(mono_audio_16k, sample_rate=16000)
            normalized = ((pair_feature - pair["feature_mean"]) / pair["feature_std"]).reshape(1, -1)
            pair_probs = np.asarray(pair["classifier"].predict_proba(normalized)[0], dtype=np.float32)
            if pair_probs.size != 2:
                continue
            confidence = abs(float(pair_probs[1]) - float(pair_probs[0]))
            if confidence < float(pair["min_confidence"]):
                continue

            pair_mass = max(left_score + right_score, EPS)
            specialist_sum = max(float(np.sum(pair_probs)), EPS)
            target_left = pair_mass * (float(pair_probs[0]) / specialist_sum)
            target_right = pair_mass * (float(pair_probs[1]) / specialist_sum)
            activation = np.clip((float(pair["trigger_margin"]) - margin) / max(float(pair["trigger_margin"]), EPS), 0.0, 1.0)
            alpha = float(pair["blend_alpha"]) * (0.35 + (0.65 * activation))
            current[left_index] = np.clip(((1.0 - alpha) * left_score) + (alpha * target_left), 0.0, 1.0)
            current[right_index] = np.clip(((1.0 - alpha) * right_score) + (alpha * target_right), 0.0, 1.0)

        return current


class _RailSpeechSpecialist:
    def __init__(self, class_names: list[str]) -> None:
        self.rail_index = class_names.index("rail") if "rail" in class_names else -1
        self.speech_index = class_names.index("speech") if "speech" in class_names else -1
        history_len = max(1, int(RAIL_SPEECH_MEMORY_WINDOWS))
        self.recent_rail: deque[float] = deque(maxlen=history_len)
        self.recent_speech: deque[float] = deque(maxlen=history_len)
        self.recent_voicing: deque[float] = deque(maxlen=history_len)
        self.recent_tonality: deque[float] = deque(maxlen=history_len)

    def reset(self) -> None:
        self.recent_rail.clear()
        self.recent_speech.clear()
        self.recent_voicing.clear()
        self.recent_tonality.clear()

    def refine(self, mono_audio_16k: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        if self.rail_index < 0 or self.speech_index < 0:
            return probabilities

        current = np.asarray(probabilities, dtype=np.float32).copy()
        rail_score = float(current[self.rail_index])
        speech_score = float(current[self.speech_index])
        voicing, tonality = _estimate_voicing_profile(mono_audio_16k, sample_rate=16000)
        rail_memory = float(np.mean(self.recent_rail)) if self.recent_rail else 0.0
        speech_memory = float(np.mean(self.recent_speech)) if self.recent_speech else 0.0
        voicing_memory = float(np.mean(self.recent_voicing)) if self.recent_voicing else voicing
        tonality_memory = float(np.mean(self.recent_tonality)) if self.recent_tonality else tonality

        margin = speech_score - rail_score
        should_apply_basic = (
            speech_score >= float(RAIL_SPEECH_MIN_SPEECH_PROB)
            and margin > 0.0
            and margin <= float(RAIL_SPEECH_TRIGGER_MARGIN)
            and voicing <= float(RAIL_SPEECH_VOICING_MAX)
            and tonality <= float(RAIL_SPEECH_TONALITY_MAX)
        )
        if should_apply_basic:
            activation = np.clip(
                (float(RAIL_SPEECH_TRIGGER_MARGIN) - margin) / max(float(RAIL_SPEECH_TRIGGER_MARGIN), EPS),
                0.0,
                1.0,
            )
            alpha = float(RAIL_SPEECH_ALPHA) * (0.35 + (0.65 * activation))
            redistribution = max(min(speech_score * alpha, speech_score), 0.0)
            current[self.speech_index] = np.clip(speech_score - redistribution, 0.0, 1.0)
            current[self.rail_index] = np.clip(rail_score + redistribution + float(RAIL_SPEECH_RAIL_BOOST), 0.0, 1.0)

        if RAIL_SPEECH_MEMORY_REFINEMENT_ENABLED:
            updated_rail = float(current[self.rail_index])
            updated_speech = float(current[self.speech_index])
            memory_margin = updated_speech - updated_rail
            low_voicing = max(voicing, voicing_memory) <= float(RAIL_SPEECH_MEMORY_MAX_VOICING)
            low_tonality = max(tonality, tonality_memory) <= float(RAIL_SPEECH_MEMORY_MAX_TONALITY)
            rail_supported = max(rail_memory, rail_score) >= float(RAIL_SPEECH_MEMORY_MIN_RAIL)
            speech_not_stable = speech_memory <= max(updated_speech + 0.02, 0.18)
            if (
                updated_speech >= float(RAIL_SPEECH_MIN_SPEECH_PROB)
                and rail_supported
                and low_voicing
                and low_tonality
                and speech_not_stable
                and memory_margin > 0.0
                and memory_margin <= float(RAIL_SPEECH_TRIGGER_MARGIN)
            ):
                activation = np.clip(
                    (float(RAIL_SPEECH_TRIGGER_MARGIN) - memory_margin) / max(float(RAIL_SPEECH_TRIGGER_MARGIN), EPS),
                    0.0,
                    1.0,
                )
                alpha = float(RAIL_SPEECH_MEMORY_ALPHA) * (0.35 + (0.65 * activation))
                redistribution = max(min(updated_speech * alpha, updated_speech), 0.0)
                current[self.speech_index] = np.clip(updated_speech - redistribution, 0.0, 1.0)
                current[self.rail_index] = np.clip(
                    updated_rail + redistribution + float(RAIL_SPEECH_MEMORY_BOOST),
                    0.0,
                    1.0,
                )

        self.recent_rail.append(float(current[self.rail_index]))
        self.recent_speech.append(float(current[self.speech_index]))
        self.recent_voicing.append(float(voicing))
        self.recent_tonality.append(float(tonality))
        return current


class _GlassImpulseSpecialist:
    def __init__(self, class_names: list[str]) -> None:
        self.glass_index = class_names.index("glass_break") if "glass_break" in class_names else -1
        self.competitor_labels = [label for label in ("cat", "music", "explosion_gunshot", "speech") if label in class_names]
        self.competitor_indices = [class_names.index(label) for label in self.competitor_labels]
        history_len = max(1, int(GLASS_EVENT_MEMORY_WINDOWS))
        self.recent_burst: deque[float] = deque(maxlen=history_len)
        self.recent_decay: deque[float] = deque(maxlen=history_len)

    def reset(self) -> None:
        self.recent_burst.clear()
        self.recent_decay.clear()

    def refine(self, mono_audio_16k: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        if self.glass_index < 0 or not self.competitor_indices:
            return probabilities

        current = np.asarray(probabilities, dtype=np.float32).copy()
        glass_score = float(current[self.glass_index])
        competitor_index = max(self.competitor_indices, key=lambda idx: float(current[idx]))
        competitor_score = float(current[competitor_index])
        margin = competitor_score - glass_score
        if (
            glass_score < float(GLASS_IMPULSE_MIN_GLASS_PROB)
            or competitor_score < float(GLASS_IMPULSE_MIN_COMPETITOR_PROB)
            or margin <= 0.0
            or margin > float(GLASS_IMPULSE_TRIGGER_MARGIN)
        ):
            return current

        impulse_score = _estimate_glass_impulse_score(mono_audio_16k, sample_rate=16000)
        burst_score, decay_score = _estimate_glass_burst_profile(mono_audio_16k, sample_rate=16000)
        if impulse_score < float(GLASS_IMPULSE_MIN_SCORE):
            self.recent_burst.append(float(burst_score))
            self.recent_decay.append(float(decay_score))
            return current

        activation = np.clip((impulse_score - float(GLASS_IMPULSE_MIN_SCORE)) / max(1.0 - float(GLASS_IMPULSE_MIN_SCORE), EPS), 0.0, 1.0)
        alpha = float(GLASS_IMPULSE_ALPHA) * (0.40 + (0.60 * activation))
        redistribution = max(min(competitor_score * alpha, competitor_score), 0.0)
        current[competitor_index] = np.clip(competitor_score - redistribution, 0.0, 1.0)
        current[self.glass_index] = np.clip(glass_score + redistribution + float(GLASS_IMPULSE_GLASS_BOOST), 0.0, 1.0)

        if GLASS_EVENT_CONFIRM_ENABLED:
            burst_memory = max([burst_score, *self.recent_burst], default=burst_score)
            decay_memory = max([decay_score, *self.recent_decay], default=decay_score)
            updated_glass = float(current[self.glass_index])
            updated_competitor = float(current[competitor_index])
            updated_margin = updated_competitor - updated_glass
            if (
                burst_memory >= float(GLASS_EVENT_MIN_BURST)
                and decay_memory >= float(GLASS_EVENT_DECAY_MIN)
                and updated_margin > -0.02
                and updated_margin <= float(GLASS_IMPULSE_TRIGGER_MARGIN)
            ):
                activation = np.clip(
                    (burst_memory - float(GLASS_EVENT_MIN_BURST)) / max(1.0 - float(GLASS_EVENT_MIN_BURST), EPS),
                    0.0,
                    1.0,
                )
                alpha = float(GLASS_EVENT_ALPHA) * (0.35 + (0.65 * activation))
                redistribution = max(min(updated_competitor * alpha, updated_competitor), 0.0)
                current[competitor_index] = np.clip(updated_competitor - redistribution, 0.0, 1.0)
                current[self.glass_index] = np.clip(
                    updated_glass + redistribution + float(GLASS_EVENT_BOOST),
                    0.0,
                    1.0,
                )

        self.recent_burst.append(float(burst_score))
        self.recent_decay.append(float(decay_score))
        return current


class YamnetClassifier:
    """Semantic classifier interface with pluggable backbone implementations."""

    def __init__(self) -> None:
        if not CLASSIFIER_ENABLED:
            raise RuntimeError(
                "Classifier disabled: no usable semantic backbone artifact was found. "
                f"Expected spectral={SPECTRAL_MODEL_PATH} or PANNs checkpoint via PANNS_MODEL_PATH."
            )

        self.mode = CLASSIFIER_MODE
        self.target_sr = 16000
        self.backends: list[tuple[str, _SemanticBackbone, float]] = []
        self.stereo_beme_backbone: _SemanticBackbone | None = None
        self.pairwise_refiner: _PairwiseSemanticRefiner | None = None
        self.rail_speech_specialist: _RailSpeechSpecialist | None = None
        self.glass_impulse_specialist: _GlassImpulseSpecialist | None = None
        self._delegate = None
        semantic_savedmodel_exists = os.path.exists(os.path.join(YAMNET_MODEL_DIR, "saved_model.pb")) or os.path.exists(
            os.path.join(YAMNET_MODEL_DIR, "saved_model.pbtxt")
        )
        learned_head_exists = os.path.exists(LEARNED_HEAD_MODEL_PATH)
        beme_head_exists = os.path.exists(BEME_HEAD_MODEL_PATH)
        beme_stereo_head_exists = os.path.exists(BEME_STEREO_HEAD_MODEL_PATH)

        semantic_labels: list[str] = []
        if os.path.exists(SEMANTIC_LABELS_JSON):
            with open(SEMANTIC_LABELS_JSON, "r", encoding="utf-8") as handle:
                semantic_labels = list(json.load(handle))
        elif USE_SEMANTIC_YAMNET or CLASSIFIER_MODE == "hybrid":
            semantic_labels = list(YAMNET_LABEL_MAP.keys())

        if CLASSIFIER_MODE == "panns":
            from panns_pipeline.classifier import PannsClassifier

            print("Loading classifier in 'panns' mode: separate PANNs semantic backbone...")
            self._delegate = PannsClassifier(labels_path=SEMANTIC_LABELS_JSON)
            self.class_names = list(self._delegate.class_names)
            self.target_sr = int(self._delegate.target_sr)
            if CONFUSION_PAIR_REFINEMENT_ENABLED and os.path.exists(PAIRWISE_REFINER_MODEL_PATH):
                self.pairwise_refiner = _PairwiseSemanticRefiner(PAIRWISE_REFINER_MODEL_PATH, self.class_names)
            if RAIL_SPEECH_REFINEMENT_ENABLED:
                self.rail_speech_specialist = _RailSpeechSpecialist(self.class_names)
            if GLASS_IMPULSE_REFINEMENT_ENABLED:
                self.glass_impulse_specialist = _GlassImpulseSpecialist(self.class_names)
            print("[OK] Semantic classifier ready with backbones: panns")
            return

        if CLASSIFIER_MODE in {"semantic", "hybrid"} and TF_AVAILABLE and semantic_labels and semantic_savedmodel_exists:
            print(f"Loading classifier in '{CLASSIFIER_MODE}' mode: semantic YAMNet SavedModel...")
            self.backends.append(("semantic", _SemanticYamnetBackbone(semantic_labels), HYBRID_SEMANTIC_WEIGHT))

        if CLASSIFIER_MODE in {"learned", "hybrid"} and TF_AVAILABLE and learned_head_exists:
            print(f"Loading classifier in '{CLASSIFIER_MODE}' mode: YAMNet learnable semantic head...")
            self.backends.append(("learned", _LearnedSemanticHeadBackbone(LEARNED_HEAD_MODEL_PATH), HYBRID_LEARNED_WEIGHT))

        if CLASSIFIER_MODE == "beme" and TF_AVAILABLE and beme_head_exists:
            print("Loading classifier in 'beme' mode: YAMNet embedding head with AutoBEME...")
            self.backends.append(("beme", _BemeSemanticHeadBackbone(BEME_HEAD_MODEL_PATH), 1.0))
            if beme_stereo_head_exists:
                print("Loading stereo-aware BEME head for stereo windows...")
                self.stereo_beme_backbone = _BemeSemanticHeadBackbone(BEME_STEREO_HEAD_MODEL_PATH)

        if CLASSIFIER_MODE in {"full", "hybrid"} and TF_AVAILABLE:
            print(f"Loading classifier in '{CLASSIFIER_MODE}' mode: Full YAMNet semantic projection...")
            self.backends.append(("full", _FullProjectedYamnetBackbone(), HYBRID_FULL_WEIGHT))

        spectral_expected_labels = list(YAMNET_LABEL_MAP.keys()) if CLASSIFIER_MODE == "hybrid" else None
        if CLASSIFIER_MODE in {"spectral", "hybrid"} and os.path.exists(SPECTRAL_MODEL_PATH):
            print(f"Loading classifier in '{CLASSIFIER_MODE}' mode: spectral backbone...")
            self.backends.append(("spectral", _SpectralBackbone(SPECTRAL_MODEL_PATH, expected_labels=spectral_expected_labels), HYBRID_SPECTRAL_WEIGHT))

        if not self.backends:
            raise RuntimeError(
                "No classifier backbone could be initialized for "
                f"CLASSIFIER_MODE={CLASSIFIER_MODE}. "
                f"TF_AVAILABLE={TF_AVAILABLE}, learned_head_artifact={learned_head_exists}, "
                f"beme_head_artifact={beme_head_exists}, "
                f"spectral_artifact={os.path.exists(SPECTRAL_MODEL_PATH)}"
            )

        primary_name, primary_backbone, _primary_weight = self.backends[0]
        self.class_names = list(primary_backbone.class_names)
        for backend_name, backend, _weight in self.backends[1:]:
            if list(backend.class_names) != self.class_names:
                raise RuntimeError(
                    f"Backbone label mismatch between '{primary_name}' and '{backend_name}'. "
                    "All backbones must emit the same semantic label order."
                )
        if CONFUSION_PAIR_REFINEMENT_ENABLED and os.path.exists(PAIRWISE_REFINER_MODEL_PATH):
            self.pairwise_refiner = _PairwiseSemanticRefiner(PAIRWISE_REFINER_MODEL_PATH, self.class_names)
        if RAIL_SPEECH_REFINEMENT_ENABLED:
            self.rail_speech_specialist = _RailSpeechSpecialist(self.class_names)
        if GLASS_IMPULSE_REFINEMENT_ENABLED:
            self.glass_impulse_specialist = _GlassImpulseSpecialist(self.class_names)
        print(f"[OK] Semantic classifier ready with backbones: {', '.join(name for name, _b, _w in self.backends)}")

    def predict_all(
        self,
        mono_audio: np.ndarray,
        input_sr: int,
        frames_audio: np.ndarray | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Return per-class scores for every label."""
        if self._delegate is not None:
            labels, probabilities = self._delegate.predict_all(mono_audio, input_sr)
            if self.pairwise_refiner is not None or self.rail_speech_specialist is not None or self.glass_impulse_specialist is not None:
                mono_audio_16k = resample_linear(np.asarray(mono_audio, dtype=np.float32).reshape(-1), input_sr, 16000)
            else:
                mono_audio_16k = None
            if self.pairwise_refiner is not None and mono_audio_16k is not None:
                probabilities = self.pairwise_refiner.refine(mono_audio_16k, probabilities, labels)
            if self.rail_speech_specialist is not None and mono_audio_16k is not None:
                probabilities = self.rail_speech_specialist.refine(mono_audio_16k, probabilities)
            if self.glass_impulse_specialist is not None and mono_audio_16k is not None:
                probabilities = self.glass_impulse_specialist.refine(mono_audio_16k, probabilities)
            return labels, probabilities

        if input_sr != self.target_sr:
            mono_audio = resample_linear(mono_audio, input_sr, self.target_sr)
            frames_audio = (
                resample_frame_major_audio(frames_audio, input_sr, self.target_sr)
                if frames_audio is not None
                else None
            )
        elif frames_audio is not None:
            frames_audio = ensure_frame_major(frames_audio)

        mono_audio = np.asarray(mono_audio, dtype=np.float32)
        probabilities = self._predict_scores_16k(mono_audio, frames_audio)
        probabilities = self._apply_temporal_frontend(mono_audio, probabilities)
        probabilities = self._apply_multires_frontend(mono_audio, probabilities)
        if self.pairwise_refiner is not None:
            probabilities = self.pairwise_refiner.refine(mono_audio, probabilities, self.class_names)
        if self.rail_speech_specialist is not None:
            probabilities = self.rail_speech_specialist.refine(mono_audio, probabilities)
        if self.glass_impulse_specialist is not None:
            probabilities = self.glass_impulse_specialist.refine(mono_audio, probabilities)

        if len(self.class_names) != int(np.asarray(probabilities).shape[-1]):
            raise RuntimeError(
                "Classifier output contract mismatch: "
                f"mode={self.mode}, labels={len(self.class_names)}, "
                f"probs={int(np.asarray(probabilities).shape[-1])}. "
                "Check CLASSIFIER_MODE, SEMANTIC_LABELS_JSON, and SavedModel export."
            )

        # L1 normalisation removed: the model (LSE pooling + no REST bucket) now
        # produces per-class scores that should NOT be forced onto a simplex.
        # Forcing them would artificially suppress minority classes whenever the
        # dominant class score is high.
        return self.class_names, probabilities

    def reset_temporal_state(self) -> None:
        """Clear stateful temporal refiners between unrelated clips."""
        for component in (self.rail_speech_specialist, self.glass_impulse_specialist):
            if component is not None and hasattr(component, "reset"):
                component.reset()

    def _compute_stereo_gate(
        self,
        mono_probabilities: np.ndarray,
        stereo_delta: np.ndarray,
        stereo_frames: np.ndarray,
    ) -> StereoGateDecision:
        mono_scores = np.asarray(mono_probabilities, dtype=np.float32).reshape(-1)
        delta_scores = np.asarray(stereo_delta, dtype=np.float32).reshape(-1)
        label_count = int(mono_scores.shape[0])
        zero_mask = np.zeros(label_count, dtype=np.float32)
        if label_count <= 0:
            return StereoGateDecision(0.0, 0.0, 0.0, 0.0, zero_mask)

        sorted_scores = np.sort(mono_scores)[::-1]
        top1 = float(sorted_scores[0]) if sorted_scores.size else 0.0
        top2 = float(sorted_scores[1]) if sorted_scores.size > 1 else 0.0
        margin = max(0.0, top1 - top2)
        entropy_norm = _normalized_score_entropy(mono_scores)

        confidence_score = float(np.clip((float(BEME_STEREO_GATE_TOP1_MAX) - top1) / max(float(BEME_STEREO_GATE_TOP1_MAX), EPS), 0.0, 1.0))
        margin_score = float(np.clip((float(BEME_STEREO_GATE_MARGIN_MAX) - margin) / max(float(BEME_STEREO_GATE_MARGIN_MAX), EPS), 0.0, 1.0))
        entropy_score = float(
            np.clip(
                (entropy_norm - float(BEME_STEREO_GATE_ENTROPY_MIN)) / max(1.0 - float(BEME_STEREO_GATE_ENTROPY_MIN), EPS),
                0.0,
                1.0,
            )
        )
        uncertainty_score = max(confidence_score, margin_score, entropy_score)

        spatial = summarize_spatial_audio(stereo_frames, float(self.target_sr))
        decorrelation = float(np.clip(1.0 - max(spatial.correlation, 0.0), 0.0, 1.0))
        direction_score = float(
            np.clip(
                (float(spatial.direction_confidence) - float(BEME_STEREO_GATE_MIN_DIRECTION_CONF))
                / max(1.0 - float(BEME_STEREO_GATE_MIN_DIRECTION_CONF), EPS),
                0.0,
                1.0,
            )
        )
        decorrelation_score = float(
            np.clip(
                (decorrelation - float(BEME_STEREO_GATE_MIN_DECORRELATION))
                / max(1.0 - float(BEME_STEREO_GATE_MIN_DECORRELATION), EPS),
                0.0,
                1.0,
            )
        )
        ild_score = float(
            np.clip(
                (abs(float(spatial.ild_db)) - float(BEME_STEREO_GATE_MIN_ILD_DB))
                / max(6.0 - float(BEME_STEREO_GATE_MIN_ILD_DB), EPS),
                0.0,
                1.0,
            )
        )
        spatial_score = max(direction_score, decorrelation_score, ild_score)

        target_mask = np.zeros(label_count, dtype=np.float32)
        target_score = 0.0
        if BEME_STEREO_GATE_TARGET_LABELS:
            topk = min(label_count, int(BEME_STEREO_GATE_TOPK))
            topk_indices = set(np.argsort(mono_scores)[::-1][:topk].tolist())
            for index, label in enumerate(self.class_names):
                if label not in BEME_STEREO_GATE_TARGET_LABELS:
                    continue
                base_score = float(mono_scores[index])
                uplift = float(max(delta_scores[index], 0.0))
                is_topk = index in topk_indices
                active = (
                    is_topk
                    or base_score >= float(BEME_STEREO_GATE_MIN_TARGET_PROB)
                    or uplift >= float(BEME_STEREO_GATE_MIN_STEREO_DELTA)
                )
                if not active:
                    continue
                target_mask[index] = 1.0
                probability_score = float(
                    np.clip(
                        (base_score - float(BEME_STEREO_GATE_MIN_TARGET_PROB))
                        / max(1.0 - float(BEME_STEREO_GATE_MIN_TARGET_PROB), EPS),
                        0.0,
                        1.0,
                    )
                )
                uplift_score = float(np.clip(uplift / max(0.12, float(BEME_STEREO_GATE_MIN_STEREO_DELTA)), 0.0, 1.0))
                topk_score = 1.0 if is_topk else 0.0
                target_score = max(target_score, probability_score, uplift_score, topk_score)
        else:
            target_mask[:] = 1.0
            target_score = 1.0 if np.any(delta_scores > 0.0) else 0.0

        if target_score <= 0.0:
            return StereoGateDecision(0.0, uncertainty_score, target_score, spatial_score, zero_mask)

        gate_score = float(np.clip((0.45 * uncertainty_score) + (0.35 * target_score) + (0.20 * spatial_score), 0.0, 1.0))
        if uncertainty_score <= 0.0 or gate_score < float(BEME_STEREO_GATE_MIN_SCORE):
            return StereoGateDecision(gate_score, uncertainty_score, target_score, spatial_score, zero_mask)
        if not bool(BEME_STEREO_GATE_TARGET_ONLY):
            target_mask[:] = 1.0
        return StereoGateDecision(gate_score, uncertainty_score, target_score, spatial_score, target_mask.astype(np.float32, copy=False))

    def _predict_scores_16k(self, mono_audio_16k: np.ndarray, frames_audio_16k: np.ndarray | None = None) -> np.ndarray:
        if self.stereo_beme_backbone is not None and frames_audio_16k is not None:
            stereo_frames = ensure_frame_major(frames_audio_16k)
            if stereo_frames.shape[1] >= 2:
                mono_probabilities = self.backends[0][1].predict(mono_audio_16k, frames_audio_16k)
                stereo_probabilities = self.stereo_beme_backbone.predict(mono_audio_16k, stereo_frames)
                stereo_delta = np.asarray(stereo_probabilities, dtype=np.float32) - np.asarray(
                    mono_probabilities,
                    dtype=np.float32,
                )
                if BEME_STEREO_UPLIFT_ONLY:
                    stereo_delta = np.maximum(stereo_delta, 0.0)
                if BEME_STEREO_GATE_ENABLED:
                    gate = self._compute_stereo_gate(mono_probabilities, stereo_delta, stereo_frames)
                    if gate.score <= 0.0 or not np.any(gate.target_mask > 0.0):
                        return np.asarray(mono_probabilities, dtype=np.float32)
                    stereo_delta = stereo_delta * gate.target_mask
                    blend_alpha = float(BEME_STEREO_BLEND_ALPHA) * float(gate.score)
                else:
                    blend_alpha = float(BEME_STEREO_BLEND_ALPHA)
                fused = np.asarray(mono_probabilities, dtype=np.float32) + (blend_alpha * stereo_delta)
                return np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)
        if len(self.backends) == 1:
            return self.backends[0][1].predict(mono_audio_16k, frames_audio_16k)

        weighted_sum = np.zeros(len(self.class_names), dtype=np.float64)
        total_weight = 0.0
        for _name, backbone, weight in self.backends:
            weighted_sum += float(weight) * np.asarray(backbone.predict(mono_audio_16k, frames_audio_16k), dtype=np.float64)
            total_weight += float(weight)
        if total_weight <= 0.0:
            total_weight = float(len(self.backends))
        return np.clip(weighted_sum / total_weight, 0.0, 1.0).astype(np.float32, copy=False)

    def _apply_temporal_frontend(self, mono_audio_16k: np.ndarray, base_probabilities: np.ndarray) -> np.ndarray:
        if not TEMPORAL_FRONTEND_ENABLED or mono_audio_16k.size == 0:
            return base_probabilities

        label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        branch_stats: list[dict[str, np.ndarray | float]] = []
        for branch in TEMPORAL_FRONTEND_BRANCHES:
            window_samples = max(1, int(round(self.target_sr * float(branch["window_s"]))))
            hop_samples = max(1, int(round(self.target_sr * float(branch["hop_s"]))))
            spans = _iter_overlapping_spans(mono_audio_16k.shape[0], window_samples, hop_samples)
            if not spans:
                continue

            branch_scores = [
                np.asarray(self._predict_scores_16k(mono_audio_16k[start:end]), dtype=np.float64)
                for start, end in spans
                if end > start
            ]
            if not branch_scores:
                continue

            stacked = np.vstack(branch_scores)
            top_k = min(2, stacked.shape[0])
            strongest = np.partition(stacked, -top_k, axis=0)[-top_k:, :]
            branch_stats.append(
                {
                    "window_s": float(branch["window_s"]),
                    "peak": np.max(stacked, axis=0),
                    "support": np.mean(strongest, axis=0),
                }
            )

        if not branch_stats:
            return base_probabilities

        fused = np.asarray(base_probabilities, dtype=np.float64).copy()
        for label, family in TEMPORAL_FAMILY_BY_LABEL.items():
            idx = label_to_index.get(label)
            if idx is None:
                continue

            base_score = float(fused[idx])
            if family == "transient":
                cfg = TEMPORAL_TRANSIENT_CONFIG.get(label, {})
                peak_score = max(float(item["peak"][idx]) for item in branch_stats)
                support_score = max(float(item["support"][idx]) for item in branch_stats)
                support_weight = float(cfg.get("support_weight", 0.20))
                rescue_score = ((1.0 - support_weight) * peak_score) + (support_weight * support_score)
                if peak_score < float(cfg.get("min_peak", 0.0)):
                    continue
                if rescue_score < (base_score + float(cfg.get("margin", 0.0))):
                    continue
                alpha = float(cfg.get("alpha", 0.25))
                fused[idx] = base_score + alpha * (rescue_score - base_score)
                continue

            if family != "sustained":
                continue

            sustained_branches = [item for item in branch_stats if float(item["window_s"]) >= 0.50]
            if not sustained_branches:
                sustained_branches = branch_stats
            peak_score = max(float(item["peak"][idx]) for item in sustained_branches)
            support_score = max(float(item["support"][idx]) for item in sustained_branches)
            if peak_score > (support_score + float(TEMPORAL_SUSTAINED_PEAK_MARGIN)):
                pull_alpha = float(TEMPORAL_SUSTAINED_PULL_ALPHA)
                fused[idx] = ((1.0 - pull_alpha) * base_score) + (pull_alpha * support_score)
            elif support_score > base_score:
                boost_alpha = float(TEMPORAL_SUSTAINED_BOOST_ALPHA)
                fused[idx] = base_score + boost_alpha * (support_score - base_score)

        return np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)

    def _apply_multires_frontend(self, mono_audio_16k: np.ndarray, base_probabilities: np.ndarray) -> np.ndarray:
        if not MULTIRES_FRONTEND_ENABLED:
            return base_probabilities
        if mono_audio_16k.size == 0:
            return base_probabilities

        label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        transient_items = [(label, cfg) for label, cfg in MULTIRES_TRANSIENT_CONFIG.items() if label in label_to_index]
        if not transient_items:
            return base_probabilities

        branch_stats: list[dict[str, np.ndarray]] = []
        for branch in MULTIRES_BRANCHES:
            window_samples = max(1, int(round(self.target_sr * float(branch["window_s"]))))
            hop_samples = max(1, int(round(self.target_sr * float(branch["hop_s"]))))
            if mono_audio_16k.shape[0] <= window_samples:
                continue

            branch_scores = []
            for start, end in _iter_overlapping_spans(mono_audio_16k.shape[0], window_samples, hop_samples):
                subwindow = mono_audio_16k[start:end]
                branch_scores.append(np.asarray(self._predict_scores_16k(subwindow), dtype=np.float64))

            if not branch_scores:
                continue

            stacked = np.vstack(branch_scores)
            top_k = min(2, stacked.shape[0])
            strongest = np.partition(stacked, -top_k, axis=0)[-top_k:, :]
            branch_stats.append(
                {
                    "peak": np.max(stacked, axis=0),
                    "support": np.mean(strongest, axis=0),
                }
            )

        if not branch_stats:
            return base_probabilities

        fused = np.asarray(base_probabilities, dtype=np.float64).copy()
        onset_strength = _estimate_onset_strength(
            mono_audio_16k,
            self.target_sr,
            attack_s=float(MULTIRES_ONSET_ATTACK_S),
            release_s=float(MULTIRES_ONSET_RELEASE_S),
        )
        for label, cfg in transient_items:
            if MULTIRES_FAMILY_BY_LABEL.get(label) != "transient":
                continue
            idx = label_to_index[label]
            base_score = float(fused[idx])
            peak_score = max(float(item["peak"][idx]) for item in branch_stats)
            support_score = max(float(item["support"][idx]) for item in branch_stats)
            support_weight = float(cfg.get("support_weight", 0.33))
            rescue_score = ((1.0 - support_weight) * peak_score) + (support_weight * support_score)
            margin = float(cfg.get("margin", 0.0))
            min_peak = float(cfg.get("min_peak", 0.0))
            alpha = float(cfg.get("alpha", 0.5))
            alpha = min(0.85, alpha + (float(MULTIRES_ONSET_ALPHA_BOOST) * onset_strength))
            if peak_score < min_peak:
                continue
            if rescue_score < (base_score + margin):
                continue
            fused[idx] = base_score + alpha * (rescue_score - base_score)

        return fused.astype(np.float32, copy=False)

    def predict_top(self, mono_audio: np.ndarray, input_sr: int) -> tuple[str, float]:
        """Return the top label and its probability."""
        labels, probabilities = self.predict_all(mono_audio, input_sr)
        index = int(np.argmax(probabilities))
        if 0 <= index < len(labels):
            return labels[index], float(probabilities[index])
        return "Unknown", float(probabilities[index])


def _iter_overlapping_spans(total_samples: int, window_samples: int, hop_samples: int) -> list[tuple[int, int]]:
    if total_samples <= 0:
        return []
    if total_samples <= window_samples:
        return [(0, total_samples)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start + window_samples <= total_samples:
        spans.append((start, start + window_samples))
        start += hop_samples

    last_start = max(0, total_samples - window_samples)
    if not spans or spans[-1][0] != last_start:
        spans.append((last_start, total_samples))
    return spans


def _estimate_onset_strength(
    mono_audio_16k: np.ndarray,
    sample_rate: int,
    *,
    attack_s: float,
    release_s: float,
) -> float:
    if mono_audio_16k.size == 0 or sample_rate <= 0:
        return 0.0

    attack_samples = max(1, min(mono_audio_16k.size, int(round(sample_rate * attack_s))))
    release_samples = max(1, min(mono_audio_16k.size, int(round(sample_rate * release_s))))
    envelope = np.abs(mono_audio_16k.astype(np.float64, copy=False))
    attack_energy = float(np.mean(envelope[:attack_samples]))
    release_energy = float(np.mean(envelope[-release_samples:]))
    rms = float(np.sqrt(np.mean(np.square(mono_audio_16k.astype(np.float64, copy=False))) + EPS))
    crest = float(np.max(envelope) / max(rms, EPS))
    attack_advantage = max(0.0, attack_energy - release_energy) / max(attack_energy + release_energy + EPS, EPS)
    crest_bonus = np.clip((crest - 1.6) / 3.2, 0.0, 1.0)
    return float(np.clip((0.65 * attack_advantage) + (0.35 * crest_bonus), 0.0, 1.0))


def _estimate_voicing_profile(mono_audio_16k: np.ndarray, sample_rate: int) -> tuple[float, float]:
    mono = np.asarray(mono_audio_16k, dtype=np.float32).reshape(-1)
    if mono.size < max(64, sample_rate // 20):
        return 0.0, 0.0

    frame_length = max(256, int(round(sample_rate * 0.040)))
    hop_length = max(128, int(round(sample_rate * 0.020)))
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1

    _freqs, _times, stft = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=max(frame_length - hop_length, 0),
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float32, copy=False)
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        return 0.0, 0.0

    power = magnitude ** 2
    freq_axis = np.linspace(0.0, sample_rate * 0.5, magnitude.shape[0], dtype=np.float32)
    low_mask = (freq_axis >= 85.0) & (freq_axis <= 350.0)
    mid_mask = (freq_axis >= 350.0) & (freq_axis <= 1800.0)
    if not np.any(low_mask) or not np.any(mid_mask):
        return 0.0, 0.0

    low_energy = np.mean(power[low_mask, :], axis=0)
    mid_energy = np.mean(power[mid_mask, :], axis=0)
    voicing_ratio = float(np.mean(low_energy / (mid_energy + EPS)))

    dominant_bins = np.argmax(power, axis=0)
    dominant_freqs = freq_axis[dominant_bins]
    tonal_frames = (dominant_freqs >= 85.0) & (dominant_freqs <= 600.0)
    tonal_stability = float(np.mean(tonal_frames.astype(np.float32)))
    voiced = float(np.clip(voicing_ratio / 2.5, 0.0, 1.0))
    return voiced, tonal_stability


def _estimate_glass_impulse_score(mono_audio_16k: np.ndarray, sample_rate: int) -> float:
    mono = np.asarray(mono_audio_16k, dtype=np.float32).reshape(-1)
    if mono.size < max(128, sample_rate // 20):
        return 0.0

    onset = _estimate_onset_strength(mono, sample_rate, attack_s=0.04, release_s=0.20)
    frame_length = max(256, int(round(sample_rate * 0.025)))
    hop_length = max(128, int(round(sample_rate * 0.010)))
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1

    _freqs, _times, stft = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=max(frame_length - hop_length, 0),
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float32, copy=False)
    if magnitude.ndim != 2 or magnitude.shape[1] == 0:
        return onset

    power = magnitude ** 2
    freq_axis = np.linspace(0.0, sample_rate * 0.5, magnitude.shape[0], dtype=np.float32)
    high_mask = freq_axis >= 2500.0
    low_mask = freq_axis <= 1200.0
    if not np.any(high_mask) or not np.any(low_mask):
        return onset

    high_energy = np.mean(power[high_mask, :], axis=0)
    low_energy = np.mean(power[low_mask, :], axis=0)
    high_ratio = float(np.max(high_energy / (low_energy + EPS)))
    frame_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]) ** 2, axis=0))
    flux_peak = float(np.max(frame_flux) / max(np.mean(frame_flux) + EPS, EPS))
    crest = float(np.max(np.abs(mono)) / max(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + EPS), EPS))

    high_score = np.clip((high_ratio - 0.9) / 2.5, 0.0, 1.0)
    flux_score = np.clip((flux_peak - 1.4) / 3.5, 0.0, 1.0)
    crest_score = np.clip((crest - 1.8) / 4.0, 0.0, 1.0)
    return float(np.clip((0.40 * onset) + (0.25 * high_score) + (0.20 * flux_score) + (0.15 * crest_score), 0.0, 1.0))


def _estimate_glass_burst_profile(mono_audio_16k: np.ndarray, sample_rate: int) -> tuple[float, float]:
    mono = np.asarray(mono_audio_16k, dtype=np.float32).reshape(-1)
    if mono.size < max(256, sample_rate // 10):
        return 0.0, 0.0

    spans = _iter_overlapping_spans(mono.size, max(1, int(round(sample_rate * 0.12))), max(1, int(round(sample_rate * 0.06))))
    if not spans:
        return 0.0, 0.0

    burst_scores: list[float] = []
    for start, end in spans:
        burst_scores.append(_estimate_glass_impulse_score(mono[start:end], sample_rate))

    burst_peak = float(max(burst_scores)) if burst_scores else 0.0
    if len(burst_scores) >= 2:
        ordered = sorted(burst_scores, reverse=True)
        burst_peak = max(burst_peak, float(0.65 * ordered[0] + 0.35 * ordered[1]))

    head_len = max(1, min(mono.size, int(round(sample_rate * 0.08))))
    tail_len = max(1, min(mono.size, int(round(sample_rate * 0.18))))
    envelope = np.abs(mono.astype(np.float64, copy=False))
    head_energy = float(np.mean(envelope[:head_len]))
    tail_energy = float(np.mean(envelope[-tail_len:]))
    decay = max(0.0, head_energy - tail_energy) / max(head_energy + tail_energy + EPS, EPS)
    return float(np.clip(burst_peak, 0.0, 1.0)), float(np.clip(decay, 0.0, 1.0))
