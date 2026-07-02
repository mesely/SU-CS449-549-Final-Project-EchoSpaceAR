"""Audio preprocessing utilities for PANNs-style models."""

from __future__ import annotations

import numpy as np
from scipy import signal


EPS = 1e-10


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert arbitrary mono/stereo arrays into a flat mono waveform."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        return samples.reshape(-1)
    if samples.ndim != 2:
        raise ValueError(f"Unsupported audio rank: {samples.ndim}")
    if samples.shape[0] <= 4 and samples.shape[1] > samples.shape[0]:
        samples = samples.T
    return np.mean(samples, axis=1).astype(np.float32, copy=False)


def resample_linear(samples: np.ndarray, input_sr: float, output_sr: int) -> np.ndarray:
    """Linearly resample a waveform to the target sample rate."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size == 0 or float(input_sr) == float(output_sr):
        return mono.astype(np.float32, copy=False)
    duration = mono.size / float(input_sr)
    source_times = np.linspace(0.0, duration, num=mono.size, endpoint=False)
    target_count = int(round(duration * output_sr))
    target_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_times, source_times, mono).astype(np.float32, copy=False)


def pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad short clips with zeros or trim long clips to a fixed length."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == target_samples:
        return mono
    if mono.size > target_samples:
        return mono[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: mono.size] = mono
    return padded


def iter_clip_spans(total_samples: int, clip_samples: int, hop_samples: int) -> list[tuple[int, int]]:
    """Yield overlapping fixed-length clip spans."""
    if total_samples <= clip_samples:
        return [(0, total_samples)]
    spans: list[tuple[int, int]] = []
    start = 0
    while start + clip_samples < total_samples:
        spans.append((start, start + clip_samples))
        start += hop_samples
    spans.append((max(0, total_samples - clip_samples), total_samples))
    return spans


def _hz_to_mel(value_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (value_hz / 700.0))


def _mel_to_hz(value_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (value_mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Build a triangular mel filterbank."""
    mel_min = float(_hz_to_mel(np.asarray([fmin], dtype=np.float64))[0])
    mel_max = float(_hz_to_mel(np.asarray([fmax], dtype=np.float64))[0])
    mel_points = np.linspace(mel_min, mel_max, num=n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(int)
    filters = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)
    for index in range(n_mels):
        left = max(0, bins[index])
        center = max(left + 1, bins[index + 1])
        right = max(center + 1, bins[index + 2])
        for bin_index in range(left, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, min(right, filters.shape[1])):
            filters[index, bin_index] = (right - bin_index) / max(right - center, 1)
    return filters


def extract_logmel(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Convert a mono waveform into a frame-major log-mel spectrogram."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return np.zeros((1, n_mels), dtype=np.float32)

    _, _, stft = signal.stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop_length),
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(stft).astype(np.float32, copy=False) ** 2
    mel_basis = build_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_spec = np.maximum(mel_basis @ magnitude, EPS)
    return np.log(mel_spec).T.astype(np.float32, copy=False)
