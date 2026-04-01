"""Stereo side-channel helpers for the mono baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    SPATIAL_CENTER_CORRELATION,
    SPATIAL_CENTER_ILD_DB,
    SPATIAL_DIRECTION_MIN_ILD_DB,
    SPATIAL_MAX_GCC_DELAY_S,
    SPATIAL_MAX_ILD_DB,
    STEREO_SIDE_CHANNEL_ENABLED,
)


EPS = 1e-9


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


def summarize_spatial_audio(samples: np.ndarray, sample_rate: float) -> SpatialSnapshot:
    """
    mono + stereo:
    - mono yol reduced YAMNet'i besler
    - stereo side-channel ILD/IPD/GCC-PHAT ile yonu sezdirir
    """
    frames = ensure_frame_major(samples)
    channels = int(frames.shape[1])

    if not STEREO_SIDE_CHANNEL_ENABLED or channels < 2:
        mono = downmix_to_mono(frames)
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + EPS))
        return SpatialSnapshot(
            mode="mono",
            channels=channels,
            direction="unknown",
            direction_confidence=0.0,
            ild_db=0.0,
            ipd_rad=0.0,
            gcc_delay_s=0.0,
            correlation=1.0,
            left_rms=mono_rms,
            right_rms=mono_rms,
        )

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
        mode="stereo",
        channels=channels,
        direction=direction,
        direction_confidence=direction_conf,
        ild_db=ild_db,
        ipd_rad=ipd_rad,
        gcc_delay_s=gcc_delay_s,
        correlation=correlation,
        left_rms=left_rms,
        right_rms=right_rms,
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
        center_conf = min(1.0, correlation)
        return "center", float(center_conf)

    if abs_ild < SPATIAL_DIRECTION_MIN_ILD_DB:
        return "unknown", float(abs_ild / max(SPATIAL_DIRECTION_MIN_ILD_DB, EPS))

    direction = "left" if ild_db > 0.0 else "right"
    ild_conf = min(1.0, abs_ild / max(SPATIAL_MAX_ILD_DB, EPS))
    decorrelation_bonus = float(np.clip(1.0 - max(correlation, 0.0), 0.0, 1.0))
    confidence = np.clip(0.65 * ild_conf + 0.35 * decorrelation_bonus, 0.0, 1.0)
    return direction, float(confidence)
