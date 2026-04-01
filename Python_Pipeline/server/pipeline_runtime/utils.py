"""Small reusable helpers shared across pipeline modules."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def resample_linear(samples: np.ndarray, input_sr: float, output_sr: int) -> np.ndarray:
    """Linearly resample a mono waveform to the target sample rate."""
    if samples.size == 0 or input_sr == output_sr:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / float(input_sr)
    source_times = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_count = int(round(duration * output_sr))
    target_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32, copy=False)


def unix_to_local_iso(timestamp_unix: float) -> str:
    """Format a Unix timestamp in local time with millisecond precision."""
    return datetime.fromtimestamp(timestamp_unix, tz=timezone.utc).astimezone().isoformat(
        timespec="milliseconds"
    )
