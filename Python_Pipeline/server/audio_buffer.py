"""Simple thread-safe audio buffering primitives."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class AudioBuffer:
    """Store recent mono float32 chunks for downstream consumers."""

    def __init__(self, samplerate_hz: int = 16000, max_seconds: int = 60) -> None:
        self.samplerate_hz = samplerate_hz
        self.max_samples = int(max_seconds * samplerate_hz)
        self.buffer: deque[np.ndarray] = deque()
        self.total_samples = 0
        self.lock = threading.Lock()

    def push_chunk(self, samples: np.ndarray) -> None:
        """Append a chunk and trim old samples when the ring grows too large."""
        if samples.ndim != 1:
            samples = samples.reshape(-1)

        with self.lock:
            chunk = samples.astype(np.float32)
            self.buffer.append(chunk)
            self.total_samples += chunk.shape[0]

            while self.total_samples > self.max_samples and self.buffer:
                old_chunk = self.buffer.popleft()
                self.total_samples -= old_chunk.shape[0]

    def pop_window(self, window_samples: int, hop_samples: int):
        """
        Placeholder for a future consumer-facing window API.
        """
        del window_samples, hop_samples
        raise NotImplementedError("Implement window extraction to match your pipeline loop.")
