# audio_buffer.py (or inside your main file)

import threading
import numpy as np
from collections import deque

class AudioBuffer:
    """
    Thread-safe buffer for mono float32 samples.
    Unity pushes chunks here; your pipeline pulls
    continuous windows from here.
    """

    def __init__(self, samplerate_hz=16000, max_seconds=60):
        self.samplerate_hz = samplerate_hz
        self.max_samples = int(max_seconds * samplerate_hz)
        self.buffer = deque()  # list of small np arrays
        self.total_samples = 0
        self.lock = threading.Lock()

    def push_chunk(self, samples: np.ndarray):
        """
        samples: 1D float32 numpy array, mono [-1, 1]
        """
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        with self.lock:
            self.buffer.append(samples.astype(np.float32))
            self.total_samples += samples.shape[0]

            # Trim if too big
            while self.total_samples > self.max_samples and self.buffer:
                old = self.buffer.popleft()
                self.total_samples -= old.shape[0]

    def pop_window(self, window_samples: int, hop_samples: int):
        """
        Generator: yields windows of 'window_samples',
        advancing by hop_samples each time.
        You can call this in a loop from your YAMNet / STT workers.
        """
        # This is one possible pattern; you may already have your own
        # loop logic and just need a "get_latest" method instead.
        raise NotImplementedError("Implement according to your existing loop.")
