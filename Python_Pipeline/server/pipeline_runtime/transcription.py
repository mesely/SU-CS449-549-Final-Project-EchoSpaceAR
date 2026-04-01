"""Whisper-based speech transcription helpers."""

from __future__ import annotations

import threading
import time

import numpy as np

from .config import (
    DECODE_HOP_S,
    PREROLL_S,
    STT_MIN_TEXT_LEN,
    STT_WINDOW_S,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)
from .utils import resample_linear


class WhisperTranscriber:
    """
    Stream audio into faster-whisper while keeping a short pre-roll buffer.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        target_sr: int = 16000,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, compute_type=compute_type)
        self.target_sr = target_sr
        self.language = WHISPER_LANGUAGE or None

        self._buf = np.zeros(0, dtype=np.float32)
        self._pre = np.zeros(0, dtype=np.float32)
        self._pre_cap = int(round(PREROLL_S * self.target_sr))
        self._lock = threading.Lock()

        self.session_active = False
        self._session_start_ts: float | None = None
        self._last_decode_ts = 0.0
        self.latest_text = ""
        self.latest_language = self.language or "auto"
        self.latest_language_probability = 0.0

    def feed_preroll(self, samples: np.ndarray, input_sr: float) -> None:
        """Keep the latest `PREROLL_S` seconds available for the next session."""
        if input_sr != self.target_sr:
            samples = resample_linear(samples, input_sr, self.target_sr)
        if samples.size == 0:
            return

        with self._lock:
            if self._pre.size == 0:
                self._pre = samples.copy()
            else:
                self._pre = np.concatenate([self._pre, samples])[-self._pre_cap :]

    def append_audio(self, samples: np.ndarray, input_sr: float) -> None:
        """Append audio to the active STT session."""
        if not self.session_active:
            return
        if input_sr != self.target_sr:
            samples = resample_linear(samples, input_sr, self.target_sr)
        if samples.size == 0:
            return

        with self._lock:
            self._buf = np.concatenate([self._buf, samples])

    def start_session(self, use_preroll: bool = True) -> None:
        """Begin a transcription session, optionally seeded with pre-roll audio."""
        with self._lock:
            self._buf = self._pre.copy() if use_preroll else np.zeros(0, dtype=np.float32)

        self.session_active = True
        self._session_start_ts = time.time()
        self._last_decode_ts = 0.0
        self.latest_text = ""
        self.latest_language = self.language or "auto"
        self.latest_language_probability = 0.0

    def stop_session(self, finalize: bool = True) -> tuple[str | None, float]:
        """Stop the session and optionally decode the buffered audio one last time."""
        self.session_active = False
        text, effective_window = (None, 0.0)

        if finalize:
            text, effective_window = self._decode_now(force=True)
            if text:
                self.latest_text = text

        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)

        self._session_start_ts = None
        return text, effective_window

    def session_duration_s(self) -> float:
        """Return the current session duration in seconds."""
        if not self.session_active or self._session_start_ts is None:
            return 0.0
        return time.time() - self._session_start_ts

    def maybe_decode(
        self,
        now_ts: float,
        window_s: float = STT_WINDOW_S,
        hop_s: float = DECODE_HOP_S,
    ) -> tuple[str | None, float]:
        """Decode only when the hop interval has elapsed."""
        if not self.session_active:
            return None, 0.0
        if now_ts - self._last_decode_ts < hop_s:
            return None, 0.0
        return self._decode_now(force=False, window_s=window_s)

    def _decode_now(self, force: bool = False, window_s: float = STT_WINDOW_S) -> tuple[str | None, float]:
        with self._lock:
            if self._buf.size == 0:
                return None, 0.0
            if force:
                audio = self._buf.copy()
            else:
                samples_to_keep = int(window_s * self.target_sr)
                audio = self._buf[-samples_to_keep:] if self._buf.size > samples_to_keep else self._buf.copy()

        effective_window = float(audio.size) / float(self.target_sr)
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=6,
            patience=0.2,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        self.latest_language = getattr(info, "language", None) or self.language or "unknown"
        self.latest_language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        self._last_decode_ts = time.time()
        return (text if len(text) >= STT_MIN_TEXT_LEN else None, effective_window)
