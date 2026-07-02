"""End-to-end live pipeline entrypoint, bridge, device selection, and UI."""

from __future__ import annotations

import csv
import os
import queue
import base64
import json
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from textwrap import shorten
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

from .b_config import (
    CLASSIFIER_ENABLED,
    CLASSIFY_HOP_S,
    DATA_COLLECTION_MODE,
    DECISION_CSV_PATH,
    DECODE_HOP_S,
    GLOBAL_LLM_COOLDOWN_S,
    IMPORTANT_SOUND_LABELS,
    IMPORTANT_SOUND_THRESH,
    LLM_ENABLED,
    LOG_DIR,
    MAX_SESSION_S,
    PLOT_UPDATE_INTERVAL_S,
    PRINT_ALL_TO_CONSOLE,
    REQUIRE_TOP_IS_SPEECH,
    SOUND_LLM_COOLDOWN_S,
    SOUND_LLM_WINDOW_S,
    SPEECH_ENERGY_MARGIN_DB,
    SPEECH_GATE_ENABLED,
    SPEECH_LABEL,
    SPEECH_MIN_OFF_S,
    SPEECH_MIN_ON_S,
    SPEECH_OFF_THRESH,
    SPEECH_ON_THRESH,
    STT_CSV_PATH,
    STT_ENABLED,
    STT_WINDOW_S,
    TOPK_OVERLAY,
    WIDE_CSV_PATH,
    unix_to_local_iso,
)
from .c_audio_core import YamnetClassifier, downmix_to_mono, ensure_frame_major, summarize_spatial_audio
from .d_decision import DecisionSnapshot, PriorityDecisionLayer
from .e_transcription import WhisperTranscriber
from .f_llm import GeminiEventAnalyzer


EVENTS: dict[str, list[dict]] = {}
EVENTS_LOCK = threading.Lock()
UNITY_AUDIO_SINK = None
UNITY_AUDIO_LOCK = threading.Lock()


def register_unity_audio_sink(func) -> None:
    """Register the callable that should receive Unity audio chunks."""
    global UNITY_AUDIO_SINK
    with UNITY_AUDIO_LOCK:
        UNITY_AUDIO_SINK = func
    print(f"[HTTP BRIDGE] Registered Unity audio sink: {func}")


def _ensure_session(session_id: str) -> None:
    with EVENTS_LOCK:
        EVENTS.setdefault(session_id, [])


def _append_event(session_id: str, event: dict) -> None:
    _ensure_session(session_id)
    with EVENTS_LOCK:
        EVENTS[session_id].append(event)


def push_yamnet_event(
    session_id: str,
    timestamp_unix: float,
    window_start: float,
    window_end: float,
    top5,
    dominant_label: str,
    dominant_prob: float,
    spl_dbfs: float,
    *,
    decision: dict | None = None,
    spatial: dict | None = None,
    raw_top5: list[dict] | None = None,
) -> None:
    """Store one classifier window for later polling by Unity."""
    yamnet_payload = {
        "window_start_unix": window_start,
        "window_end_unix": window_end,
        "top5": top5,
        "dominant_label": dominant_label,
        "dominant_prob": dominant_prob,
        "spl_dbfs": spl_dbfs,
    }
    if decision is not None:
        yamnet_payload["decision"] = decision
    if spatial is not None:
        yamnet_payload["spatial"] = spatial
    if raw_top5 is not None:
        yamnet_payload["raw_top5"] = raw_top5
    event = {
        "kind": "yamnet",
        "timestamp_unix": timestamp_unix,
        "session_id": session_id,
        "yamnet": yamnet_payload,
    }
    _append_event(session_id, event)


def push_stt_event(
    session_id: str,
    segment_id: str,
    start_unix: float,
    end_unix: float,
    eff_window_s: float,
    samplerate_hz: int,
    text: str,
    language: str,
    confidence: float,
) -> None:
    """Store one STT segment for later polling by Unity."""
    event = {
        "kind": "stt",
        "timestamp_unix": end_unix,
        "session_id": session_id,
        "stt": {
            "segment_id": segment_id,
            "start_unix": start_unix,
            "end_unix": end_unix,
            "eff_window_s": eff_window_s,
            "samplerate_hz": samplerate_hz,
            "text": text,
            "language": language,
            "confidence": confidence,
        },
    }
    _append_event(session_id, event)


def push_llm_event(
    session_id: str,
    window_start: float,
    window_end: float,
    brief_summary: str,
    user_message: str,
    important_events: list,
) -> None:
    """Store one LLM summary window for later polling by Unity."""
    event = {
        "kind": "llm",
        "timestamp_unix": time.time(),
        "session_id": session_id,
        "llm": {
            "window_start_unix": window_start,
            "window_end_unix": window_end,
            "brief_summary": brief_summary,
            "user_message": user_message,
            "important_events": important_events,
        },
    }
    _append_event(session_id, event)


def list_input_devices() -> list[tuple[int, str, int, int]]:
    """Return `(index, name, max_input_channels, default_samplerate)` rows."""
    devices = sd.query_devices()
    rows: list[tuple[int, str, int, int]] = []
    for index, device in enumerate(devices):
        rows.append(
            (
                index,
                device.get("name", ""),
                device.get("max_input_channels", 0),
                int(device.get("default_samplerate", 0) or 0),
            )
        )
    return rows


def print_input_devices(chosen_index: int | None = None) -> None:
    """Pretty-print available inputs and highlight the selected one."""
    print("\nAvailable audio devices:")
    print(f"{'Idx':>3}  {'InCh':>4}  {'DefSR':>6}  Name")
    for index, name, input_channels, sample_rate in list_input_devices():
        marker = "*" if chosen_index == index else " "
        print(f"{index:>3}{marker}  {input_channels:>4}  {sample_rate:>6}  {name}")
    print("('*' marks the selected input device)\n")


def pick_input_device() -> tuple[int, str]:
    """Choose an input device using the original fallback order."""
    devices = sd.query_devices()
    candidate = None
    try:
        default_input = sd.default.device[0]
    except Exception:
        default_input = None
    if isinstance(default_input, int) and 0 <= default_input < len(devices):
        if devices[default_input].get("max_input_channels", 0) > 0:
            candidate = default_input
    if candidate is None:
        keywords = ("microphone", "mic", "built-in", "external", "usb")
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) <= 0:
                continue
            name = (device.get("name") or "").lower()
            if any(keyword in name for keyword in keywords):
                candidate = index
                break
    if candidate is None:
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                candidate = index
                break
    if candidate is None:
        raise RuntimeError("No input device with capture channels found.")
    return candidate, devices[candidate].get("name", "")


def pick_sample_rate(device_index: int) -> int | None:
    """Choose the device default, or fall back to 48000/44100 if needed."""
    info = sd.query_devices(device_index)
    sample_rate = float(info.get("default_samplerate", 0) or 0)
    if sample_rate > 0:
        return int(sample_rate)
    for candidate in (48000, 44100):
        try:
            sd.check_input_settings(device=device_index, samplerate=candidate, channels=1)
            return candidate
        except Exception:
            continue
    return None


class PipelineHttpHandler(BaseHTTPRequestHandler):
    """Serve Unity requests and expose queued pipeline events."""

    def send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def do_POST(self) -> None:
        if self.path == "/client_hello":
            data = self.read_json_body()
            session_id = data.get("session_id") or ""
            _ensure_session(session_id)
            self.send_json({"status": "ok", "message": "hello_received"})
            return
        if self.path == "/audio_chunk":
            data = self.read_json_body()
            session_id = data.get("session_id") or ""
            sequence = data.get("seq")
            samplerate_hz = int(data.get("samplerate_hz") or 0)
            channels = max(1, int(data.get("channels") or 1))
            frame_count = int(data.get("frame_count") or 0)
            try:
                raw_bytes = base64.b64decode(data["pcm_base64"])
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
                payload = samples.reshape(-1, channels) if channels > 1 and samples.size % channels == 0 else samples
                chunk_info = {
                    "session_id": session_id,
                    "seq": sequence,
                    "timestamp_unix": data.get("timestamp_unix"),
                    "samplerate_hz": samplerate_hz,
                    "channels": channels,
                    "sample_format": data.get("sample_format") or "float32",
                    "frame_count": frame_count,
                    "device_unix_time_start": data.get("device_unix_time_start"),
                    "device_unix_time_end": data.get("device_unix_time_end"),
                }
                with UNITY_AUDIO_LOCK:
                    sink = UNITY_AUDIO_SINK
                if sink is not None:
                    try:
                        sink(payload, chunk_info)
                    except TypeError:
                        sink(payload)
            except Exception as exc:
                print(f"[HTTP] Error decoding/forwarding audio_chunk: {exc}")
            self.send_json({"status": "received"})
            return
        self.send_json({"error": "Not found"}, status=404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/events":
            self.send_json({"error": "Not found"}, status=404)
            return
        query = parse_qs(parsed.query)
        session_id = query.get("session_id", [""])[0]
        raw_since = query.get("since_unix", ["0"])[0].replace(",", ".")
        try:
            since_unix = float(raw_since)
        except ValueError:
            since_unix = 0.0
        with EVENTS_LOCK:
            session_events = EVENTS.get(session_id, [])
            new_events = [event for event in session_events if event["timestamp_unix"] > since_unix]
        last_timestamp = new_events[-1]["timestamp_unix"] if new_events else since_unix
        self.send_json({"events": new_events, "last_timestamp_unix": last_timestamp})

    def log_message(self, format, *args) -> None:
        return


def start_http_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the HTTP bridge and block forever."""
    httpd = HTTPServer((host, port), PipelineHttpHandler)
    print(f"HTTP server at http://{host}:{port}")
    httpd.serve_forever()


class DummyStream:
    """Minimal stream object used when Unity feeds the audio."""

    def __init__(self) -> None:
        self.active = True

    def __enter__(self) -> "DummyStream":
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.active = False
        return False

    def abort(self) -> None:
        self.active = False


class RealTimeSPLVisualizer:
    """Coordinate live plotting, classification, STT, and optional LLM summaries."""

    def __init__(
        self,
        update_interval_s: float = PLOT_UPDATE_INTERVAL_S,
        history_s: float = 20.0,
        min_db: float = -120.0,
        max_db: float = 0.0,
        classify: bool = True,
        classify_window_s: float = 2.0,
        use_local_mic: bool = False,
        log_csv_path: str = os.path.join(LOG_DIR, "classification_log.csv"),
    ) -> None:
        self.update_interval_s = float(update_interval_s)
        self.history_s = float(history_s)
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.classify_window_s = float(classify_window_s)
        self.classify_hop_s = float(CLASSIFY_HOP_S)
        self.use_local_mic = use_local_mic
        self.log_csv_path = log_csv_path
        self.session_id = "default"

        self._gate_state = "IDLE"
        self._gate_last_change = time.time()
        self._speech_on_since: float | None = None
        self._speech_off_since: float | None = None
        self._last_sound_llm_ts = 0.0
        self._last_any_llm_ts = 0.0
        self._zero_blocks_seen = 0

        self.q_levels: queue.Queue[tuple[float, float]] = queue.Queue()
        self.q_audio: queue.Queue[dict] = queue.Queue()
        self.max_points = int(np.ceil(self.history_s / self.update_interval_s)) + 10
        self.times: deque[float] = deque(maxlen=self.max_points)
        self.levels: deque[float] = deque(maxlen=self.max_points)

        self.q_audio_stt: queue.Queue[np.ndarray] | None = None
        self._stt_lock = threading.Lock()
        self.latest_transcript = ""

        self._llm_queue: queue.Queue[dict] | None = None
        self._llm_events_buffer: deque[dict] = deque(maxlen=600)
        self._llm_buf_lock = threading.Lock()

        self.latest_label: str | None = None
        self.latest_conf: float | None = None
        self.latest_decision: DecisionSnapshot | None = None
        self._label_lock = threading.Lock()

        self.classifier = None
        self.stt = None
        self.llm_analyzer = None
        self.decision_layer = PriorityDecisionLayer()

        self._latest_samplerate = 48000.0
        self._latest_channels = 1
        self._latest_mode_label = "mono"

        self._configure_audio_device()
        self._build_plot()
        self.stream = self._create_stream()
        self.ani = FuncAnimation(
            self.fig,
            self.on_timer,
            interval=int(self.update_interval_s * 1000),
            blit=False,
        )

        signal.signal(signal.SIGINT, self._sigint_handler)

        self._initialize_classifier(classify)
        self._initialize_stt()
        self._initialize_llm()

    def set_session_id(self, session_id: str) -> None:
        """Update the session id shared by HTTP events and LLM output."""
        self.session_id = session_id or "default"
        if self.llm_analyzer is not None:
            self.llm_analyzer.session_id = self.session_id

    def _configure_audio_device(self) -> None:
        """Keep the original device-selection flow and console output."""
        self.device_index, self.device_name = pick_input_device()
        self.samplerate = pick_sample_rate(self.device_index)
        self.stream_samplerate = self.samplerate if self.samplerate else None
        self._latest_samplerate = float(self.stream_samplerate or 48000.0)
        self.blocksize = (
            None
            if self.stream_samplerate is None
            else max(1, int(self.stream_samplerate * self.update_interval_s))
        )

        print_input_devices(self.device_index)
        print(f"Selected input device #{self.device_index}: {self.device_name}")
        print(
            f"Sample rate: {self.stream_samplerate or 'default'} Hz | "
            f"Blocksize: {self.blocksize or 'default'}"
        )
        print(
            "If the plot is flat: ensure macOS Microphone permission for Python is enabled.\n"
            "System Settings -> Privacy & Security -> Microphone -> enable for your Python/Terminal app.\n"
        )

    def _build_plot(self) -> None:
        """Create the live matplotlib figure and text overlays."""
        self.fig, self.ax = plt.subplots(figsize=(11, 4))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(self.min_db, self.max_db)
        self.ax.set_xlim(-self.history_s, 0.0)
        title_sr = int(self._latest_samplerate)
        short_name = shorten(self.device_name, width=48)
        self.ax.set_title(
            f"Real-Time SPL (RMS dBFS) - {short_name} @ {title_sr} Hz ({self._latest_mode_label})"
        )
        self.ax.set_xlabel("Time (s) relative to now")
        self.ax.set_ylabel("SPL (dBFS)")
        self.ax.grid(True, linestyle="--", alpha=0.4)

        self.text_readout = self.ax.text(
            0.01,
            0.95,
            "- dBFS",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_label = self.ax.text(
            0.99,
            0.95,
            "Classifier: disabled",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_stt = self.ax.text(
            0.99,
            0.05,
            "STT: disabled",
            transform=self.ax.transAxes,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_llm = self.ax.text(
            0.5,
            0.02,
            "Env: -",
            transform=self.ax.transAxes,
            va="bottom",
            ha="center",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )

    def _create_stream(self):
        """Use `sounddevice` for local audio or a dummy stream for Unity input."""
        if not self.use_local_mic:
            return DummyStream()

        return sd.InputStream(
            device=self.device_index,
            channels=1,
            dtype="float32",
            samplerate=self.stream_samplerate,
            blocksize=self.blocksize,
            callback=self.audio_callback,
        )

    def _initialize_classifier(self, classify: bool) -> None:
        self.classifier_enabled = classify and CLASSIFIER_ENABLED
        if not self.classifier_enabled:
            self.text_label.set_text("Classifier: disabled")
            self.log_csv_path = None
            return

        try:
            self.classifier = YamnetClassifier()
            self.text_label.set_text("Classifier: loading...")
            self._classifier_thread = threading.Thread(
                target=self._classification_worker,
                daemon=True,
                name="ClassifierWorker",
            )
            self._classifier_thread.start()
        except Exception as exc:
            print(f"[Classifier] Disabled: {exc}")
            self.classifier_enabled = False
            self.log_csv_path = None
            self.text_label.set_text("Classifier: disabled")
            return

        if self.log_csv_path:
            self._ensure_csv_header()
        self._ensure_decision_csv_header()

    def _initialize_stt(self) -> None:
        self.stt_enabled = STT_ENABLED
        if not self.stt_enabled:
            self.text_stt.set_text("STT: disabled")
            return

        self.q_audio_stt = queue.Queue()
        try:
            self.stt = WhisperTranscriber()
            self.text_stt.set_text("STT: loading...")
            self._stt_thread = threading.Thread(
                target=self._stt_worker,
                daemon=True,
                name="STTWorker",
            )
            self._stt_thread.start()
            self._ensure_stt_csv_header()
        except Exception as exc:
            print(f"[STT] Disabled: {exc}")
            self.stt_enabled = False
            self.q_audio_stt = None
            self.text_stt.set_text("STT: disabled")

    def _initialize_llm(self) -> None:
        self.llm_enabled = LLM_ENABLED
        if not self.llm_enabled:
            return

        self._llm_queue = queue.Queue()
        try:
            self.llm_analyzer = GeminiEventAnalyzer()
            self.llm_analyzer.session_id = self.session_id
            self._llm_thread = threading.Thread(
                target=self._llm_worker,
                daemon=True,
                name="LLMWorker",
            )
            self._llm_thread.start()
            print("[LLM] GeminiEventAnalyzer initialized.")
        except Exception as exc:
            print(f"[LLM] disabled (init error): {exc}")
            self.llm_enabled = False
            self._llm_queue = None
            self.llm_analyzer = None

    def feed_unity_chunk(self, samples: np.ndarray, chunk_info: dict | None = None) -> None:
        """Route Unity audio through the same callback path as local microphone audio."""
        if samples is None:
            return

        chunk_info = chunk_info or {}
        try:
            channels_hint = int(chunk_info.get("channels") or 1)
            sample_rate = float(chunk_info.get("samplerate_hz") or self.stream_samplerate or 48000.0)
            frames = ensure_frame_major(samples, channels_hint=channels_hint)
            self._route_audio_block(frames, sample_rate)
        except Exception as exc:
            print(f"[UnityAudio] Failed to route Unity chunk: {exc}")

    def audio_callback(self, indata, frames, time_info, status) -> None:
        """Capture audio, update SPL, and fan out chunks to background workers."""
        del time_info
        if status:
            print(status, file=sys.stderr)
        if frames <= 0:
            return

        try:
            frame_block = ensure_frame_major(np.asarray(indata, dtype=np.float32), channels_hint=1)
            sample_rate = float(self.stream_samplerate or 48000.0)
            self._route_audio_block(frame_block, sample_rate)
        except Exception as exc:
            print(f"[AudioCallback] Failed to process audio block: {exc}")

    def _route_audio_block(self, frames: np.ndarray, sample_rate: float) -> None:
        """
        mono + stereo:
        - mono downmix klasifikasyon ve STT icin korunur
        - stereo side-channel spatial ozellik olarak saklanir
        """
        frame_block = ensure_frame_major(frames)
        mono = downmix_to_mono(frame_block)
        if mono.size == 0:
            return

        self._latest_samplerate = float(sample_rate)
        self._latest_channels = int(frame_block.shape[1])
        self._latest_mode_label = "mono + stereo side-channel" if self._latest_channels >= 2 else "mono"

        if np.allclose(mono, 0.0):
            self._zero_blocks_seen += 1
        else:
            self._zero_blocks_seen = 0

        rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-9)
        dbfs = 20.0 * np.log10(rms + 1e-9)

        try:
            self.q_levels.put_nowait((time.monotonic(), float(dbfs)))
        except queue.Full:
            pass

        payload = {
            "ts_unix": time.time(),
            "sample_rate": float(sample_rate),
            "channels": int(frame_block.shape[1]),
            "frames": frame_block.copy(),
            "mono": mono.copy(),
        }

        if self.classifier_enabled:
            try:
                self.q_audio.put_nowait(payload)
            except queue.Full:
                pass

        if self.stt_enabled and self.q_audio_stt is not None:
            try:
                self.q_audio_stt.put_nowait(mono.copy())
            except queue.Full:
                pass

    def _classification_worker(self) -> None:
        """Classify rolling audio windows and feed downstream workers."""
        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        channels_in = 1
        samples_target = max(1, int(round(sr_in * self.classify_window_s)))
        hop_samples = max(1, int(round(sr_in * self.classify_hop_s)))
        buffer_samples = 0
        samples_since_emit = 0
        block_buffer: deque[dict] = deque()

        with self._label_lock:
            self.latest_label = "Ready"
            self.latest_conf = None
        self.text_label.set_text("Classifier: ready")

        while True:
            try:
                block = self.q_audio.get(timeout=1.0)
            except queue.Empty:
                continue

            block_sr = float(block["sample_rate"])
            block_channels = int(block["channels"])
            if abs(block_sr - sr_in) > 1e-3 or block_channels != channels_in:
                sr_in = block_sr
                channels_in = block_channels
                samples_target = max(1, int(round(sr_in * self.classify_window_s)))
                hop_samples = max(1, int(round(sr_in * self.classify_hop_s)))
                buffer_samples = 0
                samples_since_emit = 0
                block_buffer.clear()

            block_buffer.append(block)
            buffer_samples += int(block["mono"].size)
            samples_since_emit += int(block["mono"].size)

            max_keep = max(samples_target * 4, hop_samples * 6)
            while buffer_samples > max_keep and len(block_buffer) > 1:
                old = block_buffer.popleft()
                buffer_samples -= int(old["mono"].size)

            if buffer_samples < samples_target or samples_since_emit < hop_samples:
                continue

            mono_window = self._tail_concat(block_buffer, "mono", samples_target)
            frames_window = self._tail_concat(block_buffer, "frames", samples_target)
            if mono_window.size < samples_target:
                continue

            samples_since_emit = 0
            self._process_classification_window(
                mono_window=mono_window,
                frames_window=frames_window,
                sr_in=sr_in,
                window_end_unix=float(block["ts_unix"]),
            )

    def _tail_concat(self, block_buffer: deque[dict], key: str, samples_target: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining = int(samples_target)
        for block in reversed(block_buffer):
            array = block[key]
            if array is None or array.size == 0:
                continue
            take = min(int(array.shape[0]), remaining)
            parts.append(array[-take:])
            remaining -= take
            if remaining <= 0:
                break

        if not parts:
            return np.zeros((0,), dtype=np.float32) if key == "mono" else np.zeros((0, 1), dtype=np.float32)

        return np.concatenate(list(reversed(parts)), axis=0)

    def _process_classification_window(
        self,
        mono_window: np.ndarray,
        frames_window: np.ndarray,
        sr_in: float,
        window_end_unix: float,
    ) -> None:
        """Run classification, decision smoothing, and publish events."""
        spl_dbfs = 20.0 * np.log10(float(np.sqrt(np.mean(mono_window.astype(np.float64) ** 2) + 1e-9)) + 1e-9)

        overlay_text = "Classification error"
        top5 = [{"label": "error", "prob": 1.0}]
        raw_top5 = top5
        dominant_label = "error"
        dominant_prob = 1.0
        decision_payload = None
        spatial_payload = None
        decision: DecisionSnapshot | None = None

        try:
            labels, probabilities = self.classifier.predict_all(mono_window, int(sr_in), frames_audio=frames_window)
            raw_top5 = self._topk(labels, probabilities, TOPK_OVERLAY)
            spatial = summarize_spatial_audio(frames_window, sr_in)
            decision = self.decision_layer.update(labels, probabilities, spl_dbfs, spatial, window_end_unix)

            top5 = decision.top5
            dominant_label = decision.label
            dominant_prob = decision.confidence
            overlay_text = self._build_overlay_text(decision)
            decision_payload = decision.to_dict()
            spatial_payload = spatial.to_dict()

            if DATA_COLLECTION_MODE:
                self._append_wide_csv_row(window_end_unix, sr_in, labels, probabilities)

            if PRINT_ALL_TO_CONSOLE:
                self._print_probabilities(labels, probabilities)

            self._record_llm_event(window_end_unix, top5)
            self._maybe_enqueue_sound_llm_window(window_end_unix, top5)

            if SPEECH_GATE_ENABLED and self.stt_enabled:
                top_is_speech = raw_top5[0]["label"] == SPEECH_LABEL if raw_top5 else False
                self._update_speech_gate(
                    speech_prob=decision.speech_prob,
                    top_is_speech=top_is_speech,
                    spl_dbfs=spl_dbfs,
                    sr_in=sr_in,
                    now_ts=window_end_unix,
                )

            if raw_top5:
                self._append_csv_row(
                    timestamp_unix=window_end_unix,
                    sr_hz=sr_in,
                    label=raw_top5[0]["label"],
                    conf=float(raw_top5[0]["prob"]),
                )
            self._append_decision_csv_row(window_end_unix, sr_in, decision)
        except Exception as exc:
            print(f"[Classifier] Inference error: {exc}")

        with self._label_lock:
            self.latest_label = overlay_text
            self.latest_conf = None
            self.latest_decision = decision

        self._push_yamnet_event(
            window_end=window_end_unix,
            top5=top5,
            dominant_label=dominant_label,
            dominant_prob=dominant_prob,
            spl_dbfs=spl_dbfs,
            decision_payload=decision_payload,
            spatial_payload=spatial_payload,
            raw_top5=raw_top5,
        )

    def _topk(self, labels: list[str], probabilities: np.ndarray, limit: int) -> list[dict]:
        order = np.argsort(probabilities)[::-1]
        return [
            {"label": labels[index], "prob": float(probabilities[index])}
            for index in order[: min(limit, len(order))]
        ]

    def _print_probabilities(self, labels: list[str], probabilities: np.ndarray) -> None:
        order = np.argsort(probabilities)[::-1]
        max_width = max(len(name) for name in labels)
        print("\n--- probabilities ---")
        for index in order:
            print(f"{labels[index]:<{max_width}}  {probabilities[index]:.4f}")

    def _build_overlay_text(self, decision: DecisionSnapshot) -> str:
        """
        Arastirma ruhu:
        - EMA ile p(c|z) yumusat
        - hysteresis ile flapping'i kes
        - OOD gate ile bilmediginde sus
        """
        head = f"{decision.label} | state={decision.state} | priority={decision.priority}"
        direction = ""
        if decision.direction not in {"unknown", ""}:
            direction = f" | dir={decision.direction}:{decision.direction_confidence:.2f}"
        stats = (
            f"p={decision.confidence:.2f} raw={decision.raw_confidence:.2f} "
            f"H={decision.entropy_norm:.2f} speech={decision.speech_prob:.2f}"
        )
        top_lines = [f"{item['label']} ({item['prob'] * 100:.0f}%)" for item in decision.top5[:3]]
        return "\n".join([head + direction, stats, *top_lines])

    def _record_llm_event(self, timestamp_unix: float, top5: list[dict]) -> None:
        """Keep a short rolling buffer of classifier results for LLM windows."""
        if not self.llm_enabled:
            return
        with self._llm_buf_lock:
            self._llm_events_buffer.append({"ts": timestamp_unix, "top5": top5})

    def _maybe_enqueue_sound_llm_window(self, now_ts: float, top5: list[dict]) -> None:
        """Trigger an LLM window when important sounds cross the threshold."""
        if not self.llm_enabled or self._llm_queue is None:
            return

        important_hits = [
            (item["label"], item["prob"])
            for item in top5
            if item["label"] in IMPORTANT_SOUND_LABELS and item["prob"] >= IMPORTANT_SOUND_THRESH
        ]
        if not important_hits:
            return
        if now_ts - self._last_sound_llm_ts < SOUND_LLM_COOLDOWN_S:
            return

        self._last_sound_llm_ts = now_ts
        start_ts = now_ts - SOUND_LLM_WINDOW_S
        with self._stt_lock:
            stt_context = (self.latest_transcript or "").strip()

        window = {
            "start_ts": start_ts,
            "end_ts": now_ts,
            "stt_text": stt_context,
            "yamnet_events": self._slice_llm_events(start_ts, now_ts),
        }

        try:
            self._llm_queue.put_nowait(window)
        except queue.Full:
            print("[LLM] queue full; dropping sound-trigger window.")

    def _slice_llm_events(self, start_ts: float, end_ts: float) -> list[dict]:
        """Collect buffered YAMNet events within a time window."""
        with self._llm_buf_lock:
            buffered_events = list(self._llm_events_buffer)

        sliced_events = []
        for event in buffered_events:
            event_ts = event.get("ts", 0.0)
            if start_ts <= event_ts <= end_ts:
                sliced_events.append(
                    {
                        "ts": event_ts,
                        "rel_t": event_ts - end_ts,
                        "top5": event.get("top5", []),
                    }
                )
        return sliced_events

    def _update_speech_gate(
        self,
        speech_prob: float,
        top_is_speech: bool,
        spl_dbfs: float,
        sr_in: float,
        now_ts: float,
    ) -> None:
        """Use smoothed speech probability + hysteresis to start and stop Whisper sessions."""
        energy_ok = spl_dbfs >= (self.decision_layer.noise_floor_dbfs + SPEECH_ENERGY_MARGIN_DB)
        cond_on = speech_prob >= SPEECH_ON_THRESH and energy_ok and (
            top_is_speech if REQUIRE_TOP_IS_SPEECH else True
        )
        cond_off = (speech_prob < SPEECH_OFF_THRESH) or (not energy_ok) or (
            REQUIRE_TOP_IS_SPEECH and not top_is_speech
        )

        if cond_on:
            if self._speech_on_since is None:
                self._speech_on_since = now_ts
            self._speech_off_since = None
        else:
            self._speech_on_since = None

        if cond_off:
            if self._speech_off_since is None:
                self._speech_off_since = now_ts
        else:
            self._speech_off_since = None

        if self._gate_state == "IDLE":
            if self._speech_on_since is not None and (now_ts - self._speech_on_since) >= SPEECH_MIN_ON_S:
                self._gate_state = "RECORDING"
                self._gate_last_change = now_ts
                self.stt.start_session()
                self.text_stt.set_text("STT: listening...")
            return

        if self._gate_state == "RECORDING":
            if self._speech_off_since is not None and (now_ts - self._speech_off_since) >= SPEECH_MIN_OFF_S:
                final_text, effective_window = self.stt.stop_session(finalize=True)
                if final_text:
                    self._stt_publish(final_text, sr_in, effective_window)
                self.text_stt.set_text("STT: idle")
                self._gate_state = "COOLDOWN"
                self._gate_last_change = now_ts
            return

        if self._gate_state == "COOLDOWN" and not cond_off:
            self._gate_state = "IDLE"
            self._gate_last_change = now_ts

    def _push_yamnet_event(
        self,
        window_end: float,
        top5: list[dict],
        dominant_label: str,
        dominant_prob: float,
        spl_dbfs: float,
        decision_payload: dict | None,
        spatial_payload: dict | None,
        raw_top5: list[dict],
    ) -> None:
        """Publish the current classifier window through the HTTP bridge."""
        try:
            push_yamnet_event(
                session_id=self.session_id,
                timestamp_unix=window_end,
                window_start=window_end - self.classify_window_s,
                window_end=window_end,
                top5=top5,
                dominant_label=dominant_label,
                dominant_prob=dominant_prob,
                spl_dbfs=float(spl_dbfs),
                decision=decision_payload,
                spatial=spatial_payload,
                raw_top5=raw_top5,
            )
        except Exception as exc:
            print(f"[Classifier] Failed to push YAMNet event: {exc}")

    def _stt_worker(self) -> None:
        """Maintain the Whisper pre-roll buffer and publish decoded text."""
        if not self.stt_enabled or self.q_audio_stt is None:
            return

        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        self.text_stt.set_text("STT: ready")

        while True:
            try:
                chunk = self.q_audio_stt.get(timeout=1.0)
            except queue.Empty:
                text, effective_window = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
                if text:
                    self._stt_publish(text, sr_in, effective_window)
                continue

            self.stt.feed_preroll(chunk, sr_in)
            if self.stt.session_active:
                self.stt.append_audio(chunk, sr_in)

            text, effective_window = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
            if text:
                self._stt_publish(text, sr_in, effective_window)

            if self.stt.session_active and self.stt.session_duration_s() >= MAX_SESSION_S:
                final_text, effective_window = self.stt.stop_session(finalize=True)
                if final_text:
                    self._stt_publish(final_text, sr_in, effective_window)

                if self._gate_state == "RECORDING":
                    self.stt.start_session(use_preroll=False)
                    self.text_stt.set_text("STT: listening...")
                else:
                    self.text_stt.set_text("STT: idle")

    def _llm_worker(self) -> None:
        """Summarize buffered sound windows in a background thread."""
        if not self.llm_enabled or self._llm_queue is None or self.llm_analyzer is None:
            return

        while True:
            window = self._llm_queue.get()
            try:
                result = self.llm_analyzer.analyze_window(window)
            except Exception as exc:
                print(f"[LLM] worker error: {exc}")
                continue

            summary = result.get("brief_summary") or result.get("user_message") or ""
            if summary:
                short_summary = summary if len(summary) <= 80 else (summary[:77] + "...")
                self.text_llm.set_text(f"Env: {short_summary}")

    def _stt_publish(self, text: str, sr_in: float, effective_window: float) -> None:
        """Update the UI, persist STT output, and trigger an LLM window."""
        if not text:
            return

        with self._stt_lock:
            self.latest_transcript = text

        language = "unknown"
        language_confidence = 0.0
        if self.stt is not None:
            language = getattr(self.stt, "latest_language", "unknown") or "unknown"
            language_confidence = float(getattr(self.stt, "latest_language_probability", 0.0) or 0.0)

        shown = text if len(text) <= 60 else f"{text[:57]}..."
        self.text_stt.set_text(f"STT[{language}]: {shown}")

        try:
            timestamp_unix = time.time()
            with open(STT_CSV_PATH, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        unix_to_local_iso(timestamp_unix),
                        f"{timestamp_unix:.3f}",
                        f"{effective_window:.3f}",
                        int(sr_in),
                        language,
                        f"{language_confidence:.6f}",
                        text,
                    ]
                )
        except Exception as exc:
            print(f"[CSV] STT write error: {exc}")
            timestamp_unix = time.time()

        try:
            push_stt_event(
                session_id=self.session_id,
                segment_id=f"stt-{int(timestamp_unix * 1000)}",
                start_unix=timestamp_unix - float(effective_window),
                end_unix=timestamp_unix,
                eff_window_s=float(effective_window),
                samplerate_hz=int(sr_in),
                text=text,
                language=language,
                confidence=language_confidence,
            )
        except Exception as exc:
            print(f"[STT] Failed to push STT event: {exc}")

        if not self.llm_enabled or self._llm_queue is None or effective_window <= 0.0 or not text.strip():
            return

        end_ts = time.time()
        now_ts = end_ts
        if now_ts - self._last_any_llm_ts < GLOBAL_LLM_COOLDOWN_S:
            return

        self._last_any_llm_ts = now_ts
        start_ts = end_ts - effective_window
        window = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "stt_text": text,
            "yamnet_events": self._slice_llm_events(start_ts, end_ts),
        }

        try:
            self._llm_queue.put_nowait(window)
        except queue.Full:
            print("[LLM] queue full; dropping this window.")

    def on_timer(self, _frame):
        """Drain SPL values and update the plot."""
        updated = False
        now = time.monotonic()

        try:
            while True:
                timestamp_monotonic, dbfs = self.q_levels.get_nowait()
                self.times.append(timestamp_monotonic)
                self.levels.append(float(np.clip(dbfs, self.min_db, self.max_db)))
                updated = True
        except queue.Empty:
            pass

        if not updated:
            return (self.line,)

        rel_times = np.array(self.times, dtype=np.float64) - now
        mask = rel_times >= -self.history_s
        rel_times = rel_times[mask]
        rel_levels = np.array(self.levels, dtype=np.float64)[mask]

        self.line.set_data(rel_times, rel_levels)
        self.ax.set_xlim(-self.history_s, 0.0)

        if rel_levels.size > 0:
            self.text_readout.set_text(f"{rel_levels[-1]:.1f} dBFS")

        if self._zero_blocks_seen >= 5:
            self.ax.set_title("Real-Time SPL (RMS dBFS) - No signal detected (check mic permission / device)")
        else:
            short_name = shorten(self.device_name, width=48)
            self.ax.set_title(
                f"Real-Time SPL (RMS dBFS) - {short_name} @ {int(self._latest_samplerate)} Hz "
                f"({self._latest_mode_label})"
            )

        if self.classifier_enabled:
            with self._label_lock:
                if self.latest_label is not None:
                    self.text_label.set_text(self.latest_label)
        else:
            self.text_label.set_text("Classifier: disabled")

        return (self.line,)

    def run(self) -> None:
        """Start the stream context and block on the matplotlib window."""
        try:
            with self.stream:
                plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            if getattr(self.stream, "active", False):
                self.stream.abort()

    def _sigint_handler(self, *_args) -> None:
        plt.close(self.fig)

    def _ensure_wide_csv_header(self, labels: list[str]) -> bool:
        """Create the full-probability CSV header on first write."""
        try:
            needs_header = (not os.path.exists(WIDE_CSV_PATH)) or os.path.getsize(WIDE_CSV_PATH) == 0
            if needs_header:
                with open(WIDE_CSV_PATH, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["iso_time", "unix_time", "window_s", "samplerate_hz", *labels])
                print(f"[CSV] Logging full probabilities to: {WIDE_CSV_PATH}")
        except Exception as exc:
            print(f"[CSV] Could not prepare wide CSV: {exc}")
            return False
        return True

    def _append_wide_csv_row(
        self,
        timestamp_unix: float,
        sr_hz: float,
        labels: list[str],
        probabilities: np.ndarray,
    ) -> None:
        """Append one full label-probability row to the wide CSV."""
        try:
            if not os.path.exists(WIDE_CSV_PATH) or os.path.getsize(WIDE_CSV_PATH) == 0:
                if not self._ensure_wide_csv_header(labels):
                    return
            row = [
                unix_to_local_iso(timestamp_unix),
                f"{timestamp_unix:.3f}",
                f"{self.classify_window_s:.3f}",
                int(sr_hz),
                *[f"{prob:.6f}" for prob in probabilities.tolist()],
            ]
            with open(WIDE_CSV_PATH, "a", newline="") as handle:
                csv.writer(handle).writerow(row)
        except Exception as exc:
            print(f"[CSV] wide write error: {exc}")

    def _ensure_csv_header(self) -> None:
        """Create the top-1 classification log on first write."""
        try:
            needs_header = (not os.path.exists(self.log_csv_path)) or os.path.getsize(self.log_csv_path) == 0
            if needs_header:
                with open(self.log_csv_path, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["iso_time", "unix_time", "window_s", "samplerate_hz", "label", "confidence"]
                    )
                print(f"[CSV] Logging classifications to: {self.log_csv_path}")
        except Exception as exc:
            print(f"[CSV] Could not prepare log file: {exc}")
            self.log_csv_path = None

    def _append_csv_row(self, timestamp_unix: float, sr_hz: float, label: str, conf: float) -> None:
        """Append one raw top-1 classification result."""
        if not self.log_csv_path:
            return
        try:
            with open(self.log_csv_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        unix_to_local_iso(timestamp_unix),
                        f"{timestamp_unix:.3f}",
                        f"{self.classify_window_s:.3f}",
                        int(sr_hz),
                        label,
                        f"{conf:.6f}",
                    ]
                )
        except Exception as exc:
            print(f"[CSV] write error: {exc}")

    def _ensure_decision_csv_header(self) -> None:
        """Create the stabilized decision log once the classifier is enabled."""
        try:
            needs_header = (not os.path.exists(DECISION_CSV_PATH)) or os.path.getsize(DECISION_CSV_PATH) == 0
            if needs_header:
                with open(DECISION_CSV_PATH, "a", newline="") as handle:
                    csv.writer(handle).writerow(
                        [
                            "iso_time",
                            "unix_time",
                            "window_s",
                            "samplerate_hz",
                            "label",
                            "confidence",
                            "raw_label",
                            "raw_confidence",
                            "state",
                            "hud_action",
                            "priority",
                            "entropy_norm",
                            "margin",
                            "spl_dbfs",
                            "noise_floor_dbfs",
                            "speech_prob",
                            "direction",
                            "direction_confidence",
                            "onset_unix",
                            "offset_unix",
                            "is_silence",
                            "is_ood",
                        ]
                    )
                print(f"[CSV] Logging stabilized decisions to: {DECISION_CSV_PATH}")
        except Exception as exc:
            print(f"[CSV] Could not prepare decision log: {exc}")

    def _append_decision_csv_row(self, timestamp_unix: float, sr_hz: float, decision: DecisionSnapshot) -> None:
        """Append one EMA + hysteresis + OOD gate decision row."""
        try:
            with open(DECISION_CSV_PATH, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        unix_to_local_iso(timestamp_unix),
                        f"{timestamp_unix:.3f}",
                        f"{self.classify_window_s:.3f}",
                        int(sr_hz),
                        decision.label,
                        f"{decision.confidence:.6f}",
                        decision.raw_label,
                        f"{decision.raw_confidence:.6f}",
                        decision.state,
                        decision.hud_action,
                        decision.priority,
                        f"{decision.entropy_norm:.6f}",
                        f"{decision.margin:.6f}",
                        f"{decision.spl_dbfs:.6f}",
                        f"{decision.noise_floor_dbfs:.6f}",
                        f"{decision.speech_prob:.6f}",
                        decision.direction,
                        f"{decision.direction_confidence:.6f}",
                        "" if decision.onset_unix is None else f"{decision.onset_unix:.3f}",
                        "" if decision.offset_unix is None else f"{decision.offset_unix:.3f}",
                        int(decision.is_silence),
                        int(decision.is_ood),
                    ]
                )
        except Exception as exc:
            print(f"[CSV] decision write error: {exc}")

    def _ensure_stt_csv_header(self) -> None:
        """Create the STT CSV file once the transcriber is enabled."""
        try:
            needs_header = (not os.path.exists(STT_CSV_PATH)) or os.path.getsize(STT_CSV_PATH) == 0
            if needs_header:
                with open(STT_CSV_PATH, "a", newline="") as handle:
                    csv.writer(handle).writerow(
                        ["iso_time", "unix_time", "window_s", "samplerate_hz", "language", "language_confidence", "text"]
                    )
            print(f"[CSV] Logging STT to: {STT_CSV_PATH}")
        except Exception as exc:
            print(f"[CSV] Could not prepare STT log: {exc}")


def main() -> None:
    """Start the live pipeline and its Unity-facing HTTP bridge."""
    visualizer = RealTimeSPLVisualizer(
        update_interval_s=0.1,
        history_s=20.0,
        min_db=-120.0,
        max_db=0.0,
        classify=True,
        classify_window_s=1.0,
        use_local_mic=False,
    )
    visualizer.set_session_id("default")
    register_unity_audio_sink(visualizer.feed_unity_chunk)

    http_host = os.environ.get("PIPELINE_HTTP_HOST", "0.0.0.0")
    http_port = int(os.environ.get("PIPELINE_HTTP_PORT", "8000"))
    server_thread = threading.Thread(
        target=start_http_server,
        kwargs={"host": http_host, "port": http_port},
        daemon=True,
        name="PipelineHTTPServer",
    )
    server_thread.start()
    print(f"[MAIN] HTTP server started on {http_host}:{http_port}")
    visualizer.run()


if __name__ == "__main__":
    main()
