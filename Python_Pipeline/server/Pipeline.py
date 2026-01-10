#! source /Users/alpercamli/ENS492/.venv311/bin/activate python3
# -*- coding: utf-8 -*-
#set GEMINI_API_KEY=AIzaSyALM5J0bN5QtDx6PfKPvml_R4ewEf7hxVc

"""
RealTimeSPLVisualizer.py

Real-time mono audio capture with sounddevice and live RMS dBFS plot.
Optionally runs a real-time audio classifier (YAMNet via TensorFlow Hub) on 1-second windows
and overlays the predicted class + confidence on the plot.

Dependencies:
  - Required for SPL meter: numpy, sounddevice, matplotlib
  - Optional for classification: tensorflow, tensorflow-hub

Install:
  pip install numpy sounddevice matplotlib
  # (optional, to enable classification)
  pip install tensorflow tensorflow-hub
"""

import sys
import time
import queue
import signal
import threading
from collections import deque
import os
import csv
from datetime import datetime, timezone
import json
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False
from pipeline_http_bridge import push_important_event, push_llm_event, push_stt_event, push_yamnet_event
from pipeline_http_bridge import start_http_server, register_unity_audio_sink
from pipeline_http_bridge import UNITY_AUDIO_LOCK
from audio_buffer import AudioBuffer
from load_env import load_env
load_env(".env")

[
  "speech",
  "crowd",
  "music",
  "dog",
  "cat",
  "bird",
  "vehicle_horn",
  "traffic_road",
  "car_bus_truck",
  "sirens",
  "rail",
  "aircraft",
  "engine_motion",
  "alarms_buzzer",
  "phone_ring",
  "wind_rain",
  "door_knock",
  "glass_break",
  "explosion_gunshot",
  "Silence",
  "other"
]

# -------- Set matplotlib backend EARLY for macOS GUI safety --------
import matplotlib
if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textwrap import shorten

# -------- Optional imports for classification (handled gracefully) --------
# Try to import TensorFlow / TF-Hub; if unavailable, disable classifier support.
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except Exception:
    tf = None
    hub = None
    TF_AVAILABLE = False

CLASSIFIER_ENABLED = TF_AVAILABLE





# ===================== Device utilities =====================

def list_input_devices():
    """Return list of (index, name, max_input_channels, default_samplerate)."""
    devs = sd.query_devices()
    rows = []
    for i, d in enumerate(devs):
        rows.append((
            i,
            d.get("name", ""),
            d.get("max_input_channels", 0),
            int(d.get("default_samplerate", 0) or 0),
        ))
    return rows


def print_input_devices(chosen_index=None):
    """Pretty print available input devices, marking the chosen one."""
    rows = list_input_devices()
    print("\nAvailable audio devices:")
    print(f"{'Idx':>3}  {'InCh':>4}  {'DefSR':>6}  Name")
    for i, name, in_ch, sr in [(r[0], r[1], r[2], r[3]) for r in rows]:
        mark = "*" if chosen_index == i else " "
        print(f"{i:>3}{mark}  {in_ch:>4}  {sr:>6}  {name}")
    print("('*' marks the selected input device)\n")


def pick_input_device():
    """
    Choose an input device:
      1) sd.default.device[0] if valid input device
      2) mic-like device name heuristic
      3) first device with input channels
    Returns (device_index, device_name).
    """
    devs = sd.query_devices()
    cand = None

    # 1) default input
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    if isinstance(default_in, int) and 0 <= default_in < len(devs):
        if devs[default_in]["max_input_channels"] > 0:
            cand = default_in

    # 2) heuristic
    if cand is None:
        keywords = ("microphone", "mic", "built-in", "external", "usb")
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                name = (d.get("name") or "").lower()
                if any(k in name for k in keywords):
                    cand = i
                    break

    # 3) first input-capable device
    if cand is None:
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                cand = i
                break

    if cand is None:
        raise RuntimeError("No input device with capture channels found.")

    return cand, devs[cand]["name"]


def pick_sample_rate(device_index):
    """
    Choose a sensible samplerate:
      - device default if available
      - else try 48000 then 44100
      - else None (let PortAudio choose)
    """
    info = sd.query_devices(device_index)
    sr = float(info.get("default_samplerate", 0) or 0)
    if sr > 0:
        return int(sr)
    for candidate in (48000, 44100):
        try:
            sd.check_input_settings(device=device_index, samplerate=candidate, channels=1)
            return candidate
        except Exception:
            continue
    return None


# ===================== Optional YAMNet Classifier =====================

# Paths for the reduced model (used only if USE_REDUCED=True)
# Use workspace-relative paths by default. Place your reduced model and labels
# under a `models/` folder in the project root, or set these via environment
# variables `REDUCED_MODEL_DIR` and `REDUCED_LABELS_JSON`.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("PIPELINE_LOG_DIR") or os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.environ.get("PIPELINE_MODELS_DIR") or os.path.join(BASE_DIR, "models")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
except Exception:
    pass

def _resolve_reduced_model_dir():
    env = os.environ.get("REDUCED_MODEL_DIR")
    if env:
        return env
    # Prefer models/ folder
    p1 = os.path.join(MODELS_DIR, "reduced_yamnet_savedmodel")
    # Fallback to project root if user ran builder here
    p2 = os.path.join(BASE_DIR, "reduced_yamnet_savedmodel")
    def has_savedmodel(p):
        return os.path.exists(os.path.join(p, "saved_model.pb")) or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
    return p1 if has_savedmodel(p1) else p2

def _resolve_reduced_labels_json():
    env = os.environ.get("REDUCED_LABELS_JSON")
    if env:
        return env
    p1 = os.path.join(MODELS_DIR, "reduced_labels.json")
    p2 = os.path.join(BASE_DIR, "reduced_labels.json")
    return p1 if os.path.exists(p1) else p2

REDUCED_MODEL_DIR = _resolve_reduced_model_dir()
REDUCED_LABELS_JSON = _resolve_reduced_labels_json()
USE_REDUCED = True  # set True to use your reduced model, False for full YAMNet (521 classes)

# ==== Data collection mode ====
DATA_COLLECTION_MODE = True          # <— turn on/off
WIDE_CSV_PATH = os.path.join(LOG_DIR, "classification_probs.csv")
TOPK_OVERLAY = 5                     # keep your overlay readable
PRINT_ALL_TO_CONSOLE = False         # optional: dump full list to console

# ==== Speech-to-Text (Whisper) ====
STT_ENABLED = True
WHISPER_MODEL_SIZE = "small"   # "tiny", "base", "small", "medium", "large-v3"
WHISPER_COMPUTE_TYPE = "int8"  # "int8"/"float16"/"float32"
STT_WINDOW_S = 5.0             # rolling window length sent to Whisper
STT_HOP_S = 1.0                # how often to re-decode (seconds)
STT_MIN_TEXT_LEN = 1           # ignore empty/short results
# STT and other CSV/jsonl files live under `logs/` by default
STT_CSV_PATH = os.path.join(LOG_DIR, "transcription_log.csv")

# ==== STT gating by YAMNet (speech VAD) ====
SPEECH_GATE_ENABLED = True
SPEECH_LABEL = "speech"       # name in your reduced label set
SPEECH_ON_THRESH = 0.40       # normalized prob to turn ON
SPEECH_OFF_THRESH = 0.30      # normalized prob to turn OFF (hysteresis)
SPEECH_MIN_ON_S = 0.50         # avoid rapid toggling: min time to stay ON
SPEECH_MIN_OFF_S = 0.00        # avoid rapid toggling: min time to stay OFF
WHISPER_LANGUAGE = "en"      # set None for auto-detect
REQUIRE_TOP_IS_SPEECH = False    # require 'speech' to be top-1 class

# STT session bounds
PREROLL_S       = 1.5           # audio kept before trigger
MAX_SESSION_S   = 5.0           # hard cap per utterance (seconds)
DECODE_HOP_S    = 1.5           # how often to re-decode during session

# ===== Gemini / LLM config =====
LLM_ENABLED = True
LLM_DRY_RUN = False  # True: don't actually call API, just log prompts & fake response
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # or updated flash model name

LLM_CSV_PATH = os.path.join(LOG_DIR, "llm_events.csv")
LLM_JSONL_PATH = os.path.join(LOG_DIR, "llm_events.jsonl")

# how much context we send per window
LLM_MAX_EVENTS_PER_WINDOW = 40     # cap events to keep prompt small
LLM_MAX_STT_CHARS = 600           # cap transcript snippet length

# ===== LLM sound-event triggers (no speech needed) =====

# Important reduced YAMNet labels in *your* pooled model
IMPORTANT_SOUND_LABELS = {
  "music",
  "dog",
  "cat",
  "bird",
  "vehicle_horn",
  "traffic_road",
  "car_bus_truck",
  "sirens",
  "rail",
  "aircraft",
  "engine_motion",
  "alarms_buzzer",
  "phone_ring",
  "door_knock",
  "glass_break",
  "explosion_gunshot"
}

# probability threshold for "important" detection
IMPORTANT_SOUND_THRESH = 0.55

# Time context sent to LLM around important event
SOUND_LLM_WINDOW_S = 5.0   # look ~5s back from trigger

# Minimum time between sound-only LLM triggers
SOUND_LLM_COOLDOWN_S = 10.0

GLOBAL_LLM_COOLDOWN_S = 2.0



DEBUG_STT = False
def _dbg(tag, msg):
    if DEBUG_STT:
        import threading, time as _t
        ts = _t.strftime("%H:%M:%S")
        print(f"[{ts}] [{threading.current_thread().name}] {tag}: {msg}")


# ====== NEW: WhisperTranscriber (faster-whisper) ======
class WhisperTranscriber:
    """
    Streaming Whisper with:
      - pre-roll ring buffer (PREROLL_S)
      - session start/stop controlled by YAMNet
      - per-session max duration (MAX_SESSION_S)
    """
    def __init__(self, model_size=WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE, target_sr=16000):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, compute_type=compute_type)
        self.target_sr = target_sr

        # main buffer (active session only)
        self._buf = np.zeros(0, dtype=np.float32)

        # pre-roll ring buffer (always running)
        self._pre = np.zeros(0, dtype=np.float32)
        self._pre_cap = int(round(PREROLL_S * self.target_sr))

        # session control/time
        self.session_active = False
        self._session_start_ts = None
        self._last_decode_ts = 0.0

        # misc
        self._lock = threading.Lock()
        self.latest_text = ""
        self.language = WHISPER_LANGUAGE

    # ---------- feeding ----------
    def feed_preroll(self, x, sr_in):
        """Always called: maintain last PREROLL_S seconds in a ring buffer."""
        if sr_in != self.target_sr:
            x = self._resample_linear(x, sr_in, self.target_sr)
        if x.size == 0:
            return
        with self._lock:
            if self._pre.size == 0:
                self._pre = x.copy()
            else:
                self._pre = np.concatenate([self._pre, x])[-self._pre_cap:]

    def append_audio(self, x, sr_in):
        """Append only when a session is active."""
        if not self.session_active:
            return
        if sr_in != self.target_sr:
            x = self._resample_linear(x, sr_in, self.target_sr)
        if x.size == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, x])

    # ---------- session lifecycle ----------
    def start_session(self, use_preroll: bool = True):
        """Start a session. If use_preroll=False, start with an empty buffer."""
        with self._lock:
            if use_preroll:
                pre = self._pre.copy()
                self._buf = pre
            else:
                self._buf = np.zeros(0, dtype=np.float32)
        self.session_active = True
        self._session_start_ts = time.time()
        self._last_decode_ts = 0.0
        self.latest_text = ""

    def stop_session(self, finalize=True):
        """Stop and optionally decode everything collected so far."""
        self.session_active = False
        text, eff = (None, 0.0)
        if finalize:
            text, eff = self._decode_now(force=True)
            if text:
                self.latest_text = text
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
        self._session_start_ts = None
        return text, eff

    def session_duration_s(self):
        if not self.session_active or self._session_start_ts is None:
            return 0.0
        return time.time() - self._session_start_ts

    # ---------- periodic decode during session ----------
    def maybe_decode(self, now_ts, window_s=STT_WINDOW_S, hop_s=DECODE_HOP_S):
        if not self.session_active:
            return None, 0.0
        if now_ts - self._last_decode_ts < hop_s:
            return None, 0.0
        return self._decode_now(force=False, window_s=window_s)

    # ---------- core decode ----------
    def _decode_now(self, force=False, window_s=STT_WINDOW_S):
        with self._lock:
            if self._buf.size == 0:
                return (None, 0.0)
            if force:
                audio = self._buf.copy()
            else:
                n_keep = int(window_s * self.target_sr)
                audio = self._buf[-n_keep:] if self._buf.size > n_keep else self._buf.copy()

        eff_window_s = float(audio.size) / float(self.target_sr)

        # Optional light preproc/denoise hooks if you kept them:
        # audio = self._preprocess(audio, self.target_sr)
        # audio = self._denoise(audio, self.target_sr)

        segments, info = self.model.transcribe(
            audio,
            language=self.language, task="transcribe",
            beam_size=6, patience=0.2, temperature=[0.0, 0.2, 0.4],
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False, without_timestamps=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        self._last_decode_ts = time.time()
        return (text if len(text) >= STT_MIN_TEXT_LEN else None, eff_window_s)

    @staticmethod
    def _resample_linear(x, sr_in, sr_out):
        if x.size == 0:
            return x
        dur = x.size / float(sr_in)
        t_in  = np.linspace(0.0, dur, num=x.size, endpoint=False)
        n_out = int(round(dur * sr_out))
        t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32, copy=False)

class GeminiEventAnalyzer:
    """
    Wraps a Gemini model for sequential sound-event interpretation
    using the new `google.genai` client.
    """
    def __init__(
        self,
        model_name=GEMINI_MODEL_NAME,
        dry_run=LLM_DRY_RUN,
        csv_path=LLM_CSV_PATH,
        jsonl_path=LLM_JSONL_PATH,
    ):
        self.model_name = model_name
        self.dry_run = dry_run
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

        print(f"[LLM] init: model_name={self.model_name}, dry_run={self.dry_run}")

        self._ensure_logs()

        self._client = None

        if not self.dry_run:
            # Try both names, since docs differ
            api_key = (os.environ.get("GEMINI_API_KEY"))
            print(f"[LLM] init: API key present? {bool(api_key)}")
            if not api_key:
                print("[LLM] No API key in GEMINI_API_KEY or GOOGLE_API_KEY; forcing dry_run=True")
                self.dry_run = True
            else:
                if not GENAI_AVAILABLE:
                    print("[LLM] google-genai package not installed; forcing dry_run=True")
                    self.dry_run = True
                else:
                    # IMPORTANT: This is genai.Client, not google.Client
                    try:
                        self._client = genai.Client()
                        print("[LLM] Gemini client initialized.")
                    except Exception as e:
                        print(f"[LLM] Failed to initialize Gemini client: {e}; falling back to dry_run")
                        self._client = None
                        self.dry_run = True
        else:
            print("[LLM] init: staying in dry_run mode (no API calls).")

    def _ensure_logs(self):
        try:
            needs_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
            if needs_header:
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "iso_time", "unix_time",
                        "window_start_unix", "window_end_unix",
                        "stt_text",
                        "yamnet_events_json",
                        "llm_brief_summary",
                        "llm_user_message",
                        "model_name",
                    ])
                print(f"[LLM] Logging CSV to {self.csv_path}")
        except Exception as e:
            print(f"[LLM] Could not prepare CSV log: {e}")

    @staticmethod
    def _build_prompt(window_dict: dict) -> str:
        start_ts = window_dict.get("start_ts")
        end_ts = window_dict.get("end_ts")
        stt_text = window_dict.get("stt_text") or ""
        events = window_dict.get("yamnet_events") or []

        if len(stt_text) > LLM_MAX_STT_CHARS:
            stt_text = stt_text[:LLM_MAX_STT_CHARS] + "…"

        lines = []
        for ev in events[:LLM_MAX_EVENTS_PER_WINDOW]:
            rel_t = ev.get("rel_t")
            top5 = ev.get("top5", [])
            label_str = ", ".join([f"{x['label']} ({x['prob']:.2f})" for x in top5])
            lines.append(f"t={rel_t:+.1f}s -> {label_str}")

        events_block = "\n".join(lines) if lines else "(no YAMNet events captured)"

        prompt = f"""
You are assisting a Deaf or Hard-of-Hearing person by summarizing the acoustic environment.

You receive:
1) A short recent transcript of speech (STT).
2) A timeline of sound event probabilities from an audio classifier (YAMNet).
   We are mostly interested in alarms, vehicles, announcements, footsteps, doors, glass breaking, and other important events.
   Music and generic background noise can usually be ignored unless it conveys an important context.

Your goals:
- Briefly explain what seems to be happening around the user in plain English.
- If appropriate, warn the user about relevant events (e.g., approaching station, alarms, vehicles, glass breaking).
- Be concise (1–3 short sentences), not verbose.
- Use the transcript to infer context creatively (e.g., if someone says "next station is New York", say "You are approaching New York station.").

Constraints:
- Respond strictly in JSON with the following keys:
  {{
    "brief_summary": "one short sentence about the environment",
    "user_message": "one or two short sentences to show to the user",
    "important_events": [
      {{
        "type": "speech" | "alarm" | "vehicle" | "impact" | "other",
        "description": "short description",
        "priority": "low" | "medium" | "high"
      }}
    ]
  }}
- Do NOT include any explanation, markdown, or text outside the JSON.
- Your entire response MUST be valid JSON only.
Now the current window:

Time window (unix): start={start_ts}, end={end_ts}

Recent speech transcript:
\"\"\"{stt_text}\"\"\"

YAMNet top-5 timeline:
{events_block}
"""
        return prompt

    def analyze_window(self, window_dict: dict) -> dict:
        prompt = self._build_prompt(window_dict)

        ts_now = time.time()
        iso = datetime.fromtimestamp(ts_now, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")

        default = {
            "brief_summary": "Environment summary unavailable.",
            "user_message": "The assistant could not generate an environment summary for this window.",
            "important_events": [],
            "raw_response": "",
        }


        # just logging mode
        print(f"[LLM] analyze_window: dry_run={self.dry_run}, client_none={self._client is None}")

        if self.dry_run or self._client is None:
            try:
                self._log(iso, ts_now, window_dict, default)
            except Exception as e:
                print(f"[LLM] dry_run log error: {e}")
            print("[LLM] dry_run: would send prompt to Gemini, but skipping.")
            return default

        # real API call using new client
        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # With google-genai, text is usually in resp.text or resp.candidates[0].content.parts
            text = getattr(resp, "text", None)
            if text is None and getattr(resp, "candidates", None):
                # fallback: join parts if needed
                parts = []
                for c in resp.candidates:
                    for p in getattr(c.content, "parts", []):
                        if hasattr(p, "text"):
                            parts.append(p.text)
                text = "\n".join(parts)
            if text is None:
                raise RuntimeError("LLM response has no text")
        except Exception as e:
            print(f"[LLM] ERROR calling Gemini: {e}")
            try:
                self._log(iso, ts_now, window_dict, default)
            except Exception as e2:
                print(f"[LLM] log error after Gemini failure: {e2}")
            return default

        import re

        # Try to extract JSON even if the model wraps it in text/markdown
        parsed = None
        try:
            # common case: the whole response is JSON
            parsed = json.loads(text)
        except Exception:
            # try to find the first {...} block
            try:
                m = re.search(r'\{.*\}', text, re.S)
                if m:
                    json_str = m.group(0)
                    parsed = json.loads(json_str)
            except Exception:
                parsed = None

        if parsed is None:
            print("[LLM] JSON parse failed, raw response (first 300 chars):")
            print(repr(text[:300]))
            parsed = dict(default)
            parsed["raw_response"] = text


        brief = parsed.get("brief_summary") or default["brief_summary"]
        msg = parsed.get("user_message") or brief
        evs = parsed.get("important_events") or []

        result = {
            "brief_summary": brief,
            "user_message": msg,
            "important_events": evs,
            "raw_response": text,
        }

        try:
            session_id = getattr(self, "session_id", "default")
            start_ts = window_dict.get("start_ts")
            end_ts = window_dict.get("end_ts")

            push_llm_event(
                session_id=session_id,
                window_start=start_ts,
                window_end=end_ts,
                brief_summary=result.get("brief_summary", ""),
                user_message=result.get("user_message", ""),
                important_events=result.get("important_events", []),
            )
        except Exception as e:
            print("[LLM] Failed to push LLM event:", e)

        try:
            self._log(iso, ts_now, window_dict, result)
        except Exception as e:
            print(f"[LLM] log error: {e}")

        return result

    def _log(self, iso_time, unix_time, window_dict, result_dict):
        try:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    iso_time,
                    f"{unix_time:.3f}",
                    f"{window_dict.get('start_ts', 0.0):.3f}",
                    f"{window_dict.get('end_ts', 0.0):.3f}",
                    (window_dict.get("stt_text") or "").replace("\n", " "),
                    json.dumps(window_dict.get("yamnet_events") or []),
                    result_dict.get("brief_summary", ""),
                    result_dict.get("user_message", ""),
                    self.model_name,
                ])
        except Exception as e:
            print(f"[LLM] CSV log error: {e}")

        try:
            rec = {
                "iso_time": iso_time,
                "unix_time": unix_time,
                "window": window_dict,
                "result": result_dict,
                "model_name": self.model_name,
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LLM] JSONL log error: {e}")




class YamnetClassifier:
    """
    Supports:
      - Reduced SavedModel (your pooled labels)  -> self.infer(...)
      - Full YAMNet (TF-Hub)                     -> self.yamnet(...)
    Provides:
      - predict_all() -> (labels, probs[K])  with probs normalized to sum=1
      - predict_top() -> (label, prob)
    """
    def __init__(self):


        if not CLASSIFIER_ENABLED:
            raise RuntimeError("TensorFlow not available; classifier disabled.")
        self.target_sr = 16000


        if USE_REDUCED:
            # Reduced SavedModel
            from json import load
            global REDUCED_MODEL_DIR, REDUCED_LABELS_JSON
            print("Loading Reduced-YAMNet SavedModel…")
            mod = tf.saved_model.load(REDUCED_MODEL_DIR)
            self.infer = mod.__call__.get_concrete_function()
            with open(REDUCED_LABELS_JSON, "r", encoding="utf-8") as f:
                self.class_names = load(f)
            self.mode = "reduced"
            print("[OK] Reduced-YAMNet loaded.")
        else:
            # Full YAMNet
            import csv, tensorflow_hub as hub
            print("Loading full YAMNet from TF-Hub…")
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self.yamnet.class_map_path().numpy().decode("utf-8")
            names = []
            with tf.io.gfile.GFile(class_map_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    names.append(row["display_name"])
            self.class_names = names
            self.mode = "full"
            print("[OK] Full YAMNet loaded (521 classes).")

    def predict_all(self, mono_audio_1s, input_sr):
        """Return (labels, probs[K]) where probs are normalized (sum=1)."""
        if input_sr != self.target_sr:
            mono_audio_1s = self._resample_linear(mono_audio_1s, input_sr, self.target_sr)
        y = tf.convert_to_tensor(mono_audio_1s, dtype=tf.float32)

        if self.mode == "reduced":
            out = self.infer(waveform_16k=y)             # {"probs": [K]}
            probs = out["probs"].numpy()
        else:
            scores, embeddings, _ = self.yamnet(y)       # scores: [T,521]
            probs = tf.reduce_mean(scores, axis=0).numpy()

        # Option 1: runtime normalization so values <= 1 and sum to 1
        probs = probs / (probs.sum() + 1e-9)
        return self.class_names, probs

    def predict_top(self, mono_audio_1s, input_sr):
        labels, probs = self.predict_all(mono_audio_1s, input_sr)
        i = int(np.argmax(probs))
        return labels[i] if 0 <= i < len(labels) else "Unknown", float(probs[i])

    @staticmethod
    def _resample_linear(x, sr_in, sr_out):
        if x.size == 0:
            return x
        dur = x.size / float(sr_in)
        t_in  = np.linspace(0.0, dur, num=x.size, endpoint=False)
        n_out = int(round(dur * sr_out))
        t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32, copy=False)




# ===================== Visualizer with Optional Classification =====================

class RealTimeSPLVisualizer:
    def __init__(self,
                 update_interval_s=0.1,    # Plot update period
                 history_s=20.0,           # Seconds of history
                 min_db=-120.0,
                 max_db=0.0,
                 classify=True,            # Try to enable classifier
                 classify_window_s=2.0,
                use_local_mic=False,
                log_csv_path=os.path.join(LOG_DIR, "classification_log.csv")):   # Classification window length
        self.update_interval_s = float(update_interval_s)
        self.history_s = float(history_s)
        self.min_db = float(min_db)
        self.max_db = float(max_db)

        self.use_local_mic=use_local_mic

        self._gate_state = "IDLE"      # IDLE, RECORDING, COOLDOWN
        self._gate_last_change = time.time()

        # Queues / buffers
        self.q_levels = queue.Queue()   # (t, dBFS) for plotting
        self.q_audio = queue.Queue()    # raw audio chunks (float32 arrays) for classifier
        self.max_points = int(np.ceil(self.history_s / self.update_interval_s)) + 10
        self.times = deque(maxlen=self.max_points)
        self.levels = deque(maxlen=self.max_points)

        # Device & rate
        self.device_index, self.device_name = pick_input_device()
        self.samplerate = pick_sample_rate(self.device_index)
        self.stream_samplerate = self.samplerate if self.samplerate else None
        self.blocksize = (None if self.stream_samplerate is None
                          else max(1, int(self.stream_samplerate * self.update_interval_s)))

        print_input_devices(self.device_index)
        print(f"Selected input device #{self.device_index}: {self.device_name}")
        print(f"Sample rate: {self.stream_samplerate or 'default'} Hz | Blocksize: {self.blocksize or 'default'}")
        print("If the plot is flat: ensure macOS Microphone permission for Python is enabled.\n"
              "System Settings → Privacy & Security → Microphone → enable for your Python/Terminal app.\n")

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(11, 4))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(self.min_db, self.max_db)
        self.ax.set_xlim(-self.history_s, 0.0)
        title_sr = self.stream_samplerate or "default"
        self.ax.set_title(f"Real-Time SPL (RMS dBFS) — {shorten(self.device_name, width=48)} @ {title_sr} Hz (mono)")
        self.ax.set_xlabel("Time (s) relative to now")
        self.ax.set_ylabel("SPL (dBFS)")
        self.ax.grid(True, linestyle="--", alpha=0.4)

        # Overlays: numeric dBFS and classifier label
        self.text_readout = self.ax.text(
            0.01, 0.95, "— dBFS",
            transform=self.ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w")
        )
        self.text_label = self.ax.text(
            0.99, 0.95, "Classifier: disabled",
            transform=self.ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w")
        )
        # STT overlay (bottom-right)
        self.text_stt = self.ax.text(
            0.99, 0.05, "STT: disabled",
            transform=self.ax.transAxes, va="bottom", ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w")
        )


        # Audio stream (mono)
        if self.use_local_mic:
            self._zero_blocks_seen = 0
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                dtype="float32",
                samplerate=self.stream_samplerate,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
        else:
            self.stream = DummyStream()



        # Animation (GUI main thread)
        self.ani = FuncAnimation(self.fig, self.on_timer,
                                 interval=int(self.update_interval_s * 1000),
                                 blit=False)

        # Ctrl+C handler
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Classification support
        self.classifier_enabled = classify and CLASSIFIER_ENABLED
        self.classify_window_s = float(classify_window_s)
        self.latest_label = None
        self.latest_conf = None
        self._label_lock = threading.Lock()

        if self.classifier_enabled:
            try:
                self.classifier = YamnetClassifier()
                self.text_label.set_text("Classifier: loading…")
                # Start background classifier worker
                self._classifier_thread = threading.Thread(
                    target=self._classification_worker, daemon=True)
                self._classifier_thread.start()
            except Exception as e:
                print(f"[Classifier] Disabled: {e}")
                self.classifier_enabled = False
                self.text_label.set_text("Classifier: disabled")
        else:
            self.text_label.set_text("Classifier: disabled")
        self.log_csv_path = log_csv_path if self.classifier_enabled else None

        # --- STT (Whisper) support ---
        self.stt_enabled = STT_ENABLED
        self.q_audio_stt = queue.Queue() if self.stt_enabled else None
        self._stt_lock = threading.Lock()
        self.latest_transcript = ""

        if self.stt_enabled:
            try:
                self.stt = WhisperTranscriber()
                self.text_stt.set_text("STT: loading…")
                self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
                self._stt_thread.start()
            except Exception as e:
                print(f"[STT] Disabled: {e}")
                self.stt_enabled = False
                self.text_stt.set_text("STT: disabled")
        else:
            self.text_stt.set_text("STT: disabled")

        # ensure STT CSV header
        if self.stt_enabled:
            try:
                needs_header = (not os.path.exists(STT_CSV_PATH)) or os.path.getsize(STT_CSV_PATH) == 0
                if needs_header:
                    with open(STT_CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow(["iso_time", "unix_time", "window_s", "samplerate_hz", "text"])
                print(f"[CSV] Logging STT to: {STT_CSV_PATH}")
            except Exception as e:
                print(f"[CSV] Could not prepare STT log: {e}")


        if self.log_csv_path:
            self._ensure_csv_header()


        # --- LLM / Gemini integration ---
        self._last_sound_llm_ts = 0.0
        self._last_any_llm_ts = 0.0

        self.llm_enabled = LLM_ENABLED
        self.llm_analyzer = None
        self._llm_queue = queue.Queue() if self.llm_enabled else None
        self._llm_events_buffer = deque(maxlen=600)  # ~60s if called ~10Hz
        self._llm_buf_lock = threading.Lock()

        # LLM summary overlay
        self.text_llm = self.ax.text(
            0.5, 0.02, "Env: —",
            transform=self.ax.transAxes, va="bottom", ha="center",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w")
        )

        if self.llm_enabled:
            try:
                self.llm_analyzer = GeminiEventAnalyzer()
                self._llm_thread = threading.Thread(
                    target=self._llm_worker, daemon=True
                )
                self._llm_thread.start()
                print("[LLM] GeminiEventAnalyzer initialized.")
            except Exception as e:
                print(f"[LLM] disabled (init error): {e}")
                self.llm_enabled = False




    def feed_unity_chunk(self, samples: np.ndarray):
        """
        Unity → Python audio bridge.
        In Unity mode (use_local_mic=False), we simulate a sounddevice
        callback so all existing logic (YAMNet + STT pre-roll + gating)
        keeps working unchanged.
        """
        if samples is None or samples.size == 0:
            return

        try:
            # Ensure 1D float32
            if samples.ndim != 1:
                samples = samples.reshape(-1)
            samples = samples.astype(np.float32, copy=False)

            # DEBUG (optional)
            max_abs = float(np.max(np.abs(samples)))
            print(f"[UnityAudio] received chunk len={samples.size}, max_abs={max_abs:.4f}")

            # In local-mic mode, sounddevice already calls audio_callback.
            # In Unity mode (use_local_mic=False), we must call it manually.
            if not getattr(self, "use_local_mic", True):
                # sounddevice passes indata shape (frames, channels)
                frames = samples.shape[0]
                indata = samples.reshape(-1, 1)  # mono: (N, 1)

                dummy_time = None   # you can keep None, callback usually doesn't need it
                dummy_status = None # same here

                # This will:
                #  - push into self.q_audio
                #  - feed STT pre-roll / buffers
                #  - keep everything consistent
                self.audio_callback(indata, frames, dummy_time, dummy_status)
            else:
                # fallback: if you ever use Unity while local mic is on (unlikely)
                # you can still just push to q_audio
                self.q_audio.put(samples)

            # Optional: see how full q_audio gets
            try:
                qsize = self.q_audio.qsize()
            except Exception:
                qsize = -1
            print(f"[UnityAudio] q_audio size now: {qsize}")

        except Exception as e:
            print(f"[UnityAudio] Failed to route samples via audio_callback: {e}")



    # ----------------- audio callback ----------------- #
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        if frames <= 0:
            return

        x = indata[:, 0]  # float32 mono
        # zero-block detection
        if np.allclose(x, 0.0):
            self._zero_blocks_seen += 1
        else:
            self._zero_blocks_seen = 0

        # dBFS for the block
        # use float64 for RMS to improve numerical stability
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        dbfs = 20.0 * np.log10(rms + 1e-9)
        t = time.monotonic()

        # push to queues (non-blocking)
        try:
            self.q_levels.put_nowait((t, dbfs))
        except queue.Full:
            pass

        if self.classifier_enabled:
            # copy to avoid re-use of memory
            print("q audio class ok")
            try:
                self.q_audio.put_nowait(x.copy())
            except queue.Full:
                pass
        #check for stt error
        if self.stt_enabled and self.q_audio_stt is not None:
            print("q audio stt ok")
            try:
                self.q_audio_stt.put_nowait(x.copy())
            except queue.Full:
                pass


    def _classification_worker(self):
        """
        Background thread:
        - Collects ~1 second of audio
        - Runs classifier
        - Logs to CSV
        - (optional) writes all label confidences in data collection mode
        """
        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        chunk_buffer = []
        samples_target = int(round(sr_in * self.classify_window_s))
        samples_accum = 0

        # indicate ready
        with self._label_lock:
            self.latest_label = "Ready"
            self.latest_conf = None
        self.text_label.set_text("Classifier: ready")

        while True:
            try:
                chunk = self.q_audio.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_buffer.append(chunk)
            # print("Audio Size: ", len(chunk_buffer) , chunk.size)
            samples_accum += chunk.size

            if samples_accum >= samples_target:
                # === combine chunks to ~1s window ===
                audio_1s = np.concatenate(chunk_buffer, axis=0)
                audio_1s = audio_1s[-samples_target:]  # keep last N samples
                chunk_buffer.clear()
                samples_accum = 0

                # --- compute SPL in dBFS for this window (for Unity / logging) ---  # NEW
                rms = float(np.sqrt(np.mean(audio_1s ** 2) + 1e-9))
                spl_dbfs = 20.0 * np.log10(rms + 1e-9)  # NEW

                # === run classifier ===
                try:
                    # We'll prepare variables used later for push_yamnet_event      # NEW
                    top5 = None                                                     # NEW
                    dominant_label = None                                           # NEW
                    dominant_prob = None                                            # NEW

                    if DATA_COLLECTION_MODE:
                        print("DATA COLLECTION: TRUE")
                        labels, probs = self.classifier.predict_all(audio_1s, int(sr_in))

                        order = np.argsort(probs)[::-1]
                        k = min(TOPK_OVERLAY, len(order))
                        lines = [f"{labels[i]} ({probs[i]*100:.0f}%)" for i in order[:k]]
                        overlay_text = "\n".join(lines)

                        # top-1 for legacy CSV
                        top_i = order[0]
                        label, conf = labels[top_i], float(probs[top_i])

                        # --- build top5 for Unity / LLM buffer ---                 # NEW
                        top5_idx = order[:5]
                        top5 = [
                            {"label": labels[i], "prob": float(probs[i])}
                            for i in top5_idx
                        ]
                        dominant_label = top5[0]["label"]
                        dominant_prob = top5[0]["prob"]

                        # --- LLM: feed buffer with YAMNet events ---               # (slightly refactored)
                        if self.llm_enabled:
                            now = time.time()
                            with self._llm_buf_lock:
                                self._llm_events_buffer.append({
                                    "ts": now,
                                    "top5": top5,
                                })

                        # --- LLM: sound-only trigger (no speech required) ---
                        if self.llm_enabled and self._llm_queue is not None:
                            # check if any important label is in top-5 with high prob
                            important_hits = [
                                (l["label"], l["prob"])
                                for l in top5
                                if l["label"] in IMPORTANT_SOUND_LABELS and l["prob"] >= IMPORTANT_SOUND_THRESH
                            ]

                            if important_hits:
                                now_ts = now
                                # rate-limit triggers
                                if now_ts - self._last_sound_llm_ts >= SOUND_LLM_COOLDOWN_S:
                                    self._last_sound_llm_ts = now_ts

                                    # Build a time window [start_ts, now_ts]
                                    start_ts = now_ts - SOUND_LLM_WINDOW_S

                                    # Snapshot YAMNet events in this interval
                                    with self._llm_buf_lock:
                                        buf = list(self._llm_events_buffer)

                                    slice_events = []
                                    for ev in buf:
                                        ts_ev = ev.get("ts", 0.0)
                                        if start_ts <= ts_ev <= now_ts:
                                            slice_events.append({
                                                "ts": ts_ev,
                                                "rel_t": ts_ev - now_ts,   # negative = in the past
                                                "top5": ev.get("top5", []),
                                            })

                                    # Use latest STT text (if any) as weak context
                                    with self._stt_lock:
                                        stt_context = (self.latest_transcript or "").strip()

                                    window = {
                                        "start_ts": start_ts,
                                        "end_ts": now_ts,
                                        "stt_text": stt_context,      # may be "" if no one spoke
                                        "yamnet_events": slice_events,
                                    }

                                    try:
                                        self._llm_queue.put_nowait(window)
                                        # print(f"[LLM] sound-trigger window enqueued, hits={important_hits}")
                                    except queue.Full:
                                        print("[LLM] queue full; dropping sound-trigger window.")

                        # print all if desired
                        if PRINT_ALL_TO_CONSOLE:
                            maxw = max(len(s) for s in labels)
                            print("\n--- probabilities ---")
                            for i in order:
                                print(f"{labels[i]:<{maxw}}  {probs[i]:.4f}")

                        # wide CSV (every label)
                        ts_now = time.time()
                        self._append_wide_csv_row(ts_now, sr_in, labels, probs)

                    else:
                        # Simple mode: only top-1
                        label, conf = self.classifier.predict_top(audio_1s, int(sr_in))
                        overlay_text = f"{label} ({int(round(conf*100))}%)"

                        # Build a degenerate top5 for Unity: just top-1           # NEW
                        top5 = [{"label": label, "prob": float(conf)}]            # NEW
                        dominant_label = label                                     # NEW
                        dominant_prob = float(conf)                                # NEW

                    # --- Speech gate / STT trigger logic unchanged ---             # (only reads labels/probs)

                    if SPEECH_GATE_ENABLED and self.stt_enabled:
                        print("SPEECH_GATE_ENABLED: TRUE")
                        now = time.time()

                        # speech probability & top-1 check
                        try:
                            # In DATA_COLLECTION_MODE we have labels/probs; in else we don't,
                            # so we guard this whole block with DATA_COLLECTION_MODE.         # NEW
                            if DATA_COLLECTION_MODE:
                                print("DATA COLLECTION SPEECH: TRUE")
                                speech_idx = labels.index(SPEECH_LABEL)
                                speech_prob = float(probs[speech_idx])
                                top_i = int(np.argmax(probs))
                                top_is_speech = (labels[top_i] == SPEECH_LABEL)
                            else:
                                # If not in data collection mode, we don't have full probs;
                                # you can keep your old logic here or just approximate.       # NEW
                                speech_prob = conf if (label == SPEECH_LABEL) else 0.0
                                top_is_speech = (label == SPEECH_LABEL)

                        except Exception:
                            speech_prob = 0.0
                            top_is_speech = False

                        cond_on = (speech_prob >= SPEECH_ON_THRESH) and (top_is_speech if REQUIRE_TOP_IS_SPEECH else True)
                        cond_off = (speech_prob < SPEECH_OFF_THRESH) or (not top_is_speech and REQUIRE_TOP_IS_SPEECH)

                        if self._gate_state == "IDLE":
                            if cond_on:
                                self._gate_state = "RECORDING"
                                self.stt.start_session()
                                self.text_stt.set_text("STT: listening…")

                        elif self._gate_state == "RECORDING":
                            if cond_off:
                                final_txt, eff = self.stt.stop_session(finalize=True)
                                if final_txt:
                                    self._stt_publish(final_txt, float(self.stream_samplerate or 48000.0), eff)
                                self.text_stt.set_text("STT: idle")
                                self._gate_state = "COOLDOWN"

                        elif self._gate_state == "COOLDOWN":
                            if cond_off:
                                pass
                            else:
                                self._gate_state = "IDLE"

                except Exception as e:
                    overlay_text = "Classification error"
                    label, conf = "Classification error", 0.0
                    print(f"[Classifier] Inference error: {e}")
                    # Fallback values for Unity event in error case              # NEW
                    top5 = [{"label": "error", "prob": 1.0}]                     # NEW
                    dominant_label = "error"                                     # NEW
                    dominant_prob = 1.0                                          # NEW

                # === update UI and CSV (your existing behavior) ===
                with self._label_lock:
                    self.latest_label = overlay_text
                    self.latest_conf = None if DATA_COLLECTION_MODE else conf

                ts_now = time.time()
                self._append_csv_row(ts_now, sr_in, label, conf)

                # === NEW: push YAMNet event to Unity/HTTP bridge ===            # NEW
                try:
                    window_end = ts_now
                    window_start = window_end - self.classify_window_s

                    # You can decide what session_id means; if you don't have one, use "default"
                    session_id = getattr(self, "session_id", "default")

                    if top5 is not None and dominant_label is not None:
                        push_yamnet_event(
                            session_id=session_id,
                            timestamp_unix=window_end,
                            window_start=window_start,
                            window_end=window_end,
                            top5=top5,
                            dominant_label=dominant_label,
                            dominant_prob=dominant_prob,
                            spl_dbfs=float(spl_dbfs),
                        )
                except Exception as e:
                    print(f"[Classifier] Failed to push YAMNet event: {e}")




    def _stt_worker(self):
        if not self.stt_enabled:
            return
        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        self.text_stt.set_text("STT: ready")

        while True:
            try:
                chunk = self.q_audio_stt.get(timeout=1.0)
            except queue.Empty:
                # still try periodic decode during an active session
                txt, eff = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
                if txt:
                    self._stt_publish(txt, sr_in, eff)
                continue

            # Always maintain pre-roll ring
            self.stt.feed_preroll(chunk, sr_in)

            # Only append to active session
            if self.stt.session_active:
                self.stt.append_audio(chunk, sr_in)

            # Periodic decode while session is active
            txt, eff = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
            if txt:
                self._stt_publish(txt, sr_in, eff)

            # Hard stop by time cap (purely time-based)
            if self.stt.session_active and (self.stt.session_duration_s() >= MAX_SESSION_S):
                # 1) finalize current slice
                final_txt, eff = self.stt.stop_session(finalize=True)
                if final_txt:
                    self._stt_publish(final_txt, sr_in, eff)

                # 2) If we're still in speech (gate is RECORDING), roll into a new session
                if getattr(self, "_gate_state", None) == "RECORDING":
                    # restart immediately, but WITHOUT pre-roll to avoid overlap/duplication
                    self.stt.start_session(use_preroll=False)
                    self.text_stt.set_text("STT: listening…")
                else:
                    # gate already turned off elsewhere
                    self.text_stt.set_text("STT: idle")

    def _llm_worker(self):
        """Background thread: consume windows, call Gemini, update overlay + logs."""
        if not self.llm_enabled or self.llm_analyzer is None:
            return
        while True:
            window = self._llm_queue.get()
            try:
                result = self.llm_analyzer.analyze_window(window)
            except Exception as e:
                print(f"[LLM] worker error: {e}")
                continue

            # Update overlay text from result
            summary = result.get("brief_summary") or result.get("user_message") or ""
            if summary:
                short = summary if len(summary) <= 80 else (summary[:77] + "…")
                # Note: this is in a background thread; matplotlib is usually okay with
                # text changes, but if you ever see warnings you can gate this via a flag
                self.text_llm.set_text(f"Env: {short}")




    def _stt_publish(self, text, sr_in, eff_window_s):
        """Publish latest transcript to the overlay and CSV with the actual window length."""
        if not text:
            return
        with self._stt_lock:
            self.latest_transcript = text

        shown = text if len(text) <= 60 else (text[:57] + "…")
        self.text_stt.set_text(f"STT: {shown}")

        # Log with the actual effective window length
        try:
            ts = time.time()
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            with open(STT_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([iso, f"{ts:.3f}", f"{eff_window_s:.3f}", int(sr_in), text])
        except Exception as e:
            print(f"[CSV] STT write error: {e}")

        # --- NEW: push STT segment event to Unity/HTTP bridge ---
        try:
            session_id = getattr(self, "session_id", "default")
            end_unix = ts
            start_unix = end_unix - float(eff_window_s)
            segment_id = f"stt-{int(end_unix * 1000)}"

            push_stt_event(
                session_id=session_id,
                segment_id=segment_id,
                start_unix=start_unix,
                end_unix=end_unix,
                eff_window_s=float(eff_window_s),
                samplerate_hz=int(sr_in),
                text=text,
                language="en",    # or whatever you actually detect/use
                confidence=1.0    # if you get confidences from Whisper, plug them in here
            )
        except Exception as e:
            print(f"[STT] Failed to push STT event: {e}")

        # ---- LLM window trigger ----
        if self.llm_enabled and self._llm_queue is not None and eff_window_s > 0.0 and text.strip():
            end_ts = time.time()
            start_ts = end_ts - eff_window_s

            # snapshot of YAMNet events in this interval
            with self._llm_buf_lock:
                events = list(self._llm_events_buffer)

            # filter & add relative time
            slice_events = []
            for ev in events:
                ts_ev = ev.get("ts", 0.0)
                if start_ts <= ts_ev <= end_ts:
                    slice_events.append({
                        "ts": ts_ev,
                        "rel_t": ts_ev - end_ts,  # negative = in the past
                        "top5": ev.get("top5", []),
                    })

            window = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "stt_text": text,
                "yamnet_events": slice_events,
            }

            now_ts = time.time()
            if now_ts - self._last_any_llm_ts < GLOBAL_LLM_COOLDOWN_S:
                # skip this one to avoid near-duplicate LLM calls
                return  # depending on context
            self._last_any_llm_ts = now_ts

            try:
                self._llm_queue.put_nowait(window)
            except queue.Full:
                print("[LLM] queue full; dropping this window.")





    # ----------------- plot updater ----------------- #
    def on_timer(self, _frame):
        updated = False
        now = time.monotonic()

        # Drain level queue
        try:
            while True:
                t, dbfs = self.q_levels.get_nowait()
                self.times.append(t)
                dbfs = float(np.clip(dbfs, self.min_db, self.max_db))
                self.levels.append(dbfs)
                updated = True
        except queue.Empty:
            pass

        if not updated:
            return self.line,

        # Convert times to "seconds ago"
        rel_times = np.array(self.times, dtype=np.float64) - now
        mask = rel_times >= -self.history_s
        rel_times = rel_times[mask]
        rel_levels = np.array(self.levels, dtype=np.float64)[mask]

        self.line.set_data(rel_times, rel_levels)
        self.ax.set_xlim(-self.history_s, 0.0)

        # Update numeric dBFS readout
        if rel_levels.size > 0:
            self.text_readout.set_text(f"{rel_levels[-1]:.1f} dBFS")

        # Mic zero hint
        if self._zero_blocks_seen >= 5:
            self.ax.set_title("Real-Time SPL (RMS dBFS) — No signal detected (check mic permission / device)")
        else:
            # keep original informative title
            title_sr = self.stream_samplerate or "default"
            self.ax.set_title(f"Real-Time SPL (RMS dBFS) — {shorten(self.device_name, width=48)} @ {title_sr} Hz (mono)")

        # Update classifier label overlay
        if self.classifier_enabled:
            with self._label_lock:
                if self.latest_label is not None:
                    if self.latest_conf is None:
                        self.text_label.set_text(f"{self.latest_label}")
                    else:
                        pct = int(round(self.latest_conf * 100))
                        self.text_label.set_text(f"{self.latest_label} ({pct}%)")
        else:
            self.text_label.set_text("Classifier: disabled")

        return self.line,

    # ----------------- run / cleanup ----------------- #
    def run(self):
        try:
            with self.stream:
                plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            if self.stream.active:
                self.stream.abort()

    def _sigint_handler(self, *_):
        plt.close(self.fig)

    def _ensure_wide_csv_header(self, labels):
        """Create wide CSV with one column per label."""
        try:
            needs_header = (not os.path.exists(WIDE_CSV_PATH)) or os.path.getsize(WIDE_CSV_PATH) == 0
            if needs_header:
                with open(WIDE_CSV_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    header = ["iso_time", "unix_time", "window_s", "samplerate_hz"] + list(labels)
                    w.writerow(header)
                print(f"[CSV] Logging full probabilities to: {WIDE_CSV_PATH}")
        except Exception as e:
            print(f"[CSV] Could not prepare wide CSV: {e}")
            return False
        return True

    def _append_wide_csv_row(self, ts_unix, sr_hz, labels, probs):
        """Append one row of per-label probabilities to the wide CSV."""
        try:
            if not os.path.exists(WIDE_CSV_PATH) or os.path.getsize(WIDE_CSV_PATH) == 0:
                if not self._ensure_wide_csv_header(labels):
                    return
            iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            row = [iso, f"{ts_unix:.3f}", f"{self.classify_window_s:.3f}", int(sr_hz)] + [f"{p:.6f}" for p in probs.tolist()]
            with open(WIDE_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[CSV] wide write error: {e}")

    def _ensure_csv_header(self):
        """Create the CSV file and write a header row if needed."""
        try:
            needs_header = (not os.path.exists(self.log_csv_path)) or os.path.getsize(self.log_csv_path) == 0
            if needs_header:
                with open(self.log_csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iso_time", "unix_time", "window_s", "samplerate_hz", "label", "confidence"])
                print(f"[CSV] Logging classifications to: {self.log_csv_path}")
        except Exception as e:
            print(f"[CSV] Could not prepare log file: {e}")
            self.log_csv_path = None  # disable logging if we can’t write

    def _append_csv_row(self, ts_unix, sr_hz, label, conf):
        """Append one classification result row to the CSV."""
        if not self.log_csv_path:
            return
        try:
            iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            with open(self.log_csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([iso, f"{ts_unix:.3f}", f"{self.classify_window_s:.3f}", int(sr_hz), label, f"{conf:.6f}"])
        except Exception as e:
            print(f"[CSV] write error: {e}")



def main():
    vis = RealTimeSPLVisualizer(
        update_interval_s=0.1,  # UI update rate (s)
        history_s=20.0,         # seconds of plot history
        min_db=-120.0,
        max_db=0.0,
        classify=True,          # set False to force-disable classification
        classify_window_s=1.0,  # model window (s)
        use_local_mic=False     # turn OFF sounddevice, Unity will feed audio
    )

    # Session ID for events
    vis.session_id = "default"

    # Unity audio → pipeline (q_audio)
    register_unity_audio_sink(vis.feed_unity_chunk)

    # HTTP server in background thread
    http_host = os.environ.get("PIPELINE_HTTP_HOST", "0.0.0.0")
    http_port = int(os.environ.get("PIPELINE_HTTP_PORT", "8000"))
    server_thread = threading.Thread(
        target=start_http_server,
        kwargs={"host": http_host, "port": http_port},
        daemon=True
    )
    server_thread.start()
    print(f"[MAIN] HTTP server started on {http_host}:{http_port}")

    # Run your existing pipeline (blocking)
    vis.run()

class DummyStream:
    """A no-op stream used when Unity is providing audio instead of sounddevice."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def start(self):
        # No-op
        pass

    def stop(self):
        # No-op
        pass

    @property
    def active(self):
        # Pretend it's not active (or True if your logic needs it)
        return False


if __name__ == "__main__":
    main()
'''
#! source /Users/alpercamli/ENS492/.venv311/bin/activate python3
# -*- coding: utf-8 -*-
# set GEMINI_API_KEY=...

"""
RealTimeSPLVisualizer.py

Real-time mono audio capture (sounddevice or Unity feed) + live RMS dBFS plot.
Optional:
  - Reduced YAMNet (SavedModel) or full YAMNet (TF-Hub)
  - Whisper STT (faster-whisper) gated by YAMNet speech prob
  - Gemini LLM summary windows
  - Experiment runner (HTTP server) in the SAME FILE (no extra .py)

Dependencies:
  - Required: numpy, sounddevice, matplotlib
  - Optional: tensorflow, tensorflow-hub
  - Optional: faster-whisper
  - Optional: google-genai

Install:
  pip install numpy sounddevice matplotlib
  pip install tensorflow tensorflow-hub
  pip install faster-whisper
  pip install google-genai
"""

import sys
import time
import queue
import signal
import threading
from collections import deque
import os
import csv
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# ---- project-local imports (you already have these files) ----
from pipeline_http_bridge import (
    push_important_event,
    push_llm_event,
    push_stt_event,
    push_yamnet_event,
    start_http_server,
    register_unity_audio_sink,
    UNITY_AUDIO_LOCK,
)
from audio_buffer import AudioBuffer
from load_env import load_env

# Load .env early
load_env(".env")

# ===================== Paths / Directories (MUST be defined early) =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("PIPELINE_LOG_DIR") or os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.environ.get("PIPELINE_MODELS_DIR") or os.path.join(BASE_DIR, "models")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiment")

for _p in (LOG_DIR, MODELS_DIR, EXPERIMENT_DIR):
    try:
        os.makedirs(_p, exist_ok=True)
    except Exception:
        pass

# ===================== Reduced label set (keep as a constant; was a stray list before) =====================
REDUCED_LABEL_SET = [
    "speech",
    "crowd",
    "music",
    "dog",
    "cat",
    "bird",
    "vehicle_horn",
    "traffic_road",
    "car_bus_truck",
    "sirens",
    "rail",
    "aircraft",
    "engine_motion",
    "alarms_buzzer",
    "phone_ring",
    "wind_rain",
    "door_knock",
    "glass_break",
    "explosion_gunshot",
    "Silence",
    "other",
]

# ===================== Experiment Runner Config (FIX: BASE_DIR/LOG_DIR now exist) =====================
EXPERIMENT_ENABLED = True
EXPERIMENT_HTTP_HOST = os.environ.get("EXPERIMENT_HTTP_HOST", "0.0.0.0")
EXPERIMENT_HTTP_PORT = int(os.environ.get("EXPERIMENT_HTTP_PORT", "9001"))

EXPERIMENT_TRIALS_CSV = os.environ.get(
    "EXPERIMENT_TRIALS_CSV",
    os.path.join(EXPERIMENT_DIR, "trials.csv")
)
EXPERIMENT_RESULTS_CSV = os.environ.get(
    "EXPERIMENT_RESULTS_CSV",
    os.path.join(LOG_DIR, "experiment_results.csv")
)

# -------- Set matplotlib backend EARLY for macOS GUI safety --------
import matplotlib
if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from textwrap import shorten

# -------- Optional imports for Gemini (handled gracefully) --------
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# -------- Optional imports for classification (handled gracefully) --------
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except Exception:
    tf = None
    hub = None
    TF_AVAILABLE = False

CLASSIFIER_ENABLED = TF_AVAILABLE

# ===================== DummyStream (FIX: was missing) =====================
class DummyStream:
    """
    Used when use_local_mic=False (Unity pushes audio).
    Behaves like a minimal sounddevice stream context.
    """
    def __init__(self):
        self.active = True

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active = False
        return False

    def abort(self):
        self.active = False

# ===================== Device utilities =====================

def list_input_devices():
    """Return list of (index, name, max_input_channels, default_samplerate)."""
    devs = sd.query_devices()
    rows = []
    for i, d in enumerate(devs):
        rows.append((
            i,
            d.get("name", ""),
            d.get("max_input_channels", 0),
            int(d.get("default_samplerate", 0) or 0),
        ))
    return rows

def print_input_devices(chosen_index=None):
    """Pretty print available input devices, marking the chosen one."""
    rows = list_input_devices()
    print("\nAvailable audio devices:")
    print(f"{'Idx':>3}  {'InCh':>4}  {'DefSR':>6}  Name")
    for i, name, in_ch, sr in [(r[0], r[1], r[2], r[3]) for r in rows]:
        mark = "*" if chosen_index == i else " "
        print(f"{i:>3}{mark}  {in_ch:>4}  {sr:>6}  {name}")
    print("('*' marks the selected input device)\n")

def pick_input_device():
    """
    Choose an input device:
      1) sd.default.device[0] if valid input device
      2) mic-like device name heuristic
      3) first device with input channels
    Returns (device_index, device_name).
    """
    devs = sd.query_devices()
    cand = None

    # 1) default input
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    if isinstance(default_in, int) and 0 <= default_in < len(devs):
        if devs[default_in].get("max_input_channels", 0) > 0:
            cand = default_in

    # 2) heuristic
    if cand is None:
        keywords = ("microphone", "mic", "built-in", "external", "usb")
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                name = (d.get("name") or "").lower()
                if any(k in name for k in keywords):
                    cand = i
                    break

    # 3) first input-capable device
    if cand is None:
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                cand = i
                break

    if cand is None:
        raise RuntimeError("No input device with capture channels found.")

    return cand, devs[cand].get("name", "")

def pick_sample_rate(device_index):
    """
    Choose a sensible samplerate:
      - device default if available
      - else try 48000 then 44100
      - else None (let PortAudio choose)
    """
    info = sd.query_devices(device_index)
    sr = float(info.get("default_samplerate", 0) or 0)
    if sr > 0:
        return int(sr)
    for candidate in (48000, 44100):
        try:
            sd.check_input_settings(device=device_index, samplerate=candidate, channels=1)
            return candidate
        except Exception:
            continue
    return None

# ===================== Reduced YAMNet (SavedModel) resolve =====================

def _resolve_reduced_model_dir():
    env = os.environ.get("REDUCED_MODEL_DIR")
    if env:
        return env
    p1 = os.path.join(MODELS_DIR, "reduced_yamnet_savedmodel")
    p2 = os.path.join(BASE_DIR, "reduced_yamnet_savedmodel")

    def has_savedmodel(p: str) -> bool:
        return (
            os.path.exists(os.path.join(p, "saved_model.pb")) or
            os.path.exists(os.path.join(p, "saved_model.pbtxt"))
        )

    return p1 if has_savedmodel(p1) else p2

def _resolve_reduced_labels_json():
    env = os.environ.get("REDUCED_LABELS_JSON")
    if env:
        return env
    p1 = os.path.join(MODELS_DIR, "reduced_labels.json")
    p2 = os.path.join(BASE_DIR, "reduced_labels.json")
    return p1 if os.path.exists(p1) else p2

REDUCED_MODEL_DIR = _resolve_reduced_model_dir()
REDUCED_LABELS_JSON = _resolve_reduced_labels_json()
USE_REDUCED = True  # True: reduced pooled model, False: full YAMNet (521 classes)

# ==== Data collection mode ====
DATA_COLLECTION_MODE = True
WIDE_CSV_PATH = os.path.join(LOG_DIR, "classification_probs.csv")
TOPK_OVERLAY = 5
PRINT_ALL_TO_CONSOLE = False

# ==== Speech-to-Text (Whisper) ====
STT_ENABLED = True
WHISPER_MODEL_SIZE = "small"     # "tiny", "base", "small", "medium", "large-v3"
WHISPER_COMPUTE_TYPE = "int8"    # "int8"/"float16"/"float32"
STT_WINDOW_S = 5.0               # rolling window length sent to Whisper
STT_HOP_S = 1.0                  # (kept for legacy, worker uses DECODE_HOP_S)
STT_MIN_TEXT_LEN = 1
STT_CSV_PATH = os.path.join(LOG_DIR, "transcription_log.csv")

# ==== STT gating by YAMNet (speech VAD) ====
SPEECH_GATE_ENABLED = True
SPEECH_LABEL = "speech"
SPEECH_ON_THRESH = 0.40
SPEECH_OFF_THRESH = 0.30
SPEECH_MIN_ON_S = 0.50
SPEECH_MIN_OFF_S = 0.00
WHISPER_LANGUAGE = "en"          # None for auto-detect
REQUIRE_TOP_IS_SPEECH = False

# STT session bounds
PREROLL_S     = 1.5
MAX_SESSION_S = 5.0
DECODE_HOP_S  = 1.5

# ===== Gemini / LLM config =====
LLM_ENABLED = True
LLM_DRY_RUN = False
GEMINI_MODEL_NAME = "gemini-2.5-flash"

LLM_CSV_PATH = os.path.join(LOG_DIR, "llm_events.csv")
LLM_JSONL_PATH = os.path.join(LOG_DIR, "llm_events.jsonl")

LLM_MAX_EVENTS_PER_WINDOW = 40
LLM_MAX_STT_CHARS = 600

# ===== LLM sound-event triggers =====
IMPORTANT_SOUND_LABELS = {
    "music",
    "dog",
    "cat",
    "bird",
    "vehicle_horn",
    "traffic_road",
    "car_bus_truck",
    "sirens",
    "rail",
    "aircraft",
    "engine_motion",
    "alarms_buzzer",
    "phone_ring",
    "door_knock",
    "glass_break",
    "explosion_gunshot",
}
IMPORTANT_SOUND_THRESH = 0.55
SOUND_LLM_WINDOW_S = 5.0
SOUND_LLM_COOLDOWN_S = 10.0
GLOBAL_LLM_COOLDOWN_S = 2.0

# ===================== Debug helper =====================
DEBUG_STT = False
def _dbg(tag, msg):
    if DEBUG_STT:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{threading.current_thread().name}] {tag}: {msg}")
# ====== WhisperTranscriber (faster-whisper) ======
class WhisperTranscriber:
    """
    Streaming Whisper with:
      - pre-roll ring buffer (PREROLL_S)
      - session start/stop controlled by YAMNet (outside)
      - per-session max duration (MAX_SESSION_S)
    """
    def __init__(self, model_size=WHISPER_MODEL_SIZE, compute_type=WHISPER_COMPUTE_TYPE, target_sr=16000):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, compute_type=compute_type)
        self.target_sr = int(target_sr)

        # main buffer (active session only)
        self._buf = np.zeros(0, dtype=np.float32)

        # pre-roll ring buffer (always running)
        self._pre = np.zeros(0, dtype=np.float32)
        self._pre_cap = int(round(PREROLL_S * self.target_sr))

        # session control/time
        self.session_active = False
        self._session_start_ts = None
        self._last_decode_ts = 0.0

        self._lock = threading.Lock()
        self.latest_text = ""
        self.language = WHISPER_LANGUAGE

    def feed_preroll(self, x: np.ndarray, sr_in: float):
        """Always called: maintain last PREROLL_S seconds in a ring buffer."""
        sr_in = int(sr_in)
        if sr_in != self.target_sr:
            x = self._resample_linear(x, sr_in, self.target_sr)
        if x.size == 0:
            return
        with self._lock:
            if self._pre.size == 0:
                self._pre = x.copy()
            else:
                self._pre = np.concatenate([self._pre, x])[-self._pre_cap:]

    def append_audio(self, x: np.ndarray, sr_in: float):
        """Append only when a session is active."""
        if not self.session_active:
            return
        sr_in = int(sr_in)
        if sr_in != self.target_sr:
            x = self._resample_linear(x, sr_in, self.target_sr)
        if x.size == 0:
            return
        with self._lock:
            self._buf = np.concatenate([self._buf, x])

    def start_session(self, use_preroll: bool = True):
        """Start a session. If use_preroll=False, start with an empty buffer."""
        with self._lock:
            if use_preroll:
                self._buf = self._pre.copy()
            else:
                self._buf = np.zeros(0, dtype=np.float32)
        self.session_active = True
        self._session_start_ts = time.time()
        self._last_decode_ts = 0.0
        self.latest_text = ""

    def stop_session(self, finalize=True):
        """Stop and optionally decode everything collected so far."""
        self.session_active = False
        text, eff = (None, 0.0)
        if finalize:
            text, eff = self._decode_now(force=True)
            if text:
                self.latest_text = text
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
        self._session_start_ts = None
        return text, eff

    def session_duration_s(self) -> float:
        if not self.session_active or self._session_start_ts is None:
            return 0.0
        return time.time() - float(self._session_start_ts)

    def maybe_decode(self, now_ts: float, window_s=STT_WINDOW_S, hop_s=DECODE_HOP_S):
        if not self.session_active:
            return None, 0.0
        if now_ts - self._last_decode_ts < hop_s:
            return None, 0.0
        return self._decode_now(force=False, window_s=window_s)

    def _decode_now(self, force=False, window_s=STT_WINDOW_S):
        with self._lock:
            if self._buf.size == 0:
                return (None, 0.0)
            if force:
                audio = self._buf.copy()
            else:
                n_keep = int(max(1, round(window_s * self.target_sr)))
                audio = self._buf[-n_keep:] if self._buf.size > n_keep else self._buf.copy()

        eff_window_s = float(audio.size) / float(self.target_sr)

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
        text = "".join(seg.text for seg in segments).strip()
        self._last_decode_ts = time.time()
        return (text if len(text) >= STT_MIN_TEXT_LEN else None, eff_window_s)

    @staticmethod
    def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if x.size == 0:
            return x
        dur = x.size / float(sr_in)
        t_in  = np.linspace(0.0, dur, num=x.size, endpoint=False)
        n_out = int(round(dur * sr_out))
        t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


class GeminiEventAnalyzer:
    """
    Wraps a Gemini model for sequential sound-event interpretation
    using `google.genai` client (if installed).
    """
    def __init__(
        self,
        model_name=GEMINI_MODEL_NAME,
        dry_run=LLM_DRY_RUN,
        csv_path=LLM_CSV_PATH,
        jsonl_path=LLM_JSONL_PATH,
    ):
        self.model_name = model_name
        self.dry_run = bool(dry_run)
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

        print(f"[LLM] init: model_name={self.model_name}, dry_run={self.dry_run}")
        self._ensure_logs()

        self._client = None
        if not self.dry_run:
            api_key = os.environ.get("GEMINI_API_KEY")
            print(f"[LLM] init: API key present? {bool(api_key)}")
            if not api_key:
                print("[LLM] No GEMINI_API_KEY; forcing dry_run=True")
                self.dry_run = True
            elif not GENAI_AVAILABLE:
                print("[LLM] google-genai not installed; forcing dry_run=True")
                self.dry_run = True
            else:
                try:
                    # google-genai reads GEMINI_API_KEY from env
                    self._client = genai.Client()
                    print("[LLM] Gemini client initialized.")
                except Exception as e:
                    print(f"[LLM] Failed to init Gemini client: {e}; dry_run=True")
                    self._client = None
                    self.dry_run = True
        else:
            print("[LLM] init: staying in dry_run mode (no API calls).")

    def _ensure_logs(self):
        try:
            needs_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
            if needs_header:
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "iso_time", "unix_time",
                        "window_start_unix", "window_end_unix",
                        "stt_text",
                        "yamnet_events_json",
                        "llm_brief_summary",
                        "llm_user_message",
                        "model_name",
                    ])
                print(f"[LLM] Logging CSV to {self.csv_path}")
        except Exception as e:
            print(f"[LLM] Could not prepare CSV log: {e}")

    @staticmethod
    def _build_prompt(window_dict: dict) -> str:
        start_ts = window_dict.get("start_ts")
        end_ts = window_dict.get("end_ts")
        stt_text = window_dict.get("stt_text") or ""
        events = window_dict.get("yamnet_events") or []

        if len(stt_text) > LLM_MAX_STT_CHARS:
            stt_text = stt_text[:LLM_MAX_STT_CHARS] + "…"

        lines = []
        for ev in events[:LLM_MAX_EVENTS_PER_WINDOW]:
            rel_t = ev.get("rel_t")
            top5 = ev.get("top5", [])
            label_str = ", ".join([f"{x['label']} ({x['prob']:.2f})" for x in top5])
            lines.append(f"t={rel_t:+.1f}s -> {label_str}")

        events_block = "\n".join(lines) if lines else "(no YAMNet events captured)"

        prompt = f"""
You are assisting a Deaf or Hard-of-Hearing person by summarizing the acoustic environment.

You receive:
1) A short recent transcript of speech (STT).
2) A timeline of sound event probabilities from an audio classifier (YAMNet).

Your goals:
- Briefly explain what seems to be happening around the user in plain English.
- If appropriate, warn the user about relevant events (alarms, vehicles, glass breaking, etc.).
- Be concise (1–3 short sentences), not verbose.

Constraints:
- Respond strictly in JSON with the following keys:
  {{
    "brief_summary": "one short sentence about the environment",
    "user_message": "one or two short sentences to show to the user",
    "important_events": [
      {{
        "type": "speech" | "alarm" | "vehicle" | "impact" | "other",
        "description": "short description",
        "priority": "low" | "medium" | "high"
      }}
    ]
  }}
- Do NOT include any text outside the JSON.
- Entire response MUST be valid JSON only.

Time window (unix): start={start_ts}, end={end_ts}

Recent speech transcript:
\"\"\"{stt_text}\"\"\"

YAMNet top-5 timeline:
{events_block}
"""
        return prompt

    def analyze_window(self, window_dict: dict) -> dict:
        prompt = self._build_prompt(window_dict)

        ts_now = time.time()
        iso = datetime.fromtimestamp(ts_now, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")

        default = {
            "brief_summary": "Environment summary unavailable.",
            "user_message": "The assistant could not generate an environment summary for this window.",
            "important_events": [],
            "raw_response": "",
        }

        print(f"[LLM] analyze_window: dry_run={self.dry_run}, client_none={self._client is None}")

        if self.dry_run or self._client is None:
            try:
                self._log(iso, ts_now, window_dict, default)
            except Exception as e:
                print(f"[LLM] dry_run log error: {e}")
            print("[LLM] dry_run: skipping Gemini call.")
            return default

        # real API call
        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(resp, "text", None)

            if text is None and getattr(resp, "candidates", None):
                parts = []
                for c in resp.candidates:
                    for p in getattr(c.content, "parts", []):
                        if hasattr(p, "text"):
                            parts.append(p.text)
                text = "\n".join(parts)

            if text is None:
                raise RuntimeError("LLM response has no text")
        except Exception as e:
            print(f"[LLM] ERROR calling Gemini: {e}")
            try:
                self._log(iso, ts_now, window_dict, default)
            except Exception as e2:
                print(f"[LLM] log error after Gemini failure: {e2}")
            return default

        import re
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

        if parsed is None:
            print("[LLM] JSON parse failed, raw response (first 300 chars):")
            print(repr(text[:300]))
            parsed = dict(default)
            parsed["raw_response"] = text

        brief = parsed.get("brief_summary") or default["brief_summary"]
        msg = parsed.get("user_message") or brief
        evs = parsed.get("important_events") or []

        result = {
            "brief_summary": brief,
            "user_message": msg,
            "important_events": evs,
            "raw_response": text,
        }

        # push to Unity/bridge
        try:
            session_id = getattr(self, "session_id", "default")
            start_ts = window_dict.get("start_ts")
            end_ts = window_dict.get("end_ts")
            push_llm_event(
                session_id=session_id,
                window_start=start_ts,
                window_end=end_ts,
                brief_summary=result.get("brief_summary", ""),
                user_message=result.get("user_message", ""),
                important_events=result.get("important_events", []),
            )
        except Exception as e:
            print("[LLM] Failed to push LLM event:", e)

        try:
            self._log(iso, ts_now, window_dict, result)
        except Exception as e:
            print(f"[LLM] log error: {e}")

        return result

    def _log(self, iso_time, unix_time, window_dict, result_dict):
        try:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    iso_time,
                    f"{unix_time:.3f}",
                    f"{window_dict.get('start_ts', 0.0):.3f}",
                    f"{window_dict.get('end_ts', 0.0):.3f}",
                    (window_dict.get("stt_text") or "").replace("\n", " "),
                    json.dumps(window_dict.get("yamnet_events") or []),
                    result_dict.get("brief_summary", ""),
                    result_dict.get("user_message", ""),
                    self.model_name,
                ])
        except Exception as e:
            print(f"[LLM] CSV log error: {e}")

        try:
            rec = {
                "iso_time": iso_time,
                "unix_time": unix_time,
                "window": window_dict,
                "result": result_dict,
                "model_name": self.model_name,
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LLM] JSONL log error: {e}")


class YamnetClassifier:
    """
    Supports:
      - Reduced SavedModel (pooled labels) -> self.infer(...)
      - Full YAMNet (TF-Hub)              -> self.yamnet(...)
    Provides:
      - predict_all() -> (labels, probs[K]) normalized sum=1
      - predict_top() -> (label, prob)
    """
    def __init__(self):
        if not CLASSIFIER_ENABLED:
            raise RuntimeError("TensorFlow not available; classifier disabled.")

        self.target_sr = 16000

        if USE_REDUCED:
            from json import load as _json_load
            print("Loading Reduced-YAMNet SavedModel…")
            mod = tf.saved_model.load(REDUCED_MODEL_DIR)
            self.infer = mod.__call__.get_concrete_function()
            with open(REDUCED_LABELS_JSON, "r", encoding="utf-8") as f:
                self.class_names = _json_load(f)
            self.mode = "reduced"
            print("[OK] Reduced-YAMNet loaded.")
        else:
            import csv as _csv
            print("Loading full YAMNet from TF-Hub…")
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self.yamnet.class_map_path().numpy().decode("utf-8")
            names = []
            with tf.io.gfile.GFile(class_map_path, "r") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    names.append(row["display_name"])
            self.class_names = names
            self.mode = "full"
            print("[OK] Full YAMNet loaded (521 classes).")

    def predict_all(self, mono_audio: np.ndarray, input_sr: int):
        """Return (labels, probs[K]) where probs are normalized (sum=1)."""
        if int(input_sr) != self.target_sr:
            mono_audio = self._resample_linear(mono_audio, int(input_sr), self.target_sr)

        y = tf.convert_to_tensor(mono_audio, dtype=tf.float32)

        if self.mode == "reduced":
            out = self.infer(waveform_16k=y)  # {"probs": [K]}
            probs = out["probs"].numpy()
        else:
            scores, embeddings, _ = self.yamnet(y)  # scores: [T,521]
            probs = tf.reduce_mean(scores, axis=0).numpy()

        probs = probs / (probs.sum() + 1e-9)
        return self.class_names, probs

    def predict_top(self, mono_audio: np.ndarray, input_sr: int):
        labels, probs = self.predict_all(mono_audio, int(input_sr))
        i = int(np.argmax(probs))
        return labels[i] if 0 <= i < len(labels) else "Unknown", float(probs[i])

    @staticmethod
    def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if x.size == 0:
            return x
        dur = x.size / float(sr_in)
        t_in = np.linspace(0.0, dur, num=x.size, endpoint=False)
        n_out = int(round(dur * sr_out))
        t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32, copy=False)
# ===================== Visualizer with Optional Classification =====================

class RealTimeSPLVisualizer:
    def __init__(
        self,
        update_interval_s=0.1,    # Plot update period
        history_s=20.0,           # Seconds of history
        min_db=-120.0,
        max_db=0.0,
        classify=True,            # Try to enable classifier
        classify_window_s=2.0,
        use_local_mic=False,
        log_csv_path=os.path.join(LOG_DIR, "classification_log.csv"),
    ):
        self.update_interval_s = float(update_interval_s)
        self.history_s = float(history_s)
        self.min_db = float(min_db)
        self.max_db = float(max_db)

        self.use_local_mic = bool(use_local_mic)

        # Gate state for STT
        self._gate_state = "IDLE"  # IDLE, RECORDING, COOLDOWN
        self._gate_last_change = time.time()

        # Queues / buffers
        self.q_levels = queue.Queue()       # (t_monotonic, dBFS) for plotting
        self.q_audio = queue.Queue()        # audio chunks for classifier
        self.max_points = int(np.ceil(self.history_s / self.update_interval_s)) + 10
        self.times = deque(maxlen=self.max_points)
        self.levels = deque(maxlen=self.max_points)

        # Device & rate (only meaningful if local mic)
        self.device_index, self.device_name = pick_input_device()
        self.samplerate = pick_sample_rate(self.device_index)
        self.stream_samplerate = self.samplerate if self.samplerate else None
        self.blocksize = (
            None if self.stream_samplerate is None
            else max(1, int(self.stream_samplerate * self.update_interval_s))
        )

        print_input_devices(self.device_index)
        print(f"Selected input device #{self.device_index}: {self.device_name}")
        print(f"Sample rate: {self.stream_samplerate or 'default'} Hz | Blocksize: {self.blocksize or 'default'}")
        print(
            "If the plot is flat: ensure macOS Microphone permission for Python is enabled.\n"
            "System Settings → Privacy & Security → Microphone → enable for your Python/Terminal app.\n"
        )

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(11, 4))
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(self.min_db, self.max_db)
        self.ax.set_xlim(-self.history_s, 0.0)
        title_sr = self.stream_samplerate or "default"
        self.ax.set_title(
            f"Real-Time SPL (RMS dBFS) — {shorten(self.device_name, width=48)} @ {title_sr} Hz (mono)"
        )
        self.ax.set_xlabel("Time (s) relative to now")
        self.ax.set_ylabel("SPL (dBFS)")
        self.ax.grid(True, linestyle="--", alpha=0.4)

        # Overlays (matplotlib only — Unity tarafında UI yok)
        self.text_readout = self.ax.text(
            0.01, 0.95, "— dBFS",
            transform=self.ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_label = self.ax.text(
            0.99, 0.95, "Classifier: disabled",
            transform=self.ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_stt = self.ax.text(
            0.99, 0.05, "STT: disabled",
            transform=self.ax.transAxes, va="bottom", ha="right",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )
        self.text_llm = self.ax.text(
            0.5, 0.02, "Env: —",
            transform=self.ax.transAxes, va="bottom", ha="center",
            bbox=dict(boxstyle="round", alpha=0.15, fc="w"),
        )

        # zero-block detector must exist in BOTH modes (Unity feed or local mic)
        self._zero_blocks_seen = 0

        # Audio stream (mono)
        if self.use_local_mic:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                dtype="float32",
                samplerate=self.stream_samplerate,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
        else:
            self.stream = DummyStream()

        # Animation (GUI main thread)
        self.ani = FuncAnimation(
            self.fig,
            self.on_timer,
            interval=int(self.update_interval_s * 1000),
            blit=False,
        )

        # Ctrl+C handler
        signal.signal(signal.SIGINT, self._sigint_handler)

        # --- Classifier support ---
        self.classifier_enabled = bool(classify) and CLASSIFIER_ENABLED
        self.classify_window_s = float(classify_window_s)

        self.latest_label = None
        self.latest_conf = None
        self._label_lock = threading.Lock()

        # --- Experiment: latest model prediction snapshot (for RT/accuracy logging) ---
        self._exp_pred_lock = threading.Lock()
        self.exp_latest_pred_label = None
        self.exp_latest_pred_prob = None
        self.exp_latest_pred_unix = None

        if self.classifier_enabled:
            try:
                self.classifier = YamnetClassifier()
                self.text_label.set_text("Classifier: loading…")
                self._classifier_thread = threading.Thread(
                    target=self._classification_worker,
                    daemon=True,
                    name="ClassifierWorker",
                )
                self._classifier_thread.start()
            except Exception as e:
                print(f"[Classifier] Disabled: {e}")
                self.classifier_enabled = False
                self.text_label.set_text("Classifier: disabled")
        else:
            self.text_label.set_text("Classifier: disabled")

        self.log_csv_path = log_csv_path if self.classifier_enabled else None
        if self.log_csv_path:
            self._ensure_csv_header()

        # --- STT (Whisper) support ---
        self.stt_enabled = bool(STT_ENABLED)
        self.q_audio_stt = queue.Queue() if self.stt_enabled else None
        self._stt_lock = threading.Lock()
        self.latest_transcript = ""

        if self.stt_enabled:
            try:
                self.stt = WhisperTranscriber()
                self.text_stt.set_text("STT: loading…")
                self._stt_thread = threading.Thread(
                    target=self._stt_worker,
                    daemon=True,
                    name="STTWorker",
                )
                self._stt_thread.start()
            except Exception as e:
                print(f"[STT] Disabled: {e}")
                self.stt_enabled = False
                self.text_stt.set_text("STT: disabled")
        else:
            self.text_stt.set_text("STT: disabled")

        # ensure STT CSV header
        if self.stt_enabled:
            try:
                needs_header = (not os.path.exists(STT_CSV_PATH)) or os.path.getsize(STT_CSV_PATH) == 0
                if needs_header:
                    with open(STT_CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow(["iso_time", "unix_time", "window_s", "samplerate_hz", "text"])
                print(f"[CSV] Logging STT to: {STT_CSV_PATH}")
            except Exception as e:
                print(f"[CSV] Could not prepare STT log: {e}")

        # --- LLM / Gemini integration ---
        self._last_sound_llm_ts = 0.0
        self._last_any_llm_ts = 0.0

        self.llm_enabled = bool(LLM_ENABLED)
        self.llm_analyzer = None
        self._llm_queue = queue.Queue() if self.llm_enabled else None
        self._llm_events_buffer = deque(maxlen=600)  # ~60s at ~10Hz
        self._llm_buf_lock = threading.Lock()

        if self.llm_enabled:
            try:
                self.llm_analyzer = GeminiEventAnalyzer()
                # allow analyzer to read session_id if we set it later
                self._llm_thread = threading.Thread(
                    target=self._llm_worker,
                    daemon=True,
                    name="LLMWorker",
                )
                self._llm_thread.start()
                print("[LLM] GeminiEventAnalyzer initialized.")
            except Exception as e:
                print(f"[LLM] disabled (init error): {e}")
                self.llm_enabled = False

    # ----------------- Unity feed ----------------- #
    def feed_unity_chunk(self, samples: np.ndarray):
        """
        Unity → Python audio bridge.
        In Unity mode (use_local_mic=False), we simulate a sounddevice callback
        so all existing logic (YAMNet + STT pre-roll + gating) works unchanged.
        """
        if samples is None or samples.size == 0:
            return

        try:
            if samples.ndim != 1:
                samples = samples.reshape(-1)
            samples = samples.astype(np.float32, copy=False)

            if not getattr(self, "use_local_mic", True):
                frames = samples.shape[0]
                indata = samples.reshape(-1, 1)
                self.audio_callback(indata, frames, None, None)
            else:
                # if local mic is on but Unity still sends (unlikely)
                try:
                    self.q_audio.put_nowait(samples)
                except Exception:
                    pass

        except Exception as e:
            print(f"[UnityAudio] Failed to route samples via audio_callback: {e}")

    # ----------------- audio callback ----------------- #
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        if frames <= 0:
            return

        x = indata[:, 0]  # float32 mono

        # zero-block detection
        if np.allclose(x, 0.0):
            self._zero_blocks_seen += 1
        else:
            self._zero_blocks_seen = 0

        # dBFS for the block (float64 for stability)
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        dbfs = 20.0 * np.log10(rms + 1e-9)
        t = time.monotonic()

        # push to level queue
        try:
            self.q_levels.put_nowait((t, float(dbfs)))
        except Exception:
            pass

        if self.classifier_enabled:
            try:
                self.q_audio.put_nowait(x.copy())
            except Exception:
                pass

        if self.stt_enabled and self.q_audio_stt is not None:
            try:
                self.q_audio_stt.put_nowait(x.copy())
            except Exception:
                pass

    # ----------------- classifier worker ----------------- #
    def _classification_worker(self):
        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        chunk_buffer = []
        samples_target = int(round(sr_in * self.classify_window_s))
        samples_accum = 0

        with self._label_lock:
            self.latest_label = "Ready"
            self.latest_conf = None
        self.text_label.set_text("Classifier: ready")

        while True:
            try:
                chunk = self.q_audio.get(timeout=1.0)
            except queue.Empty:
                continue

            chunk_buffer.append(chunk)
            samples_accum += chunk.size

            if samples_accum < samples_target:
                continue

            audio_win = np.concatenate(chunk_buffer, axis=0)[-samples_target:]
            chunk_buffer.clear()
            samples_accum = 0

            rms = float(np.sqrt(np.mean(audio_win ** 2) + 1e-9))
            spl_dbfs = 20.0 * np.log10(rms + 1e-9)

            try:
                top5 = None
                dominant_label = None
                dominant_prob = None

                if DATA_COLLECTION_MODE:
                    labels, probs = self.classifier.predict_all(audio_win, int(sr_in))
                    order = np.argsort(probs)[::-1]
                    k = min(TOPK_OVERLAY, len(order))
                    overlay_lines = [f"{labels[i]} ({probs[i]*100:.0f}%)" for i in order[:k]]
                    overlay_text = "\n".join(overlay_lines)

                    top_i = int(order[0])
                    label, conf = labels[top_i], float(probs[top_i])

                    top5_idx = order[:5]
                    top5 = [{"label": labels[i], "prob": float(probs[i])} for i in top5_idx]
                    dominant_label = top5[0]["label"]
                    dominant_prob = top5[0]["prob"]

                    # buffer yamnet for LLM
                    if self.llm_enabled:
                        now = time.time()
                        with self._llm_buf_lock:
                            self._llm_events_buffer.append({"ts": now, "top5": top5})

                    # sound-only LLM triggers
                    if self.llm_enabled and self._llm_queue is not None:
                        important_hits = [
                            (l["label"], l["prob"])
                            for l in top5
                            if l["label"] in IMPORTANT_SOUND_LABELS and l["prob"] >= IMPORTANT_SOUND_THRESH
                        ]
                        if important_hits:
                            now_ts = time.time()
                            if now_ts - self._last_sound_llm_ts >= SOUND_LLM_COOLDOWN_S:
                                self._last_sound_llm_ts = now_ts
                                start_ts = now_ts - SOUND_LLM_WINDOW_S

                                with self._llm_buf_lock:
                                    buf = list(self._llm_events_buffer)

                                slice_events = []
                                for ev in buf:
                                    ts_ev = ev.get("ts", 0.0)
                                    if start_ts <= ts_ev <= now_ts:
                                        slice_events.append({
                                            "ts": ts_ev,
                                            "rel_t": ts_ev - now_ts,
                                            "top5": ev.get("top5", []),
                                        })

                                with self._stt_lock:
                                    stt_context = (self.latest_transcript or "").strip()

                                window = {
                                    "start_ts": start_ts,
                                    "end_ts": now_ts,
                                    "stt_text": stt_context,
                                    "yamnet_events": slice_events,
                                }
                                try:
                                    self._llm_queue.put_nowait(window)
                                except Exception:
                                    print("[LLM] queue full; dropping sound-trigger window.")

                    # log full distribution (wide)
                    ts_now = time.time()
                    self._append_wide_csv_row(ts_now, sr_in, labels, probs)

                else:
                    label, conf = self.classifier.predict_top(audio_win, int(sr_in))
                    overlay_text = f"{label} ({int(round(conf*100))}%)"
                    top5 = [{"label": label, "prob": float(conf)}]
                    dominant_label = label
                    dominant_prob = float(conf)

                # snapshot latest dominant prediction (for ExperimentRunner)
                snap_ts = time.time()
                with self._exp_pred_lock:
                    self.exp_latest_pred_label = dominant_label
                    self.exp_latest_pred_prob = float(dominant_prob) if dominant_prob is not None else None
                    self.exp_latest_pred_unix = snap_ts

                # --- STT gating by speech prob ---
                if SPEECH_GATE_ENABLED and self.stt_enabled:
                    now = time.time()

                    # compute speech_prob + top_is_speech
                    try:
                        if DATA_COLLECTION_MODE:
                            speech_idx = labels.index(SPEECH_LABEL)
                            speech_prob = float(probs[speech_idx])
                            top_i = int(np.argmax(probs))
                            top_is_speech = (labels[top_i] == SPEECH_LABEL)
                        else:
                            speech_prob = float(conf) if (label == SPEECH_LABEL) else 0.0
                            top_is_speech = (label == SPEECH_LABEL)
                    except Exception:
                        speech_prob = 0.0
                        top_is_speech = False

                    cond_on = (speech_prob >= SPEECH_ON_THRESH) and (top_is_speech if REQUIRE_TOP_IS_SPEECH else True)
                    cond_off = (speech_prob < SPEECH_OFF_THRESH) or (not top_is_speech and REQUIRE_TOP_IS_SPEECH)

                    elapsed = now - float(self._gate_last_change)

                    if self._gate_state == "IDLE":
                        if cond_on and elapsed >= SPEECH_MIN_OFF_S:
                            self._gate_state = "RECORDING"
                            self._gate_last_change = now
                            self.stt.start_session()
                            self.text_stt.set_text("STT: listening…")

                    elif self._gate_state == "RECORDING":
                        if cond_off and elapsed >= SPEECH_MIN_ON_S:
                            final_txt, eff = self.stt.stop_session(finalize=True)
                            if final_txt:
                                self._stt_publish(final_txt, sr_in, eff)
                            self.text_stt.set_text("STT: idle")
                            self._gate_state = "COOLDOWN"
                            self._gate_last_change = now

                    elif self._gate_state == "COOLDOWN":
                        # cooldown is just a one-cycle buffer to avoid thrash
                        self._gate_state = "IDLE"
                        self._gate_last_change = now

            except Exception as e:
                overlay_text = "Classification error"
                label, conf = "Classification error", 0.0
                print(f"[Classifier] Inference error: {e}")
                top5 = [{"label": "error", "prob": 1.0}]
                dominant_label = "error"
                dominant_prob = 1.0

                snap_ts = time.time()
                with self._exp_pred_lock:
                    self.exp_latest_pred_label = dominant_label
                    self.exp_latest_pred_prob = float(dominant_prob)
                    self.exp_latest_pred_unix = snap_ts

            # update overlay
            with self._label_lock:
                self.latest_label = overlay_text
                self.latest_conf = None if DATA_COLLECTION_MODE else conf

            # log label+conf
            ts_now = time.time()
            self._append_csv_row(ts_now, sr_in, label, conf)

            # push yamnet event
            try:
                window_end = ts_now
                window_start = window_end - self.classify_window_s
                session_id = getattr(self, "session_id", "default")

                if top5 is not None and dominant_label is not None:
                    push_yamnet_event(
                        session_id=session_id,
                        timestamp_unix=window_end,
                        window_start=window_start,
                        window_end=window_end,
                        top5=top5,
                        dominant_label=dominant_label,
                        dominant_prob=dominant_prob,
                        spl_dbfs=float(spl_dbfs),
                    )
            except Exception as e:
                print(f"[Classifier] Failed to push YAMNet event: {e}")

    # ----------------- STT worker ----------------- #
    def _stt_worker(self):
        if not self.stt_enabled:
            return

        sr_in = float(self.stream_samplerate) if self.stream_samplerate else 48000.0
        self.text_stt.set_text("STT: ready")

        while True:
            try:
                chunk = self.q_audio_stt.get(timeout=1.0)
            except queue.Empty:
                txt, eff = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
                if txt:
                    self._stt_publish(txt, sr_in, eff)
                continue

            # Always maintain pre-roll ring
            self.stt.feed_preroll(chunk, sr_in)

            # Only append to active session
            if self.stt.session_active:
                self.stt.append_audio(chunk, sr_in)

            # Periodic decode
            txt, eff = self.stt.maybe_decode(time.time(), STT_WINDOW_S, DECODE_HOP_S)
            if txt:
                self._stt_publish(txt, sr_in, eff)

            # Hard stop by time cap
            if self.stt.session_active and (self.stt.session_duration_s() >= MAX_SESSION_S):
                final_txt, eff = self.stt.stop_session(finalize=True)
                if final_txt:
                    self._stt_publish(final_txt, sr_in, eff)

                if getattr(self, "_gate_state", None) == "RECORDING":
                    self.stt.start_session(use_preroll=False)
                    self.text_stt.set_text("STT: listening…")
                else:
                    self.text_stt.set_text("STT: idle")

    # ----------------- LLM worker ----------------- #
    def _llm_worker(self):
        if not self.llm_enabled or self.llm_analyzer is None or self._llm_queue is None:
            return

        while True:
            window = self._llm_queue.get()
            try:
                # allow analyzer to see same session_id as visualizer
                self.llm_analyzer.session_id = getattr(self, "session_id", "default")
                result = self.llm_analyzer.analyze_window(window)
            except Exception as e:
                print(f"[LLM] worker error: {e}")
                continue

            summary = result.get("brief_summary") or result.get("user_message") or ""
            if summary:
                short = summary if len(summary) <= 80 else (summary[:77] + "…")
                self.text_llm.set_text(f"Env: {short}")

    def _stt_publish(self, text: str, sr_in: float, eff_window_s: float):
        """Publish latest transcript to overlay + CSV + bridge."""
        if not text:
            return

        with self._stt_lock:
            self.latest_transcript = text

        shown = text if len(text) <= 60 else (text[:57] + "…")
        self.text_stt.set_text(f"STT: {shown}")

        # Log STT
        try:
            ts = time.time()
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            with open(STT_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([iso, f"{ts:.3f}", f"{eff_window_s:.3f}", int(sr_in), text])
        except Exception as e:
            print(f"[CSV] STT write error: {e}")

        # Push STT event
        try:
            session_id = getattr(self, "session_id", "default")
            end_unix = time.time()
            start_unix = end_unix - float(eff_window_s)
            segment_id = f"stt-{int(end_unix * 1000)}"
            push_stt_event(
                session_id=session_id,
                segment_id=segment_id,
                start_unix=start_unix,
                end_unix=end_unix,
                eff_window_s=float(eff_window_s),
                samplerate_hz=int(sr_in),
                text=text,
                language=str(WHISPER_LANGUAGE or ""),
                confidence=1.0,
            )
        except Exception as e:
            print(f"[STT] Failed to push STT event: {e}")

        # LLM window trigger (speech-based)
        if self.llm_enabled and self._llm_queue is not None and eff_window_s > 0.0 and text.strip():
            end_ts = time.time()
            start_ts = end_ts - float(eff_window_s)

            with self._llm_buf_lock:
                events = list(self._llm_events_buffer)

            slice_events = []
            for ev in events:
                ts_ev = ev.get("ts", 0.0)
                if start_ts <= ts_ev <= end_ts:
                    slice_events.append({
                        "ts": ts_ev,
                        "rel_t": ts_ev - end_ts,
                        "top5": ev.get("top5", []),
                    })

            window = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "stt_text": text,
                "yamnet_events": slice_events,
            }

            now_ts = time.time()
            if now_ts - self._last_any_llm_ts < GLOBAL_LLM_COOLDOWN_S:
                return
            self._last_any_llm_ts = now_ts

            try:
                self._llm_queue.put_nowait(window)
            except Exception:
                print("[LLM] queue full; dropping this window.")

    # ----------------- plot updater ----------------- #
    def on_timer(self, _frame):
        updated = False
        now = time.monotonic()

        # Drain level queue
        try:
            while True:
                t, dbfs = self.q_levels.get_nowait()
                self.times.append(t)
                dbfs = float(np.clip(dbfs, self.min_db, self.max_db))
                self.levels.append(dbfs)
                updated = True
        except queue.Empty:
            pass

        if not updated:
            return self.line,

        rel_times = np.array(self.times, dtype=np.float64) - now
        mask = rel_times >= -self.history_s
        rel_times = rel_times[mask]
        rel_levels = np.array(self.levels, dtype=np.float64)[mask]

        self.line.set_data(rel_times, rel_levels)
        self.ax.set_xlim(-self.history_s, 0.0)

        if rel_levels.size > 0:
            self.text_readout.set_text(f"{rel_levels[-1]:.1f} dBFS")

        if self._zero_blocks_seen >= 5:
            self.ax.set_title("Real-Time SPL (RMS dBFS) — No signal detected (check mic permission / device)")
        else:
            title_sr = self.stream_samplerate or "default"
            self.ax.set_title(
                f"Real-Time SPL (RMS dBFS) — {shorten(self.device_name, width=48)} @ {title_sr} Hz (mono)"
            )

        if self.classifier_enabled:
            with self._label_lock:
                if self.latest_label is not None:
                    if self.latest_conf is None:
                        self.text_label.set_text(f"{self.latest_label}")
                    else:
                        pct = int(round(self.latest_conf * 100))
                        self.text_label.set_text(f"{self.latest_label} ({pct}%)")
        else:
            self.text_label.set_text("Classifier: disabled")

        return self.line,

    # ----------------- run / cleanup ----------------- #
    def run(self):
        try:
            with self.stream:
                plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if getattr(self.stream, "active", False):
                    self.stream.abort()
            except Exception:
                pass

    def _sigint_handler(self, *_):
        plt.close(self.fig)

    # ----------------- CSV helpers ----------------- #
    def _ensure_wide_csv_header(self, labels):
        try:
            needs_header = (not os.path.exists(WIDE_CSV_PATH)) or os.path.getsize(WIDE_CSV_PATH) == 0
            if needs_header:
                with open(WIDE_CSV_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    header = ["iso_time", "unix_time", "window_s", "samplerate_hz"] + list(labels)
                    w.writerow(header)
                print(f"[CSV] Logging full probabilities to: {WIDE_CSV_PATH}")
        except Exception as e:
            print(f"[CSV] Could not prepare wide CSV: {e}")
            return False
        return True

    def _append_wide_csv_row(self, ts_unix, sr_hz, labels, probs):
        try:
            if not os.path.exists(WIDE_CSV_PATH) or os.path.getsize(WIDE_CSV_PATH) == 0:
                if not self._ensure_wide_csv_header(labels):
                    return
            iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            row = [iso, f"{ts_unix:.3f}", f"{self.classify_window_s:.3f}", int(sr_hz)] + [f"{p:.6f}" for p in probs.tolist()]
            with open(WIDE_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[CSV] wide write error: {e}")

    def _ensure_csv_header(self):
        try:
            needs_header = (not os.path.exists(self.log_csv_path)) or os.path.getsize(self.log_csv_path) == 0
            if needs_header:
                with open(self.log_csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iso_time", "unix_time", "window_s", "samplerate_hz", "label", "confidence"])
                print(f"[CSV] Logging classifications to: {self.log_csv_path}")
        except Exception as e:
            print(f"[CSV] Could not prepare log file: {e}")
            self.log_csv_path = None

    def _append_csv_row(self, ts_unix, sr_hz, label, conf):
        if not self.log_csv_path:
            return
        try:
            iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            with open(self.log_csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([iso, f"{ts_unix:.3f}", f"{self.classify_window_s:.3f}", int(sr_hz), label, f"{float(conf):.6f}"])
        except Exception as e:
            print(f"[CSV] write error: {e}")
# ===================== Experiment Runner (Single-file) =====================
# ===================== Experiment Runner (Unity JSON Compatible) =====================

class ExperimentRunner:
    """
    Unity (JsonUtility) uyumlu API:

    GET  /api/current  -> flat JSON:
      {
        "done": false,
        "trial_id": "...",
        "target_label": "...",
        "prompt": "...",
        "options": [...],
        "started_unix": 123.456
      }

    POST /api/confirm  body:
      { "trial_id": "...", "selected_label": "...", "rt_ms": 1234 }

    POST /api/confirm response:
      { "ok": true, "correct": true/false, "done": true/false, "expected": "...", "got": "...", "error": "" }
    """

    # Unity tarafındaki fallbackOptions ile aynı olsun diye:
    DEFAULT_OPTIONS = [
        "speech","crowd","music","dog","cat","bird","vehicle_horn","traffic_road","car_bus_truck",
        "sirens","rail","aircraft","engine_motion","alarms_buzzer","phone_ring","wind_rain",
        "door_knock","glass_break","explosion_gunshot","Silence","other"
    ]

    def __init__(self, visualizer: "RealTimeSPLVisualizer"):
        self.vis = visualizer
        self.trials_csv = EXPERIMENT_TRIALS_CSV
        self.results_csv = EXPERIMENT_RESULTS_CSV

        self._lock = threading.Lock()
        self._trials = []
        self._idx = 0
        self._done = False

        # timestamps
        self._trial_presented_unix = None  # first time /api/current hit for that trial

        self._load_trials()
        self._ensure_results_header()

        print(f"[EXP] Loaded {len(self._trials)} trials from: {self.trials_csv}")
        if len(self._trials) == 0:
            print("[EXP] WARNING: No trials loaded. Check EXPERIMENT_TRIALS_CSV.")

    # ---------- CSV ----------
    def _load_trials(self):
        with self._lock:
            self._trials = []
            self._idx = 0
            self._done = False
            self._trial_presented_unix = None

            if not self.trials_csv or not os.path.exists(self.trials_csv):
                print(f"[EXP] trials.csv not found: {self.trials_csv}")
                return

            try:
                with open(self.trials_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trial_id = (row.get("trial_id") or row.get("id") or "").strip()
                        target = (row.get("target_label") or row.get("label") or row.get("target") or "").strip()
                        prompt = (row.get("prompt") or row.get("text") or "").strip()

                        # opsiyonel: csv içinde "options" sütunu varsa (pipe ile)
                        # ör: options = "dog|cat|speech"
                        options_raw = (row.get("options") or "").strip()
                        options = []
                        if options_raw:
                            options = [x.strip() for x in options_raw.split("|") if x.strip()]

                        if not target:
                            continue
                        if not trial_id:
                            trial_id = f"trial-{len(self._trials)+1:03d}"

                        self._trials.append({
                            "trial_id": trial_id,
                            "target_label": target,
                            "prompt": prompt,
                            "options": options,
                        })
            except Exception as e:
                print(f"[EXP] Failed to read trials CSV: {e}")
                self._trials = []

    def _ensure_results_header(self):
        try:
            os.makedirs(os.path.dirname(self.results_csv), exist_ok=True)
        except Exception:
            pass

        try:
            needs = (not os.path.exists(self.results_csv)) or os.path.getsize(self.results_csv) == 0
            if needs:
                with open(self.results_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "iso_time","unix_time",
                        "trial_id","trial_index",
                        "target_label",
                        "selected_label","is_correct",
                        "rt_ms",
                        # model snapshot
                        "model_pred_label","model_pred_prob","model_pred_unix",
                    ])
                print(f"[EXP] Logging results CSV to: {self.results_csv}")
        except Exception as e:
            print(f"[EXP] Could not prepare results CSV: {e}")

    def _append_result(self, trial_id, trial_index, target_label, selected_label, is_correct, rt_ms,
                       model_pred_label, model_pred_prob, model_pred_unix):
        try:
            ts = time.time()
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
            with open(self.results_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    iso, f"{ts:.3f}",
                    trial_id, trial_index,
                    target_label,
                    selected_label, int(is_correct),
                    int(rt_ms),
                    model_pred_label or "",
                    f"{model_pred_prob:.6f}" if model_pred_prob is not None else "",
                    f"{model_pred_unix:.3f}" if model_pred_unix is not None else "",
                ])
        except Exception as e:
            print(f"[EXP] Results write error: {e}")

    # ---------- state ----------
    def _current_trial(self):
        with self._lock:
            if self._done:
                return None
            if not self._trials:
                return None
            if self._idx < 0:
                self._idx = 0
            if self._idx >= len(self._trials):
                self._idx = len(self._trials) - 1
            return self._trials[self._idx], self._idx, len(self._trials)

    def _advance(self):
        with self._lock:
            if not self._trials:
                self._done = True
                return
            self._idx += 1
            self._trial_presented_unix = None
            if self._idx >= len(self._trials):
                self._done = True

    # ---------- API ----------
    def api_get_current(self):
        tinfo = self._current_trial()
        if tinfo is None:
            return {
                "done": True,
                "trial_id": "",
                "target_label": "",
                "prompt": "",
                "options": self.DEFAULT_OPTIONS,
                "started_unix": time.time(),
            }

        trial, idx, n = tinfo

        now = time.time()
        with self._lock:
            if self._trial_presented_unix is None:
                self._trial_presented_unix = now
            started_unix = self._trial_presented_unix

        options = trial.get("options") or []
        if not options:
            options = self.DEFAULT_OPTIONS

        return {
            "done": False,
            "trial_id": trial.get("trial_id", ""),
            "target_label": trial.get("target_label", ""),
            "prompt": trial.get("prompt", ""),
            "options": options,
            "started_unix": float(started_unix),
        }

    def api_post_confirm(self, payload: dict):
        # expect: trial_id, selected_label, rt_ms
        tinfo = self._current_trial()
        if tinfo is None:
            return {"ok": False, "correct": False, "done": True, "error": "no_active_trial", "expected": "", "got": ""}

        trial, idx, n = tinfo
        expected = (trial.get("target_label") or "").strip()
        got = (payload.get("selected_label") or "").strip()
        trial_id_client = (payload.get("trial_id") or "").strip()

        if not got:
            return {"ok": False, "correct": False, "done": False, "error": "missing_selected_label", "expected": expected, "got": ""}

        # trial id mismatch -> warn but continue
        if trial_id_client and trial_id_client != (trial.get("trial_id") or ""):
            print(f"[EXP] WARNING trial_id mismatch: client={trial_id_client} server={trial.get('trial_id')} (continuing)")

        # rt
        rt_ms = payload.get("rt_ms", 0)
        try:
            rt_ms = int(rt_ms)
        except Exception:
            rt_ms = 0

        correct = (got.lower() == expected.lower())

        # model snapshot
        with self.vis._exp_pred_lock:
            model_pred_label = self.vis.exp_latest_pred_label
            model_pred_prob = self.vis.exp_latest_pred_prob
            model_pred_unix = self.vis.exp_latest_pred_unix

        self._append_result(
            trial_id=trial.get("trial_id", ""),
            trial_index=idx,
            target_label=expected,
            selected_label=got,
            is_correct=correct,
            rt_ms=rt_ms,
            model_pred_label=model_pred_label,
            model_pred_prob=model_pred_prob,
            model_pred_unix=model_pred_unix,
        )

        # optional: pipeline push
        try:
            session_id = getattr(self.vis, "session_id", "default")
            push_important_event(
                session_id=session_id,
                timestamp_unix=time.time(),
                event_type="experiment_confirm",
                label=got,
                confidence=float(model_pred_prob) if model_pred_prob is not None else None,
                metadata={
                    "trial_id": trial.get("trial_id", ""),
                    "expected": expected,
                    "correct": bool(correct),
                    "rt_ms": int(rt_ms),
                }
            )
        except Exception as e:
            print(f"[EXP] push_important_event failed: {e}")

        # move next
        self._advance()

        return {
            "ok": True,
            "correct": bool(correct),
            "done": bool(self._done),
            "error": "",
            "expected": expected,
            "got": got,
        }

    # ---------- HTTP ----------
    def start_http_server(self, host="0.0.0.0", port=9001):
        runner = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, code: int, payload: dict):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/api/current"):
                    out = runner.api_get_current()
                    self._send_json(200, out)
                else:
                    self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self):
                if self.path.startswith("/api/confirm"):
                    try:
                        n = int(self.headers.get("Content-Length", "0"))
                        raw = self.rfile.read(n) if n > 0 else b"{}"
                        payload = json.loads(raw.decode("utf-8") or "{}")
                    except Exception:
                        payload = {}

                    out = runner.api_post_confirm(payload)
                    self._send_json(200 if out.get("ok") else 400, out)
                else:
                    self._send_json(404, {"ok": False, "error": "not_found"})

            def log_message(self, fmt, *args):
                return  # silence

        httpd = HTTPServer((host, int(port)), Handler)
        print(f"[EXP] HTTP server listening on http://{host}:{int(port)}")
        httpd.serve_forever()

def main():
    vis = RealTimeSPLVisualizer(
        update_interval_s=0.1,
        history_s=20.0,
        min_db=-120.0,
        max_db=0.0,
        classify=True,
        classify_window_s=1.0,
        use_local_mic=False,   # Unity feeds audio
    )

    vis.session_id = "default"

    # Unity audio -> Python pipeline
    register_unity_audio_sink(vis.feed_unity_chunk)

    # Pipeline HTTP server (your existing bridge)
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

    # Experiment runner
    if EXPERIMENT_ENABLED:
        exp = ExperimentRunner(vis)

        exp_server_thread = threading.Thread(
            target=exp.start_http_server,
            kwargs={"host": EXPERIMENT_HTTP_HOST, "port": EXPERIMENT_HTTP_PORT},
            daemon=True,
            name="ExperimentHTTPServer",
        )
        exp_server_thread.start()


        print(f"[EXP] Results CSV: {EXPERIMENT_RESULTS_CSV}")
        print(f"[EXP] Current-trial endpoint: http://{EXPERIMENT_HTTP_HOST}:{EXPERIMENT_HTTP_PORT}/api/current")
        print(f"[EXP] Confirm endpoint:       http://{EXPERIMENT_HTTP_HOST}:{EXPERIMENT_HTTP_PORT}/api/confirm")

    # Run (blocking)
    vis.run()


if __name__ == "__main__":
    main()

'''