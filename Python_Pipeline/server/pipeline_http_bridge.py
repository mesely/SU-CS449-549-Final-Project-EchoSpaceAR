"""HTTP bridge between Unity clients and the Python audio pipeline."""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np


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
) -> None:
    """Store one YAMNet window for later polling by Unity."""
    event = {
        "kind": "yamnet",
        "timestamp_unix": timestamp_unix,
        "session_id": session_id,
        "yamnet": {
            "window_start_unix": window_start,
            "window_end_unix": window_end,
            "top5": top5,
            "dominant_label": dominant_label,
            "dominant_prob": dominant_prob,
            "spl_dbfs": spl_dbfs,
        },
    }
    _append_event(session_id, event)
    with EVENTS_LOCK:
        total = len(EVENTS[session_id])
    print(f"[EVENT] yamnet -> session={session_id}, total={total}")


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
    with EVENTS_LOCK:
        total = len(EVENTS[session_id])
    print(f"[EVENT] stt -> session={session_id}, total={total}")


def push_llm_event(
    session_id: str,
    window_start: float,
    window_end: float,
    brief_summary: str,
    user_message: str,
    important_events: list,
) -> None:
    """Store one LLM summary window for later polling by Unity."""
    now = time.time()
    event = {
        "kind": "llm",
        "timestamp_unix": now,
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
    with EVENTS_LOCK:
        total = len(EVENTS[session_id])
    print(f"[EVENT] llm -> session={session_id}, total={total}")


def push_important_event(
    session_id: str,
    event_time_unix: float,
    source: str,
    label: str,
    mapped_type: str,
    priority: str,
    confidence: float,
    description: str,
) -> None:
    """Store a high-priority alert event."""
    now = time.time()
    event = {
        "kind": "important",
        "timestamp_unix": now,
        "session_id": session_id,
        "important": {
            "event_id": f"imp-{int(now * 1000)}",
            "event_time_unix": event_time_unix,
            "source": source,
            "label": label,
            "mapped_type": mapped_type,
            "priority": priority,
            "confidence": confidence,
            "description": description,
        },
    }
    _append_event(session_id, event)


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
            self._handle_client_hello()
            return

        if self.path == "/audio_chunk":
            self._handle_audio_chunk()
            return

        self.send_json({"error": "Not found"}, status=404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/events":
            self.send_json({"error": "Not found"}, status=404)
            return

        query = parse_qs(parsed.query)
        session_id = query.get("session_id", [""])[0]
        since_unix = self._parse_since_unix(query.get("since_unix", ["0"])[0])

        with EVENTS_LOCK:
            session_events = EVENTS.get(session_id, [])
            new_events = [event for event in session_events if event["timestamp_unix"] > since_unix]

        last_timestamp = new_events[-1]["timestamp_unix"] if new_events else since_unix
        self.send_json({"events": new_events, "last_timestamp_unix": last_timestamp})

    def log_message(self, format, *args) -> None:
        """Silence the default `BaseHTTPRequestHandler` access log."""
        return

    def _handle_client_hello(self) -> None:
        data = self.read_json_body()
        session_id = data.get("session_id") or ""
        print(f"CLIENT_HELLO: {data}")
        _ensure_session(session_id)
        self.send_json({"status": "ok", "message": "hello_received"})

    def _handle_audio_chunk(self) -> None:
        data = self.read_json_body()
        session_id = data.get("session_id") or ""
        sequence = data.get("seq")
        timestamp_unix = data.get("timestamp_unix")

        print(f"[HTTP] Received audio_chunk seq={sequence} from session={session_id} at {timestamp_unix}")

        try:
            pcm_base64 = data["pcm_base64"]
            raw_bytes = base64.b64decode(pcm_base64)
            samples = np.frombuffer(raw_bytes, dtype=np.float32)

            if samples.size == 0:
                print("[HTTP]   samples.size = 0 (empty chunk!)")
            else:
                max_abs = float(np.max(np.abs(samples)))
                print(f"[HTTP]   samples.shape={samples.shape}, max_abs={max_abs:.4f}")

            with UNITY_AUDIO_LOCK:
                sink = UNITY_AUDIO_SINK

            if sink is not None:
                sink(samples)
            else:
                print("[HTTP]   WARNING: UNITY_AUDIO_SINK is None; audio ignored.")
        except Exception as exc:
            print(f"[HTTP] Error decoding/forwarding audio_chunk: {exc}")

        self.send_json({"status": "received"})

    @staticmethod
    def _parse_since_unix(raw_since: str) -> float:
        safe_value = raw_since.replace(",", ".")
        try:
            return float(safe_value)
        except ValueError:
            print(f"[HTTP] Warning: bad since_unix='{raw_since}', defaulting to 0")
            return 0.0


def start_http_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the HTTP bridge and block forever."""
    httpd = HTTPServer((host, port), PipelineHttpHandler)
    print(f"HTTP server at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    start_http_server("172.20.10.2")
