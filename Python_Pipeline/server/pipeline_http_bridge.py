# pipeline_http_bridge.py (or inside your main file)

import base64
import json
import threading
import numpy as np
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np  # needed to decode PCM

from audio_buffer import AudioBuffer           # if you put AudioBuffer elsewhere


# events[session_id] = list of event dicts (yamnet/stt/llm)
# Global event storage (you already have this)
EVENTS = {}
EVENTS_LOCK = threading.Lock()

# NEW: global hook for Unity audio → pipeline
UNITY_AUDIO_SINK = None  # will be set from Pipeline.py
UNITY_AUDIO_LOCK = threading.Lock()

def register_unity_audio_sink(func):
    """
    Register a callable that accepts a 1D np.float32 array (mono audio chunk).
    Pipeline.py will call this with vis.feed_unity_chunk or directly vis.q_audio.put.
    """
    global UNITY_AUDIO_SINK
    with UNITY_AUDIO_LOCK:
        UNITY_AUDIO_SINK = func
    print("[HTTP BRIDGE] Registered Unity audio sink:", func)


def _ensure_session(session_id: str):
    with EVENTS_LOCK:
        if session_id not in EVENTS:
            EVENTS[session_id] = []


def push_yamnet_event(session_id: str, timestamp_unix: float, window_start: float, window_end: float,
                      top5, dominant_label: str, dominant_prob: float, spl_dbfs: float):
    """
    top5: list of dicts: { 'label': str, 'prob': float }
    """
    _ensure_session(session_id)
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
    with EVENTS_LOCK:
        EVENTS[session_id].append(event)
        print(f"[EVENT] yamnet -> session={session_id}, total={len(EVENTS[session_id])}")


def push_stt_event(session_id: str, segment_id: str, start_unix: float, end_unix: float,
                   eff_window_s: float, samplerate_hz: int, text: str,
                   language: str, confidence: float):
    _ensure_session(session_id)
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
    with EVENTS_LOCK:
        EVENTS[session_id].append(event)
        print(f"[EVENT] stt -> session={session_id}, total={len(EVENTS[session_id])}")

def push_llm_event(session_id: str, window_start: float, window_end: float,
                   brief_summary: str, user_message: str, important_events: list):
    """
    important_events: list of dicts:
      {
        "type": "speech" | "alarm" | "vehicle" | "impact" | "other",
        "description": "...",
        "priority": "low" | "medium" | "high"
      }
    """
    now = time.time()
    _ensure_session(session_id)
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
    with EVENTS_LOCK:
        EVENTS[session_id].append(event)
        print(f"[EVENT] llm -> session={session_id}, total={len(EVENTS[session_id])}")


def push_important_event(session_id: str, event_time_unix: float, source: str,
                         label: str, mapped_type: str, priority: str,
                         confidence: float, description: str):
    """
    This is your fast alert (sirens, horns, glass, etc.)
    """
    now = time.time()
    _ensure_session(session_id)
    event = {
        "kind": "important",
        "timestamp_unix": now,
        "session_id": session_id,
        "important": {
            "event_id": f"imp-{int(now * 1000)}",
            "event_time_unix": event_time_unix,
            "source": source,          # "yamnet" | "llm" | "stt"
            "label": label,            # e.g., "siren_emergency"
            "mapped_type": mapped_type,# "alarm" / "vehicle" / etc
            "priority": priority,
            "confidence": confidence,
            "description": description,
        },
    }
    with EVENTS_LOCK:
        EVENTS[session_id].append(event)



class PipelineHttpHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def do_POST(self):
        global PIPELINE, AUDIO_BUFFER

        path = self.path

        if path == "/client_hello":
            data = self.read_json_body()
            session_id = data.get("session_id")
            print("CLIENT_HELLO:", data)

            _ensure_session(session_id)
            # You can store per-session config if wanted (samplerate, etc.)

            return self.send_json({"status": "ok", "message": "hello_received"})

        elif path == "/audio_chunk":
            data = self.read_json_body()
            session_id = data.get("session_id")
            seq = data.get("seq")
            ts = data.get("timestamp_unix")

            print(f"[HTTP] Received audio_chunk seq={seq} from session={session_id} at {ts}")

            try:
                pcm_b64 = data["pcm_base64"]
                raw_bytes = base64.b64decode(pcm_b64)
                samples = np.frombuffer(raw_bytes, dtype=np.float32)

                # DEBUG: print basic stats
                if samples.size == 0:
                    print("[HTTP]   samples.size = 0 (empty chunk!)")
                else:
                    max_abs = float(np.max(np.abs(samples)))
                    print(f"[HTTP]   samples.shape={samples.shape}, max_abs={max_abs:.4f}")

                # forward to Unity audio sink
                with UNITY_AUDIO_LOCK:
                    sink = UNITY_AUDIO_SINK

                if sink is not None:
                    sink(samples)
                else:
                    print("[HTTP]   WARNING: UNITY_AUDIO_SINK is None; audio ignored.")

            except Exception as e:
                print("[HTTP] Error decoding/forwarding audio_chunk:", e)

            return self.send_json({"status": "received"})



        self.send_json({"error": "Not found"}, status=404)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/events":
            qs = parse_qs(parsed.query)
            session_id = qs.get("session_id", [""])[0]
            raw_since = qs.get("since_unix", ["0"])[0]
            # Be robust to comma decimal or weird formats from the client
            raw_since = raw_since.replace(",", ".")   # "1764,29" -> "1764.29"
            try:
                since_unix = float(raw_since)
            except ValueError:
                print(f"[HTTP] Warning: bad since_unix='{raw_since}', defaulting to 0")
                since_unix = 0.0


            with EVENTS_LOCK:
                session_events = EVENTS.get(session_id, [])

                new_events = [e for e in session_events if e["timestamp_unix"] > since_unix]
                last_ts = new_events[-1]["timestamp_unix"] if new_events else since_unix

            return self.send_json({
                "events": new_events,
                "last_timestamp_unix": last_ts
            })

        self.send_json({"error": "Not found"}, status=404)


def start_http_server(host="0.0.0.0", port=8000):
    httpd = HTTPServer((host, port), PipelineHttpHandler)
    print(f"HTTP server at http://{host}:{port}")
    httpd.serve_forever()



if __name__ == "__main__":
    start_http_server("172.20.10.2")

