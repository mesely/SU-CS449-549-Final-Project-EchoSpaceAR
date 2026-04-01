"""Gemini-based summarization for acoustic event windows."""

from __future__ import annotations

import csv
import json
import os
import re
import time

from .config import (
    GENAI_AVAILABLE,
    GEMINI_MODEL_NAME,
    LLM_CSV_PATH,
    LLM_DRY_RUN,
    LLM_JSONL_PATH,
    LLM_MAX_EVENTS_PER_WINDOW,
    LLM_MAX_STT_CHARS,
    genai,
)
from .utils import unix_to_local_iso
from pipeline_http_bridge import push_llm_event


class GeminiEventAnalyzer:
    """Summarize YAMNet and STT windows with Gemini."""

    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        dry_run: bool = LLM_DRY_RUN,
        csv_path: str = LLM_CSV_PATH,
        jsonl_path: str = LLM_JSONL_PATH,
    ) -> None:
        self.model_name = model_name
        self.dry_run = dry_run
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.session_id = "default"
        self._client = None

        print(f"[LLM] init: model_name={self.model_name}, dry_run={self.dry_run}")
        self._ensure_logs()

        if self.dry_run:
            print("[LLM] init: staying in dry_run mode (no API calls).")
            return

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        print(f"[LLM] init: API key present? {bool(api_key)}")
        if not api_key:
            print("[LLM] No API key in GEMINI_API_KEY or GOOGLE_API_KEY; forcing dry_run=True")
            self.dry_run = True
            return

        if not GENAI_AVAILABLE:
            print("[LLM] google-genai package not installed; forcing dry_run=True")
            self.dry_run = True
            return

        try:
            self._client = genai.Client()
            print("[LLM] Gemini client initialized.")
        except Exception as exc:
            print(f"[LLM] Failed to initialize Gemini client: {exc}; falling back to dry_run")
            self._client = None
            self.dry_run = True

    def _ensure_logs(self) -> None:
        """Create CSV headers used for LLM window logging."""
        try:
            needs_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
            if needs_header:
                with open(self.csv_path, "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "iso_time",
                            "unix_time",
                            "window_start_unix",
                            "window_end_unix",
                            "stt_text",
                            "yamnet_events_json",
                            "llm_brief_summary",
                            "llm_user_message",
                            "model_name",
                        ]
                    )
                print(f"[LLM] Logging CSV to {self.csv_path}")
        except Exception as exc:
            print(f"[LLM] Could not prepare CSV log: {exc}")

    @staticmethod
    def _build_prompt(window_dict: dict) -> str:
        start_ts = window_dict.get("start_ts")
        end_ts = window_dict.get("end_ts")
        stt_text = window_dict.get("stt_text") or ""
        events = window_dict.get("yamnet_events") or []

        if len(stt_text) > LLM_MAX_STT_CHARS:
            stt_text = stt_text[:LLM_MAX_STT_CHARS] + "..."

        lines = []
        for event in events[:LLM_MAX_EVENTS_PER_WINDOW]:
            rel_t = event.get("rel_t")
            top5 = event.get("top5", [])
            label_str = ", ".join(f"{item['label']} ({item['prob']:.2f})" for item in top5)
            lines.append(f"t={rel_t:+.1f}s -> {label_str}")

        events_block = "\n".join(lines) if lines else "(no YAMNet events captured)"

        return f"""
You are assisting a Deaf or Hard-of-Hearing person by summarizing the acoustic environment.

You receive:
1) A short recent transcript of speech (STT).
2) A timeline of sound event probabilities from an audio classifier (YAMNet).
   We are mostly interested in alarms, vehicles, announcements, footsteps, doors, glass breaking, and other important events.
   Music and generic background noise can usually be ignored unless it conveys an important context.

Your goals:
- Briefly explain what seems to be happening around the user in plain English.
- If appropriate, warn the user about relevant events (e.g., approaching station, alarms, vehicles, glass breaking).
- Be concise (1-3 short sentences), not verbose.
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

    def analyze_window(self, window_dict: dict) -> dict:
        """Call Gemini when available, otherwise log the window in dry-run mode."""
        prompt = self._build_prompt(window_dict)
        timestamp_unix = time.time()
        iso_time = unix_to_local_iso(timestamp_unix)
        default = {
            "brief_summary": "Environment summary unavailable.",
            "user_message": "The assistant could not generate an environment summary for this window.",
            "important_events": [],
            "raw_response": "",
        }

        print(f"[LLM] analyze_window: dry_run={self.dry_run}, client_none={self._client is None}")

        if self.dry_run or self._client is None:
            try:
                self._log(iso_time, timestamp_unix, window_dict, default)
            except Exception as exc:
                print(f"[LLM] dry_run log error: {exc}")
            print("[LLM] dry_run: would send prompt to Gemini, but skipping.")
            return default

        try:
            response = self._client.models.generate_content(model=self.model_name, contents=prompt)
            text = getattr(response, "text", None)
            if text is None and getattr(response, "candidates", None):
                parts = []
                for candidate in response.candidates:
                    for part in getattr(candidate.content, "parts", []):
                        if hasattr(part, "text"):
                            parts.append(part.text)
                text = "\n".join(parts)
            if text is None:
                raise RuntimeError("LLM response has no text")
        except Exception as exc:
            print(f"[LLM] ERROR calling Gemini: {exc}")
            try:
                self._log(iso_time, timestamp_unix, window_dict, default)
            except Exception as log_exc:
                print(f"[LLM] log error after Gemini failure: {log_exc}")
            return default

        parsed = self._parse_json_response(text)
        if parsed is None:
            print("[LLM] JSON parse failed, raw response (first 300 chars):")
            print(repr(text[:300]))
            parsed = dict(default)
            parsed["raw_response"] = text

        result = {
            "brief_summary": parsed.get("brief_summary") or default["brief_summary"],
            "user_message": parsed.get("user_message") or parsed.get("brief_summary") or default["user_message"],
            "important_events": parsed.get("important_events") or [],
            "raw_response": text,
        }

        try:
            push_llm_event(
                session_id=self.session_id,
                window_start=window_dict.get("start_ts"),
                window_end=window_dict.get("end_ts"),
                brief_summary=result["brief_summary"],
                user_message=result["user_message"],
                important_events=result["important_events"],
            )
        except Exception as exc:
            print(f"[LLM] Failed to push LLM event: {exc}")

        try:
            self._log(iso_time, timestamp_unix, window_dict, result)
        except Exception as exc:
            print(f"[LLM] log error: {exc}")
        return result

    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _log(self, iso_time: str, unix_time: float, window_dict: dict, result_dict: dict) -> None:
        try:
            with open(self.csv_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        iso_time,
                        f"{unix_time:.3f}",
                        f"{window_dict.get('start_ts', 0.0):.3f}",
                        f"{window_dict.get('end_ts', 0.0):.3f}",
                        (window_dict.get("stt_text") or "").replace("\n", " "),
                        json.dumps(window_dict.get("yamnet_events") or []),
                        result_dict.get("brief_summary", ""),
                        result_dict.get("user_message", ""),
                        self.model_name,
                    ]
                )
        except Exception as exc:
            print(f"[LLM] CSV log error: {exc}")

        try:
            record = {
                "iso_time": iso_time,
                "unix_time": unix_time,
                "window": window_dict,
                "result": result_dict,
                "model_name": self.model_name,
            }
            with open(self.jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[LLM] JSONL log error: {exc}")
