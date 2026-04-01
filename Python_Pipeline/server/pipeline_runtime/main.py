"""Executable entrypoint for the refactored pipeline."""

from __future__ import annotations

import os
import threading

from pipeline_http_bridge import register_unity_audio_sink, start_http_server

from .visualizer import RealTimeSPLVisualizer


def main() -> None:
    """Start the visualizer and the HTTP bridge used by Unity."""
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
