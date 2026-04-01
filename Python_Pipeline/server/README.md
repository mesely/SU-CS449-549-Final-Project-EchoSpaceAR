# Python Pipeline Server

This folder contains the refactored Python runtime used by EchoSpaceAR for:

- real-time SPL visualization
- reduced YAMNet sound classification
- Whisper speech transcription
- optional Gemini-based acoustic summaries
- Unity-to-Python HTTP bridging

## What Changed

The runtime keeps the same behavior and entrypoint (`Pipeline.py`), but the large monolithic script is now split into smaller modules under `pipeline_runtime/`.

- `Pipeline.py`: compatibility launcher
- `pipeline_runtime/config.py`: runtime constants and path resolution
- `pipeline_runtime/device_utils.py`: audio device discovery helpers
- `pipeline_runtime/classification.py`: reduced/full YAMNet loading and inference
- `pipeline_runtime/transcription.py`: Whisper session and pre-roll handling
- `pipeline_runtime/llm.py`: Gemini prompt building, parsing, and logging
- `pipeline_runtime/visualizer.py`: plotting, queues, workers, and event publishing
- `pipeline_http_bridge.py`: Unity HTTP event bridge

## Quick Start

1. Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Make sure the reduced model exists under `models/reduced_yamnet_savedmodel/`, or rebuild it:

```bash
python build_reduced_yamnet.py
```

3. Run the pipeline:

```bash
python Pipeline.py
```

4. Optional host overrides:

```bash
PIPELINE_HTTP_HOST=0.0.0.0 PIPELINE_HTTP_PORT=8000 python Pipeline.py
```

## HTTP Endpoints

- `POST /client_hello`
- `POST /audio_chunk`
- `GET /events?session_id=<id>&since_unix=<timestamp>`

## Logs

Generated logs are written under `logs/`:

- `classification_log.csv`
- `classification_probs.csv`
- `transcription_log.csv`
- `llm_events.csv`
- `llm_events.jsonl`

## Visual Documentation

Open the static explainer page at:

- `docs/python_pipeline_refactor.html`

It summarizes the runtime flow, module map, and the refactor decisions in a browser-friendly format.
