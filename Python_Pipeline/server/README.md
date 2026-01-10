# Pipeline

This repo runs a real-time audio pipeline with optional YAMNet classification, Whisper STT (faster-whisper), and Gemini (google-genai) integration.

Quick start (macOS):

1. Create and activate a Python venv (recommended Python 3.11+).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place model files:

Alternatively set env vars in `.env`:

3. Run the pipeline (Unity mode example):

```bash
python Pipeline.py
```

Files:

If you want me to place a small placeholder model or add more setup automation, tell me and I'll add it.
Quick model setup
-----------------
- A small example `models/reduced_labels.json` is included.
- Place your SavedModel under `models/reduced_yamnet_savedmodel/` (the project will call `tf.saved_model.load`).
- To use the TF-Hub full YAMNet instead, set `USE_REDUCED = False` in `Pipeline.py`.

macOS notes
-----------
- If you see `sounddevice` errors, install PortAudio on macOS:

```bash
brew install portaudio
source .venv/bin/activate
pip install -r requirements.txt
```

Then run `python Pipeline.py`.

If you want me to place a small placeholder model or add more setup automation, tell me and I'll add it.

BURAYI OKU AMINA KODUMUN HAYATINDA BASH MI GORDUN
KESİN ÇALIŞTIRMA BOŞ YYAPMAYAN KISIM:

Çalıştıran tanrısal kod 
bash -lc 'set -e
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python -m pip install --upgrade pip certifi || true
export SSL_CERT_FILE="$(python -m certifi)"
echo "Using SSL_CERT_FILE=$SSL_CERT_FILE"
python -u build_reduced_yamnet.py


Sonra şu: 
source .venv/bin/activate
python -m pip install --upgrade pip certifi
export SSL_CERT_FILE="$(python -m certifi)"
echo "Using SSL_CERT_FILE=$SSL_CERT_FILE"
python -u build_reduced_yamnet.py


Sonra: