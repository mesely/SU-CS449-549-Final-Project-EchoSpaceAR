PROJECT NOTES (Windows + macOS)
==============================

1) Create venv
-------------
macOS:
  python3 -m venv .venv
  source .venv/bin/activate

Windows (PowerShell):
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1

2) Install deps
---------------
  python -m pip install --upgrade pip
  pip install -r requirements.txt

3) Audio permissions (macOS)
----------------------------
If the plot is flat / no audio:
System Settings → Privacy & Security → Microphone
Enable microphone permission for:
- Terminal / iTerm OR your IDE (VSCode/PyCharm), whichever launches Python.

4) PortAudio / sounddevice notes
--------------------------------
macOS:
  Usually works out-of-box. If sounddevice fails:
    brew install portaudio

Windows:
  sounddevice uses PortAudio bundled via wheels most of the time.
  If device list is empty, install correct audio drivers and try again.

5) TensorFlow notes
-------------------
macOS:
  tensorflow==2.15.0 runs CPU. (OK for YAMNet.)
Windows:
  tensorflow==2.15.0 CPU works.
  If you want GPU, that’s a separate CUDA setup (not required here).

6) Reduced YAMNet paths
-----------------------
If USE_REDUCED=True:
  - Make sure REDUCED_MODEL_DIR and REDUCED_LABELS_JSON paths are correct for your OS.
  - Prefer relative paths like:
      reduced_yamnet_savedmodel/
      reduced_labels.json

7) Environment variables / API keys
-----------------------------------
We store keys in .env (never commit it).

Example .env:
  GEMINI_API_KEY=YOUR_KEY_HERE

Run:
  python Pipeline.py

8) Networking
-------------
If Unity posts audio to Python via HTTP:
- Ensure host IP in Unity matches your machine IP
- Ensure firewall allows inbound TCP:8000 (or chosen port)

Test endpoint:
  GET http://<HOST>:8000/events?session_id=default&since_unix=0

9) Common quick fixes
---------------------
- "ModuleNotFoundError: audio_buffer":
  Ensure audio_buffer.py is in same folder as Pipeline.py
  Or run from project root:
    python Pipeline.py

- "No module named tensorflow":
  pip install -r requirements.txt
  Confirm you’re in the correct venv.

- Whisper is slow:
  set WHISPER_MODEL_SIZE="small" or "base"
  and WHISPER_COMPUTE_TYPE="int8"
