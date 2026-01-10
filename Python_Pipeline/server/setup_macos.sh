# Remove leading and trailing markdown fences
# 
# 
#!/usr/bin/env bash
set -e

echo "[macOS] Setting up venv + dependencies..."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "[macOS] Created .venv"
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[macOS] Done."
echo "If sounddevice fails, install PortAudio:"
echo "  brew install portaudio"
echo ""
echo "Run:"
echo "  source .venv/bin/activate"
echo "  python Pipeline.py"
