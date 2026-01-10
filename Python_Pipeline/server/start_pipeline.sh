#!/bin/bash

# Pipeline başlatma script'i - 172.20.10.2:8000 üzerinde
# Çalıştır: ./start_pipeline.sh

cd "$(dirname "$0")"

# Venv'i aktifleştir
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Bağımlılıkları kontrol et
echo "Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null

# Pipeline'ı başlat
echo "================================"
echo "Starting Pipeline.py"
echo "HTTP Server: http://172.20.10.2:8000"
echo "================================"
echo ""

export PIPELINE_HTTP_HOST=172.20.10.2
export PIPELINE_HTTP_PORT=8000

python Pipeline.py
