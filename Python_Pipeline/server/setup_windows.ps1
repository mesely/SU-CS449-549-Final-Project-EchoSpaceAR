$ErrorActionPreference = "Stop"

Write-Host "[Windows] Setting up venv + dependencies..."

if (!(Test-Path ".\.venv")) {
  py -m venv .venv
  Write-Host "[Windows] Created .venv"
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "[Windows] Done."
Write-Host "Run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python Pipeline.py"
