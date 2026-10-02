#!/bin/bash
# Double-click to install (first time only) and run Layers.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Install it from the page that opens now, then run this file again."
  open https://www.python.org/downloads/
  read -r -p "Press Enter to close"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating environment - first time only..."
  python3 -m venv .venv || exit 1
fi
source .venv/bin/activate

echo "Installing packages - first time takes several minutes..."
python -m pip install --disable-pip-version-check -q -r requirements.txt || { read -r -p "Package installation failed. Press Enter"; exit 1; }
python scripts/download_models.py || { read -r -p "Model download failed. Press Enter"; exit 1; }

echo
echo "Starting Layers at http://localhost:8000 - keep this window open, close it to stop."
(sleep 10; open http://localhost:8000) &
python -m uvicorn app.server:app --port 8000
