@echo off
rem Double-click to install (first time only) and run Layers.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python is not installed.
  echo Install it from the page that opens now - tick "Add python.exe to PATH" - then run this file again.
  start https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist .venv (
  echo Creating environment - first time only...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing packages - first time takes several minutes...
python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 ( echo Package installation failed. & pause & exit /b 1 )

python scripts\download_models.py
if errorlevel 1 ( echo Model download failed. & pause & exit /b 1 )

echo.
echo Starting Layers at http://localhost:8000  - keep this window open, close it to stop.
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8000"
python -m uvicorn app.server:app --port 8000
pause
