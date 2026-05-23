@echo off
setlocal

where py >nul 2>nul
if %errorlevel% neq 0 (
    echo Python launcher py was not found. Install Python 3.10 from python.org first.
    exit /b 1
)

py -3.10 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.10 was not found. Install Python 3.10, then rerun this file.
    exit /b 1
)

py -3.10 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-windows.txt
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('Devices:', tf.config.list_physical_devices())"

echo.
echo Setup complete. To train, run: run_all_windows.bat
endlocal
