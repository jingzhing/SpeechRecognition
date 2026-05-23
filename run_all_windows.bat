@echo off
setlocal

if not exist .venv\Scripts\activate.bat (
    echo Virtual environment not found. Run setup_windows.bat first.
    exit /b 1
)

call .venv\Scripts\activate.bat
python scripts\download_dataset.py
if %errorlevel% neq 0 exit /b %errorlevel%

python scripts\train.py
if %errorlevel% neq 0 exit /b %errorlevel%

python scripts\evaluate_model.py
if %errorlevel% neq 0 exit /b %errorlevel%

python scripts\predict_audio.py --audio data\filtered\5\5_A_0.wav
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Finished. Check outputs\model and outputs\reports.
endlocal
