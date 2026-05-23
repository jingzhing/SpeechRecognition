@echo off
if not exist outputs\model (
    echo outputs\model not found. Train locally first by running train_windows.bat or run_all_windows.bat.
    exit /b 1
)

docker build -t spoken-digit-recognition:windows-inference .
